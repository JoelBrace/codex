use std::sync::Arc;

use axum::Router;
use axum::body::Body;
use axum::body::Bytes;
use axum::extract::State;
use axum::extract::rejection::JsonRejection;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::response::Response;
use axum::routing::get;
use axum::routing::post;
use chrono::Utc;
use codex_protocol::error::CodexErr;
use codex_protocol::openai_models::ReasoningEffort as ReasoningEffortConfig;
use serde_json::Value;
use tokio::time::Duration;
use tokio::time::MissedTickBehavior;
use tokio_stream::StreamExt;

use super::pool::append_log;
use super::pool::cache_session;
use super::pool::checkout_or_create_client;
use super::pool::transport_key;
use super::reasoning::RequestContext;
use super::reasoning::RetryHints;
use super::reasoning::attach_correlation_headers;
use super::reasoning::parse_request_reasoning_effort;
use super::reasoning::parse_service_tier;
use super::reasoning::request_context_from_headers;
use super::reasoning::resolve_for_request;
use super::reasoning::stream_retry_hints;
use super::state::HttpLogEntry;
use super::state::HttpServerState;
use super::stream::StreamTranslator;
use super::stream::format_sse;
use super::translation::proxy_model_info;
use super::translation::translate_request;
use super::wire_types::AnthropicRequest;

/// Claude Code assumes this context window size (tokens) when computing usage percentage.
/// It is hardcoded in the Claude Code binary by model slug and cannot be overridden via API.
const CLAUDE_CODE_ASSUMED_CONTEXT_WINDOW: f64 = 200_000.0;
const STREAM_PING_INTERVAL_SECS: u64 = 15;

pub(super) async fn health() -> &'static str {
    "ok"
}

pub(super) async fn list_models(State(state): State<Arc<HttpServerState>>) -> impl IntoResponse {
    let model_id = {
        let cfg = state.dynamic_config.read().await;
        cfg.model.clone()
    };
    let body = serde_json::json!({
        "object": "list",
        "data": [{
            "id": model_id,
            "object": "model",
            "created": 0,
            "owned_by": "codex-proxy"
        }]
    });
    axum::Json(body)
}

pub(super) async fn handle_messages(
    State(state): State<Arc<HttpServerState>>,
    headers: axum::http::HeaderMap,
    body: Result<axum::extract::Json<AnthropicRequest>, JsonRejection>,
) -> Response {
    let req_ctx = request_context_from_headers(&headers);
    let req = match body {
        Ok(body) => body.0,
        Err(e) => {
            return anthropic_error_response(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                format!("Invalid JSON body: {e}"),
                &req_ctx,
                Some(RetryHints {
                    should_retry: false,
                    retry_after_secs: None,
                }),
            );
        }
    };
    let stream_mode = req.stream;
    // Capture the model string Claude Code sent so we can echo it back in
    // responses — Claude Code uses this for telemetry and routing checks.
    let request_model = req.model.clone();

    let session_id = headers
        .get("x-codex-session-id")
        .or_else(|| headers.get("x-claude-code-session-id"))
        .and_then(|v| v.to_str().ok())
        .map(str::to_owned);

    // Read dynamic config snapshot.
    let (
        model,
        named_models,
        model_info,
        provider,
        reasoning_effort,
        reasoning_summary,
        enable_request_compression,
        beta_features_header,
    ) = {
        let cfg = state.dynamic_config.read().await;
        (
            cfg.model.clone(),
            cfg.named_models.clone(),
            cfg.model_info().clone(),
            cfg.provider.clone(),
            cfg.reasoning_effort,
            cfg.reasoning_summary.clone(),
            cfg.enable_request_compression,
            cfg.beta_features_header.clone(),
        )
    };

    let request_effort = parse_request_reasoning_effort(&req);
    let effective_reasoning_effort = request_effort.or(reasoning_effort);

    // Unified reasoning resolution: used for both logging and the stream call.
    let (resolved_model, stream_effort, log_effort, reasoning_source) = resolve_for_request(
        &req.model,
        &named_models,
        effective_reasoning_effort,
        &model_info,
        &model,
    );
    // If the named model lookup resolved to a different slug, build matching model info.
    let resolved_model_info = if resolved_model != model {
        proxy_model_info(&resolved_model)
    } else {
        model_info
    };

    // Build prompt.
    let tool_names: Vec<String> = req.tools.iter().map(|t| t.name.clone()).collect();
    let items_count = req.messages.len();

    let prompt = match translate_request(&req, &model) {
        Ok(p) => p,
        Err(e) => {
            log_error(
                &state,
                &model,
                log_effort,
                reasoning_source,
                items_count,
                &tool_names,
                stream_mode,
                format!("Failed to translate request: {e}"),
                None,
                session_id.as_deref(),
            )
            .await;
            return anthropic_error_response(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                format!("Bad request: {e}"),
                &req_ctx,
                Some(RetryHints {
                    should_retry: false,
                    retry_after_secs: None,
                }),
            );
        }
    };

    // Log the incoming request.
    {
        let entry = HttpLogEntry {
            timestamp: Utc::now(),
            model: resolved_model.clone(),
            reasoning_effort: log_effort,
            reasoning_source: reasoning_source.to_string(),
            items: items_count,
            tools: tool_names.clone(),
            streaming: stream_mode,
            session_id: session_id.clone(),
            status: None,
            stop_reason: None,
            input_tokens: None,
            output_tokens: None,
            error: None,
        };
        tracing::info!("{}", entry.request_line());
        append_log(&state.log_buffer, entry).await;
    }

    // Compute scale factor to normalise codex's effective context window to Claude Code's assumed 200k.
    let context_window_scale = {
        let codex_window = resolved_model_info.context_window.unwrap_or(272_000) as f64;
        let effective = resolved_model_info.effective_context_window_percent as f64 / 100.0;
        CLAUDE_CODE_ASSUMED_CONTEXT_WINDOW / (codex_window * effective)
    };

    // Resolve which ModelClient to use for this request.
    //
    // If the caller sends `X-Codex-Session-Id`, we reuse a cached client for
    // that session so WebSocket connections (and server-side sticky routing) are
    // preserved across tool-call turns. Each agent session is isolated from
    // others via its own `conversation_id`.
    //
    // Without the header we check out a pre-warmed client from the pool (or
    // create a fresh one). Each such request gets its own `conversation_id`,
    // preventing backend serialisation of unrelated concurrent requests and
    // eliminating stale-WebSocket reuse.
    let key = transport_key(&provider, enable_request_compression);

    let client = if let Some(ref sid) = session_id {
        // Take the entry out of the cache (preventing concurrent reuse).
        let entry = state.session_cache.lock().await.remove(sid);
        match entry {
            Some(e) if e.key == key => e.client,
            _ => {
                checkout_or_create_client(
                    &state,
                    &provider,
                    enable_request_compression,
                    &resolved_model_info,
                    beta_features_header,
                )
                .await
            }
        }
    } else {
        checkout_or_create_client(
            &state,
            &provider,
            enable_request_compression,
            &resolved_model_info,
            beta_features_header,
        )
        .await
    };
    let mut session = client.new_session();
    session.prompt_cache_key = session_id.clone();

    let service_tier = parse_service_tier(&req);
    let stream_result = session
        .stream(
            &prompt,
            &resolved_model_info,
            &state.session_telemetry,
            stream_effort,
            reasoning_summary,
            service_tier,
            None,
        )
        .await;

    let mut response_stream = match stream_result {
        Ok(s) => s,
        Err(e) => {
            let msg = format!("Stream error: {e}");
            log_error(
                &state,
                &resolved_model,
                log_effort,
                reasoning_source,
                items_count,
                &tool_names,
                stream_mode,
                msg.clone(),
                Some(503),
                session_id.as_deref(),
            )
            .await;
            return anthropic_error_response(
                StatusCode::SERVICE_UNAVAILABLE,
                "api_error",
                msg,
                &req_ctx,
                Some(stream_retry_hints()),
            );
        }
    };

    if stream_mode {
        // --- Streaming response: direct stream mapping, no mpsc bridge ---
        let log_buffer = state.log_buffer.clone();
        let state_for_stream = Arc::clone(&state);
        let model_clone = resolved_model.clone();

        let req_ctx_for_stream = req_ctx.clone();
        let request_model_for_stream = request_model.clone();
        let estimated_tokens = req.estimate_input_tokens();
        let sse_stream = async_stream::stream! {
            let mut translator = StreamTranslator::new(request_model_for_stream, context_window_scale, estimated_tokens);
            let mut had_stream_error = false;
            let mut ping_interval = tokio::time::interval(Duration::from_secs(STREAM_PING_INTERVAL_SECS));
            ping_interval.set_missed_tick_behavior(MissedTickBehavior::Delay);

            loop {
                tokio::select! {
                    _ = ping_interval.tick() => {
                        yield Ok::<Bytes, std::io::Error>(Bytes::from(format_sse("ping", &serde_json::json!({"type": "ping"}))));
                    }
                    maybe_event = response_stream.next() => {
                        match maybe_event {
                            Some(Ok(ev)) => {
                                let sse_chunk = translator.translate(&ev);
                                if !sse_chunk.is_empty() {
                                    yield Ok::<Bytes, std::io::Error>(Bytes::from(sse_chunk));
                                }
                            }
                            Some(Err(e)) => {
                                tracing::error!("Stream error: {e}");
                                if matches!(e.downcast_ref::<CodexErr>(), Some(CodexErr::ContextWindowExceeded)) {
                                    let chunks = translator.terminate_with_reason("model_context_window_exceeded");
                                    yield Ok::<Bytes, std::io::Error>(Bytes::from(chunks));
                                } else {
                                    had_stream_error = true;
                                    let error_payload = anthropic_error_body("api_error", &format!("Stream error: {e}"), &req_ctx_for_stream.request_id);
                                    yield Ok::<Bytes, std::io::Error>(Bytes::from(format_sse("error", &error_payload)));
                                }
                                break;
                            }
                            None => break,
                        }
                    }
                }
            }

            // Drop the session so the WebSocket connection is stored back into
            // `client` via `ModelClientSession::drop` before we return the
            // client to the session cache.
            drop(session);

            // Return the client to the session cache (if this was a named session).
            if let Some(ref sid) = session_id {
                cache_session(&state_for_stream.session_cache, sid, client, key.clone()).await;
            }

            // Emit final log entry after the stream is exhausted.
            let entry = HttpLogEntry {
                timestamp: Utc::now(),
                model: model_clone,
                reasoning_effort: log_effort,
                reasoning_source: reasoning_source.to_string(),
                items: 0,
                tools: Vec::new(),
                streaming: true,
                session_id: session_id.clone(),
                status: Some(if had_stream_error { 502 } else { 200 }),
                stop_reason: Some(translator.stop_reason().to_string()),
                input_tokens: translator.input_tokens(),
                output_tokens: translator.output_tokens(),
                error: if had_stream_error { Some("stream_error".to_string()) } else { None },
            };
            let response_line = entry.response_line();
            append_log(&log_buffer, entry).await;
            tracing::info!("{}", response_line);
        };

        let body = Body::from_stream(sse_stream);
        attach_correlation_headers(Response::builder().status(200), &req_ctx)
            .header("content-type", "text/event-stream")
            .header("cache-control", "no-cache")
            .header("connection", "keep-alive")
            .body(body)
            .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response())
    } else {
        // --- Non-streaming: accumulate and return JSON ---
        // Non-streaming: real tokens arrive before build_response, so no estimate needed.
        let mut translator = StreamTranslator::new(request_model.clone(), context_window_scale, 0);
        while let Some(event) = response_stream.next().await {
            match event {
                Ok(ev) => {
                    translator.consume_event(&ev);
                }
                Err(e) => {
                    if matches!(e.downcast_ref::<CodexErr>(), Some(CodexErr::ContextWindowExceeded)) {
                        translator.set_stop_reason_override("model_context_window_exceeded");
                        break;
                    }
                    let msg = format!("Stream error: {e}");
                    log_error(
                        &state,
                        &resolved_model,
                        log_effort,
                        reasoning_source,
                        items_count,
                        &tool_names,
                        false,
                        msg.clone(),
                        Some(502),
                        session_id.as_deref(),
                    )
                    .await;
                    return anthropic_error_response(
                        StatusCode::BAD_GATEWAY,
                        "api_error",
                        msg,
                        &req_ctx,
                        Some(stream_retry_hints()),
                    );
                }
            }
        }

        // Drop the session so the WebSocket connection is stored back into
        // `client` via `ModelClientSession::drop` before we return the
        // client to the session cache.
        drop(session);

        // Return the client to the session cache (if this was a named session).
        if let Some(ref sid) = session_id {
            cache_session(&state.session_cache, sid, client, key).await;
        }

        let message = translator.build_response(&request_model);
        let body = Body::from(
            serde_json::to_vec(&message).unwrap_or_else(|_| b"{\"type\":\"error\"}".to_vec()),
        );
        let entry = HttpLogEntry {
            timestamp: Utc::now(),
            model: resolved_model.clone(),
            reasoning_effort: log_effort,
            reasoning_source: reasoning_source.to_string(),
            items: items_count,
            tools: tool_names,
            streaming: false,
            session_id: session_id.clone(),
            status: Some(200),
            stop_reason: Some(translator.stop_reason().to_string()),
            input_tokens: translator.input_tokens(),
            output_tokens: translator.output_tokens(),
            error: None,
        };
        tracing::info!("{}", entry.response_line());
        append_log(&state.log_buffer, entry).await;

        attach_correlation_headers(Response::builder().status(200), &req_ctx)
            .header("content-type", "application/json")
            .body(body)
            .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response())
    }
}

fn anthropic_error_body(error_type: &str, message: &str, request_id: &str) -> Value {
    serde_json::json!({
        "type": "error",
        "error": {
            "type": error_type,
            "message": message
        },
        "request_id": request_id
    })
}

fn anthropic_error_response(
    status: StatusCode,
    error_type: &str,
    message: String,
    ctx: &RequestContext,
    retry_hints: Option<RetryHints>,
) -> Response {
    let mut builder = Response::builder().status(status);
    builder = attach_correlation_headers(builder, ctx);
    builder = builder.header("content-type", "application/json");

    if let Some(hints) = retry_hints {
        builder = builder.header(
            "x-should-retry",
            if hints.should_retry { "true" } else { "false" },
        );
        if let Some(retry_after) = hints.retry_after_secs {
            builder = builder.header("retry-after", retry_after.to_string());
        }
    }

    let body = Body::from(
        serde_json::to_vec(&anthropic_error_body(error_type, &message, &ctx.request_id))
            .unwrap_or_else(|_| b"{\"type\":\"error\"}".to_vec()),
    );

    builder
        .body(body)
        .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response())
}

async fn log_error(
    state: &HttpServerState,
    model: &str,
    reasoning_effort: ReasoningEffortConfig,
    reasoning_source: &'static str,
    items: usize,
    tools: &[String],
    streaming: bool,
    error: String,
    status: Option<u16>,
    session_id: Option<&str>,
) {
    tracing::error!("{}", error);
    let entry = HttpLogEntry {
        timestamp: Utc::now(),
        model: model.to_string(),
        reasoning_effort,
        reasoning_source: reasoning_source.to_string(),
        items,
        tools: tools.to_vec(),
        streaming,
        session_id: session_id.map(str::to_owned),
        status,
        stop_reason: None,
        input_tokens: None,
        output_tokens: None,
        error: Some(error),
    };
    append_log(&state.log_buffer, entry).await;
}

/// Build the Axum router for the HTTP proxy server.
pub fn build_router(state: Arc<HttpServerState>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(list_models))
        .route("/v1/messages", post(handle_messages))
        .with_state(state)
}
