//! HTTP server module providing an Anthropic-compatible `/v1/messages` proxy.
//!
//! When the user runs `/http-server on`, this module starts an Axum server on
//! port 8082 that:
//!
//!  * `GET  /health`       — liveness probe
//!  * `GET  /v1/models`    — returns the currently-selected model
//!  * `POST /v1/messages`  — Anthropic-format chat completion (streaming or not)
//!
//! Requests are translated from the Anthropic wire format into the internal
//! `Prompt` / `ResponseItem` types and forwarded through the existing
//! `ModelClient` / `ModelClientSession` streaming infrastructure that talks to
//! `chatgpt.com/backend-api/codex/responses`.  Responses are translated back
//! to Anthropic SSE format matching the output produced by the `codex-wrapper`
//! TypeScript proxy project.
//!
//! Log entries are written to a bounded `VecDeque` shared with the TUI so that
//! `/http-server logs` can display a live scrollable overlay.

use std::collections::HashMap;
use std::collections::VecDeque;
use std::sync::Arc;

use axum::Router;
use axum::body::Body;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::response::Response;
use axum::routing::get;
use axum::routing::post;
use chrono::DateTime;
use chrono::Utc;
use axum::body::Bytes;
use codex_core::AuthManager;
use codex_core::ModelClient;
use codex_core::ModelProviderInfo;
use codex_core::Prompt;
use codex_core::ResponseEvent;
use codex_core::ResponsesWebsocketVersion;
use codex_otel::OtelManager;
use codex_protocol::ThreadId;
use codex_protocol::config_types::ReasoningSummary as ReasoningSummaryConfig;
use codex_protocol::models::BaseInstructions;
use codex_protocol::models::ContentItem;
use codex_protocol::models::FunctionCallOutputBody;
use codex_protocol::models::FunctionCallOutputPayload;
use codex_protocol::models::ResponseItem;
use codex_protocol::openai_models::ConfigShellToolType;
use codex_protocol::openai_models::ModelInfo;
use codex_protocol::openai_models::ModelVisibility;
use codex_protocol::openai_models::ReasoningEffort as ReasoningEffortConfig;
use codex_protocol::openai_models::TruncationPolicyConfig;
use codex_protocol::openai_models::default_input_modalities;
use codex_protocol::protocol::SessionSource;
use serde::Deserialize;
use serde_json::Value;
use tokio_stream::StreamExt;
use tokio::sync::Mutex;
use tokio::sync::RwLock;
use uuid::Uuid;

/// Maximum number of log entries retained in memory.
const LOG_BUFFER_MAX: usize = 500;

// ---------------------------------------------------------------------------
// Log entry
// ---------------------------------------------------------------------------

/// A single request/response log entry for the HTTP server overlay.
#[derive(Debug, Clone)]
pub struct HttpLogEntry {
    pub timestamp: DateTime<Utc>,
    pub model: String,
    pub reasoning_effort: ReasoningEffortConfig,
    pub reasoning_source: String,
    pub items: usize,
    pub tools: Vec<String>,
    pub streaming: bool,
    pub status: Option<u16>,
    pub stop_reason: Option<String>,
    pub input_tokens: Option<u32>,
    pub output_tokens: Option<u32>,
    pub error: Option<String>,
}

impl HttpLogEntry {
    /// The `→` line emitted when a request arrives (codex-wrapper format).
    pub fn request_line(&self) -> String {
        let tools = self.tools.join(",");
        format!(
            "[codex] → model={} effort={} effort_src={} items={} tools=[{}] client={}",
            self.model,
            self.reasoning_effort,
            self.reasoning_source,
            self.items,
            tools,
            if self.streaming { "stream" } else { "sync" },
        )
    }

    /// The `←` line emitted when a response finishes.
    pub fn response_line(&self) -> String {
        if let Some(err) = &self.error {
            format!("[codex] ✗ {}", err)
        } else {
            format!(
                "[codex] ← {}  done stop_reason={} in={} out={}",
                self.status.unwrap_or(0),
                self.stop_reason.as_deref().unwrap_or("?"),
                self.input_tokens.unwrap_or(0),
                self.output_tokens.unwrap_or(0),
            )
        }
    }
}

// ---------------------------------------------------------------------------
// Dynamic config (updated from the TUI when model / reasoning changes)
// ---------------------------------------------------------------------------

/// TUI-side settings that the HTTP handler needs to forward calls correctly.
#[derive(Clone)]
pub struct HttpServerDynamicConfig {
    /// Model slug to forward to (e.g. `"gpt-5.3-codex"`).
    pub model: String,
    /// Provider configuration (base URL, auth method, etc.).
    pub provider: ModelProviderInfo,
    /// Optional reasoning effort override.
    pub reasoning_effort: Option<ReasoningEffortConfig>,
    /// Reasoning summary mode.
    pub reasoning_summary: ReasoningSummaryConfig,
    /// Optional WebSocket version for the Responses API transport.
    pub ws_version: Option<ResponsesWebsocketVersion>,
}

// ---------------------------------------------------------------------------
// Shared state
// ---------------------------------------------------------------------------

/// State shared across all HTTP handler invocations.
pub struct HttpServerState {
    pub auth_manager: Arc<AuthManager>,
    pub dynamic_config: Arc<RwLock<HttpServerDynamicConfig>>,
    /// Bounded ring buffer of log entries read by the TUI overlay.
    pub log_buffer: Arc<Mutex<VecDeque<HttpLogEntry>>>,
    pub otel_manager: OtelManager,
}

// ---------------------------------------------------------------------------
// Anthropic request / response wire types
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct AnthropicRequest {
    #[serde(default)]
    model: String,
    messages: Vec<AnthropicMessage>,
    #[serde(default)]
    system: Option<Value>,
    #[serde(default)]
    tools: Vec<AnthropicTool>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    thinking: Option<AnthropicThinking>,
    #[allow(dead_code)]
    #[serde(default)]
    max_tokens: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct AnthropicMessage {
    role: String,
    content: AnthropicContent,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum AnthropicContent {
    Text(String),
    Blocks(Vec<AnthropicBlock>),
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AnthropicBlock {
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    ToolResult {
        tool_use_id: String,
        #[serde(default)]
        content: AnthropicToolResultContent,
    },
    Thinking {
        #[allow(dead_code)]
        thinking: String,
    },
}

#[derive(Debug, Deserialize, Default)]
#[serde(untagged)]
enum AnthropicToolResultContent {
    #[default]
    Empty,
    Text(String),
    Blocks(Vec<AnthropicTextBlock>),
}

#[derive(Debug, Deserialize)]
struct AnthropicTextBlock {
    #[serde(rename = "type")]
    _type: String,
    text: String,
}

#[derive(Debug, Deserialize)]
struct AnthropicTool {
    name: String,
    #[serde(default)]
    description: String,
    input_schema: Value,
}

#[derive(Debug, Deserialize)]
struct AnthropicThinking {
    #[serde(rename = "type")]
    _type: String,
    #[serde(default)]
    budget_tokens: Option<u32>,
}

// ---------------------------------------------------------------------------
// Request translation: Anthropic → internal Prompt
// ---------------------------------------------------------------------------

/// Translate an Anthropic request body into the internal `Prompt` type.
fn translate_request(req: &AnthropicRequest, _model: &str) -> anyhow::Result<Prompt> {
    let system_text = match &req.system {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(blocks)) => {
            // Concatenate all text blocks in the system prompt array.
            blocks
                .iter()
                .filter_map(|b| {
                    if b.get("type").and_then(Value::as_str) == Some("text") {
                        b.get("text").and_then(Value::as_str).map(str::to_string)
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>()
                .join("\n")
        }
        _ => String::new(),
    };

    let mut prompt = Prompt::default();
    prompt.input = translate_messages(&req.messages);
    prompt.base_instructions = BaseInstructions { text: system_text };

    // Tools
    for tool in &req.tools {
        prompt.add_function_tool(
            tool.name.clone(),
            tool.description.clone(),
            tool.input_schema.clone(),
        )?;
    }
    prompt.set_parallel_tool_calls(true);

    Ok(prompt)
}

/// Convert Anthropic messages into `ResponseItem` list.
fn translate_messages(messages: &[AnthropicMessage]) -> Vec<ResponseItem> {
    let mut items: Vec<ResponseItem> = Vec::new();

    for msg in messages {
        match &msg.content {
            AnthropicContent::Text(text) => {
                let content_item = if msg.role == "user" {
                    ContentItem::InputText { text: text.clone() }
                } else {
                    ContentItem::OutputText { text: text.clone() }
                };
                items.push(ResponseItem::Message {
                    id: None,
                    role: msg.role.clone(),
                    content: vec![content_item],
                    end_turn: None,
                    phase: None,
                });
            }
            AnthropicContent::Blocks(blocks) => {
                let mut text_content: Vec<ContentItem> = Vec::new();

                for block in blocks {
                    match block {
                        AnthropicBlock::Thinking { .. } => {
                            // Skip thinking blocks from history.
                        }
                        AnthropicBlock::Text { text } => {
                            let item = if msg.role == "user" {
                                ContentItem::InputText { text: text.clone() }
                            } else {
                                ContentItem::OutputText { text: text.clone() }
                            };
                            text_content.push(item);
                        }
                        AnthropicBlock::ToolUse { id, name, input } => {
                            // Flush accumulated text blocks first.
                            if !text_content.is_empty() {
                                items.push(ResponseItem::Message {
                                    id: None,
                                    role: msg.role.clone(),
                                    content: std::mem::take(&mut text_content),
                                    end_turn: None,
                                    phase: None,
                                });
                            }
                            items.push(ResponseItem::FunctionCall {
                                id: None,
                                name: name.clone(),
                                arguments: serde_json::to_string(input)
                                    .unwrap_or_else(|_| "{}".to_string()),
                                call_id: id.clone(),
                            });
                        }
                        AnthropicBlock::ToolResult {
                            tool_use_id,
                            content,
                        } => {
                            // Flush accumulated text blocks first.
                            if !text_content.is_empty() {
                                items.push(ResponseItem::Message {
                                    id: None,
                                    role: msg.role.clone(),
                                    content: std::mem::take(&mut text_content),
                                    end_turn: None,
                                    phase: None,
                                });
                            }
                            let output_text = match content {
                                AnthropicToolResultContent::Empty => String::new(),
                                AnthropicToolResultContent::Text(t) => t.clone(),
                                AnthropicToolResultContent::Blocks(blocks) => blocks
                                    .iter()
                                    .map(|b| b.text.as_str())
                                    .collect::<Vec<_>>()
                                    .join("\n"),
                            };
                            items.push(ResponseItem::FunctionCallOutput {
                                call_id: tool_use_id.clone(),
                                output: FunctionCallOutputPayload {
                                    body: FunctionCallOutputBody::Text(output_text),
                                    success: None,
                                },
                            });
                        }
                    }
                }

                // Flush remaining text blocks.
                if !text_content.is_empty() {
                    items.push(ResponseItem::Message {
                        id: None,
                        role: msg.role.clone(),
                        content: text_content,
                        end_turn: None,
                        phase: None,
                    });
                }
            }
        }
    }

    items
}

// ---------------------------------------------------------------------------
// Build a minimal ModelInfo for the proxy model
// ---------------------------------------------------------------------------

/// Construct a minimal `ModelInfo` sufficient for `ModelClientSession::stream`.
///
/// Since we are proxying to a Codex model that supports reasoning, we hard-code
/// `supports_reasoning_summaries = true` and `support_verbosity = true`.
fn proxy_model_info(slug: &str) -> ModelInfo {
    ModelInfo {
        slug: slug.to_string(),
        display_name: slug.to_string(),
        description: None,
        default_reasoning_level: Some(ReasoningEffortConfig::Medium),
        supported_reasoning_levels: Vec::new(),
        shell_type: ConfigShellToolType::Default,
        visibility: ModelVisibility::None,
        supported_in_api: true,
        priority: 99,
        upgrade: None,
        base_instructions: String::new(),
        model_messages: None,
        supports_reasoning_summaries: true,
        support_verbosity: true,
        default_verbosity: None,
        apply_patch_tool_type: None,
        truncation_policy: TruncationPolicyConfig::bytes(100_000),
        supports_parallel_tool_calls: true,
        context_window: Some(200_000),
        auto_compact_token_limit: None,
        effective_context_window_percent: 95,
        experimental_supported_tools: Vec::new(),
        input_modalities: default_input_modalities(),
        prefer_websockets: false,
        used_fallback_model_metadata: true,
    }
}

// ---------------------------------------------------------------------------
// Reasoning effort helpers
// ---------------------------------------------------------------------------

fn budget_to_effort(budget: u32) -> ReasoningEffortConfig {
    if budget <= 2000 {
        ReasoningEffortConfig::Low
    } else if budget <= 8000 {
        ReasoningEffortConfig::Medium
    } else {
        ReasoningEffortConfig::High
    }
}

// ---------------------------------------------------------------------------
// Axum handlers
// ---------------------------------------------------------------------------

async fn health() -> &'static str {
    "ok"
}

async fn list_models(State(state): State<Arc<HttpServerState>>) -> impl IntoResponse {
    let cfg = state.dynamic_config.read().await;
    let model_id = cfg.model.clone();
    drop(cfg);
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

async fn handle_messages(
    State(state): State<Arc<HttpServerState>>,
    body: axum::extract::Json<AnthropicRequest>,
) -> Response {
    let req = body.0;
    let stream_mode = req.stream;

    // Read dynamic config snapshot.
    let (model, provider, reasoning_effort, reasoning_summary, ws_version) = {
        let cfg = state.dynamic_config.read().await;
        (
            cfg.model.clone(),
            cfg.provider.clone(),
            cfg.reasoning_effort,
            cfg.reasoning_summary.clone(),
            cfg.ws_version,
        )
    };

    // Resolve reasoning effort for logs.
    let model_info = proxy_model_info(&model);
    let request_budget_effort = req
        .thinking
        .as_ref()
        .and_then(|t| t.budget_tokens)
        .map(budget_to_effort);
    let (resolved_reasoning_effort, reasoning_source) = if let Some(effort) = request_budget_effort {
        (effort, "thinking".to_string())
    } else if let Some(effort) = reasoning_effort {
        (effort, "config".to_string())
    } else {
        (
            model_info
                .default_reasoning_level
                .unwrap_or(ReasoningEffortConfig::Medium),
            "model_default".to_string(),
        )
    };

    // Keep stream behavior unchanged: request thinking override (with fallback budget),
    // then dynamic config override.
    let effort = req
        .thinking
        .as_ref()
        .map(|t| budget_to_effort(t.budget_tokens.unwrap_or(10_000)))
        .or(reasoning_effort);

    // Build prompt.
    let tool_names: Vec<String> = req.tools.iter().map(|t| t.name.clone()).collect();
    let items_count = req.messages.len();

    let prompt = match translate_request(&req, &model) {
        Ok(p) => p,
        Err(e) => {
            log_error(
                &state,
                &model,
                resolved_reasoning_effort,
                reasoning_source.as_str(),
                items_count,
                &tool_names,
                stream_mode,
                format!("Failed to translate request: {e}"),
            )
            .await;
            return (
                StatusCode::BAD_REQUEST,
                format!("Bad request: {e}"),
            )
                .into_response();
        }
    };

    // Log the incoming request.
    {
        let entry = HttpLogEntry {
            timestamp: Utc::now(),
            model: model.clone(),
            reasoning_effort: resolved_reasoning_effort,
            reasoning_source: reasoning_source.clone(),
            items: items_count,
            tools: tool_names.clone(),
            streaming: stream_mode,
            status: None,
            stop_reason: None,
            input_tokens: None,
            output_tokens: None,
            error: None,
        };
        tracing::info!("{}", entry.request_line());
        let mut buf = state.log_buffer.lock().await;
        if buf.len() >= LOG_BUFFER_MAX {
            buf.pop_front();
        }
        buf.push_back(entry);
    }

    // Build ModelClient and stream.
    let conversation_id = ThreadId::new();
    let client = ModelClient::new(
        Some(state.auth_manager.clone()),
        conversation_id,
        provider,
        SessionSource::Cli,
        None, // model_verbosity
        ws_version,
        false, // enable_request_compression
        false, // include_timing_metrics
        None,  // beta_features_header
    );
    let mut session = client.new_session();

    let stream_result = session
        .stream(&prompt, &model_info, &state.otel_manager, effort, reasoning_summary, None)
        .await;

    let mut response_stream = match stream_result {
        Ok(s) => s,
        Err(e) => {
            let msg = format!("Stream error: {e}");
            log_error_complete(
                &state,
                &model,
                resolved_reasoning_effort,
                reasoning_source.as_str(),
                items_count,
                &tool_names,
                stream_mode,
                msg.clone(),
                503,
            )
            .await;
            return (StatusCode::SERVICE_UNAVAILABLE, msg).into_response();
        }
    };

    if stream_mode {
        // --- Streaming response ---
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<Bytes, std::io::Error>>(64);
        let log_buffer = state.log_buffer.clone();
        let model_clone = model.clone();
        let reasoning_effort_clone = resolved_reasoning_effort;
        let reasoning_source_clone = reasoning_source.clone();

        tokio::spawn(async move {
            let mut translator = StreamTranslator::new(model_clone.clone());

            while let Some(event) = response_stream.next().await {
                match event {
                    Ok(ev) => {
                        let sse_chunk = translator.translate(&ev);
                        if !sse_chunk.is_empty() {
                            let _ = tx.send(Ok(Bytes::from(sse_chunk))).await;
                        }
                    }
                    Err(e) => {
                        tracing::error!("Stream error: {e}");
                        break;
                    }
                }
            }

            // Emit final log entry.
            let response_line = {
                let entry = HttpLogEntry {
                    timestamp: Utc::now(),
                    model: model_clone.clone(),
                    reasoning_effort: reasoning_effort_clone,
                    reasoning_source: reasoning_source_clone.clone(),
                    items: 0,
                    tools: Vec::new(),
                    streaming: true,
                    status: Some(200),
                    stop_reason: Some(translator.stop_reason()),
                    input_tokens: translator.input_tokens(),
                    output_tokens: translator.output_tokens(),
                    error: None,
                };
                let line = entry.response_line();
                let mut buf = log_buffer.lock().await;
                if buf.len() >= LOG_BUFFER_MAX {
                    buf.pop_front();
                }
                buf.push_back(entry);
                line
            };
            tracing::info!("{}", response_line);
        });

        let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
        let body = Body::from_stream(stream);
        Response::builder()
            .status(200)
            .header("content-type", "text/event-stream")
            .header("cache-control", "no-cache")
            .header("connection", "keep-alive")
            .body(body)
            .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response())
    } else {
        // --- Non-streaming: accumulate and return JSON ---
        let mut translator = StreamTranslator::new(model.clone());
        while let Some(event) = response_stream.next().await {
            match event {
                Ok(ev) => {
                    translator.translate(&ev);
                }
                Err(e) => {
                    let msg = format!("Stream error: {e}");
                    log_error_complete(
                        &state,
                        &model,
                        resolved_reasoning_effort,
                        reasoning_source.as_str(),
                        items_count,
                        &tool_names,
                        false,
                        msg.clone(),
                        502,
                    )
                    .await;
                    return (StatusCode::BAD_GATEWAY, msg).into_response();
                }
            }
        }

        let message = translator.build_response(&req.model);
        let entry = HttpLogEntry {
            timestamp: Utc::now(),
            model: model.clone(),
            reasoning_effort: resolved_reasoning_effort,
            reasoning_source: reasoning_source.clone(),
            items: items_count,
            tools: tool_names,
            streaming: false,
            status: Some(200),
            stop_reason: Some(translator.stop_reason()),
            input_tokens: translator.input_tokens(),
            output_tokens: translator.output_tokens(),
            error: None,
        };
        tracing::info!("{}", entry.response_line());
        {
            let mut buf = state.log_buffer.lock().await;
            if buf.len() >= LOG_BUFFER_MAX {
                buf.pop_front();
            }
            buf.push_back(entry);
        }

        axum::Json(message).into_response()
    }
}

// ---------------------------------------------------------------------------
// Logging helpers
// ---------------------------------------------------------------------------

async fn log_error(
    state: &HttpServerState,
    model: &str,
    reasoning_effort: ReasoningEffortConfig,
    reasoning_source: &str,
    items: usize,
    tools: &[String],
    streaming: bool,
    error: String,
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
        status: None,
        stop_reason: None,
        input_tokens: None,
        output_tokens: None,
        error: Some(error),
    };
    let mut buf = state.log_buffer.lock().await;
    if buf.len() >= LOG_BUFFER_MAX {
        buf.pop_front();
    }
    buf.push_back(entry);
}

async fn log_error_complete(
    state: &HttpServerState,
    model: &str,
    reasoning_effort: ReasoningEffortConfig,
    reasoning_source: &str,
    items: usize,
    tools: &[String],
    streaming: bool,
    error: String,
    status: u16,
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
        status: Some(status),
        stop_reason: None,
        input_tokens: None,
        output_tokens: None,
        error: Some(error),
    };
    let mut buf = state.log_buffer.lock().await;
    if buf.len() >= LOG_BUFFER_MAX {
        buf.pop_front();
    }
    buf.push_back(entry);
}

// ---------------------------------------------------------------------------
// ResponseEvent → Anthropic SSE translator
// ---------------------------------------------------------------------------

/// Translates `ResponseEvent` stream into Anthropic SSE text chunks.
struct StreamTranslator {
    model_slug: String,
    message_id: String,
    /// Next block index to assign.
    next_block: usize,
    /// Block index currently receiving text deltas.
    current_text_block: Option<usize>,
    /// Map from function call_id → block index.
    call_id_to_block: HashMap<String, usize>,
    /// True if at least one function_call was received.
    has_function_call: bool,
    input_tokens: u32,
    output_tokens: u32,
    /// Accumulated content for non-streaming path.
    blocks: Vec<BlockState>,
}

#[derive(Debug)]
enum BlockState {
    Text { text: String },
    ToolUse { id: String, name: String, input: String },
}

impl StreamTranslator {
    fn new(model_slug: String) -> Self {
        Self {
            model_slug,
            message_id: format!("msg_{}", Uuid::new_v4().simple()),
            next_block: 0,
            current_text_block: None,
            call_id_to_block: HashMap::new(),
            has_function_call: false,
            input_tokens: 0,
            output_tokens: 0,
            blocks: Vec::new(),
        }
    }

    fn stop_reason(&self) -> String {
        if self.has_function_call {
            "tool_use".to_string()
        } else {
            "end_turn".to_string()
        }
    }

    fn input_tokens(&self) -> Option<u32> {
        if self.input_tokens > 0 {
            Some(self.input_tokens)
        } else {
            None
        }
    }

    fn output_tokens(&self) -> Option<u32> {
        if self.output_tokens > 0 {
            Some(self.output_tokens)
        } else {
            None
        }
    }

    /// Translate a single `ResponseEvent` into one or more Anthropic SSE chunks.
    ///
    /// Returns an empty string when the event produces no output.
    fn translate(&mut self, event: &ResponseEvent) -> String {
        match event {
            ResponseEvent::Created => self.handle_created(),
            ResponseEvent::OutputItemAdded(item) => self.handle_item_added(item),
            ResponseEvent::OutputTextDelta(delta) => self.handle_text_delta(delta),
            ResponseEvent::OutputItemDone(item) => self.handle_item_done(item),
            ResponseEvent::Completed { token_usage, .. } => {
                if let Some(usage) = token_usage {
                    self.input_tokens = usage.input_tokens as u32;
                    self.output_tokens = usage.output_tokens as u32;
                }
                self.handle_completed()
            }
            // Ignore the rest.
            _ => String::new(),
        }
    }

    fn handle_created(&mut self) -> String {
        let message_start = serde_json::json!({
            "type": "message_start",
            "message": {
                "id": self.message_id,
                "type": "message",
                "role": "assistant",
                "model": self.model_slug,
                "content": [],
                "stop_reason": null,
                "stop_sequence": null,
                "usage": { "input_tokens": self.input_tokens, "output_tokens": 0 }
            }
        });
        format_sse("message_start", &message_start) + &format_sse("ping", &serde_json::json!({"type": "ping"}))
    }

    fn handle_item_added(&mut self, item: &ResponseItem) -> String {
        match item {
            ResponseItem::Message { .. } => {
                // Open a text block.
                let idx = self.next_block;
                self.next_block += 1;
                self.current_text_block = Some(idx);
                self.blocks.push(BlockState::Text { text: String::new() });
                let block_start = serde_json::json!({
                    "type": "content_block_start",
                    "index": idx,
                    "content_block": { "type": "text", "text": "" }
                });
                format_sse("content_block_start", &block_start)
            }
            ResponseItem::FunctionCall { name, call_id, .. } => {
                self.has_function_call = true;
                let idx = self.next_block;
                self.next_block += 1;
                self.call_id_to_block.insert(call_id.clone(), idx);
                let block_id = call_id.clone();
                self.blocks.push(BlockState::ToolUse {
                    id: block_id.clone(),
                    name: name.clone(),
                    input: String::new(),
                });
                let block_start = serde_json::json!({
                    "type": "content_block_start",
                    "index": idx,
                    "content_block": {
                        "type": "tool_use",
                        "id": block_id,
                        "name": name,
                        "input": {}
                    }
                });
                format_sse("content_block_start", &block_start)
            }
            _ => String::new(),
        }
    }

    fn handle_text_delta(&mut self, delta: &str) -> String {
        let Some(idx) = self.current_text_block else {
            return String::new();
        };
        // Accumulate for non-streaming path.
        if let Some(BlockState::Text { text }) = self.blocks.get_mut(idx) {
            text.push_str(delta);
        }
        let content_delta = serde_json::json!({
            "type": "content_block_delta",
            "index": idx,
            "delta": { "type": "text_delta", "text": delta }
        });
        format_sse("content_block_delta", &content_delta)
    }

    fn handle_item_done(&mut self, item: &ResponseItem) -> String {
        match item {
            ResponseItem::Message { .. } => {
                let Some(idx) = self.current_text_block.take() else {
                    return String::new();
                };
                let stop = serde_json::json!({
                    "type": "content_block_stop",
                    "index": idx
                });
                format_sse("content_block_stop", &stop)
            }
            ResponseItem::FunctionCall {
                call_id, arguments, ..
            } => {
                let Some(&idx) = self.call_id_to_block.get(call_id) else {
                    return String::new();
                };
                // Accumulate for non-streaming path.
                if let Some(BlockState::ToolUse { input, .. }) = self.blocks.get_mut(idx) {
                    *input = arguments.clone();
                }
                let args_delta = serde_json::json!({
                    "type": "content_block_delta",
                    "index": idx,
                    "delta": { "type": "input_json_delta", "partial_json": arguments }
                });
                let stop = serde_json::json!({
                    "type": "content_block_stop",
                    "index": idx
                });
                format_sse("content_block_delta", &args_delta)
                    + &format_sse("content_block_stop", &stop)
            }
            _ => String::new(),
        }
    }

    fn handle_completed(&self) -> String {
        let stop_reason = self.stop_reason();
        let message_delta = serde_json::json!({
            "type": "message_delta",
            "delta": { "stop_reason": stop_reason, "stop_sequence": null },
            "usage": { "output_tokens": self.output_tokens }
        });
        let message_stop = serde_json::json!({ "type": "message_stop" });
        format_sse("message_delta", &message_delta) + &format_sse("message_stop", &message_stop)
    }

    /// Build a complete non-streaming Anthropic response message.
    fn build_response(&self, original_model: &str) -> Value {
        let mut content: Vec<Value> = Vec::new();
        for block in &self.blocks {
            match block {
                BlockState::Text { text } => {
                    content.push(serde_json::json!({ "type": "text", "text": text }));
                }
                BlockState::ToolUse { id, name, input } => {
                    let parsed_input: Value = serde_json::from_str(input)
                        .unwrap_or(serde_json::json!({}));
                    content.push(serde_json::json!({
                        "type": "tool_use",
                        "id": id,
                        "name": name,
                        "input": parsed_input
                    }));
                }
            }
        }
        serde_json::json!({
            "id": self.message_id,
            "type": "message",
            "role": "assistant",
            "model": original_model,
            "content": content,
            "stop_reason": self.stop_reason(),
            "stop_sequence": null,
            "usage": {
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens
            }
        })
    }
}

fn format_sse(event_type: &str, data: &Value) -> String {
    format!(
        "event: {}\ndata: {}\n\n",
        event_type,
        serde_json::to_string(data).unwrap_or_default()
    )
}

// ---------------------------------------------------------------------------
// Router construction
// ---------------------------------------------------------------------------

/// Build the Axum router for the HTTP proxy server.
pub fn build_router(state: Arc<HttpServerState>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(list_models))
        .route("/v1/messages", post(handle_messages))
        .with_state(state)
}
