use std::collections::HashMap;
use std::str::FromStr;

use axum::http::HeaderValue;
use codex_protocol::config_types::ServiceTier;
use codex_protocol::openai_models::ModelInfo;
use codex_protocol::openai_models::ReasoningEffort as ReasoningEffortConfig;
use uuid::Uuid;

use super::wire_types::AnthropicRequest;

const DEFAULT_RETRY_AFTER_SECS: u32 = 2;

#[derive(Clone)]
pub(super) struct RequestContext {
    pub(super) request_id: String,
    pub(super) client_request_id: Option<String>,
}

pub(super) struct RetryHints {
    pub(super) should_retry: bool,
    pub(super) retry_after_secs: Option<u32>,
}

pub(super) fn request_context_from_headers(headers: &axum::http::HeaderMap) -> RequestContext {
    let client_request_id = headers
        .get("x-client-request-id")
        .and_then(|v| v.to_str().ok())
        .map(str::to_owned);
    RequestContext {
        request_id: format!("req_{}", Uuid::new_v4().simple()),
        client_request_id,
    }
}

pub(super) fn attach_correlation_headers(
    mut builder: axum::http::response::Builder,
    ctx: &RequestContext,
) -> axum::http::response::Builder {
    builder = builder
        .header("request-id", ctx.request_id.as_str())
        .header("x-request-id", ctx.request_id.as_str());
    if let Some(client_id) = &ctx.client_request_id {
        if let Ok(value) = HeaderValue::from_str(client_id) {
            builder = builder.header("x-client-request-id", value);
        }
    }
    builder
}

/// Resolve model slug, stream effort, log effort, and source for a request.
///
/// Resolution order:
/// 1. Named model lookup: if `requested_model` contains a key from `named_models`
///    (case-insensitive), use that entry's model slug and effort.
/// 2. TUI config effort with the default model slug.
/// 3. Model default effort with the default model slug.
///
/// Returns `(resolved_model, stream_effort, log_effort, source)`.
/// `stream_effort` is `None` when falling through to the model default so the
/// model can use its own default instead of being forced to Medium.
pub(super) fn resolve_for_request(
    requested_model: &str,
    named_models: &HashMap<String, (String, Option<ReasoningEffortConfig>)>,
    config_effort: Option<ReasoningEffortConfig>,
    model_info: &ModelInfo,
    default_model: &str,
) -> (
    String,
    Option<ReasoningEffortConfig>,
    ReasoningEffortConfig,
    &'static str,
) {
    let lower = requested_model.to_ascii_lowercase();
    if let Some((named_slug, named_effort)) = named_models
        .iter()
        .find(|(name, _)| lower.contains(name.as_str()))
        .map(|(_, cfg)| cfg)
    {
        let log_effort = named_effort.unwrap_or_else(|| {
            model_info
                .default_reasoning_level
                .unwrap_or(ReasoningEffortConfig::Medium)
        });
        (named_slug.clone(), *named_effort, log_effort, "named_model")
    } else if let Some(e) = config_effort {
        (default_model.to_string(), Some(e), e, "config")
    } else {
        let e = model_info
            .default_reasoning_level
            .unwrap_or(ReasoningEffortConfig::Medium);
        // Pass `None` so the model uses its own default rather than forcing Medium.
        (default_model.to_string(), None, e, "model_default")
    }
}

pub(super) fn parse_request_reasoning_effort(
    req: &AnthropicRequest,
) -> Option<ReasoningEffortConfig> {
    let output_effort = req
        .output_config
        .as_ref()
        .and_then(|cfg| cfg.get("effort"))
        .and_then(|v| v.as_str());
    let thinking_effort = req
        .thinking
        .as_ref()
        .and_then(|t| t.get("effort"))
        .and_then(|v| v.as_str());

    output_effort
        .or(thinking_effort)
        .and_then(|s| ReasoningEffortConfig::from_str(s).ok())
}

pub(super) fn parse_service_tier(req: &AnthropicRequest) -> Option<ServiceTier> {
    match req.speed.as_deref().map(str::to_ascii_lowercase).as_deref() {
        Some("fast") => Some(ServiceTier::Fast),
        Some("flex") => Some(ServiceTier::Flex),
        _ => None,
    }
}

pub(super) fn stream_retry_hints() -> RetryHints {
    RetryHints {
        should_retry: true,
        retry_after_secs: Some(DEFAULT_RETRY_AFTER_SECS),
    }
}
