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
use axum::body::Bytes;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::response::Response;
use axum::routing::get;
use axum::routing::post;
use chrono::DateTime;
use chrono::Utc;
use codex_core::AuthManager;
use codex_core::ModelClient;
use codex_core::ModelProviderInfo;
use codex_core::Prompt;
use codex_core::ResponseEvent;
use codex_otel::OtelManager;
use codex_protocol::ThreadId;
use codex_protocol::config_types::ReasoningSummary as ReasoningSummaryConfig;
use codex_protocol::models::BaseInstructions;
use codex_protocol::models::ContentItem;
use codex_protocol::models::FunctionCallOutputBody;
use codex_protocol::models::FunctionCallOutputContentItem;
use codex_protocol::models::FunctionCallOutputPayload;
use codex_protocol::models::ResponseItem;
use codex_protocol::openai_models::ConfigShellToolType;
use codex_protocol::openai_models::ModelInfo;
use codex_protocol::openai_models::ModelVisibility;
use codex_protocol::openai_models::ReasoningEffort as ReasoningEffortConfig;
use codex_protocol::openai_models::TruncationPolicyConfig;
use codex_protocol::openai_models::WebSearchToolType;
use codex_protocol::openai_models::default_input_modalities;
use codex_protocol::protocol::SessionSource;
use serde::Deserialize;
use serde_json::Value;
use tokio_stream::StreamExt;
use tokio::sync::Mutex;
use tokio::sync::RwLock;
use uuid::Uuid;

// ---------------------------------------------------------------------------
// Transport key and warm pool / session cache
// ---------------------------------------------------------------------------

/// Identifies the transport-relevant subset of configuration.
///
/// Used to invalidate warm-pool and session-cache entries when provider
/// config changes.
#[derive(Clone, PartialEq, Eq)]
struct TransportKey {
    provider_name: String,
    ws_enabled: bool,
    enable_request_compression: bool,
}

/// A pre-warmed `ModelClient` held in the warm pool.
///
/// Each entry is a *fresh* client (never used for a real request) so it
/// cannot carry a stale WebSocket connection.
struct WarmPoolEntry {
    client: ModelClient,
    key: TransportKey,
}

/// A cached `ModelClient` for an identified agent session.
///
/// Keyed by the `X-Codex-Session-Id` request header. Reusing the same
/// client across tool-call turns preserves the WebSocket connection (and
/// thus server-side sticky routing) for the duration of a single agent
/// conversation, while keeping each agent session isolated from others.
struct SessionCacheEntry {
    client: ModelClient,
    last_used: std::time::Instant,
    key: TransportKey,
}

/// Maximum number of log entries retained in memory.
const LOG_BUFFER_MAX: usize = 500;

/// Claude Code assumes this context window size (tokens) when computing usage percentage.
/// It is hardcoded in the Claude Code binary by model slug and cannot be overridden via API.
const CLAUDE_CODE_ASSUMED_CONTEXT_WINDOW: f64 = 200_000.0;

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
    /// Whether WebSocket transport is enabled for the Responses API.
    pub ws_enabled: bool,
    /// Whether to enable zstd request compression (mirrors core feature flag).
    pub enable_request_compression: bool,
    model_info_cache: ModelInfo,
}

impl HttpServerDynamicConfig {
    pub fn new(
        model: String,
        provider: ModelProviderInfo,
        reasoning_effort: Option<ReasoningEffortConfig>,
        reasoning_summary: Option<ReasoningSummaryConfig>,
        ws_enabled: bool,
        enable_request_compression: bool,
    ) -> Self {
        let model_info_cache = proxy_model_info(&model);
        Self {
            model,
            provider,
            reasoning_effort,
            reasoning_summary: reasoning_summary.unwrap_or(ReasoningSummaryConfig::Auto),
            ws_enabled,
            enable_request_compression,
            model_info_cache,
        }
    }

    pub fn set_model(&mut self, model: String) {
        self.model = model;
        self.refresh_model_info_cache();
    }

    /// Like `set_model` but uses real `ModelInfo` from the live models cache
    /// instead of the static proxy stub.
    pub fn set_model_with_info(&mut self, model: String, model_info: ModelInfo) {
        self.model = model;
        self.model_info_cache = model_info;
    }

    pub fn set_reasoning_effort(&mut self, reasoning_effort: Option<ReasoningEffortConfig>) {
        self.reasoning_effort = reasoning_effort;
        self.refresh_model_info_cache();
    }

    pub fn model_info(&self) -> &ModelInfo {
        &self.model_info_cache
    }

    fn refresh_model_info_cache(&mut self) {
        self.model_info_cache = proxy_model_info(&self.model);
    }
}

// ---------------------------------------------------------------------------
// Shared state
// ---------------------------------------------------------------------------

/// How many pre-warmed clients to keep in the pool.
const WARM_POOL_TARGET: usize = 2;

/// Evict session-cache entries idle for longer than this.
const SESSION_CACHE_TTL: std::time::Duration = std::time::Duration::from_secs(300);

/// State shared across all HTTP handler invocations.
pub struct HttpServerState {
    pub auth_manager: Arc<AuthManager>,
    pub dynamic_config: Arc<RwLock<HttpServerDynamicConfig>>,
    /// Bounded ring buffer of log entries read by the TUI overlay.
    pub log_buffer: Arc<Mutex<VecDeque<HttpLogEntry>>>,
    pub otel_manager: OtelManager,
    /// Pool of pre-warmed `ModelClient` instances for fast cold-start.
    ///
    /// Each entry is a fresh client (never used for a real request).
    /// The mutex is held only for the deque operation — never during I/O.
    warm_pool: Mutex<VecDeque<WarmPoolEntry>>,
    /// Per-agent-session `ModelClient` cache, keyed by `X-Codex-Session-Id`.
    ///
    /// Entries are taken out (removed) at request start and re-inserted
    /// after the response completes, so two concurrent requests for the
    /// same session ID never share a client.
    session_cache: Mutex<HashMap<String, SessionCacheEntry>>,
}

impl HttpServerState {
    pub fn new(
        auth_manager: Arc<AuthManager>,
        dynamic_config: Arc<RwLock<HttpServerDynamicConfig>>,
        log_buffer: Arc<Mutex<VecDeque<HttpLogEntry>>>,
        otel_manager: OtelManager,
    ) -> Self {
        Self {
            auth_manager,
            dynamic_config,
            log_buffer,
            otel_manager,
            warm_pool: Mutex::new(VecDeque::new()),
            session_cache: Mutex::new(HashMap::new()),
        }
    }
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
enum AnthropicImageSource {
    Base64 {
        media_type: String,
        data: String,
    },
    Url {
        url: String,
    },
}

impl AnthropicImageSource {
    fn to_image_url(&self) -> String {
        match self {
            Self::Base64 { media_type, data } => {
                format!("data:{media_type};base64,{data}")
            }
            Self::Url { url } => url.clone(),
        }
    }
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
    Image {
        source: AnthropicImageSource,
    },
    Document {
        #[serde(default)]
        title: Option<String>,
    },
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AnthropicToolResultBlock {
    Text { text: String },
    Image { source: AnthropicImageSource },
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Deserialize, Default)]
#[serde(untagged)]
enum AnthropicToolResultContent {
    #[default]
    Empty,
    Text(String),
    Blocks(Vec<AnthropicToolResultBlock>),
}

#[derive(Debug, Deserialize)]
struct AnthropicTool {
    #[serde(rename = "type", default)]
    tool_type: String,
    name: String,
    #[serde(default)]
    description: String,
    #[serde(default)]
    input_schema: Option<Value>,
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
        if tool.tool_type.starts_with("web_search_") {
            prompt.add_web_search_tool(true);
        } else if let Some(schema) = &tool.input_schema {
            prompt.add_function_tool(tool.name.clone(), tool.description.clone(), schema.clone())?;
        }
        // Other unknown built-in tool types (computer_use, etc.) are silently skipped
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
                        AnthropicBlock::Image { source } => {
                            let item = ContentItem::InputImage {
                                image_url: source.to_image_url(),
                            };
                            text_content.push(item);
                        }
                        AnthropicBlock::Document { title } => {
                            let label = title.as_deref().unwrap_or("untitled");
                            let placeholder =
                                format!("[Document: {label} — not supported by this proxy]");
                            let item = if msg.role == "user" {
                                ContentItem::InputText { text: placeholder }
                            } else {
                                ContentItem::OutputText { text: placeholder }
                            };
                            text_content.push(item);
                        }
                        AnthropicBlock::Unknown => {
                            // Silently skip unrecognised block types.
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
                            match content {
                                AnthropicToolResultContent::Empty => {
                                    items.push(ResponseItem::FunctionCallOutput {
                                        call_id: tool_use_id.clone(),
                                        output: FunctionCallOutputPayload {
                                            body: FunctionCallOutputBody::Text(String::new()),
                                            success: None,
                                        },
                                    });
                                }
                                AnthropicToolResultContent::Text(t) => {
                                    items.push(ResponseItem::FunctionCallOutput {
                                        call_id: tool_use_id.clone(),
                                        output: FunctionCallOutputPayload {
                                            body: FunctionCallOutputBody::Text(t.clone()),
                                            success: None,
                                        },
                                    });
                                }
                                AnthropicToolResultContent::Blocks(blocks) => {
                                    let content_items: Vec<FunctionCallOutputContentItem> =
                                        blocks
                                            .iter()
                                            .filter_map(|b| match b {
                                                AnthropicToolResultBlock::Text { text } => {
                                                    Some(FunctionCallOutputContentItem::InputText {
                                                        text: text.clone(),
                                                    })
                                                }
                                                AnthropicToolResultBlock::Image { source } => {
                                                    Some(
                                                        FunctionCallOutputContentItem::InputImage {
                                                            image_url: source.to_image_url(),
                                                            detail: None,
                                                        },
                                                    )
                                                }
                                                AnthropicToolResultBlock::Unknown => None,
                                            })
                                            .collect();
                                    items.push(ResponseItem::FunctionCallOutput {
                                        call_id: tool_use_id.clone(),
                                        output: FunctionCallOutputPayload::from_content_items(
                                            content_items,
                                        ),
                                    });
                                }
                            }
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
        availability_nux: None,
        supports_reasoning_summaries: true,
        default_reasoning_summary: ReasoningSummaryConfig::Auto,
        support_verbosity: true,
        default_verbosity: None,
        apply_patch_tool_type: None,
        truncation_policy: TruncationPolicyConfig::bytes(100_000),
        supports_parallel_tool_calls: true,
        context_window: None,
        auto_compact_token_limit: None,
        effective_context_window_percent: 95,
        experimental_supported_tools: Vec::new(),
        input_modalities: default_input_modalities(),
        prefer_websockets: false,
        used_fallback_model_metadata: true,
        supports_image_detail_original: false,
        web_search_tool_type: WebSearchToolType::Text,
    }
}

// ---------------------------------------------------------------------------
// Reasoning effort helpers
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Proxy client slot helpers
// ---------------------------------------------------------------------------

/// Build a fresh `ModelClient` suitable for the HTTP proxy path.
fn create_proxy_client(
    auth_manager: &Arc<AuthManager>,
    provider: &ModelProviderInfo,
    ws_enabled: bool,
    enable_request_compression: bool,
) -> ModelClient {
    ModelClient::new(
        Some(auth_manager.clone()),
        ThreadId::new(), // stable conversation_id for the slot lifetime
        provider.clone(),
        SessionSource::Cli,
        None,  // model_verbosity
        ws_enabled,
        enable_request_compression,
        false, // include_timing_metrics
        None,  // beta_features_header
    )
}

/// Compute a transport key from the current config snapshot.
fn transport_key(
    provider: &ModelProviderInfo,
    ws_enabled: bool,
    enable_request_compression: bool,
) -> TransportKey {
    TransportKey {
        provider_name: provider.name.clone(),
        ws_enabled,
        enable_request_compression,
    }
}

/// Spawn best-effort background prewarm tasks to refill the warm pool.
///
/// Creates fresh clients (never used for real requests — no stale connections
/// possible) up to `WARM_POOL_TARGET`, prewarming each WebSocket connection
/// before pushing it into the pool.
fn spawn_pool_refill(
    state: Arc<HttpServerState>,
    key: TransportKey,
    model_info: ModelInfo,
) {
    tokio::spawn(async move {
        let current_len = state.warm_pool.lock().await.len();
        let needed = WARM_POOL_TARGET.saturating_sub(current_len);
        let provider = state.dynamic_config.read().await.provider.clone();
        for _ in 0..needed {
            let client = create_proxy_client(
                &state.auth_manager,
                &provider,
                key.ws_enabled,
                key.enable_request_compression,
            );
            let mut session = client.new_session();
            let prompt = Prompt::default();
            if let Err(e) = session
                .prewarm_websocket(
                    &prompt,
                    &model_info,
                    &state.otel_manager,
                    None,
                    ReasoningSummaryConfig::Auto,
                    None,
                    None,
                )
                .await
            {
                tracing::debug!("warm pool prewarm failed (best-effort): {e}");
                // Don't push a failed prewarm; just skip this slot.
                continue;
            }
            // Drop session here: the prewarmed connection is stored back into
            // `client` via `ModelClientSession::drop`.
            drop(session);
            state
                .warm_pool
                .lock()
                .await
                .push_back(WarmPoolEntry { client, key: key.clone() });
        }
    });
}

/// Check out a pre-warmed client from the pool, or create a fresh one.
///
/// The pool mutex is held only for the deque pop — never during network I/O.
/// After handing out a client, a background task refills the pool.
async fn checkout_or_create_client(
    state: &Arc<HttpServerState>,
    provider: &ModelProviderInfo,
    ws_enabled: bool,
    enable_request_compression: bool,
    model_info: &ModelInfo,
) -> ModelClient {
    let key = transport_key(provider, ws_enabled, enable_request_compression);

    // Try to pop a matching warm entry (lock held briefly, no I/O).
    let warm = {
        let mut pool = state.warm_pool.lock().await;
        let pos = pool.iter().position(|e| e.key == key);
        pos.map(|i| pool.remove(i).unwrap())
    };

    let client = if let Some(entry) = warm {
        entry.client
    } else {
        // Cold fallback: fresh client, fresh (empty) WebSocket session.
        create_proxy_client(&state.auth_manager, provider, ws_enabled, enable_request_compression)
    };

    // Refill pool in background so future requests are warm.
    spawn_pool_refill(Arc::clone(state), key, model_info.clone());

    client
}

// ---------------------------------------------------------------------------
// Unified reasoning effort resolver
// ---------------------------------------------------------------------------

/// Map the Anthropic model name requested by the client to a reasoning effort
/// override based on model tier. Returns `Some(Low)` for haiku requests so
/// they always run with minimal reasoning, regardless of the TUI config.
fn effort_for_model_tier(requested_model: &str) -> Option<ReasoningEffortConfig> {
    let model = requested_model.to_ascii_lowercase();
    if model.contains("haiku") {
        Some(ReasoningEffortConfig::Low)
    } else if model.contains("opus") {
        Some(ReasoningEffortConfig::High)
    } else {
        None
    }
}

/// Resolve the reasoning effort for both logging and the stream call.
///
/// Resolution order:
/// 1. Model-tier override (haiku → Low).
/// 2. TUI config.
/// 3. Model default.
///
/// Returns `(stream_effort, log_effort, source)`.  `stream_effort` is `None`
/// when falling through to the model default so the model can use its own
/// default instead of being forced to Medium.
fn resolve_reasoning(
    requested_model: &str,
    config_effort: Option<ReasoningEffortConfig>,
    model_info: &ModelInfo,
) -> (Option<ReasoningEffortConfig>, ReasoningEffortConfig, &'static str) {
    if let Some(e) = effort_for_model_tier(requested_model) {
        (Some(e), e, "model_tier")
    } else if let Some(e) = config_effort {
        (Some(e), e, "config")
    } else {
        let e = model_info
            .default_reasoning_level
            .unwrap_or(ReasoningEffortConfig::Medium);
        // Pass `None` so the model uses its own default rather than forcing Medium.
        (None, e, "model_default")
    }
}

// ---------------------------------------------------------------------------
// Axum handlers
// ---------------------------------------------------------------------------

async fn health() -> &'static str {
    "ok"
}

async fn list_models(State(state): State<Arc<HttpServerState>>) -> impl IntoResponse {
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

async fn handle_messages(
    State(state): State<Arc<HttpServerState>>,
    headers: axum::http::HeaderMap,
    body: axum::extract::Json<AnthropicRequest>,
) -> Response {
    let req = body.0;
    let stream_mode = req.stream;

    // Read dynamic config snapshot.
    let (model, model_info, provider, reasoning_effort, reasoning_summary, ws_enabled, enable_request_compression) = {
        let cfg = state.dynamic_config.read().await;
        (
            cfg.model.clone(),
            cfg.model_info().clone(),
            cfg.provider.clone(),
            cfg.reasoning_effort,
            cfg.reasoning_summary.clone(),
            cfg.ws_enabled,
            cfg.enable_request_compression,
        )
    };

    // Unified reasoning resolution: used for both logging and the stream call.
    let (stream_effort, log_effort, reasoning_source) =
        resolve_reasoning(&req.model, reasoning_effort, &model_info);

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
            )
            .await;
            return (StatusCode::BAD_REQUEST, format!("Bad request: {e}")).into_response();
        }
    };

    // Log the incoming request.
    {
        let entry = HttpLogEntry {
            timestamp: Utc::now(),
            model: model.clone(),
            reasoning_effort: log_effort,
            reasoning_source: reasoning_source.to_string(),
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
        append_log(&state.log_buffer, entry).await;
    }

    // Compute scale factor to normalise codex's effective context window to Claude Code's assumed 200k.
    let context_window_scale = {
        let codex_window = model_info.context_window.unwrap_or(272_000) as f64;
        let effective = model_info.effective_context_window_percent as f64 / 100.0;
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
    let session_id = headers
        .get("x-codex-session-id")
        .and_then(|v| v.to_str().ok())
        .map(str::to_owned);

    let key = transport_key(&provider, ws_enabled, enable_request_compression);

    let client = if let Some(ref sid) = session_id {
        // Take the entry out of the cache (preventing concurrent reuse).
        let entry = state.session_cache.lock().await.remove(sid);
        match entry {
            Some(e) if e.key == key => e.client,
            _ => create_proxy_client(&state.auth_manager, &provider, ws_enabled, enable_request_compression),
        }
    } else {
        checkout_or_create_client(&state, &provider, ws_enabled, enable_request_compression, &model_info).await
    };

    let mut session = client.new_session();

    let stream_result = session
        .stream(&prompt, &model_info, &state.otel_manager, stream_effort, reasoning_summary, None, None)
        .await;

    let mut response_stream = match stream_result {
        Ok(s) => s,
        Err(e) => {
            let msg = format!("Stream error: {e}");
            log_error(
                &state,
                &model,
                log_effort,
                reasoning_source,
                items_count,
                &tool_names,
                stream_mode,
                msg.clone(),
                Some(503),
            )
            .await;
            return (StatusCode::SERVICE_UNAVAILABLE, msg).into_response();
        }
    };

    if stream_mode {
        // --- Streaming response: direct stream mapping, no mpsc bridge ---
        let log_buffer = state.log_buffer.clone();
        let state_for_stream = Arc::clone(&state);
        let model_clone = model.clone();

        let sse_stream = async_stream::stream! {
            let mut translator = StreamTranslator::new(model_clone.clone(), context_window_scale);

            while let Some(event) = response_stream.next().await {
                match event {
                    Ok(ev) => {
                        let sse_chunk = translator.translate(&ev);
                        if !sse_chunk.is_empty() {
                            yield Ok::<Bytes, std::io::Error>(Bytes::from(sse_chunk));
                        }
                    }
                    Err(e) => {
                        tracing::error!("Stream error: {e}");
                        break;
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
                status: Some(200),
                stop_reason: Some(translator.stop_reason().to_string()),
                input_tokens: translator.input_tokens(),
                output_tokens: translator.output_tokens(),
                error: None,
            };
            let response_line = entry.response_line();
            append_log(&log_buffer, entry).await;
            tracing::info!("{}", response_line);
        };

        let body = Body::from_stream(sse_stream);
        Response::builder()
            .status(200)
            .header("content-type", "text/event-stream")
            .header("cache-control", "no-cache")
            .header("connection", "keep-alive")
            .body(body)
            .unwrap_or_else(|_| StatusCode::INTERNAL_SERVER_ERROR.into_response())
    } else {
        // --- Non-streaming: accumulate and return JSON ---
        let mut translator = StreamTranslator::new(model.clone(), context_window_scale);
        while let Some(event) = response_stream.next().await {
            match event {
                Ok(ev) => {
                    translator.consume_event(&ev);
                }
                Err(e) => {
                    let msg = format!("Stream error: {e}");
                    log_error(
                        &state,
                        &model,
                        log_effort,
                        reasoning_source,
                        items_count,
                        &tool_names,
                        false,
                        msg.clone(),
                        Some(502),
                    )
                    .await;
                    return (StatusCode::BAD_GATEWAY, msg).into_response();
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

        let message = translator.build_response(&req.model);
        let entry = HttpLogEntry {
            timestamp: Utc::now(),
            model: model.clone(),
            reasoning_effort: log_effort,
            reasoning_source: reasoning_source.to_string(),
            items: items_count,
            tools: tool_names,
            streaming: false,
            status: Some(200),
            stop_reason: Some(translator.stop_reason().to_string()),
            input_tokens: translator.input_tokens(),
            output_tokens: translator.output_tokens(),
            error: None,
        };
        tracing::info!("{}", entry.response_line());
        append_log(&state.log_buffer, entry).await;

        axum::Json(message).into_response()
    }
}

// ---------------------------------------------------------------------------
// Logging helpers
// ---------------------------------------------------------------------------

async fn append_log(buffer: &Arc<Mutex<VecDeque<HttpLogEntry>>>, entry: HttpLogEntry) {
    let mut buf = buffer.lock().await;
    if buf.len() >= LOG_BUFFER_MAX {
        buf.pop_front();
    }
    buf.push_back(entry);
}

async fn cache_session(
    cache: &Mutex<HashMap<String, SessionCacheEntry>>,
    sid: &str,
    client: ModelClient,
    key: TransportKey,
) {
    let mut c = cache.lock().await;
    let now = std::time::Instant::now();
    c.retain(|_, e| now.duration_since(e.last_used) < SESSION_CACHE_TTL);
    c.insert(sid.to_owned(), SessionCacheEntry { client, last_used: now, key });
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
        status,
        stop_reason: None,
        input_tokens: None,
        output_tokens: None,
        error: Some(error),
    };
    append_log(&state.log_buffer, entry).await;
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
    /// Scale factor applied to input_tokens before reporting to Claude Code.
    /// Normalises codex's effective context window to Claude Code's assumed 200k window.
    context_window_scale: f64,
}

#[derive(Debug)]
enum BlockState {
    Text { text: String },
    ToolUse { id: String, name: String, input: String },
}

impl StreamTranslator {
    fn new(model_slug: String, context_window_scale: f64) -> Self {
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
            context_window_scale,
        }
    }

    fn stop_reason(&self) -> &'static str {
        if self.has_function_call { "tool_use" } else { "end_turn" }
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

    /// Consume a single `ResponseEvent` and mutate accumulated state without
    /// emitting SSE output.
    fn consume_event(&mut self, event: &ResponseEvent) {
        match event {
            ResponseEvent::Created => {
                self.handle_created(false);
            }
            ResponseEvent::OutputItemAdded(item) => {
                self.handle_item_added(item, false);
            }
            ResponseEvent::OutputTextDelta(delta) => {
                self.handle_text_delta(delta, false);
            }
            ResponseEvent::OutputItemDone(item) => {
                self.handle_item_done(item, false);
            }
            ResponseEvent::Completed { token_usage, .. } => {
                if let Some(usage) = token_usage {
                    self.input_tokens = usage.input_tokens as u32;
                    self.output_tokens = usage.output_tokens as u32;
                }
                self.handle_completed(false);
            }
            // Ignore the rest.
            _ => {}
        }
    }

    /// Translate a single `ResponseEvent` into one or more Anthropic SSE chunks.
    ///
    /// Returns an empty string when the event produces no output.
    fn translate(&mut self, event: &ResponseEvent) -> String {
        match event {
            ResponseEvent::Created => self.handle_created(true),
            ResponseEvent::OutputItemAdded(item) => self.handle_item_added(item, true),
            ResponseEvent::OutputTextDelta(delta) => self.handle_text_delta(delta, true),
            ResponseEvent::OutputItemDone(item) => self.handle_item_done(item, true),
            ResponseEvent::Completed { token_usage, .. } => {
                if let Some(usage) = token_usage {
                    self.input_tokens = usage.input_tokens as u32;
                    self.output_tokens = usage.output_tokens as u32;
                }
                self.handle_completed(true)
            }
            // Ignore the rest.
            _ => String::new(),
        }
    }

    fn handle_created(&mut self, emit_sse: bool) -> String {
        if !emit_sse {
            return String::new();
        }

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

    fn handle_item_added(&mut self, item: &ResponseItem, emit_sse: bool) -> String {
        match item {
            ResponseItem::Message { .. } => {
                // Open a text block.
                let idx = self.next_block;
                self.next_block += 1;
                self.current_text_block = Some(idx);
                self.blocks.push(BlockState::Text { text: String::new() });

                if !emit_sse {
                    return String::new();
                }

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
                self.blocks.push(BlockState::ToolUse {
                    id: call_id.clone(),
                    name: name.clone(),
                    input: String::new(),
                });

                if !emit_sse {
                    return String::new();
                }

                let block_start = serde_json::json!({
                    "type": "content_block_start",
                    "index": idx,
                    "content_block": {
                        "type": "tool_use",
                        "id": call_id,
                        "name": name,
                        "input": {}
                    }
                });
                format_sse("content_block_start", &block_start)
            }
            _ => String::new(),
        }
    }

    fn handle_text_delta(&mut self, delta: &str, emit_sse: bool) -> String {
        let Some(idx) = self.current_text_block else {
            return String::new();
        };
        // Accumulate for non-streaming path.
        if let Some(BlockState::Text { text }) = self.blocks.get_mut(idx) {
            text.push_str(delta);
        }

        if !emit_sse {
            return String::new();
        }

        let content_delta = serde_json::json!({
            "type": "content_block_delta",
            "index": idx,
            "delta": { "type": "text_delta", "text": delta }
        });
        format_sse("content_block_delta", &content_delta)
    }

    fn handle_item_done(&mut self, item: &ResponseItem, emit_sse: bool) -> String {
        match item {
            ResponseItem::Message { .. } => {
                let Some(idx) = self.current_text_block.take() else {
                    return String::new();
                };

                if !emit_sse {
                    return String::new();
                }

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

                if !emit_sse {
                    return String::new();
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

    fn handle_completed(&self, emit_sse: bool) -> String {
        if !emit_sse {
            return String::new();
        }

        let reported_input_tokens = (self.input_tokens as f64 * self.context_window_scale).round() as u32;
        let stop_reason = self.stop_reason();
        let message_delta = serde_json::json!({
            "type": "message_delta",
            "delta": { "stop_reason": stop_reason, "stop_sequence": null },
            "usage": { "output_tokens": self.output_tokens, "input_tokens": reported_input_tokens }
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
        let reported_input_tokens = (self.input_tokens as f64 * self.context_window_scale).round() as u32;
        serde_json::json!({
            "id": self.message_id,
            "type": "message",
            "role": "assistant",
            "model": original_model,
            "content": content,
            "stop_reason": self.stop_reason(),
            "stop_sequence": null,
            "usage": {
                "input_tokens": reported_input_tokens,
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
