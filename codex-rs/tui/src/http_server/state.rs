use std::collections::HashMap;
use std::collections::VecDeque;
use std::sync::Arc;

use chrono::DateTime;
use chrono::Utc;
use codex_core::ModelClient;
use codex_login::AuthManager;
use codex_model_provider_info::ModelProviderInfo;
use codex_otel::SessionTelemetry;
use codex_protocol::config_types::ReasoningSummary as ReasoningSummaryConfig;
use codex_protocol::openai_models::ModelInfo;
use codex_protocol::openai_models::ReasoningEffort as ReasoningEffortConfig;
use tokio::sync::Mutex;
use tokio::sync::RwLock;

use super::translation::proxy_model_info;

// ---------------------------------------------------------------------------
// Pool / session cache types (defined here to avoid a circular dependency
// with pool.rs, which imports HttpServerState)
// ---------------------------------------------------------------------------

/// Identifies the transport-relevant subset of configuration.
///
/// Used to invalidate warm-pool and session-cache entries when provider
/// config changes.
#[derive(Clone, PartialEq, Eq)]
pub(super) struct TransportKey {
    pub(super) provider_name: String,
    pub(super) enable_request_compression: bool,
}

/// A pre-warmed `ModelClient` held in the warm pool.
///
/// Each entry is a *fresh* client (never used for a real request) so it
/// cannot carry a stale WebSocket connection.
pub(super) struct WarmPoolEntry {
    pub(super) client: ModelClient,
    pub(super) key: TransportKey,
}

/// A cached `ModelClient` for an identified agent session.
///
/// Keyed by the `X-Codex-Session-Id` request header. Reusing the same
/// client across tool-call turns preserves the WebSocket connection (and
/// thus server-side sticky routing) for the duration of a single agent
/// conversation, while keeping each agent session isolated from others.
pub(super) struct SessionCacheEntry {
    pub(super) client: ModelClient,
    pub(super) last_used: std::time::Instant,
    pub(super) key: TransportKey,
}

/// Maximum number of log entries retained in memory.
pub(super) const LOG_BUFFER_MAX: usize = 500;

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
    pub session_id: Option<String>,
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
        let session_full = self.session_id.as_deref().unwrap_or("-");
        let session = &session_full[..session_full.len().min(8)];
        format!(
            "[codex] → model={} effort={} effort_src={} items={} tools=[{}] client={} session={}",
            self.model,
            self.reasoning_effort,
            self.reasoning_source,
            self.items,
            tools,
            if self.streaming { "stream" } else { "sync" },
            session,
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
    /// Whether to enable zstd request compression (mirrors core feature flag).
    pub enable_request_compression: bool,
    /// Named model overrides for the HTTP proxy.
    ///
    /// Key = name (e.g. "haiku"), value = (model slug, optional effort).
    /// When a request model contains a key as a substring, the matching
    /// model + effort is used instead of the TUI default.
    pub named_models: HashMap<String, (String, Option<ReasoningEffortConfig>)>,
    /// Comma-separated list of enabled experimental feature keys for the
    /// `x-codex-beta-features` header (mirrors `Session::build_model_client_beta_features_header`).
    pub beta_features_header: Option<String>,
    model_info_cache: ModelInfo,
}

impl HttpServerDynamicConfig {
    pub fn new(
        model: String,
        provider: ModelProviderInfo,
        reasoning_effort: Option<ReasoningEffortConfig>,
        reasoning_summary: Option<ReasoningSummaryConfig>,
        enable_request_compression: bool,
        beta_features_header: Option<String>,
    ) -> Self {
        let model_info_cache = proxy_model_info(&model);
        Self {
            model,
            provider,
            reasoning_effort,
            reasoning_summary: reasoning_summary.unwrap_or(ReasoningSummaryConfig::Auto),
            enable_request_compression,
            named_models: HashMap::new(),
            beta_features_header,
            model_info_cache,
        }
    }

    /// Replace the entire named models map.
    pub fn set_named_models(
        &mut self,
        named_models: HashMap<String, (String, Option<ReasoningEffortConfig>)>,
    ) {
        self.named_models = named_models;
    }

    /// Insert or update a single named model entry.
    pub fn set_named_model(
        &mut self,
        name: String,
        model: String,
        effort: Option<ReasoningEffortConfig>,
    ) {
        self.named_models.insert(name, (model, effort));
    }

    /// Remove a named model entry.
    pub fn remove_named_model(&mut self, name: &str) {
        self.named_models.remove(name);
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

/// State shared across all HTTP handler invocations.
pub struct HttpServerState {
    pub auth_manager: Arc<AuthManager>,
    pub dynamic_config: Arc<RwLock<HttpServerDynamicConfig>>,
    /// Bounded ring buffer of log entries read by the TUI overlay.
    pub log_buffer: Arc<Mutex<VecDeque<HttpLogEntry>>>,
    pub session_telemetry: SessionTelemetry,
    /// Pool of pre-warmed `ModelClient` instances for fast cold-start.
    ///
    /// Each entry is a fresh client (never used for a real request).
    /// The mutex is held only for the deque operation — never during I/O.
    pub(super) warm_pool: Mutex<VecDeque<WarmPoolEntry>>,
    /// Per-agent-session `ModelClient` cache, keyed by `X-Codex-Session-Id`.
    ///
    /// Entries are taken out (removed) at request start and re-inserted
    /// after the response completes, so two concurrent requests for the
    /// same session ID never share a client.
    pub(super) session_cache: Mutex<HashMap<String, SessionCacheEntry>>,
}

impl HttpServerState {
    pub fn new(
        auth_manager: Arc<AuthManager>,
        dynamic_config: Arc<RwLock<HttpServerDynamicConfig>>,
        log_buffer: Arc<Mutex<VecDeque<HttpLogEntry>>>,
        session_telemetry: SessionTelemetry,
    ) -> Self {
        Self {
            auth_manager,
            dynamic_config,
            log_buffer,
            session_telemetry,
            warm_pool: Mutex::new(VecDeque::new()),
            session_cache: Mutex::new(HashMap::new()),
        }
    }
}
