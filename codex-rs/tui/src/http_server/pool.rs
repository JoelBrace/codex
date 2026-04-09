use std::collections::HashMap;
use std::collections::VecDeque;
use std::sync::Arc;

use codex_core::ModelClient;
use codex_login::AuthManager;
use codex_model_provider_info::ModelProviderInfo;
use codex_core::Prompt;
use codex_protocol::config_types::ReasoningSummary as ReasoningSummaryConfig;
use codex_protocol::openai_models::ModelInfo;
use codex_protocol::protocol::SessionSource;
use codex_protocol::ThreadId;
use tokio::sync::Mutex;

use super::state::HttpLogEntry;
use super::state::HttpServerState;
use super::state::LOG_BUFFER_MAX;
use super::state::SessionCacheEntry;
use super::state::TransportKey;
use super::state::WarmPoolEntry;

/// How many pre-warmed clients to keep in the pool.
pub(super) const WARM_POOL_TARGET: usize = 4;

/// Evict session-cache entries idle for longer than this.
pub(super) const SESSION_CACHE_TTL: std::time::Duration = std::time::Duration::from_secs(300);

/// Build a fresh `ModelClient` suitable for the HTTP proxy path.
pub(super) fn create_proxy_client(
    auth_manager: &Arc<AuthManager>,
    provider: &ModelProviderInfo,
    enable_request_compression: bool,
    beta_features_header: Option<String>,
) -> ModelClient {
    ModelClient::new(
        Some(auth_manager.clone()),
        ThreadId::new(), // stable conversation_id for the slot lifetime
        String::new(),   // installation_id
        provider.clone(),
        SessionSource::Cli,
        None,  // model_verbosity
        enable_request_compression,
        false, // include_timing_metrics
        beta_features_header,
    )
}

/// Compute a transport key from the current config snapshot.
pub(super) fn transport_key(
    provider: &ModelProviderInfo,
    enable_request_compression: bool,
) -> TransportKey {
    TransportKey {
        provider_name: provider.name.clone(),
        enable_request_compression,
    }
}

/// Spawn best-effort background prewarm tasks to refill the warm pool.
///
/// Creates fresh clients (never used for real requests — no stale connections
/// possible) up to `WARM_POOL_TARGET`, prewarming each WebSocket connection
/// before pushing it into the pool.
pub(super) fn spawn_pool_refill(
    state: Arc<HttpServerState>,
    key: TransportKey,
    model_info: ModelInfo,
) {
    tokio::spawn(async move {
        let current_len = state.warm_pool.lock().await.len();
        let needed = WARM_POOL_TARGET.saturating_sub(current_len);
        let (provider, beta_features_header) = {
            let cfg = state.dynamic_config.read().await;
            (cfg.provider.clone(), cfg.beta_features_header.clone())
        };
        for _ in 0..needed {
            let client = create_proxy_client(
                &state.auth_manager,
                &provider,
                key.enable_request_compression,
                beta_features_header.clone(),
            );
            let mut session = client.new_session();
            let prompt = Prompt::default();
            if let Err(e) = session
                .prewarm_websocket(
                    &prompt,
                    &model_info,
                    &state.session_telemetry,
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
            state.warm_pool.lock().await.push_back(WarmPoolEntry {
                client,
                key: key.clone(),
            });
        }
    });
}

/// Check out a pre-warmed client from the pool, or create a fresh one.
///
/// The pool mutex is held only for the deque pop — never during network I/O.
/// After handing out a client, a background task refills the pool.
pub(super) async fn checkout_or_create_client(
    state: &Arc<HttpServerState>,
    provider: &ModelProviderInfo,
    enable_request_compression: bool,
    model_info: &ModelInfo,
    beta_features_header: Option<String>,
) -> ModelClient {
    let key = transport_key(provider, enable_request_compression);

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
        create_proxy_client(
            &state.auth_manager,
            provider,
            enable_request_compression,
            beta_features_header,
        )
    };

    // Refill pool in background so future requests are warm.
    spawn_pool_refill(Arc::clone(state), key, model_info.clone());

    client
}

pub(super) async fn append_log(
    buffer: &Arc<Mutex<VecDeque<HttpLogEntry>>>,
    entry: HttpLogEntry,
) {
    let mut buf = buffer.lock().await;
    if buf.len() >= LOG_BUFFER_MAX {
        buf.pop_front();
    }
    buf.push_back(entry);
}

pub(super) async fn cache_session(
    cache: &Mutex<HashMap<String, SessionCacheEntry>>,
    sid: &str,
    client: ModelClient,
    key: TransportKey,
) {
    let mut c = cache.lock().await;
    let now = std::time::Instant::now();
    c.retain(|_, e| now.duration_since(e.last_used) < SESSION_CACHE_TTL);
    c.insert(
        sid.to_owned(),
        SessionCacheEntry {
            client,
            last_used: now,
            key,
        },
    );
}
