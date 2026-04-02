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
//!
//! # Module layout
//!
//! | Module | Contents |
//! |---|---|
//! | [`wire_types`] | Anthropic JSON deserialization types |
//! | [`state`] | [`HttpLogEntry`], [`HttpServerDynamicConfig`], [`HttpServerState`] |
//! | [`translation`] | Anthropic → internal format conversion |
//! | [`reasoning`] | Per-request reasoning/effort resolution |
//! | [`pool`] | Warm pool and session cache management |
//! | [`stream`] | [`StreamTranslator`] SSE translation |
//! | [`handlers`] | Axum handlers and [`build_router`] |

mod handlers;
mod pool;
mod reasoning;
mod state;
mod stream;
mod translation;
mod wire_types;

pub use handlers::build_router;
pub use state::HttpLogEntry;
pub use state::HttpServerDynamicConfig;
pub use state::HttpServerState;
