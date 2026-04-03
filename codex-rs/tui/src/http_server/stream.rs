use std::collections::HashMap;

use codex_core::ResponseEvent;
use codex_protocol::models::ResponseItem;
use serde_json::Value;
use uuid::Uuid;

/// Translates `ResponseEvent` stream into Anthropic SSE text chunks.
pub(super) struct StreamTranslator {
    model_slug: String,
    message_id: String,
    /// Next block index to assign.
    next_block: usize,
    /// Block index allocated for a text block whose `content_block_start` has
    /// not yet been emitted (waiting for the first text delta).
    pending_text_block: Option<usize>,
    /// Block index currently receiving text deltas (start already emitted).
    current_text_block: Option<usize>,
    /// Active thinking block: (block_index, summary_index).
    current_thinking_block: Option<(usize, i64)>,
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
    /// If set, overrides the stop_reason returned by stop_reason().
    stop_reason_override: Option<&'static str>,
}

#[derive(Debug)]
pub(super) enum BlockState {
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: String,
    },
    Thinking {
        text: String,
    },
}

impl StreamTranslator {
    pub(super) fn new(model_slug: String, context_window_scale: f64, estimated_input_tokens: u32) -> Self {
        Self {
            model_slug,
            message_id: format!("msg_{}", Uuid::new_v4().simple()),
            next_block: 0,
            pending_text_block: None,
            current_text_block: None,
            current_thinking_block: None,
            call_id_to_block: HashMap::new(),
            has_function_call: false,
            input_tokens: estimated_input_tokens,
            output_tokens: 0,
            blocks: Vec::new(),
            context_window_scale,
            stop_reason_override: None,
        }
    }

    pub(super) fn stop_reason(&self) -> &str {
        if let Some(r) = self.stop_reason_override {
            return r;
        }
        if self.has_function_call { "tool_use" } else { "end_turn" }
    }

    pub(super) fn set_stop_reason_override(&mut self, reason: &'static str) {
        self.stop_reason_override = Some(reason);
    }

    pub(super) fn terminate_with_reason(&mut self, stop_reason: &str) -> String {
        let mut out = String::new();
        // Close any open thinking block before the text block.
        out += &self.close_thinking_block(true);
        if let Some(idx) = self.current_text_block {
            out += &format_sse("content_block_stop", &serde_json::json!({"type": "content_block_stop", "index": idx}));
        }
        let reported_input = (self.input_tokens as f64 * self.context_window_scale).round() as u32;
        out += &format_sse("message_delta", &serde_json::json!({
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": null},
            "usage": {"output_tokens": self.output_tokens, "input_tokens": reported_input, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
        }));
        out += &format_sse("message_stop", &serde_json::json!({"type": "message_stop"}));
        out
    }

    pub(super) fn input_tokens(&self) -> Option<u32> {
        if self.input_tokens > 0 {
            Some(self.input_tokens)
        } else {
            None
        }
    }

    pub(super) fn output_tokens(&self) -> Option<u32> {
        if self.output_tokens > 0 {
            Some(self.output_tokens)
        } else {
            None
        }
    }

    /// Consume a single `ResponseEvent` and mutate accumulated state without
    /// emitting SSE output.
    pub(super) fn consume_event(&mut self, event: &ResponseEvent) {
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
            ResponseEvent::ReasoningSummaryPartAdded { summary_index } => {
                self.handle_thinking_part_added(*summary_index, false);
            }
            ResponseEvent::ReasoningSummaryDelta { delta, .. } => {
                self.handle_thinking_delta(delta, false);
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
    pub(super) fn translate(&mut self, event: &ResponseEvent) -> String {
        match event {
            ResponseEvent::Created => self.handle_created(true),
            ResponseEvent::OutputItemAdded(item) => self.handle_item_added(item, true),
            ResponseEvent::OutputTextDelta(delta) => self.handle_text_delta(delta, true),
            ResponseEvent::OutputItemDone(item) => self.handle_item_done(item, true),
            ResponseEvent::ReasoningSummaryPartAdded { summary_index } => {
                self.handle_thinking_part_added(*summary_index, true)
            }
            ResponseEvent::ReasoningSummaryDelta { delta, .. } => {
                self.handle_thinking_delta(delta, true)
            }
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
                "usage": { "input_tokens": self.input_tokens, "output_tokens": 0, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0 }
            }
        });
        format_sse("message_start", &message_start)
            + &format_sse("ping", &serde_json::json!({"type": "ping"}))
    }

    fn handle_item_added(&mut self, item: &ResponseItem, emit_sse: bool) -> String {
        match item {
            ResponseItem::Message { .. } => {
                // Allocate a text block but defer emitting content_block_start
                // until the first delta arrives. If no delta ever comes (tool-
                // only responses), we suppress the empty block entirely.
                let idx = self.next_block;
                self.next_block += 1;
                self.pending_text_block = Some(idx);
                self.blocks.push(BlockState::Text {
                    text: String::new(),
                });
                String::new()
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
        // Promote a pending block to active on the first delta.
        if let Some(pending_idx) = self.pending_text_block.take() {
            self.current_text_block = Some(pending_idx);
            if emit_sse {
                let block_start = serde_json::json!({
                    "type": "content_block_start",
                    "index": pending_idx,
                    "content_block": { "type": "text", "text": "" }
                });
                let start_sse = format_sse("content_block_start", &block_start);
                if let Some(BlockState::Text { text }) = self.blocks.get_mut(pending_idx) {
                    text.push_str(delta);
                }
                let content_delta = serde_json::json!({
                    "type": "content_block_delta",
                    "index": pending_idx,
                    "delta": { "type": "text_delta", "text": delta }
                });
                return start_sse + &format_sse("content_block_delta", &content_delta);
            }
        }

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

    /// Opens a new thinking block for `summary_index`, closing any previous
    /// thinking block with a different index first.
    fn handle_thinking_part_added(&mut self, summary_index: i64, emit_sse: bool) -> String {
        let mut out = String::new();

        // Close the previous thinking block if it belongs to a different part.
        if let Some((prev_idx, prev_summary)) = self.current_thinking_block {
            if prev_summary != summary_index {
                self.current_thinking_block = None;
                if emit_sse {
                    out += &format_sse(
                        "content_block_stop",
                        &serde_json::json!({"type": "content_block_stop", "index": prev_idx}),
                    );
                }
            } else {
                // Same part already open — nothing to do.
                return out;
            }
        }

        let idx = self.next_block;
        self.next_block += 1;
        self.current_thinking_block = Some((idx, summary_index));
        self.blocks.push(BlockState::Thinking { text: String::new() });

        if emit_sse {
            out += &format_sse(
                "content_block_start",
                &serde_json::json!({
                    "type": "content_block_start",
                    "index": idx,
                    "content_block": { "type": "thinking", "thinking": "" }
                }),
            );
        }

        out
    }

    fn handle_thinking_delta(&mut self, delta: &str, emit_sse: bool) -> String {
        let Some((idx, _)) = self.current_thinking_block else {
            return String::new();
        };

        if let Some(BlockState::Thinking { text }) = self.blocks.get_mut(idx) {
            text.push_str(delta);
        }

        if !emit_sse {
            return String::new();
        }

        format_sse(
            "content_block_delta",
            &serde_json::json!({
                "type": "content_block_delta",
                "index": idx,
                "delta": { "type": "thinking_delta", "thinking": delta }
            }),
        )
    }

    /// Close the active thinking block, if any. Returns SSE chunks for the
    /// `signature_delta` and `content_block_stop` when `emit_sse` is true.
    fn close_thinking_block(&mut self, emit_sse: bool) -> String {
        let Some((idx, _)) = self.current_thinking_block.take() else {
            return String::new();
        };
        if !emit_sse {
            return String::new();
        }
        // Emit a proxy signature so Claude Code's SDK validation passes.
        let signature = format!("proxy_{idx:04}");
        let sig_delta = format_sse(
            "content_block_delta",
            &serde_json::json!({
                "type": "content_block_delta",
                "index": idx,
                "delta": { "type": "signature_delta", "signature": signature }
            }),
        );
        let stop = format_sse(
            "content_block_stop",
            &serde_json::json!({"type": "content_block_stop", "index": idx}),
        );
        sig_delta + &stop
    }

    fn handle_item_done(&mut self, item: &ResponseItem, emit_sse: bool) -> String {
        match item {
            ResponseItem::Message { .. } => {
                // If the block never received a delta, discard it — no start
                // was emitted so no stop is needed either.
                if let Some(pending_idx) = self.pending_text_block.take() {
                    self.blocks.truncate(pending_idx);
                    self.next_block = pending_idx;
                    return String::new();
                }

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
            ResponseItem::Reasoning { .. } => self.close_thinking_block(emit_sse),
            _ => String::new(),
        }
    }

    fn handle_completed(&mut self, emit_sse: bool) -> String {
        // Always close any dangling thinking block before message_delta.
        let thinking_close = self.close_thinking_block(emit_sse);

        if !emit_sse {
            return String::new();
        }

        let reported_input_tokens =
            (self.input_tokens as f64 * self.context_window_scale).round() as u32;
        let stop_reason = self.stop_reason();
        let message_delta = serde_json::json!({
            "type": "message_delta",
            "delta": { "stop_reason": stop_reason, "stop_sequence": null },
            "usage": { "output_tokens": self.output_tokens, "input_tokens": reported_input_tokens, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0 }
        });
        let message_stop = serde_json::json!({ "type": "message_stop" });
        thinking_close
            + &format_sse("message_delta", &message_delta)
            + &format_sse("message_stop", &message_stop)
    }

    /// Build a complete non-streaming Anthropic response message.
    pub(super) fn build_response(&self, original_model: &str) -> Value {
        let mut content: Vec<Value> = Vec::new();
        for block in &self.blocks {
            match block {
                BlockState::Text { text } => {
                    content.push(serde_json::json!({ "type": "text", "text": text }));
                }
                BlockState::ToolUse { id, name, input } => {
                    let parsed_input: Value = serde_json::from_str(input)
                        .unwrap_or_else(|_| serde_json::json!({ "_raw": input }));
                    content.push(serde_json::json!({
                        "type": "tool_use",
                        "id": id,
                        "name": name,
                        "input": parsed_input
                    }));
                }
                BlockState::Thinking { text } => {
                    content.push(serde_json::json!({
                        "type": "thinking",
                        "thinking": text,
                        "signature": format!("proxy_{i:04}", i = content.len())
                    }));
                }
            }
        }
        let reported_input_tokens =
            (self.input_tokens as f64 * self.context_window_scale).round() as u32;
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
                "output_tokens": self.output_tokens,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0
            }
        })
    }
}

pub(super) fn format_sse(event_type: &str, data: &Value) -> String {
    format!(
        "event: {}\ndata: {}\n\n",
        event_type,
        serde_json::to_string(data).unwrap_or_default()
    )
}
