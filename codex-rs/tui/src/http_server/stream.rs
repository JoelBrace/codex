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
}

impl StreamTranslator {
    pub(super) fn new(model_slug: String, context_window_scale: f64) -> Self {
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

    pub(super) fn terminate_with_reason(&self, stop_reason: &str) -> String {
        let mut out = String::new();
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
                // Open a text block.
                let idx = self.next_block;
                self.next_block += 1;
                self.current_text_block = Some(idx);
                self.blocks.push(BlockState::Text {
                    text: String::new(),
                });

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

        let reported_input_tokens =
            (self.input_tokens as f64 * self.context_window_scale).round() as u32;
        let stop_reason = self.stop_reason();
        let message_delta = serde_json::json!({
            "type": "message_delta",
            "delta": { "stop_reason": stop_reason, "stop_sequence": null },
            "usage": { "output_tokens": self.output_tokens, "input_tokens": reported_input_tokens, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0 }
        });
        let message_stop = serde_json::json!({ "type": "message_stop" });
        format_sse("message_delta", &message_delta) + &format_sse("message_stop", &message_stop)
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
                "output_tokens": self.output_tokens
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
