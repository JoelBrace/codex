use codex_core::Prompt;
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
use serde_json::Value;

use super::wire_types::AnthropicBlock;
use super::wire_types::AnthropicContent;
use super::wire_types::AnthropicMessage;
use super::wire_types::AnthropicRequest;
use super::wire_types::AnthropicToolResultBlock;
use super::wire_types::AnthropicToolResultContent;

/// Translate an Anthropic request body into the internal `Prompt` type.
pub(super) fn translate_request(req: &AnthropicRequest, _model: &str) -> anyhow::Result<Prompt> {
    let _ = (
        &req.metadata,
        &req.temperature,
        &req.context_management,
        &req.thinking,
        &req.output_config,
        &req.speed,
        &req.betas,
    );

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

    let tool_choice_none = req
        .tool_choice
        .as_ref()
        .and_then(|choice| match choice {
            Value::String(s) => Some(s.eq_ignore_ascii_case("none")),
            Value::Object(map) => map
                .get("type")
                .and_then(Value::as_str)
                .map(|s| s.eq_ignore_ascii_case("none")),
            _ => None,
        })
        .unwrap_or(false);

    // Tools
    if !tool_choice_none {
        for tool in &req.tools {
            if tool.tool_type.starts_with("web_search_") {
                prompt.add_web_search_tool(true);
            } else if let Some(schema) = &tool.input_schema {
                prompt.add_function_tool(
                    tool.name.clone(),
                    tool.description.clone(),
                    schema.clone(),
                )?;
            }
            // Other unknown built-in tool types (computer_use, etc.) are silently skipped
        }
    }
    prompt.set_parallel_tool_calls(!tool_choice_none);

    Ok(prompt)
}

/// Convert Anthropic messages into `ResponseItem` list.
pub(super) fn translate_messages(messages: &[AnthropicMessage]) -> Vec<ResponseItem> {
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
                                arguments: input.to_string(),
                                call_id: id.clone(),
                                namespace: None,
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
                                    let content_items: Vec<FunctionCallOutputContentItem> = blocks
                                        .iter()
                                        .filter_map(|b| match b {
                                            AnthropicToolResultBlock::Text { text } => {
                                                Some(FunctionCallOutputContentItem::InputText {
                                                    text: text.clone(),
                                                })
                                            }
                                            AnthropicToolResultBlock::Image { source } => {
                                                Some(FunctionCallOutputContentItem::InputImage {
                                                    image_url: source.to_image_url(),
                                                    detail: None,
                                                })
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

/// Construct a minimal `ModelInfo` sufficient for `ModelClientSession::stream`.
///
/// Since we are proxying to a Codex model that supports reasoning, we hard-code
/// `supports_reasoning_summaries = true` and `support_verbosity = true`.
pub(super) fn proxy_model_info(slug: &str) -> ModelInfo {
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
        used_fallback_model_metadata: true,
        supports_image_detail_original: false,
        web_search_tool_type: WebSearchToolType::Text,
        supports_search_tool: false,
    }
}
