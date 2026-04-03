use serde::Deserialize;
use serde_json::Value;

#[derive(Debug, Deserialize)]
pub(super) struct AnthropicRequest {
    #[serde(default)]
    pub(super) model: String,
    pub(super) messages: Vec<AnthropicMessage>,
    #[serde(default)]
    pub(super) system: Option<Value>,
    #[serde(default)]
    pub(super) tools: Vec<AnthropicTool>,
    #[serde(default)]
    pub(super) stream: bool,
    #[allow(dead_code)]
    #[serde(default)]
    pub(super) max_tokens: Option<u32>,
    #[serde(default)]
    pub(super) tool_choice: Option<Value>,
    #[serde(default)]
    pub(super) metadata: Option<Value>,
    #[serde(default)]
    pub(super) thinking: Option<Value>,
    #[serde(default)]
    pub(super) temperature: Option<f64>,
    #[serde(default)]
    pub(super) context_management: Option<Value>,
    #[serde(default)]
    pub(super) output_config: Option<Value>,
    #[serde(default)]
    pub(super) speed: Option<String>,
    #[serde(default)]
    pub(super) betas: Option<Vec<String>>,
}

#[derive(Debug, Deserialize)]
pub(super) struct AnthropicMessage {
    pub(super) role: String,
    pub(super) content: AnthropicContent,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub(super) enum AnthropicContent {
    Text(String),
    Blocks(Vec<AnthropicBlock>),
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(super) enum AnthropicImageSource {
    Base64 { media_type: String, data: String },
    Url { url: String },
}

impl AnthropicImageSource {
    pub(super) fn to_image_url(&self) -> String {
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
pub(super) enum AnthropicBlock {
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
        /// When true the tool call failed; mapped to success=false in OpenAI format.
        #[serde(default)]
        is_error: Option<bool>,
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
        /// Raw source object; text-type sources have their content extracted.
        #[serde(default)]
        source: Option<Value>,
    },
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(super) enum AnthropicToolResultBlock {
    Text {
        text: String,
    },
    Image {
        source: AnthropicImageSource,
    },
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Deserialize, Default)]
#[serde(untagged)]
pub(super) enum AnthropicToolResultContent {
    #[default]
    Empty,
    Text(String),
    Blocks(Vec<AnthropicToolResultBlock>),
}

#[derive(Debug, Deserialize)]
pub(super) struct AnthropicTool {
    #[serde(rename = "type", default)]
    pub(super) tool_type: String,
    pub(super) name: String,
    #[serde(default)]
    pub(super) description: String,
    #[serde(default)]
    pub(super) input_schema: Option<Value>,
}
