//! Anthropic Messages front end. Both upstreams only speak chat completions,
//! so requests and responses (plain and streamed) are translated here.

pub mod request;
pub mod response;
pub mod stream;

use serde_json::Value;

use crate::decider::{Conversation, Role};

/// Build the decider's view of a Messages request.
pub fn conversation(body: &Value) -> Conversation {
    let mut conv = Conversation {
        system: request::system_text(body.get("system")),
        tools: body
            .get("tools")
            .and_then(Value::as_array)
            .map_or(0, Vec::len),
        ..Default::default()
    };
    let messages = body.get("messages").and_then(Value::as_array);
    for message in messages.into_iter().flatten() {
        let role = match message.get("role").and_then(Value::as_str) {
            Some("assistant") => Role::Assistant,
            _ => Role::User,
        };
        let blocks = match message.get("content") {
            Some(Value::String(s)) => {
                conv.push(role, s.as_str());
                continue;
            }
            Some(Value::Array(blocks)) => blocks,
            _ => continue,
        };
        let mut text = Vec::new();
        for block in blocks {
            match block.get("type").and_then(Value::as_str) {
                Some("text") => {
                    text.extend(
                        block
                            .get("text")
                            .and_then(Value::as_str)
                            .map(str::to_string),
                    );
                }
                Some("image" | "document") => conv.has_images = true,
                Some("tool_use") => {
                    let name = block.get("name").and_then(Value::as_str).unwrap_or("?");
                    text.push(format!("[tool call: {name}]"));
                }
                Some("tool_result") => {
                    let (result, images) = request::tool_result_content(block.get("content"));
                    conv.has_images |= !images.is_empty();
                    conv.push(Role::Tool, result);
                }
                _ => {}
            }
        }
        if !text.is_empty() {
            conv.push(role, text.join("\n"));
        }
    }
    conv
}

/// Rough token count for `/v1/messages/count_tokens`; neither upstream offers one.
pub fn estimate_tokens(body: &Value) -> u64 {
    let chars: usize = ["system", "messages", "tools"]
        .iter()
        .filter_map(|key| body.get(*key))
        .map(|v| v.to_string().chars().count())
        .sum();
    (chars as u64).div_ceil(4).max(1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn tool_results_are_not_user_text() {
        let body = json!({
            "system": [{"type": "text", "text": "you are helpful"}],
            "tools": [{"name": "read", "input_schema": {}}],
            "messages": [
                {"role": "user", "content": "open main.rs"},
                {"role": "assistant", "content": [
                    {"type": "text", "text": "reading"},
                    {"type": "tool_use", "id": "t1", "name": "read", "input": {"path": "main.rs"}}
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "fn main() {}"}
                ]}
            ]
        });
        let conv = conversation(&body);
        assert_eq!(conv.system, "you are helpful");
        assert_eq!(conv.tools, 1);
        assert_eq!(conv.turns.last().unwrap().role, Role::Tool);
        assert_eq!(conv.state()["latest_user_request"], "open main.rs");
    }
}
