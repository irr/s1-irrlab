//! OpenAI chat-completions response → Anthropic Messages response.

use serde_json::{json, Value};

pub fn message_id() -> String {
    format!("msg_{}", uuid::Uuid::new_v4().simple())
}

pub fn tool_use_id() -> String {
    format!("toolu_{}", uuid::Uuid::new_v4().simple())
}

/// `finish_reason` → `stop_reason`. Some servers report `stop` even when the
/// turn ended in tool calls, so their presence wins.
pub fn stop_reason(finish_reason: Option<&str>, has_tool_use: bool) -> &'static str {
    match finish_reason {
        Some("length") => "max_tokens",
        Some("tool_calls" | "function_call") => "tool_use",
        _ if has_tool_use => "tool_use",
        _ => "end_turn",
    }
}

/// Reasoning text, under either name vLLM has used for it.
pub fn reasoning(message: &Value) -> Option<&str> {
    ["reasoning_content", "reasoning"]
        .iter()
        .find_map(|key| message.get(*key).and_then(Value::as_str))
        .filter(|s| !s.is_empty())
}

/// Anthropic `usage` from a chat-completions `usage` object.
pub fn usage(usage: Option<&Value>) -> Value {
    let get = |pointer: &str| {
        usage
            .and_then(|u| u.pointer(pointer))
            .and_then(Value::as_u64)
            .unwrap_or(0)
    };
    let cached = get("/prompt_tokens_details/cached_tokens");
    json!({
        "input_tokens": get("/prompt_tokens").saturating_sub(cached),
        "output_tokens": get("/completion_tokens"),
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": cached,
    })
}

/// Translate a complete (non-streamed) response; `model` is the model that answered.
pub fn from_chat(resp: &Value, model: &str) -> Value {
    let choice = resp.pointer("/choices/0");
    let message = choice
        .and_then(|c| c.get("message"))
        .unwrap_or(&Value::Null);

    let mut content = Vec::new();
    if let Some(thinking) = reasoning(message) {
        content.push(json!({"type": "thinking", "thinking": thinking, "signature": ""}));
    }
    let text = match message.get("content") {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(parts)) => parts
            .iter()
            .filter_map(|p| p.get("text").and_then(Value::as_str))
            .collect(),
        _ => String::new(),
    };
    if !text.is_empty() {
        content.push(json!({"type": "text", "text": text}));
    }
    let calls = message.get("tool_calls").and_then(Value::as_array);
    let mut has_tool_use = false;
    for call in calls.into_iter().flatten() {
        let arguments = call.pointer("/function/arguments").and_then(Value::as_str);
        let input = arguments
            .and_then(|a| serde_json::from_str::<Value>(a).ok())
            .filter(Value::is_object)
            .unwrap_or_else(|| json!({}));
        let id = call
            .get("id")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty());
        content.push(json!({
            "type": "tool_use",
            "id": id.map_or_else(tool_use_id, str::to_string),
            "name": call.pointer("/function/name").and_then(Value::as_str).unwrap_or(""),
            "input": input,
        }));
        has_tool_use = true;
    }
    if content.is_empty() {
        content.push(json!({"type": "text", "text": ""}));
    }

    let finish = choice
        .and_then(|c| c.get("finish_reason"))
        .and_then(Value::as_str);
    json!({
        "id": message_id(),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop_reason(finish, has_tool_use),
        "stop_sequence": null,
        "usage": usage(resp.get("usage")),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn translates_text_reasoning_and_tool_calls() {
        let resp = json!({
            "choices": [{
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "reasoning_content": "let me look",
                    "content": "checking",
                    "tool_calls": [
                        {"id": "call_1", "type": "function", "function": {"name": "read", "arguments": "{\"path\":\"a.rs\"}"}}
                    ]
                }
            }],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20, "prompt_tokens_details": {"cached_tokens": 40}}
        });
        let msg = from_chat(&resp, "qwen");
        assert_eq!(msg["type"], "message");
        assert_eq!(msg["model"], "qwen");
        assert_eq!(msg["stop_reason"], "tool_use");
        assert_eq!(msg["content"][0]["type"], "thinking");
        assert_eq!(
            msg["content"][1],
            json!({"type": "text", "text": "checking"})
        );
        assert_eq!(
            msg["content"][2],
            json!({"type": "tool_use", "id": "call_1", "name": "read", "input": {"path": "a.rs"}})
        );
        assert_eq!(msg["usage"]["input_tokens"], 60);
        assert_eq!(msg["usage"]["cache_read_input_tokens"], 40);
        assert_eq!(msg["usage"]["output_tokens"], 20);
    }

    #[test]
    fn maps_stop_reasons() {
        assert_eq!(stop_reason(Some("stop"), false), "end_turn");
        assert_eq!(stop_reason(Some("stop"), true), "tool_use");
        assert_eq!(stop_reason(Some("length"), true), "max_tokens");
        assert_eq!(stop_reason(None, false), "end_turn");
    }

    #[test]
    fn empty_reply_still_has_a_block() {
        let msg = from_chat(
            &json!({"choices": [{"finish_reason": "stop", "message": {"content": null}}]}),
            "m",
        );
        assert_eq!(msg["content"], json!([{"type": "text", "text": ""}]));
        assert_eq!(msg["usage"]["input_tokens"], 0);
    }
}
