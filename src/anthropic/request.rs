//! Anthropic Messages request → OpenAI chat-completions request.

use serde_json::{json, Map, Value};

/// Translate a Messages request for `model`. `Err` is a client-facing message.
pub fn to_chat(req: &Value, model: &str, max_output: Option<u64>) -> Result<Value, String> {
    let messages = req
        .get("messages")
        .and_then(Value::as_array)
        .ok_or("`messages` must be an array")?;

    let mut out = Vec::new();
    let system = system_text(req.get("system"));
    if !system.is_empty() {
        out.push(json!({"role": "system", "content": system}));
    }
    for (i, message) in messages.iter().enumerate() {
        let content = message.get("content").unwrap_or(&Value::Null);
        match message.get("role").and_then(Value::as_str) {
            Some("user") => push_user(&mut out, content),
            Some("assistant") => out.push(assistant(content)),
            Some("system") => {
                out.push(json!({"role": "system", "content": system_text(Some(content))}))
            }
            Some(other) => return Err(format!("messages[{i}].role `{other}` is not supported")),
            None => return Err(format!("messages[{i}].role is required")),
        }
    }

    let mut chat = Map::new();
    chat.insert("model".into(), json!(model));
    chat.insert("messages".into(), Value::Array(out));

    if let Some(max) = req.get("max_tokens").and_then(Value::as_u64) {
        chat.insert(
            "max_tokens".into(),
            json!(max_output.map_or(max, |cap| max.min(cap))),
        );
    }
    for key in ["temperature", "top_p"] {
        if let Some(v) = req.get(key).filter(|v| v.is_number()) {
            chat.insert(key.into(), v.clone());
        }
    }
    if let Some(stop) = req.get("stop_sequences").and_then(Value::as_array) {
        if !stop.is_empty() {
            chat.insert("stop".into(), json!(stop));
        }
    }
    if req.get("stream").and_then(Value::as_bool) == Some(true) {
        chat.insert("stream".into(), json!(true));
        chat.insert("stream_options".into(), json!({"include_usage": true}));
    }

    let tools = tools(req.get("tools"));
    if !tools.is_empty() {
        chat.insert("tools".into(), Value::Array(tools));
        if let Some(choice) = req.get("tool_choice") {
            if let Some(mapped) = tool_choice(choice) {
                chat.insert("tool_choice".into(), mapped);
            }
            if choice
                .get("disable_parallel_tool_use")
                .and_then(Value::as_bool)
                == Some(true)
            {
                chat.insert("parallel_tool_calls".into(), json!(false));
            }
        }
    }
    Ok(Value::Object(chat))
}

/// `system` is either a string or a list of text blocks.
pub fn system_text(system: Option<&Value>) -> String {
    match system {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(blocks)) => blocks
            .iter()
            .filter_map(|b| b.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join("\n\n"),
        _ => String::new(),
    }
}

/// A user message becomes one `tool` message per tool result, followed by a
/// `user` message with whatever text and images remain. Tool messages go first
/// because chat completions requires them directly after the assistant's calls.
fn push_user(out: &mut Vec<Value>, content: &Value) {
    let blocks = match content {
        Value::String(text) => {
            out.push(json!({"role": "user", "content": text}));
            return;
        }
        Value::Array(blocks) => blocks,
        _ => return,
    };
    let mut parts = Vec::new();
    for block in blocks {
        match block.get("type").and_then(Value::as_str) {
            Some("text") => {
                if let Some(text) = block.get("text").and_then(Value::as_str) {
                    parts.push(json!({"type": "text", "text": text}));
                }
            }
            Some("image") => parts.extend(image_part(block)),
            Some("document") => {
                parts.push(json!({"type": "text", "text": "[document omitted: not supported by this model]"}));
            }
            Some("tool_result") => {
                let (text, images) = tool_result_content(block.get("content"));
                out.push(json!({
                    "role": "tool",
                    "tool_call_id": block.get("tool_use_id").cloned().unwrap_or(Value::Null),
                    "content": text,
                }));
                parts.extend(images);
            }
            _ => {}
        }
    }
    if parts.is_empty() {
        return;
    }
    let all_text = parts.iter().all(|p| p["type"] == "text");
    let content = if all_text {
        let texts: Vec<&str> = parts.iter().filter_map(|p| p["text"].as_str()).collect();
        json!(texts.join("\n\n"))
    } else {
        Value::Array(parts)
    };
    out.push(json!({"role": "user", "content": content}));
}

/// Text of a `tool_result` plus any images it carries (as chat-completions parts).
pub fn tool_result_content(content: Option<&Value>) -> (String, Vec<Value>) {
    match content {
        Some(Value::String(text)) => (text.clone(), Vec::new()),
        Some(Value::Array(blocks)) => {
            let mut text = Vec::new();
            let mut images = Vec::new();
            for block in blocks {
                match block.get("type").and_then(Value::as_str) {
                    Some("text") => text.extend(block.get("text").and_then(Value::as_str)),
                    Some("image") => images.extend(image_part(block)),
                    _ => {}
                }
            }
            (text.join("\n"), images)
        }
        _ => (String::new(), Vec::new()),
    }
}

fn image_part(block: &Value) -> Option<Value> {
    let source = block.get("source")?;
    let url = match source.get("type").and_then(Value::as_str)? {
        "base64" => format!(
            "data:{};base64,{}",
            source.get("media_type")?.as_str()?,
            source.get("data")?.as_str()?
        ),
        "url" => source.get("url")?.as_str()?.to_string(),
        _ => return None,
    };
    Some(json!({"type": "image_url", "image_url": {"url": url}}))
}

/// Assistant text blocks become `content`, `tool_use` blocks become
/// `tool_calls`; thinking blocks have no chat-completions equivalent.
fn assistant(content: &Value) -> Value {
    let mut text = String::new();
    let mut calls = Vec::new();
    match content {
        Value::String(s) => text.push_str(s),
        Value::Array(blocks) => {
            for block in blocks {
                match block.get("type").and_then(Value::as_str) {
                    Some("text") => {
                        text.push_str(block.get("text").and_then(Value::as_str).unwrap_or(""))
                    }
                    Some("tool_use") => calls.push(json!({
                        "id": block.get("id").cloned().unwrap_or(Value::Null),
                        "type": "function",
                        "function": {
                            "name": block.get("name").cloned().unwrap_or(Value::Null),
                            "arguments": block.get("input").unwrap_or(&json!({})).to_string(),
                        }
                    })),
                    _ => {}
                }
            }
        }
        _ => {}
    }
    let mut message = Map::new();
    message.insert("role".into(), json!("assistant"));
    if text.is_empty() && !calls.is_empty() {
        message.insert("content".into(), Value::Null);
    } else {
        message.insert("content".into(), json!(text));
    }
    if !calls.is_empty() {
        message.insert("tool_calls".into(), Value::Array(calls));
    }
    Value::Object(message)
}

/// Client tools only: server tools (web search, etc.) carry no `input_schema`
/// and cannot run on a chat-completions upstream.
fn tools(tools: Option<&Value>) -> Vec<Value> {
    let Some(tools) = tools.and_then(Value::as_array) else {
        return Vec::new();
    };
    tools
        .iter()
        .filter_map(|tool| {
            let mut function = Map::new();
            function.insert("name".into(), tool.get("name")?.clone());
            if let Some(description) = tool.get("description").filter(|d| d.is_string()) {
                function.insert("description".into(), description.clone());
            }
            function.insert("parameters".into(), tool.get("input_schema")?.clone());
            Some(json!({"type": "function", "function": function}))
        })
        .collect()
}

fn tool_choice(choice: &Value) -> Option<Value> {
    match choice.get("type").and_then(Value::as_str)? {
        "auto" => Some(json!("auto")),
        "any" => Some(json!("required")),
        "none" => Some(json!("none")),
        "tool" => Some(json!({"type": "function", "function": {"name": choice.get("name")?}})),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn translates_a_tool_round_trip() {
        let req = json!({
            "model": "claude-whatever",
            "max_tokens": 64000,
            "system": [{"type": "text", "text": "be terse", "cache_control": {"type": "ephemeral"}}],
            "stream": true,
            "stop_sequences": ["END"],
            "thinking": {"type": "enabled", "budget_tokens": 1024},
            "tools": [
                {"name": "read", "description": "read a file", "input_schema": {"type": "object"}},
                {"type": "web_search_20250305", "name": "web_search"}
            ],
            "tool_choice": {"type": "any", "disable_parallel_tool_use": true},
            "messages": [
                {"role": "user", "content": "open main.rs"},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "hmm", "signature": "sig"},
                    {"type": "text", "text": "reading"},
                    {"type": "tool_use", "id": "t1", "name": "read", "input": {"path": "main.rs"}}
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": [{"type": "text", "text": "fn main() {}"}]},
                    {"type": "text", "text": "now explain it"}
                ]}
            ]
        });
        let chat = to_chat(&req, "qwen", Some(32768)).unwrap();
        assert_eq!(chat["model"], "qwen");
        assert_eq!(chat["max_tokens"], 32768);
        assert_eq!(chat["stop"], json!(["END"]));
        assert_eq!(chat["stream_options"], json!({"include_usage": true}));
        assert_eq!(chat["tool_choice"], "required");
        assert_eq!(chat["parallel_tool_calls"], false);
        assert!(chat.get("thinking").is_none());
        assert_eq!(chat["tools"].as_array().unwrap().len(), 1);
        assert_eq!(
            chat["tools"][0]["function"]["parameters"],
            json!({"type": "object"})
        );

        let m = chat["messages"].as_array().unwrap();
        assert_eq!(m.len(), 5);
        assert_eq!(m[0], json!({"role": "system", "content": "be terse"}));
        assert_eq!(m[1], json!({"role": "user", "content": "open main.rs"}));
        assert_eq!(m[2]["content"], "reading");
        assert_eq!(m[2]["tool_calls"][0]["id"], "t1");
        assert_eq!(
            m[2]["tool_calls"][0]["function"]["arguments"],
            r#"{"path":"main.rs"}"#
        );
        assert_eq!(
            m[3],
            json!({"role": "tool", "tool_call_id": "t1", "content": "fn main() {}"})
        );
        assert_eq!(m[4], json!({"role": "user", "content": "now explain it"}));
    }

    #[test]
    fn translates_images_and_tool_only_turns() {
        let req = json!({
            "max_tokens": 100,
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": "what is this?"},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}}
                ]},
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "t1", "name": "look", "input": {}}
                ]}
            ]
        });
        let chat = to_chat(&req, "m", None).unwrap();
        assert_eq!(chat["max_tokens"], 100);
        assert!(chat.get("tools").is_none());
        let m = chat["messages"].as_array().unwrap();
        assert_eq!(
            m[0]["content"][1],
            json!({"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}})
        );
        assert_eq!(m[1]["content"], Value::Null);
        assert_eq!(m[1]["tool_calls"][0]["function"]["arguments"], "{}");
    }

    #[test]
    fn rejects_malformed_requests() {
        assert!(to_chat(&json!({"messages": "hi"}), "m", None).is_err());
        assert!(to_chat(
            &json!({"messages": [{"role": "robot", "content": "x"}]}),
            "m",
            None
        )
        .is_err());
    }
}
