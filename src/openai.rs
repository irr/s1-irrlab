//! OpenAI chat-completions front end. Requests are forwarded as-is apart from
//! the `model` rewrite, so this module only reads the body for the decider and
//! patches the two fields the proxy owns.

use serde_json::{json, Value};

use crate::{
    decider::{Conversation, Role},
    upstream::Upstream,
};

/// Build the decider's view of a chat-completions request.
pub fn conversation(body: &Value) -> Conversation {
    let mut conv = Conversation {
        tools: body
            .get("tools")
            .and_then(Value::as_array)
            .map_or(0, Vec::len),
        ..Default::default()
    };
    let messages = body.get("messages").and_then(Value::as_array);
    for message in messages.into_iter().flatten() {
        let (mut text, images) = content_text(message.get("content"));
        conv.has_images |= images;
        match message.get("role").and_then(Value::as_str) {
            Some("system" | "developer") => {
                if !conv.system.is_empty() {
                    conv.system.push_str("\n\n");
                }
                conv.system.push_str(&text);
            }
            Some("user") => conv.push(Role::User, text),
            Some("assistant") => {
                let calls = message.get("tool_calls").and_then(Value::as_array);
                for call in calls.into_iter().flatten() {
                    let name = call.pointer("/function/name").and_then(Value::as_str);
                    text.push_str(&format!("\n[tool call: {}]", name.unwrap_or("?")));
                }
                conv.push(Role::Assistant, text.trim());
            }
            Some("tool" | "function") => conv.push(Role::Tool, text),
            _ => {}
        }
    }
    conv
}

/// Text of a message's `content` (string or parts) and whether it carries media.
fn content_text(content: Option<&Value>) -> (String, bool) {
    match content {
        Some(Value::String(s)) => (s.clone(), false),
        Some(Value::Array(parts)) => {
            let mut text = Vec::new();
            let mut media = false;
            for part in parts {
                match part.get("text").and_then(Value::as_str) {
                    Some(t) => text.push(t),
                    None => media = true,
                }
            }
            (text.join("\n"), media)
        }
        _ => (String::new(), false),
    }
}

/// Point the request at the chosen tier: its model name and output-token limit.
pub fn prepare(body: &mut Value, upstream: &Upstream) {
    body["model"] = json!(upstream.model);
    let Some(limit) = upstream.max_output_tokens else {
        return;
    };
    for key in ["max_tokens", "max_completion_tokens"] {
        if body
            .get(key)
            .and_then(Value::as_u64)
            .is_some_and(|n| n > limit)
        {
            body[key] = json!(limit);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_conversation_from_mixed_content() {
        let body = json!({
            "tools": [{"type": "function"}, {"type": "function"}],
            "messages": [
                {"role": "system", "content": "be brief"},
                {"role": "user", "content": [
                    {"type": "text", "text": "what is this?"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
                ]},
                {"role": "assistant", "content": null, "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "look", "arguments": "{}"}}
                ]},
                {"role": "tool", "tool_call_id": "c1", "content": "a cat"}
            ]
        });
        let conv = conversation(&body);
        assert_eq!(conv.system, "be brief");
        assert_eq!(conv.tools, 2);
        assert!(conv.has_images);
        assert_eq!(conv.turns.len(), 3);
        assert_eq!(conv.turns[0].text, "what is this?");
        assert_eq!(conv.turns[1].text, "[tool call: look]");
        assert_eq!(conv.turns[2].role, Role::Tool);
    }
}
