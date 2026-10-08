//! OpenAI `chat.completion.chunk` stream → Anthropic Messages event stream.

use std::{collections::HashSet, convert::Infallible, time::Duration};

use bytes::Bytes;
use eventsource_stream::Eventsource;
use futures::{Stream, StreamExt};
use serde_json::{json, Value};
use tracing::warn;

use super::response::{message_id, reasoning, stop_reason, tool_use_id, usage};

const PING_EVERY: Duration = Duration::from_secs(15);

#[derive(Debug, Clone, PartialEq)]
pub struct SseEvent {
    pub event: &'static str,
    pub data: Value,
}

impl SseEvent {
    fn new(event: &'static str, mut data: Value) -> Self {
        data["type"] = json!(event);
        Self { event, data }
    }

    pub fn encode(&self) -> Bytes {
        Bytes::from(format!("event: {}\ndata: {}\n\n", self.event, self.data))
    }
}

/// The content block currently receiving deltas.
#[derive(Debug, PartialEq)]
enum Open {
    None,
    Thinking,
    Text,
    Tool { index: u64, id: String },
}

/// Turns chunks into Anthropic events. Anthropic content blocks are strictly
/// sequential, so a block is closed as soon as output moves on to another one.
pub struct StreamTranslator {
    model: String,
    open: Open,
    /// Blocks started so far; the open block's index is `blocks - 1`.
    blocks: usize,
    /// Upstream tool-call indices that already have a block.
    tools_seen: HashSet<u64>,
    finish_reason: Option<String>,
    usage: Option<Value>,
}

impl StreamTranslator {
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            open: Open::None,
            blocks: 0,
            tools_seen: HashSet::new(),
            finish_reason: None,
            usage: None,
        }
    }

    pub fn start(&self) -> Vec<SseEvent> {
        vec![SseEvent::new(
            "message_start",
            json!({"message": {
                "id": message_id(),
                "type": "message",
                "role": "assistant",
                "model": self.model,
                "content": [],
                "stop_reason": null,
                "stop_sequence": null,
                "usage": usage(None),
            }}),
        )]
    }

    pub fn on_chunk(&mut self, chunk: &Value) -> Vec<SseEvent> {
        let mut out = Vec::new();
        if let Some(u) = chunk.get("usage").filter(|u| u.is_object()) {
            self.usage = Some(u.clone());
        }
        let Some(choice) = chunk.pointer("/choices/0") else {
            return out;
        };
        if let Some(reason) = choice.get("finish_reason").and_then(Value::as_str) {
            self.finish_reason = Some(reason.to_string());
        }
        let Some(delta) = choice.get("delta") else {
            return out;
        };

        if let Some(thinking) = reasoning(delta) {
            if self.open != Open::Thinking {
                self.open_block(
                    &mut out,
                    Open::Thinking,
                    json!({"type": "thinking", "thinking": "", "signature": ""}),
                );
            }
            self.delta(
                &mut out,
                json!({"type": "thinking_delta", "thinking": thinking}),
            );
        }
        if let Some(text) = delta
            .get("content")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
        {
            if self.open != Open::Text {
                self.open_block(&mut out, Open::Text, json!({"type": "text", "text": ""}));
            }
            self.delta(&mut out, json!({"type": "text_delta", "text": text}));
        }
        let calls = delta.get("tool_calls").and_then(Value::as_array);
        for call in calls.into_iter().flatten() {
            self.tool_delta(&mut out, call);
        }
        out
    }

    fn tool_delta(&mut self, out: &mut Vec<SseEvent>, call: &Value) {
        let index = call.get("index").and_then(Value::as_u64).unwrap_or(0);
        let id = call
            .get("id")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty());
        // Continues the open block unless the index changes, or a new id shows up
        // on the same index (some gateways number every parallel call 0).
        let continues = matches!(&self.open, Open::Tool { index: i, id: cur }
            if *i == index && id.is_none_or(|id| id == cur));
        if !continues {
            if id.is_none() && self.tools_seen.contains(&index) {
                warn!(
                    index,
                    "dropping tool-call delta for an already closed block"
                );
                return;
            }
            let id = id.map_or_else(tool_use_id, str::to_string);
            let name = call
                .pointer("/function/name")
                .and_then(Value::as_str)
                .unwrap_or("");
            let block = json!({"type": "tool_use", "id": id, "name": name, "input": {}});
            self.tools_seen.insert(index);
            self.open_block(out, Open::Tool { index, id }, block);
        }
        let arguments = call.pointer("/function/arguments").and_then(Value::as_str);
        if let Some(partial) = arguments.filter(|s| !s.is_empty()) {
            self.delta(
                out,
                json!({"type": "input_json_delta", "partial_json": partial}),
            );
        }
    }

    fn open_block(&mut self, out: &mut Vec<SseEvent>, open: Open, block: Value) {
        self.close_block(out);
        self.open = open;
        out.push(SseEvent::new(
            "content_block_start",
            json!({"index": self.blocks, "content_block": block}),
        ));
        self.blocks += 1;
    }

    fn close_block(&mut self, out: &mut Vec<SseEvent>) {
        if self.open != Open::None {
            self.open = Open::None;
            out.push(SseEvent::new(
                "content_block_stop",
                json!({"index": self.blocks - 1}),
            ));
        }
    }

    fn delta(&self, out: &mut Vec<SseEvent>, delta: Value) {
        out.push(SseEvent::new(
            "content_block_delta",
            json!({"index": self.blocks - 1, "delta": delta}),
        ));
    }

    /// Close the message once the upstream stream has ended.
    pub fn finish(&mut self) -> Vec<SseEvent> {
        let mut out = Vec::new();
        if self.blocks == 0 {
            self.open_block(&mut out, Open::Text, json!({"type": "text", "text": ""}));
        }
        self.close_block(&mut out);
        let reason = stop_reason(self.finish_reason.as_deref(), !self.tools_seen.is_empty());
        out.push(SseEvent::new(
            "message_delta",
            json!({
                "delta": {"stop_reason": reason, "stop_sequence": null},
                "usage": usage(self.usage.as_ref()),
            }),
        ));
        out.push(SseEvent::new("message_stop", json!({})));
        out
    }

    pub fn error(message: &str) -> SseEvent {
        SseEvent::new(
            "error",
            json!({"error": {"type": "api_error", "message": message}}),
        )
    }
}

enum Step {
    Events(Vec<SseEvent>),
    Ping,
    Failed(String),
    Done,
}

/// Translate an upstream SSE response body into an Anthropic SSE body.
pub fn translate(
    upstream: reqwest::Response,
    model: String,
) -> impl Stream<Item = Result<Bytes, Infallible>> {
    async_stream::stream! {
        let mut translator = StreamTranslator::new(model);
        for event in translator.start() {
            yield Ok(event.encode());
        }
        let mut events = Box::pin(upstream.bytes_stream().eventsource());
        let mut ping = tokio::time::interval_at(tokio::time::Instant::now() + PING_EVERY, PING_EVERY);
        loop {
            let step = tokio::select! {
                item = events.next() => match item {
                    None => Step::Done,
                    Some(Err(err)) => Step::Failed(format!("upstream stream failed: {err}")),
                    Some(Ok(event)) if event.data.trim() == "[DONE]" => Step::Done,
                    Some(Ok(event)) => match serde_json::from_str::<Value>(&event.data) {
                        Ok(chunk) => match chunk.get("error") {
                            Some(err) => Step::Failed(crate::error::upstream_message(&json!({"error": err}).to_string())),
                            None => Step::Events(translator.on_chunk(&chunk)),
                        },
                        Err(_) => Step::Events(Vec::new()),
                    },
                },
                _ = ping.tick() => Step::Ping,
            };
            match step {
                Step::Events(batch) => {
                    for event in batch {
                        yield Ok(event.encode());
                    }
                }
                Step::Ping => yield Ok(SseEvent::new("ping", json!({})).encode()),
                Step::Failed(message) => {
                    warn!(%message, "anthropic stream aborted");
                    yield Ok(StreamTranslator::error(&message).encode());
                    return;
                }
                Step::Done => break,
            }
        }
        for event in translator.finish() {
            yield Ok(event.encode());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(chunks: &[Value]) -> Vec<SseEvent> {
        let mut t = StreamTranslator::new("qwen");
        let mut events = t.start();
        for chunk in chunks {
            events.extend(t.on_chunk(chunk));
        }
        events.extend(t.finish());
        events
    }

    fn delta(delta: Value) -> Value {
        json!({"choices": [{"index": 0, "delta": delta, "finish_reason": null}]})
    }

    fn names(events: &[SseEvent]) -> Vec<&'static str> {
        events.iter().map(|e| e.event).collect()
    }

    #[test]
    fn text_stream() {
        let events = run(&[
            delta(json!({"role": "assistant", "content": ""})),
            delta(json!({"content": "Hel"})),
            delta(json!({"content": "lo"})),
            json!({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
            json!({"choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": 2}}),
        ]);
        assert_eq!(
            names(&events),
            [
                "message_start",
                "content_block_start",
                "content_block_delta",
                "content_block_delta",
                "content_block_stop",
                "message_delta",
                "message_stop"
            ]
        );
        assert_eq!(events[0].data["message"]["model"], "qwen");
        assert_eq!(
            events[2].data["delta"],
            json!({"type": "text_delta", "text": "Hel"})
        );
        assert_eq!(events[5].data["delta"]["stop_reason"], "end_turn");
        assert_eq!(events[5].data["usage"]["input_tokens"], 12);
        assert_eq!(events[5].data["usage"]["output_tokens"], 2);
        assert_eq!(events[5].data["type"], "message_delta");
    }

    #[test]
    fn thinking_then_text_then_parallel_tools() {
        let tool = |index: u64, id: Option<&str>, name: Option<&str>, args: &str| {
            delta(
                json!({"tool_calls": [{"index": index, "id": id, "function": {"name": name, "arguments": args}}]}),
            )
        };
        let events = run(&[
            delta(json!({"reasoning_content": "think"})),
            delta(json!({"content": "ok"})),
            tool(0, Some("call_a"), Some("read"), ""),
            tool(0, None, None, "{\"path\":"),
            tool(0, None, None, "\"a.rs\"}"),
            tool(1, Some("call_b"), Some("grep"), "{}"),
            json!({"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}),
        ]);
        let starts: Vec<&Value> = events
            .iter()
            .filter(|e| e.event == "content_block_start")
            .map(|e| &e.data)
            .collect();
        assert_eq!(starts.len(), 4);
        assert_eq!(starts[0]["content_block"]["type"], "thinking");
        assert_eq!(starts[1]["content_block"]["type"], "text");
        assert_eq!(
            starts[2]["content_block"],
            json!({"type": "tool_use", "id": "call_a", "name": "read", "input": {}})
        );
        assert_eq!(
            (&starts[3]["index"], &starts[3]["content_block"]["id"]),
            (&json!(3), &json!("call_b"))
        );

        // Every block is stopped before the next one starts.
        let mut open = None;
        for e in &events {
            match e.event {
                "content_block_start" => assert_eq!(open.replace(e.data["index"].clone()), None),
                "content_block_delta" => assert_eq!(open.as_ref(), Some(&e.data["index"])),
                "content_block_stop" => assert_eq!(open.take().as_ref(), Some(&e.data["index"])),
                _ => {}
            }
        }
        let json_parts: String = events
            .iter()
            .filter(|e| e.data["index"] == 2 && e.data["delta"]["type"] == "input_json_delta")
            .map(|e| e.data["delta"]["partial_json"].as_str().unwrap())
            .collect();
        assert_eq!(json_parts, "{\"path\":\"a.rs\"}");
        let message_delta = events.iter().find(|e| e.event == "message_delta").unwrap();
        assert_eq!(message_delta.data["delta"]["stop_reason"], "tool_use");
    }

    #[test]
    fn same_index_new_id_starts_a_new_tool_block() {
        let call = |id: &str| {
            delta(
                json!({"tool_calls": [{"index": 0, "id": id, "function": {"name": "f", "arguments": "{}"}}]}),
            )
        };
        let events = run(&[call("a"), call("b")]);
        assert_eq!(
            events
                .iter()
                .filter(|e| e.event == "content_block_start")
                .count(),
            2
        );
    }

    #[test]
    fn empty_stream_emits_one_empty_text_block() {
        let events = run(&[]);
        assert_eq!(
            names(&events),
            [
                "message_start",
                "content_block_start",
                "content_block_stop",
                "message_delta",
                "message_stop"
            ]
        );
    }
}
