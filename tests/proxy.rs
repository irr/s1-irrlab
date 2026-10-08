//! End-to-end tests: the real router on an ephemeral port, with wiremock
//! standing in for the decider and both model tiers.

use s1_irrlab::{
    config::{Config, DeciderConfig, OnError, ServerConfig, TierConfig},
    server,
};
use serde_json::{json, Value};
use wiremock::{
    matchers::{body_partial_json, header, method, path},
    Mock, MockServer, ResponseTemplate,
};

struct Harness {
    proxy: String,
    decider: MockServer,
    top: MockServer,
    flash: MockServer,
    http: reqwest::Client,
}

impl Harness {
    async fn start() -> Self {
        Self::start_with(|_| {}).await
    }

    async fn start_with(tweak: impl FnOnce(&mut Config)) -> Self {
        let (decider, top, flash) = (
            MockServer::start().await,
            MockServer::start().await,
            MockServer::start().await,
        );
        let tier = |server: &MockServer, model: &str, key: Option<&str>| TierConfig {
            base_url: format!("{}/v1", server.uri()),
            api_key: key.map(str::to_string),
            model: model.to_string(),
            timeout_secs: None,
            max_output_tokens: None,
        };
        let mut cfg = Config {
            server: ServerConfig {
                host: "127.0.0.1".into(),
                port: 0,
            },
            top: tier(&top, "top-model", Some("top-secret")),
            flash: tier(&flash, "flash-model", None),
            decider: DeciderConfig {
                url: format!("{}/v1/decisions", decider.uri()),
                api_key: None,
                model: "clef-flash".into(),
                timeout_ms: 300,
                threshold: 0.5,
                on_error: OnError::Top,
                sticky_ttl_secs: 900,
            },
        };
        tweak(&mut cfg);

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let proxy = format!("http://{}", listener.local_addr().unwrap());
        let app = server::router(cfg).unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        Self {
            proxy,
            decider,
            top,
            flash,
            http: reqwest::Client::new(),
        }
    }

    /// The decider answers every request with this p(top).
    async fn decide(&self, p_top: f64) {
        let choice = if p_top >= 0.5 { "top" } else { "flash" };
        Mock::given(method("POST"))
            .and(path("/v1/decisions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "model": "clef-flash",
                "answers": {"tier": {
                    "type": "choice",
                    "choice": choice,
                    "confidence": p_top.max(1.0 - p_top),
                    "probabilities": {"flash": 1.0 - p_top, "top": p_top},
                }},
                "usage": {"input_tokens": 200, "output_tokens": 0},
            })))
            .mount(&self.decider)
            .await;
    }

    async fn post(&self, route: &str, body: Value) -> reqwest::Response {
        self.http
            .post(format!("{}{route}", self.proxy))
            .header("authorization", "Bearer client-key")
            .header("x-api-key", "client-key")
            .json(&body)
            .send()
            .await
            .unwrap()
    }
}

fn completion(text: &str) -> ResponseTemplate {
    ResponseTemplate::new(200).set_body_json(json!({
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": text}}],
        "usage": {"prompt_tokens": 9, "completion_tokens": 3, "total_tokens": 12},
    }))
}

fn chat_endpoint() -> wiremock::MockBuilder {
    Mock::given(method("POST")).and(path("/v1/chat/completions"))
}

fn chat_request(prompt: &str) -> Value {
    json!({"model": "anything", "messages": [{"role": "user", "content": prompt}]})
}

fn header_of<'a>(response: &'a reqwest::Response, name: &str) -> &'a str {
    response.headers().get(name).unwrap().to_str().unwrap()
}

#[tokio::test]
async fn complex_prompt_goes_to_top_with_its_key_and_model() {
    let h = Harness::start().await;
    h.decide(0.9).await;
    chat_endpoint()
        .and(header("authorization", "Bearer top-secret"))
        .and(body_partial_json(
            json!({"model": "top-model", "temperature": 0.2}),
        ))
        .respond_with(completion("from top"))
        .expect(1)
        .mount(&h.top)
        .await;
    chat_endpoint()
        .respond_with(completion("from flash"))
        .expect(0)
        .mount(&h.flash)
        .await;

    let mut request = chat_request("design a distributed lock service");
    request["temperature"] = json!(0.2);
    let response = h.post("/v1/chat/completions", request).await;

    assert_eq!(response.status(), 200);
    assert_eq!(header_of(&response, "x-s1-tier"), "top");
    assert_eq!(
        header_of(&response, "x-s1-decision"),
        "p=0.9000;src=decider"
    );
    let body: Value = response.json().await.unwrap();
    assert_eq!(body["choices"][0]["message"]["content"], "from top");

    // The decider saw the prompt, and was asked the tier question.
    let asked: Value = h.decider.received_requests().await.unwrap()[0]
        .body_json()
        .unwrap();
    assert_eq!(
        asked["state"]["latest_user_request"],
        "design a distributed lock service"
    );
    assert_eq!(asked["questions"]["tier"]["type"], "choice");
    assert!(asked["questions"]["tier"]["criteria"]["top"].is_string());
}

#[tokio::test]
async fn simple_prompt_goes_to_flash_without_credentials() {
    let h = Harness::start().await;
    h.decide(0.1).await;
    chat_endpoint()
        .respond_with(completion("from top"))
        .expect(0)
        .mount(&h.top)
        .await;
    chat_endpoint()
        .and(body_partial_json(json!({"model": "flash-model"})))
        .respond_with(completion("from flash"))
        .expect(1)
        .mount(&h.flash)
        .await;

    let response = h.post("/v1/chat/completions", chat_request("say hi")).await;

    assert_eq!(header_of(&response, "x-s1-tier"), "flash");
    let seen = &h.flash.received_requests().await.unwrap()[0];
    assert!(
        !seen.headers.contains_key("authorization"),
        "client key leaked upstream"
    );
    assert!(
        !seen.headers.contains_key("x-api-key"),
        "client key leaked upstream"
    );
}

#[tokio::test]
async fn threshold_is_configurable() {
    let h = Harness::start_with(|cfg| cfg.decider.threshold = 0.8).await;
    h.decide(0.7).await;
    chat_endpoint()
        .respond_with(completion("from flash"))
        .expect(1)
        .mount(&h.flash)
        .await;

    let response = h
        .post("/v1/chat/completions", chat_request("medium task"))
        .await;
    assert_eq!(header_of(&response, "x-s1-tier"), "flash");
}

#[tokio::test]
async fn decider_failures_fall_back_to_top() {
    for failure in [
        ResponseTemplate::new(400)
            .set_body_json(json!({"error": "schema requires too many tokens"})),
        ResponseTemplate::new(200).set_body_json(json!({"answers": {}})),
        ResponseTemplate::new(200)
            .set_body_json(json!({"answers": {"tier": {"choice": "top"}}}))
            .set_delay(std::time::Duration::from_millis(1500)),
    ] {
        let h = Harness::start().await;
        Mock::given(path("/v1/decisions"))
            .respond_with(failure)
            .mount(&h.decider)
            .await;
        chat_endpoint()
            .respond_with(completion("from top"))
            .expect(1)
            .mount(&h.top)
            .await;

        let response = h.post("/v1/chat/completions", chat_request("say hi")).await;
        assert_eq!(response.status(), 200);
        assert_eq!(header_of(&response, "x-s1-tier"), "top");
        assert_eq!(header_of(&response, "x-s1-decision"), "src=fallback");
    }
}

#[tokio::test]
async fn unreachable_decider_can_fail_the_request_instead() {
    let h = Harness::start_with(|cfg| {
        cfg.decider.url = "http://127.0.0.1:1/v1/decisions".into();
        cfg.decider.on_error = OnError::Fail;
    })
    .await;
    chat_endpoint()
        .respond_with(completion("x"))
        .expect(0)
        .mount(&h.top)
        .await;

    let openai = h.post("/v1/chat/completions", chat_request("hi")).await;
    assert_eq!(openai.status(), 503);
    assert!(openai.json::<Value>().await.unwrap()["error"]["message"].is_string());

    let anthropic = h
        .post(
            "/v1/messages",
            json!({"max_tokens": 10, "messages": [{"role": "user", "content": "hi"}]}),
        )
        .await;
    assert_eq!(anthropic.status(), 503);
    let body: Value = anthropic.json().await.unwrap();
    assert_eq!(
        (&body["type"], &body["error"]["type"]),
        (&json!("error"), &json!("overloaded_error"))
    );
}

#[tokio::test]
async fn ha_fails_over_to_the_other_tier_when_the_model_errors() {
    let h = Harness::start_with(|cfg| cfg.decider.on_error = OnError::Ha).await;
    h.decide(0.9).await;
    chat_endpoint()
        .respond_with(ResponseTemplate::new(503).set_body_string("overloaded"))
        .expect(1)
        .mount(&h.top)
        .await;
    chat_endpoint()
        .and(body_partial_json(json!({"model": "flash-model"})))
        .respond_with(completion("from flash"))
        .expect(1)
        .mount(&h.flash)
        .await;

    let response = h
        .post("/v1/chat/completions", chat_request("refactor everything"))
        .await;
    assert_eq!(response.status(), 200);
    assert_eq!(header_of(&response, "x-s1-tier"), "flash");
    assert_eq!(
        header_of(&response, "x-s1-decision"),
        "p=0.9000;src=decider;failover_from=top"
    );
    let body: Value = response.json().await.unwrap();
    assert_eq!(body["choices"][0]["message"]["content"], "from flash");
}

#[tokio::test]
async fn ha_survives_decider_and_top_tier_both_being_down() {
    let h = Harness::start_with(|cfg| {
        cfg.decider.on_error = OnError::Ha;
        cfg.decider.url = "http://127.0.0.1:1/v1/decisions".into();
        cfg.top.base_url = "http://127.0.0.1:1/v1".into();
    })
    .await;
    chat_endpoint()
        .and(body_partial_json(json!({"model": "flash-model"})))
        .respond_with(completion("still here"))
        .expect(2)
        .mount(&h.flash)
        .await;

    let openai = h.post("/v1/chat/completions", chat_request("hi")).await;
    assert_eq!(openai.status(), 200);
    assert_eq!(
        header_of(&openai, "x-s1-decision"),
        "src=fallback;failover_from=top"
    );

    // The Anthropic body is rebuilt for the tier that actually answers.
    let anthropic = h
        .post(
            "/v1/messages",
            json!({"max_tokens": 10, "messages": [{"role": "user", "content": "hi"}]}),
        )
        .await;
    assert_eq!(anthropic.status(), 200);
    assert_eq!(header_of(&anthropic, "x-s1-tier"), "flash");
    let body: Value = anthropic.json().await.unwrap();
    assert_eq!(body["model"], "flash-model");
    assert_eq!(body["content"][0]["text"], "still here");
}

#[tokio::test]
async fn ha_does_not_retry_requests_the_model_rejected() {
    let h = Harness::start_with(|cfg| cfg.decider.on_error = OnError::Ha).await;
    h.decide(0.9).await;
    chat_endpoint()
        .respond_with(
            ResponseTemplate::new(400)
                .set_body_json(json!({"error": {"message": "bad tool schema"}})),
        )
        .expect(1)
        .mount(&h.top)
        .await;
    chat_endpoint()
        .respond_with(completion("x"))
        .expect(0)
        .mount(&h.flash)
        .await;

    let response = h.post("/v1/chat/completions", chat_request("hi")).await;
    assert_eq!(response.status(), 400);
    assert_eq!(
        header_of(&response, "x-s1-decision"),
        "p=0.9000;src=decider"
    );
}

#[tokio::test]
async fn ha_reports_the_last_error_when_both_tiers_fail() {
    let h = Harness::start_with(|cfg| cfg.decider.on_error = OnError::Ha).await;
    h.decide(0.1).await;
    chat_endpoint()
        .respond_with(ResponseTemplate::new(500))
        .expect(1)
        .mount(&h.flash)
        .await;
    chat_endpoint()
        .respond_with(
            ResponseTemplate::new(429).set_body_json(json!({"error": {"message": "slow down"}})),
        )
        .expect(1)
        .mount(&h.top)
        .await;

    let response = h.post("/v1/chat/completions", chat_request("hi")).await;
    assert_eq!(response.status(), 429);
    assert_eq!(header_of(&response, "x-s1-tier"), "top");
    assert_eq!(
        header_of(&response, "x-s1-decision"),
        "p=0.1000;src=decider;failover_from=flash"
    );
}

#[tokio::test]
async fn without_ha_a_failing_model_is_not_replaced() {
    let h = Harness::start().await;
    h.decide(0.9).await;
    chat_endpoint()
        .respond_with(ResponseTemplate::new(503))
        .expect(1)
        .mount(&h.top)
        .await;
    chat_endpoint()
        .respond_with(completion("x"))
        .expect(0)
        .mount(&h.flash)
        .await;

    let response = h.post("/v1/chat/completions", chat_request("hi")).await;
    assert_eq!(response.status(), 503);
    assert_eq!(header_of(&response, "x-s1-tier"), "top");
}

#[tokio::test]
async fn tool_loop_reuses_the_decision_and_new_turns_ask_again() {
    let h = Harness::start().await;
    h.decide(0.9).await;
    chat_endpoint()
        .respond_with(completion("ok"))
        .mount(&h.top)
        .await;

    let first = chat_request("refactor the parser");
    let mut follow_up = first.clone();
    follow_up["messages"].as_array_mut().unwrap().extend([
        json!({"role": "assistant", "content": null, "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "read", "arguments": "{}"}}
        ]}),
        json!({"role": "tool", "tool_call_id": "c1", "content": "fn parse() {}"}),
    ]);
    let mut next_turn = follow_up.clone();
    next_turn["messages"].as_array_mut().unwrap().extend([
        json!({"role": "assistant", "content": "done"}),
        json!({"role": "user", "content": "thanks"}),
    ]);

    let a = h.post("/v1/chat/completions", first).await;
    let b = h.post("/v1/chat/completions", follow_up).await;
    assert_eq!(header_of(&a, "x-s1-decision"), "p=0.9000;src=decider");
    assert_eq!(header_of(&b, "x-s1-decision"), "p=0.9000;src=cache");
    assert_eq!(h.decider.received_requests().await.unwrap().len(), 1);

    let c = h.post("/v1/chat/completions", next_turn).await;
    assert_eq!(header_of(&c, "x-s1-decision"), "p=0.9000;src=decider");
    assert_eq!(h.decider.received_requests().await.unwrap().len(), 2);
}

#[tokio::test]
async fn openai_stream_and_upstream_errors_pass_through_untouched() {
    let h = Harness::start().await;
    h.decide(0.1).await;
    let sse = "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\ndata: [DONE]\n\n";
    chat_endpoint()
        .and(body_partial_json(json!({"stream": true})))
        .respond_with(ResponseTemplate::new(200).set_body_raw(sse, "text/event-stream"))
        .mount(&h.flash)
        .await;
    chat_endpoint()
        .respond_with(
            ResponseTemplate::new(429).set_body_json(json!({"error": {"message": "slow down"}})),
        )
        .mount(&h.flash)
        .await;

    let mut streaming = chat_request("say hi");
    streaming["stream"] = json!(true);
    let response = h.post("/v1/chat/completions", streaming).await;
    assert_eq!(header_of(&response, "content-type"), "text/event-stream");
    assert_eq!(response.text().await.unwrap(), sse);

    let response = h
        .post("/v1/chat/completions", chat_request("say hi again"))
        .await;
    assert_eq!(response.status(), 429);
    assert_eq!(header_of(&response, "x-s1-tier"), "flash");
    assert_eq!(
        response.json::<Value>().await.unwrap()["error"]["message"],
        "slow down"
    );
}

#[tokio::test]
async fn anthropic_request_is_translated_both_ways() {
    let h = Harness::start().await;
    h.decide(0.1).await;
    chat_endpoint()
        .and(body_partial_json(json!({
            "model": "flash-model",
            "max_tokens": 256,
            "messages": [
                {"role": "system", "content": "be terse"},
                {"role": "user", "content": "what is in main.rs?"},
            ],
            "tools": [{"type": "function", "function": {"name": "read", "parameters": {"type": "object"}}}],
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                "role": "assistant",
                "content": null,
                "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "read", "arguments": "{\"path\":\"main.rs\"}"}}],
            }}],
            "usage": {"prompt_tokens": 30, "completion_tokens": 8},
        })))
        .expect(1)
        .mount(&h.flash)
        .await;

    let response = h
        .post(
            "/v1/messages",
            json!({
                "model": "claude-something",
                "max_tokens": 256,
                "system": "be terse",
                "tools": [{"name": "read", "input_schema": {"type": "object"}}],
                "messages": [{"role": "user", "content": [{"type": "text", "text": "what is in main.rs?"}]}],
            }),
        )
        .await;

    assert_eq!(response.status(), 200);
    assert_eq!(header_of(&response, "x-s1-tier"), "flash");
    let body: Value = response.json().await.unwrap();
    assert_eq!(body["type"], "message");
    assert_eq!(body["role"], "assistant");
    assert_eq!(body["model"], "flash-model");
    assert_eq!(body["stop_reason"], "tool_use");
    assert_eq!(
        body["content"],
        json!([{"type": "tool_use", "id": "call_1", "name": "read", "input": {"path": "main.rs"}}])
    );
    assert_eq!(body["usage"]["input_tokens"], 30);
    assert_eq!(body["usage"]["output_tokens"], 8);
}

#[tokio::test]
async fn anthropic_stream_is_translated_to_message_events() {
    let h = Harness::start().await;
    h.decide(0.9).await;
    let upstream = [
        json!({"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}]}),
        json!({"choices": [{"index": 0, "delta": {"content": "Hel"}}]}),
        json!({"choices": [{"index": 0, "delta": {"content": "lo"}}]}),
        json!({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
        json!({"choices": [], "usage": {"prompt_tokens": 11, "completion_tokens": 2}}),
    ]
    .iter()
    .map(|chunk| format!("data: {chunk}\n\n"))
    .collect::<String>()
        + "data: [DONE]\n\n";
    chat_endpoint()
        .and(body_partial_json(
            json!({"stream": true, "stream_options": {"include_usage": true}}),
        ))
        .respond_with(ResponseTemplate::new(200).set_body_raw(upstream, "text/event-stream"))
        .expect(1)
        .mount(&h.top)
        .await;

    let response = h
        .post(
            "/v1/messages",
            json!({"max_tokens": 64, "stream": true, "messages": [{"role": "user", "content": "greet me"}]}),
        )
        .await;
    assert_eq!(header_of(&response, "content-type"), "text/event-stream");
    assert_eq!(header_of(&response, "x-s1-tier"), "top");

    let text = response.text().await.unwrap();
    let events: Vec<(&str, Value)> = text
        .split("\n\n")
        .filter(|frame| !frame.is_empty())
        .map(|frame| {
            let (event, data) = frame.split_once('\n').unwrap();
            (
                event.strip_prefix("event: ").unwrap(),
                serde_json::from_str(data.strip_prefix("data: ").unwrap()).unwrap(),
            )
        })
        .collect();
    let names: Vec<&str> = events.iter().map(|(name, _)| *name).collect();
    assert_eq!(
        names,
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
    let said: String = events
        .iter()
        .filter_map(|(_, data)| data["delta"]["text"].as_str())
        .collect();
    assert_eq!(said, "Hello");
    assert_eq!(events[0].1["message"]["model"], "top-model");
    assert_eq!(events[5].1["delta"]["stop_reason"], "end_turn");
    assert_eq!(events[5].1["usage"]["output_tokens"], 2);
}

#[tokio::test]
async fn anthropic_callers_get_anthropic_shaped_errors() {
    let h = Harness::start().await;
    h.decide(0.9).await;
    chat_endpoint()
        .respond_with(
            ResponseTemplate::new(429).set_body_json(json!({"error": {"message": "slow down"}})),
        )
        .mount(&h.top)
        .await;

    let upstream_error = h
        .post(
            "/v1/messages",
            json!({"max_tokens": 10, "messages": [{"role": "user", "content": "hi"}]}),
        )
        .await;
    assert_eq!(upstream_error.status(), 429);
    assert_eq!(
        upstream_error.json::<Value>().await.unwrap(),
        json!({"type": "error", "error": {"type": "rate_limit_error", "message": "slow down"}})
    );

    let bad_request = h.post("/v1/messages", json!({"messages": "nope"})).await;
    assert_eq!(bad_request.status(), 400);
    assert_eq!(
        bad_request.json::<Value>().await.unwrap()["error"]["type"],
        "invalid_request_error"
    );
}

#[tokio::test]
async fn auxiliary_endpoints() {
    let h = Harness::start().await;

    let models: Value = h
        .http
        .get(format!("{}/v1/models", h.proxy))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let ids: Vec<&str> = models["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["id"].as_str().unwrap())
        .collect();
    assert_eq!(ids, ["s1-auto", "top-model", "flash-model"]);

    let health = h
        .http
        .get(format!("{}/health", h.proxy))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(health.contains("\"status\":\"ok\""));
    assert!(
        !health.contains("top-secret"),
        "health must not expose keys"
    );

    let count: Value = h
        .post(
            "/v1/messages/count_tokens",
            json!({"messages": [{"role": "user", "content": "hello there"}]}),
        )
        .await
        .json()
        .await
        .unwrap();
    assert!(count["input_tokens"].as_u64().unwrap() > 0);
    assert!(h.decider.received_requests().await.unwrap().is_empty());

    let missing = h.post("/v1/responses", json!({})).await;
    assert_eq!(missing.status(), 404);
}
