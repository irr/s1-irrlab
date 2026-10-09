//! The two model tiers and the HTTP client that talks to each.

use std::time::Duration;

use reqwest::header::HeaderMap;
use serde_json::{json, Value};

use crate::config::TierConfig;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Tier {
    Top,
    Flash,
}

impl Tier {
    pub fn as_str(self) -> &'static str {
        match self {
            Tier::Top => "top",
            Tier::Flash => "flash",
        }
    }

    pub fn other(self) -> Self {
        match self {
            Tier::Top => Tier::Flash,
            Tier::Flash => Tier::Top,
        }
    }
}

/// Client headers that must reach a Messages-capable upstream. Nothing else,
/// in particular no credentials, is forwarded.
#[derive(Debug, Default, Clone)]
pub struct MessagesHeaders {
    pub version: Option<String>,
    pub beta: Option<String>,
}

impl MessagesHeaders {
    pub fn extract(client: &HeaderMap) -> Self {
        let keep = |name: &str| {
            client
                .get(name)
                .and_then(|v| v.to_str().ok())
                .map(str::to_string)
        };
        Self {
            version: keep("anthropic-version"),
            beta: keep("anthropic-beta"),
        }
    }

    fn apply(&self, req: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        let mut req = req.header(
            "anthropic-version",
            self.version.as_deref().unwrap_or(DEFAULT_VERSION),
        );
        if let Some(beta) = &self.beta {
            req = req.header("anthropic-beta", beta);
        }
        req
    }
}

const DEFAULT_VERSION: &str = "2023-06-01";

/// One upstream, reachable both as a chat-completions server and, for servers
/// that serve it, as an Anthropic Messages server (`base_url` minus a trailing
/// `/v1` is not assumed: both paths hang off the configured `base_url`).
pub struct Upstream {
    client: reqwest::Client,
    chat_url: String,
    messages_url: String,
    api_key: Option<String>,
    pub model: String,
    pub max_output_tokens: Option<u64>,
}

impl Upstream {
    pub fn new(cfg: &TierConfig) -> reqwest::Result<Self> {
        let mut builder = reqwest::Client::builder().connect_timeout(Duration::from_secs(15));
        if let Some(secs) = cfg.timeout_secs {
            builder = builder.timeout(Duration::from_secs(secs));
        }
        Ok(Self {
            client: builder.build()?,
            chat_url: format!("{}/chat/completions", cfg.base_url),
            messages_url: format!("{}/messages", cfg.base_url),
            api_key: cfg.api_key.clone(),
            model: cfg.model.clone(),
            max_output_tokens: cfg.max_output_tokens,
        })
    }

    /// POST a chat-completions body. The caller's own credentials are never
    /// forwarded; only this tier's configured key (if any) is sent.
    pub async fn send(&self, body: &Value) -> reqwest::Result<reqwest::Response> {
        let mut req = self.client.post(&self.chat_url).json(body);
        if let Some(key) = &self.api_key {
            req = req.bearer_auth(key);
        }
        req.send().await
    }

    /// POST a Messages body verbatim, with the client's dialect headers kept.
    pub async fn send_messages(
        &self,
        body: &Value,
        dialect: &MessagesHeaders,
    ) -> reqwest::Result<reqwest::Response> {
        let mut req = self.client.post(&self.messages_url).json(body);
        req = dialect.apply(req);
        if let Some(key) = &self.api_key {
            req = req.bearer_auth(key);
        }
        req.send().await
    }

    /// The client's Messages body retargeted at this tier: its model name and
    /// output-token cap; everything else travels untouched.
    pub fn native_body(&self, request: &Value) -> Value {
        let mut body = request.clone();
        body["model"] = json!(&self.model);
        if let Some(cap) = self.max_output_tokens {
            if let Some(n) = body.get("max_tokens").and_then(Value::as_u64) {
                body["max_tokens"] = json!(n.min(cap));
            }
        }
        body
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn tier() -> TierConfig {
        TierConfig {
            base_url: "http://127.0.0.1:1/v1".into(),
            api_key: None,
            model: "qwen".into(),
            timeout_secs: None,
            max_output_tokens: Some(32768),
        }
    }

    #[test]
    fn native_body_keeps_the_client_body_and_rewrites_tier_fields() {
        let up = Upstream::new(&tier()).unwrap();
        let request = json!({
            "model": "claude-something",
            "max_tokens": 64000,
            "stream": true,
            "system": [{"type": "text", "text": "be terse", "cache_control": {"type": "ephemeral"}}],
            "metadata": {"user_id": "u"},
        });
        let body = up.native_body(&request);
        assert_eq!(body["model"], "qwen");
        assert_eq!(body["max_tokens"], 32768);
        assert_eq!(body["stream"], json!(true));
        assert_eq!(body["metadata"]["user_id"], "u");
        // Dialect fields travel verbatim, cache_control included.
        assert_eq!(body["system"][0]["cache_control"]["type"], "ephemeral");
    }

    #[test]
    fn native_body_without_cap_or_max_tokens_is_untouched_apart_from_model() {
        let up = Upstream::new(&TierConfig {
            max_output_tokens: None,
            ..tier()
        })
        .unwrap();
        let body = up.native_body(&json!({"model": "claude", "messages": []}));
        assert_eq!(body, json!({"model": "qwen", "messages": []}));
    }

    #[test]
    fn dialect_headers_only_keep_anthropic_metadata() {
        let mut headers = HeaderMap::new();
        headers.insert("anthropic-version", "2023-06-01".parse().unwrap());
        headers.insert("anthropic-beta", "tools-2024-04-04".parse().unwrap());
        headers.insert("x-api-key", "secret".parse().unwrap());
        let kept = MessagesHeaders::extract(&headers);
        assert_eq!(kept.version.as_deref(), Some("2023-06-01"));
        assert_eq!(kept.beta.as_deref(), Some("tools-2024-04-04"));
    }

    #[test]
    fn tier_alternation() {
        assert_eq!(Tier::Top.other(), Tier::Flash);
        assert_eq!(Tier::Flash.other(), Tier::Top);
        assert_eq!(Tier::Top.as_str(), "top");
    }
}
