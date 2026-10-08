//! The two model tiers and the HTTP client that talks to each.

use std::time::Duration;

use serde_json::Value;

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

/// One OpenAI-compatible chat-completions upstream.
pub struct Upstream {
    client: reqwest::Client,
    url: String,
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
            url: format!("{}/chat/completions", cfg.base_url),
            api_key: cfg.api_key.clone(),
            model: cfg.model.clone(),
            max_output_tokens: cfg.max_output_tokens,
        })
    }

    /// POST a chat-completions body. The caller's own credentials are never
    /// forwarded; only this tier's configured key (if any) is sent.
    pub async fn send(&self, body: &Value) -> reqwest::Result<reqwest::Response> {
        let mut req = self.client.post(&self.url).json(body);
        if let Some(key) = &self.api_key {
            req = req.bearer_auth(key);
        }
        req.send().await
    }
}
