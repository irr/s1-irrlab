//! The routing decision: build a compact description of the conversation, ask
//! the System One decider (`POST /v1/decisions`) which tier should answer, and
//! remember the answer for follow-up steps of the same user turn.

use std::{
    collections::{hash_map::DefaultHasher, HashMap},
    hash::{Hash, Hasher},
    sync::Mutex,
    time::{Duration, Instant},
};

use serde_json::{json, Value};
use thiserror::Error;
use tracing::{debug, warn};

use crate::{
    config::{DeciderConfig, OnError},
    upstream::Tier,
};

const QUESTION_ID: &str = "tier";
const INSTRUCTIONS: &str = "A coding agent is working in a software project. Which model tier \
     should write its next turn for the latest user request?";
const FLASH_CRITERIA: &str = "Routine coding work a small fast model handles reliably: reading, \
     searching or listing files, running a command and reporting its output, a single-file or \
     few-line edit with clear instructions, renames, formatting, lint or typo fixes, boilerplate, \
     simple tests, commit messages, short explanations of code, summaries, titles, \
     acknowledgements and other mechanical follow-ups";
const TOP_CRITERIA: &str = "Demanding software engineering that needs the strongest model: \
     planning or implementing a feature across several files, architecture or API design, \
     debugging a failure whose root cause is unknown, large refactors or migrations, \
     concurrency, performance or security work, code review, long autonomous multi-step tasks \
     with many tool calls, and vague or underspecified requirements";

// Character budgets keeping the state far below the decider's 16384-token cap.
const REQUEST_HEAD: usize = 4000;
const REQUEST_TAIL: usize = 2000;
const CONTEXT_TURNS: usize = 6;
const CONTEXT_CHARS: usize = 500;
const SYSTEM_CHARS: usize = 800;
const CACHE_PRUNE_AT: usize = 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone)]
pub struct Turn {
    pub role: Role,
    pub text: String,
}

/// Protocol-neutral view of a request, built by the OpenAI and Anthropic front ends.
#[derive(Debug, Clone, Default)]
pub struct Conversation {
    pub system: String,
    pub turns: Vec<Turn>,
    pub tools: usize,
    pub has_images: bool,
}

impl Conversation {
    pub fn push(&mut self, role: Role, text: impl Into<String>) {
        self.turns.push(Turn {
            role,
            text: text.into(),
        });
    }

    /// Index of the last turn the user actually wrote (not a tool result).
    fn latest_user(&self) -> Option<usize> {
        self.turns
            .iter()
            .rposition(|t| t.role == Role::User && !t.text.trim().is_empty())
    }

    /// The `state` handed to the decider.
    pub fn state(&self) -> Value {
        let latest = self.latest_user();
        let request = latest
            .map(|i| strip_reminders(&self.turns[i].text))
            .unwrap_or_default();
        let context: Vec<String> = self
            .turns
            .iter()
            .enumerate()
            .filter(|(i, t)| Some(*i) != latest && !t.text.trim().is_empty())
            .map(|(_, t)| {
                let role = match t.role {
                    Role::User => "user",
                    Role::Assistant => "assistant",
                    Role::Tool => "tool",
                };
                format!(
                    "{role}: {}",
                    clip(strip_reminders(&t.text).trim(), CONTEXT_CHARS, 0)
                )
            })
            .collect();
        let skip = context.len().saturating_sub(CONTEXT_TURNS);
        json!({
            "latest_user_request": clip(request.trim(), REQUEST_HEAD, REQUEST_TAIL),
            "recent_context": &context[skip..],
            "system_prompt_excerpt": clip(self.system.trim(), SYSTEM_CHARS, 0),
            "turns": self.turns.len(),
            "tools_available": self.tools,
            "has_images": self.has_images,
            "last_turn_is_tool_result": self.turns.last().is_some_and(|t| t.role == Role::Tool),
        })
    }

    /// Identifies one user turn of one conversation, so every step of an agent's
    /// tool loop maps to the same decision.
    fn sticky_key(&self) -> u64 {
        let mut h = DefaultHasher::new();
        self.system.hash(&mut h);
        let first = self.turns.iter().find(|t| t.role == Role::User);
        first.map(|t| t.text.as_str()).hash(&mut h);
        let latest = self.latest_user();
        latest.hash(&mut h);
        latest.map(|i| self.turns[i].text.as_str()).hash(&mut h);
        h.finish()
    }
}

/// Keep `head` leading and `tail` trailing characters, eliding the middle.
fn clip(text: &str, head: usize, tail: usize) -> String {
    let total = text.chars().count();
    if total <= head + tail {
        return text.to_string();
    }
    let start: String = text.chars().take(head).collect();
    let end: String = text.chars().skip(total - tail).collect();
    format!("{start} … {end}")
}

/// Agents inject `<system-reminder>` blocks into user turns; they are not part
/// of what the user asked, so they should not drive the decision.
fn strip_reminders(text: &str) -> String {
    const OPEN: &str = "<system-reminder>";
    const CLOSE: &str = "</system-reminder>";
    let mut out = String::new();
    let mut rest = text;
    while let Some(start) = rest.find(OPEN) {
        let Some(len) = rest[start..].find(CLOSE) else {
            break;
        };
        out.push_str(&rest[..start]);
        rest = &rest[start + len + CLOSE.len()..];
    }
    out.push_str(rest);
    if out.trim().is_empty() {
        text.to_string()
    } else {
        out
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    Decider,
    Cache,
    Fallback,
}

impl Source {
    pub fn as_str(self) -> &'static str {
        match self {
            Source::Decider => "decider",
            Source::Cache => "cache",
            Source::Fallback => "fallback",
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Decision {
    pub tier: Tier,
    /// Probability the decider gave the top tier; absent on fallback.
    pub p_top: Option<f64>,
    pub source: Source,
    pub elapsed: Duration,
    /// Set when the decided tier's model failed and the other tier answered instead.
    pub failed_over_from: Option<Tier>,
}

impl Decision {
    /// Value of the `x-s1-decision` response header.
    pub fn header(&self) -> String {
        let mut header = match self.p_top {
            Some(p) => format!("p={p:.4};src={}", self.source.as_str()),
            None => format!("src={}", self.source.as_str()),
        };
        if let Some(from) = self.failed_over_from {
            header.push_str(&format!(";failover_from={}", from.as_str()));
        }
        header
    }
}

#[derive(Debug, Error)]
pub enum DeciderError {
    #[error("decider request failed: {0}")]
    Http(#[from] reqwest::Error),
    #[error("decider returned HTTP {status}: {body}")]
    Status { status: u16, body: String },
    #[error("decider response has no usable `tier` answer: {0}")]
    Answer(String),
}

pub struct Decider {
    client: reqwest::Client,
    cfg: DeciderConfig,
    cache: Mutex<HashMap<u64, (Tier, f64, Instant)>>,
}

impl Decider {
    pub fn new(cfg: DeciderConfig) -> reqwest::Result<Self> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_millis(cfg.timeout_ms))
            .build()?;
        Ok(Self {
            client,
            cfg,
            cache: Mutex::new(HashMap::new()),
        })
    }

    /// Pick a tier. Only fails when `on_error = "fail"` and the decider is unusable.
    pub async fn decide(&self, conv: &Conversation) -> Result<Decision, DeciderError> {
        let started = Instant::now();
        let key = conv.sticky_key();
        if let Some((tier, p)) = self.cached(key) {
            return Ok(Decision {
                tier,
                p_top: Some(p),
                source: Source::Cache,
                elapsed: started.elapsed(),
                failed_over_from: None,
            });
        }
        match self.ask(conv).await {
            Ok(p) => {
                let tier = if p >= self.cfg.threshold {
                    Tier::Top
                } else {
                    Tier::Flash
                };
                self.store(key, tier, p);
                Ok(Decision {
                    tier,
                    p_top: Some(p),
                    source: Source::Decider,
                    elapsed: started.elapsed(),
                    failed_over_from: None,
                })
            }
            Err(err) => {
                let tier = match self.cfg.on_error {
                    OnError::Fail => return Err(err),
                    OnError::Top | OnError::Ha => Tier::Top,
                    OnError::Flash => Tier::Flash,
                };
                warn!(error = %err, fallback = tier.as_str(), "decider unavailable");
                Ok(Decision {
                    tier,
                    p_top: None,
                    source: Source::Fallback,
                    elapsed: started.elapsed(),
                    failed_over_from: None,
                })
            }
        }
    }

    /// Ask the decider and return p(top).
    async fn ask(&self, conv: &Conversation) -> Result<f64, DeciderError> {
        let body = json!({
            "model": self.cfg.model,
            "state": conv.state(),
            "questions": {
                QUESTION_ID: {
                    "type": "choice",
                    "instructions": INSTRUCTIONS,
                    "criteria": {"flash": FLASH_CRITERIA, "top": TOP_CRITERIA},
                }
            }
        });
        debug!(state = %body["state"], "asking decider");
        let mut req = self.client.post(&self.cfg.url).json(&body);
        if let Some(key) = &self.cfg.api_key {
            req = req.bearer_auth(key);
        }
        let resp = req.send().await?;
        let status = resp.status();
        if !status.is_success() {
            let body: String = resp.text().await.unwrap_or_default();
            return Err(DeciderError::Status {
                status: status.as_u16(),
                body: body.chars().take(500).collect(),
            });
        }
        let answer: Value = resp.json().await?;
        debug!(answer = %answer["answers"], "decider answered");
        p_top(&answer).ok_or_else(|| DeciderError::Answer(answer.to_string()))
    }

    fn cached(&self, key: u64) -> Option<(Tier, f64)> {
        let ttl = Duration::from_secs(self.cfg.sticky_ttl_secs);
        let cache = self.cache.lock().unwrap();
        let (tier, p, at) = cache.get(&key)?;
        (at.elapsed() < ttl).then_some((*tier, *p))
    }

    fn store(&self, key: u64, tier: Tier, p: f64) {
        if self.cfg.sticky_ttl_secs == 0 {
            return;
        }
        let ttl = Duration::from_secs(self.cfg.sticky_ttl_secs);
        let mut cache = self.cache.lock().unwrap();
        if cache.len() >= CACHE_PRUNE_AT {
            cache.retain(|_, (_, _, at)| at.elapsed() < ttl);
        }
        cache.insert(key, (tier, p, Instant::now()));
    }
}

/// p(top) from a `/v1/decisions` response, falling back to the hard `choice`.
fn p_top(answer: &Value) -> Option<f64> {
    let tier = answer.get("answers")?.get(QUESTION_ID)?;
    if let Some(p) = tier.pointer("/probabilities/top").and_then(Value::as_f64) {
        return Some(p);
    }
    match tier.get("choice")?.as_str()? {
        "top" => Some(1.0),
        "flash" => Some(0.0),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn conv(turns: &[(Role, &str)]) -> Conversation {
        let mut c = Conversation::default();
        for (role, text) in turns {
            c.push(*role, *text);
        }
        c
    }

    #[test]
    fn state_uses_latest_user_text_not_tool_result() {
        let c = conv(&[
            (Role::User, "refactor the parser"),
            (Role::Assistant, "[tool call: read_file]"),
            (Role::Tool, "fn parse() {}"),
        ]);
        let state = c.state();
        assert_eq!(state["latest_user_request"], "refactor the parser");
        assert_eq!(state["last_turn_is_tool_result"], true);
        assert_eq!(state["recent_context"].as_array().unwrap().len(), 2);
        assert_eq!(state["turns"], 3);
    }

    #[test]
    fn state_strips_reminders_and_clips() {
        let long = "x".repeat(10_000);
        let c = conv(&[(
            Role::User,
            &format!("<system-reminder>ignore me</system-reminder>{long}"),
        )]);
        let request = c.state()["latest_user_request"]
            .as_str()
            .unwrap()
            .to_string();
        assert!(!request.contains("ignore me"));
        assert_eq!(request.chars().count(), REQUEST_HEAD + REQUEST_TAIL + 3);
    }

    #[test]
    fn sticky_key_is_stable_across_a_tool_loop() {
        let first = conv(&[(Role::User, "fix the bug")]);
        let later = conv(&[
            (Role::User, "fix the bug"),
            (Role::Assistant, "[tool call: grep]"),
            (Role::Tool, "match"),
        ]);
        let next = conv(&[
            (Role::User, "fix the bug"),
            (Role::Assistant, "done"),
            (Role::User, "thanks"),
        ]);
        assert_eq!(first.sticky_key(), later.sticky_key());
        assert_ne!(first.sticky_key(), next.sticky_key());
    }

    #[test]
    fn reads_probability_then_choice() {
        let full = json!({"answers": {"tier": {"choice": "flash", "probabilities": {"flash": 0.3, "top": 0.7}}}});
        assert_eq!(p_top(&full), Some(0.7));
        assert_eq!(
            p_top(&json!({"answers": {"tier": {"choice": "flash"}}})),
            Some(0.0)
        );
        assert_eq!(p_top(&json!({"answers": {}})), None);
    }
}
