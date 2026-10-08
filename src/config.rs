//! `s1.toml` loading, `{env:VAR}` expansion and validation.
//!
//! Both tiers and the decider must be fully configured; the proxy refuses to
//! start otherwise.

use std::path::Path;

use serde::Deserialize;
use thiserror::Error;

const PLACEHOLDER: &str = "REPLACE_ME";

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("cannot read config {path}: {source}")]
    Read {
        path: String,
        source: std::io::Error,
    },
    #[error("invalid TOML in config: {0}")]
    Parse(#[from] toml::de::Error),
    #[error("invalid config:\n  - {}", .0.join("\n  - "))]
    Invalid(Vec<String>),
}

/// What to do with a request when the decider cannot produce a decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OnError {
    Top,
    Flash,
    Fail,
    /// High availability: like `Top`, and additionally a request whose model
    /// fails is retried on the other tier.
    Ha,
}

#[derive(Debug, Clone)]
pub struct Config {
    pub server: ServerConfig,
    pub top: TierConfig,
    pub flash: TierConfig,
    pub decider: DeciderConfig,
}

#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
}

#[derive(Debug, Clone)]
pub struct TierConfig {
    /// OpenAI-compatible base URL, without a trailing slash (e.g. `http://host:8000/v1`).
    pub base_url: String,
    pub api_key: Option<String>,
    pub model: String,
    /// Whole-request timeout; `None` means wait forever.
    pub timeout_secs: Option<u64>,
    /// Clamp for the client's requested output tokens.
    pub max_output_tokens: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct DeciderConfig {
    pub url: String,
    pub api_key: Option<String>,
    pub model: String,
    pub timeout_ms: u64,
    /// Route to the top tier when p(top) >= threshold.
    pub threshold: f64,
    pub on_error: OnError,
    /// How long a decision is reused for follow-up steps of the same user turn; 0 disables.
    pub sticky_ttl_secs: u64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawConfig {
    server: Option<RawServer>,
    top: Option<RawTier>,
    flash: Option<RawTier>,
    decider: Option<RawDecider>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawServer {
    host: Option<String>,
    port: Option<u16>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawTier {
    base_url: Option<String>,
    api_key: Option<String>,
    model: Option<String>,
    timeout_secs: Option<u64>,
    max_output_tokens: Option<u64>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawDecider {
    url: Option<String>,
    api_key: Option<String>,
    model: Option<String>,
    timeout_ms: Option<u64>,
    threshold: Option<f64>,
    on_error: Option<OnError>,
    sticky_ttl_secs: Option<u64>,
}

impl Config {
    pub fn load(path: &Path) -> Result<Self, ConfigError> {
        let text = std::fs::read_to_string(path).map_err(|source| ConfigError::Read {
            path: path.display().to_string(),
            source,
        })?;
        Self::parse(&text, &|name| std::env::var(name).ok())
    }

    /// Parse and validate; `env` resolves `{env:VAR}` references.
    pub fn parse(text: &str, env: &dyn Fn(&str) -> Option<String>) -> Result<Self, ConfigError> {
        let raw: RawConfig = toml::from_str(text)?;
        let mut v = Validator {
            env,
            problems: Vec::new(),
        };

        let server = ServerConfig {
            host: raw
                .server
                .as_ref()
                .and_then(|s| s.host.clone())
                .unwrap_or_else(|| "127.0.0.1".into()),
            port: raw.server.as_ref().and_then(|s| s.port).unwrap_or(1970),
        };
        let top = v.tier("top", raw.top);
        let flash = v.tier("flash", raw.flash);
        let decider = v.decider(raw.decider);

        match (top, flash, decider) {
            (Some(top), Some(flash), Some(decider)) if v.problems.is_empty() => Ok(Config {
                server,
                top,
                flash,
                decider,
            }),
            _ => Err(ConfigError::Invalid(v.problems)),
        }
    }
}

struct Validator<'a> {
    env: &'a dyn Fn(&str) -> Option<String>,
    problems: Vec<String>,
}

impl Validator<'_> {
    /// A required string: present, env-expanded, non-empty and not a placeholder.
    fn required(&mut self, key: &str, value: Option<String>) -> Option<String> {
        let Some(value) = value else {
            self.problems.push(format!("{key} is required"));
            return None;
        };
        let value = self.expand(key, &value)?;
        if value.trim().is_empty() || value.contains(PLACEHOLDER) {
            self.problems
                .push(format!("{key} is not set (found {value:?})"));
            return None;
        }
        Some(value)
    }

    fn optional(&mut self, key: &str, value: Option<String>) -> Option<String> {
        let value = self.expand(key, &value?)?;
        (!value.trim().is_empty()).then_some(value)
    }

    fn expand(&mut self, key: &str, value: &str) -> Option<String> {
        match expand_env(value, self.env) {
            Ok(v) => Some(v),
            Err(var) => {
                self.problems.push(format!(
                    "{key} references environment variable {var}, which is not set"
                ));
                None
            }
        }
    }

    fn url(&mut self, key: &str, value: Option<String>) -> Option<String> {
        let value = self.required(key, value)?;
        match reqwest::Url::parse(&value) {
            Ok(u) if matches!(u.scheme(), "http" | "https") && u.has_host() => {
                Some(value.trim_end_matches('/').to_string())
            }
            _ => {
                self.problems
                    .push(format!("{key} is not a valid http(s) URL: {value:?}"));
                None
            }
        }
    }

    fn tier(&mut self, name: &str, raw: Option<RawTier>) -> Option<TierConfig> {
        let Some(raw) = raw else {
            self.problems
                .push(format!("[{name}] section is required (base_url, model)"));
            return None;
        };
        let base_url = self.url(&format!("{name}.base_url"), raw.base_url);
        let model = self.required(&format!("{name}.model"), raw.model);
        let api_key = self.optional(&format!("{name}.api_key"), raw.api_key);
        if raw.max_output_tokens == Some(0) {
            self.problems
                .push(format!("{name}.max_output_tokens must be greater than 0"));
        }
        Some(TierConfig {
            base_url: base_url?,
            api_key,
            model: model?,
            timeout_secs: raw.timeout_secs.filter(|s| *s > 0),
            max_output_tokens: raw.max_output_tokens,
        })
    }

    fn decider(&mut self, raw: Option<RawDecider>) -> Option<DeciderConfig> {
        let Some(raw) = raw else {
            self.problems
                .push("[decider] section is required (url)".to_string());
            return None;
        };
        let url = self.url("decider.url", raw.url);
        let api_key = self.optional("decider.api_key", raw.api_key);
        let model = match raw.model {
            Some(m) => self.required("decider.model", Some(m)),
            None => Some("clef-flash".to_string()),
        };
        let threshold = raw.threshold.unwrap_or(0.5);
        if !(0.0..=1.0).contains(&threshold) {
            self.problems.push(format!(
                "decider.threshold must be within 0..=1, got {threshold}"
            ));
        }
        let timeout_ms = raw.timeout_ms.unwrap_or(3000);
        if timeout_ms == 0 {
            self.problems
                .push("decider.timeout_ms must be greater than 0".to_string());
        }
        Some(DeciderConfig {
            url: url?,
            api_key,
            model: model?,
            timeout_ms,
            threshold,
            on_error: raw.on_error.unwrap_or(OnError::Top),
            sticky_ttl_secs: raw.sticky_ttl_secs.unwrap_or(900),
        })
    }
}

/// Replace every `{env:VAR}` in `value`; `Err` carries the first unset variable.
fn expand_env(value: &str, env: &dyn Fn(&str) -> Option<String>) -> Result<String, String> {
    const OPEN: &str = "{env:";
    let mut out = String::with_capacity(value.len());
    let mut rest = value;
    while let Some(start) = rest.find(OPEN) {
        let after = &rest[start + OPEN.len()..];
        let Some(end) = after.find('}') else { break };
        let name = after[..end].trim();
        out.push_str(&rest[..start]);
        out.push_str(&env(name).ok_or_else(|| name.to_string())?);
        rest = &after[end + 1..];
    }
    out.push_str(rest);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    const FULL: &str = r#"
        [top]
        base_url = "https://gw.example.com/v1/"
        api_key = "{env:GW_KEY}"
        model = "big"

        [flash]
        base_url = "http://192.168.1.209:8000/v1"
        model = "Qwen/Qwen3.8-Flash-Next"

        [decider]
        url = "http://127.0.0.1:8000/v1/decisions"
    "#;

    fn env(name: &str) -> Option<String> {
        (name == "GW_KEY").then(|| "secret".to_string())
    }

    fn problems(text: &str) -> Vec<String> {
        match Config::parse(text, &env) {
            Err(ConfigError::Invalid(p)) => p,
            other => panic!("expected Invalid, got {other:?}"),
        }
    }

    #[test]
    fn parses_with_defaults() {
        let cfg = Config::parse(FULL, &env).unwrap();
        assert_eq!(
            (cfg.server.host.as_str(), cfg.server.port),
            ("127.0.0.1", 1970)
        );
        assert_eq!(cfg.top.base_url, "https://gw.example.com/v1");
        assert_eq!(cfg.top.api_key.as_deref(), Some("secret"));
        assert_eq!(cfg.flash.api_key, None);
        assert_eq!(cfg.flash.timeout_secs, None);
        assert_eq!(cfg.decider.model, "clef-flash");
        assert_eq!(cfg.decider.threshold, 0.5);
        assert_eq!(cfg.decider.on_error, OnError::Top);
        assert_eq!(cfg.decider.sticky_ttl_secs, 900);
    }

    #[test]
    fn reports_every_missing_piece() {
        let p = problems("[top]\nmodel = \"big\"\n");
        assert!(
            p.iter().any(|m| m.contains("top.base_url is required")),
            "{p:?}"
        );
        assert!(
            p.iter().any(|m| m.contains("[flash] section is required")),
            "{p:?}"
        );
        assert!(
            p.iter()
                .any(|m| m.contains("[decider] section is required")),
            "{p:?}"
        );
    }

    #[test]
    fn rejects_placeholders_and_bad_urls() {
        let text = FULL
            .replace("model = \"big\"", "model = \"REPLACE_ME\"")
            .replace("http://127.0.0.1:8000/v1/decisions", "localhost:8000");
        let p = problems(&text);
        assert!(
            p.iter().any(|m| m.contains("top.model is not set")),
            "{p:?}"
        );
        assert!(
            p.iter().any(|m| m.contains("decider.url is not a valid")),
            "{p:?}"
        );
    }

    #[test]
    fn rejects_unset_env_var() {
        let p = problems(&FULL.replace("GW_KEY", "MISSING_KEY"));
        assert!(p.iter().any(|m| m.contains("MISSING_KEY")), "{p:?}");
    }

    #[test]
    fn rejects_unknown_keys() {
        let text = format!("{FULL}\nsticky = 1\n");
        assert!(matches!(
            Config::parse(&text, &env),
            Err(ConfigError::Parse(_))
        ));
    }
}
