//! Errors shaped for whichever API the caller is speaking.

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::{json, Value};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Protocol {
    OpenAi,
    Anthropic,
}

impl Protocol {
    pub fn as_str(self) -> &'static str {
        match self {
            Protocol::OpenAi => "openai",
            Protocol::Anthropic => "anthropic",
        }
    }
}

#[derive(Debug)]
pub struct ApiError {
    pub protocol: Protocol,
    pub status: StatusCode,
    pub message: String,
}

impl ApiError {
    pub fn new(protocol: Protocol, status: StatusCode, message: impl Into<String>) -> Self {
        Self {
            protocol,
            status,
            message: message.into(),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let body = match self.protocol {
            Protocol::OpenAi => json!({
                "error": {
                    "message": self.message,
                    "type": if self.status.is_client_error() { "invalid_request_error" } else { "api_error" },
                    "param": null,
                    "code": null,
                }
            }),
            Protocol::Anthropic => anthropic_error(self.status, &self.message),
        };
        (self.status, Json(body)).into_response()
    }
}

pub fn anthropic_error(status: StatusCode, message: &str) -> Value {
    let kind = match status.as_u16() {
        400 | 422 => "invalid_request_error",
        401 => "authentication_error",
        403 => "permission_error",
        404 => "not_found_error",
        413 => "request_too_large",
        429 => "rate_limit_error",
        503 | 529 => "overloaded_error",
        _ => "api_error",
    };
    json!({"type": "error", "error": {"type": kind, "message": message}})
}

/// Pull a human-readable message out of an upstream error body.
pub fn upstream_message(body: &str) -> String {
    let parsed = serde_json::from_str::<Value>(body).ok();
    let found = parsed.as_ref().and_then(|v| {
        v.pointer("/error/message")
            .or_else(|| v.get("error"))
            .or_else(|| v.get("message"))
            .or_else(|| v.get("detail"))
            .and_then(Value::as_str)
    });
    let message = found.unwrap_or(body).trim();
    if message.is_empty() {
        "upstream returned an error".to_string()
    } else {
        message.chars().take(2000).collect()
    }
}
