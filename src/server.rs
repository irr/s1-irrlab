//! HTTP surface: decide first, then forward to the chosen tier.

use std::{sync::Arc, time::Instant};

use axum::{
    body::{Body, Bytes},
    extract::{DefaultBodyLimit, State},
    http::{header, HeaderName, HeaderValue, StatusCode, Uri},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde_json::{json, Value};
use tracing::info;

use crate::{
    anthropic,
    config::{Config, OnError},
    decider::{Conversation, Decider, Decision},
    error::{upstream_message, ApiError, Protocol},
    openai,
    upstream::{Tier, Upstream},
};

const MAX_BODY_BYTES: usize = 64 * 1024 * 1024;
/// Listed by `/v1/models` for clients that insist on picking a model; the
/// requested model name never affects routing.
const VIRTUAL_MODEL: &str = "s1-auto";
/// Response headers that describe the upstream connection, not the payload.
const HOP_BY_HOP: [&str; 5] = [
    "connection",
    "content-length",
    "keep-alive",
    "transfer-encoding",
    "upgrade",
];

pub struct AppState {
    cfg: Config,
    decider: Decider,
    top: Upstream,
    flash: Upstream,
}

impl AppState {
    fn upstream(&self, tier: Tier) -> &Upstream {
        match tier {
            Tier::Top => &self.top,
            Tier::Flash => &self.flash,
        }
    }
}

pub fn router(cfg: Config) -> anyhow::Result<Router> {
    let state = Arc::new(AppState {
        decider: Decider::new(cfg.decider.clone())?,
        top: Upstream::new(&cfg.top)?,
        flash: Upstream::new(&cfg.flash)?,
        cfg,
    });
    Ok(Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/messages", post(messages))
        .route("/v1/messages/count_tokens", post(count_tokens))
        .fallback(not_found)
        .layer(DefaultBodyLimit::max(MAX_BODY_BYTES))
        .with_state(state))
}

async fn chat_completions(State(state): State<Arc<AppState>>, body: Bytes) -> Response {
    let protocol = Protocol::OpenAi;
    let started = Instant::now();
    let request = match parse_object(protocol, &body) {
        Ok(request) => request,
        Err(err) => return err.into_response(),
    };
    let mut decision = match decide(&state, protocol, &openai::conversation(&request)).await {
        Ok(decision) => decision,
        Err(err) => return err.into_response(),
    };
    let build = |upstream: &Upstream| {
        let mut body = request.clone();
        openai::prepare(&mut body, upstream);
        Ok(body)
    };
    let (reply, _) = match dispatch(&state, protocol, &mut decision, &build).await {
        Ok(sent) => sent,
        Err(response) => return *response,
    };
    log(protocol, &decision, reply.status(), started);

    // Success or error, the upstream already speaks the caller's protocol.
    let mut builder = Response::builder().status(reply.status());
    for (name, value) in reply.headers() {
        if !HOP_BY_HOP.contains(&name.as_str()) {
            builder = builder.header(name, value);
        }
    }
    let response = builder
        .body(Body::from_stream(reply.bytes_stream()))
        .expect("upstream status and headers are valid");
    stamp(response, &decision)
}

async fn messages(State(state): State<Arc<AppState>>, body: Bytes) -> Response {
    let protocol = Protocol::Anthropic;
    let started = Instant::now();
    let request = match parse_object(protocol, &body) {
        Ok(request) => request,
        Err(err) => return err.into_response(),
    };
    let mut decision = match decide(&state, protocol, &anthropic::conversation(&request)).await {
        Ok(decision) => decision,
        Err(err) => return err.into_response(),
    };
    let build = |upstream: &Upstream| {
        anthropic::request::to_chat(&request, &upstream.model, upstream.max_output_tokens)
    };
    let (reply, chat) = match dispatch(&state, protocol, &mut decision, &build).await {
        Ok(sent) => sent,
        Err(response) => return *response,
    };
    let upstream = state.upstream(decision.tier);
    let status = reply.status();
    log(protocol, &decision, status, started);

    let response = if !status.is_success() {
        let text = reply.text().await.unwrap_or_default();
        ApiError::new(protocol, status, upstream_message(&text)).into_response()
    } else if chat.get("stream").and_then(Value::as_bool) == Some(true) {
        let events = anthropic::stream::translate(reply, upstream.model.clone());
        (
            [
                (header::CONTENT_TYPE, "text/event-stream"),
                (header::CACHE_CONTROL, "no-cache"),
            ],
            Body::from_stream(events),
        )
            .into_response()
    } else {
        match reply.json::<Value>().await {
            Ok(completion) => {
                Json(anthropic::response::from_chat(&completion, &upstream.model)).into_response()
            }
            Err(err) => ApiError::new(
                protocol,
                StatusCode::BAD_GATEWAY,
                format!(
                    "{} tier returned an unreadable response: {err}",
                    decision.tier.as_str()
                ),
            )
            .into_response(),
        }
    };
    stamp(response, &decision)
}

async fn count_tokens(body: Bytes) -> Response {
    match parse_object(Protocol::Anthropic, &body) {
        Ok(request) => {
            Json(json!({"input_tokens": anthropic::estimate_tokens(&request)})).into_response()
        }
        Err(err) => err.into_response(),
    }
}

async fn models(State(state): State<Arc<AppState>>) -> Json<Value> {
    let mut ids = vec![VIRTUAL_MODEL, &state.top.model, &state.flash.model];
    ids.dedup();
    // Each entry carries both the OpenAI and the Anthropic field names.
    let data: Vec<Value> = ids
        .iter()
        .map(|id| {
            json!({
                "id": id,
                "object": "model",
                "type": "model",
                "display_name": id,
                "created": 0,
                "created_at": "1970-01-01T00:00:00Z",
                "owned_by": "s1-irrlab",
            })
        })
        .collect();
    Json(json!({"object": "list", "data": data, "has_more": false}))
}

async fn health(State(state): State<Arc<AppState>>) -> Json<Value> {
    let tier = |cfg: &crate::config::TierConfig| json!({"base_url": cfg.base_url, "model": cfg.model, "api_key": cfg.api_key.is_some()});
    let cfg = &state.cfg;
    Json(json!({
        "status": "ok",
        "top": tier(&cfg.top),
        "flash": tier(&cfg.flash),
        "decider": {
            "url": cfg.decider.url,
            "model": cfg.decider.model,
            "threshold": cfg.decider.threshold,
            "timeout_ms": cfg.decider.timeout_ms,
            "sticky_ttl_secs": cfg.decider.sticky_ttl_secs,
        },
    }))
}

async fn not_found(uri: Uri) -> Response {
    let protocol = if uri.path().starts_with("/v1/messages") {
        Protocol::Anthropic
    } else {
        Protocol::OpenAi
    };
    ApiError::new(
        protocol,
        StatusCode::NOT_FOUND,
        format!("no route for {}", uri.path()),
    )
    .into_response()
}

fn parse_object(protocol: Protocol, body: &[u8]) -> Result<Value, ApiError> {
    match serde_json::from_slice::<Value>(body) {
        Ok(value) if value.is_object() => Ok(value),
        Ok(_) => Err(ApiError::new(
            protocol,
            StatusCode::BAD_REQUEST,
            "request body must be a JSON object",
        )),
        Err(err) => Err(ApiError::new(
            protocol,
            StatusCode::BAD_REQUEST,
            format!("invalid JSON body: {err}"),
        )),
    }
}

/// The decision is always resolved (decider, cache or fallback) before any upstream call.
async fn decide(
    state: &AppState,
    protocol: Protocol,
    conv: &Conversation,
) -> Result<Decision, ApiError> {
    state.decider.decide(conv).await.map_err(|err| {
        ApiError::new(
            protocol,
            StatusCode::SERVICE_UNAVAILABLE,
            format!("routing decision failed: {err}"),
        )
    })
}

/// One try at one tier: the reply (with the body that was sent) or a transport failure.
enum Attempt {
    Reply(reqwest::Response, Value),
    Unreachable(reqwest::Error),
}

impl Attempt {
    /// Failures that say the model is unavailable, not that the request is
    /// wrong: the other tier may well succeed.
    fn model_failed(&self) -> Option<String> {
        match self {
            Attempt::Unreachable(err) => Some(err.to_string()),
            Attempt::Reply(reply, _) => {
                let status = reply.status();
                let unavailable = status.is_server_error()
                    || matches!(status.as_u16(), 401 | 403 | 404 | 408 | 429);
                unavailable.then(|| format!("HTTP {status}"))
            }
        }
    }
}

/// `build` makes the chat-completions body for a tier; `Err` is the client's fault.
type Build<'a> = &'a (dyn Fn(&Upstream) -> Result<Value, String> + Sync);

async fn attempt(upstream: &Upstream, build: Build<'_>) -> Result<Attempt, String> {
    let body = build(upstream)?;
    Ok(match upstream.send(&body).await {
        Ok(reply) => Attempt::Reply(reply, body),
        Err(err) => Attempt::Unreachable(err),
    })
}

/// Send to the decided tier. With `on_error = "ha"`, a failing model is
/// replaced by the other tier, and `decision` is updated to say so.
async fn dispatch(
    state: &AppState,
    protocol: Protocol,
    decision: &mut Decision,
    build: Build<'_>,
) -> Result<(reqwest::Response, Value), Box<Response>> {
    let bad_request = |message| {
        Box::new(ApiError::new(protocol, StatusCode::BAD_REQUEST, message).into_response())
    };
    let mut result = attempt(state.upstream(decision.tier), build)
        .await
        .map_err(bad_request)?;
    if state.cfg.decider.on_error == OnError::Ha {
        if let Some(reason) = result.model_failed() {
            let failed = decision.tier;
            tracing::warn!(
                failed = failed.as_str(),
                failover = failed.other().as_str(),
                %reason,
                "model failed, failing over"
            );
            decision.tier = failed.other();
            decision.failed_over_from = Some(failed);
            result = attempt(state.upstream(decision.tier), build)
                .await
                .map_err(bad_request)?;
        }
    }
    match result {
        Attempt::Reply(reply, body) => Ok((reply, body)),
        Attempt::Unreachable(err) => Err(Box::new(bad_gateway(protocol, decision, err))),
    }
}

fn bad_gateway(protocol: Protocol, decision: &Decision, err: reqwest::Error) -> Response {
    let tier = decision.tier.as_str();
    tracing::warn!(protocol = protocol.as_str(), tier, error = %err, "upstream unreachable");
    let response = ApiError::new(
        protocol,
        StatusCode::BAD_GATEWAY,
        format!("{tier} tier request failed: {err}"),
    );
    stamp(response.into_response(), decision)
}

fn log(protocol: Protocol, decision: &Decision, status: StatusCode, started: Instant) {
    info!(
        protocol = protocol.as_str(),
        tier = decision.tier.as_str(),
        p_top = decision.p_top,
        source = decision.source.as_str(),
        failover_from = decision.failed_over_from.map(Tier::as_str),
        decider_ms = decision.elapsed.as_millis() as u64,
        status = status.as_u16(),
        total_ms = started.elapsed().as_millis() as u64,
        "routed"
    );
}

/// Tell the caller which tier answered and why.
fn stamp(mut response: Response, decision: &Decision) -> Response {
    let headers = response.headers_mut();
    headers.insert(
        HeaderName::from_static("x-s1-tier"),
        HeaderValue::from_static(decision.tier.as_str()),
    );
    if let Ok(value) = HeaderValue::from_str(&decision.header()) {
        headers.insert(HeaderName::from_static("x-s1-decision"), value);
    }
    response
}
