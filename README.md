# s1-irrlab

A local proxy for coding agents that sends every LLM request to either a **top** model or a
**flash** model. A [System One](https://huggingface.co/Cloudflare/clef-flash) decision model
(clef-flash, Jev) judges whether the agent's next turn is routine coding work or demanding
engineering, and that call is made **before** the request is forwarded.

Agents talk to one endpoint, `http://127.0.0.1:1970`, in either dialect:

| Route | For |
|---|---|
| `POST /v1/chat/completions` | OpenAI-compatible agents (opencode, AI SDK, ...) |
| `POST /v1/messages` | Anthropic-compatible agents (Claude Code, Anthropic SDKs) |
| `POST /v1/messages/count_tokens` | token count for Anthropic clients (see [Limits](#limits)) |
| `GET /v1/models`, `GET /health` | model list, config summary (keys redacted) |

```text
client ─► :1970 ─► build state ─► POST decider /v1/decisions ─► p(top) >= threshold ?
                                                                  ├─ yes ─► top tier
                                                                  └─ no  ─► flash tier
```

Both tiers are OpenAI chat-completions upstreams. OpenAI requests are forwarded untouched apart
from the `model` name; Anthropic requests are translated to chat completions and the reply
(plain or streamed, including tool calls and reasoning) is translated back.

## Quick start

```sh
# 1. start the decider in another terminal; it must answer on decider.url
(cd ../clef-flash && make run)

# 2. configure and start the proxy
cp s1.example.toml s1.toml     # then fill in [top], [flash] and [decider]
make run                       # or: make build && dist/s1 --config s1.toml
curl -s http://127.0.0.1:1970/health

# 3. watch it route
make sample
```

The proxy refuses to start until both tiers and the decider are configured, and lists
everything that is missing.

It does **not** check that the decider is reachable. With the decider down the proxy keeps
working, but every request goes to the `decider.on_error` tier (top by default) and responses
carry `src=fallback`.

## Trying it: `make sample`

With the proxy running, `make sample` sends two chat-completions requests, a routine coding
task and a demanding one, and prints where each one went. It needs `curl` (7.84 or newer)
and `jq`.

````text
────────────────────────────────────────────────────────────
prompt    Write a one-line Python function that returns the square of a number.
tier      flash
decision  p=0.1000;src=decider
model     Qwen/Qwen3.8-Flash-Next
http      200
tokens    66 in / 100 out

```python
square = lambda x: x * x
```

────────────────────────────────────────────────────────────
prompt    Our Rust HTTP service intermittently deadlocks under load since we added a connection pool ...
tier      top
decision  p=0.9000;src=decider
...
````

| What you see | Meaning |
|---|---|
| `src=decider` | the decider was asked for this request |
| `src=cache` | same prompt as a recent run; the sticky decision was reused |
| `src=fallback` | the decider was unreachable or unusable, so `decider.on_error` chose the tier |
| `failover_from=top` | with `on_error = "ha"`: that tier's model failed and the other one answered |
| `http 000`, empty tier | the proxy itself is not reachable |
| `(no answer; finish_reason=length)` | the model spent its whole token budget before answering |

Both requests are real and are billed by whichever upstream answers. Use
`make sample URL=http://host:port` to target a proxy on another address.

## Configuration

`--config <path>`, else `$S1_CONFIG`, else `./s1.toml`. Strings may use `{env:VAR}`.

| Key | Required | Default | Meaning |
|---|---|---|---|
| `server.host` / `server.port` | no | `127.0.0.1` / `1970` | listen address |
| `top.base_url`, `flash.base_url` | **yes** | | OpenAI-compatible base URL (`.../v1`) |
| `top.model`, `flash.model` | **yes** | | model name sent to that upstream |
| `*.api_key` | no | none | sent as `Authorization: Bearer`; omitted = no header |
| `*.timeout_secs` | no | none | whole-request timeout; omitted = wait forever |
| `*.max_output_tokens` | no | none | clamps the client's `max_tokens` |
| `decider.url` | **yes** | | System One `/v1/decisions` endpoint |
| `decider.api_key` | no | none | bearer token for a hosted decider |
| `decider.model` | no | `clef-flash` | model name sent to the decider |
| `decider.timeout_ms` | no | `3000` | budget for the decision |
| `decider.threshold` | no | `0.5` | route to top when p(top) >= threshold |
| `decider.on_error` | no | `top` | when the decider is unusable: `top`, `flash`, `fail` (503), or `ha` (see [High availability](#high-availability)) |
| `decider.sticky_ttl_secs` | no | `900` | reuse a decision within one user turn; `0` = always ask |

The client's own `model` name and API key are ignored: any model name works, and only the
chosen tier's configured key is ever sent upstream.

### High availability

`on_error = "ha"` keeps requests succeeding when either the decider or one model is down:

- **Decider unusable:** the request goes to the top tier, as with `on_error = "top"`.
- **Chosen model fails:** the request is sent again to the other tier, whichever way the
  decision went, and the response says so:

  ```text
  x-s1-tier: flash
  x-s1-decision: p=0.9000;src=decider;failover_from=top
  ```

A model counts as failed when it cannot be reached or times out, or answers with a 5xx, 401,
403, 404, 408 or 429. Other 4xx replies mean the request itself was rejected, so they are
returned as they are. If the second tier fails too, its error is the one the client sees.

Things to know:

- Failover happens before the first byte of the reply. A stream that breaks midway is not
  restarted on the other tier.
- The sticky decision is not changed by a failover, so while a model stays down every request
  routed to it pays for one failed attempt first. Set that tier's `timeout_secs` if it tends
  to hang rather than refuse.
- A failover from flash to top spends top-tier money on work judged routine, and one from top
  to flash gives demanding work to the weaker model.

### Choosing a threshold

The decider splits probability between `flash` and `top`, so `0.5` follows whichever it finds
more likely. Moving the threshold only changes what happens to prompts it is unsure about.

| Threshold | Effect | Use when |
|---|---|---|
| `0.3`–`0.4` | unsure prompts go to top | quality matters more than cost |
| `0.5` | follows the decider's own pick | default |
| `0.6`–`0.7` | only clearly demanding prompts go to top | top-tier spend is the main concern |

The two mistakes are not equal: an easy task sent to top costs some money, while a hard task
sent to flash can fail a whole agent run, because the decision sticks for that turn's entire
tool loop. To tune it, run real work and read the `p_top` values in the `routed` log lines.
If they cluster near 0 and 1 the threshold barely matters; if many sit in the middle, check
where those prompts should have gone and move it that way.

## How the decision is made

The decider gets one `choice` question and a compact `state`. The question and its two
options are written for coding agents (`src/decider.rs`):

| Option | Described to the decider as |
|---|---|
| `flash` | routine coding work: reading or searching files, running a command, small clearly specified edits, renames, formatting, lint and typo fixes, boilerplate, simple tests, commit messages, short explanations, mechanical follow-ups |
| `top` | demanding engineering: multi-file features, architecture or API design, debugging an unknown root cause, large refactors or migrations, concurrency, performance or security work, code review, long autonomous multi-step tasks, vague requirements |

The state:

```jsonc
{
  "latest_user_request": "...",        // last text the user wrote, <system-reminder> blocks removed
  "recent_context": ["assistant: ..."], // up to 6 other recent turns, clipped
  "system_prompt_excerpt": "...",
  "turns": 14, "tools_available": 23, "has_images": false,
  "last_turn_is_tool_result": true
}
```

**Sticky decisions.** In an agent's tool loop most requests end in a tool result, not a new
prompt. Re-deciding each step would flip models mid-task, so the decision for a user turn is
cached (keyed by system prompt, first and latest user message) and reused until the user
writes something new. Set `sticky_ttl_secs = 0` to consult the decider on every request.

The cache lives in memory only: it is empty after a restart, and an entry expires
`sticky_ttl_secs` after it was decided even if the tool loop is still running, at which point
the decider is asked again for the same turn.

Every response says what happened:

```text
x-s1-tier: top
x-s1-decision: p=0.8312;src=decider      # src = decider | cache | fallback; ;failover_from=<tier> under ha
```

and the proxy logs one line per request (protocol, tier, p(top), source, timings, status).
Set `RUST_LOG=s1_irrlab=debug` to also log the state sent to the decider and its raw answer.

## Pointing agents at it

opencode:

```json
"s1": {
  "npm": "@ai-sdk/openai-compatible",
  "name": "s1 router",
  "options": { "baseURL": "http://127.0.0.1:1970/v1", "timeout": false },
  "models": { "s1-auto": { "name": "s1 auto", "tool_call": true, "attachment": true } }
}
```

Claude Code:

```sh
ANTHROPIC_BASE_URL=http://127.0.0.1:1970 ANTHROPIC_API_KEY=unused claude
```

Whatever model the agent selects is ignored. Claude Code also sends background requests under
a small/fast model name (titles, summaries); those are routed by the decider like any other
request, not pinned to the flash tier.

## Limits

- Anthropic features with no chat-completions equivalent are dropped: extended-thinking
  settings, `cache_control`, server tools (web search, code execution), documents.
- Reasoning from the upstream is returned to Anthropic clients as `thinking` blocks, but those
  blocks are discarded when the client sends the conversation back.
- The decider never sees image data, only a `has_images` flag, so a request is not judged by
  what its images contain.
- If a client asks for more output tokens than a model allows (Claude Code does), the
  upstream rejects the request; set that tier's `max_output_tokens` to clamp it.
- `count_tokens` is a local estimate (characters / 4), not a tokenizer count.
- The OpenAI Responses API (`/v1/responses`), embeddings and other endpoints are not proxied.
- No inbound authentication; keep it bound to loopback.

## Development

```sh
make build    # release binary → dist/s1
make run      # run with CONFIG=s1.toml
make sample   # two live requests through a running proxy (curl + jq)
make test     # unit tests + end-to-end tests against mock decider and upstreams
make vet      # clippy -D warnings, cargo fmt --check
make install  # copy dist/s1 to ~/.local/bin
```

| File | Purpose |
|---|---|
| `src/config.rs` | TOML loading, `{env:VAR}` expansion, validation |
| `src/decider.rs` | decision state, `/v1/decisions` client, threshold, sticky cache |
| `src/upstream.rs` | the two tiers and their HTTP clients |
| `src/server.rs` | routes: decide, then forward |
| `src/openai.rs` | chat-completions passthrough |
| `src/anthropic/` | Messages ↔ chat-completions translation (request, response, stream) |
| `tests/proxy.rs` | end-to-end tests |
| `s1.example.toml` | config template with example providers to paste under a tier |
| `Makefile` | build, run, sample, test, vet, install |
