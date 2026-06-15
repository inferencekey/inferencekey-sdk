//! Data-plane pipelines: OpenAI-compatible inference over [`HttpPort`].
//!
//! These run against the *data plane* (`ik_live_` keys) under
//! `/endpoint/:project_slug/:workload_slug/v1/...`. Everything here is split
//! into PURE builders/parsers (no IO, unit-tested below) and three thin EFFECT
//! entry points that wire those steps to the transport port via the canonical
//! [`HttpPort`] API: [`HttpPort::request_json`] for unary calls and
//! [`HttpPort::stream_sse`] for streaming, each taking a fully-built
//! [`HttpRequest`].
//!
//! Pure steps:
//! - [`build_endpoint_url`] — join the data-plane path.
//! - [`resolve_messages`] — `prompt` XOR `messages`, validated non-empty.
//! - [`build_chat_body`] / [`build_embed_body`] — request JSON.
//! - [`parse_chat_result`] / [`parse_chunk`] / [`parse_embed_result`] — decode
//!   opaque [`serde_json::Value`] into typed results via `.get()` + `ok_or_else`.

use futures_util::stream::{BoxStream, StreamExt};
use serde_json::{json, Map, Value};

use crate::errors::{CoreError, CoreResult};
use crate::ports::http::{HttpMethod, HttpPort, HttpRequest};

// ---------------------------------------------------------------------------
// Public parameter / result types
// ---------------------------------------------------------------------------

/// A single chat turn. `role` is one of `system` / `user` / `assistant`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// Inputs for [`generate_text`] / [`generate_text_stream`].
///
/// Exactly one of `prompt` or `messages` must be supplied (see
/// [`resolve_messages`]); a bare `prompt` is lifted into a single `user` turn.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GenerateTextParams {
    pub prompt: Option<String>,
    pub messages: Option<Vec<ChatMessage>>,
    pub temperature: Option<f64>,
    pub max_tokens: Option<u32>,
}

/// A completed (non-streamed) chat result.
#[derive(Debug, Clone, PartialEq)]
pub struct TextResult {
    pub text: String,
    pub model: String,
    pub finish_reason: Option<String>,
    /// The full upstream response, untouched.
    pub raw: Value,
}

/// One streamed chunk (`chat.completion.chunk`).
#[derive(Debug, Clone, PartialEq)]
pub struct TextChunk {
    pub text: String,
    pub finish_reason: Option<String>,
    /// The full chunk JSON, untouched.
    pub raw: Value,
}

/// Inputs for [`embed`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EmbedParams {
    pub input: Vec<String>,
}

/// An embeddings result, one vector per input in request order.
#[derive(Debug, Clone, PartialEq)]
pub struct EmbedResult {
    pub embeddings: Vec<Vec<f64>>,
    pub model: String,
    /// The full upstream response, untouched.
    pub raw: Value,
}

// ---------------------------------------------------------------------------
// EFFECT: public entry points
// ---------------------------------------------------------------------------

/// Run a non-streaming chat completion and decode the result.
pub async fn generate_text(
    http: &dyn HttpPort,
    base_url: &str,
    project_slug: &str,
    workload_slug: &str,
    api_key: &str,
    params: GenerateTextParams,
) -> CoreResult<TextResult> {
    let url = build_endpoint_url(base_url, project_slug, workload_slug, "chat/completions");
    let body = build_chat_body(workload_slug, &params, false)?;
    let req = HttpRequest::with_body(HttpMethod::Post, url, api_key, body);
    let value = http.request_json(req).await?;
    parse_chat_result(&value)
}

/// Run a streaming chat completion, yielding one [`TextChunk`] per SSE chunk.
///
/// The terminal `data: [DONE]` sentinel and empty keep-alive lines are dropped
/// by the port; this maps every remaining data frame through [`parse_chunk`].
pub async fn generate_text_stream(
    http: &dyn HttpPort,
    base_url: &str,
    project_slug: &str,
    workload_slug: &str,
    api_key: &str,
    params: GenerateTextParams,
) -> CoreResult<BoxStream<'static, CoreResult<TextChunk>>> {
    let url = build_endpoint_url(base_url, project_slug, workload_slug, "chat/completions");
    let body = build_chat_body(workload_slug, &params, true)?;
    let req = HttpRequest::with_body(HttpMethod::Post, url, api_key, body);
    let frames = http.stream_sse(req).await?;
    let chunks = frames.map(|frame| frame.and_then(|value| parse_chunk(&value)));
    Ok(chunks.boxed())
}

/// Run an embeddings request and decode the result.
pub async fn embed(
    http: &dyn HttpPort,
    base_url: &str,
    project_slug: &str,
    workload_slug: &str,
    api_key: &str,
    params: EmbedParams,
) -> CoreResult<EmbedResult> {
    let url = build_endpoint_url(base_url, project_slug, workload_slug, "embeddings");
    let body = build_embed_body(workload_slug, &params)?;
    let req = HttpRequest::with_body(HttpMethod::Post, url, api_key, body);
    let value = http.request_json(req).await?;
    parse_embed_result(&value)
}

// ---------------------------------------------------------------------------
// PURE: url + request builders
// ---------------------------------------------------------------------------

/// Join the data-plane endpoint URL:
/// `{base}/endpoint/{project}/{workload}/v1/{path}` with no doubled slashes.
fn build_endpoint_url(base_url: &str, project_slug: &str, workload_slug: &str, path: &str) -> String {
    let base = base_url.trim_end_matches('/');
    let path = path.trim_start_matches('/');
    format!("{base}/endpoint/{project_slug}/{workload_slug}/v1/{path}")
}

/// Resolve `prompt` XOR `messages` into the chat message list.
///
/// - both set or neither set → [`CoreError::Validation`];
/// - empty `messages`, or a blank `prompt`, or any blank message content →
///   [`CoreError::Validation`].
fn resolve_messages(params: &GenerateTextParams) -> CoreResult<Vec<ChatMessage>> {
    match (&params.prompt, &params.messages) {
        (Some(_), Some(_)) => Err(invalid("provide either prompt or messages, not both")),
        (None, None) => Err(invalid("provide a prompt or messages")),
        (Some(prompt), None) => resolve_prompt(prompt),
        (None, Some(messages)) => resolve_message_list(messages),
    }
}

/// Lift a single non-blank `prompt` into one `user` turn.
fn resolve_prompt(prompt: &str) -> CoreResult<Vec<ChatMessage>> {
    match prompt.trim().is_empty() {
        true => Err(invalid("prompt must not be empty")),
        false => Ok(vec![ChatMessage {
            role: "user".to_string(),
            content: prompt.to_string(),
        }]),
    }
}

/// Validate an explicit message list: non-empty, no blank content.
fn resolve_message_list(messages: &[ChatMessage]) -> CoreResult<Vec<ChatMessage>> {
    match messages.is_empty() {
        true => Err(invalid("messages must not be empty")),
        false => validate_message_contents(messages).map(|_| messages.to_vec()),
    }
}

/// Reject any message with blank `content`.
fn validate_message_contents(messages: &[ChatMessage]) -> CoreResult<()> {
    match messages.iter().any(|m| m.content.trim().is_empty()) {
        true => Err(invalid("message content must not be empty")),
        false => Ok(()),
    }
}

/// Build the `/chat/completions` request body.
fn build_chat_body(workload_slug: &str, params: &GenerateTextParams, stream: bool) -> CoreResult<Value> {
    let messages = resolve_messages(params)?;
    let mut body = Map::new();
    body.insert("model".to_string(), json!(resolve_model(workload_slug)));
    body.insert("messages".to_string(), encode_messages(&messages));
    body.insert("stream".to_string(), Value::Bool(stream));
    insert_opt(&mut body, "temperature", params.temperature.map(Value::from));
    insert_opt(&mut body, "max_tokens", params.max_tokens.map(Value::from));
    Ok(Value::Object(body))
}

/// Build the `/embeddings` request body.
fn build_embed_body(workload_slug: &str, params: &EmbedParams) -> CoreResult<Value> {
    match params.input.is_empty() {
        true => Err(invalid("input must not be empty")),
        false => Ok(json!({
            "model": resolve_model(workload_slug),
            "input": params.input,
        })),
    }
}

/// The OpenAI `model` field: the workload slug, or `"default"` when unnamed.
fn resolve_model(workload_slug: &str) -> &str {
    match workload_slug.trim().is_empty() {
        true => "default",
        false => workload_slug,
    }
}

/// Encode chat turns into the wire array.
fn encode_messages(messages: &[ChatMessage]) -> Value {
    let turns: Vec<Value> = messages
        .iter()
        .map(|m| json!({ "role": m.role, "content": m.content }))
        .collect();
    Value::Array(turns)
}

/// Insert a key only when its value is present (keeps bodies minimal).
fn insert_opt(body: &mut Map<String, Value>, key: &str, value: Option<Value>) {
    if let Some(value) = value {
        body.insert(key.to_string(), value);
    }
}

// ---------------------------------------------------------------------------
// PURE: response parsers
// ---------------------------------------------------------------------------

/// Decode a `chat.completion` response into a [`TextResult`].
fn parse_chat_result(value: &Value) -> CoreResult<TextResult> {
    let choice = first_choice(value)?;
    let text = require_str(choice, &["message", "content"], "choices[0].message.content")?;
    let model = require_str(value, &["model"], "model")?;
    Ok(TextResult {
        text,
        model,
        finish_reason: optional_str(choice, "finish_reason"),
        raw: value.clone(),
    })
}

/// Decode one `chat.completion.chunk` into a [`TextChunk`].
///
/// A chunk's delta may legitimately omit `content` (role-only or terminal
/// frames), so missing/non-string deltas decode to empty text rather than error.
fn parse_chunk(value: &Value) -> CoreResult<TextChunk> {
    let choice = first_choice(value)?;
    let text = choice
        .get("delta")
        .and_then(|d| d.get("content"))
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    Ok(TextChunk {
        text,
        finish_reason: optional_str(choice, "finish_reason"),
        raw: value.clone(),
    })
}

/// Decode an embeddings response into an [`EmbedResult`].
fn parse_embed_result(value: &Value) -> CoreResult<EmbedResult> {
    let data = value
        .get("data")
        .and_then(Value::as_array)
        .ok_or_else(|| drift("data"))?;
    let embeddings = data.iter().map(parse_embedding).collect::<CoreResult<_>>()?;
    let model = require_str(value, &["model"], "model")?;
    Ok(EmbedResult {
        embeddings,
        model,
        raw: value.clone(),
    })
}

/// Decode one `data[i].embedding` array of floats.
fn parse_embedding(item: &Value) -> CoreResult<Vec<f64>> {
    item.get("embedding")
        .and_then(Value::as_array)
        .ok_or_else(|| drift("data[].embedding"))?
        .iter()
        .map(|n| n.as_f64().ok_or_else(|| drift("data[].embedding[]")))
        .collect()
}

/// Borrow `choices[0]` or report drift naming the missing field.
fn first_choice(value: &Value) -> CoreResult<&Value> {
    value
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .ok_or_else(|| drift("choices[0]"))
}

/// Read a required string at a nested key path, reporting `label` on drift.
fn require_str(value: &Value, path: &[&str], label: &str) -> CoreResult<String> {
    path.iter()
        .try_fold(value, |node, key| node.get(*key))
        .and_then(Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| drift(label))
}

/// Read an optional string field, `None` when absent, null, or non-string.
fn optional_str(value: &Value, key: &str) -> Option<String> {
    value.get(key).and_then(Value::as_str).map(str::to_string)
}

// ---------------------------------------------------------------------------
// Error helpers
// ---------------------------------------------------------------------------

fn invalid(message: &str) -> CoreError {
    CoreError::Validation(message.to_string())
}

/// An upstream body that omitted a field we contractually require.
fn drift(field: &str) -> CoreError {
    CoreError::Drift {
        fields: field.to_string(),
    }
}

// ---------------------------------------------------------------------------
// Tests (pure functions only — no transport)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn params(prompt: Option<&str>, messages: Option<Vec<ChatMessage>>) -> GenerateTextParams {
        GenerateTextParams {
            prompt: prompt.map(str::to_string),
            messages,
            ..GenerateTextParams::default()
        }
    }

    fn msg(role: &str, content: &str) -> ChatMessage {
        ChatMessage {
            role: role.to_string(),
            content: content.to_string(),
        }
    }

    #[test]
    fn resolve_messages_cases() {
        struct Case {
            name: &'static str,
            input: GenerateTextParams,
            expect: Result<Vec<ChatMessage>, ()>,
        }

        let cases = [
            Case {
                name: "prompt lifts to one user turn",
                input: params(Some("Hola"), None),
                expect: Ok(vec![msg("user", "Hola")]),
            },
            Case {
                name: "explicit messages pass through",
                input: params(None, Some(vec![msg("system", "be terse"), msg("user", "hi")])),
                expect: Ok(vec![msg("system", "be terse"), msg("user", "hi")]),
            },
            Case {
                name: "both set is rejected",
                input: params(Some("hi"), Some(vec![msg("user", "hi")])),
                expect: Err(()),
            },
            Case {
                name: "neither set is rejected",
                input: params(None, None),
                expect: Err(()),
            },
            Case {
                name: "blank prompt is rejected",
                input: params(Some("   "), None),
                expect: Err(()),
            },
            Case {
                name: "empty message list is rejected",
                input: params(None, Some(vec![])),
                expect: Err(()),
            },
            Case {
                name: "blank message content is rejected",
                input: params(None, Some(vec![msg("user", "  ")])),
                expect: Err(()),
            },
        ];

        for case in cases {
            match (resolve_messages(&case.input), case.expect) {
                (Ok(got), Ok(want)) => assert_eq!(got, want, "{}", case.name),
                (Err(_), Err(())) => {}
                (got, want) => panic!("{}: got {got:?}, want {want:?}", case.name),
            }
        }
    }

    #[test]
    fn build_chat_body_omits_absent_options_and_sets_stream() {
        let body = build_chat_body("support-bot", &params(Some("hi"), None), true).expect("body");
        assert_eq!(body.get("model").and_then(Value::as_str), Some("support-bot"));
        assert_eq!(body.get("stream"), Some(&Value::Bool(true)));
        assert!(body.get("temperature").is_none());
        assert!(body.get("max_tokens").is_none());
    }

    #[test]
    fn build_chat_body_includes_set_options() {
        let p = GenerateTextParams {
            prompt: Some("hi".to_string()),
            messages: None,
            temperature: Some(0.2),
            max_tokens: Some(300),
        };
        let body = build_chat_body("bot", &p, false).expect("body");
        assert_eq!(body.get("temperature").and_then(Value::as_f64), Some(0.2));
        assert_eq!(body.get("max_tokens").and_then(Value::as_u64), Some(300));
        assert_eq!(body.get("stream"), Some(&Value::Bool(false)));
    }

    #[test]
    fn build_embed_body_rejects_empty_and_uses_default_model() {
        assert!(build_embed_body("bot", &EmbedParams { input: vec![] }).is_err());
        let body = build_embed_body("", &EmbedParams { input: vec!["a".to_string()] }).expect("body");
        assert_eq!(body.get("model").and_then(Value::as_str), Some("default"));
    }

    #[test]
    fn build_endpoint_url_normalizes_slashes() {
        let url = build_endpoint_url("https://api.inferencekey.com/", "acme", "bot", "/chat/completions");
        assert_eq!(
            url,
            "https://api.inferencekey.com/endpoint/acme/bot/v1/chat/completions"
        );
    }

    #[test]
    fn parse_chat_result_cases() {
        struct Case {
            name: &'static str,
            body: Value,
            expect: Result<(&'static str, &'static str, Option<&'static str>), ()>,
        }

        let cases = [
            Case {
                name: "happy path",
                body: json!({
                    "model": "llama-3.1-8b",
                    "choices": [{
                        "message": { "role": "assistant", "content": "Hola!" },
                        "finish_reason": "stop"
                    }]
                }),
                expect: Ok(("Hola!", "llama-3.1-8b", Some("stop"))),
            },
            Case {
                name: "missing finish_reason decodes to None",
                body: json!({
                    "model": "m",
                    "choices": [{ "message": { "content": "ok" } }]
                }),
                expect: Ok(("ok", "m", None)),
            },
            Case {
                name: "no choices is drift",
                body: json!({ "model": "m", "choices": [] }),
                expect: Err(()),
            },
            Case {
                name: "missing content is drift",
                body: json!({ "model": "m", "choices": [{ "message": {} }] }),
                expect: Err(()),
            },
            Case {
                name: "missing model is drift",
                body: json!({ "choices": [{ "message": { "content": "x" } }] }),
                expect: Err(()),
            },
        ];

        for case in cases {
            match (parse_chat_result(&case.body), case.expect) {
                (Ok(got), Ok((text, model, finish))) => {
                    assert_eq!(got.text, text, "{}", case.name);
                    assert_eq!(got.model, model, "{}", case.name);
                    assert_eq!(got.finish_reason.as_deref(), finish, "{}", case.name);
                    assert_eq!(got.raw, case.body, "{} raw passthrough", case.name);
                }
                (Err(_), Err(())) => {}
                (got, want) => panic!("{}: got {got:?}, want {want:?}", case.name),
            }
        }
    }

    #[test]
    fn parse_chunk_tolerates_empty_delta() {
        let role_only = json!({ "choices": [{ "delta": { "role": "assistant" } }] });
        let chunk = parse_chunk(&role_only).expect("chunk");
        assert_eq!(chunk.text, "");
        assert_eq!(chunk.finish_reason, None);

        let with_text = json!({
            "choices": [{ "delta": { "content": "Hi" }, "finish_reason": "stop" }]
        });
        let chunk = parse_chunk(&with_text).expect("chunk");
        assert_eq!(chunk.text, "Hi");
        assert_eq!(chunk.finish_reason.as_deref(), Some("stop"));
    }

    #[test]
    fn parse_embed_result_decodes_vectors() {
        let body = json!({
            "model": "embed-m",
            "data": [
                { "embedding": [0.1, 0.2] },
                { "embedding": [0.3, 0.4] }
            ]
        });
        let result = parse_embed_result(&body).expect("embed");
        assert_eq!(result.model, "embed-m");
        assert_eq!(result.embeddings, vec![vec![0.1, 0.2], vec![0.3, 0.4]]);
    }

    #[test]
    fn parse_embed_result_rejects_non_numeric_and_missing_data() {
        assert!(parse_embed_result(&json!({ "model": "m" })).is_err());
        let bad = json!({ "model": "m", "data": [{ "embedding": ["x"] }] });
        assert!(parse_embed_result(&bad).is_err());
    }
}
