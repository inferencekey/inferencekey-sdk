//! `reqwest`-backed implementation of [`HttpPort`].
//!
//! This is the only place in the core that talks to the network. Everything
//! else stays pure and drives this adapter through the [`HttpPort`] trait, so
//! the pipelines never see `reqwest` types and tests can swap in a fake port.
//!
//! Responsibilities, kept strictly separated:
//! * PURE — [`build_request`] turns a [`HttpRequest`] into a
//!   [`reqwest::RequestBuilder`]; [`map_status_error`] / [`decode_body`] map
//!   bytes to typed JSON or a [`CoreError`].
//! * EFFECT — the trait methods do the actual IO inside a boxed future.
//!
//! Async is modelled with `BoxFuture` (a `Pin<Box<dyn Future + Send>>`) rather
//! than `async fn` in the trait, so the body of each method is a
//! `Box::pin(async move { ... })`. Every `reqwest` failure is mapped to
//! [`CoreError::Network`] at the boundary via [`map_send_error`]; non-2xx
//! responses are mapped to the typed control-plane errors via
//! [`map_status_error`], mirroring the wire codes.

use std::pin::Pin;
use std::task::{Context, Poll};

use futures_util::stream::{Stream, StreamExt};
use serde_json::Value;

use crate::domain::sse::{parse_sse_line, SseLine};
use crate::errors::{CoreError, CoreResult, PermissionCode};
use crate::ports::http::{BoxFuture, HttpMethod, HttpPort, HttpRequest, JsonStream};

/// `reqwest`-backed [`HttpPort`]. Cheap to clone (the inner client is an `Arc`).
#[derive(Clone)]
pub struct ReqwestHttp {
    client: reqwest::Client,
}

impl ReqwestHttp {
    /// Build an adapter over a default `reqwest` client.
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
        }
    }

    /// Wrap an already-configured client (timeouts, proxy, …) chosen by the caller.
    pub fn with_client(client: reqwest::Client) -> Self {
        Self { client }
    }

    /// PURE-ish: assemble the `reqwest` request from our transport-agnostic
    /// [`HttpRequest`]. Sets the bearer token and JSON body when present.
    fn build_request(&self, req: &HttpRequest) -> reqwest::RequestBuilder {
        let builder = self
            .client
            .request(to_method(req.method), &req.url)
            .bearer_auth(&req.token);
        with_json_body(builder, req.body.as_ref())
    }
}

impl Default for ReqwestHttp {
    fn default() -> Self {
        Self::new()
    }
}

impl HttpPort for ReqwestHttp {
    fn request_json<'a>(&'a self, req: HttpRequest) -> BoxFuture<'a, Value> {
        Box::pin(async move {
            let response = self
                .build_request(&req)
                .send()
                .await
                .map_err(map_send_error)?;

            let status = response.status();
            let bytes = response.bytes().await.map_err(map_send_error)?;

            match status.is_success() {
                true => decode_body(bytes.as_ref()),
                false => Err(map_status_error(status.as_u16(), bytes.as_ref())),
            }
        })
    }

    fn stream_sse<'a>(&'a self, req: HttpRequest) -> BoxFuture<'a, JsonStream> {
        Box::pin(async move {
            let response = self
                .build_request(&req)
                .header(reqwest::header::ACCEPT, "text/event-stream")
                .send()
                .await
                .map_err(map_send_error)?;

            let status = response.status();
            if !status.is_success() {
                let bytes = response.bytes().await.map_err(map_send_error)?;
                return Err(map_status_error(status.as_u16(), bytes.as_ref()));
            }

            Ok(into_json_stream(response))
        })
    }
}

/// PURE: map our verb enum onto the concrete `reqwest::Method`. Infallible — no
/// fallible string parse, no `as` cast.
fn to_method(method: HttpMethod) -> reqwest::Method {
    match method {
        HttpMethod::Get => reqwest::Method::GET,
        HttpMethod::Post => reqwest::Method::POST,
        HttpMethod::Patch => reqwest::Method::PATCH,
        HttpMethod::Delete => reqwest::Method::DELETE,
    }
}

/// PURE: attach a JSON body when present, leaving the builder untouched otherwise.
fn with_json_body(
    builder: reqwest::RequestBuilder,
    body: Option<&Value>,
) -> reqwest::RequestBuilder {
    match body {
        Some(value) => builder.json(value),
        None => builder,
    }
}

/// PURE: decode a successful response body. An empty body (e.g. `204 No Content`)
/// decodes to [`Value::Null`] rather than a JSON error.
fn decode_body(bytes: &[u8]) -> CoreResult<Value> {
    if bytes.is_empty() {
        return Ok(Value::Null);
    }
    serde_json::from_slice(bytes).map_err(CoreError::from)
}

/// PURE: map a non-2xx response to a typed [`CoreError`], mirroring the
/// control-plane codes. The body is `{ "error": "<code>" }` on the control
/// plane; we fall back to the raw text when it is not that shape.
fn map_status_error(status: u16, body: &[u8]) -> CoreError {
    let code = extract_error_code(body);
    let message = error_message(&code, body);

    match status {
        401 => CoreError::Auth(message),
        403 => CoreError::Permission {
            code: PermissionCode::from_code(code.as_deref()),
            message,
        },
        404 => CoreError::NotFound(message),
        400 => CoreError::BadRequest(message),
        other => CoreError::Api {
            status: other,
            message,
        },
    }
}

/// PURE: pull the machine-readable `error` code out of a control-plane body.
fn extract_error_code(body: &[u8]) -> Option<String> {
    serde_json::from_slice::<Value>(body)
        .ok()
        .as_ref()
        .and_then(|v| v.get("error"))
        .and_then(Value::as_str)
        .map(str::to_owned)
}

/// PURE: a human-readable message — the wire code if we have one, else the raw
/// body text, else a generic placeholder.
fn error_message(code: &Option<String>, body: &[u8]) -> String {
    code.clone()
        .or_else(|| non_empty_text(body))
        .unwrap_or_else(|| "request failed".to_owned())
}

/// PURE: the body as trimmed UTF-8 text, or `None` when empty/non-text.
fn non_empty_text(body: &[u8]) -> Option<String> {
    std::str::from_utf8(body)
        .ok()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
}

/// PURE: collapse a `reqwest` transport failure into [`CoreError::Network`].
fn map_send_error(err: reqwest::Error) -> CoreError {
    CoreError::Network(err.to_string())
}

/// EFFECT→PURE bridge: turn the raw byte stream into a [`JsonStream`] of parsed
/// SSE values, mapping transport errors to [`CoreError::Network`] and stopping
/// at the terminal `[DONE]` sentinel.
fn into_json_stream(response: reqwest::Response) -> JsonStream {
    let chunks = response.bytes_stream().map(|chunk| chunk.map_err(map_send_error));
    Box::new(SseLines::new(chunks))
}

/// Buffers raw response bytes, splits them on newlines, and yields a parsed
/// [`Value`] for every `data:` line until the terminal sentinel is seen.
///
/// Only `data:` lines carry payloads; comments, blank lines, and other SSE
/// fields are skipped by [`parse_sse_line`] (they produce [`SseLine::Skip`]).
///
/// Generic over the chunk type `C: AsRef<[u8]>` so this adapter never names the
/// `bytes` crate: the concrete stream yields `bytes::Bytes`, which satisfies the
/// bound, and we only ever touch its bytes via `as_ref`.
struct SseLines<S> {
    /// Upstream chunks, transport errors already mapped to [`CoreError`].
    inner: S,
    /// Bytes received but not yet terminated by a newline.
    buffer: Vec<u8>,
    /// Set once the `[DONE]` sentinel is parsed so the stream ends cleanly.
    done: bool,
}

impl<S> SseLines<S> {
    fn new(inner: S) -> Self {
        Self {
            inner,
            buffer: Vec::new(),
            done: false,
        }
    }

    /// Take the next complete line (without its `\n`) out of the buffer, if any.
    fn take_line(&mut self) -> Option<Vec<u8>> {
        let newline = self.buffer.iter().position(|&b| b == b'\n')?;
        let mut line: Vec<u8> = self.buffer.drain(..=newline).collect();
        line.pop(); // drop the trailing '\n'
        if line.last() == Some(&b'\r') {
            line.pop(); // tolerate CRLF line endings
        }
        Some(line)
    }

    /// Parse one buffered line into a stream item, or `None` to keep draining.
    /// `Some(value)` yields a payload; a `[DONE]` sentinel flips `done` and
    /// yields nothing; skipped lines are silently consumed.
    fn next_from_buffer(&mut self) -> Option<Value> {
        while let Some(line) = self.take_line() {
            match classify(&line) {
                SseLine::Data(value) => return Some(value),
                SseLine::Done => {
                    self.done = true;
                    return None;
                }
                SseLine::Skip => continue,
            }
        }
        None
    }

    /// On end-of-stream, parse any trailing line that lacked a final newline.
    fn flush_tail(&mut self) -> Option<Value> {
        if self.buffer.is_empty() {
            return None;
        }
        let line = std::mem::take(&mut self.buffer);
        match classify(&line) {
            SseLine::Data(value) => Some(value),
            _ => None,
        }
    }
}

impl<S, C> Stream for SseLines<S>
where
    S: Stream<Item = CoreResult<C>> + Unpin,
    C: AsRef<[u8]>,
{
    type Item = CoreResult<Value>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        loop {
            if self.done {
                return Poll::Ready(None);
            }

            // Drain whatever complete lines we already have first.
            if let Some(value) = self.next_from_buffer() {
                return Poll::Ready(Some(Ok(value)));
            }
            if self.done {
                return Poll::Ready(None);
            }

            // Buffer exhausted of complete lines — pull the next chunk.
            match self.inner.poll_next_unpin(cx) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(None) => return Poll::Ready(self.flush_tail().map(Ok)),
                Poll::Ready(Some(Ok(chunk))) => self.buffer.extend_from_slice(chunk.as_ref()),
                Poll::Ready(Some(Err(e))) => return Poll::Ready(Some(Err(e))),
            }
        }
    }
}

/// PURE: decode a single raw line as UTF-8 and classify it. Non-UTF-8 lines are
/// not valid in the OpenAI-compatible event stream, so we treat them as noise
/// to skip rather than fail the whole stream.
fn classify(line: &[u8]) -> SseLine {
    match std::str::from_utf8(line) {
        Ok(text) => parse_sse_line(text),
        Err(_) => SseLine::Skip,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::stream;
    use serde_json::json;

    /// Drive an `SseLines` adapter over fixed chunks and collect its output.
    async fn run(chunks: Vec<&'static str>) -> Vec<CoreResult<Value>> {
        let inner = stream::iter(
            chunks
                .into_iter()
                .map(|c| Ok::<_, CoreError>(c.as_bytes().to_vec())),
        );
        SseLines::new(inner).collect().await
    }

    #[tokio::test]
    async fn yields_data_frames_and_stops_at_done() {
        let out = run(vec![
            "data: {\"n\":1}\n",
            "data: {\"n\":2}\n",
            "data: [DONE]\n",
            "data: {\"n\":3}\n",
        ])
        .await;

        let values: Vec<Value> = out.into_iter().filter_map(Result::ok).collect();
        assert_eq!(values, vec![json!({"n": 1}), json!({"n": 2})]);
    }

    #[tokio::test]
    async fn reassembles_lines_split_across_chunks() {
        let out = run(vec!["data: {\"n\"", ":42}\n", "data: [DONE]\n"]).await;
        let values: Vec<Value> = out.into_iter().filter_map(Result::ok).collect();
        assert_eq!(values, vec![json!({"n": 42})]);
    }

    #[tokio::test]
    async fn skips_comments_blanks_and_flushes_unterminated_tail() {
        let out = run(vec![": keep-alive\n", "\n", "data: {\"tail\":true}"]).await;
        let values: Vec<Value> = out.into_iter().filter_map(Result::ok).collect();
        assert_eq!(values, vec![json!({"tail": true})]);
    }

    #[test]
    fn maps_status_codes_to_typed_errors() {
        let auth = map_status_error(401, b"{\"error\":\"bad token\"}");
        assert!(matches!(auth, CoreError::Auth(_)));

        let perm = map_status_error(403, b"{\"error\":\"wrong_credential_type\"}");
        match perm {
            CoreError::Permission { code, .. } => {
                assert_eq!(code, PermissionCode::WrongCredentialType)
            }
            other => panic!("expected Permission, got {other:?}"),
        }

        assert!(matches!(map_status_error(404, b""), CoreError::NotFound(_)));
        assert!(matches!(map_status_error(400, b""), CoreError::BadRequest(_)));
        assert!(matches!(
            map_status_error(503, b"upstream down"),
            CoreError::Api { status: 503, .. }
        ));
    }

    #[test]
    fn empty_success_body_decodes_to_null() {
        assert_eq!(decode_body(b"").expect("null"), Value::Null);
    }

    #[test]
    fn extracts_control_plane_error_code() {
        let code = extract_error_code(b"{\"error\":\"scope_insufficient\"}");
        assert_eq!(code.as_deref(), Some("scope_insufficient"));
        assert_eq!(extract_error_code(b"not json"), None);
    }

    #[test]
    fn maps_verbs_to_reqwest_methods() {
        assert_eq!(to_method(HttpMethod::Get), reqwest::Method::GET);
        assert_eq!(to_method(HttpMethod::Post), reqwest::Method::POST);
        assert_eq!(to_method(HttpMethod::Patch), reqwest::Method::PATCH);
        assert_eq!(to_method(HttpMethod::Delete), reqwest::Method::DELETE);
    }
}
