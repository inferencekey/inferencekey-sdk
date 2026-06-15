//! The HTTP port: the boundary the pipelines depend on instead of `reqwest`.
//!
//! Pipelines build a [`HttpRequest`] (pure: method, url, token, optional JSON
//! body) and hand it to an [`HttpPort`] implementation. The concrete adapter
//! (built on `reqwest`, or a fake in tests) performs the IO and maps transport
//! failures to [`CoreError`]. Keeping this trait reqwest-free lets us swap the
//! transport, mock it, and keep the core logic synchronous and pure.
//!
//! Async is modelled with `Pin<Box<dyn Future<...> + Send + '_>>` rather than
//! `async fn` in traits so the trait is object-safe and compiles on older
//! toolchains. The borrow of `&self` is captured by the `'a` lifetime on the
//! returned future, so implementations may hold a client across the `.await`.

use std::future::Future;
use std::pin::Pin;

use crate::errors::CoreResult;

/// The HTTP verbs the SDK issues. `Delete` is reserved for future surfaces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HttpMethod {
    Get,
    Post,
    Patch,
    Delete,
}

impl HttpMethod {
    /// The uppercase wire token for this verb.
    pub fn as_str(&self) -> &'static str {
        match self {
            HttpMethod::Get => "GET",
            HttpMethod::Post => "POST",
            HttpMethod::Patch => "PATCH",
            HttpMethod::Delete => "DELETE",
        }
    }
}

/// A fully-resolved HTTP request: everything the transport needs, nothing it
/// has to decide. `token` is the bearer value only (without the `Bearer `
/// prefix); the adapter is responsible for redacting it in logs. `body` is
/// already-encoded opaque JSON, present only for verbs that carry one.
#[derive(Debug, Clone)]
pub struct HttpRequest {
    /// HTTP verb to use.
    pub method: HttpMethod,
    /// Absolute request URL, already joined from base + path.
    pub url: String,
    /// Bearer credential value (prefix-redacted before it reaches any log).
    pub token: String,
    /// Optional JSON request body. `None` for bodyless verbs such as `GET`.
    pub body: Option<serde_json::Value>,
}

impl HttpRequest {
    /// Build a request that carries a JSON body.
    pub fn with_body(
        method: HttpMethod,
        url: impl Into<String>,
        token: impl Into<String>,
        body: serde_json::Value,
    ) -> Self {
        HttpRequest {
            method,
            url: url.into(),
            token: token.into(),
            body: Some(body),
        }
    }

    /// Build a bodyless request (e.g. a `GET` or `DELETE`).
    pub fn empty(
        method: HttpMethod,
        url: impl Into<String>,
        token: impl Into<String>,
    ) -> Self {
        HttpRequest {
            method,
            url: url.into(),
            token: token.into(),
            body: None,
        }
    }
}

/// A boxed future returning `T` or a [`CoreError`], borrowing `&self` for `'a`.
///
/// Used so the trait stays object-safe (`dyn HttpPort`) without depending on
/// unstable `async fn` in traits.
pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = CoreResult<T>> + Send + 'a>>;

/// A boxed stream of decoded JSON values, each item fallible. `Unbox`-friendly:
/// `Unpin` so callers can `.next().await` without pinning ceremony.
pub type JsonStream = Box<dyn futures_util::Stream<Item = CoreResult<serde_json::Value>> + Send + Unpin>;

/// The transport boundary the pipelines depend on.
///
/// Implementations turn an [`HttpRequest`] into either a single decoded JSON
/// body or a stream of decoded JSON events, mapping every transport or non-2xx
/// outcome to [`CoreError`] before returning. The trait is `Send + Sync` so a
/// single instance can be shared across tasks behind an `Arc`.
pub trait HttpPort: Send + Sync {
    /// Perform a unary request and decode the response body as JSON.
    ///
    /// The adapter maps `401`/`403`/`404`/`400`/other non-2xx and connection
    /// failures to the matching [`CoreError`] variant; on success it returns
    /// the parsed body.
    fn request_json<'a>(&'a self, req: HttpRequest) -> BoxFuture<'a, serde_json::Value>;

    /// Perform a streaming request and yield decoded SSE data frames as JSON.
    ///
    /// Each `data:` line is parsed into a [`serde_json::Value`]; the terminal
    /// `data: [DONE]` sentinel ends the stream without producing an item. Per
    /// the OpenAI-compatible data plane, chunks carry `chat.completion.chunk`.
    fn stream_sse<'a>(&'a self, req: HttpRequest) -> BoxFuture<'a, JsonStream>;
}
