//! Node/TypeScript binding for the InferenceKey SDK.
//!
//! A thin napi-rs shell over `inferencekey-core`. Methods are async and return
//! JS Promises (napi drives them on its own Tokio runtime), so Node's event
//! loop is never blocked. JSON crosses the boundary as strings; the ergonomic
//! typed surface lives in the TypeScript wrapper that ships alongside the addon.
//!
//! Core pipelines borrow their inputs by reference, but a napi async task must
//! be `'static`; we therefore own (clone) every input and share the transport
//! behind an `Arc` so each call awaits only data it owns.

use std::sync::Arc;

use futures_util::stream::{BoxStream, StreamExt};
use napi::bindgen_prelude::*;
use napi_derive::napi;
use tokio::sync::Mutex;

use inferencekey_core::ports::http::HttpPort;
use inferencekey_core::{
    embed, ensure, generate_text, generate_text_stream, readiness_events, CoreError, EmbedParams,
    GenerateTextParams, OnDrift, ReadinessEvent, ReqwestHttp, TextChunk,
};
use inferencekey_core::WorkloadSpec;

/// The native client handed to JavaScript.
#[napi]
pub struct Client {
    base_url: String,
    http: Arc<ReqwestHttp>,
}

#[napi]
impl Client {
    /// Build a client bound to `base_url`.
    #[napi(constructor)]
    pub fn new(base_url: String) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_owned(),
            http: Arc::new(ReqwestHttp::new()),
        }
    }

    /// Provision/reconcile a workload. `spec_json` is a JSON `WorkloadSpec`;
    /// resolves to an `EndpointRef` as JSON.
    #[napi]
    pub async fn ensure(
        &self,
        sdk_token: String,
        project_id: String,
        spec_json: String,
        on_drift: String,
    ) -> Result<String> {
        let spec: WorkloadSpec = parse_json(&spec_json)?;
        let policy = parse_on_drift(&on_drift)?;
        let http = self.http.clone();
        let base_url = self.base_url.clone();
        let result = ensure(
            port(&http),
            &base_url,
            &sdk_token,
            &project_id,
            &spec,
            policy,
        )
        .await
        .map_err(map_core_error)?;
        to_json(result)
    }

    /// Open the readiness progress stream for a workload (control plane).
    /// Resolves to a [`ReadinessStream`] whose `next()` yields one
    /// `ReadinessEvent` JSON string per SSE frame and `null` at end of stream.
    #[napi]
    pub async fn readiness_events(
        &self,
        sdk_token: String,
        project_id: String,
        workload_slug: String,
    ) -> Result<ReadinessStream> {
        let http = self.http.clone();
        let base_url = self.base_url.clone();
        let stream = readiness_events(
            http.clone() as Arc<dyn HttpPort>,
            &base_url,
            &sdk_token,
            &project_id,
            &workload_slug,
        )
        .await
        .map_err(map_core_error)?;
        Ok(ReadinessStream::new(http, stream))
    }

    /// Run a non-streaming chat completion. `params_json` is a JSON
    /// `GenerateTextParams`; resolves to a `TextResult` as JSON.
    #[napi]
    pub async fn generate_text(
        &self,
        project_slug: String,
        workload_slug: String,
        api_key: String,
        params_json: String,
    ) -> Result<String> {
        let params: GenerateTextParams = parse_json(&params_json)?;
        let http = self.http.clone();
        let base_url = self.base_url.clone();
        let result = generate_text(
            port(&http),
            &base_url,
            &project_slug,
            &workload_slug,
            &api_key,
            params,
        )
        .await
        .map_err(map_core_error)?;
        to_json(result)
    }

    /// Open a streaming chat completion. `params_json` is a JSON
    /// `GenerateTextParams`; resolves to a [`ChatStream`] handle whose `next()`
    /// yields one chunk-JSON string per SSE frame and `null` at end of stream.
    ///
    /// The SSE connection is established here (so auth/4xx errors surface up
    /// front, before iteration); each chunk is then pulled lazily by `next()`.
    #[napi]
    pub async fn generate_text_stream(
        &self,
        project_slug: String,
        workload_slug: String,
        api_key: String,
        params_json: String,
    ) -> Result<ChatStream> {
        let params: GenerateTextParams = parse_json(&params_json)?;
        let http = self.http.clone();
        let base_url = self.base_url.clone();
        let stream = generate_text_stream(
            port(&http),
            &base_url,
            &project_slug,
            &workload_slug,
            &api_key,
            params,
        )
        .await
        .map_err(map_core_error)?;
        // Keep the transport alive for the stream's lifetime: the core's
        // BoxStream borrows nothing (it is `'static`), but the underlying
        // reqwest connection lives inside it, so the Arc must outlive iteration.
        Ok(ChatStream::new(http, stream))
    }

    /// Run an embeddings request. `params_json` is a JSON `EmbedParams`;
    /// resolves to an `EmbedResult` as JSON.
    #[napi]
    pub async fn embed(
        &self,
        project_slug: String,
        workload_slug: String,
        api_key: String,
        params_json: String,
    ) -> Result<String> {
        let params: EmbedParams = parse_json(&params_json)?;
        let http = self.http.clone();
        let base_url = self.base_url.clone();
        let result = embed(
            port(&http),
            &base_url,
            &project_slug,
            &workload_slug,
            &api_key,
            params,
        )
        .await
        .map_err(map_core_error)?;
        to_json(result)
    }
}

/// A live streaming chat completion handed to JavaScript.
///
/// Holds the core's chunk stream behind a `Mutex` (napi may call `next()` from
/// any worker thread) and a clone of the transport `Arc` so the underlying
/// connection outlives iteration. Each `next()` pulls one chunk; the TS wrapper
/// adapts this into an `AsyncIterable` for `for await`.
#[napi]
pub struct ChatStream {
    // Held only to keep the transport alive while the stream is consumed.
    _http: Arc<ReqwestHttp>,
    inner: Mutex<BoxStream<'static, inferencekey_core::CoreResult<TextChunk>>>,
}

impl ChatStream {
    fn new(
        http: Arc<ReqwestHttp>,
        inner: BoxStream<'static, inferencekey_core::CoreResult<TextChunk>>,
    ) -> Self {
        Self {
            _http: http,
            inner: Mutex::new(inner),
        }
    }
}

#[napi]
impl ChatStream {
    /// Pull the next chunk as a `TextChunk` JSON string, or `null` when the
    /// stream is exhausted. A transport/parse error rejects the promise.
    #[napi]
    pub async fn next(&self) -> Result<Option<String>> {
        let mut stream = self.inner.lock().await;
        match stream.next().await {
            None => Ok(None),
            Some(Ok(chunk)) => to_json(chunk).map(Some),
            Some(Err(e)) => Err(map_core_error(e)),
        }
    }
}

/// A live readiness progress stream handed to JavaScript. Same shape as
/// [`ChatStream`]: `next()` pulls one `ReadinessEvent` JSON string, or `null`
/// when the stream ends.
#[napi]
pub struct ReadinessStream {
    _http: Arc<ReqwestHttp>,
    inner: Mutex<BoxStream<'static, inferencekey_core::CoreResult<ReadinessEvent>>>,
    // Signalled by `close()`. A pending `next()` races the stream against this
    // notify, so a finished/timed-out wait can unblock the in-flight pull and
    // let the SSE connection drop instead of keeping the event loop alive.
    closed: tokio::sync::Notify,
    is_closed: std::sync::atomic::AtomicBool,
}

impl ReadinessStream {
    fn new(
        http: Arc<ReqwestHttp>,
        inner: BoxStream<'static, inferencekey_core::CoreResult<ReadinessEvent>>,
    ) -> Self {
        Self {
            _http: http,
            inner: Mutex::new(inner),
            closed: tokio::sync::Notify::new(),
            is_closed: std::sync::atomic::AtomicBool::new(false),
        }
    }
}

#[napi]
impl ReadinessStream {
    /// Pull the next `ReadinessEvent` as a JSON string, or `null` when the
    /// stream is exhausted or has been closed.
    #[napi]
    pub async fn next(&self) -> Result<Option<String>> {
        use std::sync::atomic::Ordering;
        if self.is_closed.load(Ordering::Relaxed) {
            return Ok(None);
        }
        let mut stream = self.inner.lock().await;
        tokio::select! {
            biased;
            _ = self.closed.notified() => Ok(None),
            item = stream.next() => match item {
                None => Ok(None),
                Some(Ok(ev)) => to_json(ev).map(Some),
                Some(Err(e)) => Err(map_core_error(e)),
            },
        }
    }

    /// Close the stream so a pending `next()` unblocks and the SSE connection
    /// is released. Idempotent. The wrapper calls this once the wait is decided.
    #[napi]
    pub fn close(&self) {
        self.is_closed
            .store(true, std::sync::atomic::Ordering::Relaxed);
        self.closed.notify_waiters();
    }
}

/// Borrow the shared transport as the port the core expects.
fn port(http: &Arc<ReqwestHttp>) -> &dyn HttpPort {
    http.as_ref()
}

/// Parse a `Deserialize` value from a JSON string.
fn parse_json<T: serde::de::DeserializeOwned>(json: &str) -> Result<T> {
    serde_json::from_str(json).map_err(|e| Error::from_reason(format!("invalid json: {e}")))
}

/// Serialize a `Serialize` value to a JSON string.
fn to_json<T: serde::Serialize>(value: T) -> Result<String> {
    serde_json::to_string(&value).map_err(|e| Error::from_reason(format!("serialize failed: {e}")))
}

/// Map an `on_drift` wire string to the enum.
fn parse_on_drift(raw: &str) -> Result<OnDrift> {
    match raw {
        "" | "reconcile" => Ok(OnDrift::Reconcile),
        "fail" => Ok(OnDrift::Fail),
        "dry_run" => Ok(OnDrift::DryRun),
        "warn" => Ok(OnDrift::Warn),
        "ignore" => Ok(OnDrift::Ignore),
        other => Err(Error::from_reason(format!("unknown on_drift: {other}"))),
    }
}

/// Map a [`CoreError`] to a napi error; the JS wrapper refines these into typed
/// SDK error classes by inspecting the message/code.
fn map_core_error(err: CoreError) -> Error {
    Error::from_reason(err.to_string())
}
