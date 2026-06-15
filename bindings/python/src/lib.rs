//! Python binding for the InferenceKey SDK.
//!
//! A thin pyo3 shell over `inferencekey-core`: it owns a blocking Tokio runtime,
//! marshals JSON in/out at the boundary, and maps [`CoreError`] to typed Python
//! exceptions. The ergonomic, dataclass-based surface lives in the pure-Python
//! wrapper (`inferencekey/__init__.py`); this layer stays minimal.

// pyo3's `#[pymethods]` expansion wraps `?` in a `PyErr: From<PyErr>` conversion
// that clippy flags as useless on every fallible method's return type. It is a
// macro-generated false positive, not our code — silence it crate-wide.
#![allow(clippy::useless_conversion)]

use std::sync::Arc;

use futures_util::stream::{BoxStream, StreamExt};
use pyo3::exceptions::{PyPermissionError, PyRuntimeError, PyStopIteration, PyValueError};
use pyo3::prelude::*;

use inferencekey_core::ports::http::HttpPort;
use inferencekey_core::{
    delete as core_delete, embed, ensure, generate_text, generate_text_stream, readiness_events,
    CoreError, CoreResult, EmbedParams, GenerateTextParams, OnDrift, ReadinessEvent, ReqwestHttp,
    TextChunk, WorkloadSpec,
};

/// The native client: a base URL, the HTTP transport, and a blocking runtime.
///
/// The runtime is shared (via `Arc`) with any [`ChatStream`] this client opens,
/// since each `__next__` drives the same reactor to pull one chunk.
#[pyclass]
struct Client {
    base_url: String,
    http: Arc<ReqwestHttp>,
    runtime: Arc<tokio::runtime::Runtime>,
}

#[pymethods]
impl Client {
    /// Build a client bound to `base_url`.
    #[new]
    fn new(base_url: &str) -> PyResult<Self> {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| PyRuntimeError::new_err(format!("failed to start runtime: {e}")))?;
        Ok(Self {
            base_url: base_url.trim_end_matches('/').to_owned(),
            http: Arc::new(ReqwestHttp::new()),
            runtime: Arc::new(runtime),
        })
    }

    /// Provision/reconcile a workload. `spec_json` is a JSON `WorkloadSpec`;
    /// returns an `EndpointRef` as JSON.
    fn ensure(
        &self,
        py: Python<'_>,
        sdk_token: &str,
        project_id: &str,
        spec_json: &str,
        on_drift: &str,
    ) -> PyResult<String> {
        let spec: WorkloadSpec = parse_json(spec_json)?;
        let policy = parse_on_drift(on_drift)?;
        // Release the GIL while the blocking network call runs.
        let result = py.allow_threads(|| {
            self.block_on(ensure(
                self.port(),
                &self.base_url,
                sdk_token,
                project_id,
                &spec,
                policy,
            ))
        });
        result.map_err(map_core_error).and_then(to_json)
    }

    /// Delete a workload by slug (control plane). Returns `True` if it existed,
    /// `False` if it was already absent (idempotent). Cloud GPUs the autoscaler
    /// provisioned are torn down server-side.
    fn delete(
        &self,
        py: Python<'_>,
        sdk_token: &str,
        project_id: &str,
        workload_slug: &str,
    ) -> PyResult<bool> {
        let result = py.allow_threads(|| {
            self.block_on(core_delete(
                self.port(),
                &self.base_url,
                sdk_token,
                project_id,
                workload_slug,
            ))
        });
        result.map_err(map_core_error)
    }

    /// Open the readiness progress stream for a workload (control plane).
    /// Returns a [`ReadinessStream`] iterator whose `__next__` yields one
    /// `ReadinessEvent` JSON string per SSE frame.
    fn readiness_events(
        &self,
        py: Python<'_>,
        sdk_token: &str,
        project_id: &str,
        workload_slug: &str,
    ) -> PyResult<ReadinessStream> {
        let http = self.http.clone() as Arc<dyn HttpPort>;
        let stream = py.allow_threads(|| {
            self.block_on(readiness_events(
                http,
                &self.base_url,
                sdk_token,
                project_id,
                workload_slug,
            ))
        });
        let stream = stream.map_err(map_core_error)?;
        Ok(ReadinessStream {
            runtime: self.runtime.clone(),
            _http: self.http.clone(),
            inner: Some(stream),
        })
    }

    /// Run a non-streaming chat completion. `params_json` is a JSON
    /// `GenerateTextParams`; returns a `TextResult` as JSON.
    fn generate_text(
        &self,
        py: Python<'_>,
        project_slug: &str,
        workload_slug: &str,
        api_key: &str,
        params_json: &str,
    ) -> PyResult<String> {
        let params: GenerateTextParams = parse_json(params_json)?;
        let result = py.allow_threads(|| {
            self.block_on(generate_text(
                self.port(),
                &self.base_url,
                project_slug,
                workload_slug,
                api_key,
                params,
            ))
        });
        result.map_err(map_core_error).and_then(to_json)
    }

    /// Open a streaming chat completion. `params_json` is a JSON
    /// `GenerateTextParams`; returns a [`ChatStream`] iterator whose `__next__`
    /// yields one `TextChunk` JSON string per SSE frame.
    ///
    /// The SSE connection is opened here (so auth/4xx errors raise immediately),
    /// then chunks are pulled lazily as the iterator is advanced.
    fn generate_text_stream(
        &self,
        py: Python<'_>,
        project_slug: &str,
        workload_slug: &str,
        api_key: &str,
        params_json: &str,
    ) -> PyResult<ChatStream> {
        let params: GenerateTextParams = parse_json(params_json)?;
        let stream = py.allow_threads(|| {
            self.block_on(generate_text_stream(
                self.port(),
                &self.base_url,
                project_slug,
                workload_slug,
                api_key,
                params,
            ))
        });
        let stream = stream.map_err(map_core_error)?;
        Ok(ChatStream {
            runtime: self.runtime.clone(),
            // Keep the transport alive for as long as the stream is consumed.
            _http: self.http.clone(),
            inner: Some(stream),
        })
    }

    /// Run an embeddings request. `params_json` is a JSON `EmbedParams`;
    /// returns an `EmbedResult` as JSON.
    fn embed(
        &self,
        py: Python<'_>,
        project_slug: &str,
        workload_slug: &str,
        api_key: &str,
        params_json: &str,
    ) -> PyResult<String> {
        let params: EmbedParams = parse_json(params_json)?;
        let result = py.allow_threads(|| {
            self.block_on(embed(
                self.port(),
                &self.base_url,
                project_slug,
                workload_slug,
                api_key,
                params,
            ))
        });
        result.map_err(map_core_error).and_then(to_json)
    }
}

impl Client {
    fn port(&self) -> &dyn HttpPort {
        self.http.as_ref()
    }

    fn block_on<F: std::future::Future>(&self, fut: F) -> F::Output {
        self.runtime.block_on(fut)
    }
}

/// A live streaming chat completion exposed to Python as an iterator.
///
/// Implements the iterator protocol: `__iter__` returns self and each
/// `__next__` drives the shared runtime to pull one chunk, releasing the GIL
/// across the blocking await. Exhaustion raises `StopIteration`; the typed
/// Python wrapper turns each chunk-JSON string into a `TextChunk`.
#[pyclass]
struct ChatStream {
    runtime: Arc<tokio::runtime::Runtime>,
    // Held only to keep the transport alive while the stream is consumed.
    _http: Arc<ReqwestHttp>,
    // `None` once the stream is exhausted, so further `__next__` calls are a
    // clean `StopIteration` rather than re-polling a finished stream.
    inner: Option<BoxStream<'static, CoreResult<TextChunk>>>,
}

#[pymethods]
impl ChatStream {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    /// Advance the stream by one chunk. Returns the chunk as a JSON string, or
    /// raises `StopIteration` at end of stream.
    fn __next__(&mut self, py: Python<'_>) -> PyResult<String> {
        let runtime = self.runtime.clone();
        let Some(stream) = self.inner.as_mut() else {
            return Err(PyStopIteration::new_err(()));
        };
        let next = py.allow_threads(|| runtime.block_on(stream.next()));
        match next {
            Some(Ok(chunk)) => to_json(chunk),
            Some(Err(e)) => Err(map_core_error(e)),
            None => {
                self.inner = None;
                Err(PyStopIteration::new_err(()))
            }
        }
    }
}

/// A live readiness progress stream exposed to Python as an iterator. Same
/// shape as [`ChatStream`]: each `__next__` pulls one `ReadinessEvent` JSON
/// string (raising `StopIteration` at end of stream).
#[pyclass]
struct ReadinessStream {
    runtime: Arc<tokio::runtime::Runtime>,
    _http: Arc<ReqwestHttp>,
    inner: Option<BoxStream<'static, CoreResult<ReadinessEvent>>>,
}

#[pymethods]
impl ReadinessStream {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self, py: Python<'_>) -> PyResult<String> {
        let runtime = self.runtime.clone();
        let Some(stream) = self.inner.as_mut() else {
            return Err(PyStopIteration::new_err(()));
        };
        let next = py.allow_threads(|| runtime.block_on(stream.next()));
        match next {
            Some(Ok(ev)) => to_json(ev),
            Some(Err(e)) => Err(map_core_error(e)),
            None => {
                self.inner = None;
                Err(PyStopIteration::new_err(()))
            }
        }
    }
}

/// Parse a `Deserialize` value from a JSON string, mapping errors to `ValueError`.
fn parse_json<T: serde::de::DeserializeOwned>(json: &str) -> PyResult<T> {
    serde_json::from_str(json).map_err(|e| PyValueError::new_err(format!("invalid json: {e}")))
}

/// Serialize a `Serialize` value to a JSON string.
fn to_json<T: serde::Serialize>(value: T) -> PyResult<String> {
    serde_json::to_string(&value)
        .map_err(|e| PyRuntimeError::new_err(format!("serialize failed: {e}")))
}

/// Map an `on_drift` wire string to the enum.
fn parse_on_drift(raw: &str) -> PyResult<OnDrift> {
    match raw {
        "" | "reconcile" => Ok(OnDrift::Reconcile),
        "fail" => Ok(OnDrift::Fail),
        "dry_run" => Ok(OnDrift::DryRun),
        "warn" => Ok(OnDrift::Warn),
        "ignore" => Ok(OnDrift::Ignore),
        other => Err(PyValueError::new_err(format!("unknown on_drift: {other}"))),
    }
}

/// Map a [`CoreError`] to the closest Python exception type.
fn map_core_error(err: CoreError) -> PyErr {
    let message = err.to_string();
    match err {
        CoreError::Permission { .. } | CoreError::Auth(_) => PyPermissionError::new_err(message),
        CoreError::Config(_) | CoreError::Validation(_) | CoreError::BadRequest(_) => {
            PyValueError::new_err(message)
        }
        _ => PyRuntimeError::new_err(message),
    }
}

/// The native extension module (imported by the Python wrapper as `_inferencekey`).
#[pymodule]
fn _inferencekey(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Client>()?;
    m.add_class::<ChatStream>()?;
    m.add_class::<ReadinessStream>()?;
    Ok(())
}
