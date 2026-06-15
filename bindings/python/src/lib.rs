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

use pyo3::exceptions::{PyPermissionError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use inferencekey_core::ports::http::HttpPort;
use inferencekey_core::{
    embed, ensure, generate_text, CoreError, EmbedParams, GenerateTextParams, OnDrift, ReqwestHttp,
    WorkloadSpec,
};

/// The native client: a base URL, the HTTP transport, and a blocking runtime.
#[pyclass]
struct Client {
    base_url: String,
    http: ReqwestHttp,
    runtime: tokio::runtime::Runtime,
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
            http: ReqwestHttp::new(),
            runtime,
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
        &self.http
    }

    fn block_on<F: std::future::Future>(&self, fut: F) -> F::Output {
        self.runtime.block_on(fut)
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
    Ok(())
}
