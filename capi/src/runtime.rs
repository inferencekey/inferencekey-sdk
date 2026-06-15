//! The blocking client: an internal Tokio runtime + the core's HTTP adapter.
//!
//! The core is async; the C ABI is synchronous. This wraps a current-thread
//! runtime so each fallible entry point can `block_on` a core pipeline without
//! the caller ever seeing a future.

use std::future::Future;

use inferencekey_core::{CoreError, CoreResult, ReqwestHttp};
use inferencekey_core::ports::http::HttpPort;

/// An opaque, owned client handle handed across the C ABI.
pub struct Client {
    base_url: String,
    http: ReqwestHttp,
    runtime: tokio::runtime::Runtime,
}

impl Client {
    /// Build a client bound to `base_url` over the given transport. Fails only
    /// if a Tokio runtime cannot be created.
    pub fn new(base_url: &str, http: ReqwestHttp) -> CoreResult<Self> {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| CoreError::Network(format!("failed to start runtime: {e}")))?;
        Ok(Self {
            base_url: base_url.trim_end_matches('/').to_owned(),
            http,
            runtime,
        })
    }

    /// The configured base URL (no trailing slash).
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// The transport, borrowed as the port the core pipelines expect.
    pub fn http(&self) -> &dyn HttpPort {
        &self.http
    }

    /// Drive a core future to completion on the internal runtime.
    pub fn block_on<F: Future>(&self, fut: F) -> F::Output {
        self.runtime.block_on(fut)
    }
}
