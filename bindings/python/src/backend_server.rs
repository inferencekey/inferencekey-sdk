//! Native HTTP/SSE transport for `inferencekey.backend.serve`.
//!
//! A pyo3 shell over the `inferencekey-backend-server` crate. Rust accepts
//! connections, parses HTTP, answers `/healthz`, `/meta`, `/v1/models` and
//! `/debug/stats`, and writes every byte; the Python object handed in as `app`
//! (`serve._NativeApp`) decides what each POST request means.
//!
//! GIL rules:
//! 1. Only blocking-pool threads call into Python (`Python::with_gil`); the
//!    tokio IO threads never touch a Python object, only `Bytes`.
//! 2. Every wait on the Rust side (binding, shutdown, join, a send into a full
//!    stream channel) runs inside `py.allow_threads`.
//! 3. No Rust lock is held across a Python call.
//! 4. Python owns SIGTERM and SIGINT; the runtime installs no signal handler.

use std::io;
use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use pyo3::exceptions::{PyOSError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use inferencekey_backend_server::{
    internal_error, start, AppHandler, JsonReply, Route, SendOutcome, ServerConfig, ServerHandle,
    StreamSink as Sink,
};

/// Bump when the Python side needs something the binding did not have before;
/// `serve._select_transport` only picks the native transport on a match.
pub const BACKEND_SERVER_ABI: u32 = 1;

/// How often a blocking `wait()` comes back to let Python run signal handlers.
const SIGNAL_CHECK_INTERVAL: Duration = Duration::from_millis(200);

/// The application object (`serve._NativeApp`) seen from the transport.
struct PyApp {
    app: Py<PyAny>,
}

impl PyApp {
    /// Log an exception that escaped the Python application (a bug in
    /// `serve.py`, since backend errors are mapped there) with its traceback.
    fn report(py: Python<'_>, what: &str, err: PyErr) {
        eprintln!("native transport: {what} raised:");
        err.print(py);
    }
}

impl AppHandler for PyApp {
    fn call_json(&self, route: Route, body: Bytes) -> JsonReply {
        Python::with_gil(|py| {
            let result = self
                .app
                .call_method1(py, "handle_json", (route.path(), PyBytes::new_bound(py, &body)))
                .and_then(|reply| {
                    let (status, body): (u16, Bound<'_, PyBytes>) = reply.extract(py)?;
                    Ok(JsonReply::new(status, Bytes::copy_from_slice(body.as_bytes())))
                });
            result.unwrap_or_else(|err| {
                Self::report(py, "handle_json", err);
                JsonReply::new(500, internal_error(route, "internal server error"))
            })
        })
    }

    fn call_stream(&self, body: Bytes, sink: Sink) {
        Python::with_gil(|py| {
            let result = Py::new(py, StreamSink { inner: sink.clone() }).and_then(|py_sink| {
                self.app
                    .call_method1(py, "handle_stream", (PyBytes::new_bound(py, &body), py_sink))
                    .map(drop)
            });
            if let Err(err) = result {
                Self::report(py, "handle_stream", err);
                // Only answers if nothing was sent yet; a started stream just
                // ends (the transport finishes it when this call returns).
                sink.reply(500, internal_error(Route::Chat, "internal server error"));
            }
        })
    }
}

/// Map a bind or runtime error to the `OSError` the stdlib server would raise
/// (`errno` set, so `EADDRINUSE` reads the same).
fn os_error(err: io::Error) -> PyErr {
    match err.raw_os_error() {
        Some(errno) => {
            let text = err.to_string();
            let message = text.split(" (os error").next().unwrap_or(&text).to_owned();
            PyOSError::new_err((errno, message))
        }
        None => PyOSError::new_err(err.to_string()),
    }
}

/// A native HTTP server bound to `127.0.0.1:port`, serving `app`.
///
/// The socket is bound and listening when the constructor returns, so slow
/// setup can run afterwards while `/healthz` answers 503 from Rust.
#[pyclass(frozen, module = "inferencekey._inferencekey")]
pub struct BackendServer {
    handle: ServerHandle,
    io_threads: usize,
    backlog: u32,
    max_inflight: usize,
}

#[pymethods]
impl BackendServer {
    #[new]
    #[pyo3(signature = (
        port,
        app,
        backlog = 4096,
        io_threads = None,
        max_inflight = 1024,
        stream_max_bytes = 8 << 20,
        sse_connection_close = false,
        shutdown_grace = 5.0,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        port: u16,
        app: Py<PyAny>,
        backlog: u32,
        io_threads: Option<usize>,
        max_inflight: usize,
        stream_max_bytes: usize,
        sse_connection_close: bool,
        shutdown_grace: f64,
    ) -> PyResult<Self> {
        if !(shutdown_grace.is_finite() && shutdown_grace >= 0.0) {
            return Err(PyValueError::new_err("shutdown_grace must be a non-negative number of seconds"));
        }
        let cfg = ServerConfig {
            port,
            backlog: backlog.max(1),
            io_threads: io_threads.unwrap_or_else(ServerConfig::default_io_threads).max(1),
            max_inflight: max_inflight.max(1),
            stream_max_bytes: stream_max_bytes.max(1),
            sse_connection_close,
            shutdown_grace: Duration::from_secs_f64(shutdown_grace),
            ..ServerConfig::default()
        };
        let (io_threads, backlog, max_inflight) = (cfg.io_threads, cfg.backlog, cfg.max_inflight);
        let app: Arc<dyn AppHandler> = Arc::new(PyApp { app });
        let handle = py.allow_threads(|| start(cfg, app)).map_err(os_error)?;
        Ok(BackendServer { handle, io_threads, backlog, max_inflight })
    }

    /// The bound port (the real one when 0 was requested).
    #[getter]
    fn local_port(&self) -> u16 {
        self.handle.local_addr().port()
    }

    /// `("127.0.0.1", port)`, like `socketserver.TCPServer.server_address`.
    #[getter]
    fn server_address(&self) -> (String, u16) {
        let addr = self.handle.local_addr();
        (addr.ip().to_string(), addr.port())
    }

    #[getter]
    fn io_threads(&self) -> usize {
        self.io_threads
    }

    #[getter]
    fn backlog(&self) -> u32 {
        self.backlog
    }

    #[getter]
    fn max_inflight(&self) -> usize {
        self.max_inflight
    }

    /// Flip `/healthz` between 503 `loading` and 200 `ok`.
    fn set_ready(&self, ready: bool) {
        self.handle.set_ready(ready);
    }

    /// The manifest JSON served verbatim at `/meta` (`/v1/models` derives from it).
    fn set_manifest(&self, manifest: &[u8]) {
        self.handle.set_manifest(manifest);
    }

    /// The `/debug/stats` counters as a JSON string.
    fn stats(&self) -> String {
        self.handle.stats()
    }

    /// Stop accepting, mark open streams closed, and let in-flight responses
    /// finish for up to the grace period. Blocks with the GIL released, so
    /// application calls can finish meanwhile. Idempotent.
    fn shutdown(&self, py: Python<'_>) {
        py.allow_threads(|| self.handle.shutdown());
    }

    /// Shut down (if needed) and release the runtime. Idempotent.
    fn close(&self, py: Python<'_>) {
        py.allow_threads(|| self.handle.close());
    }

    /// Block until the server stops, or `timeout` seconds pass; returns whether
    /// it stopped. Signal handlers run in between, so Ctrl-C still works.
    #[pyo3(signature = (timeout = None))]
    fn wait(&self, py: Python<'_>, timeout: Option<f64>) -> PyResult<bool> {
        let deadline = match timeout {
            Some(t) if !(t.is_finite() && t >= 0.0) => {
                return Err(PyValueError::new_err("timeout must be a non-negative number of seconds"))
            }
            Some(t) => Some(std::time::Instant::now() + Duration::from_secs_f64(t)),
            None => None,
        };
        loop {
            let slice = match deadline {
                Some(deadline) => {
                    let left = deadline.saturating_duration_since(std::time::Instant::now());
                    left.min(SIGNAL_CHECK_INTERVAL)
                }
                None => SIGNAL_CHECK_INTERVAL,
            };
            if py.allow_threads(|| self.handle.join_timeout(slice)) {
                return Ok(true);
            }
            py.check_signals()?;
            if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
                return Ok(false);
            }
        }
    }
}

impl Drop for BackendServer {
    fn drop(&mut self) {
        // Dropping a ServerHandle closes it, which waits for the shutdown
        // grace period. Application threads may be waiting for the GIL to
        // finish their calls, so release it while waiting; otherwise they
        // could not run and the wait would always last the whole grace.
        Python::with_gil(|py| py.allow_threads(|| self.handle.close()));
    }
}

/// The producer half of one `/v1/chat/completions` response, handed to
/// `serve._NativeApp.handle_stream` as the `out` of `drive_chat_stream`.
///
/// Pushing is a memcpy into a bounded channel, with no syscall and no GIL
/// release; only a full channel (a slow client) waits, with the GIL released.
#[pyclass(frozen, name = "StreamSink", module = "inferencekey._inferencekey")]
pub struct StreamSink {
    inner: Sink,
}

impl StreamSink {
    /// Finish a send: wait (GIL released) only when the channel was full.
    fn complete(&self, py: Python<'_>, outcome: SendOutcome) -> bool {
        match outcome {
            SendOutcome::Sent => true,
            SendOutcome::Closed => false,
            SendOutcome::Full(pending) => py.allow_threads(|| self.inner.wait_send(pending)),
        }
    }
}

#[pymethods]
impl StreamSink {
    /// Answer with a plain JSON response instead of a stream (only before
    /// `start`). Returns whether the reply was taken.
    fn reply(&self, status: u16, body: &[u8]) -> bool {
        self.inner.reply(status, Bytes::copy_from_slice(body))
    }

    /// Send the 200 `text/event-stream` head. Returns whether it was taken.
    fn start(&self) -> bool {
        self.inner.start()
    }

    /// Queue one `data: <payload>` event. `False` once the client is gone, the
    /// stream fell too far behind, or the server is shutting down.
    fn push(&self, py: Python<'_>, payload: &[u8]) -> bool {
        let outcome = self.inner.try_push(payload);
        self.complete(py, outcome)
    }

    /// Queue several events at once (one channel item).
    fn push_many(&self, py: Python<'_>, payloads: Vec<Bound<'_, PyBytes>>) -> bool {
        let outcome = self.inner.try_push_many(payloads.iter().map(|p| p.as_bytes()));
        self.complete(py, outcome)
    }

    /// Queue the `: keepalive` comment; same result as `push`.
    fn keepalive(&self, py: Python<'_>) -> bool {
        let outcome = self.inner.try_keepalive();
        self.complete(py, outcome)
    }

    /// Queue `data: [DONE]` and end the stream.
    fn done(&self, py: Python<'_>) {
        let outcome = self.inner.try_done();
        self.complete(py, outcome);
        self.inner.finish();
    }

    /// Whether producing more is pointless (client gone, stream closed, or
    /// server shutting down).
    #[getter]
    fn closed(&self) -> bool {
        self.inner.is_closed()
    }
}

/// Register the native transport in the extension module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<BackendServer>()?;
    m.add_class::<StreamSink>()?;
    m.add("BACKEND_SERVER_ABI", BACKEND_SERVER_ABI)?;
    Ok(())
}
