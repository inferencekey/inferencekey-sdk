//! Native HTTP/SSE transport for InferenceKey custom backends.
//!
//! Rust decides how bytes move; the application (in production the Python
//! `inferencekey.backend.serve` functions) decides what a request means.
//! This crate accepts connections on `127.0.0.1`, parses HTTP/1, answers
//! `/healthz`, `/meta`, `/v1/models` and `/debug/stats` itself, enforces the
//! body rules, and writes every byte of every response, coalescing SSE events
//! into as few writes as the client's pace allows. POST requests go to an
//! [`AppHandler`] on a blocking pool, one thread per in-flight request.
//!
//! ```no_run
//! use std::sync::Arc;
//! use inferencekey_backend_server::{start, AppHandler, JsonReply, Route, ServerConfig, StreamSink};
//! use bytes::Bytes;
//!
//! struct Echo;
//! impl AppHandler for Echo {
//!     fn call_json(&self, _route: Route, body: Bytes) -> JsonReply { JsonReply::new(200, body) }
//!     fn call_stream(&self, _body: Bytes, sink: StreamSink) {
//!         sink.start();
//!         sink.push(br#"{"choices":[]}"#);
//!         sink.done();
//!     }
//! }
//!
//! let server = start(ServerConfig { port: 0, ..ServerConfig::default() }, Arc::new(Echo)).unwrap();
//! server.set_manifest(br#"{"name": "echo"}"#);
//! server.set_ready(true);
//! server.join();
//! ```

mod app;
mod body;
mod conn;
mod errors;
mod listener;
mod router;
mod sse;
mod stats;

use std::io;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Duration;

use bytes::Bytes;
use tokio::runtime::Runtime;
use tokio::sync::{watch, Semaphore};

pub use app::{AppHandler, JsonReply, Route};
pub use errors::{internal_error, ErrorPage};
pub use sse::{PendingFrame, SendOutcome, StreamSink, CHANNEL_CAPACITY, COALESCE_CAP};

use conn::Stopped;
use stats::Stats;

/// Thread name prefix of the IO (tokio worker) threads.
pub const IO_THREAD_PREFIX: &str = "ik-http-io";
/// Thread name of the blocking-pool threads that run application calls.
pub const REQUEST_THREAD_NAME: &str = "ik-backend-req";

/// Default [`ServerConfig::shutdown_grace`]. The worker sends SIGTERM and
/// escalates to SIGKILL after 5 s, so drain plus runtime teardown must fit
/// well inside that, or the backend's own cleanup never runs.
pub const DEFAULT_SHUTDOWN_GRACE: Duration = Duration::from_secs(2);

/// Server settings. [`ServerConfig::default`] gives the production defaults.
#[derive(Clone, Debug)]
pub struct ServerConfig {
    /// Port on 127.0.0.1; 0 picks a free one.
    pub port: u16,
    /// Listen backlog (capped by the kernel's `net.core.somaxconn`).
    pub backlog: u32,
    /// Tokio IO worker threads.
    pub io_threads: usize,
    /// Concurrent application calls; excess requests wait for a permit.
    pub max_inflight: usize,
    /// Largest accepted request body.
    pub max_body: usize,
    /// Queued bytes per stream before it is closed as a slow client.
    pub stream_max_bytes: usize,
    /// Close-delimited SSE framing for every client, not only HTTP/1.0 ones.
    pub sse_connection_close: bool,
    /// How long shutdown lets in-flight responses finish. Keep the whole stop
    /// well under the worker's SIGTERM budget (5 s before SIGKILL).
    pub shutdown_grace: Duration,
    /// How long a client may take to send its request head.
    pub header_read_timeout: Duration,
    /// The 501 page for methods other than GET and POST.
    pub unsupported_method_page: ErrorPage,
}

impl ServerConfig {
    /// `min(4, available CPUs)`.
    pub fn default_io_threads() -> usize {
        std::thread::available_parallelism().map_or(1, |n| n.get()).min(4)
    }
}

impl Default for ServerConfig {
    fn default() -> Self {
        ServerConfig {
            port: 0,
            backlog: 4096,
            io_threads: Self::default_io_threads(),
            max_inflight: 1024,
            max_body: 16 * 1024 * 1024,
            stream_max_bytes: 8 * 1024 * 1024,
            sse_connection_close: false,
            shutdown_grace: DEFAULT_SHUTDOWN_GRACE,
            header_read_timeout: Duration::from_secs(30),
            unsupported_method_page: ErrorPage::default(),
        }
    }
}

/// Cached `/meta` and `/v1/models` bodies.
#[derive(Clone)]
pub(crate) struct Manifest {
    pub meta: Bytes,
    pub models: Bytes,
}

impl Manifest {
    fn from_bytes(meta: &[u8]) -> Self {
        Manifest { meta: Bytes::copy_from_slice(meta), models: errors::models_body(meta) }
    }
}

/// State shared by the accept loop, every connection and the handle.
pub(crate) struct Shared {
    pub cfg: ServerConfig,
    pub app: Arc<dyn AppHandler>,
    pub ready: AtomicBool,
    manifest: RwLock<Manifest>,
    pub stats: Arc<Stats>,
    pub admission: Arc<Semaphore>,
    /// Set when shutdown starts; every sink reports closed from then on.
    pub closing: Arc<AtomicBool>,
}

impl Shared {
    pub fn manifest(&self) -> Manifest {
        self.manifest.read().unwrap_or_else(|p| p.into_inner()).clone()
    }
}

/// A running server. Dropping it closes it.
pub struct ServerHandle {
    local_addr: SocketAddr,
    shared: Arc<Shared>,
    shutdown_tx: watch::Sender<bool>,
    stopped: Arc<Stopped>,
    runtime: Mutex<Option<Runtime>>,
}

/// Bind and start serving. The socket is bound (and listening) when this
/// returns, so a caller can run slow setup afterwards while `/healthz`
/// answers 503.
///
/// Builds a dedicated multi-thread runtime: `io_threads` workers named
/// `ik-http-io-N`, and a blocking pool of up to `max_inflight + 16` threads
/// (tokio's default cap of 512 would silently queue streams beyond it) whose
/// OS-level name becomes `ik-backend-req` once they run an application call.
pub fn start(cfg: ServerConfig, app: Arc<dyn AppHandler>) -> io::Result<ServerHandle> {
    let io_threads = cfg.io_threads.max(1);
    let max_inflight = cfg.max_inflight.max(1);
    // tokio names every runtime thread through one function, workers and
    // blocking threads alike; blocking threads rename themselves to
    // REQUEST_THREAD_NAME when they run an application call (see
    // `name_request_thread`).
    let spawned = AtomicUsize::new(0);
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(io_threads)
        .max_blocking_threads(max_inflight + 16)
        .thread_name_fn(move || format!("{IO_THREAD_PREFIX}-{}", spawned.fetch_add(1, Ordering::Relaxed)))
        .thread_keep_alive(Duration::from_secs(30))
        .enable_io()
        .enable_time()
        .build()?;

    let listener = {
        let _context = runtime.enter();
        listener::bind(cfg.port, cfg.backlog)?
    };
    let local_addr = listener.local_addr()?;

    let shared = Arc::new(Shared {
        admission: Arc::new(Semaphore::new(max_inflight)),
        cfg: ServerConfig { io_threads, max_inflight, ..cfg },
        app,
        ready: AtomicBool::new(false),
        manifest: RwLock::new(Manifest::from_bytes(b"{}")),
        stats: Arc::new(Stats::default()),
        closing: Arc::new(AtomicBool::new(false)),
    });
    let (shutdown_tx, shutdown_rx) = watch::channel(false);
    let stopped = Arc::new(Stopped::default());
    runtime.spawn(conn::serve(listener, shared.clone(), shutdown_rx, stopped.clone()));

    Ok(ServerHandle { local_addr, shared, shutdown_tx, stopped, runtime: Mutex::new(Some(runtime)) })
}

impl ServerHandle {
    /// The bound address (the real port when 0 was requested).
    pub fn local_addr(&self) -> SocketAddr {
        self.local_addr
    }

    /// Flip `/healthz` between 503 `loading` and 200 `ok`.
    pub fn set_ready(&self, ready: bool) {
        self.shared.ready.store(ready, Ordering::Release);
    }

    pub fn is_ready(&self) -> bool {
        self.shared.ready.load(Ordering::Acquire)
    }

    /// Set the manifest JSON served verbatim at `/meta`; `/v1/models` is
    /// derived from its `name`.
    pub fn set_manifest(&self, manifest: &[u8]) {
        *self.shared.manifest.write().unwrap_or_else(|p| p.into_inner()) = Manifest::from_bytes(manifest);
    }

    /// The `/debug/stats` counters as a JSON object.
    pub fn stats(&self) -> String {
        self.shared.stats.to_json()
    }

    /// Stop accepting, mark every open stream closed, and wait for in-flight
    /// responses to finish (up to the grace period, after which the remaining
    /// connections are aborted). Blocks; idempotent.
    pub fn shutdown(&self) {
        self.shared.closing.store(true, Ordering::Release);
        self.shared.admission.close();
        let _ = self.shutdown_tx.send(true);
        self.stopped.wait(Some(self.shared.cfg.shutdown_grace + Duration::from_secs(2)));
    }

    /// Block until the server has stopped (after a [`ServerHandle::shutdown`]
    /// from another thread).
    pub fn join(&self) {
        self.stopped.wait(None);
    }

    /// [`ServerHandle::join`] with a timeout; returns whether it stopped.
    pub fn join_timeout(&self, timeout: Duration) -> bool {
        self.stopped.wait(Some(timeout))
    }

    /// Shut down and release the runtime. Application calls still running
    /// after the grace period are abandoned (their threads finish on their
    /// own, and are not waited for). Idempotent.
    pub fn close(&self) {
        self.shutdown();
        let runtime = self.runtime.lock().unwrap_or_else(|p| p.into_inner()).take();
        if let Some(runtime) = runtime {
            if self.stopped.wait(Some(Duration::ZERO)) || tokio::runtime::Handle::try_current().is_ok() {
                // Stopped: the listener and every connection are gone, so the
                // only threads left are stuck application calls, which would
                // be abandoned anyway; do not wait for them. Inside another
                // runtime, dropping this one would panic.
                runtime.shutdown_background();
            } else {
                runtime.shutdown_timeout(Duration::from_secs(1));
            }
        }
    }
}

/// Give the current blocking-pool thread its OS-level name (what `ps`,
/// `pidstat -t` and `/proc/<pid>/task/*/comm` show), once per thread.
pub(crate) fn name_request_thread() {
    thread_local! {
        static NAMED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    }
    NAMED.with(|named| {
        if !named.replace(true) {
            set_os_thread_name(REQUEST_THREAD_NAME);
        }
    });
}

#[cfg(target_os = "linux")]
fn set_os_thread_name(name: &str) {
    let mut buf = [0u8; 16];
    let len = name.len().min(15);
    buf[..len].copy_from_slice(&name.as_bytes()[..len]);
    // SAFETY: PR_SET_NAME reads a NUL-terminated string of at most 16 bytes,
    // which `buf` is.
    unsafe {
        libc::prctl(libc::PR_SET_NAME, buf.as_ptr() as libc::c_ulong, 0, 0, 0);
    }
}

#[cfg(not(target_os = "linux"))]
fn set_os_thread_name(_name: &str) {}

impl Drop for ServerHandle {
    fn drop(&mut self) {
        self.close();
    }
}
