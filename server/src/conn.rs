//! The accept loop, per-connection HTTP/1 serving and graceful shutdown.

use std::sync::atomic::Ordering;
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper_util::rt::{TokioIo, TokioTimer};
use hyper_util::server::graceful::GracefulShutdown;
use tokio::net::TcpListener;
use tokio::sync::watch;
use tokio::task::JoinSet;

use crate::stats::{self, GaugeGuard};
use crate::{router, Shared};

/// A one-way "the server has stopped" flag that blocking threads can wait on.
#[derive(Default)]
pub(crate) struct Stopped {
    done: Mutex<bool>,
    cv: Condvar,
}

impl Stopped {
    pub fn set(&self) {
        *self.done.lock().unwrap_or_else(|p| p.into_inner()) = true;
        self.cv.notify_all();
    }

    /// Wait until stopped; `None` waits forever. Returns whether it stopped.
    pub fn wait(&self, timeout: Option<Duration>) -> bool {
        let guard = self.done.lock().unwrap_or_else(|p| p.into_inner());
        match timeout {
            None => *self.cv.wait_while(guard, |done| !*done).unwrap_or_else(|p| p.into_inner()),
            Some(timeout) => {
                let (guard, _) = self
                    .cv
                    .wait_timeout_while(guard, timeout, |done| !*done)
                    .unwrap_or_else(|p| p.into_inner());
                *guard
            }
        }
    }
}

/// Accept and serve connections until `shutdown` fires, then drain.
///
/// On shutdown: stop accepting and release the listener, flip every open
/// stream closed (pushes return `false`, so producers end their streams),
/// let in-flight responses finish for up to `shutdown_grace`, then abort the
/// connections that remain.
pub(crate) async fn serve(
    listener: TcpListener,
    shared: Arc<Shared>,
    mut shutdown: watch::Receiver<bool>,
    stopped: Arc<Stopped>,
) {
    let mut builder = http1::Builder::new();
    builder
        .keep_alive(true)
        .timer(TokioTimer::new())
        .header_read_timeout(shared.cfg.header_read_timeout);
    let graceful = GracefulShutdown::new();
    let mut connections = JoinSet::new();

    loop {
        tokio::select! {
            biased;
            _ = shutdown.changed() => break,
            Some(_) = connections.join_next(), if !connections.is_empty() => {}
            accepted = listener.accept() => match accepted {
                Ok((stream, _peer)) => {
                    // Small SSE writes must not wait for delayed ACKs.
                    let _ = stream.set_nodelay(true);
                    stats::inc(&shared.stats.connections_total);
                    let service_state = shared.clone();
                    let service = service_fn(move |req| router::handle(req, service_state.clone()));
                    let connection = graceful.watch(builder.serve_connection(TokioIo::new(stream), service));
                    let conn_state = shared.clone();
                    connections.spawn(async move {
                        let _open = GaugeGuard::new(&conn_state.stats.connections);
                        if let Err(error) = connection.await {
                            tracing::debug!(%error, "connection ended with an error");
                        }
                    });
                }
                Err(error) => {
                    // EMFILE and friends: back off instead of spinning.
                    stats::inc(&shared.stats.accept_errors);
                    tracing::warn!(%error, "accept failed");
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
            }
        }
    }

    drop(listener);
    shared.closing.store(true, Ordering::Release);
    if tokio::time::timeout(shared.cfg.shutdown_grace, graceful.shutdown()).await.is_err() {
        tracing::warn!("shutdown grace period expired; aborting open connections");
    }
    connections.shutdown().await;
    stopped.set();
}
