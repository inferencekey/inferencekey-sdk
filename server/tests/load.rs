//! Concurrency at scale (DESIGN.md 9.1 items 10 and 12).

mod common;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use common::*;
use inferencekey_backend_server::{JsonReply, ServerConfig};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

fn client_runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_multi_thread().worker_threads(4).enable_all().build().unwrap()
}

/// Open `n` connections at once, send `request` on each, read each to EOF.
/// Returns (responses that start with `expect_prefix`, error descriptions).
fn burst(addr: std::net::SocketAddr, n: usize, request: &'static [u8], expect_prefix: &'static [u8]) -> (usize, Vec<String>) {
    client_runtime().block_on(async move {
        let start = Arc::new(tokio::sync::Barrier::new(n));
        let mut tasks = Vec::with_capacity(n);
        for _ in 0..n {
            let start = start.clone();
            tasks.push(tokio::spawn(async move {
                start.wait().await;
                let mut stream = tokio::net::TcpStream::connect(addr).await.map_err(|e| format!("connect: {e}"))?;
                stream.write_all(request).await.map_err(|e| format!("write: {e}"))?;
                let mut raw = Vec::new();
                tokio::time::timeout(Duration::from_secs(60), stream.read_to_end(&mut raw))
                    .await
                    .map_err(|_| "timeout".to_owned())?
                    .map_err(|e| format!("read: {e}"))?;
                Ok::<Vec<u8>, String>(raw)
            }));
        }
        let mut ok = 0;
        let mut errors = Vec::new();
        for task in tasks {
            match task.await.unwrap() {
                Ok(raw) if raw.starts_with(expect_prefix) => ok += 1,
                Ok(raw) => errors.push(format!("unexpected: {:?}", String::from_utf8_lossy(&raw[..raw.len().min(80)]))),
                Err(error) => errors.push(error),
            }
        }
        (ok, errors)
    })
}

#[test]
fn burst_of_2000_connects_has_no_resets() {
    let server = serve(FakeApp::echo());
    server.set_ready(true);
    let (ok, errors) = burst(
        server.local_addr(),
        2000,
        b"GET /healthz HTTP/1.1\r\nHost: t\r\nConnection: close\r\n\r\n",
        b"HTTP/1.1 200 OK",
    );
    assert!(errors.is_empty(), "{} errors, first: {:?}", errors.len(), errors.first());
    assert_eq!(ok, 2000);
    assert_eq!(stat(&server, "accept_errors"), 0);
}

#[test]
fn six_hundred_streams_run_at_once_past_tokios_default_blocking_cap() {
    const STREAMS: usize = 600;
    let arrived = Arc::new((Mutex::new(0usize), Condvar::new()));
    let peak = Arc::new(AtomicUsize::new(0));
    let (app_arrived, app_peak) = (arrived.clone(), peak.clone());
    let app = FakeApp::new(
        |_route, _body| JsonReply::new(200, "{}"),
        move |_body, sink| {
            let (lock, cv) = &*app_arrived;
            let mut count = lock.lock().unwrap();
            *count += 1;
            app_peak.fetch_max(*count, Ordering::Relaxed);
            cv.notify_all();
            // Every call waits until all of them run concurrently; with
            // tokio's default of 512 blocking threads the rest would queue
            // and this would time out.
            let (count, timeout) = cv.wait_timeout_while(count, Duration::from_secs(30), |c| *c < STREAMS).unwrap();
            drop(count);
            if timeout.timed_out() {
                sink.reply(500, r#"{"error": "not all streams ran at once"}"#);
                return;
            }
            sink.start();
            sink.push(b"{\"ok\":true}");
            sink.done();
        },
    );
    let server = serve_with(ServerConfig { max_inflight: 1024, ..config() }, app);
    let (ok, errors) = burst(
        server.local_addr(),
        STREAMS,
        b"POST /v1/chat/completions HTTP/1.1\r\nHost: t\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}",
        b"HTTP/1.1 200 OK",
    );
    assert!(errors.is_empty(), "{} errors, first: {:?}", errors.len(), errors.first());
    assert_eq!(ok, STREAMS);
    assert_eq!(peak.load(Ordering::Relaxed), STREAMS);
}
