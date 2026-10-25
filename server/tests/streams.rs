//! Event streams over real sockets (DESIGN.md 9.1 items 6 to 9 and 14).

mod common;

use std::io::{Read, Write};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc};
use std::time::{Duration, Instant};

use common::*;
use inferencekey_backend_server::{JsonReply, ServerConfig, StreamSink};
use inferencekey_core::domain::sse::{format_data_event, DONE_EVENT, KEEPALIVE_EVENT};

const CHAT_REQUEST: &[u8] =
    b"POST /v1/chat/completions HTTP/1.1\r\nHost: t\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}";

fn stream_app(stream: impl Fn(StreamSink) + Send + Sync + 'static) -> FakeApp {
    FakeApp::new(|_route, _body| JsonReply::new(200, "{}"), move |_body, sink| stream(sink))
}

#[test]
fn start_sends_the_event_stream_head() {
    let server = serve(FakeApp::echo());
    let response = exchange(server.local_addr(), CHAT_REQUEST);
    assert_eq!(response.status, 200);
    assert_eq!(response.status_line, "HTTP/1.1 200 OK");
    assert_eq!(response.header("content-type"), Some("text/event-stream"));
    assert_eq!(response.header("cache-control"), Some("no-cache"));
    assert_eq!(response.header("connection"), Some("close"));
    assert!(response.header("content-length").is_none());
    assert!(response.is_chunked(), "HTTP/1.1 clients get chunked framing");
    assert_eq!(response.text(), "data: {\"n\":1}\n\ndata: {\"n\":2}\n\ndata: [DONE]\n\n");
}

#[test]
fn reply_before_start_is_a_plain_json_response() {
    let server = serve(stream_app(|sink| {
        sink.reply(400, r#"{"error": {"message": "messages must not be empty", "type": "invalid_request_error"}}"#);
    }));
    let response = exchange(server.local_addr(), CHAT_REQUEST);
    assert_eq!(response.status, 400);
    assert_eq!(response.header("content-type"), Some("application/json"));
    assert!(response.header("cache-control").is_none());
    assert!(!response.is_chunked());
    assert_eq!(
        response.text(),
        r#"{"error": {"message": "messages must not be empty", "type": "invalid_request_error"}}"#
    );
}

#[test]
fn coalesced_stream_keeps_bytes_and_order_including_keepalives() {
    let server = serve(stream_app(|sink| {
        sink.start();
        for i in 0..1000 {
            assert!(sink.push(format!("{{\"n\":{i},\"text\":\"caf\u{e9}\"}}").as_bytes()));
            if i % 10 == 0 {
                assert!(sink.keepalive());
            }
        }
        sink.done();
    }));
    let response = exchange(server.local_addr(), CHAT_REQUEST);
    let mut expected = Vec::new();
    for i in 0..1000 {
        format_data_event(format!("{{\"n\":{i},\"text\":\"caf\u{e9}\"}}").as_bytes(), &mut expected);
        if i % 10 == 0 {
            expected.extend_from_slice(KEEPALIVE_EVENT);
        }
    }
    expected.extend_from_slice(DONE_EVENT);
    assert_eq!(response.body, expected);

    let frames = stat(&server, "frames_written");
    let chunks = stat(&server, "chunks_pushed");
    assert_eq!(chunks, 1000);
    assert_eq!(stat(&server, "keepalives_pushed"), 100);
    assert!(frames <= chunks + 101, "never more frames than events");
    println!("coalescing over HTTP with a reading client: {frames} frames for 1101 events");
    assert!(wait_until(Duration::from_secs(2), || stat(&server, "open_streams") == 0));
    assert_eq!(stat(&server, "cancelled"), 0);
}

/// The client reads the first event, then drops the socket. Returns how long
/// after the drop the producer saw `false`, and how many sends it took.
fn disconnect_detection(keepalive_only: bool) -> (Duration, u32) {
    let (tx, rx) = mpsc::channel::<(Instant, u32)>();
    let tx = std::sync::Mutex::new(tx);
    let server = serve(stream_app(move |sink| {
        sink.start();
        let first = if keepalive_only { sink.keepalive() } else { sink.push(b"{\"n\":0}") };
        assert!(first);
        let mut sends = 0;
        loop {
            std::thread::sleep(Duration::from_millis(10));
            sends += 1;
            let alive = if keepalive_only { sink.keepalive() } else { sink.push(b"{\"n\":1}") };
            if !alive {
                break;
            }
            assert!(sends < 2000, "never noticed the disconnect");
        }
        tx.lock().unwrap().send((Instant::now(), sends)).unwrap();
    }));
    let mut stream = connect(server.local_addr());
    stream.write_all(CHAT_REQUEST).unwrap();
    let mut seen = Vec::new();
    let mut buf = [0u8; 4096];
    let marker: &[u8] = if keepalive_only { b": keepalive" } else { b"data: " };
    while !seen.windows(marker.len()).any(|w| w == marker) {
        let n = stream.read(&mut buf).unwrap();
        assert!(n > 0);
        seen.extend_from_slice(&buf[..n]);
    }
    drop(stream);
    let dropped = Instant::now();
    let (noticed, sends_after_drop) = rx.recv_timeout(Duration::from_secs(10)).expect("producer stopped");
    assert!(wait_until(Duration::from_secs(2), || stat(&server, "cancelled") == 1));
    (noticed.saturating_duration_since(dropped), sends_after_drop)
}

#[test]
fn push_returns_false_soon_after_the_client_drops() {
    let (after, sends) = disconnect_detection(false);
    println!("push noticed the disconnect after {after:?} ({sends} sends)");
    assert!(after < Duration::from_secs(1), "{after:?}");
    // Within a write or two (RST, then EPIPE on loopback), not just "soon".
    assert!(sends <= 3, "{sends} sends before the disconnect was noticed");
}

#[test]
fn keepalive_returns_false_soon_after_the_client_drops() {
    let (after, sends) = disconnect_detection(true);
    println!("keepalive noticed the disconnect after {after:?} ({sends} sends)");
    assert!(after < Duration::from_secs(1), "{after:?}");
    // Within a write or two (RST, then EPIPE on loopback), not just "soon".
    assert!(sends <= 3, "{sends} sends before the disconnect was noticed");
}

#[test]
fn non_reading_client_blocks_the_producer_at_the_channel_bound() {
    let pushed = Arc::new(AtomicUsize::new(0));
    let finished = Arc::new(AtomicU64::new(0));
    let (app_pushed, app_finished) = (pushed.clone(), finished.clone());
    let payload = vec![b'x'; 1024];
    let server = serve(stream_app(move |sink| {
        sink.start();
        while sink.push(&payload) {
            app_pushed.fetch_add(1, Ordering::Relaxed);
        }
        app_finished.store(1, Ordering::Release);
    }));
    let mut stream = connect(server.local_addr());
    stream.write_all(CHAT_REQUEST).unwrap();

    // Wait for the count to stop moving: the producer is blocked.
    let mut last = usize::MAX;
    assert!(wait_until(Duration::from_secs(10), || {
        std::thread::sleep(Duration::from_millis(300));
        let now = pushed.load(Ordering::Relaxed);
        let stalled = now == last && now > 0;
        last = now;
        stalled
    }));
    let stalled_at = pushed.load(Ordering::Relaxed);
    println!("producer blocked after {stalled_at} KiB pushed to a client that never reads");
    assert_eq!(finished.load(Ordering::Acquire), 0, "blocked, not closed");
    assert!(stalled_at < 64 * 1024, "memory bounded: {stalled_at} KiB in flight");
    assert!(stat(&server, "slow_client_closes") == 0, "1 KiB events never reach the 8 MiB bound");

    // The client leaves: the blocked push returns false.
    drop(stream);
    assert!(wait_until(Duration::from_secs(5), || finished.load(Ordering::Acquire) == 1));
}

#[test]
fn exceeding_stream_max_bytes_closes_the_stream() {
    let pushed_bytes = Arc::new(AtomicUsize::new(0));
    let (tx, rx) = mpsc::channel::<()>();
    let tx = std::sync::Mutex::new(tx);
    let app_bytes = pushed_bytes.clone();
    let payload = vec![b'y'; 32 * 1024];
    let cfg = ServerConfig { stream_max_bytes: 256 * 1024, ..config() };
    let server = serve_with(cfg, stream_app(move |sink| {
        sink.start();
        while sink.push(&payload) {
            app_bytes.fetch_add(payload.len(), Ordering::Relaxed);
        }
        tx.lock().unwrap().send(()).unwrap();
    }));
    let mut stream = connect(server.local_addr());
    stream.write_all(CHAT_REQUEST).unwrap();
    rx.recv_timeout(Duration::from_secs(10)).expect("the bound closes the stream instead of blocking");
    let total = pushed_bytes.load(Ordering::Relaxed);
    println!("slow client closed after {} KiB pushed", total / 1024);
    assert!(total < 64 * 1024 * 1024);
    assert_eq!(stat(&server, "slow_client_closes"), 1);

    // The connection is aborted rather than drained: the client sees EOF
    // (or a reset) without a terminating [DONE].
    let mut rest = Vec::new();
    let _ = stream.read_to_end(&mut rest);
    assert!(!rest.ends_with(b"data: [DONE]\n\n"));
}

fn expected_echo_events() -> &'static [u8] {
    b"data: {\"n\":1}\n\ndata: {\"n\":2}\n\ndata: [DONE]\n\n"
}

#[test]
fn http_10_request_gets_a_close_delimited_stream() {
    let server = serve(FakeApp::echo());
    let response = exchange(
        server.local_addr(),
        b"POST /v1/chat/completions HTTP/1.0\r\nHost: t\r\nContent-Length: 2\r\n\r\n{}",
    );
    assert_eq!(response.status_line, "HTTP/1.0 200 OK");
    assert!(!response.is_chunked());
    assert!(response.header("content-length").is_none());
    assert_eq!(response.raw_body, expected_echo_events());
}

#[test]
fn sse_connection_close_makes_http_11_streams_close_delimited() {
    let server = serve_with(ServerConfig { sse_connection_close: true, ..config() }, FakeApp::echo());
    let response = exchange(server.local_addr(), CHAT_REQUEST);
    assert_eq!(response.status_line, "HTTP/1.0 200 OK");
    assert!(!response.is_chunked());
    assert_eq!(response.header("connection"), Some("close"));
    assert_eq!(response.raw_body, expected_echo_events());
    // JSON routes keep HTTP/1.1.
    assert_eq!(post(server.local_addr(), "/process", b"{}").status_line, "HTTP/1.1 200 OK");
}

#[test]
fn stream_handler_returning_without_done_ends_the_body() {
    let server = serve(stream_app(|sink| {
        sink.start();
        sink.push(b"{\"n\":1}");
    }));
    let response = exchange(server.local_addr(), CHAT_REQUEST);
    assert_eq!(response.status, 200);
    assert_eq!(response.text(), "data: {\"n\":1}\n\n");
}

#[test]
fn a_sink_kept_by_the_app_does_not_hold_the_stream_open() {
    let kept = Arc::new(std::sync::Mutex::new(Vec::<StreamSink>::new()));
    let app_kept = kept.clone();
    let server = serve(stream_app(move |sink| {
        sink.start();
        sink.push(b"{}");
        app_kept.lock().unwrap().push(sink);
    }));
    let response = exchange(server.local_addr(), CHAT_REQUEST);
    assert_eq!(response.text(), "data: {}\n\n");
    assert!(kept.lock().unwrap()[0].is_closed() || !kept.lock().unwrap()[0].push(b"{}"));
}
