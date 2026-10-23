//! Graceful shutdown (DESIGN.md 9.1 item 13).

mod common;

use std::io::{Read, Write};
use std::sync::{mpsc, Mutex};
use std::time::{Duration, Instant};

use common::*;
use inferencekey_backend_server::{JsonReply, ServerConfig};

const CHAT_REQUEST: &[u8] =
    b"POST /v1/chat/completions HTTP/1.1\r\nHost: t\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}";

#[test]
fn shutdown_flips_open_sinks_closed_and_exits_within_the_grace() {
    let (tx, rx) = mpsc::channel::<Instant>();
    let tx = Mutex::new(tx);
    let app = FakeApp::new(
        |_route, _body| JsonReply::new(200, "{}"),
        move |_body, sink| {
            sink.start();
            while sink.keepalive() {
                std::thread::sleep(Duration::from_millis(20));
            }
            tx.lock().unwrap().send(Instant::now()).unwrap();
        },
    );
    let server = serve_with(ServerConfig { shutdown_grace: Duration::from_secs(2), ..config() }, app);
    let addr = server.local_addr();
    let mut stream = connect(addr);
    stream.write_all(CHAT_REQUEST).unwrap();
    let mut buf = [0u8; 256];
    assert!(stream.read(&mut buf).unwrap() > 0, "stream started");

    let started = Instant::now();
    server.shutdown();
    let took = started.elapsed();
    let noticed = rx.recv_timeout(Duration::from_secs(5)).expect("producer stopped");
    println!("shutdown returned after {took:?}; producer stopped after {:?}", noticed - started);
    assert!(noticed - started < Duration::from_millis(500));
    assert!(took < Duration::from_secs(2), "the stream ended, so no need to wait for the grace: {took:?}");

    // The open stream ended, and nothing accepts anymore.
    let mut rest = Vec::new();
    stream.read_to_end(&mut rest).unwrap();
    assert!(std::net::TcpStream::connect(addr).is_err());
    assert!(server.join_timeout(Duration::from_millis(10)));
}

#[test]
fn a_producer_that_ignores_closed_is_aborted_after_the_grace() {
    let app = FakeApp::new(
        |_route, _body| JsonReply::new(200, "{}"),
        |_body, sink| {
            sink.start();
            // Never checks: blocked in an engine wait, for example.
            std::thread::sleep(Duration::from_secs(8));
        },
    );
    let server = serve_with(ServerConfig { shutdown_grace: Duration::from_secs(1), ..config() }, app);
    let mut stream = connect(server.local_addr());
    stream.write_all(CHAT_REQUEST).unwrap();
    let mut buf = [0u8; 256];
    assert!(stream.read(&mut buf).unwrap() > 0);

    let started = Instant::now();
    server.close();
    let took = started.elapsed();
    println!("close with a stuck producer took {took:?}");
    // The grace, and no extra wait for the stuck blocking thread.
    assert!(took >= Duration::from_millis(900) && took < Duration::from_millis(1600), "{took:?}");
    let mut rest = Vec::new();
    let _ = stream.read_to_end(&mut rest);
}

#[test]
fn a_stuck_json_call_does_not_hold_close_past_the_default_grace() {
    // The worker sends SIGTERM and kills after 5 s; a sleeping /process must
    // not keep close() (drain plus runtime teardown) anywhere near that.
    assert_eq!(ServerConfig::default().shutdown_grace, Duration::from_secs(2));
    let app = FakeApp::new(
        |_route, _body| {
            std::thread::sleep(Duration::from_secs(30));
            JsonReply::new(200, "{}")
        },
        |_body, sink| {
            sink.reply(200, &b"{}"[..]);
        },
    );
    let server = serve_with(ServerConfig { shutdown_grace: ServerConfig::default().shutdown_grace, ..config() }, app);
    let mut stream = connect(server.local_addr());
    stream.write_all(b"POST /process HTTP/1.1\r\nHost: t\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}").unwrap();
    assert!(wait_until(Duration::from_secs(2), || stat(&server, "inflight") == 1), "call started");

    let started = Instant::now();
    server.close();
    let took = started.elapsed();
    println!("close with a sleeping /process took {took:?}");
    assert!(took < Duration::from_millis(2600), "{took:?}");
}

#[test]
fn idle_keep_alive_connections_do_not_delay_shutdown() {
    let server = serve(FakeApp::echo());
    let mut stream = connect(server.local_addr());
    stream.write_all(b"GET /healthz HTTP/1.1\r\nHost: t\r\n\r\n").unwrap();
    let mut buf = [0u8; 512];
    assert!(stream.read(&mut buf).unwrap() > 0);
    let started = Instant::now();
    server.shutdown();
    assert!(started.elapsed() < Duration::from_secs(1), "{:?}", started.elapsed());
}

#[test]
fn requests_waiting_for_admission_end_at_shutdown_instead_of_after_the_grace() {
    let app = FakeApp::new(
        |_route, _body| {
            // The one permitted call is stuck for longer than the grace.
            std::thread::sleep(Duration::from_secs(4));
            JsonReply::new(200, "{}")
        },
        |_body, sink| {
            sink.reply(200, &b"{}"[..]);
        },
    );
    let cfg = ServerConfig { max_inflight: 1, shutdown_grace: Duration::from_secs(2), ..config() };
    let server = std::sync::Arc::new(serve_with(cfg, app));
    let addr = server.local_addr();
    let blocked = std::thread::spawn(move || post(addr, "/process", b"{}"));
    assert!(wait_until(Duration::from_secs(2), || stat(&server, "inflight") == 1));
    let queued = std::thread::spawn(move || {
        let started = Instant::now();
        (post(addr, "/v1/chat/completions", b"{}"), started.elapsed())
    });
    assert!(wait_until(Duration::from_secs(2), || stat(&server, "queued") == 1));

    let stopper = {
        let server = server.clone();
        std::thread::spawn(move || server.shutdown())
    };
    let (response, waited) = queued.join().unwrap();
    println!("the queued request ended after {waited:?}");
    assert_eq!(response.status, 503);
    assert_eq!(response.body, br#"{"error": {"message": "backend not ready", "type": "server_error"}}"#);
    assert!(waited < Duration::from_secs(1), "{waited:?}");
    stopper.join().unwrap();
    drop(blocked);
}
