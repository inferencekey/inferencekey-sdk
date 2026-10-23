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
    assert!(took >= Duration::from_millis(900) && took < Duration::from_secs(4), "{took:?}");
    let mut rest = Vec::new();
    let _ = stream.read_to_end(&mut rest);
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
