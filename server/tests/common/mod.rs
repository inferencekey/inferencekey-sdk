//! Shared helpers for the integration tests: a configurable fake application
//! and a raw-socket HTTP client that shows exactly what is on the wire.

#![allow(dead_code)]

use std::io::{Read, Write};
use std::net::{Shutdown, SocketAddr, TcpStream};
use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use inferencekey_backend_server::{start, AppHandler, JsonReply, Route, ServerConfig, ServerHandle, StreamSink};

type JsonFn = dyn Fn(Route, Bytes) -> JsonReply + Send + Sync;
type StreamFn = dyn Fn(Bytes, StreamSink) + Send + Sync;

/// An application made of two closures.
pub struct FakeApp {
    json: Box<JsonFn>,
    stream: Box<StreamFn>,
}

impl FakeApp {
    pub fn new(
        json: impl Fn(Route, Bytes) -> JsonReply + Send + Sync + 'static,
        stream: impl Fn(Bytes, StreamSink) + Send + Sync + 'static,
    ) -> Self {
        FakeApp { json: Box::new(json), stream: Box::new(stream) }
    }

    /// JSON routes echo `{"route": ..., "len": ...}`; chat streams two events.
    pub fn echo() -> Self {
        FakeApp::new(
            |route, body| {
                JsonReply::new(200, format!("{{\"route\": \"{}\", \"len\": {}}}", route.path(), body.len()))
            },
            |_body, sink| {
                sink.start();
                sink.push(br#"{"n":1}"#);
                sink.push(br#"{"n":2}"#);
                sink.done();
            },
        )
    }
}

impl AppHandler for FakeApp {
    fn call_json(&self, route: Route, body: Bytes) -> JsonReply {
        (self.json)(route, body)
    }
    fn call_stream(&self, body: Bytes, sink: StreamSink) {
        (self.stream)(body, sink)
    }
}

pub fn config() -> ServerConfig {
    ServerConfig { port: 0, shutdown_grace: Duration::from_secs(2), ..ServerConfig::default() }
}

pub fn serve(app: FakeApp) -> ServerHandle {
    serve_with(config(), app)
}

pub fn serve_with(cfg: ServerConfig, app: FakeApp) -> ServerHandle {
    start(cfg, Arc::new(app)).expect("server starts")
}

/// One response as it came off the socket.
#[derive(Debug)]
pub struct RawResponse {
    pub status_line: String,
    pub status: u16,
    pub headers: Vec<(String, String)>,
    /// The body bytes exactly as sent (still chunk-encoded if chunked).
    pub raw_body: Vec<u8>,
    /// The body with chunked transfer encoding removed.
    pub body: Vec<u8>,
}

impl RawResponse {
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(key, _)| key.eq_ignore_ascii_case(name))
            .map(|(_, value)| value.as_str())
    }

    pub fn is_chunked(&self) -> bool {
        self.header("transfer-encoding").map_or(false, |v| v.eq_ignore_ascii_case("chunked"))
    }

    pub fn text(&self) -> &str {
        std::str::from_utf8(&self.body).expect("utf-8 body")
    }
}

pub fn connect(addr: SocketAddr) -> TcpStream {
    let stream = TcpStream::connect(addr).expect("connect");
    stream.set_read_timeout(Some(Duration::from_secs(15))).unwrap();
    stream
}

/// Send raw request bytes, read until the server closes, parse.
pub fn exchange(addr: SocketAddr, request: &[u8]) -> RawResponse {
    let mut stream = connect(addr);
    stream.write_all(request).unwrap();
    let mut raw = Vec::new();
    stream.read_to_end(&mut raw).expect("read response");
    parse(&raw).unwrap_or_else(|| panic!("unparseable response: {:?}", String::from_utf8_lossy(&raw)))
}

/// Like [`exchange`], but half-closes the request side after writing, and
/// returns `None` when the server closed without answering.
pub fn exchange_half_closed(addr: SocketAddr, request: &[u8]) -> Option<RawResponse> {
    let mut stream = connect(addr);
    stream.write_all(request).unwrap();
    let _ = stream.shutdown(Shutdown::Write);
    let mut raw = Vec::new();
    let _ = stream.read_to_end(&mut raw);
    parse(&raw)
}

pub fn get(addr: SocketAddr, path: &str) -> RawResponse {
    exchange(addr, format!("GET {path} HTTP/1.1\r\nHost: t\r\nConnection: close\r\n\r\n").as_bytes())
}

pub fn post(addr: SocketAddr, path: &str, body: &[u8]) -> RawResponse {
    let mut request = format!(
        "POST {path} HTTP/1.1\r\nHost: t\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    )
    .into_bytes();
    request.extend_from_slice(body);
    exchange(addr, &request)
}

pub fn parse(raw: &[u8]) -> Option<RawResponse> {
    let split = raw.windows(4).position(|w| w == b"\r\n\r\n")?;
    let head = std::str::from_utf8(&raw[..split]).ok()?;
    let raw_body = raw[split + 4..].to_vec();
    let mut lines = head.split("\r\n");
    let status_line = lines.next()?.to_owned();
    let status = status_line.split(' ').nth(1)?.parse().ok()?;
    let headers: Vec<(String, String)> = lines
        .filter_map(|line| line.split_once(':'))
        .map(|(k, v)| (k.trim().to_owned(), v.trim().to_owned()))
        .collect();
    let mut response = RawResponse { status_line, status, headers, body: Vec::new(), raw_body };
    response.body = if response.is_chunked() { dechunk(&response.raw_body) } else { response.raw_body.clone() };
    Some(response)
}

pub fn dechunk(mut raw: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    loop {
        let Some(eol) = raw.windows(2).position(|w| w == b"\r\n") else { break };
        let size_text = std::str::from_utf8(&raw[..eol]).unwrap();
        let size = usize::from_str_radix(size_text.split(';').next().unwrap().trim(), 16).unwrap();
        raw = &raw[eol + 2..];
        if size == 0 {
            break;
        }
        out.extend_from_slice(&raw[..size]);
        raw = &raw[size + 2..];
    }
    out
}

/// Poll `condition` every 5 ms for up to `timeout`.
pub fn wait_until(timeout: Duration, mut condition: impl FnMut() -> bool) -> bool {
    let deadline = std::time::Instant::now() + timeout;
    while std::time::Instant::now() < deadline {
        if condition() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    condition()
}

/// One counter from `/debug/stats`.
pub fn stat(server: &ServerHandle, name: &str) -> u64 {
    let value: serde_json::Value = serde_json::from_str(&server.stats()).unwrap();
    value[name].as_u64().unwrap_or_else(|| panic!("no stat {name}"))
}
