//! Routing, native endpoints, body rules, admission and the runtime shape
//! (DESIGN.md 9.1 items 2 to 5, 11 and 12).

mod common;

use std::io::{Read, Write};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Barrier, Condvar, Mutex};
use std::time::{Duration, Instant};

use bytes::Bytes;
use common::*;
use inferencekey_backend_server::{start, ErrorPage, JsonReply, Route, ServerConfig};

#[test]
fn unknown_paths_get_the_python_404_body() {
    let server = serve(FakeApp::echo());
    let addr = server.local_addr();
    for response in [get(addr, "/nope"), post(addr, "/nope", b"{}"), get(addr, "/process"), post(addr, "/healthz", b"")] {
        assert_eq!(response.status, 404);
        assert_eq!(response.text(), r#"{"error": "not found"}"#);
        assert_eq!(response.header("content-type"), Some("application/json"));
    }
    // An absolute-form target is a path that matches nothing, as in stdlib.
    let absolute = exchange(addr, b"GET http://t/healthz HTTP/1.1\r\nHost: t\r\nConnection: close\r\n\r\n");
    assert_eq!(absolute.status, 404);
}

#[test]
fn unknown_post_path_is_404_before_the_body_rules() {
    let server = serve(FakeApp::echo());
    let response = exchange(
        server.local_addr(),
        b"POST /nope HTTP/1.1\r\nHost: t\r\nContent-Length: 99999999\r\nConnection: close\r\n\r\n",
    );
    assert_eq!(response.status, 404);
}

/// A body over the limit is read and dropped before the 400, so a client that
/// sends it whole (reqwest in the worker) gets the answer instead of a reset.
#[test]
fn too_large_body_sent_whole_gets_the_400_not_a_reset() {
    let server = serve(FakeApp::echo());
    server.set_ready(true);
    let addr = server.local_addr();
    let body = vec![b'x'; 16 * 1024 * 1024 + 1];
    for route in [Route::Process, Route::Forecast, Route::Chat] {
        let mut stream = connect(addr);
        stream.set_read_timeout(Some(Duration::from_secs(30))).unwrap();
        let head = format!("POST {} HTTP/1.1\r\nHost: t\r\nContent-Length: {}\r\n\r\n", route.path(), body.len());
        stream.write_all(head.as_bytes()).expect("head written");
        stream.write_all(&body).expect("the whole body is accepted, not reset");
        let mut raw = Vec::new();
        stream.read_to_end(&mut raw).expect("the answer, then a clean close");
        let response = parse(&raw).expect("an answer");
        assert_eq!(response.status, 400, "{route:?}");
        assert_eq!(response.text(), r#"{"error": "request body too large"}"#, "{route:?}");
        assert_eq!(response.header("connection"), Some("close"), "{route:?}");
    }
}

#[test]
fn query_string_is_stripped() {
    let server = serve(FakeApp::echo());
    server.set_ready(true);
    let addr = server.local_addr();
    assert_eq!(get(addr, "/healthz?probe=1&x=y").text(), r#"{"status": "ok"}"#);
    let response = post(addr, "/process?trace=1", b"{}");
    assert_eq!(response.text(), r#"{"route": "/process", "len": 2}"#);
}

#[test]
fn leading_slashes_collapse_like_cpython() {
    let server = serve(FakeApp::echo());
    server.set_ready(true);
    let addr = server.local_addr();
    assert_eq!(get(addr, "//healthz").status, 200);
    assert_eq!(get(addr, "///healthz?x=1").status, 200);
    let response = post(addr, "//process", b"{}");
    assert_eq!(response.text(), r#"{"route": "/process", "len": 2}"#);
    // Inner double slashes stay, as in stdlib.
    assert_eq!(get(addr, "/v1//models").status, 404);
}

#[test]
fn other_methods_get_501_with_the_http_server_page() {
    let server = serve(FakeApp::echo());
    let addr = server.local_addr();
    for method in ["PUT", "DELETE", "PATCH", "OPTIONS", "TRACE", "FOO"] {
        let response = exchange(
            addr,
            format!("{method} /process HTTP/1.1\r\nHost: t\r\nContent-Length: 0\r\nConnection: close\r\n\r\n").as_bytes(),
        );
        assert_eq!(response.status, 501, "{method}");
        assert_eq!(response.header("content-type"), Some("text/html;charset=utf-8"));
        assert!(response.text().contains(&format!("<p>Message: Unsupported method ('{method}').</p>")));
    }
    // HEAD has no do_HEAD in the stdlib handler either; no body is sent.
    let head = exchange(addr, b"HEAD /healthz HTTP/1.1\r\nHost: t\r\nConnection: close\r\n\r\n");
    assert_eq!(head.status, 501);
    assert!(head.raw_body.is_empty());
}

#[test]
fn the_501_page_comes_from_the_config() {
    // What the Python binding passes on CPython 3.9 and 3.10, whose
    // DEFAULT_ERROR_MESSAGE differs from 3.11+.
    let page = ErrorPage {
        template: "<!DOCTYPE HTML PUBLIC>\n<p>Error code: 501</p>\n<p>Message: %(message)s.</p>\n\
                   <p>Error code explanation: HTTPStatus.NOT_IMPLEMENTED - Server does not support this operation.</p>\n"
            .to_owned(),
        content_type: "text/html; charset=latin-1".to_owned(),
    };
    let server = serve_with(ServerConfig { unsupported_method_page: page, ..config() }, FakeApp::echo());
    let response = exchange(server.local_addr(), b"PUT /process HTTP/1.1\r\nHost: t\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
    assert_eq!(response.status, 501);
    assert_eq!(response.header("content-type"), Some("text/html; charset=latin-1"));
    assert_eq!(
        response.text(),
        "<!DOCTYPE HTML PUBLIC>\n<p>Error code: 501</p>\n<p>Message: Unsupported method ('PUT').</p>\n\
         <p>Error code explanation: HTTPStatus.NOT_IMPLEMENTED - Server does not support this operation.</p>\n"
    );
}

#[test]
fn healthz_flips_while_every_app_call_is_blocked() {
    let gate = Arc::new(Barrier::new(3));
    let app_gate = gate.clone();
    let app = FakeApp::new(
        move |_route, _body| {
            app_gate.wait();
            JsonReply::new(200, "{}")
        },
        |_body, sink| {
            sink.reply(200, "{}");
        },
    );
    let server = serve_with(ServerConfig { max_inflight: 2, ..config() }, app);
    let addr = server.local_addr();
    let blocked: Vec<_> = (0..2).map(|_| std::thread::spawn(move || post(addr, "/process", b"{}"))).collect();
    assert!(wait_until(Duration::from_secs(5), || stat(&server, "inflight") == 2));

    let started = Instant::now();
    let loading = get(addr, "/healthz");
    assert_eq!((loading.status, loading.text()), (503, r#"{"status": "loading"}"#));
    server.set_ready(true);
    let ok = get(addr, "/healthz");
    assert_eq!((ok.status, ok.text()), (200, r#"{"status": "ok"}"#));
    assert_eq!(get(addr, "/meta").status, 200, "meta is native too");
    assert!(started.elapsed() < Duration::from_secs(1), "{:?}", started.elapsed());

    gate.wait();
    for handle in blocked {
        assert_eq!(handle.join().unwrap().status, 200);
    }
}

#[test]
fn meta_and_models_serve_the_manifest() {
    let server = serve(FakeApp::echo());
    let addr = server.local_addr();
    assert_eq!(get(addr, "/meta").text(), "{}");
    let manifest = r#"{"name": "my-llm", "version": "1.0", "api": "openai-chat"}"#;
    server.set_manifest(manifest.as_bytes());
    let meta = get(addr, "/meta");
    assert_eq!((meta.status, meta.text()), (200, manifest));
    assert_eq!(meta.header("content-type"), Some("application/json"));
    assert_eq!(meta.header("content-length"), Some(manifest.len().to_string().as_str()));
    assert_eq!(
        get(addr, "/v1/models").text(),
        r#"{"object": "list", "data": [{"id": "my-llm", "object": "model", "owned_by": "custom"}]}"#
    );
}

#[test]
fn debug_stats_is_json() {
    let server = serve(FakeApp::echo());
    let response = get(server.local_addr(), "/debug/stats");
    let value: serde_json::Value = serde_json::from_slice(&response.body).unwrap();
    assert!(value["requests"].as_u64().unwrap() >= 1);
    assert!(value.get("frames_written").is_some() && value.get("chunks_pushed").is_some());
}

/// Every POST route, with a body recorder.
fn recording_app(seen: Arc<Mutex<Vec<(Route, Bytes)>>>) -> FakeApp {
    let stream_seen = seen.clone();
    FakeApp::new(
        move |route, body| {
            seen.lock().unwrap().push((route, body));
            JsonReply::new(200, "{}")
        },
        move |body, sink| {
            stream_seen.lock().unwrap().push((Route::Chat, body));
            sink.reply(200, "{}");
        },
    )
}

#[test]
fn missing_content_length_means_an_empty_body() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let server = serve(recording_app(seen.clone()));
    let addr = server.local_addr();
    for route in [Route::Process, Route::Forecast, Route::Chat] {
        let response = exchange(
            addr,
            format!("POST {} HTTP/1.1\r\nHost: t\r\nConnection: close\r\n\r\n", route.path()).as_bytes(),
        );
        assert_eq!(response.status, 200, "{route:?}");
    }
    let seen = seen.lock().unwrap();
    assert_eq!(seen.len(), 3);
    assert!(seen.iter().all(|(_, body)| body.is_empty()));
}

#[test]
fn body_is_delivered_verbatim() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let server = serve(recording_app(seen.clone()));
    let payload = "{\"messages\": [{\"content\": \"caf\u{e9}\"}]}".as_bytes();
    assert_eq!(post(server.local_addr(), "/v1/chat/completions", payload).status, 200);
    assert_eq!(&seen.lock().unwrap()[0].1[..], payload);
}

/// What each Content-Length value gets on the native transport.
///
/// Known transport differences (DESIGN.md 3.2 step 4), recorded here so they
/// cannot drift silently: hyper parses `Content-Length` for framing and
/// answers every value that is not plain digits, `-1` included, and values
/// that overflow u64, with its own 400 and an EMPTY body before the
/// application runs; the stdlib transport answers those with
/// `{"error": "invalid Content-Length"}` or `{"error": "request body too
/// large"}`. Leading and trailing whitespace is optional whitespace to hyper,
/// so ` 5` and `5 ` frame a 5-byte body. Only an over-limit value of plain
/// digits reaches `body.rs` and gets the JSON body.
#[test]
fn content_length_values_on_all_post_routes() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let server = serve(recording_app(seen.clone()));
    let addr = server.local_addr();
    let hyper_rejects = ["abc", "+5", "1_0", "0x10", "", "-", "-1", "5.0", "\u{665}", "99999999999999999999999"];
    for route in [Route::Process, Route::Forecast, Route::Chat] {
        for value in hyper_rejects {
            let request = format!(
                "POST {} HTTP/1.1\r\nHost: t\r\nContent-Length: {value}\r\nConnection: close\r\n\r\n{{}}",
                route.path()
            );
            let response = exchange_half_closed(addr, request.as_bytes()).expect("an answer");
            assert_eq!((response.status, response.body.len()), (400, 0), "{route:?} {value:?}");
        }

        let too_large = exchange_half_closed(
            addr,
            format!("POST {} HTTP/1.1\r\nHost: t\r\nContent-Length: 16777217\r\nConnection: close\r\n\r\n", route.path())
                .as_bytes(),
        )
        .expect("an answer");
        assert_eq!(too_large.status, 400);
        assert_eq!(too_large.text(), r#"{"error": "request body too large"}"#, "plain shape on {route:?}");

        for value in [" 5", "5 "] {
            let request = format!(
                "POST {} HTTP/1.1\r\nHost: t\r\nContent-Length: {value}\r\nConnection: close\r\n\r\n12345",
                route.path()
            );
            assert_eq!(exchange(addr, request.as_bytes()).status, 200, "{route:?} {value:?}");
        }
    }
    let seen = seen.lock().unwrap();
    assert_eq!(seen.len(), 6, "only the whitespace cases reached the app");
    assert!(seen.iter().all(|(_, body)| &body[..] == b"12345"));
}

#[test]
fn exactly_the_body_limit_is_accepted() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let server = serve(recording_app(seen.clone()));
    let body = vec![b'a'; 16 * 1024 * 1024];
    assert_eq!(post(server.local_addr(), "/process", &body).status, 200);
    assert_eq!(seen.lock().unwrap()[0].1.len(), body.len());
}

#[test]
fn json_routes_keep_the_connection_alive() {
    let server = serve(FakeApp::echo());
    let mut stream = connect(server.local_addr());
    stream
        .write_all(b"POST /process HTTP/1.1\r\nHost: t\r\nContent-Length: 2\r\n\r\n{}POST /forecast HTTP/1.1\r\nHost: t\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
        .unwrap();
    let mut raw = String::new();
    stream.read_to_string(&mut raw).unwrap();
    assert_eq!(raw.matches("HTTP/1.1 200 OK").count(), 2, "{raw}");
    assert!(raw.contains(r#"{"route": "/process", "len": 2}"#));
    assert!(raw.contains(r#"{"route": "/forecast", "len": 0}"#));
    assert_eq!(stat(&server, "connections_total"), 1);
}

#[test]
fn status_codes_from_the_app_pass_through() {
    let app = FakeApp::new(|_route, _body| JsonReply::new(501, r#"{"error": "nope"}"#), |_b, s| {
        s.reply(400, r#"{"error": {"message": "bad", "type": "invalid_request_error"}}"#);
    });
    let server = serve(app);
    let addr = server.local_addr();
    let process = post(addr, "/forecast", b"{}");
    assert_eq!((process.status, process.text()), (501, r#"{"error": "nope"}"#));
    let chat = post(addr, "/v1/chat/completions", b"{}");
    assert_eq!(chat.status, 400);
    assert_eq!(chat.header("content-type"), Some("application/json"));
    assert!(chat.header("cache-control").is_none());
}

#[test]
fn app_panics_become_500_in_the_route_shape() {
    let app = FakeApp::new(|_route, _body| panic!("bug in the app"), |_body, _sink| panic!("bug in the app"));
    let server = serve(app);
    let addr = server.local_addr();
    let process = post(addr, "/process", b"{}");
    assert_eq!((process.status, process.text()), (500, r#"{"error": "internal server error"}"#));
    let chat = post(addr, "/v1/chat/completions", b"{}");
    assert_eq!(chat.status, 500);
    assert_eq!(
        chat.text(),
        r#"{"error": {"message": "stream handler returned without a response", "type": "server_error"}}"#
    );
    // The server is still up.
    assert_eq!(get(addr, "/healthz").status, 503);
}

#[test]
fn admission_queues_the_request_over_max_inflight_and_never_503s() {
    let release = Arc::new((Mutex::new(false), Condvar::new()));
    let app_release = release.clone();
    let app = FakeApp::new(
        move |_route, _body| {
            let (lock, cv) = &*app_release;
            let _open = cv.wait_while(lock.lock().unwrap(), |open| !*open).unwrap();
            JsonReply::new(200, "{}")
        },
        |_b, s| {
            s.reply(200, "{}");
        },
    );
    let server = serve_with(ServerConfig { max_inflight: 2, ..config() }, app);
    let addr = server.local_addr();
    let clients: Vec<_> = (0..3).map(|_| std::thread::spawn(move || post(addr, "/process", b"{}"))).collect();
    assert!(wait_until(Duration::from_secs(5), || stat(&server, "inflight") == 2 && stat(&server, "queued") == 1));
    std::thread::sleep(Duration::from_millis(200));
    assert_eq!(stat(&server, "queued"), 1, "the third request waits in Rust");

    let (lock, cv) = &*release;
    *lock.lock().unwrap() = true;
    cv.notify_all();
    for client in clients {
        assert_eq!(client.join().unwrap().status, 200);
    }
    assert_eq!(stat(&server, "queued"), 0);
    assert_eq!(stat(&server, "inflight"), 0);
}

#[test]
fn second_bind_on_a_busy_port_fails_with_addr_in_use() {
    let server = serve(FakeApp::echo());
    let port = server.local_addr().port();
    let error = start(ServerConfig { port, ..config() }, Arc::new(FakeApp::echo())).err().expect("must fail");
    assert_eq!(error.kind(), std::io::ErrorKind::AddrInUse);
}

#[test]
fn port_is_free_again_after_close() {
    let server = serve(FakeApp::echo());
    let port = server.local_addr().port();
    server.close();
    let again = start(ServerConfig { port, ..config() }, Arc::new(FakeApp::echo())).expect("re-bind");
    assert_eq!(again.local_addr().port(), port);
}

#[cfg(target_os = "linux")]
#[test]
fn io_and_request_threads_are_named() {
    fn thread_names() -> Vec<String> {
        std::fs::read_dir("/proc/self/task")
            .unwrap()
            .filter_map(|entry| std::fs::read_to_string(entry.ok()?.path().join("comm")).ok())
            .map(|name| name.trim().to_owned())
            .collect()
    }
    let seen_name = Arc::new(Mutex::new(String::new()));
    let app_seen = seen_name.clone();
    let app = FakeApp::new(
        move |_route, _body| {
            let comm = std::fs::read_to_string("/proc/thread-self/comm").unwrap();
            *app_seen.lock().unwrap() = comm.trim().to_owned();
            JsonReply::new(200, "{}")
        },
        |_b, s| {
            s.reply(200, "{}");
        },
    );
    let server = serve_with(ServerConfig { io_threads: 3, ..config() }, app);
    let names = thread_names();
    for n in 0..3 {
        assert!(names.contains(&format!("ik-http-io-{n}")), "{names:?}");
    }
    assert_eq!(post(server.local_addr(), "/process", b"{}").status, 200);
    assert_eq!(*seen_name.lock().unwrap(), "ik-backend-req", "app calls run on request threads");
    assert!(thread_names().iter().any(|name| name == "ik-backend-req"));
}

#[test]
fn many_sequential_requests_do_not_leak_gauges() {
    let counter = Arc::new(AtomicUsize::new(0));
    let app_counter = counter.clone();
    let app = FakeApp::new(
        move |_route, _body| {
            app_counter.fetch_add(1, Ordering::Relaxed);
            JsonReply::new(200, "{}")
        },
        |_b, s| {
            s.reply(200, "{}");
        },
    );
    let server = serve(app);
    for _ in 0..50 {
        assert_eq!(post(server.local_addr(), "/process", b"{}").status, 200);
    }
    assert_eq!(counter.load(Ordering::Relaxed), 50);
    assert!(wait_until(Duration::from_secs(2), || stat(&server, "connections") == 0));
    assert_eq!(stat(&server, "inflight"), 0);
    assert_eq!(stat(&server, "queued"), 0);
}
