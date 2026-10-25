//! Method and path dispatch: native endpoints answered here without calling
//! the application, POST routes handed to it through admission and the
//! blocking pool.

use std::io;
use std::pin::Pin;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::task::{Context, Poll};

use bytes::Bytes;
use hyper::body::{Body, Frame, Incoming, SizeHint};
use hyper::header::{HeaderValue, CACHE_CONTROL, CONNECTION, CONTENT_TYPE};
use hyper::{Method, Request, Response, StatusCode, Uri, Version};

use crate::app::Route;
use crate::errors;
use crate::sse::{self, Head, SseBody, StreamSink};
use crate::stats::{self, GaugeGuard};
use crate::{body, name_request_thread, Shared};

/// A response body: a complete buffer, or an event stream.
pub(crate) enum AppBody {
    Full(Option<Bytes>),
    Stream(SseBody),
}

impl Body for AppBody {
    type Data = Bytes;
    type Error = io::Error;

    fn poll_frame(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Bytes>, io::Error>>> {
        match self.get_mut() {
            AppBody::Full(data) => Poll::Ready(data.take().map(|bytes| Ok(Frame::data(bytes)))),
            AppBody::Stream(stream) => Pin::new(stream).poll_frame(cx),
        }
    }

    fn is_end_stream(&self) -> bool {
        match self {
            AppBody::Full(data) => data.as_ref().map_or(true, Bytes::is_empty),
            AppBody::Stream(stream) => stream.is_end_stream(),
        }
    }

    fn size_hint(&self) -> SizeHint {
        match self {
            AppBody::Full(data) => SizeHint::with_exact(data.as_ref().map_or(0, |b| b.len() as u64)),
            AppBody::Stream(stream) => stream.size_hint(),
        }
    }
}

const APPLICATION_JSON: HeaderValue = HeaderValue::from_static("application/json");

/// A JSON response with an exact `Content-Length`.
fn json(status: u16, body: impl Into<Bytes>) -> Response<AppBody> {
    let status = StatusCode::from_u16(status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    let mut response = Response::new(AppBody::Full(Some(body.into())));
    *response.status_mut() = status;
    response.headers_mut().insert(CONTENT_TYPE, APPLICATION_JSON);
    response
}

/// The path a route is matched on: the query string is stripped. An
/// absolute-form target (`GET http://host/healthz`) matches nothing, as in the
/// stdlib transport, where the whole target is the path. Leading slashes
/// collapse to one (`//healthz` is `/healthz`), as CPython's
/// `BaseHTTPRequestHandler.parse_request` does (gh-87389).
///
/// One difference cannot be mirrored: the `http` crate drops a `#fragment`
/// from the target, while stdlib compares it as part of the path, so
/// `/healthz#x` is 200 here and 404 there. Clients never send fragments.
fn route_path(uri: &Uri) -> String {
    if uri.scheme().is_some() {
        return String::new();
    }
    let path = uri.path();
    if path.starts_with("//") {
        format!("/{}", path.trim_start_matches('/'))
    } else {
        path.to_owned()
    }
}

fn post_route(path: &str) -> Option<Route> {
    match path {
        "/process" => Some(Route::Process),
        "/forecast" => Some(Route::Forecast),
        "/v1/chat/completions" => Some(Route::Chat),
        _ => None,
    }
}

pub(crate) async fn handle(
    req: Request<Incoming>,
    shared: Arc<Shared>,
) -> Result<Response<AppBody>, io::Error> {
    stats::inc(&shared.stats.requests);
    let path = route_path(req.uri());
    match *req.method() {
        Method::GET => Ok(handle_get(&path, &shared)),
        Method::POST => match post_route(&path) {
            Some(route) => handle_post(route, req, shared).await,
            None => Ok(json(404, errors::not_found())),
        },
        _ => Ok(unsupported_method(req.method(), &shared.cfg.unsupported_method_page)),
    }
}

fn handle_get(path: &str, shared: &Shared) -> Response<AppBody> {
    match path {
        "/healthz" => {
            if shared.ready.load(Ordering::Acquire) {
                json(200, Bytes::from_static(errors::HEALTH_OK))
            } else {
                json(503, Bytes::from_static(errors::HEALTH_LOADING))
            }
        }
        "/meta" => json(200, shared.manifest().meta),
        "/v1/models" => json(200, shared.manifest().models),
        "/debug/stats" => json(200, shared.stats.to_json()),
        _ => json(404, errors::not_found()),
    }
}

/// 501 for a method without a handler, as `BaseHTTPRequestHandler` answers it.
fn unsupported_method(method: &Method, page: &errors::ErrorPage) -> Response<AppBody> {
    let mut response = Response::new(AppBody::Full(Some(page.unsupported_method(method.as_str()))));
    *response.status_mut() = StatusCode::NOT_IMPLEMENTED;
    let content_type = HeaderValue::from_str(&page.content_type)
        .unwrap_or_else(|_| HeaderValue::from_static("text/html;charset=utf-8"));
    response.headers_mut().insert(CONTENT_TYPE, content_type);
    response.headers_mut().insert(CONNECTION, HeaderValue::from_static("close"));
    response
}

async fn handle_post(
    route: Route,
    req: Request<Incoming>,
    shared: Arc<Shared>,
) -> Result<Response<AppBody>, io::Error> {
    let length = match body::declared_length(req.headers(), shared.cfg.max_body as u64) {
        Ok(length) => length,
        Err(error) => {
            if error == body::LengthError::TooLarge {
                body::discard(req.into_body(), body::DISCARD_MAX, body::DISCARD_TIMEOUT).await;
            }
            let mut response = json(400, errors::plain_error(error.message()));
            response.headers_mut().insert(CONNECTION, HeaderValue::from_static("close"));
            return Ok(response);
        }
    };
    let request_version = req.version();
    let body = body::collect(req.into_body(), length)
        .await
        .map_err(io::Error::other)?;

    // Admission: excess requests wait here, never in the kernel backlog and
    // never with a 429/503.
    let permit = {
        let _queued = GaugeGuard::new(&shared.stats.queued);
        match shared.admission.clone().acquire_owned().await {
            Ok(permit) => permit,
            // Shutdown closed admission: answer instead of resetting, in the
            // route's "backend not ready" shape.
            Err(_) => return Ok(json(503, errors::internal_error(route, "backend not ready"))),
        }
    };

    let app = shared.app.clone();
    let stats = shared.stats.clone();
    match route {
        Route::Process | Route::Forecast => {
            let call = tokio::task::spawn_blocking(move || {
                name_request_thread();
                let _permit = permit;
                let _inflight = GaugeGuard::new(&stats.inflight);
                app.call_json(route, body)
            });
            Ok(match call.await {
                Ok(reply) => json(reply.status, reply.body),
                Err(_) => json(500, errors::internal_error(route, "internal server error")),
            })
        }
        Route::Chat => {
            let (sink, head, mut stream) =
                sse::channel(
                shared.cfg.stream_max_bytes,
                shared.closing.clone(),
                shared.stats.clone(),
                shared.flusher.clone(),
            );
            tokio::task::spawn_blocking(move || {
                name_request_thread();
                let _permit = permit;
                let _inflight = GaugeGuard::new(&stats.inflight);
                let _finish = FinishOnDrop(sink.clone());
                app.call_stream(body, sink);
            });
            Ok(match head.await {
                Ok(Head::Reply { status, body }) => json(status, body),
                Ok(Head::Start) => {
                    stream.arm();
                    event_stream(stream, request_version, shared.cfg.sse_connection_close)
                }
                Err(_) => json(
                    500,
                    errors::internal_error(Route::Chat, "stream handler returned without a response"),
                ),
            })
        }
    }
}

/// Ends the stream when the application call returns or panics, even if the
/// application kept a clone of the sink.
struct FinishOnDrop(StreamSink);

impl Drop for FinishOnDrop {
    fn drop(&mut self) {
        self.0.finish();
    }
}

/// The 200 event-stream head. HTTP/1.1 clients get chunked framing; HTTP/1.0
/// clients, or every client with `sse_connection_close`, get a close-delimited
/// body (an HTTP/1.0 status line makes hyper frame it that way).
fn event_stream(stream: SseBody, request_version: Version, force_close: bool) -> Response<AppBody> {
    let mut response = Response::new(AppBody::Stream(stream));
    let headers = response.headers_mut();
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));
    headers.insert(CACHE_CONTROL, HeaderValue::from_static("no-cache"));
    headers.insert(CONNECTION, HeaderValue::from_static("close"));
    if force_close || request_version == Version::HTTP_10 {
        *response.version_mut() = Version::HTTP_10;
    }
    response
}
