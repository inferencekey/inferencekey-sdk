//! The boundary between the transport and the code that decides what a
//! request means (in production, the Python `serve.py` functions).

use bytes::Bytes;

use crate::sse::StreamSink;

/// A POST route the application answers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Route {
    /// `POST /process`
    Process,
    /// `POST /forecast`
    Forecast,
    /// `POST /v1/chat/completions` (always dispatched to
    /// [`AppHandler::call_stream`], which may answer with plain JSON).
    Chat,
}

impl Route {
    /// The request path, which is also the route name handed to Python.
    pub fn path(self) -> &'static str {
        match self {
            Route::Process => "/process",
            Route::Forecast => "/forecast",
            Route::Chat => "/v1/chat/completions",
        }
    }
}

/// A complete JSON response: HTTP status and the exact body bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JsonReply {
    pub status: u16,
    pub body: Bytes,
}

impl JsonReply {
    pub fn new(status: u16, body: impl Into<Bytes>) -> Self {
        JsonReply { status, body: body.into() }
    }
}

/// What the transport calls for each POST request.
///
/// Both methods block. They run on the server's blocking pool, one thread per
/// in-flight request, after the request has an admission permit, so an
/// implementation may block on IO, locks or a GIL for as long as it needs.
/// Calls happen concurrently from many threads.
pub trait AppHandler: Send + Sync + 'static {
    /// Answer `POST /process` or `POST /forecast` with a JSON response.
    fn call_json(&self, route: Route, body: Bytes) -> JsonReply;

    /// Answer `POST /v1/chat/completions` through `sink`: either
    /// [`StreamSink::reply`] (plain JSON, including every early error) or
    /// [`StreamSink::start`] followed by pushes and [`StreamSink::done`].
    ///
    /// Returning without calling either answers 500. Returning ends the
    /// stream even if `done` was not called (the body then ends without
    /// `[DONE]`).
    fn call_stream(&self, body: Bytes, sink: StreamSink);
}
