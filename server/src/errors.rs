//! Response bodies the transport builds itself.
//!
//! Every JSON body here is byte-equal to what the Python stdlib transport
//! sends for the same case (`json.dumps` with default separators and
//! `ensure_ascii`), so a client cannot tell the transports apart. The golden
//! test in the Python suite compares them.

use bytes::Bytes;
use inferencekey_core::domain::sse::{error_event, write_python_json_string};

use crate::app::Route;

/// `{"error": "<message>"}`, the plain error shape of every POST route.
pub(crate) fn plain_error(message: &str) -> Bytes {
    let mut out = Vec::with_capacity(message.len() + 16);
    out.extend_from_slice(b"{\"error\": ");
    write_python_json_string(message, &mut out);
    out.push(b'}');
    Bytes::from(out)
}

/// The 404 body for any unknown GET or POST path.
pub(crate) fn not_found() -> Bytes {
    plain_error("not found")
}

/// `GET /healthz` bodies.
pub(crate) const HEALTH_OK: &[u8] = b"{\"status\": \"ok\"}";
pub(crate) const HEALTH_LOADING: &[u8] = b"{\"status\": \"loading\"}";

/// The 500 body when the application itself fails (a panic, or a stream
/// handler that returned without answering), in the route's error shape:
/// the OpenAI shape on chat, the plain shape elsewhere.
pub fn internal_error(route: Route, message: &str) -> Bytes {
    match route {
        Route::Chat => Bytes::from(error_event(message)),
        Route::Process | Route::Forecast => plain_error(message),
    }
}

/// `GET /v1/models` for a manifest, as Python's
/// `json.dumps(models_payload(manifest))` writes it.
///
/// The model id is the manifest's `name` (`""` when absent or when the
/// manifest is not a JSON object). A non-string `name` is written with
/// `serde_json`, which matches Python for null, booleans and integers.
pub(crate) fn models_body(manifest: &[u8]) -> Bytes {
    let name = serde_json::from_slice::<serde_json::Value>(manifest)
        .ok()
        .and_then(|value| value.get("name").cloned());
    let mut out = Vec::with_capacity(96);
    out.extend_from_slice(b"{\"object\": \"list\", \"data\": [{\"id\": ");
    match name {
        None => out.extend_from_slice(b"\"\""),
        Some(serde_json::Value::String(name)) => write_python_json_string(&name, &mut out),
        Some(other) => out.extend_from_slice(other.to_string().as_bytes()),
    }
    out.extend_from_slice(b", \"object\": \"model\", \"owned_by\": \"custom\"}]}");
    Bytes::from(out)
}

/// Content type of [`unsupported_method_page`], as `http.server` sends it.
pub(crate) const HTML_CONTENT_TYPE: &str = "text/html;charset=utf-8";

/// The HTML page `http.server.BaseHTTPRequestHandler.send_error(501, ...)`
/// writes (CPython 3.12) for a method without a `do_*` handler.
pub(crate) fn unsupported_method_page(method: &str) -> Bytes {
    let message = html_escape(&format!("Unsupported method ('{method}')"));
    Bytes::from(format!(
        "<!DOCTYPE HTML>\n<html lang=\"en\">\n    <head>\n        <meta charset=\"utf-8\">\n        \
         <title>Error response</title>\n    </head>\n    <body>\n        <h1>Error response</h1>\n        \
         <p>Error code: 501</p>\n        <p>Message: {message}.</p>\n        \
         <p>Error code explanation: 501 - Server does not support this operation.</p>\n    \
         </body>\n</html>\n"
    ))
}

/// `html.escape(text, quote=False)`.
fn html_escape(text: &str) -> String {
    text.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_error_matches_python() {
        assert_eq!(&plain_error("not found")[..], br#"{"error": "not found"}"#);
        assert_eq!(
            &plain_error("invalid Content-Length")[..],
            br#"{"error": "invalid Content-Length"}"#
        );
    }

    #[test]
    fn models_body_matches_python() {
        assert_eq!(
            &models_body(br#"{"name": "my-llm", "api": "openai-chat"}"#)[..],
            br#"{"object": "list", "data": [{"id": "my-llm", "object": "model", "owned_by": "custom"}]}"#
        );
        assert_eq!(
            &models_body(b"{}")[..],
            br#"{"object": "list", "data": [{"id": "", "object": "model", "owned_by": "custom"}]}"#
        );
        // ensure_ascii, as json.dumps does by default.
        let body = models_body("{\"name\": \"m\u{e9}\"}".as_bytes());
        assert!(std::str::from_utf8(&body).unwrap().contains(r#""id": "m\u00e9""#));
    }

    #[test]
    fn unsupported_method_page_matches_cpython_312() {
        // Printed by CPython 3.12 from http.server.DEFAULT_ERROR_MESSAGE.
        let expected = "<!DOCTYPE HTML>\n<html lang=\"en\">\n    <head>\n        <meta charset=\"utf-8\">\n        <title>Error response</title>\n    </head>\n    <body>\n        <h1>Error response</h1>\n        <p>Error code: 501</p>\n        <p>Message: Unsupported method ('PUT').</p>\n        <p>Error code explanation: 501 - Server does not support this operation.</p>\n    </body>\n</html>\n";
        assert_eq!(std::str::from_utf8(&unsupported_method_page("PUT")).unwrap(), expected);
        assert!(std::str::from_utf8(&unsupported_method_page("A<B&")).unwrap().contains("('A&lt;B&amp;')"));
    }

    #[test]
    fn internal_error_uses_the_route_shape() {
        assert_eq!(&internal_error(Route::Process, "boom")[..], br#"{"error": "boom"}"#);
        assert_eq!(
            &internal_error(Route::Chat, "boom")[..],
            br#"{"error": {"message": "boom", "type": "server_error"}}"#
        );
    }
}
