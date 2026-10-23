//! Request body rules for the POST routes, mirroring `serve.content_length`.
//!
//! - A missing `Content-Length` means an empty body.
//! - The value must match `-?[0-9]+` exactly; anything else is
//!   `400 {"error": "invalid Content-Length"}`.
//! - A negative value, or one above the body limit, is
//!   `400 {"error": "request body too large"}`.
//!
//! A too-large body is still read and dropped, up to [`DISCARD_MAX`] bytes
//! and for at most [`DISCARD_TIMEOUT`], before the 400 goes out with
//! `Connection: close`. Closing with the body unread makes the kernel reset
//! the connection, and the client (reqwest in the worker) then reports a send
//! error instead of the 400.
//!
//! hyper parses `Content-Length` itself for framing and answers some malformed
//! values with its own bodyless 400 before the application runs; the
//! integration tests record which values reach this module.

use std::time::Duration;

use bytes::Bytes;
use http_body_util::BodyExt;
use hyper::body::Incoming;
use hyper::header::{HeaderMap, CONTENT_LENGTH};

/// Why a declared length was refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LengthError {
    Invalid,
    TooLarge,
}

impl LengthError {
    pub fn message(self) -> &'static str {
        match self {
            LengthError::Invalid => "invalid Content-Length",
            LengthError::TooLarge => "request body too large",
        }
    }
}

/// The body length a request declares, checked against `max`.
pub(crate) fn declared_length(headers: &HeaderMap, max: u64) -> Result<u64, LengthError> {
    parse_content_length(headers.get(CONTENT_LENGTH).map(|value| value.as_bytes()), max)
}

/// [`declared_length`] on the raw header value (`None` when absent).
pub(crate) fn parse_content_length(value: Option<&[u8]>, max: u64) -> Result<u64, LengthError> {
    let Some(value) = value else {
        return Ok(0);
    };
    let (negative, digits) = match value.split_first() {
        Some((b'-', rest)) => (true, rest),
        _ => (false, value),
    };
    if digits.is_empty() || !digits.iter().all(u8::is_ascii_digit) {
        return Err(LengthError::Invalid);
    }
    // Saturate instead of overflowing: any huge value is "too large", as
    // Python's arbitrary-precision int() makes it.
    let magnitude = digits
        .iter()
        .fold(0u64, |acc, digit| acc.saturating_mul(10).saturating_add(u64::from(digit - b'0')));
    if (negative && magnitude > 0) || magnitude > max {
        return Err(LengthError::TooLarge);
    }
    Ok(magnitude)
}

/// Read exactly the declared body. A zero length reads nothing, even if the
/// client sent a body without `Content-Length` (as the stdlib transport does).
pub(crate) async fn collect(body: Incoming, length: u64) -> Result<Bytes, hyper::Error> {
    if length == 0 {
        return Ok(Bytes::new());
    }
    Ok(body.collect().await?.to_bytes())
}

/// The most bytes of a refused body that are read and dropped.
pub(crate) const DISCARD_MAX: u64 = 64 * 1024 * 1024;

/// How long a refused body may take to arrive before the 400 goes out anyway.
pub(crate) const DISCARD_TIMEOUT: Duration = Duration::from_secs(10);

/// Read and drop a refused body, up to `max` bytes or until `timeout`. Errors
/// (a client that stops sending, a short body) end the drain quietly: the
/// caller answers and closes either way.
pub(crate) async fn discard(mut body: Incoming, max: u64, timeout: Duration) {
    let drain = async {
        let mut read = 0u64;
        while read < max {
            match body.frame().await {
                Some(Ok(frame)) => {
                    if let Some(data) = frame.data_ref() {
                        read = read.saturating_add(data.len() as u64);
                    }
                }
                Some(Err(_)) | None => break,
            }
        }
    };
    let _ = tokio::time::timeout(timeout, drain).await;
}

#[cfg(test)]
mod tests {
    use super::*;

    const MAX: u64 = 16 * 1024 * 1024;

    #[test]
    fn accepts_plain_digits_and_missing_header() {
        assert_eq!(parse_content_length(None, MAX), Ok(0));
        assert_eq!(parse_content_length(Some(b"0"), MAX), Ok(0));
        assert_eq!(parse_content_length(Some(b"-0"), MAX), Ok(0));
        assert_eq!(parse_content_length(Some(b"0042"), MAX), Ok(42));
        assert_eq!(parse_content_length(Some(b"16777216"), MAX), Ok(MAX));
    }

    #[test]
    fn rejects_everything_the_python_regex_rejects() {
        let invalid: [&[u8]; 12] = [
            b"abc", b"+5", b" 5", b"5 ", b"1_0", b"0x10", b"", b"-", b"5.0", b"--5",
            "\u{665}".as_bytes(), // ARABIC-INDIC DIGIT FIVE, which int() accepts
            b"5\t",
        ];
        for value in invalid {
            assert_eq!(parse_content_length(Some(value), MAX), Err(LengthError::Invalid), "{value:?}");
        }
    }

    #[test]
    fn negative_and_oversized_are_too_large() {
        for value in [&b"-1"[..], b"16777217", b"99999999999999999999999999", b"-99999999999999999999"] {
            assert_eq!(parse_content_length(Some(value), MAX), Err(LengthError::TooLarge), "{value:?}");
        }
    }
}
