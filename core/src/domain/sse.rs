//! Pure SSE line classification for the OpenAI-compatible streaming protocol.
//!
//! The data plane streams `chat.completion.chunk` objects as Server-Sent
//! Events. Each physical line is one of three things: a JSON `data:` payload,
//! the terminal `data: [DONE]` sentinel, or noise (blank lines, comments,
//! non-`data:` fields). This module turns a single raw line into that verdict.
//!
//! Pure: no IO, no time, no allocation beyond what `serde_json` needs. The
//! caller owns the byte stream and the line splitting; this only classifies.

use serde_json::Value;

/// The classification of one raw SSE line.
#[derive(Debug, Clone, PartialEq)]
pub enum SseLine {
    /// A decoded `data:` JSON payload (e.g. a `chat.completion.chunk`).
    Data(Value),
    /// The terminal `data: [DONE]` sentinel.
    Done,
    /// Noise to ignore: blanks, comments, non-`data:` fields, undecodable JSON.
    Skip,
}

/// The SSE comment prefix; lines starting with it carry no field.
const COMMENT_PREFIX: char = ':';
/// The only SSE field we consume.
const DATA_FIELD: &str = "data:";
/// The terminal sentinel value the server sends to close the stream.
const DONE_SENTINEL: &str = "[DONE]";

/// Classify one raw SSE line into an [`SseLine`].
///
/// Strips a trailing `\r`, skips blanks and `:` comments, only reads the
/// `data:` field, drops one optional leading space from the value, maps the
/// `[DONE]` sentinel to [`SseLine::Done`], and decodes the rest as JSON
/// ([`SseLine::Data`]) — falling back to [`SseLine::Skip`] when it cannot.
pub fn parse_sse_line(raw: &str) -> SseLine {
    classify_value(extract_data_value(strip_trailing_cr(raw)))
}

/// Drop a single trailing carriage return left by `\r\n` line endings.
fn strip_trailing_cr(raw: &str) -> &str {
    raw.strip_suffix('\r').unwrap_or(raw)
}

/// The trimmed value of the `data:` field, if this line carries one.
///
/// Returns `None` for blanks, comment lines, and any other SSE field, so the
/// caller need only reason about present-and-relevant values.
fn extract_data_value(line: &str) -> Option<&str> {
    match line {
        "" => None,
        l if l.starts_with(COMMENT_PREFIX) => None,
        l => l.strip_prefix(DATA_FIELD).map(strip_one_leading_space),
    }
}

/// Remove exactly one optional leading space, per the SSE field grammar.
fn strip_one_leading_space(value: &str) -> &str {
    value.strip_prefix(' ').unwrap_or(value)
}

/// Turn an extracted `data:` value into its verdict.
fn classify_value(value: Option<&str>) -> SseLine {
    match value {
        None => SseLine::Skip,
        Some(DONE_SENTINEL) => SseLine::Done,
        Some(v) => decode_json(v),
    }
}

/// Decode a JSON payload, treating undecodable input as [`SseLine::Skip`].
fn decode_json(value: &str) -> SseLine {
    match serde_json::from_str::<Value>(value) {
        Ok(json) => SseLine::Data(json),
        Err(_) => SseLine::Skip,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn classifies_each_line_kind() {
        let cases = [
            // (raw line, expected verdict)
            ("data: [DONE]", SseLine::Done),
            ("data: {\"object\":\"chat.completion.chunk\"}",
             SseLine::Data(json!({"object": "chat.completion.chunk"}))),
            ("data:{\"object\":\"chat.completion.chunk\"}",
             SseLine::Data(json!({"object": "chat.completion.chunk"}))),
            ("", SseLine::Skip),
            (": keep-alive comment", SseLine::Skip),
            ("event: message", SseLine::Skip),
            ("data: not json", SseLine::Skip),
            ("data: ", SseLine::Skip),
        ];

        for (raw, expected) in cases {
            assert_eq!(parse_sse_line(raw), expected, "raw = {raw:?}");
        }
    }

    #[test]
    fn strips_trailing_carriage_return_before_classifying() {
        assert_eq!(parse_sse_line("data: [DONE]\r"), SseLine::Done);
        assert_eq!(
            parse_sse_line("data: {\"n\":1}\r"),
            SseLine::Data(json!({"n": 1})),
        );
    }

    #[test]
    fn strips_only_one_leading_space_from_the_value() {
        // The first space is the field separator; a second belongs to the value
        // and must make the JSON undecodable rather than be silently eaten.
        assert_eq!(parse_sse_line("data:  [DONE]"), SseLine::Skip);
        assert_eq!(
            parse_sse_line("data:\"hi\""),
            SseLine::Data(json!("hi")),
        );
    }

    #[test]
    fn comment_line_is_skipped_even_when_it_mentions_data() {
        assert_eq!(parse_sse_line(":data: [DONE]"), SseLine::Skip);
    }
}
