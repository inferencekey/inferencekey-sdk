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

// --------------------------------------------------------------------------
// Producer side: the bytes a server writes.
//
// The backend runtime (the `inferencekey-backend-server` crate and the Python
// stdlib transport) frames events exactly like this, so a client that reads
// them with `parse_sse_line` sees the same values the producer encoded. Pure:
// they only append to buffers.

/// The terminal event that ends every successful stream.
pub const DONE_EVENT: &[u8] = b"data: [DONE]\n\n";

/// The SSE comment a backend's "nothing to send yet" (`None`) chunk becomes.
/// Clients ignore it; writing it is how a server notices a client that left.
pub const KEEPALIVE_EVENT: &[u8] = b": keepalive\n\n";

/// The `data:` field prefix a producer writes (with its separator space).
const DATA_PREFIX: &[u8] = b"data: ";
/// The blank line that ends one event.
const EVENT_END: &[u8] = b"\n\n";

/// Append one event, `data: <json>\n\n`, to `out`.
///
/// `json` must be a single line (no raw `\n` or `\r`), as any serialized JSON
/// value is; the caller guarantees it.
pub fn format_data_event(json: &[u8], out: &mut Vec<u8>) {
    out.reserve(DATA_PREFIX.len() + json.len() + EVENT_END.len());
    out.extend_from_slice(DATA_PREFIX);
    out.extend_from_slice(json);
    out.extend_from_slice(EVENT_END);
}

/// The JSON payload of a mid-stream error event (to be framed with
/// [`format_data_event`]).
///
/// The bytes equal Python's `json.dumps({"error": {"message": message,
/// "type": "server_error"}})`, which is what the Python stream driver sends,
/// so both transports emit the same event.
pub fn error_event(message: &str) -> Vec<u8> {
    let mut out = Vec::with_capacity(message.len() + 48);
    out.extend_from_slice(b"{\"error\": {\"message\": ");
    write_python_json_string(message, &mut out);
    out.extend_from_slice(b", \"type\": \"server_error\"}}");
    out
}

/// Append `value` as a JSON string literal, byte-equal to Python's
/// `json.dumps(value)` with its defaults (`ensure_ascii=True`).
///
/// Quotes and backslashes are escaped, `\n \r \t \b \f` use their short forms,
/// every other control character and everything outside printable ASCII
/// (including DEL) becomes `\uXXXX` in lowercase hex, and characters beyond
/// the BMP become a UTF-16 surrogate pair.
pub fn write_python_json_string(value: &str, out: &mut Vec<u8>) {
    out.push(b'"');
    for ch in value.chars() {
        match ch {
            '"' => out.extend_from_slice(b"\\\""),
            '\\' => out.extend_from_slice(b"\\\\"),
            '\n' => out.extend_from_slice(b"\\n"),
            '\r' => out.extend_from_slice(b"\\r"),
            '\t' => out.extend_from_slice(b"\\t"),
            '\u{08}' => out.extend_from_slice(b"\\b"),
            '\u{0c}' => out.extend_from_slice(b"\\f"),
            ' '..='~' => out.push(ch as u8),
            _ => {
                let mut units = [0u16; 2];
                for unit in ch.encode_utf16(&mut units) {
                    push_unicode_escape(*unit, out);
                }
            }
        }
    }
    out.push(b'"');
}

/// Append `\uXXXX` for one UTF-16 code unit, lowercase hex like Python.
fn push_unicode_escape(unit: u16, out: &mut Vec<u8>) {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    out.extend_from_slice(b"\\u");
    for shift in [12, 8, 4, 0] {
        out.push(HEX[usize::from((unit >> shift) & 0xf)]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Split framed bytes into the lines a client would feed `parse_sse_line`.
    fn lines(framed: &[u8]) -> Vec<String> {
        std::str::from_utf8(framed)
            .expect("utf-8")
            .split('\n')
            .map(str::to_owned)
            .collect()
    }

    #[test]
    fn data_event_round_trips_through_the_parser() {
        let values = [
            json!({"choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": null}]}),
            json!({"text": "caf\u{e9} \u{2615} \u{1F600} \u{4e2d}\u{6587}"}),
            json!({"text": "say \"quoted\" and \\ back"}),
            json!({"text": "line one\nline two\r\n\ttabbed"}),
            json!("plain string"),
            json!([1, 2.5, true, null]),
        ];
        for value in values {
            let payload = serde_json::to_vec(&value).unwrap();
            let mut framed = Vec::new();
            format_data_event(&payload, &mut framed);

            assert!(framed.starts_with(b"data: ") && framed.ends_with(b"\n\n"));
            let got = lines(&framed);
            // One data line, then the blank line that ends the event.
            assert_eq!(got.len(), 3, "an escaped newline must not split the event: {got:?}");
            assert_eq!(parse_sse_line(&got[0]), SseLine::Data(value));
            assert_eq!(parse_sse_line(&got[1]), SseLine::Skip);
            assert_eq!(got[2], "");
        }
    }

    #[test]
    fn format_data_event_appends_without_clearing() {
        let mut out = b"data: [DONE]\n\n".to_vec();
        format_data_event(b"{}", &mut out);
        assert_eq!(out, b"data: [DONE]\n\ndata: {}\n\n");
    }

    #[test]
    fn done_and_keepalive_events_classify_as_expected() {
        let done = lines(DONE_EVENT);
        assert_eq!(parse_sse_line(&done[0]), SseLine::Done);
        assert_eq!(&done[1..], ["", ""]);

        let keepalive = lines(KEEPALIVE_EVENT);
        assert_eq!(parse_sse_line(&keepalive[0]), SseLine::Skip);
        assert_eq!(&keepalive[1..], ["", ""]);
    }

    #[test]
    fn error_event_round_trips_with_unicode_quotes_and_newlines() {
        let messages = [
            "disk",
            "q\"uote\\back",
            "line\nbreak\r\t\u{08}\u{0c}\u{01}\u{1f}\u{7f}",
            "caf\u{e9} \u{2615} \u{1F600}",
            "",
        ];
        for message in messages {
            let mut framed = Vec::new();
            format_data_event(&error_event(message), &mut framed);
            let got = lines(&framed);
            assert_eq!(got.len(), 3, "{message:?}");
            assert_eq!(
                parse_sse_line(&got[0]),
                SseLine::Data(json!({"error": {"message": message, "type": "server_error"}})),
            );
        }
    }

    #[test]
    fn error_event_bytes_equal_python_json_dumps() {
        // Expected values printed by CPython 3.12:
        // json.dumps({"error": {"message": m, "type": "server_error"}})
        let cases: [(&str, &str); 5] = [
            ("disk", r#"{"error": {"message": "disk", "type": "server_error"}}"#),
            ("q\"uote\\back", r#"{"error": {"message": "q\"uote\\back", "type": "server_error"}}"#),
            (
                "line\nbreak\r\t\u{08}\u{0c}\u{01}\u{1f}\u{7f}",
                r#"{"error": {"message": "line\nbreak\r\t\b\f\u0001\u001f\u007f", "type": "server_error"}}"#,
            ),
            (
                "caf\u{e9} \u{2615} \u{1F600}",
                r#"{"error": {"message": "caf\u00e9 \u2615 \ud83d\ude00", "type": "server_error"}}"#,
            ),
            ("</script>&", r#"{"error": {"message": "</script>&", "type": "server_error"}}"#),
        ];
        for (message, expected) in cases {
            assert_eq!(
                std::str::from_utf8(&error_event(message)).unwrap(),
                expected,
                "message = {message:?}"
            );
        }
    }

    #[test]
    fn python_json_string_is_pure_ascii_and_decodes_back() {
        let value = "\u{0}\u{1f} ~\u{7f}\u{80}\u{ffff}\u{10000}\u{10ffff}";
        let mut out = Vec::new();
        write_python_json_string(value, &mut out);
        assert!(out.iter().all(|b| (b' '..=b'~').contains(b)));
        assert_eq!(serde_json::from_slice::<String>(&out).unwrap(), value);
        assert_eq!(
            std::str::from_utf8(&out).unwrap(),
            r#""\u0000\u001f ~\u007f\u0080\uffff\ud800\udc00\udbff\udfff""#
        );
    }

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
