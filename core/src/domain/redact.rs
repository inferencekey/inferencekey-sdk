//! Token redaction. A single pure function that strips the secret tail off an
//! InferenceKey credential so it can be logged safely.
//!
//! Pure: string slicing only, no IO. The full token is `ik_<kind>_<id>_<secret>`
//! (e.g. `ik_sdk_<8hex>_<64hex>` or `ik_live_<id>_<secret>`). We keep the
//! credential prefix and a short head of the id — never the secret tail — so
//! the result is enough to correlate a credential in logs without leaking it.

/// How many leading characters of the id segment we keep.
const ID_HEAD_LEN: usize = 8;

/// The ellipsis marker that stands in for everything we dropped.
const ELLIPSIS: &str = "\u{2026}";

/// Head used when the token does not match the `ik_<kind>_<id>_…` shape
/// (empty, truncated, or otherwise unknown). Deliberately tiny so an unknown
/// or malformed secret never survives into a log line.
const UNKNOWN_HEAD: &str = "ik";

/// Redact a credential to its prefix plus a short hex head.
///
/// `ik_<kind>_<id>_<secret>` becomes `ik_<kind>_<first 8 of id>…`. Anything that
/// does not parse into at least `ik`, a kind, an id, and a secret segment falls
/// back to a fixed `ik…` head. The secret tail is never returned.
pub fn redact(token: &str) -> String {
    parse_segments(token)
        .map(render_known)
        .unwrap_or_else(render_unknown)
}

/// The parts of a well-formed credential we are willing to surface.
struct Parts<'a> {
    kind: &'a str,
    id: &'a str,
}

/// Split `ik_<kind>_<id>_<secret>` into its `kind` and `id`, requiring the
/// literal `ik` prefix and a non-empty secret tail. Returns `None` for any
/// other shape.
fn parse_segments(token: &str) -> Option<Parts<'_>> {
    let rest = token.strip_prefix("ik_")?;
    let (kind, rest) = rest.split_once('_')?;
    let (id, secret) = rest.split_once('_')?;
    match (kind.is_empty(), id.is_empty(), secret.is_empty()) {
        (false, false, false) => Some(Parts { kind, id }),
        _ => None,
    }
}

/// Build `ik_<kind>_<head>…` from a parsed credential.
fn render_known(parts: Parts<'_>) -> String {
    let head = head_of(parts.id);
    format!("ik_{}_{}{}", parts.kind, head, ELLIPSIS)
}

/// The fixed fallback head for unparseable input.
fn render_unknown() -> String {
    format!("{}{}", UNKNOWN_HEAD, ELLIPSIS)
}

/// The first [`ID_HEAD_LEN`] characters of `id` (fewer if it is shorter),
/// counted by `char` so we never split a multi-byte boundary.
fn head_of(id: &str) -> String {
    id.chars().take(ID_HEAD_LEN).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Secrets used in the fixtures. Every redaction must drop these entirely.
    const SDK_SECRET: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    const LIVE_SECRET: &str = "supersecretlivetail";

    #[test]
    fn redacts_to_prefix_and_head() {
        let cases = [
            // (token, expected redaction)
            (
                format!("ik_sdk_a1b2c3d4_{SDK_SECRET}"),
                "ik_sdk_a1b2c3d4\u{2026}",
            ),
            (
                format!("ik_live_proj1234_{LIVE_SECRET}"),
                "ik_live_proj1234\u{2026}",
            ),
            // id shorter than the head length is kept whole.
            ("ik_sdk_abc_tail".to_string(), "ik_sdk_abc\u{2026}"),
            // id exactly the head length.
            ("ik_live_12345678_tail".to_string(), "ik_live_12345678\u{2026}"),
            // unexpected kind still redacts; kind is not a secret.
            ("ik_foo_deadbeef99_tail".to_string(), "ik_foo_deadbeef\u{2026}"),
        ];
        for (token, expected) in cases {
            assert_eq!(redact(&token), expected, "token: {token}");
        }
    }

    #[test]
    fn unknown_shapes_fall_back_to_fixed_head() {
        let cases = [
            "",
            "ik",
            "ik_",
            "ik_sdk",
            "ik_sdk_",
            "ik_sdk_onlyid",   // missing secret segment
            "ik_sdk__tail",    // empty id
            "ik__id_tail",     // empty kind
            "not_a_token",
            "bearer ik_sdk_a1b2c3d4_secret", // not a bare token
        ];
        for token in cases {
            assert_eq!(redact(token), "ik\u{2026}", "token: {token:?}");
        }
    }

    #[test]
    fn secret_tail_never_survives() {
        let tokens = [
            format!("ik_sdk_a1b2c3d4_{SDK_SECRET}"),
            format!("ik_live_proj1234_{LIVE_SECRET}"),
            // secret embedded right after a short id.
            format!("ik_sdk_abc_{SDK_SECRET}"),
        ];
        for token in tokens {
            let out = redact(&token);
            assert!(!out.contains(SDK_SECRET), "leaked sdk secret in {out:?}");
            assert!(!out.contains(LIVE_SECRET), "leaked live secret in {out:?}");
            // The full token must never appear verbatim.
            assert!(!out.contains(&token), "leaked full token in {out:?}");
        }
    }

    #[test]
    fn keeps_at_most_head_len_of_id() {
        let out = redact("ik_sdk_0123456789abcdef_tail");
        // 8-char head, no more.
        assert_eq!(out, "ik_sdk_01234567\u{2026}");
        assert!(!out.contains("89abcdef"), "kept too much of the id: {out:?}");
    }

    #[test]
    fn never_panics_on_multibyte_input() {
        // Non-ASCII id must not split a char boundary.
        let out = redact("ik_sdk_\u{00e9}\u{00e9}\u{00e9}_secret");
        assert!(out.starts_with("ik_sdk_"));
        assert!(out.ends_with('\u{2026}'));
    }
}
