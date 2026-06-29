//! Workload specification: the declarative shape a caller hands to `ensure()`.
//!
//! A [`WorkloadSpec`] is pure data. This module owns the *pure* rules that guard
//! it before any request is built:
//!
//! * [`validate_spec`] — structural invariants plus the hard rule that a secret
//!   (`ik_live_…` / `ik_sdk_…`) must never be smuggled into a spec field.
//! * [`assert_sdk_token`] — the management plane only accepts an `ik_sdk_` token;
//!   an `ik_live_` data key is a *permission* failure, anything else is a config
//!   failure. The offending token is redacted to a prefix before it reaches the
//!   error message.
//!
//! By design neither `provider` nor `min_vram_gb` is part of the spec: the
//! platform derives placement, and we never let callers pin it.

use serde::{Deserialize, Serialize};

use crate::domain::enums::{Backend, ExecutionPolicy, TaskType};
use crate::domain::redact::redact;
use crate::errors::{CoreError, CoreResult, PermissionCode};

/// Prefix carried by data-plane keys (OpenAI-compatible surface).
const DATA_KEY_PREFIX: &str = "ik_live_";
/// Prefix carried by management-plane tokens (control surface).
const SDK_TOKEN_PREFIX: &str = "ik_sdk_";

/// A declarative workload definition.
///
/// `name`, `slug` and `model` are required; everything else is optional and
/// omitted from the wire body when `None`. `provider` and `min_vram_gb` are
/// deliberately absent — placement is the platform's job, never the caller's.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WorkloadSpec {
    pub name: String,
    pub slug: String,
    pub model: String,
    pub backend: Backend,
    pub project: Option<String>,
    pub description: Option<String>,
    pub command: Option<String>,
    pub vllm_version: Option<String>,
    pub task_type: Option<TaskType>,
    pub config: Option<serde_json::Value>,
    pub execution_policy: Option<ExecutionPolicy>,
    pub execution_policy_config: Option<serde_json::Value>,
    pub worker_id: Option<String>,
    pub gpu_resource_id: Option<String>,
}

/// Canonicalise a slug the **same way the control plane does** before it is
/// persisted: every non-alphanumeric run collapses to a single `-`, the whole
/// thing is lowercased, and leading/trailing dashes are trimmed. Keeping this in
/// lockstep with the server's `slugify` is what makes `ensure()` idempotent —
/// the slug we search by and send equals the slug the server stores.
///
/// (Mirror of `crates/manager/src/repo/workloads.rs::slugify`.)
pub fn canonical_slug(slug: &str) -> String {
    slug.chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .collect::<String>()
        .to_lowercase()
        .split('-')
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join("-")
}

/// A named string field of a spec, paired for required-field and secret checks.
struct NamedField<'a> {
    label: &'a str,
    value: &'a str,
}

/// The fields that must be present (non-empty after trimming).
fn required_fields(spec: &WorkloadSpec) -> [NamedField<'_>; 3] {
    [
        NamedField {
            label: "name",
            value: &spec.name,
        },
        NamedField {
            label: "slug",
            value: &spec.slug,
        },
        NamedField {
            label: "model",
            value: &spec.model,
        },
    ]
}

/// Every string-bearing field, used to scan for leaked secrets.
fn string_fields(spec: &WorkloadSpec) -> Vec<NamedField<'_>> {
    let mut fields = vec![
        NamedField {
            label: "name",
            value: &spec.name,
        },
        NamedField {
            label: "slug",
            value: &spec.slug,
        },
        NamedField {
            label: "model",
            value: &spec.model,
        },
    ];
    let optional: [(&str, &Option<String>); 6] = [
        ("project", &spec.project),
        ("description", &spec.description),
        ("command", &spec.command),
        ("vllm_version", &spec.vllm_version),
        ("worker_id", &spec.worker_id),
        ("gpu_resource_id", &spec.gpu_resource_id),
    ];
    optional
        .iter()
        .filter_map(|(label, opt)| opt.as_deref().map(|value| NamedField { label, value }))
        .for_each(|field| fields.push(field));
    fields
}

/// True when a value looks like any credential the caller must never inline.
fn looks_like_secret(value: &str) -> bool {
    value.starts_with(DATA_KEY_PREFIX) || value.starts_with(SDK_TOKEN_PREFIX)
}

/// Ensure a required field is non-empty (ignoring surrounding whitespace).
fn ensure_present(field: &NamedField<'_>) -> CoreResult<()> {
    match field.value.trim().is_empty() {
        true => Err(CoreError::Validation(format!(
            "{} must not be empty",
            field.label
        ))),
        false => Ok(()),
    }
}

/// Ensure a field does not carry a secret token.
fn ensure_no_secret(field: &NamedField<'_>) -> CoreResult<()> {
    match looks_like_secret(field.value) {
        true => Err(CoreError::Config(
            "secrets must never go in specs".to_owned(),
        )),
        false => Ok(()),
    }
}

/// Validate a spec without performing any IO.
///
/// Rules:
/// * `name`, `slug`, `model` must be non-empty.
/// * No string field may begin with `ik_live_` or `ik_sdk_`.
///
/// `backend` is already a typed enum, so it needs no string check.
pub fn validate_spec(spec: &WorkloadSpec) -> CoreResult<()> {
    required_fields(spec).iter().try_for_each(ensure_present)?;
    string_fields(spec).iter().try_for_each(ensure_no_secret)
}

/// Assert a token is usable on the management plane.
///
/// * An `ik_live_` data key is the *wrong credential type* → permission error.
/// * Anything that is not an `ik_sdk_` token is a configuration error.
///
/// The token is redacted to its prefix in any message it appears in.
pub fn assert_sdk_token(token: &str) -> CoreResult<()> {
    match token {
        t if t.starts_with(DATA_KEY_PREFIX) => Err(wrong_credential_type(t)),
        t if t.starts_with(SDK_TOKEN_PREFIX) => Ok(()),
        t => Err(not_an_sdk_token(t)),
    }
}

/// A data key was handed to the management plane.
fn wrong_credential_type(token: &str) -> CoreError {
    CoreError::Permission {
        code: PermissionCode::WrongCredentialType,
        message: format!(
            "{} is a data key; the management plane requires an ik_sdk_ token",
            redact(token)
        ),
    }
}

/// A token with no recognised credential prefix.
fn not_an_sdk_token(token: &str) -> CoreError {
    CoreError::Config(format!(
        "{} is not an ik_sdk_ management token",
        redact(token)
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A labelled spec mutator used by the table-driven validation tests.
    type Case = (&'static str, fn(&mut WorkloadSpec));

    fn base_spec() -> WorkloadSpec {
        WorkloadSpec {
            name: "Chat".to_owned(),
            slug: "chat".to_owned(),
            model: "qwen3".to_owned(),
            backend: Backend::Vllm,
            project: None,
            description: None,
            command: None,
            vllm_version: None,
            task_type: None,
            config: None,
            execution_policy: None,
            execution_policy_config: None,
            worker_id: None,
            gpu_resource_id: None,
        }
    }

    fn is_config(err: &CoreError) -> bool {
        matches!(err, CoreError::Config(_))
    }

    fn is_validation(err: &CoreError) -> bool {
        matches!(err, CoreError::Validation(_))
    }

    #[test]
    fn validate_accepts_a_minimal_spec() {
        assert!(validate_spec(&base_spec()).is_ok());
    }

    #[test]
    fn validate_accepts_a_custom_backend() {
        let mut spec = base_spec();
        spec.backend = Backend::Custom("echo".to_owned());
        assert!(validate_spec(&spec).is_ok());
    }

    #[test]
    fn spec_with_custom_backend_serializes_backend_to_its_slug() {
        let mut spec = base_spec();
        spec.backend = Backend::Custom("echo".to_owned());
        let value = serde_json::to_value(&spec).expect("serialize");
        assert_eq!(value["backend"], serde_json::json!("echo"));
    }

    #[test]
    fn spec_with_custom_backend_round_trips_through_serde() {
        let mut spec = base_spec();
        spec.backend = Backend::Custom("echo".to_owned());
        let json = serde_json::to_string(&spec).expect("serialize");
        let parsed: WorkloadSpec = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.backend, Backend::Custom("echo".to_owned()));
    }

    #[test]
    fn canonical_slug_matches_server_slugify() {
        // Must stay in lockstep with the server's `slugify`: lowercase, every
        // non-alphanumeric run → one dash, no leading/trailing dashes.
        let cases = [
            ("support-bot", "support-bot"),
            (
                "Gemma 4 26B (llama.cpp / GGUF) on R9700",
                "gemma-4-26b-llama-cpp-gguf-on-r9700",
            ),
            ("  Mixed_Case--Slug  ", "mixed-case-slug"),
            ("already-canonical", "already-canonical"),
            ("UPPER", "upper"),
            ("a__b..c", "a-b-c"),
            ("---", ""),
        ];
        for (input, expected) in cases {
            assert_eq!(canonical_slug(input), expected, "slug for {input:?}");
        }
    }

    #[test]
    fn validate_rejects_missing_required_fields() {
        let cases: [Case; 4] = [
            ("empty name", |s| s.name = String::new()),
            ("blank slug", |s| s.slug = "   ".to_owned()),
            ("empty model", |s| s.model = String::new()),
            ("whitespace name", |s| s.name = "\t".to_owned()),
        ];
        for (label, mutate) in cases {
            let mut spec = base_spec();
            mutate(&mut spec);
            let err = validate_spec(&spec).expect_err(label);
            assert!(
                is_validation(&err),
                "{label}: expected validation error, got {err:?}"
            );
        }
    }

    #[test]
    fn validate_rejects_secrets_in_any_string_field() {
        let cases: [Case; 6] = [
            ("live in name", |s| s.name = "ik_live_abc".to_owned()),
            ("sdk in slug", |s| s.slug = "ik_sdk_abc".to_owned()),
            ("live in model", |s| s.model = "ik_live_zzz".to_owned()),
            ("sdk in command", |s| {
                s.command = Some("ik_sdk_xyz".to_owned())
            }),
            ("live in description", |s| {
                s.description = Some("ik_live_d".to_owned())
            }),
            ("sdk in worker_id", |s| {
                s.worker_id = Some("ik_sdk_w".to_owned())
            }),
        ];
        for (label, mutate) in cases {
            let mut spec = base_spec();
            mutate(&mut spec);
            let err = validate_spec(&spec).expect_err(label);
            assert!(
                is_config(&err),
                "{label}: expected config error, got {err:?}"
            );
        }
    }

    #[test]
    fn validate_ignores_secret_lookalikes_inside_a_value() {
        let mut spec = base_spec();
        spec.description = Some("see ik_live_ docs".to_owned());
        assert!(validate_spec(&spec).is_ok());
    }

    #[test]
    fn assert_sdk_token_accepts_management_tokens() {
        assert!(assert_sdk_token("ik_sdk_deadbeef_0011").is_ok());
    }

    #[test]
    fn assert_sdk_token_rejects_data_keys_as_wrong_credential() {
        let err = assert_sdk_token("ik_live_secretvalue").expect_err("data key");
        match err {
            CoreError::Permission { code, message } => {
                assert_eq!(code, PermissionCode::WrongCredentialType);
                assert!(
                    !message.contains("secretvalue"),
                    "raw token leaked: {message}"
                );
            }
            other => panic!("expected permission error, got {other:?}"),
        }
    }

    #[test]
    fn assert_sdk_token_rejects_unknown_prefixes_as_config() {
        let cases = ["", "sk-openai", "bearer abc", "ik_live", "ik_sdk"];
        for token in cases {
            let err = assert_sdk_token(token).expect_err(token);
            assert!(
                is_config(&err),
                "{token:?}: expected config error, got {err:?}"
            );
        }
    }
}
