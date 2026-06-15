//! Wire types and pure mappers between a [`WorkloadSpec`] and the control-plane
//! request/response bodies.
//!
//! Everything here is PURE: parse / validate / transform / build. No IO, no
//! time, no logging. The transport layer owns the effects and calls into these
//! mappers to shape what it sends and to compute drift from what it reads back.
//!
//! ## Spec field-access contract
//!
//! These mappers read the following fields off [`WorkloadSpec`] (authored in the
//! sibling module `crate::domain::spec`):
//! `name`, `slug`, `description: Option<String>`, `task_type: TaskType`,
//! `backend: Backend`, `model: String`, `command: Option<String>`,
//! `vllm_version: Option<String>`, `worker_id: Option<String>`,
//! `gpu_resource_id: Option<String>`, `execution_policy: Option<ExecutionPolicy>`.
//! `provider` and `min_vram_gb` are intentionally never read or emitted.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::domain::enums::{Backend, ExecutionPolicy, TaskType};
use crate::domain::spec::WorkloadSpec;

/// Control-plane create body: `POST /api/projects/:project_id/workloads`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CreateWorkloadRequest {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub task_type: TaskType,
    pub backend: Backend,
    pub model_name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub config: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub worker_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu_resource_id: Option<String>,
    // The execution policy declared on the spec must ride the create body too —
    // otherwise a workload is born with the server's default policy and only
    // reconciles to the declared one on a later PATCH. Both are omitted when the
    // spec leaves them unset, falling back to the server default.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub execution_policy: Option<ExecutionPolicy>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub execution_policy_config: Option<Value>,
}

/// Control-plane update body: `PATCH /api/workloads/:id`. Every field is
/// optional; only declared changes are sent. `task_type` is not patchable, so
/// it is absent here by design.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct UpdateWorkloadRequest {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub backend: Option<Backend>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub config: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub worker_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu_resource_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub execution_policy: Option<ExecutionPolicy>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub execution_policy_config: Option<Value>,
}

/// Control-plane workload representation returned by GET / create / update.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WorkloadResponse {
    pub id: String,
    pub project_id: String,
    pub name: String,
    pub slug: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub task_type: TaskType,
    pub backend: Backend,
    pub model_name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub config: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub worker_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu_resource_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub execution_policy: Option<ExecutionPolicy>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub execution_policy_config: Option<Value>,
    pub created_at: String,
    pub updated_at: String,
}

/// A single drifted field: what the platform currently has (`from`) versus what
/// the spec declares (`to`). Values are opaque JSON so any field type fits.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FieldDiff {
    pub field: String,
    pub from: Value,
    pub to: Value,
}

/// Backend-specific `config`. For vllm / vllm-omni this is `{ command,
/// vllm_version? }`; absent for backends that take no launch command. Returns
/// `None` when the spec declares nothing to put in `config`.
pub fn build_config(spec: &WorkloadSpec) -> Option<Value> {
    let command = spec.command.as_ref()?;
    let mut map = serde_json::Map::new();
    map.insert("command".into(), Value::String(command.clone()));
    if let Some(version) = spec.vllm_version.as_ref() {
        map.insert("vllm_version".into(), Value::String(version.clone()));
    }
    Some(Value::Object(map))
}

/// Build the create body from a spec. `task_type` carries the spec value (the
/// server default of `text2text` is applied by the spec layer, not here).
pub fn to_create_request(spec: &WorkloadSpec) -> CreateWorkloadRequest {
    CreateWorkloadRequest {
        name: spec.name.clone(),
        description: spec.description.clone(),
        // task_type is required on the wire; default to text2text when unset
        // (matches the Manager's server-side default).
        task_type: spec.task_type.unwrap_or(TaskType::Text2Text),
        backend: spec.backend,
        model_name: spec.model.clone(),
        config: build_config(spec),
        worker_id: spec.worker_id.clone(),
        gpu_resource_id: spec.gpu_resource_id.clone(),
        execution_policy: spec.execution_policy,
        execution_policy_config: spec.execution_policy_config.clone(),
    }
}

/// Build a minimal PATCH body that carries exactly the drifted fields. Each
/// diff's `to` value is the spec's desired value, decoded back into the typed
/// request field. Unknown field names are ignored.
pub fn to_update_request(spec: &WorkloadSpec, diffs: &[FieldDiff]) -> UpdateWorkloadRequest {
    diffs.iter().fold(UpdateWorkloadRequest::default(), |req, diff| {
        apply_diff_to_update(req, spec, diff)
    })
}

/// Compute drift for the fields the spec declares. `task_type` is never
/// compared because it is not patchable. Ordering is stable (declaration order).
pub fn diff_workload(spec: &WorkloadSpec, current: &WorkloadResponse) -> Vec<FieldDiff> {
    let policy_kind_changed = diff_execution_policy(&current.execution_policy, &spec.execution_policy);
    [
        diff_string("name", &current.name, &spec.name),
        diff_enum("backend", current.backend.as_str(), spec.backend.as_str()),
        diff_string("model_name", &current.model_name, &spec.model),
        diff_opt_string("description", &current.description, &spec.description),
        diff_config(&current.config, build_config(spec)),
        diff_opt_string("worker_id", &current.worker_id, &spec.worker_id),
        diff_opt_string("gpu_resource_id", &current.gpu_resource_id, &spec.gpu_resource_id),
        policy_kind_changed.clone(),
        diff_execution_policy_config(
            &current.execution_policy_config,
            &spec.execution_policy_config,
            policy_kind_changed.is_some(),
        ),
    ]
    .into_iter()
    .flatten()
    .collect()
}

/// One-line, human-readable summary of drifted field names (for logs / the
/// `Drift` error). Empty diffs render as `"<none>"`.
pub fn summarize_diffs(diffs: &[FieldDiff]) -> String {
    match diffs.is_empty() {
        true => "<none>".to_string(),
        false => diffs
            .iter()
            .map(|d| d.field.as_str())
            .collect::<Vec<_>>()
            .join(", "),
    }
}

// --- pure diff builders -------------------------------------------------------

/// Diff a required string field; `None` when equal.
fn diff_string(field: &str, current: &str, desired: &str) -> Option<FieldDiff> {
    match current == desired {
        true => None,
        false => Some(named_diff(field, json_str(current), json_str(desired))),
    }
}

/// Diff an enum rendered as its wire string; `None` when equal.
fn diff_enum(field: &str, current: &str, desired: &str) -> Option<FieldDiff> {
    diff_string(field, current, desired)
}

/// Diff an optional string the spec declares. A `None` on the spec side means
/// "not declared" and is never treated as drift.
fn diff_opt_string(
    field: &str,
    current: &Option<String>,
    desired: &Option<String>,
) -> Option<FieldDiff> {
    let want = desired.as_ref()?;
    match current.as_deref() == Some(want.as_str()) {
        true => None,
        false => Some(named_diff(field, opt_str_json(current), json_str(want))),
    }
}

/// Diff the backend `config`. Only compared when the spec declares one.
fn diff_config(current: &Option<Value>, desired: Option<Value>) -> Option<FieldDiff> {
    let want = desired?;
    let have = current.clone().unwrap_or(Value::Null);
    match have == want {
        true => None,
        false => Some(named_diff("config", have, want)),
    }
}

/// Diff `execution_policy` (enum) only when the spec declares one.
fn diff_execution_policy(
    current: &Option<ExecutionPolicy>,
    desired: &Option<ExecutionPolicy>,
) -> Option<FieldDiff> {
    let want = (*desired)?;
    match *current == Some(want) {
        true => None,
        false => Some(named_diff(
            "execution_policy",
            opt_enum_json(current.map(|p| p.as_str())),
            json_str(want.as_str()),
        )),
    }
}

/// Diff `execution_policy_config` (an opaque JSON blob), but only when the spec
/// declares one. Emitted when the desired config differs from the live config,
/// OR when the `execution_policy` kind itself changed: the Manager re-validates
/// the policy against whatever config it has on file, so switching e.g.
/// `fixed → autoscaling` MUST carry the new config in the same PATCH — otherwise
/// the server validates the new kind against the stale (often empty) config and
/// rejects it (e.g. "max_hourly_cost_usd must be greater than zero").
fn diff_execution_policy_config(
    current: &Option<Value>,
    desired: &Option<Value>,
    policy_kind_changed: bool,
) -> Option<FieldDiff> {
    let want = desired.as_ref()?;
    let have = current.clone().unwrap_or(Value::Null);
    match have == *want && !policy_kind_changed {
        true => None,
        false => Some(named_diff("execution_policy_config", have, want.clone())),
    }
}

// --- pure update assembly -----------------------------------------------------

/// Set the one [`UpdateWorkloadRequest`] field named by `diff`, pulling the
/// typed value from the spec so the wire body stays strongly typed.
fn apply_diff_to_update(
    mut req: UpdateWorkloadRequest,
    spec: &WorkloadSpec,
    diff: &FieldDiff,
) -> UpdateWorkloadRequest {
    match diff.field.as_str() {
        "name" => req.name = Some(spec.name.clone()),
        "description" => req.description = spec.description.clone(),
        "backend" => req.backend = Some(spec.backend),
        "model_name" => req.model_name = Some(spec.model.clone()),
        "config" => req.config = build_config(spec),
        "worker_id" => req.worker_id = spec.worker_id.clone(),
        "gpu_resource_id" => req.gpu_resource_id = spec.gpu_resource_id.clone(),
        "execution_policy" => req.execution_policy = spec.execution_policy,
        "execution_policy_config" => {
            req.execution_policy_config = spec.execution_policy_config.clone()
        }
        _ => {}
    }
    req
}

// --- tiny value helpers -------------------------------------------------------

fn named_diff(field: &str, from: Value, to: Value) -> FieldDiff {
    FieldDiff {
        field: field.to_string(),
        from,
        to,
    }
}

fn json_str(s: &str) -> Value {
    Value::String(s.to_string())
}

fn opt_str_json(value: &Option<String>) -> Value {
    value
        .as_ref()
        .map(|s| Value::String(s.clone()))
        .unwrap_or(Value::Null)
}

fn opt_enum_json(value: Option<&str>) -> Value {
    value.map(json_str).unwrap_or(Value::Null)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn base_spec() -> WorkloadSpec {
        WorkloadSpec {
            name: "support-bot".to_string(),
            slug: "support-bot".to_string(),
            model: "meta-llama/Llama-3.1-8B-Instruct".to_string(),
            backend: Backend::Vllm,
            project: None,
            description: None,
            command: Some("vllm serve meta-llama/Llama-3.1-8B-Instruct".to_string()),
            vllm_version: None,
            task_type: Some(TaskType::Text2Text),
            config: None,
            execution_policy: None,
            execution_policy_config: None,
            worker_id: None,
            gpu_resource_id: None,
        }
    }

    fn base_response() -> WorkloadResponse {
        WorkloadResponse {
            id: "wl_1".to_string(),
            project_id: "proj_1".to_string(),
            name: "support-bot".to_string(),
            slug: "support-bot".to_string(),
            description: None,
            task_type: TaskType::Text2Text,
            backend: Backend::Vllm,
            model_name: "meta-llama/Llama-3.1-8B-Instruct".to_string(),
            config: build_config(&base_spec()),
            worker_id: None,
            gpu_resource_id: None,
            execution_policy: None,
            execution_policy_config: None,
            created_at: "2026-06-15T00:00:00Z".to_string(),
            updated_at: "2026-06-15T00:00:00Z".to_string(),
        }
    }

    #[test]
    fn build_config_shapes_vllm_command_and_version() {
        let cases = [
            (None, None, None),
            (
                Some("vllm serve x"),
                None,
                Some(json!({ "command": "vllm serve x" })),
            ),
            (
                Some("vllm serve x"),
                Some("0.6.2"),
                Some(json!({ "command": "vllm serve x", "vllm_version": "0.6.2" })),
            ),
        ];
        for (command, version, expected) in cases {
            let spec = WorkloadSpec {
                command: command.map(str::to_string),
                vllm_version: version.map(str::to_string),
                ..base_spec()
            };
            assert_eq!(build_config(&spec), expected);
        }
    }

    #[test]
    fn to_create_request_maps_spec_fields() {
        struct Case {
            mutate: fn(WorkloadSpec) -> WorkloadSpec,
            expected: CreateWorkloadRequest,
        }
        let cases = [
            Case {
                mutate: |s| s,
                expected: CreateWorkloadRequest {
                    name: "support-bot".to_string(),
                    description: None,
                    task_type: TaskType::Text2Text,
                    backend: Backend::Vllm,
                    model_name: "meta-llama/Llama-3.1-8B-Instruct".to_string(),
                    config: Some(json!({ "command": "vllm serve meta-llama/Llama-3.1-8B-Instruct" })),
                    worker_id: None,
                    gpu_resource_id: None,
                    execution_policy: None,
                    execution_policy_config: None,
                },
            },
            Case {
                mutate: |s| WorkloadSpec {
                    description: Some("desc".to_string()),
                    task_type: Some(TaskType::Embedding),
                    backend: Backend::Ollama,
                    command: None,
                    worker_id: Some("w1".to_string()),
                    gpu_resource_id: Some("g1".to_string()),
                    execution_policy: Some(ExecutionPolicy::Autoscaling),
                    execution_policy_config: Some(json!({ "min_replicas": 1, "max_replicas": 1 })),
                    ..s
                },
                expected: CreateWorkloadRequest {
                    name: "support-bot".to_string(),
                    description: Some("desc".to_string()),
                    task_type: TaskType::Embedding,
                    backend: Backend::Ollama,
                    model_name: "meta-llama/Llama-3.1-8B-Instruct".to_string(),
                    config: None,
                    worker_id: Some("w1".to_string()),
                    gpu_resource_id: Some("g1".to_string()),
                    execution_policy: Some(ExecutionPolicy::Autoscaling),
                    execution_policy_config: Some(json!({ "min_replicas": 1, "max_replicas": 1 })),
                },
            },
        ];
        for case in cases {
            let spec = (case.mutate)(base_spec());
            assert_eq!(to_create_request(&spec), case.expected);
        }
    }

    #[test]
    fn create_request_omits_none_fields_in_json() {
        let spec = base_spec();
        let value = serde_json::to_value(to_create_request(&spec)).expect("serialize");
        let obj = value.as_object().expect("object");
        assert!(!obj.contains_key("description"));
        assert!(!obj.contains_key("worker_id"));
        assert!(obj.contains_key("task_type"));
        assert_eq!(obj.get("backend"), Some(&json!("vllm")));
        // task_type rides the existing TaskType serde, whatever it renders to.
        let expected_task = serde_json::to_value(TaskType::Text2Text).expect("task_type");
        assert_eq!(obj.get("task_type"), Some(&expected_task));
        // An undeclared policy is omitted, so the server applies its default.
        assert!(!obj.contains_key("execution_policy"));
        assert!(!obj.contains_key("execution_policy_config"));
    }

    #[test]
    fn create_request_carries_declared_execution_policy() {
        // A declared policy must ride the create body, not just a later PATCH —
        // otherwise the workload is born with the wrong scheduling.
        let spec = WorkloadSpec {
            execution_policy: Some(ExecutionPolicy::Autoscaling),
            execution_policy_config: Some(json!({ "min_replicas": 1, "max_replicas": 1 })),
            ..base_spec()
        };
        let value = serde_json::to_value(to_create_request(&spec)).expect("serialize");
        let obj = value.as_object().expect("object");
        assert_eq!(obj.get("execution_policy"), Some(&json!("autoscaling")));
        assert_eq!(
            obj.get("execution_policy_config"),
            Some(&json!({ "min_replicas": 1, "max_replicas": 1 }))
        );
    }

    #[test]
    fn diff_workload_table() {
        struct Case {
            name: &'static str,
            mutate_spec: fn(WorkloadSpec) -> WorkloadSpec,
            mutate_resp: fn(WorkloadResponse) -> WorkloadResponse,
            expected_fields: &'static [&'static str],
        }
        let cases = [
            Case {
                name: "in sync",
                mutate_spec: |s| s,
                mutate_resp: |r| r,
                expected_fields: &[],
            },
            Case {
                name: "model renamed",
                mutate_spec: |s| WorkloadSpec {
                    model: "new-model".to_string(),
                    ..s
                },
                mutate_resp: |r| r,
                expected_fields: &["model_name"],
            },
            Case {
                name: "backend changed",
                mutate_spec: |s| WorkloadSpec {
                    backend: Backend::Sglang,
                    ..s
                },
                mutate_resp: |r| r,
                expected_fields: &["backend"],
            },
            Case {
                name: "task_type ignored even when different",
                mutate_spec: |s| WorkloadSpec {
                    task_type: Some(TaskType::Reranker),
                    ..s
                },
                mutate_resp: |r| WorkloadResponse {
                    task_type: TaskType::Text2Text,
                    ..r
                },
                expected_fields: &[],
            },
            Case {
                name: "spec-undeclared optional never drifts",
                mutate_spec: |s| WorkloadSpec {
                    description: None,
                    worker_id: None,
                    ..s
                },
                mutate_resp: |r| WorkloadResponse {
                    description: Some("server set".to_string()),
                    worker_id: Some("server-w".to_string()),
                    ..r
                },
                expected_fields: &[],
            },
            Case {
                name: "declared description drifts",
                mutate_spec: |s| WorkloadSpec {
                    description: Some("want".to_string()),
                    ..s
                },
                mutate_resp: |r| WorkloadResponse {
                    description: Some("have".to_string()),
                    ..r
                },
                expected_fields: &["description"],
            },
            Case {
                name: "config command drifts",
                mutate_spec: |s| WorkloadSpec {
                    command: Some("vllm serve changed".to_string()),
                    ..s
                },
                mutate_resp: |r| r,
                expected_fields: &["config"],
            },
            Case {
                name: "execution policy drifts",
                mutate_spec: |s| WorkloadSpec {
                    execution_policy: Some(ExecutionPolicy::Autoscaling),
                    ..s
                },
                mutate_resp: |r| WorkloadResponse {
                    execution_policy: Some(ExecutionPolicy::Fixed),
                    ..r
                },
                expected_fields: &["execution_policy"],
            },
            Case {
                // Switching the policy kind must carry the new config in the same
                // PATCH; otherwise the Manager validates `autoscaling` against the
                // stale config and rejects it. Regression test for the
                // "max_hourly_cost_usd must be greater than zero" failure on a
                // re-run where the live workload was `fixed` with an empty config.
                name: "policy kind change carries its config",
                mutate_spec: |s| WorkloadSpec {
                    execution_policy: Some(ExecutionPolicy::Autoscaling),
                    execution_policy_config: Some(json!({ "max_hourly_cost_usd": 5.0 })),
                    ..s
                },
                mutate_resp: |r| WorkloadResponse {
                    execution_policy: Some(ExecutionPolicy::Fixed),
                    execution_policy_config: Some(json!({})),
                    ..r
                },
                expected_fields: &["execution_policy", "execution_policy_config"],
            },
            Case {
                // Same policy kind, but the declared config drifts from the live
                // one — the config alone must be patched.
                name: "execution policy config drifts",
                mutate_spec: |s| WorkloadSpec {
                    execution_policy: Some(ExecutionPolicy::Autoscaling),
                    execution_policy_config: Some(json!({ "max_hourly_cost_usd": 9.0 })),
                    ..s
                },
                mutate_resp: |r| WorkloadResponse {
                    execution_policy: Some(ExecutionPolicy::Autoscaling),
                    execution_policy_config: Some(json!({ "max_hourly_cost_usd": 5.0 })),
                    ..r
                },
                expected_fields: &["execution_policy_config"],
            },
            Case {
                // Spec declares a config equal to the live one and the kind is
                // unchanged — no drift, no needless PATCH.
                name: "execution policy config in sync",
                mutate_spec: |s| WorkloadSpec {
                    execution_policy: Some(ExecutionPolicy::Autoscaling),
                    execution_policy_config: Some(json!({ "max_hourly_cost_usd": 5.0 })),
                    ..s
                },
                mutate_resp: |r| WorkloadResponse {
                    execution_policy: Some(ExecutionPolicy::Autoscaling),
                    execution_policy_config: Some(json!({ "max_hourly_cost_usd": 5.0 })),
                    ..r
                },
                expected_fields: &[],
            },
            Case {
                name: "multiple fields, stable order",
                mutate_spec: |s| WorkloadSpec {
                    name: "renamed".to_string(),
                    model: "new-model".to_string(),
                    ..s
                },
                mutate_resp: |r| r,
                expected_fields: &["name", "model_name"],
            },
        ];

        for case in cases {
            let spec = (case.mutate_spec)(base_spec());
            let resp = (case.mutate_resp)(base_response());
            let diffs = diff_workload(&spec, &resp);
            let fields: Vec<&str> = diffs.iter().map(|d| d.field.as_str()).collect();
            assert_eq!(fields, case.expected_fields, "case: {}", case.name);
        }
    }

    #[test]
    fn diff_workload_carries_from_and_to_values() {
        let spec = WorkloadSpec {
            model: "desired-model".to_string(),
            ..base_spec()
        };
        let diffs = diff_workload(&spec, &base_response());
        assert_eq!(diffs.len(), 1);
        let diff = diffs.first().expect("one diff");
        assert_eq!(diff.field, "model_name");
        assert_eq!(diff.from, json!("meta-llama/Llama-3.1-8B-Instruct"));
        assert_eq!(diff.to, json!("desired-model"));
    }

    #[test]
    fn to_update_request_carries_only_drifted_fields() {
        let spec = WorkloadSpec {
            model: "new-model".to_string(),
            description: Some("new desc".to_string()),
            ..base_spec()
        };
        let diffs = diff_workload(&spec, &base_response());
        let update = to_update_request(&spec, &diffs);
        assert_eq!(update.model_name, Some("new-model".to_string()));
        assert_eq!(update.description, Some("new desc".to_string()));
        assert_eq!(update.name, None);
        assert_eq!(update.backend, None);
    }

    #[test]
    fn update_request_omits_untouched_fields_in_json() {
        let spec = WorkloadSpec {
            model: "new-model".to_string(),
            ..base_spec()
        };
        let diffs = diff_workload(&spec, &base_response());
        let value = serde_json::to_value(to_update_request(&spec, &diffs)).expect("serialize");
        let obj = value.as_object().expect("object");
        assert_eq!(obj.get("model_name"), Some(&json!("new-model")));
        assert!(!obj.contains_key("name"));
        assert!(!obj.contains_key("task_type"));
    }

    #[test]
    fn update_request_carries_policy_config_when_kind_switches() {
        // Regression: re-running ensure() against a live workload that is
        // `fixed` with an empty config must PATCH both the new kind and its
        // config, so the Manager never validates `autoscaling` against `{}`.
        let spec = WorkloadSpec {
            execution_policy: Some(ExecutionPolicy::Autoscaling),
            execution_policy_config: Some(json!({
                "min_workers": 1,
                "max_workers": 1,
                "max_hourly_cost_usd": 5.0,
            })),
            ..base_spec()
        };
        let live = WorkloadResponse {
            execution_policy: Some(ExecutionPolicy::Fixed),
            execution_policy_config: Some(json!({})),
            ..base_response()
        };
        let diffs = diff_workload(&spec, &live);
        let value = serde_json::to_value(to_update_request(&spec, &diffs)).expect("serialize");
        let obj = value.as_object().expect("object");
        assert_eq!(obj.get("execution_policy"), Some(&json!("autoscaling")));
        assert_eq!(
            obj.get("execution_policy_config"),
            Some(&json!({
                "min_workers": 1,
                "max_workers": 1,
                "max_hourly_cost_usd": 5.0,
            }))
        );
    }

    #[test]
    fn summarize_diffs_lists_or_none() {
        assert_eq!(summarize_diffs(&[]), "<none>");
        let diffs = vec![
            named_diff("name", json!("a"), json!("b")),
            named_diff("model_name", json!("c"), json!("d")),
        ];
        assert_eq!(summarize_diffs(&diffs), "name, model_name");
    }
}
