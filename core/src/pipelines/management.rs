//! `ensure()` — idempotent provision-or-reconcile for a single workload.
//!
//! The pipeline is a thin choreography over an [`HttpPort`]: list the project's
//! workloads, match the spec's slug, then either create it (absent) or act on
//! the drift per the caller's [`OnDrift`] policy. Every step is a small named
//! function; the pure ones (request building, list parsing, slug matching) are
//! kept apart from the three effectful ones that touch the network
//! (`find_by_slug`, `create_workload`, `update_workload`).
//!
//! There is no `send`/`HttpResponse` on the port: [`HttpPort::request_json`]
//! already maps every non-2xx outcome to a [`CoreError`] and hands back the
//! decoded [`serde_json::Value`] on success, which we decode into the typed
//! wire bodies. Tokens are forwarded to the port verbatim and only ever appear
//! in logs through [`redact`].

use serde_json::Value;

use crate::domain::enums::OnDrift;
use crate::domain::redact::redact;
use crate::domain::spec::WorkloadSpec;
use crate::domain::wire::{
    diff_workload, summarize_diffs, to_create_request, to_update_request, FieldDiff,
    WorkloadResponse,
};
use crate::errors::{CoreError, CoreResult};
use crate::ports::http::{HttpMethod, HttpPort, HttpRequest};

/// Addresses a workload's OpenAI-compatible data-plane endpoint
/// (`/endpoint/:project_slug/:workload_slug/v1/...`). Handed to a `DataClient`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EndpointRef {
    pub project_slug: String,
    pub workload_slug: String,
}

/// The per-call context shared by every step: the transport handle plus the
/// resolved control-plane address and credential. Bundling these keeps each
/// step's signature small and threads the same `sdk_token` to redact in logs.
struct Ctx<'a> {
    http: &'a dyn HttpPort,
    base_url: &'a str,
    sdk_token: &'a str,
    project_id: &'a str,
}

/// Idempotently make the live workload match `spec`, returning its [`EndpointRef`].
///
/// 1. List the project's workloads and find one whose slug equals `spec.slug`.
/// 2. Absent → create it (or, under [`OnDrift::DryRun`], plan the create only).
/// 3. Present → compute drift and act per `on_drift`:
///    [`Reconcile`](OnDrift::Reconcile) PATCHes the diff,
///    [`Fail`](OnDrift::Fail) returns [`CoreError::Drift`],
///    [`DryRun`](OnDrift::DryRun) plans only,
///    [`Warn`](OnDrift::Warn) logs and leaves it,
///    [`Ignore`](OnDrift::Ignore) silently leaves it.
pub async fn ensure(
    http: &dyn HttpPort,
    base_url: &str,
    sdk_token: &str,
    project_id: &str,
    spec: &WorkloadSpec,
    on_drift: OnDrift,
) -> CoreResult<EndpointRef> {
    let ctx = Ctx { http, base_url, sdk_token, project_id };
    match find_by_slug(&ctx, &spec.slug).await? {
        None => ensure_absent(&ctx, spec, on_drift).await,
        Some(live) => ensure_present(&ctx, spec, &live, on_drift).await,
    }
}

/// Slug absent on the platform: create it, or plan the create under `DryRun`.
async fn ensure_absent(ctx: &Ctx<'_>, spec: &WorkloadSpec, on_drift: OnDrift) -> CoreResult<EndpointRef> {
    match on_drift {
        OnDrift::DryRun => Ok(plan_create(ctx, spec)),
        _ => {
            let created = create_workload(ctx, spec).await?;
            Ok(build_endpoint_ref(ctx, spec, &created.slug))
        }
    }
}

/// Slug present: assess drift, then branch on the policy.
async fn ensure_present(
    ctx: &Ctx<'_>,
    spec: &WorkloadSpec,
    live: &WorkloadResponse,
    on_drift: OnDrift,
) -> CoreResult<EndpointRef> {
    let diffs = diff_workload(spec, live);
    match diffs.is_empty() {
        true => Ok(build_endpoint_ref(ctx, spec, &live.slug)),
        false => resolve_drift(ctx, spec, live, &diffs, on_drift).await,
    }
}

/// Live state drifted — apply the chosen [`OnDrift`] policy.
async fn resolve_drift(
    ctx: &Ctx<'_>,
    spec: &WorkloadSpec,
    live: &WorkloadResponse,
    diffs: &[FieldDiff],
    on_drift: OnDrift,
) -> CoreResult<EndpointRef> {
    match on_drift {
        OnDrift::Reconcile => {
            let updated = update_workload(ctx, &live.id, spec, diffs).await?;
            Ok(build_endpoint_ref(ctx, spec, &updated.slug))
        }
        OnDrift::Fail => Err(CoreError::Drift {
            fields: summarize_diffs(diffs),
        }),
        OnDrift::DryRun => Ok(plan_update(ctx, spec, live, diffs)),
        OnDrift::Warn => Ok(warn_drift(ctx, spec, live, diffs)),
        OnDrift::Ignore => Ok(build_endpoint_ref(ctx, spec, &live.slug)),
    }
}

/* ------------------------------- effects -------------------------------- */

/// `GET /api/projects/:project_id/workloads`, then find the matching slug.
async fn find_by_slug(ctx: &Ctx<'_>, slug: &str) -> CoreResult<Option<WorkloadResponse>> {
    let url = join_url(ctx.base_url, &format!("/api/projects/{}/workloads", ctx.project_id));
    let request = HttpRequest::empty(HttpMethod::Get, url, ctx.sdk_token);
    let value = ctx.http.request_json(request).await?;
    let workloads = parse_workload_list(value)?;
    Ok(find_slug(&workloads, slug))
}

/// `POST /api/projects/:project_id/workloads` with the create body.
async fn create_workload(ctx: &Ctx<'_>, spec: &WorkloadSpec) -> CoreResult<WorkloadResponse> {
    let url = join_url(ctx.base_url, &format!("/api/projects/{}/workloads", ctx.project_id));
    let body = serde_json::to_value(to_create_request(spec))?;
    let request = HttpRequest::with_body(HttpMethod::Post, url, ctx.sdk_token, body);
    let value = ctx.http.request_json(request).await?;
    decode_workload(value)
}

/// `PATCH /api/workloads/:id` carrying only the drifted fields.
async fn update_workload(
    ctx: &Ctx<'_>,
    workload_id: &str,
    spec: &WorkloadSpec,
    diffs: &[FieldDiff],
) -> CoreResult<WorkloadResponse> {
    let url = join_url(ctx.base_url, &format!("/api/workloads/{workload_id}"));
    let body = serde_json::to_value(to_update_request(spec, diffs))?;
    let request = HttpRequest::with_body(HttpMethod::Patch, url, ctx.sdk_token, body);
    let value = ctx.http.request_json(request).await?;
    decode_workload(value)
}

/* -------------------------------- parsing ------------------------------- */

/// Decode the list endpoint body, tolerating both a bare array and a
/// `{ "workloads": [...] }` envelope. A `null` body decodes to an empty list.
fn parse_workload_list(value: Value) -> CoreResult<Vec<WorkloadResponse>> {
    match value {
        Value::Null => Ok(Vec::new()),
        Value::Object(map) if map.contains_key("workloads") => decode_envelope(map),
        other => Ok(serde_json::from_value(other)?),
    }
}

/// Pull the `workloads` array out of an envelope object and decode it.
fn decode_envelope(mut map: serde_json::Map<String, Value>) -> CoreResult<Vec<WorkloadResponse>> {
    let array = map.remove("workloads").unwrap_or(Value::Null);
    match array {
        Value::Null => Ok(Vec::new()),
        other => Ok(serde_json::from_value(other)?),
    }
}

/// Decode a single workload returned by create / update.
fn decode_workload(value: Value) -> CoreResult<WorkloadResponse> {
    Ok(serde_json::from_value(value)?)
}

/// Pure: the workload whose slug matches, if any.
fn find_slug(workloads: &[WorkloadResponse], slug: &str) -> Option<WorkloadResponse> {
    workloads.iter().find(|w| w.slug == slug).cloned()
}

/* --------------------------- urls, refs, logs --------------------------- */

/// Join `base_url` + an absolute `path` into one URL with no doubled slash.
fn join_url(base_url: &str, path: &str) -> String {
    let base = base_url.trim_end_matches('/');
    let path = path.trim_start_matches('/');
    format!("{base}/{path}")
}

/// The data-plane ref for a workload: the spec's declared project slug when
/// present, else the resolved `project_id`; the workload slug as it landed.
fn build_endpoint_ref(ctx: &Ctx<'_>, spec: &WorkloadSpec, workload_slug: &str) -> EndpointRef {
    EndpointRef {
        project_slug: resolve_project_slug(spec, ctx.project_id),
        workload_slug: workload_slug.to_string(),
    }
}

/// `spec.project` when it is a non-empty string, otherwise the `project_id`.
fn resolve_project_slug(spec: &WorkloadSpec, project_id: &str) -> String {
    spec.project
        .as_deref()
        .filter(|p| !p.is_empty())
        .unwrap_or(project_id)
        .to_string()
}

/// `DryRun` of a create: log the plan, mutate nothing, hand back a usable ref
/// addressing the declared slug. The token only ever appears redacted.
fn plan_create(ctx: &Ctx<'_>, spec: &WorkloadSpec) -> EndpointRef {
    tracing::info!(
        action = "create",
        slug = %spec.slug,
        project = %resolve_project_slug(spec, ctx.project_id),
        token = %redact(ctx.sdk_token),
        "ensure dry-run: would create workload; no mutation performed",
    );
    build_endpoint_ref(ctx, spec, &spec.slug)
}

/// `DryRun` of an update: log the planned diff, mutate nothing, hand back a ref.
fn plan_update(
    ctx: &Ctx<'_>,
    spec: &WorkloadSpec,
    live: &WorkloadResponse,
    diffs: &[FieldDiff],
) -> EndpointRef {
    tracing::info!(
        action = "update",
        slug = %live.slug,
        diff = %summarize_diffs(diffs),
        token = %redact(ctx.sdk_token),
        "ensure dry-run: would reconcile drift; no mutation performed",
    );
    build_endpoint_ref(ctx, spec, &live.slug)
}

/// `Warn`: log the drift and leave the live workload untouched.
fn warn_drift(
    ctx: &Ctx<'_>,
    spec: &WorkloadSpec,
    live: &WorkloadResponse,
    diffs: &[FieldDiff],
) -> EndpointRef {
    tracing::warn!(
        slug = %live.slug,
        workload_id = %live.id,
        diff = %summarize_diffs(diffs),
        token = %redact(ctx.sdk_token),
        "workload drifted from spec but on_drift=warn; leaving it untouched",
    );
    build_endpoint_ref(ctx, spec, &live.slug)
}

/* --------------------------------- tests -------------------------------- */

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::enums::{Backend, TaskType};
    use serde_json::json;

    /// Build a fixture JSON body for one workload. Enum fields ride their real
    /// serde wire forms (not hardcoded strings) so the test stays correct
    /// regardless of how `TaskType`/`Backend` render.
    fn workload_json(slug: &str, task_type: TaskType, backend: Backend) -> Value {
        json!({
            "id": format!("wl-{slug}"),
            "project_id": "proj-uuid",
            "name": slug,
            "slug": slug,
            "task_type": task_type,
            "backend": backend,
            "model_name": "Qwen/Qwen2.5-7B-Instruct",
            "created_at": "2026-01-01T00:00:00Z",
            "updated_at": "2026-01-01T00:00:00Z"
        })
    }

    fn workload(slug: &str) -> WorkloadResponse {
        let value = workload_json(slug, TaskType::Text2Text, Backend::Vllm);
        serde_json::from_value(value).expect("decode workload fixture")
    }

    #[test]
    fn find_slug_matches_on_exact_slug() {
        let workloads = vec![workload("alpha"), workload("beta")];
        assert_eq!(
            find_slug(&workloads, "beta").map(|w| w.slug),
            Some("beta".to_string())
        );
        assert_eq!(
            find_slug(&workloads, "beta").map(|w| w.id),
            Some("wl-beta".to_string())
        );
        assert!(find_slug(&workloads, "missing").is_none());
        assert!(find_slug(&[], "anything").is_none());
    }

    #[test]
    fn parse_workload_list_tolerates_array_and_envelope() {
        let bare = json!([workload_json("x", TaskType::Text2Text, Backend::Vllm)]);
        let from_array = parse_workload_list(bare).expect("array decodes");
        assert_eq!(from_array.len(), 1);
        assert_eq!(from_array[0].backend, Backend::Vllm);
        assert_eq!(from_array[0].task_type, TaskType::Text2Text);

        let enveloped =
            json!({ "workloads": [workload_json("y", TaskType::Embedding, Backend::Ollama)] });
        let from_envelope = parse_workload_list(enveloped).expect("envelope decodes");
        assert_eq!(from_envelope.len(), 1);
        assert_eq!(from_envelope[0].slug, "y");
        assert_eq!(from_envelope[0].backend, Backend::Ollama);

        assert!(parse_workload_list(Value::Null).expect("null is empty").is_empty());
        assert!(parse_workload_list(json!({ "workloads": null }))
            .expect("null envelope is empty")
            .is_empty());
    }

    #[test]
    fn resolve_project_slug_prefers_spec_then_falls_back() {
        let mut spec = base_spec();
        spec.project = Some("acme".to_string());
        assert_eq!(resolve_project_slug(&spec, "proj-uuid"), "acme");

        spec.project = None;
        assert_eq!(resolve_project_slug(&spec, "proj-uuid"), "proj-uuid");

        spec.project = Some(String::new());
        assert_eq!(resolve_project_slug(&spec, "proj-uuid"), "proj-uuid");
    }

    #[test]
    fn join_url_normalizes_slashes() {
        assert_eq!(
            join_url("https://api.test/", "/api/workloads/wl-1"),
            "https://api.test/api/workloads/wl-1"
        );
        assert_eq!(
            join_url("https://api.test", "api/workloads/wl-1"),
            "https://api.test/api/workloads/wl-1"
        );
    }

    fn base_spec() -> WorkloadSpec {
        WorkloadSpec {
            name: "chat".to_string(),
            slug: "chat".to_string(),
            model: "qwen3".to_string(),
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
}