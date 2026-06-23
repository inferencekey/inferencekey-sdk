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

use std::sync::Arc;
use std::time::Duration;

use futures_util::stream::{self, BoxStream, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::domain::enums::OnDrift;
use crate::domain::redact::redact;
use crate::domain::spec::{canonical_slug, WorkloadSpec};
use crate::domain::wire::{
    diff_workload, summarize_diffs, to_create_request, to_update_request, FieldDiff,
    WorkloadResponse,
};
use crate::errors::{CoreError, CoreResult};
use crate::ports::http::{HttpMethod, HttpPort, HttpRequest, JsonStream};

/// Addresses a workload's OpenAI-compatible data-plane endpoint
/// (`/endpoint/:project_slug/:workload_slug/v1/...`). Handed to a `DataClient`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EndpointRef {
    pub project_slug: String,
    pub workload_slug: String,
}

/// One readiness progress update streamed while waiting for a workload to come
/// up. Mirrors the Manager's sanitized SSE payload: a normalized `phase`, a
/// human-readable `message`, milliseconds since the stream opened, and an
/// optional allow-listed bootstrap `step`. `phase == "ready"` means the
/// workload is serving; `phase == "error"` is a terminal failure.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReadinessEvent {
    pub phase: String,
    pub message: String,
    #[serde(default)]
    pub elapsed_ms: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub step: Option<String>,
}

/// Backoff schedule for reconnecting a dropped readiness stream: 1s, 2s, 4s,
/// then capped. Bounded so a flapping server can't blow the budget per attempt;
/// the caller's overall `timeoutMs` (enforced in the binding) is the hard
/// ceiling on total wait, so we don't also count attempts here.
const RECONNECT_BACKOFF_MS: [u64; 3] = [1_000, 2_000, 4_000];
const RECONNECT_BACKOFF_CAP_MS: u64 = 5_000;

/// Backoff delay (ms) for the Nth consecutive failed (re)connect, 0-indexed.
fn backoff_ms(consecutive_failures: usize) -> u64 {
    RECONNECT_BACKOFF_MS
        .get(consecutive_failures)
        .copied()
        .unwrap_or(RECONNECT_BACKOFF_CAP_MS)
}

/// The sleep primitive used between reconnect attempts. Real runs use
/// [`TokioSleeper`]; tests inject a no-op/recording sleeper so the backoff is
/// exercised without real time passing.
pub trait Sleeper: Send + Sync {
    /// Sleep for `duration`, then resolve.
    fn sleep<'a>(&'a self, duration: Duration) -> BoxFuture<'a, ()>;
}

/// A future that resolves to `()` after a delay (no error channel).
type BoxFuture<'a, T> = std::pin::Pin<Box<dyn std::future::Future<Output = T> + Send + 'a>>;

/// Production sleeper backed by `tokio::time::sleep`.
pub struct TokioSleeper;

impl Sleeper for TokioSleeper {
    fn sleep<'a>(&'a self, duration: Duration) -> BoxFuture<'a, ()> {
        Box::pin(tokio::time::sleep(duration))
    }
}

/// Is this error transient (worth reconnecting) rather than permanent? A
/// dropped/refused connection or a 502/503/504 is transient — the manager may
/// be restarting. Auth/permission/not-found/bad-request and local errors are
/// permanent: reconnecting would only repeat them.
fn is_transient(err: &CoreError) -> bool {
    match err {
        CoreError::Network(_) => true,
        CoreError::Api { status, .. } => matches!(status, 502 | 503 | 504),
        _ => false,
    }
}

/// Whether a decoded event is terminal — the only legitimate way the stream
/// ends. A stream that closes *without* a terminal frame is a mid-flight drop
/// and triggers a reconnect (this is the bug being fixed: a bare end-of-stream
/// must not look like success).
fn is_terminal(event: &ReadinessEvent) -> bool {
    event.phase == "ready" || event.phase == "error"
}

/// Open the readiness progress stream for `workload_slug`, yielding one
/// [`ReadinessEvent`] per server-sent event until a terminal (`ready`/`error`)
/// frame arrives.
///
/// **Reconnection.** The control plane has no `Last-Event-ID`/offset, but it
/// emits a snapshot of the current phase on connect. So if the underlying SSE
/// stream drops mid-flight (e.g. the manager restarts) before a terminal frame,
/// this transparently reopens it — after a bounded backoff (1s, 2s, 4s, capped
/// at 5s) — and re-syncs from the fresh snapshot. A workload that reached
/// `ready` during the gap is observed on reconnect. Only a *terminal* frame
/// ends the stream; a permanent error (401/403/404/400) ends it too, propagated
/// as-is. Reconnection is intentionally housed here in the core so both the
/// Node and Python bindings get it for free; the caller's `timeoutMs` remains
/// the hard ceiling on total wait.
///
/// Control plane: authenticated with the `ik_sdk_` token, scoped to
/// `project_id` server-side. Non-readiness SSE frames (comments/keepalives) are
/// dropped by the transport, so every item here is a decoded payload.
pub async fn readiness_events(
    http: Arc<dyn HttpPort>,
    base_url: &str,
    sdk_token: &str,
    project_id: &str,
    workload_slug: &str,
) -> CoreResult<BoxStream<'static, CoreResult<ReadinessEvent>>> {
    readiness_events_with_sleeper(
        http,
        base_url,
        sdk_token,
        project_id,
        workload_slug,
        Arc::new(TokioSleeper),
    )
    .await
}

/// State threaded through the reconnecting [`stream::unfold`]. Owns everything
/// needed to reopen the SSE connection so the produced stream is `'static`.
struct ReadinessState {
    http: Arc<dyn HttpPort>,
    sleeper: Arc<dyn Sleeper>,
    req: HttpRequest,
    slug: String,
    token: String,
    /// The live SSE frame stream, or `None` before the first / after a drop.
    frames: Option<JsonStream>,
    /// Consecutive failed (re)connect attempts, for backoff. Reset on any frame.
    failures: usize,
    /// Set once a terminal frame is seen so the stream stops cleanly.
    done: bool,
}

/// [`readiness_events`] with an injectable [`Sleeper`] for deterministic tests.
pub async fn readiness_events_with_sleeper(
    http: Arc<dyn HttpPort>,
    base_url: &str,
    sdk_token: &str,
    project_id: &str,
    workload_slug: &str,
    sleeper: Arc<dyn Sleeper>,
) -> CoreResult<BoxStream<'static, CoreResult<ReadinessEvent>>> {
    let url = join_url(
        base_url,
        &format!("/api/projects/{project_id}/workloads/{workload_slug}/readiness-events"),
    );
    let req = HttpRequest::empty(HttpMethod::Get, url, sdk_token);

    let state = ReadinessState {
        http,
        sleeper,
        req,
        slug: workload_slug.to_string(),
        token: sdk_token.to_string(),
        frames: None,
        failures: 0,
        done: false,
    };

    let stream = stream::unfold(state, |state| async move {
        next_readiness(state).await
    });
    Ok(stream.boxed())
}

/// Pull the next [`ReadinessEvent`], reconnecting on a mid-flight drop. Returns
/// `None` to end the stream after a terminal frame, a permanent error, or once
/// `done`. Drives one step of the [`stream::unfold`] state machine.
async fn next_readiness(mut state: ReadinessState) -> Option<(CoreResult<ReadinessEvent>, ReadinessState)> {
    if state.done {
        return None;
    }
    loop {
        // Ensure we have a live frame stream, reconnecting (with backoff) if not.
        if state.frames.is_none() {
            if state.failures > 0 {
                let delay = backoff_ms(state.failures - 1);
                tracing::info!(
                    workload = %state.slug,
                    attempt = state.failures,
                    backoff_ms = delay,
                    token = %redact(&state.token),
                    "readiness stream dropped; reconnecting after backoff",
                );
                state.sleeper.sleep(Duration::from_millis(delay)).await;
            }
            match state.http.stream_sse(state.req.clone()).await {
                Ok(frames) => state.frames = Some(frames),
                Err(err) if is_transient(&err) => {
                    // Couldn't reopen — count it and loop to back off and retry.
                    state.failures += 1;
                    continue;
                }
                Err(err) => {
                    // Permanent: surface once, then end the stream.
                    state.done = true;
                    return Some((Err(err), state));
                }
            }
        }

        let frames = state.frames.as_mut().expect("frames set above");
        match frames.next().await {
            Some(Ok(value)) => {
                state.failures = 0; // a healthy frame resets the backoff
                match parse_readiness_event(&value) {
                    Ok(event) => {
                        if is_terminal(&event) {
                            state.done = true;
                        }
                        return Some((Ok(event), state));
                    }
                    // A malformed payload is a local decode error, not a drop:
                    // surface it and end (reconnecting wouldn't fix bad JSON).
                    Err(err) => {
                        state.done = true;
                        return Some((Err(err), state));
                    }
                }
            }
            Some(Err(err)) if is_transient(&err) => {
                // Mid-stream transport blip: drop the stream and reconnect.
                state.frames = None;
                state.failures += 1;
            }
            Some(Err(err)) => {
                // Permanent error mid-stream: surface and end.
                state.done = true;
                return Some((Err(err), state));
            }
            None => {
                // Stream ended WITHOUT a terminal frame — a mid-flight drop, not
                // success. Reconnect. (This is the core bug fix.)
                state.frames = None;
                state.failures += 1;
            }
        }
    }
}

/// Decode one readiness SSE payload into a [`ReadinessEvent`].
fn parse_readiness_event(value: &Value) -> CoreResult<ReadinessEvent> {
    serde_json::from_value(value.clone()).map_err(CoreError::from)
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
    // Canonicalise the slug to the form the server will persist, then thread the
    // normalised spec through the whole pipeline. This keeps the slug we search
    // by (`find_by_slug`) equal to the slug we send (`to_create_request`) and to
    // the one the server stores — so a re-`ensure()` always matches, instead of
    // creating a `-1` duplicate when the caller declared a non-canonical slug.
    let spec = &normalize_slug(spec);
    match find_by_slug(&ctx, &spec.slug).await? {
        None => ensure_absent(&ctx, spec, on_drift).await,
        Some(live) => ensure_present(&ctx, spec, &live, on_drift).await,
    }
}

/// Return a copy of `spec` whose `slug` is canonicalised. Cheap clone; the spec
/// is small and this runs once per `ensure()` call.
fn normalize_slug(spec: &WorkloadSpec) -> WorkloadSpec {
    let mut normalized = spec.clone();
    normalized.slug = canonical_slug(&spec.slug);
    normalized
}

/// Delete the workload named by `slug` from `project_id`, returning whether it
/// existed. Idempotent: a slug that isn't there resolves to `Ok(false)` rather
/// than an error, so cleanup-on-exit is safe to call unconditionally.
///
/// On the platform side this also tears down any cloud GPUs the autoscaler had
/// provisioned for the workload (private workers are only unassigned), so a
/// caller doesn't leak billable capacity. Authenticated with the `ik_sdk_`
/// control token, scoped to `project_id`.
pub async fn delete(
    http: &dyn HttpPort,
    base_url: &str,
    sdk_token: &str,
    project_id: &str,
    slug: &str,
) -> CoreResult<bool> {
    let ctx = Ctx { http, base_url, sdk_token, project_id };
    match find_by_slug(&ctx, slug).await? {
        None => Ok(false),
        Some(live) => {
            delete_workload(&ctx, &live.id).await?;
            Ok(true)
        }
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

/// `DELETE /api/workloads/:id`. The server responds `204 No Content` (decoded
/// as `Value::Null`), which we discard.
async fn delete_workload(ctx: &Ctx<'_>, workload_id: &str) -> CoreResult<()> {
    let url = join_url(ctx.base_url, &format!("/api/workloads/{workload_id}"));
    let request = HttpRequest::empty(HttpMethod::Delete, url, ctx.sdk_token);
    ctx.http.request_json(request).await?;
    Ok(())
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
    fn normalize_slug_canonicalizes_before_use() {
        // A non-canonical declared slug is normalized to the form the server
        // persists, so `find_by_slug` and the create body agree on identity.
        let spec = WorkloadSpec {
            name: "Support Bot".to_string(),
            slug: "Gemma 4 26B (llama.cpp) on R9700".to_string(),
            model: "m".to_string(),
            backend: Backend::Llamacpp,
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
        };
        assert_eq!(normalize_slug(&spec).slug, "gemma-4-26b-llama-cpp-on-r9700");
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

    // ── readiness reconnection ────────────────────────────────────────────
    use crate::ports::http::{BoxFuture, JsonStream};
    use futures_util::stream;
    use std::sync::Mutex;

    /// One scripted outcome of a `stream_sse` call: either a sequence of frames
    /// (each Ok value or a transient/permanent error) that then ends, or an
    /// immediate connect error.
    enum Attempt {
        /// A stream that yields these results in order, then ends (`None`).
        Stream(Vec<CoreResult<Value>>),
        /// `stream_sse` itself fails before returning a stream.
        ConnectErr(CoreError),
    }

    /// Fake transport: hands out one scripted [`Attempt`] per `stream_sse` call,
    /// in order. Reconnects pull the next attempt — exactly the manager-restart
    /// shape. Records how many times `stream_sse` was called.
    struct FakeHttp {
        attempts: Mutex<std::collections::VecDeque<Attempt>>,
        connects: Mutex<usize>,
    }

    impl FakeHttp {
        fn new(attempts: Vec<Attempt>) -> Self {
            Self {
                attempts: Mutex::new(attempts.into_iter().collect()),
                connects: Mutex::new(0),
            }
        }
        fn connect_count(&self) -> usize {
            *self.connects.lock().unwrap()
        }
    }

    impl HttpPort for FakeHttp {
        fn request_json<'a>(&'a self, _req: HttpRequest) -> BoxFuture<'a, Value> {
            Box::pin(async { Err(CoreError::NotImplemented("request_json".into())) })
        }
        fn stream_sse<'a>(&'a self, _req: HttpRequest) -> BoxFuture<'a, JsonStream> {
            *self.connects.lock().unwrap() += 1;
            let attempt = self.attempts.lock().unwrap().pop_front();
            Box::pin(async move {
                match attempt {
                    Some(Attempt::Stream(items)) => {
                        let s: JsonStream = Box::new(stream::iter(items));
                        Ok(s)
                    }
                    Some(Attempt::ConnectErr(err)) => Err(err),
                    // Ran out of script: behave like a connection refused so a
                    // runaway reconnect loop surfaces rather than hanging.
                    None => Err(CoreError::Network("no more attempts".into())),
                }
            })
        }
    }

    /// Sleeper that never actually sleeps; it records each requested delay so a
    /// test can assert the backoff schedule without real time passing.
    struct RecordingSleeper {
        delays_ms: Mutex<Vec<u64>>,
    }
    impl RecordingSleeper {
        fn new() -> Arc<Self> {
            Arc::new(Self { delays_ms: Mutex::new(Vec::new()) })
        }
        fn delays(&self) -> Vec<u64> {
            self.delays_ms.lock().unwrap().clone()
        }
    }
    impl Sleeper for RecordingSleeper {
        fn sleep<'a>(&'a self, duration: Duration) -> super::BoxFuture<'a, ()> {
            self.delays_ms.lock().unwrap().push(duration.as_millis() as u64);
            Box::pin(async {})
        }
    }

    fn ev(phase: &str) -> Value {
        json!({ "phase": phase, "message": phase, "elapsed_ms": 0 })
    }

    async fn collect_readiness(
        http: Arc<dyn HttpPort>,
        sleeper: Arc<dyn Sleeper>,
    ) -> Vec<CoreResult<ReadinessEvent>> {
        let stream = readiness_events_with_sleeper(
            http, "https://api.test", "ik_sdk_x", "proj", "wl", sleeper,
        )
        .await
        .expect("open stream");
        stream.collect::<Vec<_>>().await
    }

    #[tokio::test]
    async fn reconnects_after_a_mid_flight_drop_and_resolves_ready() {
        // The user's case: first stream emits one progress frame then ends with
        // NO terminal frame (manager restarted); reconnect sees `ready`.
        let http = Arc::new(FakeHttp::new(vec![
            Attempt::Stream(vec![Ok(ev("provisioning"))]), // drops without terminal
            Attempt::Stream(vec![Ok(ev("ready"))]),
        ]));
        let sleeper = RecordingSleeper::new();
        let out = collect_readiness(http.clone(), sleeper.clone()).await;

        let phases: Vec<String> = out.iter().map(|r| r.as_ref().unwrap().phase.clone()).collect();
        assert_eq!(phases, vec!["provisioning", "ready"]);
        assert_eq!(http.connect_count(), 2, "should reconnect exactly once");
        assert_eq!(sleeper.delays(), vec![1_000], "one backoff before the reconnect");
    }

    #[tokio::test]
    async fn happy_path_does_not_reconnect() {
        // Regression: `ready` on the first stream means zero reconnects.
        let http = Arc::new(FakeHttp::new(vec![Attempt::Stream(vec![
            Ok(ev("scheduling")),
            Ok(ev("ready")),
        ])]));
        let sleeper = RecordingSleeper::new();
        let out = collect_readiness(http.clone(), sleeper.clone()).await;

        let phases: Vec<String> = out.iter().map(|r| r.as_ref().unwrap().phase.clone()).collect();
        assert_eq!(phases, vec!["scheduling", "ready"]);
        assert_eq!(http.connect_count(), 1);
        assert!(sleeper.delays().is_empty(), "no backoff on the happy path");
    }

    #[tokio::test]
    async fn permanent_error_on_reconnect_does_not_retry() {
        // First stream drops; the reconnect hits a 404 — surface it and stop,
        // no further attempts.
        let http = Arc::new(FakeHttp::new(vec![
            Attempt::Stream(vec![Ok(ev("scheduling"))]),
            Attempt::ConnectErr(CoreError::NotFound("workload gone".into())),
            Attempt::Stream(vec![Ok(ev("ready"))]), // must NOT be reached
        ]));
        let sleeper = RecordingSleeper::new();
        let out = collect_readiness(http.clone(), sleeper.clone()).await;

        assert_eq!(out[0].as_ref().unwrap().phase, "scheduling");
        assert!(matches!(out[1], Err(CoreError::NotFound(_))));
        assert_eq!(out.len(), 2, "stream ends after the permanent error");
        assert_eq!(http.connect_count(), 2, "no attempt after the 404");
    }

    #[tokio::test]
    async fn transient_connect_errors_back_off_then_succeed() {
        // Two failed reconnects (network), then a stream that reaches ready.
        // Asserts the 1s, 2s backoff schedule.
        let http = Arc::new(FakeHttp::new(vec![
            Attempt::Stream(vec![Ok(ev("scheduling"))]), // drop #1
            Attempt::ConnectErr(CoreError::Network("refused".into())), // reconnect fails
            Attempt::ConnectErr(CoreError::Api { status: 503, message: "starting".into() }),
            Attempt::Stream(vec![Ok(ev("ready"))]),
        ]));
        let sleeper = RecordingSleeper::new();
        let out = collect_readiness(http.clone(), sleeper.clone()).await;

        let phases: Vec<String> = out.iter().map(|r| r.as_ref().unwrap().phase.clone()).collect();
        assert_eq!(phases, vec!["scheduling", "ready"]);
        // Backoff before each of the 3 reconnect attempts: 1s, 2s, 4s.
        assert_eq!(sleeper.delays(), vec![1_000, 2_000, 4_000]);
    }

    #[tokio::test]
    async fn cancellation_drops_the_stream_without_hanging() {
        // Dropping the consumer (the stream) must not deadlock or panic even
        // mid-reconnect. Take one event, then drop.
        let http = Arc::new(FakeHttp::new(vec![
            Attempt::Stream(vec![Ok(ev("scheduling"))]),
            Attempt::Stream(vec![Ok(ev("ready"))]),
        ]));
        let sleeper = RecordingSleeper::new();
        let mut stream = readiness_events_with_sleeper(
            http, "https://api.test", "ik_sdk_x", "proj", "wl", sleeper,
        )
        .await
        .expect("open stream");
        let first = stream.next().await.unwrap().unwrap();
        assert_eq!(first.phase, "scheduling");
        drop(stream); // no hang, no panic
    }

    // ── delete ────────────────────────────────────────────────────────────

    /// Fake transport for the unary path: answers the workload-list GET with a
    /// fixed roster and records every (method, url) so a test can assert which
    /// requests `delete` issued.
    struct DeleteHttp {
        list_body: Value,
        calls: Mutex<Vec<(String, String)>>,
    }

    impl DeleteHttp {
        fn new(list_body: Value) -> Arc<Self> {
            Arc::new(Self { list_body, calls: Mutex::new(Vec::new()) })
        }
        fn calls(&self) -> Vec<(String, String)> {
            self.calls.lock().unwrap().clone()
        }
    }

    impl HttpPort for DeleteHttp {
        fn request_json<'a>(&'a self, req: HttpRequest) -> BoxFuture<'a, Value> {
            self.calls
                .lock()
                .unwrap()
                .push((req.method.as_str().to_string(), req.url.clone()));
            let body = if matches!(req.method, HttpMethod::Get) {
                self.list_body.clone()
            } else {
                // DELETE → 204 No Content, decoded as Null by the adapter.
                Value::Null
            };
            Box::pin(async move { Ok(body) })
        }
        fn stream_sse<'a>(&'a self, _req: HttpRequest) -> BoxFuture<'a, JsonStream> {
            Box::pin(async { Err(CoreError::NotImplemented("stream_sse".into())) })
        }
    }

    #[tokio::test]
    async fn delete_resolves_slug_then_issues_delete_by_id() {
        let http = DeleteHttp::new(json!({ "workloads": [
            workload_json("keep", TaskType::Text2Text, Backend::Vllm),
            workload_json("doomed", TaskType::Text2Text, Backend::Vllm),
        ]}));
        let existed = delete(http.as_ref(), "https://api.test", "ik_sdk_x", "proj", "doomed")
            .await
            .expect("delete ok");
        assert!(existed, "slug present → returns true");
        let calls = http.calls();
        // One GET to resolve the slug, then a DELETE to the matched id.
        assert_eq!(calls[0].0, "GET");
        assert_eq!(calls[1].0, "DELETE");
        assert!(
            calls[1].1.ends_with("/api/workloads/wl-doomed"),
            "deletes the resolved id, got {}",
            calls[1].1
        );
    }

    #[tokio::test]
    async fn delete_is_idempotent_for_absent_slug() {
        let http = DeleteHttp::new(json!({ "workloads": [
            workload_json("keep", TaskType::Text2Text, Backend::Vllm),
        ]}));
        let existed = delete(http.as_ref(), "https://api.test", "ik_sdk_x", "proj", "ghost")
            .await
            .expect("delete ok");
        assert!(!existed, "absent slug → returns false, not an error");
        let calls = http.calls();
        assert_eq!(calls.len(), 1, "only the list GET, no DELETE");
        assert_eq!(calls[0].0, "GET");
    }
}