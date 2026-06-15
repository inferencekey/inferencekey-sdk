//! Stable C ABI over `inferencekey-core`.
//!
//! This is the FFI surface other languages (cgo, JNI/FFM, …) link against. The
//! core is async; this layer wraps it in a blocking facade backed by an internal
//! Tokio runtime so the ABI stays simple and synchronous.
//!
//! Conventions (kept uniform across every entry point):
//! * Opaque handles are returned as `*mut` to a boxed Rust value; the caller
//!   owns them and must release them with the matching `*_free`.
//! * Strings cross the boundary as NUL-terminated UTF-8 `char*`. Inputs are
//!   borrowed; outputs are heap-allocated by this library and the caller frees
//!   them with [`ik_string_free`].
//! * Fallible calls return an `int32` status: `0` = ok, non-zero = error. The
//!   human-readable message for the most recent failure on the calling thread
//!   is retrievable via [`ik_last_error`].
//! * Complex inputs/outputs are JSON (a spec in, a result out) so the ABI is
//!   tiny and stable as the domain evolves.
//!
//! `unsafe` and raw-pointer/`as` use is confined to this crate — it is the
//! documented FFI boundary the workspace guidelines carve out as the one place
//! they are acceptable.

use std::ffi::{c_char, c_int, CString};

use inferencekey_core::{
    embed, ensure, generate_text, EmbedParams, GenerateTextParams, OnDrift, ReqwestHttp,
    WorkloadSpec,
};

mod ffi;
mod runtime;

use ffi::{borrow_str, into_c_string, set_last_error, OkOrErr};
use runtime::Client;

/// Status codes returned by fallible entry points. `0` is success.
pub const IK_OK: c_int = 0;
pub const IK_ERR_NULL_ARG: c_int = 1;
pub const IK_ERR_INVALID_UTF8: c_int = 2;
pub const IK_ERR_INVALID_JSON: c_int = 3;
pub const IK_ERR_CALL_FAILED: c_int = 4;

// ---------------------------------------------------------------------------
// Client lifecycle
// ---------------------------------------------------------------------------

/// Create a client bound to `base_url`. Returns an opaque handle, or null on
/// failure (call [`ik_last_error`] for details). Free it with [`ik_client_free`].
///
/// # Safety
/// `base_url` must be a valid NUL-terminated UTF-8 string for the duration of
/// the call.
#[no_mangle]
pub unsafe extern "C" fn ik_client_new(base_url: *const c_char) -> *mut Client {
    match build_client(base_url) {
        Ok(client) => Box::into_raw(Box::new(client)),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Release a client created by [`ik_client_new`]. Passing null is a no-op.
///
/// # Safety
/// `client` must be a handle from [`ik_client_new`] not already freed.
#[no_mangle]
pub unsafe extern "C" fn ik_client_free(client: *mut Client) {
    if !client.is_null() {
        drop(Box::from_raw(client));
    }
}

/// Build a [`Client`] from a borrowed base URL.
unsafe fn build_client(base_url: *const c_char) -> Result<Client, ()> {
    let base_url = borrow_str(base_url, "base_url")?;
    Client::new(base_url, ReqwestHttp::new()).map_err(|e| {
        set_last_error(&e.to_string());
    })
}

// ---------------------------------------------------------------------------
// Control plane: ensure()
// ---------------------------------------------------------------------------

/// Provision/reconcile a workload. `spec_json` is a `WorkloadSpec` as JSON.
/// On success writes a heap-allocated `EndpointRef` JSON string to `*out_json`
/// (free with [`ik_string_free`]) and returns [`IK_OK`].
///
/// # Safety
/// All pointers must be valid NUL-terminated UTF-8 (except `out_json`, a valid
/// `*mut *mut c_char`). The handle must come from [`ik_client_new`].
#[no_mangle]
pub unsafe extern "C" fn ik_ensure(
    client: *mut Client,
    sdk_token: *const c_char,
    project_id: *const c_char,
    spec_json: *const c_char,
    on_drift: *const c_char,
    out_json: *mut *mut c_char,
) -> c_int {
    run_ensure(client, sdk_token, project_id, spec_json, on_drift, out_json).ok_or_err()
}

/// Inner ensure() that works in `Result` so the body composes with `?`.
unsafe fn run_ensure(
    client: *mut Client,
    sdk_token: *const c_char,
    project_id: *const c_char,
    spec_json: *const c_char,
    on_drift: *const c_char,
    out_json: *mut *mut c_char,
) -> Result<(), c_int> {
    let client = as_ref(client)?;
    let sdk_token = borrow_str(sdk_token, "sdk_token").map_err(|_| IK_ERR_INVALID_UTF8)?;
    let project_id = borrow_str(project_id, "project_id").map_err(|_| IK_ERR_INVALID_UTF8)?;
    let spec = parse_spec(spec_json)?;
    let policy = parse_on_drift(on_drift)?;
    let http = client.http();
    let result = client
        .block_on(ensure(http, client.base_url(), sdk_token, project_id, &spec, policy))
        .map_err(map_call_error)?;
    write_json(&result, out_json)
}

// ---------------------------------------------------------------------------
// Data plane: generate_text / embed
// ---------------------------------------------------------------------------

/// Run a non-streaming chat completion. `params_json` is a `GenerateTextParams`
/// as JSON; on success writes a `TextResult` JSON string to `*out_json`.
///
/// # Safety
/// See [`ik_ensure`]; the same pointer rules apply.
#[no_mangle]
pub unsafe extern "C" fn ik_generate_text(
    client: *mut Client,
    project_slug: *const c_char,
    workload_slug: *const c_char,
    api_key: *const c_char,
    params_json: *const c_char,
    out_json: *mut *mut c_char,
) -> c_int {
    run_generate_text(client, project_slug, workload_slug, api_key, params_json, out_json)
        .ok_or_err()
}

unsafe fn run_generate_text(
    client: *mut Client,
    project_slug: *const c_char,
    workload_slug: *const c_char,
    api_key: *const c_char,
    params_json: *const c_char,
    out_json: *mut *mut c_char,
) -> Result<(), c_int> {
    let client = as_ref(client)?;
    let project_slug = borrow_str(project_slug, "project_slug").map_err(|_| IK_ERR_INVALID_UTF8)?;
    let workload_slug =
        borrow_str(workload_slug, "workload_slug").map_err(|_| IK_ERR_INVALID_UTF8)?;
    let api_key = borrow_str(api_key, "api_key").map_err(|_| IK_ERR_INVALID_UTF8)?;
    let params: GenerateTextParams = parse_json(params_json)?;
    let result = client
        .block_on(generate_text(
            client.http(),
            client.base_url(),
            project_slug,
            workload_slug,
            api_key,
            params,
        ))
        .map_err(map_call_error)?;
    write_json(&result, out_json)
}

/// Run an embeddings request. `params_json` is an `EmbedParams` as JSON; on
/// success writes an `EmbedResult` JSON string to `*out_json`.
///
/// # Safety
/// See [`ik_ensure`]; the same pointer rules apply.
#[no_mangle]
pub unsafe extern "C" fn ik_embed(
    client: *mut Client,
    project_slug: *const c_char,
    workload_slug: *const c_char,
    api_key: *const c_char,
    params_json: *const c_char,
    out_json: *mut *mut c_char,
) -> c_int {
    run_embed(client, project_slug, workload_slug, api_key, params_json, out_json).ok_or_err()
}

unsafe fn run_embed(
    client: *mut Client,
    project_slug: *const c_char,
    workload_slug: *const c_char,
    api_key: *const c_char,
    params_json: *const c_char,
    out_json: *mut *mut c_char,
) -> Result<(), c_int> {
    let client = as_ref(client)?;
    let project_slug = borrow_str(project_slug, "project_slug").map_err(|_| IK_ERR_INVALID_UTF8)?;
    let workload_slug =
        borrow_str(workload_slug, "workload_slug").map_err(|_| IK_ERR_INVALID_UTF8)?;
    let api_key = borrow_str(api_key, "api_key").map_err(|_| IK_ERR_INVALID_UTF8)?;
    let params: EmbedParams = parse_json(params_json)?;
    let result = client
        .block_on(embed(
            client.http(),
            client.base_url(),
            project_slug,
            workload_slug,
            api_key,
            params,
        ))
        .map_err(map_call_error)?;
    write_json(&result, out_json)
}

// ---------------------------------------------------------------------------
// Strings & errors
// ---------------------------------------------------------------------------

/// Free a string previously returned by this library (e.g. an `out_json`).
/// Passing null is a no-op.
///
/// # Safety
/// `s` must be a pointer returned by this library and not already freed.
#[no_mangle]
pub unsafe extern "C" fn ik_string_free(s: *mut c_char) {
    if !s.is_null() {
        drop(CString::from_raw(s));
    }
}

/// Return the most recent error message on the calling thread as a freshly
/// allocated string, or null if there is none. Free it with [`ik_string_free`].
#[no_mangle]
pub extern "C" fn ik_last_error() -> *mut c_char {
    ffi::take_last_error()
        .and_then(|msg| into_c_string(&msg).ok())
        .unwrap_or(std::ptr::null_mut())
}

// ---------------------------------------------------------------------------
// Small shared helpers (pure / borrow-checked)
// ---------------------------------------------------------------------------

/// Borrow an opaque client handle, recording an error and mapping to a status
/// code when it is null.
unsafe fn as_ref<'a>(client: *mut Client) -> Result<&'a Client, c_int> {
    match client.is_null() {
        true => Err(set_last_error("client handle is null")),
        false => Ok(&*client),
    }
}

/// Parse a JSON spec string into a [`WorkloadSpec`].
unsafe fn parse_spec(spec_json: *const c_char) -> Result<WorkloadSpec, c_int> {
    parse_json(spec_json)
}

/// Parse any `Deserialize` value from a borrowed JSON C string.
unsafe fn parse_json<T: serde::de::DeserializeOwned>(json: *const c_char) -> Result<T, c_int> {
    let raw = borrow_str(json, "json").map_err(|_| IK_ERR_INVALID_UTF8)?;
    serde_json::from_str(raw).map_err(|e| set_last_error(&format!("invalid json: {e}")))
}

/// Parse the optional `on_drift` selector; `None`/empty defaults to reconcile.
unsafe fn parse_on_drift(on_drift: *const c_char) -> Result<OnDrift, c_int> {
    if on_drift.is_null() {
        return Ok(OnDrift::Reconcile);
    }
    let raw = borrow_str(on_drift, "on_drift").map_err(|_| IK_ERR_INVALID_UTF8)?;
    on_drift_from_str(raw)
}

/// Map an `on_drift` wire string to the enum.
fn on_drift_from_str(raw: &str) -> Result<OnDrift, c_int> {
    match raw {
        "" | "reconcile" => Ok(OnDrift::Reconcile),
        "fail" => Ok(OnDrift::Fail),
        "dry_run" => Ok(OnDrift::DryRun),
        "warn" => Ok(OnDrift::Warn),
        "ignore" => Ok(OnDrift::Ignore),
        other => Err(set_last_error(&format!("unknown on_drift: {other}"))),
    }
}

/// Serialize a result and write it into `*out_json` as a fresh C string.
unsafe fn write_json<T: serde::Serialize>(
    value: &T,
    out_json: *mut *mut c_char,
) -> Result<(), c_int> {
    if out_json.is_null() {
        return Err(set_last_error("out_json pointer is null"));
    }
    let json = serde_json::to_string(value)
        .map_err(|e| set_last_error(&format!("serialize failed: {e}")))?;
    let c = into_c_string(&json).map_err(|_| set_last_error("result contained a NUL byte"))?;
    *out_json = c;
    Ok(())
}

/// Record a core call failure and return the generic call-failed status code.
fn map_call_error(err: inferencekey_core::CoreError) -> c_int {
    set_last_error(&err.to_string());
    IK_ERR_CALL_FAILED
}
