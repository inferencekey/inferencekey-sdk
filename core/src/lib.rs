//! # InferenceKey Core
//!
//! Shared core logic and transport for the InferenceKey SDK. This crate is the
//! single source of truth consumed by the C ABI and every per-language binding
//! (Python, Node, Go, ...); the bindings are thin shells over the contracts and
//! pipelines declared here.
//!
//! ## Two planes, two tokens
//!
//! InferenceKey exposes two distinct surfaces, each with its own credential:
//!
//! - **Control plane** — management of workloads (create / update / list / get)
//!   over JSON under `/api/...`. Authenticated with a management token
//!   (`ik_sdk_...`) sent as `Authorization: Bearer ik_sdk_<...>`. Data keys are
//!   rejected here. See [`pipelines::management`].
//! - **Data plane** — OpenAI-compatible inference (chat completions, embeddings)
//!   under `/endpoint/:project_slug/:workload_slug/v1/...`. Authenticated with a
//!   data key (`ik_live_...`) sent as `Authorization: Bearer ik_live_<...>`.
//!   Only data keys are accepted here. See [`pipelines::data`].
//!
//! ## Design boundaries
//!
//! Pure logic (parsing, validation, request building, drift diffing) is kept
//! separate from effects (HTTP, time), which sit behind the [`ports::http`]
//! abstraction and are realized by [`adapters::reqwest_http`]. Secrets are
//! never embedded in specs and are redacted to a prefix in logs via
//! [`domain::redact`]. Idempotency is keyed by an explicit slug, and drift is
//! resolved according to the caller's [`OnDrift`] policy.

pub mod errors;

pub mod domain {
    pub mod enums;
    pub mod redact;
    pub mod config;
    pub mod spec;
    pub mod wire;
    pub mod sse;
}

pub mod ports {
    pub mod http;
}

pub mod adapters {
    pub mod reqwest_http;
}

pub mod pipelines {
    pub mod management;
    pub mod data;
}

pub use errors::{CoreError, CoreResult, PermissionCode};

pub use domain::enums::*;
pub use domain::spec::WorkloadSpec;

pub use pipelines::management::{delete, ensure, readiness_events, EndpointRef, ReadinessEvent};
pub use pipelines::data::{
    embed, generate_text, generate_text_stream, ChatMessage, EmbedParams, EmbedResult,
    GenerateTextParams, TextChunk, TextResult,
};

pub use adapters::reqwest_http::ReqwestHttp;
