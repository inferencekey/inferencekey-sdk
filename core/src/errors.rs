//! Domain error type, explicit and mappable.
//!
//! Every layer maps its failure to [`CoreError`] as early as possible. The
//! `code` carried on permission failures is the machine-readable string the
//! control plane returns (`wrong_credential_type`, …) so callers can branch on
//! it without parsing messages.

use thiserror::Error;

/// Machine-readable permission codes the control plane returns on 403.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PermissionCode {
    WrongCredentialType,
    ProjectScopeMismatch,
    ScopeInsufficient,
    /// A 403 whose body code we did not recognize.
    Unknown,
}

impl PermissionCode {
    /// The wire string for this code.
    pub fn as_str(&self) -> &'static str {
        match self {
            PermissionCode::WrongCredentialType => "wrong_credential_type",
            PermissionCode::ProjectScopeMismatch => "project_scope_mismatch",
            PermissionCode::ScopeInsufficient => "scope_insufficient",
            PermissionCode::Unknown => "forbidden",
        }
    }

    /// Parse a control-plane error code into a [`PermissionCode`].
    pub fn from_code(code: Option<&str>) -> Self {
        match code {
            Some("wrong_credential_type") => PermissionCode::WrongCredentialType,
            Some("project_scope_mismatch") => PermissionCode::ProjectScopeMismatch,
            Some("scope_insufficient") => PermissionCode::ScopeInsufficient,
            _ => PermissionCode::Unknown,
        }
    }
}

/// All failures the core can surface. Bindings map these to each language's
/// idiomatic exception type.
#[derive(Debug, Error)]
pub enum CoreError {
    /// Client-side misconfiguration before any request (missing url/project/token).
    #[error("configuration error: {0}")]
    Config(String),

    /// A request argument failed local validation.
    #[error("validation error: {0}")]
    Validation(String),

    /// 401 — missing or invalid credentials.
    #[error("authentication failed: {0}")]
    Auth(String),

    /// 403 — valid request, but the credential may not perform it.
    #[error("permission denied [{code}]: {message}", code = .code.as_str())]
    Permission {
        code: PermissionCode,
        message: String,
    },

    /// 404 — the requested resource does not exist.
    #[error("not found: {0}")]
    NotFound(String),

    /// 400 — the request was malformed or rejected by the server.
    #[error("bad request: {0}")]
    BadRequest(String),

    /// `ensure()` found drift while the policy was `Fail`. `fields` lists the
    /// drifted field names.
    #[error("workload drifted from spec: {fields}")]
    Drift { fields: String },

    /// A surface declared but not implemented yet.
    #[error("not implemented: {0}")]
    NotImplemented(String),

    /// Any other non-2xx response.
    #[error("api error (status {status}): {message}")]
    Api { status: u16, message: String },

    /// A transport-level failure (DNS, connection, timeout) before a response.
    #[error("network error: {0}")]
    Network(String),

    /// Failed to (de)serialize a JSON body.
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
}

/// Convenience alias used throughout the core.
pub type CoreResult<T> = Result<T, CoreError>;
