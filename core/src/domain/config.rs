//! Pure config resolution from explicit values, environment, and defaults.
//!
//! Precedence is always **explicit > env var > default**. The only IO here is
//! reading `std::env`; everything else is string shaping. Resolution never
//! fails — a missing optional value surfaces as `None` so the caller layer can
//! decide whether its absence is a [`crate::errors::CoreError::Config`].

/// Fallback base URL when neither an explicit value nor an env var is set.
pub const DEFAULT_BASE_URL: &str = "https://api.inferencekey.com";

const ENV_BASE_URL: &str = "INFERENCEKEY_BASE_URL";
const ENV_PROJECT: &str = "INFERENCEKEY_PROJECT";
const ENV_API_KEY: &str = "INFERENCEKEY_API_KEY";
const ENV_SDK_TOKEN: &str = "INFERENCEKEY_SDK_TOKEN";

/// Resolved configuration for the data plane (OpenAI-compatible, `ik_live_`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DataConfig {
    /// Base URL with any trailing slash trimmed.
    pub base_url: String,
    /// Project slug used to build the `/endpoint/:project/...` path.
    pub project: Option<String>,
    /// The `ik_live_` data-plane key, when known.
    pub api_key: Option<String>,
}

/// Resolved configuration for the control plane (`/api/...`, `ik_sdk_`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ManagementConfig {
    /// Base URL with any trailing slash trimmed.
    pub base_url: String,
    /// Default project slug for project-scoped management calls.
    pub project: Option<String>,
    /// The `ik_sdk_` management token, when known.
    pub sdk_token: Option<String>,
}

/// Resolve the base URL: explicit, else `INFERENCEKEY_BASE_URL`, else default.
///
/// The result always has its trailing slash trimmed so path joins stay clean.
pub fn resolve_base_url(explicit: Option<&str>) -> String {
    let resolved = explicit
        .map(str::to_owned)
        .or_else(|| read_env(ENV_BASE_URL))
        .unwrap_or_else(|| DEFAULT_BASE_URL.to_owned());
    trim_trailing_slashes(&resolved).to_owned()
}

/// Resolve the project slug: explicit, else `INFERENCEKEY_PROJECT`.
pub fn resolve_project(explicit: Option<&str>) -> Option<String> {
    explicit.map(str::to_owned).or_else(|| read_env(ENV_PROJECT))
}

/// Resolve the data-plane key: explicit, else `INFERENCEKEY_API_KEY`.
pub fn resolve_data_api_key(explicit: Option<&str>) -> Option<String> {
    explicit.map(str::to_owned).or_else(|| read_env(ENV_API_KEY))
}

/// Resolve the management token: explicit, else `INFERENCEKEY_SDK_TOKEN`.
pub fn resolve_sdk_token(explicit: Option<&str>) -> Option<String> {
    explicit.map(str::to_owned).or_else(|| read_env(ENV_SDK_TOKEN))
}

/// Build a [`DataConfig`] applying the standard precedence to each field.
pub fn resolve_data_config(
    base_url: Option<&str>,
    project: Option<&str>,
    api_key: Option<&str>,
) -> DataConfig {
    DataConfig {
        base_url: resolve_base_url(base_url),
        project: resolve_project(project),
        api_key: resolve_data_api_key(api_key),
    }
}

/// Build a [`ManagementConfig`] applying the standard precedence to each field.
pub fn resolve_management_config(
    base_url: Option<&str>,
    project: Option<&str>,
    sdk_token: Option<&str>,
) -> ManagementConfig {
    ManagementConfig {
        base_url: resolve_base_url(base_url),
        project: resolve_project(project),
        sdk_token: resolve_sdk_token(sdk_token),
    }
}

/// Read an env var, treating empty/whitespace-only values as absent.
fn read_env(key: &str) -> Option<String> {
    std::env::var(key)
        .ok()
        .map(|v| v.trim().to_owned())
        .filter(|v| !v.is_empty())
}

/// Trim every trailing `/` so `https://h/` and `https://h` resolve equal.
fn trim_trailing_slashes(url: &str) -> &str {
    url.trim_end_matches('/')
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    // Env is process-global; serialize the env-touching tests behind one lock.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    const ALL_KEYS: [&str; 4] = [ENV_BASE_URL, ENV_PROJECT, ENV_API_KEY, ENV_SDK_TOKEN];

    fn clear_env() {
        for key in ALL_KEYS {
            std::env::remove_var(key);
        }
    }

    #[test]
    fn base_url_precedence_and_default() {
        let cases = [
            (Some("https://x.dev/"), None, "https://x.dev"),
            (None, Some("https://env.dev//"), "https://env.dev"),
            (None, None, DEFAULT_BASE_URL),
        ];
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        for (explicit, env, want) in cases {
            clear_env();
            if let Some(value) = env {
                std::env::set_var(ENV_BASE_URL, value);
            }
            assert_eq!(resolve_base_url(explicit), want);
        }
        clear_env();
    }

    #[test]
    fn optionals_prefer_explicit_then_env_then_none() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_env();
        assert_eq!(resolve_project(Some("p")), Some("p".to_owned()));

        std::env::set_var(ENV_PROJECT, "envp");
        std::env::set_var(ENV_API_KEY, "ik_live_x");
        std::env::set_var(ENV_SDK_TOKEN, "ik_sdk_x");
        assert_eq!(resolve_project(None), Some("envp".to_owned()));
        assert_eq!(resolve_data_api_key(None), Some("ik_live_x".to_owned()));
        assert_eq!(resolve_sdk_token(None), Some("ik_sdk_x".to_owned()));

        clear_env();
        assert_eq!(resolve_project(None), None);
        assert_eq!(resolve_data_api_key(None), None);
        assert_eq!(resolve_sdk_token(None), None);
    }

    #[test]
    fn blank_env_values_are_treated_as_absent() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_env();
        std::env::set_var(ENV_PROJECT, "   ");
        assert_eq!(resolve_project(None), None);
        clear_env();
    }

    #[test]
    fn config_builders_compose_resolution() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        clear_env();
        let data = resolve_data_config(Some("https://h/"), Some("proj"), Some("ik_live_k"));
        assert_eq!(
            data,
            DataConfig {
                base_url: "https://h".to_owned(),
                project: Some("proj".to_owned()),
                api_key: Some("ik_live_k".to_owned()),
            }
        );

        let mgmt = resolve_management_config(None, None, Some("ik_sdk_t"));
        assert_eq!(
            mgmt,
            ManagementConfig {
                base_url: DEFAULT_BASE_URL.to_owned(),
                project: None,
                sdk_token: Some("ik_sdk_t".to_owned()),
            }
        );
        clear_env();
    }
}
