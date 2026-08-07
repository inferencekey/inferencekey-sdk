//! Domain enums. Each variant maps to the exact wire string the platform uses.
//!
//! Pure: parsing and rendering only, no IO. `serde` (de)serializes them as their
//! wire strings so they drop straight into request/response bodies.

use std::borrow::Cow;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Inference backend. The serialized form is the `backend` wire string.
///
/// The native variants carry no data and serialize to their fixed wire strings
/// (`ollama`, `vllm`, `vllm-omni`, `sglang`, `llamacpp`). Any other wire string
/// deserializes to [`Backend::Custom`] carrying the slug verbatim, so a
/// `WorkloadResponse` naming a custom (SDK-published) backend never fails to
/// parse. `Custom` re-serializes to that same slug.
///
/// Carrying a `String` means `Backend` is **not** `Copy`; clone it where a value
/// is needed by ownership.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Backend {
    Ollama,
    Vllm,
    VllmOmni,
    Sglang,
    /// llama.cpp: prebuilt `llama-server` (GGUF, OpenAI-compatible). The worker
    /// resolves the install path per node hardware (ROCm/Metal tarball, or apt
    /// CUDA on NVIDIA). `text2text` only today.
    Llamacpp,
    /// stable-diffusion.cpp: resident `sd-server` (exposes the OpenAI images
    /// surface `/v1/images/generations`). The worker compiles sd.cpp with CUDA
    /// for the local GPU at bootstrap (no prebuilt Linux+CUDA release). Serves
    /// `text2image` only.
    StableDiffusionCpp,
    /// A custom backend published through the SDK, identified by its slug.
    Custom(String),
}

impl Backend {
    /// The wire string (e.g. `"vllm-omni"`, or the slug for [`Backend::Custom`]).
    ///
    /// Native variants borrow a `'static` string; `Custom` borrows its slug. The
    /// `Cow` lets both share one return type without allocating for natives.
    pub fn as_str(&self) -> Cow<'_, str> {
        match self {
            Backend::Ollama => Cow::Borrowed("ollama"),
            Backend::Vllm => Cow::Borrowed("vllm"),
            Backend::VllmOmni => Cow::Borrowed("vllm-omni"),
            Backend::Sglang => Cow::Borrowed("sglang"),
            Backend::Llamacpp => Cow::Borrowed("llamacpp"),
            Backend::StableDiffusionCpp => Cow::Borrowed("stablediffusioncpp"),
            Backend::Custom(slug) => Cow::Borrowed(slug.as_str()),
        }
    }

    /// Parse a wire string into a native [`Backend`]. Returns `None` for any
    /// string that is not a native backend (use [`Backend::from_wire`] to accept
    /// custom slugs).
    pub fn from_str_opt(s: &str) -> Option<Self> {
        match s {
            "ollama" => Some(Backend::Ollama),
            "vllm" => Some(Backend::Vllm),
            "vllm-omni" => Some(Backend::VllmOmni),
            "sglang" => Some(Backend::Sglang),
            "llamacpp" => Some(Backend::Llamacpp),
            "stablediffusioncpp" => Some(Backend::StableDiffusionCpp),
            _ => None,
        }
    }

    /// Parse any wire string. Native backends map to their variant; everything
    /// else becomes [`Backend::Custom`]. Never fails.
    pub fn from_wire(s: &str) -> Self {
        Backend::from_str_opt(s).unwrap_or_else(|| Backend::Custom(s.to_owned()))
    }
}

impl Serialize for Backend {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.as_str())
    }
}

impl<'de> Deserialize<'de> for Backend {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        Ok(Backend::from_wire(&s))
    }
}

/// Workload modality (`task_type`). Server default is `text2text`.
///
/// The wire strings are `text2text`, `text2image`, … (no underscore before the
/// digit), so the serde renames are spelled out per-variant rather than derived
/// from a `rename_all` rule — `snake_case` would wrongly emit `text2_text`, and
/// the resulting body would be rejected by the server and fail to deserialize a
/// `WorkloadResponse` read back. Keep these in lockstep with [`TaskType::as_str`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskType {
    #[serde(rename = "text2text")]
    Text2Text,
    #[serde(rename = "embedding")]
    Embedding,
    #[serde(rename = "text2image")]
    Text2Image,
    #[serde(rename = "text2video")]
    Text2Video,
    #[serde(rename = "text2audio")]
    Text2Audio,
    #[serde(rename = "image2image")]
    Image2Image,
    #[serde(rename = "image2video")]
    Image2Video,
    #[serde(rename = "audio2text")]
    Audio2Text,
    #[serde(rename = "omni")]
    Omni,
    #[serde(rename = "reranker")]
    Reranker,
    #[serde(rename = "classification")]
    Classification,
    #[serde(rename = "reward")]
    Reward,
    #[serde(rename = "forecast")]
    Forecast,
}

impl TaskType {
    /// The wire string (e.g. `"text2text"`).
    pub fn as_str(&self) -> &'static str {
        match self {
            TaskType::Text2Text => "text2text",
            TaskType::Embedding => "embedding",
            TaskType::Text2Image => "text2image",
            TaskType::Text2Video => "text2video",
            TaskType::Text2Audio => "text2audio",
            TaskType::Image2Image => "image2image",
            TaskType::Image2Video => "image2video",
            TaskType::Audio2Text => "audio2text",
            TaskType::Omni => "omni",
            TaskType::Reranker => "reranker",
            TaskType::Classification => "classification",
            TaskType::Reward => "reward",
            TaskType::Forecast => "forecast",
        }
    }
}

/// Drift-handling strategy for `ensure()`. Defaults to [`OnDrift::Reconcile`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OnDrift {
    /// Create if absent; update in place if drifted (default).
    #[default]
    Reconcile,
    /// Raise if the existing workload differs.
    Fail,
    /// Report the plan, change nothing.
    DryRun,
    /// Log a warning, leave as-is.
    Warn,
    /// Silently use the existing workload.
    Ignore,
}

/// Execution policy (`execution_policy`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionPolicy {
    Fixed,
    Scheduled,
    Autoscaling,
}

impl ExecutionPolicy {
    /// The wire string.
    pub fn as_str(&self) -> &'static str {
        match self {
            ExecutionPolicy::Fixed => "fixed",
            ExecutionPolicy::Scheduled => "scheduled",
            ExecutionPolicy::Autoscaling => "autoscaling",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_round_trips_through_wire_string() {
        for b in [
            Backend::Ollama,
            Backend::Vllm,
            Backend::VllmOmni,
            Backend::Sglang,
            Backend::Llamacpp,
            Backend::StableDiffusionCpp,
        ] {
            assert_eq!(Backend::from_str_opt(b.as_str().as_ref()), Some(b.clone()));
        }
        assert_eq!(Backend::from_str_opt("llamacpp"), Some(Backend::Llamacpp));
        assert_eq!(
            Backend::from_str_opt("stablediffusioncpp"),
            Some(Backend::StableDiffusionCpp)
        );
        assert_eq!(Backend::from_str_opt("nope"), None);
    }

    #[test]
    fn on_drift_defaults_to_reconcile() {
        assert_eq!(OnDrift::default(), OnDrift::Reconcile);
    }

    #[test]
    fn backend_serializes_as_kebab_wire_string() {
        let json = serde_json::to_string(&Backend::VllmOmni).expect("serialize");
        assert_eq!(json, "\"vllm-omni\"");
    }

    #[test]
    fn native_backends_serde_round_trip_to_their_wire_strings() {
        let cases = [
            (Backend::Ollama, "\"ollama\""),
            (Backend::Vllm, "\"vllm\""),
            (Backend::VllmOmni, "\"vllm-omni\""),
            (Backend::Sglang, "\"sglang\""),
            (Backend::Llamacpp, "\"llamacpp\""),
            (Backend::StableDiffusionCpp, "\"stablediffusioncpp\""),
        ];
        for (variant, wire) in cases {
            let json = serde_json::to_string(&variant).expect("serialize");
            assert_eq!(json, wire, "serialize {variant:?}");
            let parsed: Backend = serde_json::from_str(wire).expect("deserialize");
            assert_eq!(parsed, variant, "deserialize {wire}");
            assert_eq!(Backend::from_wire(variant.as_str().as_ref()), variant);
        }
    }

    #[test]
    fn custom_backend_serializes_to_its_slug() {
        let json = serde_json::to_string(&Backend::Custom("echo".to_owned())).expect("serialize");
        assert_eq!(json, "\"echo\"");
        assert_eq!(Backend::Custom("echo".to_owned()).as_str(), "echo");
    }

    #[test]
    fn unknown_wire_string_deserializes_to_custom() {
        let parsed: Backend = serde_json::from_str("\"echo\"").expect("deserialize");
        assert_eq!(parsed, Backend::Custom("echo".to_owned()));
        assert_eq!(
            Backend::from_wire("echo"),
            Backend::Custom("echo".to_owned())
        );
        // Native strings still map to native variants, never to Custom.
        assert_eq!(Backend::from_wire("vllm"), Backend::Vllm);
    }

    #[test]
    fn custom_backend_round_trips_through_serde() {
        let original = Backend::Custom("my-cool-backend".to_owned());
        let json = serde_json::to_string(&original).expect("serialize");
        let parsed: Backend = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed, original);
    }

    #[test]
    fn task_type_serde_matches_as_str_wire_string() {
        // serde (de)serialization must round-trip through the exact wire string
        // `as_str()` reports — otherwise a `WorkloadResponse` from the server
        // (e.g. `"text2text"`) fails to deserialize. Guards the snake_case trap.
        let variants = [
            TaskType::Text2Text,
            TaskType::Embedding,
            TaskType::Text2Image,
            TaskType::Text2Video,
            TaskType::Text2Audio,
            TaskType::Image2Image,
            TaskType::Image2Video,
            TaskType::Audio2Text,
            TaskType::Omni,
            TaskType::Reranker,
            TaskType::Classification,
            TaskType::Reward,
            TaskType::Forecast,
        ];
        for variant in variants {
            let json = serde_json::to_string(&variant).expect("serialize");
            assert_eq!(json, format!("\"{}\"", variant.as_str()));
            let parsed: TaskType = serde_json::from_str(&json).expect("deserialize");
            assert_eq!(parsed, variant);
        }
    }
}
