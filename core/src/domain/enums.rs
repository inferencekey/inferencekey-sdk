//! Domain enums. Each variant maps to the exact wire string the platform uses.
//!
//! Pure: parsing and rendering only, no IO. `serde` (de)serializes them as their
//! wire strings so they drop straight into request/response bodies.

use serde::{Deserialize, Serialize};

/// Inference backend. The serialized form is the `backend` wire string.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Backend {
    Ollama,
    Vllm,
    #[serde(rename = "vllm-omni")]
    VllmOmni,
    Sglang,
}

impl Backend {
    /// The wire string (e.g. `"vllm-omni"`).
    pub fn as_str(&self) -> &'static str {
        match self {
            Backend::Ollama => "ollama",
            Backend::Vllm => "vllm",
            Backend::VllmOmni => "vllm-omni",
            Backend::Sglang => "sglang",
        }
    }

    /// Parse a wire string into a [`Backend`].
    pub fn from_str_opt(s: &str) -> Option<Self> {
        match s {
            "ollama" => Some(Backend::Ollama),
            "vllm" => Some(Backend::Vllm),
            "vllm-omni" => Some(Backend::VllmOmni),
            "sglang" => Some(Backend::Sglang),
            _ => None,
        }
    }
}

/// Workload modality (`task_type`). Server default is `text2text`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskType {
    Text2Text,
    Embedding,
    Text2Image,
    Text2Audio,
    Audio2Text,
    Reranker,
    Classification,
    Reward,
}

impl TaskType {
    /// The wire string (e.g. `"text2text"`).
    pub fn as_str(&self) -> &'static str {
        match self {
            TaskType::Text2Text => "text2text",
            TaskType::Embedding => "embedding",
            TaskType::Text2Image => "text2image",
            TaskType::Text2Audio => "text2audio",
            TaskType::Audio2Text => "audio2text",
            TaskType::Reranker => "reranker",
            TaskType::Classification => "classification",
            TaskType::Reward => "reward",
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
        for b in [Backend::Ollama, Backend::Vllm, Backend::VllmOmni, Backend::Sglang] {
            assert_eq!(Backend::from_str_opt(b.as_str()), Some(b));
        }
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
}
