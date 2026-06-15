"""Typed enums whose values are the exact platform wire strings."""

from __future__ import annotations

from enum import Enum


class _Str(str, Enum):
    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.value


class Backend(_Str):
    """Inference backend (the ``backend`` wire string)."""

    OLLAMA = "ollama"
    VLLM = "vllm"
    VLLM_OMNI = "vllm-omni"
    SGLANG = "sglang"


class TaskType(_Str):
    """Workload modality (``task_type``); server default is ``text2text``."""

    TEXT2TEXT = "text2text"
    EMBEDDING = "embedding"
    TEXT2IMAGE = "text2image"
    TEXT2AUDIO = "text2audio"
    AUDIO2TEXT = "audio2text"
    RERANKER = "reranker"
    CLASSIFICATION = "classification"
    REWARD = "reward"


class OnDrift(_Str):
    """Drift-handling strategy for ``ensure()``; defaults to ``RECONCILE``."""

    RECONCILE = "reconcile"
    FAIL = "fail"
    DRY_RUN = "dry_run"
    WARN = "warn"
    IGNORE = "ignore"


class ExecutionPolicy(_Str):
    """How the Manager schedules the workload (``execution_policy``)."""

    FIXED = "fixed"
    SCHEDULED = "scheduled"
    AUTOSCALING = "autoscaling"
