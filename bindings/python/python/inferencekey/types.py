"""Public dataclasses: the spec a caller declares and the results returned."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from .enums import Backend, ExecutionPolicy, TaskType


@dataclass
class WorkloadSpec:
    """A declarative workload definition (the intent handed to ``ensure()``).

    ``name``, ``slug`` and ``model`` are required. ``provider`` and ``min_vram_gb``
    are intentionally absent — placement is the platform's job, never the caller's.
    """

    name: str
    slug: str
    model: str
    backend: Union[Backend, str]
    project: Optional[str] = None
    description: Optional[str] = None
    command: Optional[str] = None
    vllm_version: Optional[str] = None
    task_type: Optional[Union[TaskType, str]] = None
    config: Optional[Dict[str, Any]] = None
    execution_policy: Optional[Union[ExecutionPolicy, str]] = None
    execution_policy_config: Optional[Dict[str, Any]] = None
    worker_id: Optional[str] = None
    gpu_resource_id: Optional[str] = None

    def to_wire(self) -> Dict[str, Any]:
        """Render to the snake_case dict the core expects (omitting unset fields)."""
        out: Dict[str, Any] = {
            "name": self.name,
            "slug": self.slug,
            "model": self.model,
            "backend": _wire(self.backend),
        }
        optional = {
            "project": self.project,
            "description": self.description,
            "command": self.command,
            "vllm_version": self.vllm_version,
            "task_type": _wire(self.task_type),
            "config": self.config,
            "execution_policy": _wire(self.execution_policy),
            "execution_policy_config": self.execution_policy_config,
            "worker_id": self.worker_id,
            "gpu_resource_id": self.gpu_resource_id,
        }
        out.update({k: v for k, v in optional.items() if v is not None})
        return out


@dataclass(frozen=True)
class EndpointRef:
    """Slugs addressing a workload's data-plane endpoint."""

    project_slug: str
    workload_slug: str


@dataclass(frozen=True)
class TextResult:
    """A completed chat result."""

    text: str
    model: str
    finish_reason: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TextChunk:
    """One streamed chunk of a chat completion (a ``chat.completion.chunk``).

    ``text`` is the delta for this frame — concatenate chunks to rebuild the
    full reply. ``finish_reason`` is set only on the terminal chunk.
    """

    text: str
    finish_reason: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ReadinessEvent:
    """One readiness progress update from :meth:`ManagementClient.wait_until_ready`.

    ``phase`` is one of ``scheduling`` / ``provisioning`` / ``bootstrapping`` /
    ``ready`` / ``error``; ``ready`` means the workload is serving and ``error``
    is a terminal failure. ``elapsed_ms`` is time since the wait started; ``step``
    is an allow-listed bootstrap step name when applicable.
    """

    phase: str
    message: str
    elapsed_ms: int = 0
    step: Optional[str] = None


@dataclass(frozen=True)
class EmbedResult:
    """An embeddings result, one vector per input."""

    embeddings: List[List[float]]
    model: str
    raw: Dict[str, Any] = field(default_factory=dict)


def _wire(value: Any) -> Any:
    """Render an enum (or str) to its wire string; pass ``None`` through."""
    if value is None:
        return None
    return value.value if hasattr(value, "value") else value
