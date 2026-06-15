"""InferenceKey SDK — Python.

Declare AI workloads in code, ensure they exist on the platform, and call the
resulting OpenAI-compatible endpoints. The heavy lifting lives in the Rust core
(via the native ``_inferencekey`` extension); this package is the ergonomic,
typed surface over it.

Two planes, two tokens:

* ``ik_sdk_`` (control) — provision/reconcile workloads. See
  :class:`ManagementClient`.
* ``ik_live_`` (data, per-workload) — call inference. See :class:`DataClient`.

Quickstart::

    from inferencekey import ManagementClient, DataClient, WorkloadSpec, Backend

    mgmt = ManagementClient.from_env(project="acme")
    ref = mgmt.ensure(WorkloadSpec(
        name="support-bot", slug="support-bot",
        model="meta-llama/Llama-3.1-8B-Instruct", backend=Backend.VLLM,
        command="vllm serve meta-llama/Llama-3.1-8B-Instruct --max-model-len 8192",
    ))

    data = DataClient.from_env(project="acme")
    out = data.endpoint(ref.workload_slug, api_key="ik_live_...").generate_text(prompt="Hola")
    print(out.text)
"""

from .enums import Backend, ExecutionPolicy, OnDrift, TaskType
from .errors import (
    ApiError,
    AuthError,
    ConfigurationError,
    InferenceKeyError,
    PermissionDenied,
    ValidationError,
)
from .clients import DataClient, Endpoint, ManagementClient
from .types import EmbedResult, EndpointRef, TextResult, WorkloadSpec

__all__ = [
    "Backend",
    "TaskType",
    "OnDrift",
    "ExecutionPolicy",
    "WorkloadSpec",
    "EndpointRef",
    "TextResult",
    "EmbedResult",
    "ManagementClient",
    "DataClient",
    "Endpoint",
    "InferenceKeyError",
    "PermissionDenied",
    "AuthError",
    "ValidationError",
    "ConfigurationError",
    "ApiError",
]

__version__ = "0.1.0"
