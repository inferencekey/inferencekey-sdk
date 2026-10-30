"""Unit tests for :mod:`inferencekey.enums` and the ``backend`` wire string.

Pure: no network, no native extension calls. The native backend set must stay
identical to ``Backend`` in ``core/src/domain/enums.rs`` and ``bindings/node/lib.ts``;
any other wire string is a custom backend slug and is passed through verbatim.
"""

from __future__ import annotations

import json

from inferencekey import Backend, ExecutionPolicy, WorkloadSpec

#: The native backends of the Rust core, as wire strings (8 natives; custom
#: slugs are not enum members).
NATIVE_BACKENDS = {
    "ollama",
    "vllm",
    "vllm-omni",
    "vllm-exl3",
    "sglang",
    "llamacpp",
    "stablediffusioncpp",
    "strata",
}


def test_backend_values_are_the_native_set() -> None:
    assert {b.value for b in Backend} == NATIVE_BACKENDS
    assert len(Backend) == 8


def test_strata_wire_string() -> None:
    assert Backend.STRATA == "strata"
    assert Backend("strata") is Backend.STRATA


def test_strata_spec_serializes_backend_and_config() -> None:
    command = "strata --family coder --model IQ1_M --context 262144 --kv int8"
    spec = WorkloadSpec(
        name="Strata coder",
        slug="strata-coder",
        model="qwen3.8-flash-next-coder-iq1_m",
        backend=Backend.STRATA,
        command=command,
        task_type="text2text",
        config={"strata_version": "v0.1.40"},
        execution_policy=ExecutionPolicy.FIXED,
    )
    wire = spec.to_wire()
    assert wire["backend"] == "strata"
    assert json.loads(json.dumps(wire))["backend"] == "strata"
    assert wire["command"] == command
    assert wire["config"] == {"strata_version": "v0.1.40"}


def test_custom_backend_slug_passes_through() -> None:
    spec = WorkloadSpec(name="Echo", slug="echo", model="echo", backend="my-cool-backend")
    assert spec.to_wire()["backend"] == "my-cool-backend"
