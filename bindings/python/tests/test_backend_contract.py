"""Unit tests for the custom-backend contract — no torch, no HTTP."""

from __future__ import annotations

import pytest

from inferencekey.backend import (
    TASK_TYPES,
    BackendContext,
    BackendManifest,
    CustomBackend,
    Job,
    Result,
)


class MinimalBackend(CustomBackend):
    """A torch-free subclass: it just records setup and echoes the input."""

    def __init__(self) -> None:
        self.setups = 0

    def setup(self, ctx: BackendContext) -> None:
        self.setups += 1
        self.greeting = ctx.config.get("greeting", "hi")

    def process(self, job: Job) -> Result:
        return Result(output={"echo": job.input, "greeting": self.greeting})


def test_subclass_satisfies_contract() -> None:
    b = MinimalBackend()
    b.setup(BackendContext(config={"greeting": "hello"}, port=1234))
    assert b.setups == 1
    res = b.process(Job(id="j1", input={"x": 1}))
    assert isinstance(res, Result)
    assert res.output == {"echo": {"x": 1}, "greeting": "hello"}


def test_abc_cannot_instantiate_base() -> None:
    with pytest.raises(TypeError):
        CustomBackend()  # type: ignore[abstract]


def test_job_and_result_roundtrip_wire() -> None:
    job = Job.from_wire({"id": "j1", "input": {"a": [1, 2]}})
    assert job.id == "j1"
    assert job.input == {"a": [1, 2]}
    assert job.to_wire() == {"id": "j1", "input": {"a": [1, 2]}}
    assert Result(output={"k": "v"}).to_wire() == {"output": {"k": "v"}}


def test_job_input_defaults_to_empty_dict() -> None:
    assert Job.from_wire({"id": "j1"}).input == {}


@pytest.mark.parametrize(
    "bad",
    [
        {},  # missing id
        {"id": ""},  # empty id
        {"id": 5},  # non-string id
        {"id": "j1", "input": []},  # input not an object
        "not a dict",
    ],
)
def test_job_from_wire_rejects_bad_payloads(bad: object) -> None:
    with pytest.raises(ValueError):
        Job.from_wire(bad)  # type: ignore[arg-type]


def test_backend_context_defaults() -> None:
    ctx = BackendContext()
    assert ctx.config == {}
    assert ctx.port == 0


# --- C-1: BackendManifest / CustomBackend.manifest() ---


class DescribedBackend(CustomBackend):
    name = "described"
    version = "0.2.0"
    task_type = "embedding"
    requirements = "requirements.txt"

    def setup(self, ctx: BackendContext) -> None:  # pragma: no cover
        pass

    def process(self, job: Job) -> Result:  # pragma: no cover
        return Result()


def test_manifest_reads_declared_class_attributes() -> None:
    m = DescribedBackend().manifest()
    assert isinstance(m, BackendManifest)
    assert m.to_wire() == {
        "name": "described",
        "version": "0.2.0",
        "task_type": "embedding",
        "requirements": "requirements.txt",
    }
    assert m.task_type in TASK_TYPES


def test_manifest_defaults_to_class_name_when_undeclared() -> None:
    m = MinimalBackend().manifest()
    assert m.name == "MinimalBackend"
    assert m.version == "" and m.task_type == "" and m.requirements == ""
