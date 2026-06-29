"""Unit tests for backend packaging — torch-free, stdlib only.

Packaging is pure file manipulation: it never imports the developer's backend
nor ``torch``. These tests prove that by using a *dummy* backend module that
subclasses :class:`CustomBackend` with no torch import, and by reading the
manifest back out of the artifact without importing anything from it.
"""

from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path

import pytest

from inferencekey.backend import (
    BackendPackage,
    package_backend,
    read_manifest_from_archive,
)
from inferencekey.errors import BackendError, InferenceKeyError, PackagingError
from inferencekey.backend.packaging import MANIFEST_NAME, SDK_PROTOCOL


# A trivial torch-free backend source the tests package. It subclasses
# CustomBackend but imports no torch, so writing it to disk and bundling it is
# safe and self-contained.
_DUMMY_SOURCE = '''\
from inferencekey.backend import CustomBackend, Job, Result


class DummyBackend(CustomBackend):
    def setup(self, ctx):
        self.greeting = ctx.config.get("greeting", "hi")

    def process(self, job):
        return Result(output={"echo": job.input})
'''


@pytest.fixture()
def dummy_src(tmp_path: Path) -> Path:
    """A single-file backend on disk."""
    src = tmp_path / "src" / "backend.py"
    src.parent.mkdir(parents=True)
    src.write_text(_DUMMY_SOURCE, encoding="utf-8")
    return src


@pytest.fixture()
def dummy_requirements(tmp_path: Path) -> Path:
    req = tmp_path / "requirements.txt"
    req.write_text("torch>=2.0\n", encoding="utf-8")
    return req


def _names(archive: str) -> list:
    with tarfile.open(archive, "r:gz") as tar:
        return sorted(tar.getnames())


def test_package_backend_creates_artifact(dummy_src, dummy_requirements, tmp_path):
    out = tmp_path / "out"
    pkg = package_backend(
        src=str(dummy_src),
        entrypoint="backend:DummyBackend",
        requirements=str(dummy_requirements),
        name="dummy",
        version="0.1.0",
        task_type="text2text",
        out_dir=str(out),
    )

    assert isinstance(pkg, BackendPackage)
    assert Path(pkg.path).is_file()
    assert pkg.path == str(out / "dummy-0.1.0.tar.gz")
    assert pkg.manifest["name"] == "dummy"
    assert pkg.manifest["sdk_protocol"] == SDK_PROTOCOL


def test_artifact_contains_manifest_code_and_requirements(
    dummy_src, dummy_requirements, tmp_path
):
    pkg = package_backend(
        src=str(dummy_src),
        entrypoint="backend:DummyBackend",
        requirements=str(dummy_requirements),
        name="dummy",
        version="0.1.0",
        task_type="text2text",
        out_dir=str(tmp_path / "out"),
    )
    names = _names(pkg.path)
    assert MANIFEST_NAME in names
    assert "requirements.txt" in names
    assert "backend.py" in names

    # The bundled requirements are the dev's file, verbatim.
    with tarfile.open(pkg.path, "r:gz") as tar:
        req = tar.extractfile("requirements.txt").read().decode("utf-8")
    assert req == "torch>=2.0\n"


def test_read_manifest_from_archive_static(dummy_src, dummy_requirements, tmp_path):
    pkg = package_backend(
        src=str(dummy_src),
        entrypoint="backend:DummyBackend",
        requirements=str(dummy_requirements),
        name="dummy",
        version="2.3.4",
        task_type="classification",
        description="a dummy backend",
        out_dir=str(tmp_path / "out"),
    )
    manifest = read_manifest_from_archive(pkg.path)
    assert manifest["name"] == "dummy"
    assert manifest["version"] == "2.3.4"
    assert manifest["task_type"] == "classification"
    assert manifest["entrypoint"] == "backend:DummyBackend"
    assert manifest["sdk_protocol"] == SDK_PROTOCOL
    assert manifest["description"] == "a dummy backend"
    # Reading the manifest must not have imported the backend module.
    import sys

    assert "backend" not in sys.modules


def test_sha256_and_size_match_disk(dummy_src, dummy_requirements, tmp_path):
    pkg = package_backend(
        src=str(dummy_src),
        entrypoint="backend:DummyBackend",
        requirements=str(dummy_requirements),
        name="dummy",
        version="0.1.0",
        task_type="text2text",
        out_dir=str(tmp_path / "out"),
    )
    data = Path(pkg.path).read_bytes()
    assert pkg.size_bytes == len(data)
    assert pkg.sha256 == hashlib.sha256(data).hexdigest()


def test_package_directory_source(tmp_path):
    pkgdir = tmp_path / "mypkg"
    pkgdir.mkdir()
    (pkgdir / "__init__.py").write_text("", encoding="utf-8")
    (pkgdir / "backend.py").write_text(_DUMMY_SOURCE, encoding="utf-8")

    pkg = package_backend(
        src=str(pkgdir),
        entrypoint="mypkg.backend:DummyBackend",
        name="dummy",
        version="0.1.0",
        out_dir=str(tmp_path / "out"),
    )
    names = _names(pkg.path)
    assert "mypkg/__init__.py" in names
    assert "mypkg/backend.py" in names


def test_requirements_optional_yields_empty(dummy_src, tmp_path):
    pkg = package_backend(
        src=str(dummy_src),
        entrypoint="backend:DummyBackend",
        name="dummy",
        version="0.1.0",
        out_dir=str(tmp_path / "out"),
    )
    with tarfile.open(pkg.path, "r:gz") as tar:
        req = tar.extractfile("requirements.txt").read()
    assert req == b""


def test_task_type_optional(dummy_src, tmp_path):
    pkg = package_backend(
        src=str(dummy_src),
        entrypoint="backend:DummyBackend",
        name="dummy",
        version="0.1.0",
        out_dir=str(tmp_path / "out"),
    )
    assert pkg.manifest["task_type"] == ""


def test_invalid_entrypoint_raises_and_no_artifact(dummy_src, tmp_path):
    out = tmp_path / "out"
    with pytest.raises(PackagingError):
        package_backend(
            src=str(dummy_src),
            entrypoint="backend.DummyBackend",  # no colon
            name="dummy",
            version="0.1.0",
            out_dir=str(out),
        )
    assert not (out / "dummy-0.1.0.tar.gz").exists()


def test_empty_entrypoint_side_raises(dummy_src, tmp_path):
    with pytest.raises(PackagingError):
        package_backend(
            src=str(dummy_src),
            entrypoint="backend:",
            name="dummy",
            version="0.1.0",
            out_dir=str(tmp_path / "out"),
        )


def test_unknown_task_type_raises_and_no_artifact(dummy_src, tmp_path):
    out = tmp_path / "out"
    with pytest.raises(PackagingError):
        package_backend(
            src=str(dummy_src),
            entrypoint="backend:DummyBackend",
            name="dummy",
            version="0.1.0",
            task_type="nope",
            out_dir=str(out),
        )
    assert not (out / "dummy-0.1.0.tar.gz").exists()


def test_missing_src_raises(tmp_path):
    with pytest.raises(PackagingError):
        package_backend(
            src=str(tmp_path / "does-not-exist.py"),
            entrypoint="backend:DummyBackend",
            name="dummy",
            version="0.1.0",
            out_dir=str(tmp_path / "out"),
        )


def test_missing_requirements_file_raises(dummy_src, tmp_path):
    with pytest.raises(PackagingError):
        package_backend(
            src=str(dummy_src),
            entrypoint="backend:DummyBackend",
            requirements=str(tmp_path / "nope.txt"),
            name="dummy",
            version="0.1.0",
            out_dir=str(tmp_path / "out"),
        )


def test_packaging_error_is_in_hierarchy():
    assert issubclass(PackagingError, BackendError)
    assert issubclass(PackagingError, InferenceKeyError)


def test_read_manifest_rejects_traversal_member(tmp_path):
    # Hand-craft a malicious tar with a '..' member alongside a manifest.
    bad = tmp_path / "bad.tar.gz"
    with tarfile.open(bad, "w:gz") as tar:
        payload = json.dumps({"name": "x"}).encode("utf-8")
        info = tarfile.TarInfo(name=MANIFEST_NAME)
        info.size = len(payload)
        import io

        tar.addfile(info, io.BytesIO(payload))
        evil = tarfile.TarInfo(name="../escape.txt")
        evil.size = 0
        tar.addfile(evil, io.BytesIO(b""))

    with pytest.raises(PackagingError):
        read_manifest_from_archive(str(bad))


def test_read_manifest_missing_raises(tmp_path):
    empty = tmp_path / "no-manifest.tar.gz"
    with tarfile.open(empty, "w:gz") as tar:
        info = tarfile.TarInfo(name="backend.py")
        info.size = 0
        import io

        tar.addfile(info, io.BytesIO(b""))
    with pytest.raises(PackagingError):
        read_manifest_from_archive(str(empty))


def test_cli_creates_artifact_and_prints(dummy_src, dummy_requirements, tmp_path, capsys):
    from inferencekey.backend import package as package_cli

    out = tmp_path / "out"
    rc = package_cli.main(
        [
            "--src",
            str(dummy_src),
            "--entrypoint",
            "backend:DummyBackend",
            "--requirements",
            str(dummy_requirements),
            "--name",
            "dummy",
            "--version",
            "0.1.0",
            "--task-type",
            "text2text",
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    printed = capsys.readouterr().out.strip().splitlines()
    artifact_path, sha = printed[0], printed[1]
    assert Path(artifact_path).is_file()
    assert sha == hashlib.sha256(Path(artifact_path).read_bytes()).hexdigest()


def test_cli_invalid_entrypoint_exits_nonzero(dummy_src, tmp_path):
    from inferencekey.backend import package as package_cli

    out = tmp_path / "out"
    rc = package_cli.main(
        [
            "--src",
            str(dummy_src),
            "--entrypoint",
            "no-colon",
            "--name",
            "dummy",
            "--version",
            "0.1.0",
            "--out",
            str(out),
        ]
    )
    assert rc == 2
    assert not (out / "dummy-0.1.0.tar.gz").exists()
