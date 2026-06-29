"""CLI to package a custom backend into a ``.tar.gz`` artifact.

Thin :mod:`argparse` wrapper over :func:`inferencekey.backend.packaging.package_backend`::

    python -m inferencekey.backend.package \\
        --src examples/custom-backend-echo/backend.py \\
        --entrypoint backend:EchoLinearBackend \\
        --requirements examples/custom-backend-echo/requirements.txt \\
        --name echo --version 0.1.0 --task-type text2text \\
        --out /tmp/out

On success it prints the artifact path and its sha256 (one per line) and exits
``0``; on a :class:`~inferencekey.errors.PackagingError` it prints the message to
stderr and exits ``2``, having written no artifact. Imports only stdlib and the
pure-Python packaging module — never ``torch`` nor the backend.
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from ..errors import PackagingError
from .packaging import package_backend


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m inferencekey.backend.package",
        description="Package a custom backend into a .tar.gz artifact.",
    )
    parser.add_argument(
        "--src",
        required=True,
        help="Path to the backend code: a .py file or a package directory.",
    )
    parser.add_argument(
        "--entrypoint",
        required=True,
        help="Backend entrypoint as 'module:Class'.",
    )
    parser.add_argument(
        "--requirements",
        default=None,
        help="Path to requirements.txt (optional; empty one bundled if omitted).",
    )
    parser.add_argument("--name", required=True, help="Backend name.")
    parser.add_argument(
        "--slug",
        default=None,
        help="Publish slug the Manager registers under (optional; defaults to --name).",
    )
    parser.add_argument("--version", required=True, help="Backend version string.")
    parser.add_argument(
        "--task-type",
        dest="task_type",
        default=None,
        help="One of the SDK TASK_TYPES (optional).",
    )
    parser.add_argument(
        "--out",
        dest="out_dir",
        required=True,
        help="Output directory for the artifact.",
    )
    parser.add_argument(
        "--description",
        default=None,
        help="Optional human-readable description.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Parse ``argv`` (defaults to ``sys.argv``), package, print path + sha256."""
    args = _build_parser().parse_args(argv)
    try:
        pkg = package_backend(
            src=args.src,
            entrypoint=args.entrypoint,
            requirements=args.requirements,
            name=args.name,
            slug=args.slug,
            version=args.version,
            task_type=args.task_type,
            out_dir=args.out_dir,
            description=args.description,
        )
    except PackagingError as exc:
        print(f"packaging failed: {exc}", file=sys.stderr, flush=True)
        return 2
    print(pkg.path)
    print(pkg.sha256)
    return 0


if __name__ == "__main__":  # pragma: no cover — exercised via the CLI tests
    raise SystemExit(main())
