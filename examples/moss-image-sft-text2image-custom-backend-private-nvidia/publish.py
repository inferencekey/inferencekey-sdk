"""Package backend.py and publish it to your tenant as a custom backend.

Run once (and again whenever you change backend.py: bump VERSION first).

    python publish.py

Reads INFERENCEKEY_BASE_URL, INFERENCEKEY_TENANT_ID and INFERENCEKEY_USER_TOKEN
(a dashboard session token: the upload is a tenant-level action, so it is
authenticated as a user, not with the project's ik_sdk_ token). You can also
skip this script and drag the .tar.gz into Dashboard -> Custom backends.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from inferencekey import publish_custom_backend
from inferencekey.backend import package_backend

HERE = Path(__file__).resolve().parent
SLUG = "moss-image-text2image"
VERSION = "0.2.0"


def main() -> int:
    pkg = package_backend(
        src=str(HERE / "backend.py"),
        entrypoint="backend:MossImageText2ImageBackend",
        requirements=str(HERE / "requirements.txt"),
        name=SLUG,
        slug=SLUG,
        version=VERSION,
        task_type="text2image",
        requires_python=">=3.10",
        description="MOSS-Image-SFT (OpenMOSS): text and layout to image, 50-step SFT or 8-step DMD LoRA",
        out_dir=str(HERE / "dist"),
    )
    print(f"packaged {pkg.path} ({pkg.size_bytes} bytes, sha256 {pkg.sha256[:12]})")

    tenant_id = os.environ.get("INFERENCEKEY_TENANT_ID")
    token = os.environ.get("INFERENCEKEY_USER_TOKEN")
    if not tenant_id or not token:
        print(
            "INFERENCEKEY_TENANT_ID / INFERENCEKEY_USER_TOKEN not set: upload the "
            "package from Dashboard -> Custom backends instead.",
            file=sys.stderr,
        )
        return 0

    result = publish_custom_backend(tenant_id=tenant_id, package_path=str(pkg.path), token=token)
    print(f"published custom backend slug={result.get('slug')} id={result.get('id')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
