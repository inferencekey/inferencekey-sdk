"""Package backend.py and publish it to your tenant as a custom backend.

Run once (and again whenever you change backend.py: bump VERSION first).

    python publish.py                  # full multimodal model, slug embeddinggemma2-multimodal
    python publish.py --variant text   # text and code only, slug embeddinggemma2-text

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
VERSION = "0.1.3"
VARIANTS = {
    # variant: (backend slug, entrypoint class, description)
    "multimodal": (
        "embeddinggemma2-multimodal",
        "EmbeddingGemma2Backend",
        "EmbeddingGemma 2 (Google DeepMind): text, code, image, video and audio in one 768-d space",
    ),
    "text": (
        "embeddinggemma2-text",
        "EmbeddingGemma2TextBackend",
        "EmbeddingGemma 2 text and code only (vision and audio encoders not loaded)",
    ),
}


def _load_dotenv() -> None:
    env = HERE / ".env"
    if env.is_file():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())


def main() -> int:
    _load_dotenv()
    variant = sys.argv[2] if len(sys.argv) > 2 and sys.argv[1] == "--variant" else "multimodal"
    slug, entry_class, description = VARIANTS[variant]
    pkg = package_backend(
        src=str(HERE / "backend.py"),
        entrypoint=f"backend:{entry_class}",
        requirements=str(HERE / "requirements.txt"),
        name=slug,
        slug=slug,
        version=VERSION,
        task_type="embedding",
        requires_python=">=3.11,<3.14",
        description=description,
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
