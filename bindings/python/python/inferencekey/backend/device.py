"""Device selection helper for custom backends.

A custom backend runs on whatever GPU the assigned node has — NVIDIA (CUDA),
AMD (ROCm) or Apple (Metal/MPS) — but PyTorch spells those differently and the
right choice depends on both the installed ``torch`` build and the hardware.
:func:`pick_device` centralises that so every backend picks the accelerator the
same way instead of hand-rolling ``torch.cuda.is_available()`` checks.

**No torch at import time.** The rest of ``inferencekey.backend`` deliberately
never imports ``torch`` (the static packaging path must run without it). This
module honours that: ``torch`` is imported *lazily*, inside :func:`pick_device`,
so merely importing the subpackage still pulls in no heavy deps. Only a backend
that actually calls :func:`pick_device` — at ``setup()`` time, where torch is
already a dependency — triggers the import.

ROCm note: AMD's PyTorch build reports its GPU through the **``cuda``** device
string and ``torch.cuda.is_available()`` (ROCm reuses the CUDA API surface, with
``torch.version.hip`` set instead of ``torch.version.cuda``). So a single
``"cuda"`` probe covers both NVIDIA and AMD ROCm; Apple Silicon is the separate
``"mps"`` case.
"""

from __future__ import annotations

__all__ = ["pick_device"]


def pick_device(preferred: str = "auto") -> str:
    """Return the torch device string this backend should use.

    ``preferred``:

    * ``"auto"`` (default) — probe for an accelerator and fall back to CPU.
      Order: CUDA/ROCm (``"cuda"``) → Apple Metal (``"mps"``) → ``"cpu"``.
    * any explicit value (``"cuda"``, ``"cuda:1"``, ``"mps"``, ``"cpu"``) —
      returned as-is *if* it is actually usable; otherwise we warn and fall back
      to ``"cpu"`` rather than letting ``.to(device)`` blow up at runtime. This
      keeps an operator's explicit ``{"device": "cuda"}`` honest: on a node
      whose torch can't see a GPU it degrades to CPU with a clear log line
      instead of crashing every job.

    The probe imports ``torch`` lazily; a backend that never calls this never
    pays for it.
    """
    try:
        import torch
    except ImportError:
        # No torch at all — the only honest answer is CPU. A backend that needs
        # a GPU will fail loudly later when it actually touches torch.
        return "cpu"

    def cuda_ok() -> bool:
        # Covers NVIDIA CUDA and AMD ROCm (ROCm exposes the CUDA API surface).
        try:
            return bool(torch.cuda.is_available())
        except Exception:
            return False

    def mps_ok() -> bool:
        backend = getattr(torch.backends, "mps", None)
        if backend is None:
            return False
        try:
            return bool(backend.is_available())
        except Exception:
            return False

    if preferred == "auto":
        if cuda_ok():
            return "cuda"
        if mps_ok():
            return "mps"
        return "cpu"

    # Explicit request: validate it maps to something usable so we fail soft.
    root = preferred.split(":", 1)[0]
    if root == "cuda":
        if cuda_ok():
            return preferred
    elif root == "mps":
        if mps_ok():
            return preferred
    elif root == "cpu":
        return preferred
    else:
        # Unknown device string — trust the caller (custom torch backends can
        # register their own, e.g. "xpu"); if torch can't use it the backend's
        # own .to() will surface the error.
        return preferred

    import sys

    print(
        f"pick_device: requested device {preferred!r} is not available on this "
        f"node; falling back to 'cpu'",
        file=sys.stderr,
        flush=True,
    )
    return "cpu"
