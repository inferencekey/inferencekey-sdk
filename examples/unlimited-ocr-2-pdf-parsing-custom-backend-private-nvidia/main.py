"""Parse a whole PDF into Markdown with **Baidu Unlimited OCR 2** on one private NVIDIA GPU.

The canonical SDK shape (ensure() -> wait_until_ready() -> call -> delete()),
varying one axis vs the other examples: the whole document goes in ONE
request. The model reads every page in order and writes one Markdown answer
with a ``<PAGE>`` marker per page, so tables and paragraphs that continue on
the next page can be merged instead of being cut at the page break.

The backend is a **custom Python backend** (``unlimited-ocr-2``, published with
publish.py): it installs the authors' SGLang build from the model repository
and serves the BF16 checkpoint (6.7 GB). The endpoint is the OpenAI-compatible
``POST .../v1/chat/completions``. Send a link to the PDF (the backend
downloads it); a small local PDF can travel inline as an OpenAI ``file`` part,
but in our test a 1.7 MB PDF sent inline never reached the worker, so links
are the reliable way. Options (``pages 1-20``, ``crop_mode=false``) go in the
message text, because the platform forwards the standard chat fields only.

Outputs land in ./outputs/<pdf name>.md (Markdown) and .json (timings).

Run:  cp .env.example .env  # fill in real values
      python publish.py     # once, uploads the backend to your tenant
      python main.py
"""

from __future__ import annotations

import base64
import json
import os
import sys
import time
from pathlib import Path

import requests

from inferencekey import ExecutionPolicy, InferenceKeyError, ManagementClient, WorkloadSpec

SLUG = os.environ.get("UOCR_WORKLOAD_SLUG", "unlimited-ocr-2")
BACKEND_SLUG = "unlimited-ocr-2"  # the slug publish.py registered
MODEL = "baidu/Unlimited-OCR-2"
OUT_DIR = Path(__file__).resolve().parent / "outputs"


def _load_dotenv() -> None:
    env = Path(__file__).resolve().parent / ".env"
    if env.is_file():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())


def parse_pdf(url: str, api_key: str, source: str, pages: str | None = None, crop_mode: bool = True,
              timeout: int = 3600) -> dict:
    """One streamed chat request with the whole document; returns the Markdown and timings.

    ``source`` is a link to a PDF (the backend downloads it) or a local PDF path
    (sent inline as an OpenAI ``file`` part). Options travel in the message
    text, because the platform forwards the standard chat fields only."""
    options = (f" pages {pages}" if pages else "") + ("" if crop_mode else " crop_mode=false")
    if source.startswith(("http://", "https://")):
        content = [{"type": "text", "text": f"Parse this document: {source}{options}"}]
    else:
        pdf = Path(source)
        content = [{"type": "file", "file": {"filename": pdf.name,
                                             "file_data": "data:application/pdf;base64," + base64.b64encode(pdf.read_bytes()).decode()}},
                   {"type": "text", "text": f"Parse this document.{options}"}]
    body = {"model": "default", "stream": True, "messages": [{"role": "user", "content": content}]}
    t0 = time.time()
    first = None
    parts, usage = [], {}
    with requests.post(url, headers={"Authorization": f"Bearer {api_key}"}, json=body, stream=True,
                       timeout=timeout) as resp:
        resp.raise_for_status()
        resp.encoding = "utf-8"  # text/event-stream without a charset would decode as Latin-1
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            event = json.loads(data)
            usage = event.get("usage") or usage
            for choice in event.get("choices") or []:
                piece = (choice.get("delta") or {}).get("content") or ""
                if piece:
                    first = first or time.time() - t0
                    parts.append(piece)
    seconds = time.time() - t0
    text = "".join(parts)
    n_pages = text.count("<PAGE>")
    return {"markdown": text, "pages": n_pages, "seconds": round(seconds, 2),
            "first_token_s": round(first or 0, 2),
            "pages_per_minute": round(n_pages / seconds * 60, 1) if seconds else None,
            "completion_tokens": usage.get("completion_tokens"),
            "tokens_per_second": round(usage["completion_tokens"] / seconds, 1) if usage.get("completion_tokens") else None}


def main() -> int:
    _load_dotenv()
    source = os.environ.get("IK_PDF", "document.pdf")
    if not source.startswith(("http://", "https://")) and not Path(source).is_file():
        print(f"PDF not found: {source}. Set IK_PDF to a link to a PDF, or a local PDF of up to about 1 MB.",
              file=sys.stderr)
        return 1
    stem = Path(source.split("?")[0]).stem or "document"
    mgmt = ManagementClient.from_env()
    # Backend knobs reach the backend process as environment variables (see backend.py).
    backend_env = {k: v for k, v in os.environ.items() if k.startswith("UOCR_") and k != "UOCR_WORKLOAD_SLUG"}
    # The first start builds the SGLang environment and downloads 6.7 GB: allow 30 minutes.
    config = {"ready_timeout_secs": 1800, "request_timeout_secs": 3600, "env": backend_env}

    spec = WorkloadSpec(
        name="Unlimited OCR 2",
        slug=SLUG,
        model=MODEL,
        backend=BACKEND_SLUG,  # a custom backend slug instead of a Backend enum
        task_type="text2text",
        worker_id=os.environ["IK_WORKER_ID"],
        config=config,
        # Private + fixed: one always-on replica on your GPU until delete() runs.
        execution_policy=ExecutionPolicy.FIXED,
    )

    try:
        t0 = time.time()
        ref = mgmt.ensure(spec)
        print(f"ensured {ref.project_slug}/{ref.workload_slug}")
        print("waiting for the worker (first start: SGLang install, 6.7 GB download, CUDA graphs)...")
        mgmt.wait_until_ready(ref.workload_slug, timeout=1800)
        print(f"ready after {time.time() - t0:.0f} s")

        base = os.environ.get("INFERENCEKEY_BASE_URL", "https://cloud.inferencekey.com").rstrip("/")
        url = f"{base}/endpoint/{os.environ['INFERENCEKEY_PROJECT']}/{ref.workload_slug}/v1/chat/completions"
        # One view per page (crop_mode=false) was faster and kept long tables intact in our test;
        # IK_CROP_MODE=true uses the authors' default (page tiles), better for very small print.
        crop = os.environ.get("IK_CROP_MODE", "false").lower() in ("1", "true", "yes")
        out = parse_pdf(url, os.environ["INFERENCEKEY_API_KEY"], source, os.environ.get("IK_PAGES"), crop_mode=crop)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / f"{stem}.md").write_text(out["markdown"])
        (OUT_DIR / f"{stem}.json").write_text(json.dumps({k: v for k, v in out.items() if k != "markdown"}, indent=1))
        print(f"\n{out['pages']} pages in {out['seconds']} s ({out['pages_per_minute']} pages/min, "
              f"{out['completion_tokens']} tokens, first text after {out['first_token_s']} s)")
        print(out["markdown"][:800])
    except InferenceKeyError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except requests.HTTPError as e:
        print(f"request failed: {e}: {e.response.text[:300]}", file=sys.stderr)
        return 1
    finally:
        # Tear down so the GPU stops being reserved. IK_KEEP_ALIVE=1 keeps the workload.
        if os.environ.get("IK_KEEP_ALIVE") == "1":
            print(f"IK_KEEP_ALIVE=1: keeping {SLUG}; delete it with mgmt.delete('{SLUG}')")
        else:
            print(f"deleted={mgmt.delete(SLUG)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
