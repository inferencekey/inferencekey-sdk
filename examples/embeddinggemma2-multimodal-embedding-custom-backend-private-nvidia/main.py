"""Search text, images, audio and video with ONE embedding model: EmbeddingGemma 2 on a private NVIDIA GPU.

The canonical SDK shape (ensure() -> wait_until_ready() -> call -> delete()),
varying one axis vs the other examples: the backend is a **custom Python
backend** (`embeddinggemma2-multimodal`, published with publish.py) because no
built-in embedding runtime takes images, audio or video.

1. Text retrieval through the typed SDK call `Endpoint.embed()`.
2. Cross-modal retrieval: every file in ./media (jpg/png/webp, wav/mp3/ogg/flac,
   mp4/webm/mov) is embedded as a media item, then text queries are ranked
   against them. The data plane has no typed multimodal helper yet, so this
   part POSTs OpenAI-style JSON to `/v1/embeddings` with the `ik_live_` key.

Results go to ./outputs/results.json.

Run:  cp .env.example .env  # fill in real values
      python publish.py     # once, uploads the backend to your tenant
      python main.py [--media DIR] [--query "..."]
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import sys
import time
from pathlib import Path

import requests

from inferencekey import DataClient, ExecutionPolicy, InferenceKeyError, ManagementClient, WorkloadSpec

HERE = Path(__file__).resolve().parent
VARIANT = os.environ.get("EG2_VARIANT", "multimodal")  # "multimodal" or "text"
SLUG = f"embeddinggemma2-{VARIANT}"
BACKEND_SLUG = SLUG  # the custom backend slug publish.py registered
OUT_DIR = HERE / "outputs"
KIND_BY_EXT = {
    ".jpg": "image", ".jpeg": "image", ".png": "image", ".webp": "image",
    ".wav": "audio", ".mp3": "audio", ".ogg": "audio", ".flac": "audio", ".m4a": "audio",
    ".mp4": "video", ".webm": "video", ".mov": "video",
}
DOCS = [
    "The northern lights are caused by charged particles from the sun hitting the atmosphere.",
    "Sourdough bread needs a starter, flour, water and salt, and a long cold proof.",
    "def binary_search(xs, x):\n    lo, hi = 0, len(xs)\n    while lo < hi: ...",
    "El Real Madrid ganó la final en el último minuto con un gol desde fuera del área.",
]
TEXT_QUERY = "What causes the aurora borealis?"
DEFAULT_QUERIES = [
    "an animal sleeping in the snow",
    "a bowl of noodle soup",
    "a weather forecast with rain and wind",
    "a goal in a football match",
    "how to knead bread dough",
    "a city street at night in the rain",
]


def _load_dotenv() -> None:
    env = HERE / ".env"
    if env.is_file():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())


def _data_uri(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()


def _cos(a: list, b: list) -> float:
    # Vectors come back L2-normalized, so the dot product is the cosine.
    return sum(x * y for x, y in zip(a, b))


def main() -> int:
    _load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--media", default=str(HERE / "media"), help="folder with images, audio and video")
    ap.add_argument("--query", action="append", help="text query (repeatable)")
    args = ap.parse_args()

    mgmt = ManagementClient.from_env()
    spec = WorkloadSpec(
        name=f"EmbeddingGemma 2 ({VARIANT})",
        slug=SLUG,
        model="google/embeddinggemma-2",
        backend=BACKEND_SLUG,  # a custom backend slug instead of a Backend enum
        task_type="embedding",
        worker_id=os.environ["IK_WORKER_ID"],
        # Private + fixed: one always-on replica on your GPU until delete() runs.
        execution_policy=ExecutionPolicy.FIXED,
    )
    results: dict = {"slug": SLUG}
    try:
        t0 = time.time()
        ref = mgmt.ensure(spec)
        print(f"ensured {ref.project_slug}/{ref.workload_slug}")
        # First start: the worker builds the backend venv (torch, transformers,
        # sentence-transformers) and the backend downloads 1.5 GB of weights.
        mgmt.wait_until_ready(ref.workload_slug, timeout=1800)
        results["ready_after_s"] = round(time.time() - t0, 1)
        print(f"ready after {results['ready_after_s']} s")

        # 1. Text retrieval with the typed SDK call.
        ep = DataClient.from_env().endpoint(ref.workload_slug)
        t = time.time()
        res = ep.embed(input=[f"task: search result | query: {TEXT_QUERY}"] + [f"title: none | text: {d}" for d in DOCS])
        q, docs = res.embeddings[0], res.embeddings[1:]
        ranked = sorted(((round(_cos(q, d), 4), DOCS[i][:60]) for i, d in enumerate(docs)), reverse=True)
        results["text"] = {"query": TEXT_QUERY, "dims": len(q), "round_trip_s": round(time.time() - t, 3), "ranking": ranked}
        print(f"text: {len(q)}-d vectors, best match {ranked[0]}")

        # 2. Cross-modal retrieval over ./media (skipped by the text-only variant).
        media = sorted(p for p in Path(args.media).glob("*") if p.suffix.lower() in KIND_BY_EXT)
        if VARIANT == "multimodal" and media:
            base = os.environ.get("INFERENCEKEY_BASE_URL", "https://cloud.inferencekey.com").rstrip("/")
            url = f"{base}/endpoint/{os.environ['INFERENCEKEY_PROJECT']}/{ref.workload_slug}/v1/embeddings"
            headers = {"Authorization": f"Bearer {os.environ['INFERENCEKEY_API_KEY']}"}
            vecs, timings = {}, []
            for p in media:  # one request per file keeps each message small
                kind = KIND_BY_EXT[p.suffix.lower()]
                t = time.time()
                r = requests.post(url, headers=headers, json={"model": "default", "input": [{kind: _data_uri(p)}]}, timeout=300)
                r.raise_for_status()
                body = r.json()
                vecs[p.name] = body["data"][0]["embedding"]
                timings.append({"file": p.name, "kind": kind, "round_trip_s": round(time.time() - t, 3), "backend": body.get("timings")})
                print(f"embedded {p.name} ({kind}) in {time.time() - t:.2f} s")
            queries = args.query or DEFAULT_QUERIES
            r = requests.post(url, headers=headers, json={"model": "default", "input": queries, "prompt_name": "SearchQuery"}, timeout=120)
            r.raise_for_status()
            qv = [d["embedding"] for d in r.json()["data"]]
            matches = {}
            for query, v in zip(queries, qv):
                scored = sorted(((round(_cos(v, e), 4), name) for name, e in vecs.items()), reverse=True)
                matches[query] = scored[:3]
                print(f"{query!r:45} -> {scored[0][1]} ({scored[0][0]})")
            results["media"] = {"timings": timings, "matches": matches}

        OUT_DIR.mkdir(exist_ok=True)
        (OUT_DIR / "results.json").write_text(json.dumps(results, indent=1, ensure_ascii=False))
        print(f"wrote {OUT_DIR / 'results.json'}")
    except InferenceKeyError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except requests.HTTPError as e:
        print(f"request failed: {e}: {e.response.text[:300]}", file=sys.stderr)
        return 1
    finally:
        # Tear down so the GPU stops being reserved. IK_KEEP_ALIVE=1 keeps the
        # workload (useful to try more inputs from the dashboard).
        if os.environ.get("IK_KEEP_ALIVE") == "1":
            print(f"IK_KEEP_ALIVE=1: keeping {SLUG}; delete it with mgmt.delete('{SLUG}')")
        else:
            print(f"deleted={mgmt.delete(SLUG)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
