# EmbeddingGemma 2 (custom backend, private NVIDIA): text, code, images, video and audio in one vector space

Serves [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) from
Google DeepMind (Apache-2.0, 740M parameters), an embedding model that maps
text, code, images, video and audio into the same 768-dimensional space. A text
query can find a photo, a video clip or a spoken recording directly.

The built-in embedding runtimes (`sglang`, `llamacpp`, `ollama`) only take
text, and vLLM 0.31 has no `EmbeddingGemma2Model` architecture, so this example
ships a **custom Python backend** that runs the model the way its card
documents (`sentence-transformers`), publishes it to your tenant and deploys it
on your NVIDIA worker. The axis it varies: `backend` is a custom backend slug,
and the request carries media, not only strings.

## Compatibility
- SDK: local build in this repo (see [CONTRIBUTING](../CONTRIBUTING.md#4-depending-on-the-sdk))
- Language: Python >= 3.9 for the client scripts; the backend runs on Python
  3.11 to 3.13 (the worker provisions it with `uv`)
- Placement: private NVIDIA with bf16 (Ampere or newer). The backend installs
  the CUDA 12.6 torch build, so it does not need the 580 driver that PyPI's
  CUDA 13 torch needs (tested on driver 570). About 1.5 GB
  of VRAM loaded, more for long-document batches (see Measured)
- Backend: custom (`embeddinggemma2-multimodal`, or `embeddinggemma2-text`)   Policy: `fixed`

## Prerequisites
- An SDK token (`ik_sdk_`) and an API key (`ik_live_`) for the project, see
  [Tokens](../README.md#tokens).
- A registered NVIDIA worker assigned to the project (`IK_WORKER_ID`).
- To publish the backend from the script: your tenant id and a dashboard
  session token. Or upload `dist/*.tar.gz` from Dashboard -> Custom backends.
- Some files to search in `./media` (jpg/png/webp, wav/mp3/ogg/flac/m4a,
  mp4/webm/mov), or pass `--media DIR`.

## Run
```bash
cp .env.example .env        # fill in real values
python -m venv .venv && . .venv/bin/activate
pip install maturin requests
( cd ../../bindings/python && maturin develop --release )
python publish.py           # once: packages backend.py and uploads it (6 KB)
python main.py              # ensure -> wait until ready -> text + cross-modal search
```

Text-and-code only (the vision and audio encoders are never loaded):
`python publish.py --variant text`, then `EG2_VARIANT=text python main.py`.

## What it does
1. `publish.py` packages `backend.py` + `requirements.txt` (code only, no
   weights) and registers it as a custom backend of your tenant.
2. `main.py` calls `ensure()` with `backend="embeddinggemma2-multimodal"` and
   `task_type="embedding"`, pinned to your worker with a `fixed` policy.
3. On first start the worker builds the backend venv (torch, transformers,
   sentence-transformers, PyAV) and the backend downloads 1.5 GB of weights.
4. `main.py` embeds a query and four documents with the typed SDK call
   `Endpoint.embed()`, then embeds every file in `./media` and ranks text
   queries against them. Results go to `outputs/results.json`.

## Request shape
`POST /endpoint/<project>/<slug>/v1/embeddings`, OpenAI-compatible plus three
optional fields:

```json
{
  "model": "default",
  "input": [
    "task: search result | query: an animal sleeping in the snow",
    {"image": "data:image/jpeg;base64,..."},
    {"audio": "data:audio/wav;base64,..."},
    {"video": "https://example.com/clip.mp4"},
    {"text": "Listing: <|image|> close-up <|image|>", "image": ["data:...", "data:..."]}
  ],
  "dimensions": 256,
  "prompt_name": "SearchQuery",
  "encoding_format": "float"
}
```

- `input`: strings, or objects with `text` and/or `image`, `audio`, `video`
  (data URI, raw base64 or URL; `image` and `video` also take a list). Text with
  `<|image|>`, `<|audio|>`, `<|video|>` placeholders interleaves media into ONE
  vector.
- `prompt_name`: the card's task prefixes for plain strings (`SearchQuery`,
  `Document`, `QuestionAnswering`, `CodeRetrieval`, `Classification`,
  `Clustering`, `SentenceSimilarity`, ...). Media gets no prefix.
- `dimensions`: Matryoshka truncation to 768, 512, 256 or 128, re-normalized.
- Video is sampled at 1 frame per second (at most 32 frames), audio is
  resampled to 16 kHz mono, both decoded with PyAV (no system FFmpeg needed).
- The response adds `timings` (decode and model seconds) next to `usage`.

## Measured (RunPod NVIDIA RTX 2000 Ada 16 GB, driver 570, 2026-10-14)

| | |
| --- | --- |
| `ensure()` to ready, cold pod | 78 s (venv install 49 s, weights download and load 29 s) |
| `ensure()` to ready, weights cached | 12 s |
| Re-published backend, hot recycle | 13 s |
| VRAM, loaded (`torch.cuda.memory_allocated`) | 1,432 MiB multimodal, 517 MiB text-only |
| Peak allocated: 1 image / 8 images / 32 docs of ~500 tokens / one 8,000-token doc | 1.5 / 2.0 / 3.3 / 2.6 GB |
| Peak for 8 docs of ~6,000 tokens in one request | about 10 GB (the process held 11.6 GB) |

Model time per request (median, bf16):

| Input | ms |
| --- | --- |
| 1 short text query | 36 |
| 256 short texts | 204 |
| 32 documents, ~500 tokens each | 688 |
| 8 documents, ~6,000 tokens each | 4,416 |
| 8.6 s of speech | 158 |
| 1 image | 111 |
| 8 images | 867 |
| 6 s video clip | 307 |
| text + 2 images + audio, one vector | 232 |

The pod cost 0.24 USD per hour; the whole session ran 29 minutes, 0.12 USD.

## Cost & cleanup
`fixed` keeps one replica on your worker until you delete the workload. The
script deletes it on exit (also on error and Ctrl-C). With `IK_KEEP_ALIVE=1` it
keeps it, so you can try more inputs from the dashboard tester (text only) or
the API; then delete it with
`ManagementClient.from_env().delete("embeddinggemma2-multimodal")`. If the GPU
is a rented pod, also stop the pod: deleting the workload does not.

## Troubleshooting
- **`uv pip install -r requirements.txt failed`**: uv takes a package only
  from the FIRST index that has it, and the PyTorch index also hosts old
  `requests`, `pillow` and `numpy`. A version floor on those makes the install
  unsolvable; keep them unpinned (as here) if you add the PyTorch index.
- **Images and video slow (0.5 s per image)**: rented containers often show
  every host core to torch (64 here) while the quota is a handful (6.8 vCPU),
  and the image preprocessing on the CPU thrashes. The backend caps torch
  threads to the cgroup quota; override with `EG2_TORCH_THREADS` on the worker.
- **GPU memory stays high after a big batch**: PyTorch caches what it used. The
  backend calls `torch.cuda.empty_cache()` after requests above 16,384 tokens.
- **NaN or odd vectors**: never run this model in float16 (the card says so).
  The backend picks bf16 on GPUs that support it and float32 otherwise.
- **Large media**: a request travels as one message with an 8 MB limit; send
  big files one per request, or pass URLs.

---
See [CONTRIBUTING.md](../CONTRIBUTING.md) and the
[tokens & placement overview](../README.md#tokens).
