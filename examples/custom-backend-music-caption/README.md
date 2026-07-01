# custom-backend-music-caption

A real-world **audio2text** custom backend for the InferenceKey SDK: it wraps
**LP-MusicCaps** and turns a musical audio clip into a natural-language
description of the song. This is the representative *audio* example for the SDK
contract — load a real PyTorch model **once** in `setup()`, then caption many
audio jobs against it over a long-lived loopback HTTP server.

Unlike the sibling [`custom-backend-echo`](../custom-backend-echo/) and
[`custom-backend-text-sentiment`](../custom-backend-text-sentiment/) examples
(which build tiny in-memory models), this backend loads pretrained weights. The
weights are **not** in the repo: `setup()` downloads `transfer.pth`
(~1 GB) from Hugging Face **once**, caches it, and never re-downloads.

## What's here

| File | Purpose |
| --- | --- |
| `backend.py` | `MusicCaptionBackend` — subclasses `CustomBackend`; `task_type="audio2text"`. |
| `lpmc/` | Vendored, inference-only subset of LP-MusicCaps (`BartCaptionModel` + audio helpers). |
| `requirements.txt` | The backend's runtime deps (torch, transformers, librosa, …). The SDK runtime needs none of these. |

The runtime that serves a backend lives in the SDK itself
(`python -m inferencekey.backend.serve`) and depends only on the Python standard
library — **not** on torch, transformers, or any HTTP framework.

## The audio contract

`Job`/`Result` carry free dicts (this is **not** OpenAI-compatible). This
example uses:

* input  — `{"audio_b64": "<base64 of a .wav/.mp3 file>", "num_beams": 5}`
  (`num_beams` optional, default 5).
* output — `{"description": "<joined text>", "chunks": [{"text": ..., "time": "0:00-10:00"}, ...]}`

The audio is resampled to 16 kHz mono and split into 10-second chunks; each
chunk is captioned independently, and `description` is the chunk captions joined
with a space. A missing/invalid `audio_b64` (or undecodable audio) raises inside
`process()` → the runtime returns `500` and the server stays alive.

## Model & weights

The model is LP-MusicCaps' `BartCaptionModel`: a mel-spectrogram audio encoder
feeding a BART decoder. On the first `setup()` the backend downloads

```
https://huggingface.co/seungheondoh/lp-music-caps/resolve/main/transfer.pth
```

to `~/.cache/inferencekey/lp-music-caps/transfer.pth` (override with
`config.weights_dir`) and loads it once with `torch.load(..., map_location="cpu")`
+ `model.load_state_dict(obj["state_dict"])`, then moves the model to
`config.device` (default `"cpu"`; no GPU autodetection, no forced `.cuda()`).

## System dependency: ffmpeg

Decoding mp3 (and most compressed formats) needs **ffmpeg** on the system PATH.
Install it with your OS package manager, e.g.:

```bash
sudo apt-get install -y ffmpeg      # Debian/Ubuntu
# brew install ffmpeg               # macOS
```

WAV files decode without ffmpeg (via `soundfile`/libsndfile).

## Run it end to end (reproducible)

All commands assume you are at the SDK repo root (`external/inferencekey-sdk/`).
The SDK's Python package is pure Python here, so no native build is needed for
this example — we just put it on `PYTHONPATH`.

### 1. Create a venv and install the backend's deps

> **Use CPython 3.9–3.11.** LP-MusicCaps' `BartCaptionModel.generate()` targets
> the `transformers==4.26.1` generation API, and that transformers/torch
> combination only has wheels for 3.9–3.11 (it's the version the upstream
> Dockerfile `pytorch/pytorch:2.1.0` ships). On 3.13 pip can only resolve
> `torch>=2.5` / `transformers>=5`, whose BART generation internals differ and
> raise at inference time — so build/run this backend with a 3.9–3.11
> interpreter. Install a compatible CPU torch **first**, then the rest:

```bash
python3.11 -m venv examples/custom-backend-music-caption/.venv   # 3.9–3.11
. examples/custom-backend-music-caption/.venv/bin/activate
pip install --upgrade pip
# CPU-only torch compatible with transformers 4.26.1:
pip install "torch==2.0.1" "torchaudio==2.0.2" --index-url https://download.pytorch.org/whl/cpu
pip install -r examples/custom-backend-music-caption/requirements.txt
```

> The SDK runtime does **not** import any of these. They are needed only because
> *this example's backend* uses them.

### 2. Start the server

```bash
# Make both the SDK package and this example folder importable:
export PYTHONPATH="$PWD/bindings/python/python:$PWD/examples/custom-backend-music-caption"

python -m inferencekey.backend.serve \
    --port 8099 \
    --backend backend:MusicCaptionBackend \
    --config-json '{"device": "cpu"}'
```

You'll see one `model loaded` line once `setup()` finishes (after the one-time
weights download), then `serving on http://127.0.0.1:8099`.

> The entrypoint and config can also come from the environment:
> `IK_BACKEND_ENTRYPOINT="backend:MusicCaptionBackend"` and
> `IK_BACKEND_CONFIG='{"device":"cpu"}'`. `--port` is always required. To cache
> the weights elsewhere, pass `{"device":"cpu","weights_dir":"/path/to/dir"}`.

#### Or start it from code (the `serve_backend` helper)

```python
# run_music_caption.py  (run with PYTHONPATH set as above)
from inferencekey.backend import serve_backend
from backend import MusicCaptionBackend

serve_backend(MusicCaptionBackend, port=8099, config={"device": "cpu"})
```

### 3. Readiness and metadata

```bash
# 503 {"status":"loading"} until setup() (incl. the weights download) finishes,
# then 200 {"status":"ok"}:
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8099/healthz

curl -s http://127.0.0.1:8099/meta
# {"name": "lp-music-caps", "version": "0.1.0", "task_type": "audio2text", "requirements": "requirements.txt"}
```

### 4. Caption a real audio job (`POST /process`)

Send the base64 of an audio file as `input.audio_b64`. For example, with a short
mp3:

```bash
AUDIO_B64=$(base64 -w0 /path/to/song.mp3)   # macOS: base64 -i /path/to/song.mp3
curl -s -X POST http://127.0.0.1:8099/process \
    -d "{\"id\":\"j1\",\"input\":{\"audio_b64\":\"$AUDIO_B64\"}}"
# {"output": {"description": "...", "chunks": [{"text": "...", "time": "0:00-10:00"}, ...]}}
```

Building the request body from a file without shell-escaping surprises:

```bash
python3 - <<'PY'
import base64, json
b64 = base64.b64encode(open("/path/to/song.mp3", "rb").read()).decode()
print(json.dumps({"id": "j1", "input": {"audio_b64": b64, "num_beams": 5}}))
PY
```

A bad input (missing/invalid `audio_b64`, or undecodable audio) raises inside
`process()` → `500`, and the server stays alive:

```bash
curl -s -w " [%{http_code}]\n" -X POST http://127.0.0.1:8099/process \
    -d '{"id":"bad","input":{}}'
# {"error": "input must carry a base64 string 'audio_b64'"} [500]
```

## Package and publish with the SDK

The packager never imports the backend or its heavy deps; it bundles the code +
`requirements.txt` + a `manifest.json`. The pretrained weights are **not**
included — they are fetched in `setup()` at run time.

```python
from inferencekey.backend import package_backend

pkg = package_backend(
    src="examples/custom-backend-music-caption",          # dir with backend.py + lpmc/
    entrypoint="backend:MusicCaptionBackend",
    name="lp-music-caps",
    version="0.1.0",
    task_type="audio2text",
    requirements="examples/custom-backend-music-caption/requirements.txt",
    out_dir="dist",
    description="LP-MusicCaps: audio -> music description",
)
print(pkg.path)   # dist/lp-music-caps-0.1.0.tar.gz
```

Then publish it to the Manager:

```python
from inferencekey import publish_custom_backend

record = publish_custom_backend(
    tenant_id="<tenant-uuid>",
    package_path="dist/lp-music-caps-0.1.0.tar.gz",
    token="<api-token>",
)
```

Because the weights (~1 GB) are downloaded in `setup()` and cached, they never
bloat the package or the repo.

## Attribution & license

This example wraps **LP-MusicCaps** by SeungHeon Doh, Keunwoo Choi, Jongpil Lee,
and Juhan Nam — model `seungheondoh/lp-music-caps`, code at
<https://github.com/seungheondoh/lp-music-caps>. The `lpmc/` directory is a
vendored, inference-only subset of that project (the `BartCaptionModel`, the
audio encoder, and the `load_audio`/`load_pretrained` helpers), with imports
made self-contained and `load_pretrained` adapted to run on CPU. LP-MusicCaps is
released under the **CC BY-NC 4.0** license; respect its terms (non-commercial
use, attribution) when using this example and the downloaded weights. See the
upstream repository for the full license text and citation.
