# custom-backend-text-sentiment

A self-contained **text** custom backend for the InferenceKey SDK: a tiny
sentiment classifier. It shows the T01 contract for the product's dominant case
(**text**) — load a minuscule PyTorch model **once**, then classify many text
jobs against it over a long-lived loopback HTTP server.

The model is a tiny `nn.Embedding` + `nn.Linear` built **in memory** over a
small fixed vocabulary — no weights are downloaded. A few obvious positive and
negative words steer the prediction, so the example is fully reproducible
offline while still running a real `torch.nn.Module` forward pass.

> See the sibling [`custom-backend-echo`](../custom-backend-echo/) for the
> numeric-vector variant. This one is the **text** representative.

## What's here

| File | Purpose |
| --- | --- |
| `backend.py` | `SentimentBackend` — subclasses `CustomBackend`; declares `/meta` metadata (`task_type="classification"`). |
| `requirements.txt` | The backend's runtime deps (`torch`). The SDK runtime needs none of these. |

## The text contract

`Job`/`Result` carry free dicts; this example uses the agreed **text** mapping:

* input  — `{"text": "..."}` (also accepts `{"prompt": "..."}`)
* output — `{"label": "positive"|"negative", "score": float}`

A text-generation backend (`text2text`) would instead read
`{"messages":[...]}` or `{"prompt":"..."}` and return `{"text": "..."}` — see
the `Job` docstring in `base.py` for the full mapping.

## Run it end to end (reproducible)

All commands assume you are at the SDK repo root (`external/inferencekey-sdk/`).

### 1. Create a venv and install torch (the backend's dep)

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 2. Start the server

```bash
export PYTHONPATH="$PWD/bindings/python/python:$PWD/examples/custom-backend-text-sentiment"

python -m inferencekey.backend.serve \
    --port 8099 \
    --backend backend:SentimentBackend \
    --config-json '{"device": "cpu"}'
```

You'll see exactly one `model loaded` line, then `serving on http://127.0.0.1:8099`.

#### Or start it from code (the `serve_backend` helper)

Instead of the CLI you can boot the same server from a script or REPL:

```python
# run_sentiment.py  (run with PYTHONPATH set as above)
from inferencekey.backend import serve_backend
from backend import SentimentBackend

# Pass a class (instantiated with no args) or an instance. Blocks until SIGTERM.
serve_backend(SentimentBackend, port=8099, config={"device": "cpu"})
```

### 3. Inspect the backend metadata (`GET /meta`)

```bash
curl -s http://127.0.0.1:8099/meta
# {"name": "tiny-sentiment", "version": "0.1.0", "task_type": "classification", "requirements": "requirements.txt"}
```

### 4. Classify a real text job (`POST /process`)

```bash
curl -s -X POST http://127.0.0.1:8099/process \
    -d '{"id":"j1","input":{"text":"this is great i love it"}}'
# {"output": {"label": "positive", "score": ...}}

curl -s -X POST http://127.0.0.1:8099/process \
    -d '{"id":"j2","input":{"prompt":"awful terrible hate"}}'
# {"output": {"label": "negative", "score": ...}}
```

A bad input (missing `text`/`prompt`) raises inside `process()` → `500` and the
server stays alive:

```bash
curl -s -w " [%{http_code}]\n" -X POST http://127.0.0.1:8099/process \
    -d '{"id":"bad","input":{}}'
# {"error": "input must carry a string 'text' or 'prompt'"} [500]
```
