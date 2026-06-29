# custom-backend-echo

A minimal, self-contained **custom backend** for the InferenceKey SDK. It shows
the T01 contract end to end: load a tiny PyTorch model **once**, then process
many jobs against it over a long-lived loopback HTTP server.

The model is an identity-initialized `nn.Linear` built **in memory** — no
weights are downloaded — so it simply echoes the input vector back. That keeps
the example fully reproducible offline while still exercising a real
`torch.nn.Module` forward pass.

## What's here

| File | Purpose |
| --- | --- |
| `backend.py` | `EchoLinearBackend` — the custom backend (subclasses `CustomBackend`). |
| `failing_backend.py` | `FailingSetupBackend` — raises in `setup()`, for the failure demo. |
| `requirements.txt` | The backend's runtime deps (`torch`). The SDK runtime needs none of these. |

The runtime that serves a backend lives in the SDK itself
(`python -m inferencekey.backend.serve`) and depends only on the Python standard
library — **not** on torch or any HTTP framework.

## The contract in 30 seconds

```python
from inferencekey.backend import BackendContext, CustomBackend, Job, Result

class EchoLinearBackend(CustomBackend):
    def setup(self, ctx: BackendContext) -> None:
        ...   # called once; instantiate the nn.Module here, store it on self

    def process(self, job: Job) -> Result:
        ...   # called per job; reuse the model loaded in setup()
```

* `Job` — `{"id": str, "input": dict}`.
* `Result` — wraps a free `{"output": dict}` (this is **not** OpenAI-compatible).
* `BackendContext` — `config: dict` (resolve `device`/`model_name`/weights here;
  device defaults to `"cpu"`, no GPU autodetection) and `port: int`.

## HTTP schema (served by the runtime)

* `GET /healthz` → `503 {"status":"loading"}` until `setup()` finishes, then
  `200 {"status":"ok"}`.
* `GET /meta` → `200` with the backend's declarative metadata
  (`name`/`version`/`task_type`/`requirements`); available regardless of
  readiness. A backend declares these as class attributes (or by overriding
  `manifest()`); unset fields default to empty, with `name` falling back to the
  class name.
* `POST /process` → body `{"id":..., "input":{...}}` → `200 {"output":{...}}`.
  Bad body → `400`. A raising `process()` → `500 {"error":"..."}` and the
  server stays alive for the next job.

The server binds `127.0.0.1` only (loopback), never `0.0.0.0`.

## Starting from code (`serve_backend`)

Besides `python -m inferencekey.backend.serve`, you can boot the same server
from a script or REPL with the `serve_backend` helper (it reuses the internal
`serve()`):

```python
from inferencekey.backend import serve_backend
from backend import EchoLinearBackend

# Pass a class (instantiated with no args) or an instance. Blocks until SIGTERM;
# pass block=False to get the running server back for tests/REPL and shut it
# down yourself (httpd.shutdown(); httpd.server_close()).
serve_backend(EchoLinearBackend, port=8099, config={"device": "cpu", "size": 4})
```

---

## Run it end to end (reproducible)

All commands assume you are at the SDK repo root
(`external/inferencekey-sdk/`). The SDK's Python package is pure Python here, so
no native build is needed for this example — we just put it on `PYTHONPATH`.

### 1. Create a venv and install torch (the backend's dep)

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
# CPU-only torch is enough — no GPU, no weights downloaded:
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

> The SDK runtime does **not** import torch. Torch is needed only because *this
> example's backend* uses it.

### 2. Start the server

```bash
# Make both the SDK package and this example folder importable:
export PYTHONPATH="$PWD/bindings/python/python:$PWD/examples/custom-backend-echo"

python -m inferencekey.backend.serve \
    --port 8099 \
    --backend backend:EchoLinearBackend \
    --config-json '{"device": "cpu", "size": 4}'
```

You'll see exactly one `model loaded` line once `setup()` finishes, then
`serving on http://127.0.0.1:8099`.

> The entrypoint and config can also come from the environment instead of CLI
> flags: `IK_BACKEND_ENTRYPOINT="backend:EchoLinearBackend"` and
> `IK_BACKEND_CONFIG='{"device":"cpu","size":4}'`. `--port` is always required.

### 3. Exercise the acceptance criteria (in another terminal)

**Readiness (503 before load, 200 after):**

```bash
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8099/healthz   # 200 once loaded
```

If you query during the (very brief) load window you'll get `503`. To see it
deterministically, use a backend whose `setup()` is slow — the SDK test suite
does exactly this (`bindings/python/tests/test_backend_serve.py`).

**Backend metadata (`GET /meta`):**

```bash
curl -s http://127.0.0.1:8099/meta
# {"name": "EchoLinearBackend", "version": "", "task_type": "", "requirements": ""}
```

`EchoLinearBackend` declares no metadata, so `/meta` falls back to the class name
with empty fields — it never breaks. See the
[`custom-backend-text-sentiment`](../custom-backend-text-sentiment) example for a
backend that declares `name`/`version`/`task_type`.

**Process a job (200 + echoed vector):**

```bash
curl -s -X POST http://127.0.0.1:8099/process \
    -d '{"id":"j1","input":{"values":[1,2,3,4]}}'
# {"output": {"values": [1.0, 2.0, 3.0, 4.0], "device": "cpu"}}
```

**Model loaded once across N≥3 jobs** — send three jobs, then check the server's
stdout: `model loaded` appears exactly once.

```bash
for i in 1 2 3; do
  curl -s -X POST http://127.0.0.1:8099/process \
      -d "{\"id\":\"j$i\",\"input\":{\"values\":[$i,$i,$i,$i]}}"; echo
done
```

**Resilience (500 then a later job still works):**

```bash
# Forces a ValueError inside process() -> 500, server stays alive:
curl -s -w " [%{http_code}]\n" -X POST http://127.0.0.1:8099/process \
    -d '{"id":"bad","input":{"values":"nope"}}'
# {"error": "input.values must be a list of numbers"} [500]

# A valid job afterwards still returns 200:
curl -s -w " [%{http_code}]\n" -X POST http://127.0.0.1:8099/process \
    -d '{"id":"after","input":{"values":[1,1,1,1]}}'
# {"output": {"values": [1.0, 1.0, 1.0, 1.0], "device": "cpu"}} [200]
```

### 4. Setup failure (process exits non-zero, never ready)

```bash
python -m inferencekey.backend.serve \
    --port 8100 \
    --backend failing_backend:FailingSetupBackend
echo "exit code: $?"   # 1 — and /healthz on :8100 never reaches 200
```

The traceback is printed to stderr, readiness never flips, and the process exits
with a non-zero status the worker supervisor can act on.

---

## Run the SDK test suite (no torch required)

The contract's unit tests and the runtime's integration tests use torch-free
fake backends, so they run without torch installed:

```bash
pip install pytest
PYTHONPATH="$PWD/bindings/python/python" pytest bindings/python/tests -q
```
