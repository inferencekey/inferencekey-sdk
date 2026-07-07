# Chronos-2 forecast (private, AMD gfx120x) — forecast a time series with a custom backend

A real-world **forecast** custom backend for the InferenceKey SDK: it wraps
Amazon **Chronos-2** (`amazon/chronos-2`) and predicts the next `horizon` steps
of a time series as a median point forecast plus quantiles. It is the
representative *forecast* example — it implements the SDK's typed
`ForecastRequest → ForecastResult` contract (not the free-dict `process` shape),
loading the model **once** in `setup()` and reusing it for every request. The
one axis it varies vs the other examples is the task: **forecast** on a
**custom** backend, pinned to a private AMD box.

## Compatibility
- SDK: local build in this repo (or >= 0.1.0 once published)
- Language: Python >= 3.9 (the `main.py` client). The **backend** needs Python
  >= 3.10 — `chronos-forecasting` requires it — declared via
  `requires_python=">=3.10"`, so the worker provisions a matching interpreter.
- Placement: private (AMD R9700 / gfx120x)
- Backend: custom (`chronos-2`), forecast   Policy: fixed

## Prerequisites
- The tokens/ids this example needs (see [tokens & placement](../README.md#tokens)):
  an SDK token (`ik_sdk_`), a data-plane API key (`ik_live_`), your project slug,
  and your registered worker's id (`IK_WORKER_ID`).
- Python >= 3.9 for the client. A registered **private worker** on an AMD R9700
  (gfx120x) box — the worker installs a ROCm `torch` into the backend's venv.
- The `chronos-2` backend must be **published** to your tenant so
  `WorkloadSpec(backend="chronos-2")` resolves. Package this folder and publish
  it with `package_backend` / `publish_custom_backend` (see the SDK docs and the
  sibling [`custom-backend-music-caption`](../custom-backend-music-caption/#package-and-publish-with-the-sdk)
  README for the exact calls) before running `main.py`.

## Run
```bash
cp .env.example .env        # fill in real values
# install the local SDK — see ../CONTRIBUTING.md#4-depending-on-the-sdk
python main.py
```

## What it does
The three steps — `ensure()` → wait until ready → call → `delete()` — for a
custom forecast backend:

1. **`ensure()`** reconciles a private workload pinned to `IK_WORKER_ID`, with
   `backend="chronos-2"` (the published custom backend's *name*, not a `Backend`
   enum) and `task_type="forecast"`.
2. **`wait_until_ready()`** blocks while the cold worker builds the backend's
   venv (ROCm `torch` + `chronos-forecasting`), downloads the ~500 MB weights,
   and loads the model once.
3. **Call** — the SDK has no typed forecast helper yet, so `main.py` POSTs a
   `ForecastRequest` JSON directly to
   `…/endpoint/{project}/{slug}/forecast` with the data-plane key (`ik_live_`),
   then prints the returned `median`, `quantiles`, and `timestamps`.

## Cost & cleanup
The workload provisions **one fixed replica** on your private GPU. Because the
policy is `fixed`, that replica — and the GPU it reserves — stays up (and
billing) **until you delete it**; a `fixed` private workload does *not* scale to
zero. Chronos-2 is small: VRAM use is well under 1 GB on the R9700's 32 GB, so
the GPU is barely occupied — the cost is the *reservation*, not the memory.
`main.py` calls `mgmt.delete(SLUG)` in a `finally` block, so it tears down on
success, on error, and on Ctrl-C. If you kill it harder (SIGKILL) and the
workload is left behind, delete it by slug — re-run the script (its `finally`
deletes) or call `ManagementClient.from_env().delete("chronos2-forecast")`.

## Troubleshooting
- **`ensure` fails: unknown backend `chronos-2`.** The custom backend isn't
  published to your tenant. Package this folder and publish it first (see
  Prerequisites), then re-run.
- **`wait_until_ready` times out.** First boot is slow: the worker installs
  ROCm `torch` + `chronos-forecasting` and downloads the weights. The timeout is
  1800 s here — raise it on a slow link, and check the worker logs in the
  Manager UI for a pip/download error.
- **Backend crashes at load (host RAM).** The R9700 has plenty of VRAM, but a
  low-RAM host can OOM while loading the model into memory before moving it to
  the GPU. Add swap on the worker host if you hit this.
- **`403`/`401` on the POST.** The `/forecast` call uses the **data-plane** key
  (`ik_live_`, `INFERENCEKEY_API_KEY`), scoped to this workload — not the SDK
  token (`ik_sdk_`) that provisions it. Mint/scope the `ik_live_` key correctly.

---
See [CONTRIBUTING.md](../CONTRIBUTING.md) and the
[tokens & placement overview](../README.md#tokens).
