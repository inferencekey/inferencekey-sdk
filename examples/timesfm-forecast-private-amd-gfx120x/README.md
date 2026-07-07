# TimesFM forecast (private, AMD gfx120x) — forecast a time series with a custom backend

A real-world **forecast** custom backend for the InferenceKey SDK: it wraps
Google **TimesFM 2.5** (`google/timesfm-2.5-200m-pytorch`) and predicts the next
`horizon` steps of a time series as a median point forecast plus quantiles. It
implements the SDK's typed `ForecastRequest → ForecastResult` contract (not the
free-dict `process` shape), loading and compiling the model **once** in `setup()`
and reusing it for every request. The one axis it varies vs the other examples is
the task: **forecast** on a **custom** backend, pinned to a private AMD box.

**Univariate — no covariates.** TimesFM 2.5's basic API forecasts from the series
history alone; it does not consume `past_covariates`/`future_covariates`. This
backend **ignores** them if sent (it does not error). For a covariate-aware
forecast (a stock plus VIX / volume / calendar), use the sibling
[`chronos2-forecast-private-amd-gfx120x`](../chronos2-forecast-private-amd-gfx120x)
example, which models covariates natively.

## Compatibility
- SDK: local build in this repo (or >= 0.1.0 once published)
- Language: Python >= 3.9 (the `main.py` client). The **backend** needs Python
  >= 3.10 — declared via `requires_python=">=3.10"`, so the worker provisions a
  matching interpreter.
- Placement: private (AMD R9700 / gfx120x)
- Backend: custom (`timesfm`), forecast   Policy: fixed

## Prerequisites
- The tokens/ids this example needs (see [tokens & placement](../README.md#tokens)):
  an SDK token (`ik_sdk_`), a data-plane API key (`ik_live_`), your project slug,
  and your registered worker's id (`IK_WORKER_ID`, `wrk_…`).
- Python >= 3.9 for the client. A registered **private worker** on an AMD R9700
  (gfx120x) box — the worker installs a ROCm `torch` into the backend's venv.
- The `timesfm` backend must be **published** to your tenant so
  `WorkloadSpec(backend="timesfm")` resolves. Package this folder and publish it
  with `package_backend` / `publish_custom_backend` (see the SDK docs and the
  [`custom-backend-music-caption`](../custom-backend-music-caption/#package-and-publish-with-the-sdk)
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
   `backend="timesfm"` (the published custom backend's *name*, not a `Backend`
   enum) and `task_type="forecast"`.
2. **`wait_until_ready()`** blocks while the cold worker builds the backend's
   venv (ROCm `torch` + `timesfm`), downloads the weights, and loads + compiles
   the model once.
3. **Call** — the SDK has no typed forecast helper yet, so `main.py` POSTs a
   `ForecastRequest` JSON directly to `…/endpoint/{project}/{slug}/forecast`
   with the data-plane key (`ik_live_`), then prints the returned `median`,
   `quantiles`, and `timestamps`.

TimesFM's forecast returns a mean plus a fixed grid of quantiles (0.1, 0.2, …,
0.9). This backend reports `median` from the **q0.5** column (a true median, not
the mean), and maps each level you request in `quantile_levels` onto the
**nearest** of TimesFM's own levels. If you ask for a level TimesFM doesn't
carry (e.g. `0.05` or `0.95`), you get the nearest one it does (`0.1` / `0.9`)
— returned under the key you asked for, so mind the rounding when you request
levels outside `{0.1 … 0.9}`.

## Cost & cleanup
The workload provisions **one fixed replica** on your private GPU. Because the
policy is `fixed`, that replica — and the GPU it reserves — stays up (and
billing) **until you delete it**; a `fixed` private workload does *not* scale to
zero. TimesFM is small: VRAM use is a small fraction of the R9700's 32 GB, so
the cost is the *reservation*, not the memory. `main.py` calls `mgmt.delete(SLUG)`
in a `finally` block, so it tears down on success, on error, and on Ctrl-C. If
you kill it harder (SIGKILL) and the workload is left behind, delete it by slug —
re-run the script (its `finally` deletes) or call
`ManagementClient.from_env().delete("timesfm-forecast")`.

## Troubleshooting
- **`ensure` fails: unknown backend `timesfm`.** The custom backend isn't
  published to your tenant. Package this folder and publish it first (see
  Prerequisites), then re-run.
- **`wait_until_ready` times out.** First boot is slow: the worker installs
  ROCm `torch` + `timesfm` and downloads the weights. The timeout is 1800 s
  here — raise it on a slow link, and check the worker logs in the Manager UI
  for a pip/download error.
- **Backend crashes at load (host RAM).** The R9700 has plenty of VRAM, but a
  low-RAM host can OOM while loading the model into memory before moving it to
  the GPU. Add swap on the worker host if you hit this.
- **`horizon` too large.** The backend compiles TimesFM with a `max_horizon`
  (512 by default); a request with a larger horizon is rejected with a clear
  error. Raise it via the backend config (`max_horizon`) if you need longer
  forecasts.
- **`403`/`401` on the POST.** The `/forecast` call uses the **data-plane** key
  (`ik_live_`, `INFERENCEKEY_API_KEY`), scoped to this workload — not the SDK
  token (`ik_sdk_`) that provisions it. Mint/scope the `ik_live_` key correctly.

---
See [CONTRIBUTING.md](../CONTRIBUTING.md) and the
[tokens & placement overview](../README.md#tokens).
