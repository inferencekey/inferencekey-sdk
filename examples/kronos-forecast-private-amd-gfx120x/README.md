# Kronos OHLCV forecast (private, AMD gfx120x) — forecast candlesticks with a custom backend

A real-world **forecast** custom backend for the InferenceKey SDK: it wraps
**Kronos**, a foundation model for financial **K-line / OHLCV** candlesticks
(open, high, low, close, volume). Given a security's recent OHLCV history it
forecasts the next `horizon` close prices, as a median plus quantiles. It
implements the SDK's typed `ForecastRequest → ForecastResult` contract, loading
the model **once** in `setup()`. The one axis it varies vs the other forecast
examples is the data shape: **multivariate OHLCV** (not a single bare series).

| Path | What it is |
| --- | --- |
| `backend.py` | The `KronosBackend` (`setup`/`forecast`/`process`). |
| `kronos_model/` | **Vendored**, inference subset of [Kronos](https://github.com/shiyu-coder/Kronos) (`Kronos`, `KronosTokenizer`, `KronosPredictor`). MIT — see `kronos_model/LICENSE`. |
| `main.py` | The client: ensure → ready → POST `/forecast` → delete. |

## Mapping the contract to OHLCV
- `target` → the **close** price history (the series we forecast).
- `past_covariates` → the matching `{"open", "high", "low", "volume"}` columns
  (each the same length as `target`). Omit them and the backend forecasts from a
  **flat candle** (open=high=low=close, volume=0) — a bare `target` still works,
  just with less information.
- `median` / `quantiles` → the forecast **close**. Kronos is generative and does
  not emit quantiles directly, so the backend runs it `n_paths` times (default
  20, configurable) and takes per-step percentiles across the sample paths. More
  paths → smoother quantiles but a slower call.

## Compatibility
- SDK: local build in this repo (or >= 0.1.0 once published)
- Language: Python >= 3.9 (the `main.py` client). The **backend** needs Python
  >= 3.10 — declared via `requires_python=">=3.10"`, so the worker provisions a
  matching interpreter.
- Placement: private (AMD R9700 / gfx120x)
- Backend: custom (`kronos`), forecast   Policy: fixed

## Prerequisites
- The tokens/ids this example needs (see [tokens & placement](../README.md#tokens)):
  an SDK token (`ik_sdk_`), a data-plane API key (`ik_live_`), your project slug,
  and your registered worker's id (`IK_WORKER_ID`, `wrk_…`).
- Python >= 3.9 for the client. A registered **private worker** on an AMD R9700
  (gfx120x) box — the worker installs a ROCm `torch` into the backend's venv.
- The `kronos` backend must be **published** to your tenant so
  `WorkloadSpec(backend="kronos")` resolves. Package this folder — **including
  the `kronos_model/` directory** — and publish it with `package_backend` /
  `publish_custom_backend` (see the SDK docs and the
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
custom OHLCV forecast backend:

1. **`ensure()`** reconciles a private workload pinned to `IK_WORKER_ID`, with
   `backend="kronos"` (the published custom backend's *name*) and
   `task_type="forecast"`.
2. **`wait_until_ready()`** blocks while the cold worker builds the backend's
   venv (ROCm `torch` + Kronos's deps), downloads the weights, and loads the
   model once.
3. **Call** — `main.py` POSTs a `ForecastRequest` (close as `target`, the rest
   of OHLCV as `past_covariates`) directly to
   `…/endpoint/{project}/{slug}/forecast` with the data-plane key (`ik_live_`),
   then prints the forecast close, quantiles, and timestamps.

## Cost & cleanup
The workload provisions **one fixed replica** on your private GPU. Because the
policy is `fixed`, that replica — and the GPU it reserves — stays up (and
billing) **until you delete it**; a `fixed` private workload does *not* scale to
zero. `main.py` calls `mgmt.delete(SLUG)` in a `finally` block, so it tears down
on success, on error, and on Ctrl-C. If you kill it harder (SIGKILL) and the
workload is left behind, delete it by slug — re-run the script (its `finally`
deletes) or call `ManagementClient.from_env().delete("kronos-forecast")`.

## Troubleshooting
- **`ensure` fails: unknown backend `kronos`.** The custom backend isn't
  published to your tenant. Package this folder (with `kronos_model/`) and
  publish it first (see Prerequisites), then re-run.
- **Import error `No module named 'kronos_model'`.** The `kronos_model/`
  directory wasn't included in the published package. Re-package the whole
  folder, not just `backend.py`.
- **`wait_until_ready` times out.** First boot is slow: the worker installs
  ROCm `torch` + Kronos's deps and downloads the weights. The timeout is 1800 s
  here — raise it on a slow link, and check the worker logs in the Manager UI.
- **Forecast is slow.** Quantiles come from `n_paths` model runs (default 20).
  Lower `n_paths` via the backend config for a faster, coarser forecast; raise
  it for smoother quantiles.
- **Backend crashes at load (host RAM).** The R9700 has plenty of VRAM, but a
  low-RAM host can OOM while loading the model. Add swap on the worker host.
- **`403`/`401` on the POST.** The `/forecast` call uses the **data-plane** key
  (`ik_live_`, `INFERENCEKEY_API_KEY`), scoped to this workload — not the SDK
  token (`ik_sdk_`) that provisions it.

## Attribution & license
Kronos is by Shiyu Chen et al. — <https://github.com/shiyu-coder/Kronos>. The
`kronos_model/` directory is a vendored, inference-only subset of that project,
redistributed here under its MIT license (`kronos_model/LICENSE`). The model
weights (`NeoQuasar/Kronos-*` on Hugging Face) are downloaded at runtime and are
not shipped with this example.

---
See [CONTRIBUTING.md](../CONTRIBUTING.md) and the
[tokens & placement overview](../README.md#tokens).
