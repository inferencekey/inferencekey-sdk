"""Forecast an OHLCV series with a **Kronos** custom backend on a private GPU.

The canonical SDK shape — ensure() -> wait_until_ready() -> call -> delete() —
varying one axis vs the other examples: the backend is a **custom** backend
(``kronos``, the one this folder publishes) and the task is **forecast** over
financial candlesticks (OHLCV). The data plane has no typed forecast helper yet,
so — exactly like the audio example POSTs to ``/v1/audio/transcriptions`` — this
script POSTs a ForecastRequest JSON to the workload's ``/forecast`` endpoint with
the data-plane key (``ik_live_``) and prints the forecast close it gets back.

Kronos forecasts a candlestick series: ``target`` is the **close** history and
``past_covariates`` carry the matching open/high/low/volume. (Send just ``target``
and the backend forecasts from a flat candle — see the README.)

Placement is private: the workload is pinned to YOUR registered worker via
``worker_id``. This example targets an **AMD R9700 (gfx120x)** box — the worker
installs a ROCm ``torch`` for the backend's venv; see the README.

For a custom backend, ``WorkloadSpec.backend`` is the **name of the published
custom backend** (``"kronos"``), not one of the built-in ``Backend`` enums, and
``task_type`` is the raw string ``"forecast"``. Publish this backend first (see
the README) so that name resolves.

Run:  cp .env.example .env  # fill in real values, then
      python main.py
"""

from __future__ import annotations

import os
import sys

import requests

from inferencekey import (
    ManagementClient,
    WorkloadSpec,
    ExecutionPolicy,
    InferenceKeyError,
)

# --- The model / backend ---------------------------------------------------
# Kronos (NeoQuasar/Kronos-small): a foundation model for OHLCV candlesticks.
# The Kronos code is vendored into the backend (kronos_model/); only the weights
# are pulled from Hugging Face at cold start.
MODEL = "NeoQuasar/Kronos-small"

# The custom backend's published name (backend.py sets `name = "kronos"`).
# For a custom backend this is the `backend` field — NOT a Backend enum.
BACKEND_NAME = "kronos"

SLUG = "kronos-forecast"

# A tiny synthetic OHLCV history to forecast once the workload is ready. `close`
# is the target; open/high/low/volume ride along as past covariates. Point these
# at your own candlesticks (e.g. a stock's daily bars).
CLOSE = [
    10.0, 11.0, 12.5, 11.8, 13.0, 14.2, 13.5, 15.0,
    16.1, 15.4, 17.0, 18.3, 17.6, 19.0, 20.2, 19.5,
]
OPEN = [c - 0.3 for c in CLOSE]
HIGH = [c + 0.5 for c in CLOSE]
LOW = [c - 0.6 for c in CLOSE]
VOLUME = [1000.0 + 25 * i for i in range(len(CLOSE))]
HORIZON = 6
QUANTILE_LEVELS = [0.1, 0.5, 0.9]


def main() -> int:
    # Control plane (ik_sdk_): reads INFERENCEKEY_SDK_TOKEN / _PROJECT / _BASE_URL.
    mgmt = ManagementClient.from_env()

    # The worker id comes from the environment — never hard-code it. It's the
    # id of your registered worker (`wrk_…`; Manager UI -> Workers -> copy it).
    worker_id = os.environ["IK_WORKER_ID"]

    spec = WorkloadSpec(
        name="Kronos OHLCV forecast",
        slug=SLUG,
        model=MODEL,
        backend=BACKEND_NAME,        # the published custom backend's name
        task_type="forecast",        # raw string; the forecast task type
        worker_id=worker_id,
        # Private + fixed: one always-on replica. The GPU stays reserved (and
        # billing) until delete() runs — see the README's Cost & cleanup.
        execution_policy=ExecutionPolicy.FIXED,
        execution_policy_config={"replicas": 1},
    )

    try:
        # 1) Provision / reconcile. Idempotent by slug.
        ref = mgmt.ensure(spec)
        print(f"ensured {ref.project_slug}/{ref.workload_slug}")

        # 2) Wait for the cold worker to build the backend's venv (ROCm torch +
        #    Kronos deps), download the weights, and load the model. The first
        #    boot is slow, so give it a generous timeout.
        print("waiting for the worker to become ready (cold start can take a while)...")
        mgmt.wait_until_ready(ref.workload_slug, timeout=1800)

        # 3) Forecast via the workload's /forecast endpoint. The SDK has no typed
        #    forecast helper yet, so we POST the ForecastRequest JSON directly
        #    with the data-plane key (ik_live_). `target` is the close; the other
        #    OHLCV columns ride along as past_covariates.
        base = os.environ.get("INFERENCEKEY_BASE_URL", "https://cloud.inferencekey.com")
        project = os.environ["INFERENCEKEY_PROJECT"]
        api_key = os.environ["INFERENCEKEY_API_KEY"]
        url = f"{base}/endpoint/{project}/{ref.workload_slug}/forecast"

        body = {
            "id": "f1",
            "target": CLOSE,
            "horizon": HORIZON,
            "quantile_levels": QUANTILE_LEVELS,
            "past_covariates": {
                "open": OPEN,
                "high": HIGH,
                "low": LOW,
                "volume": VOLUME,
            },
        }
        resp = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=body,
            timeout=180,
        )
        resp.raise_for_status()
        out = resp.json()
        forecast = out.get("forecast", out)
        print(f"\nforecast close ({HORIZON} steps): {forecast.get('median')}")
        if forecast.get("quantiles"):
            print(f"quantiles: {forecast['quantiles']}")
        if forecast.get("timestamps"):
            print(f"timestamps: {forecast['timestamps']}")
        print()
    except InferenceKeyError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except requests.HTTPError as e:
        print(f"forecast request failed: {e} — {e.response.text[:200]}", file=sys.stderr)
        return 1
    finally:
        # 4) Tear down so the reserved GPU stops billing. Idempotent and safe —
        #    runs on success, error, and Ctrl-C.
        deleted = mgmt.delete(SLUG)
        print(f"deleted={deleted}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
