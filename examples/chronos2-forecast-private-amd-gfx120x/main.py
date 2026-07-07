"""Forecast a time series with a **Chronos-2** custom backend on a private GPU.

The canonical SDK shape — ensure() -> wait_until_ready() -> call -> delete() —
varying one axis vs the other examples: the backend is a **custom** backend
(``chronos-2``, the one this folder publishes) and the task is **forecast**. The
data plane has no typed forecast helper yet, so — exactly like the audio example
POSTs to ``/v1/audio/transcriptions`` — this script POSTs a ForecastRequest JSON
to the workload's ``/forecast`` endpoint with the data-plane key (``ik_live_``)
and prints the median + quantiles it gets back.

Placement is private: the workload is pinned to YOUR registered worker via
``worker_id``. This example targets an **AMD R9700 (gfx120x)** box — the worker
installs a ROCm ``torch`` for the backend's venv; see the README.

For a custom backend, ``WorkloadSpec.backend`` is the **name of the published
custom backend** (``"chronos-2"``), not one of the built-in ``Backend`` enums,
and ``task_type`` is the raw string ``"forecast"``. Publish this backend first
(see the README) so that name resolves.

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
# amazon/chronos-2: a ~120M-parameter zero-shot time-series foundation model
# (~500 MB of weights). It fits with room to spare on the R9700's 32 GB (VRAM
# use is well under 1 GB); the real constraint is host RAM at load — see README.
MODEL = "amazon/chronos-2"

# The custom backend's published name (backend.py sets `name = "chronos-2"`).
# For a custom backend this is the `backend` field — NOT a Backend enum.
BACKEND_NAME = "chronos-2"

SLUG = "chronos2-forecast"

# A tiny synthetic history to forecast once the workload is ready: a rising line
# with mild seasonality. Point this at your own series (a stock's closes, etc.).
HISTORY = [
    10.0, 11.0, 12.5, 11.8, 13.0, 14.2, 13.5, 15.0,
    16.1, 15.4, 17.0, 18.3, 17.6, 19.0, 20.2, 19.5,
]
HORIZON = 6
QUANTILE_LEVELS = [0.1, 0.5, 0.9]


def main() -> int:
    # Control plane (ik_sdk_): reads INFERENCEKEY_SDK_TOKEN / _PROJECT / _BASE_URL.
    mgmt = ManagementClient.from_env()

    # The worker id comes from the environment — never hard-code it. It's the
    # id of your registered worker (`wrk_…`; Manager UI -> Workers -> copy it).
    worker_id = os.environ["IK_WORKER_ID"]

    spec = WorkloadSpec(
        name="Chronos-2 forecast",
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
        #    chronos-forecasting), download the weights, and load the model. The
        #    first boot is slow, so give it a generous timeout.
        print("waiting for the worker to become ready (cold start can take a while)...")
        mgmt.wait_until_ready(ref.workload_slug, timeout=1800)

        # 3) Forecast via the workload's /forecast endpoint. The SDK has no typed
        #    forecast helper yet, so we POST the ForecastRequest JSON directly
        #    with the data-plane key (ik_live_).
        base = os.environ.get("INFERENCEKEY_BASE_URL", "https://cloud.inferencekey.com")
        project = os.environ["INFERENCEKEY_PROJECT"]
        api_key = os.environ["INFERENCEKEY_API_KEY"]
        url = f"{base}/endpoint/{project}/{ref.workload_slug}/forecast"

        body = {
            "id": "f1",
            "target": HISTORY,
            "horizon": HORIZON,
            "quantile_levels": QUANTILE_LEVELS,
        }
        resp = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=body,
            timeout=120,
        )
        resp.raise_for_status()
        out = resp.json()
        forecast = out.get("forecast", out)
        print(f"\nmedian ({HORIZON} steps): {forecast.get('median')}")
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
