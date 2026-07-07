"""A custom **forecast** backend wrapping Salesforce **Moirai 2.0**.

Given the history of a single time series, this backend predicts the next
``horizon`` steps as a median forecast and a set of quantiles. It wraps
``Moirai2Forecast`` over the ``Salesforce/moirai-2.0-R-small`` checkpoint (a
decoder-only, quantile-based universal forecasting model from the ``uni2ts``
library) running on standard ``torch`` — no custom kernels — via GluonTS.

**Scope — univariate.** This example forecasts a single ``target`` series; it
does not wire up covariates. Moirai supports dynamic real features, but the SDK
contract's covariates are left out here for parity with the TimesFM example and
to keep the mapping simple. For covariate-aware forecasting use the Chronos-2
backend, which models past/known-future covariates natively.

Contract shape — this backend implements the SDK's typed **forecast** contract
(:class:`ForecastRequest` in, :class:`ForecastResult` out), *not* the free-dict
``process`` shape. See ``inferencekey.backend`` for the dataclasses.

* input  — ``{"id": "f1", "target": [...], "horizon": 24, "freq": "D",
  "quantile_levels": [0.1, 0.5, 0.9]}`` (only ``id``/``target``/``horizon``
  required; covariates accepted but ignored).
* output — ``{"forecast": {"median": [...], "timestamps": [...],
  "quantiles": {"0.1": [...], "0.9": [...]}}}``.

The model weights are **not** shipped with this example: ``setup()`` downloads
``Salesforce/moirai-2.0-R-small`` from Hugging Face once (cached), then loads the
module a single time. ``forecast()`` builds a fresh GluonTS predictor per request
(``prediction_length`` is fixed at predictor-build time, so it must track the
request's ``horizon``) but reuses the loaded module.

Run it (from the SDK repo root, with this folder importable)::

    python -m inferencekey.backend.serve \\
        --port 8099 \\
        --backend backend:Moirai2Backend \\
        --config-json '{"device": "auto"}'

``process()`` is implemented as a required stub — this backend only serves
``/forecast``. See ``README.md``.

Attribution: Moirai 2.0 by Salesforce AI Research
(``Salesforce/moirai-2.0-R-small``, https://github.com/SalesforceAIResearch/uni2ts).
"""

from __future__ import annotations

import sys
from typing import Any, List, Optional

from inferencekey.backend import (
    BackendContext,
    CustomBackend,
    ForecastRequest,
    ForecastResult,
    Job,
    Result,
    pick_device,
)

#: The Hugging Face checkpoint Moirai 2.0 loads its weights from.
_MODEL_NAME = "Salesforce/moirai-2.0-R-small"
#: The small checkpoint's max sequence length — context is capped at this.
_MAX_CONTEXT = 512


class Moirai2Backend(CustomBackend):
    """Moirai 2.0 forecasting backend (univariate history in, forecast out)."""

    # Declarative metadata exposed via GET /meta.
    name = "moirai2"
    version = "0.1.0"
    task_type = "forecast"
    requirements = "requirements.txt"
    # uni2ts requires CPython 3.10+.
    requires_python = ">=3.10"

    def setup(self, ctx: BackendContext) -> None:
        # Moirai's GluonTS predictor picks its own device ("auto"); we still run
        # pick_device to validate any explicit {"device": ...} and to log the
        # accelerator the node exposes, consistent with the other examples.
        self.device = pick_device(str(ctx.config.get("device", "auto")))
        self.context_length = int(ctx.config.get("context_length", _MAX_CONTEXT))

        # Imported here (not at module top) so the SDK's static packaging path,
        # which must run without the heavy deps, never imports uni2ts/torch.
        from uni2ts.model.moirai2 import Moirai2Module

        self.module = Moirai2Module.from_pretrained(_MODEL_NAME)
        print("model loaded", file=sys.stderr, flush=True)

    def process(self, job: Job) -> Result:
        """Required stub — this backend only serves ``/forecast``.

        ``process`` is abstract on :class:`CustomBackend`, so a forecast-only
        backend must still implement it. We return a clear, non-crashing message
        pointing the caller at the right endpoint.
        """
        return Result(
            output={
                "error": (
                    "moirai2 is a forecast backend: POST /forecast with a "
                    "ForecastRequest body, not /process"
                )
            }
        )

    def forecast(self, request: ForecastRequest) -> ForecastResult:
        """Predict ``request.horizon`` future steps of ``request.target``.

        Builds a GluonTS predictor for this request's ``horizon`` (Moirai fixes
        ``prediction_length`` at predictor-build time), wraps ``target`` in a
        single-series GluonTS dataset, runs it, and reads the median (q0.5, a
        native level → a true median) plus the requested quantiles off the
        returned ``QuantileForecast``. Covariates are ignored (see the module
        docstring).
        """
        import numpy as np
        import pandas as pd
        from gluonts.dataset.pandas import PandasDataset

        from uni2ts.model.moirai2 import Moirai2Forecast

        n = len(request.target)
        model = Moirai2Forecast(
            module=self.module,
            prediction_length=request.horizon,
            context_length=min(self.context_length, n),
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )
        predictor = model.create_predictor(batch_size=1)

        # Wrap the univariate history in a GluonTS dataset. Moirai takes the
        # series *frequency* as a signal, so we honour request.freq (a pandas
        # offset alias like "D"/"H"/"W") when the caller supplies one, defaulting
        # to daily otherwise. The anchor date is arbitrary — only the frequency
        # and the values matter — so a fixed start is fine. If the given freq is
        # not a valid pandas alias, fall back to "D" rather than crashing.
        freq = request.freq or "D"
        try:
            index = pd.period_range(start="2000-01-01", periods=n, freq=freq)
        except (ValueError, TypeError):
            index = pd.period_range(start="2000-01-01", periods=n, freq="D")
        df = pd.DataFrame({"target": np.asarray(request.target, dtype=np.float32)}, index=index)
        ds = PandasDataset(df, target="target")

        forecast = next(iter(predictor.predict(ds)))

        # q0.5 is a native Moirai level, so this is a genuine median.
        median = [float(x) for x in np.asarray(forecast.quantile(0.5))][: request.horizon]
        if len(median) != request.horizon:
            raise ValueError(
                f"moirai2 produced {len(median)} median steps but horizon is "
                f"{request.horizon}"
            )

        quantiles: dict = {}
        for q in request.quantile_levels:
            # QuantileForecast.quantile(q) interpolates between native levels
            # (0.1..0.9) and returns an array of length `horizon`.
            col = [float(x) for x in np.asarray(forecast.quantile(q))][: request.horizon]
            if len(col) == request.horizon:
                quantiles[str(q)] = col

        _, _, ts_fut_iso = _build_timestamps(
            request.timestamps, request.freq, n, request.horizon
        )
        return ForecastResult(
            median=median, quantiles=(quantiles or None), timestamps=ts_fut_iso
        )


def _build_timestamps(
    timestamps: Optional[List[str]],
    freq: Optional[str],
    n: int,
    horizon: int,
):
    """Return ``(ts_hist, ts_fut, ts_fut_iso)`` for the forecast window.

    Only ``ts_fut_iso`` (ISO8601 strings for the horizon, or ``None``) is used by
    this univariate backend; the full triple mirrors the other forecast examples
    so the timestamp logic reads the same. ``None`` when the series has no real
    datetime axis — the SDK contract keeps ``timestamps`` optional.
    """
    if not timestamps and not freq:
        return list(range(n)), list(range(n, n + horizon)), None

    import pandas as pd

    if timestamps:
        ts_hist = pd.to_datetime(timestamps)
        inferred = pd.infer_freq(ts_hist) if len(ts_hist) >= 3 else None
        step = (
            (ts_hist[-1] - ts_hist[-2])
            if inferred is None and len(ts_hist) >= 2
            else None
        )
        if inferred is not None:
            ts_fut = pd.date_range(
                start=ts_hist[-1], periods=horizon + 1, freq=inferred
            )[1:]
        elif step is not None:
            ts_fut = pd.DatetimeIndex(
                [ts_hist[-1] + step * (i + 1) for i in range(horizon)]
            )
        else:  # single history point, no inferable spacing
            ts_fut = pd.DatetimeIndex([ts_hist[-1]] * horizon)
        return ts_hist, ts_fut, [t.isoformat() for t in ts_fut]

    # freq only: anchor the window at "now"; only spacing/continuation matter.
    # A freq alias pandas can't parse (e.g. a deprecated/upper-case one on newer
    # pandas) must not sink the whole forecast — degrade to no output timestamps
    # (the contract keeps them optional) rather than raising.
    try:
        ts_hist = pd.date_range(end=pd.Timestamp.now().normalize(), periods=n, freq=freq)
        ts_fut = pd.date_range(start=ts_hist[-1], periods=horizon + 1, freq=freq)[1:]
    except (ValueError, TypeError):
        return list(range(n)), list(range(n, n + horizon)), None
    return ts_hist, ts_fut, [t.isoformat() for t in ts_fut]
