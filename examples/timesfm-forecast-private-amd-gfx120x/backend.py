"""A custom **forecast** backend wrapping Google **TimesFM 2.5**.

Given the history of a single time series, this backend predicts the next
``horizon`` steps as a median point forecast and a set of quantiles. It wraps
``timesfm.TimesFM_2p5_200M_torch`` (``google/timesfm-2.5-200m-pytorch``): a ~200M
parameter, zero-shot decoder-only foundation model for time-series forecasting
that runs on a recent GPU (NVIDIA CUDA or AMD ROCm) or on CPU, on standard
``torch`` — no custom kernels.

**Scope — univariate, no covariates.** TimesFM 2.5's basic forecast API takes a
plain series and returns a point forecast plus a fixed set of quantiles; it does
*not* consume past/future covariates. This backend therefore **ignores**
``past_covariates``/``future_covariates`` if a caller sends them (it does not
error — it just forecasts from ``target`` alone). If you need covariates (a stock
plus VIX/volume/calendar), use the Chronos-2 backend, which models them natively.

Contract shape — this backend implements the SDK's typed **forecast** contract
(:class:`ForecastRequest` in, :class:`ForecastResult` out), *not* the free-dict
``process`` shape. See ``inferencekey.backend`` for the dataclasses.

* input  — ``{"id": "f1", "target": [...], "horizon": 24, "freq": "D",
  "quantile_levels": [0.1, 0.5, 0.9]}`` (only ``id``/``target``/``horizon``
  required; covariates accepted but ignored).
* output — ``{"forecast": {"median": [...], "timestamps": [...],
  "quantiles": {"0.1": [...], "0.9": [...]}}}``.

The model weights are **not** shipped with this example: ``setup()`` downloads
``google/timesfm-2.5-200m-pytorch`` from Hugging Face once (cached), then loads
and compiles the model a single time. ``forecast()`` reuses it for every request.

Run it (from the SDK repo root, with this folder importable)::

    python -m inferencekey.backend.serve \\
        --port 8099 \\
        --backend backend:TimesFMBackend \\
        --config-json '{"device": "auto"}'

``process()`` is implemented as a required stub — this backend only serves
``/forecast``. See ``README.md``.

Attribution: TimesFM by Google Research (``google/timesfm-2.5-200m-pytorch``,
https://github.com/google-research/timesfm).
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

#: The Hugging Face model id TimesFM 2.5 loads its weights from.
_MODEL_NAME = "google/timesfm-2.5-200m-pytorch"
#: TimesFM's ``forecast`` returns a fixed quantile grid alongside the mean. On
#: the quantile axis, index 0 is the **mean** and indices 1..9 are these nine
#: deciles (so the axis has ``1 + 9 = 10`` columns). We map each requested level
#: to the nearest of these — and read the median (q0.5) from index 5.
_TIMESFM_QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
#: Position of the 0.5 (median) level within _TIMESFM_QUANTILE_LEVELS.
_MEDIAN_LEVEL_INDEX = _TIMESFM_QUANTILE_LEVELS.index(0.5)


class TimesFMBackend(CustomBackend):
    """TimesFM 2.5 forecasting backend (univariate history in, forecast out)."""

    # Declarative metadata exposed via GET /meta.
    name = "timesfm"
    version = "0.1.0"
    task_type = "forecast"
    requirements = "requirements.txt"
    # timesfm (the 2.5 torch line) needs a modern CPython; 3.10+ is a safe floor
    # and matches the other forecast backends' venvs.
    requires_python = ">=3.10"

    def setup(self, ctx: BackendContext) -> None:
        # Device selection mirrors the other examples: "auto" probes for an
        # accelerator (CUDA/ROCm) and falls back to CPU. TimesFM 2.5 is small, so
        # CPU is viable; a GPU just makes it faster.
        self.device = pick_device(str(ctx.config.get("device", "auto")))

        # max_horizon caps how far a single forecast can look ahead; a request
        # with a larger horizon than this would be rejected by the model. Expose
        # it via config so an operator can raise it, with a generous default.
        self.max_horizon = int(ctx.config.get("max_horizon", 512))
        self.max_context = int(ctx.config.get("max_context", 1024))

        # Imported here (not at module top) so the SDK's static packaging path,
        # which must run without the heavy deps, never imports timesfm/torch.
        import timesfm

        self.model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(_MODEL_NAME)
        self.model.compile(
            timesfm.ForecastConfig(
                max_context=self.max_context,
                max_horizon=self.max_horizon,
                normalize_inputs=True,
            )
        )
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
                    "timesfm is a forecast backend: POST /forecast with a "
                    "ForecastRequest body, not /process"
                )
            }
        )

    def forecast(self, request: ForecastRequest) -> ForecastResult:
        """Predict ``request.horizon`` future steps of ``request.target``.

        TimesFM 2.5 is univariate — ``past_covariates``/``future_covariates`` are
        ignored (see the module docstring). We forecast from ``target`` alone and
        map TimesFM's fixed quantile grid onto the caller's requested levels.

        ``median`` is the model's **q0.5** column, not its ``point_forecast``
        (which is the *mean*): the SDK field is named ``median``, and for a skewed
        series mean != median. Reading q0.5 also keeps ``median`` consistent with
        ``quantiles["0.5"]`` when the caller asks for 0.5. If the model returns no
        usable quantile grid, we fall back to ``point_forecast``.
        """
        import numpy as np

        if request.horizon > self.max_horizon:
            raise ValueError(
                f"horizon {request.horizon} exceeds this backend's max_horizon "
                f"{self.max_horizon}; raise it via the backend config"
            )

        inputs = [np.asarray(request.target, dtype=np.float64)]

        # forecast() returns (point_forecast, quantile_forecast):
        #   point_forecast    shape (n_series, horizon)          — the mean/point.
        #   quantile_forecast shape (n_series, horizon, 1+Q)     — [mean, q1..qQ].
        point_forecast, quantile_forecast = self.model.forecast(
            horizon=request.horizon, inputs=inputs
        )

        # Median from the q0.5 column (index _MEDIAN_LEVEL_INDEX + 1, offset by
        # the mean at column 0); fall back to the point forecast if the quantile
        # grid is missing/short.
        qarr = np.asarray(quantile_forecast)
        median_col = _MEDIAN_LEVEL_INDEX + 1
        if qarr.ndim == 3 and qarr.shape[0] >= 1 and qarr.shape[2] > median_col:
            median = [float(x) for x in qarr[0, :, median_col]][: request.horizon]
        else:
            median = [float(x) for x in np.asarray(point_forecast)[0]][
                : request.horizon
            ]
        if len(median) != request.horizon:
            raise ValueError(
                f"timesfm produced {len(median)} median steps but horizon "
                f"is {request.horizon}"
            )

        quantiles = self._map_quantiles(
            np, quantile_forecast, request.quantile_levels, request.horizon
        )

        # Future timestamps, if the request carried a datetime axis.
        _, _, ts_fut_iso = _build_timestamps(
            request.timestamps, request.freq, len(request.target), request.horizon
        )
        return ForecastResult(
            median=median, quantiles=quantiles, timestamps=ts_fut_iso
        )

    # -- helpers ----------------------------------------------------------------

    @staticmethod
    def _map_quantiles(
        np: Any,
        quantile_forecast: Any,
        requested_levels: List[float],
        horizon: int,
    ) -> Optional[dict]:
        """Map TimesFM's fixed quantile grid onto the caller's requested levels.

        ``quantile_forecast`` has shape ``(n_series, horizon, 1 + Q)``: axis-2
        index 0 is the mean and indices ``1..Q`` are :data:`_TIMESFM_QUANTILE_LEVELS`.
        For each requested level we take the TimesFM column whose level is nearest
        (exact when the caller asks for one of TimesFM's own levels, e.g. 0.1/0.5).
        """
        arr = np.asarray(quantile_forecast)
        if arr.ndim != 3 or arr.shape[0] < 1:
            return None
        series0 = arr[0]  # (horizon, 1 + Q)
        n_cols = series0.shape[1]
        # Columns 1.. correspond to _TIMESFM_QUANTILE_LEVELS (truncated to fit
        # whatever the library actually returned, so we never index past the end).
        available = _TIMESFM_QUANTILE_LEVELS[: max(0, n_cols - 1)]
        if not available:
            return None

        quantiles: dict = {}
        for q in requested_levels:
            # nearest TimesFM level -> its column index (offset by 1 for the mean).
            nearest_idx = min(
                range(len(available)), key=lambda i: abs(available[i] - q)
            )
            col = [float(x) for x in series0[:, nearest_idx + 1]][:horizon]
            # Keep the contract's len(quantile) == len(median) invariant: skip a
            # column the model returned shorter than the horizon rather than
            # emitting a silently-truncated series.
            if len(col) == horizon:
                quantiles[str(q)] = col
        return quantiles or None


def _build_timestamps(
    timestamps: Optional[List[str]],
    freq: Optional[str],
    n: int,
    horizon: int,
):
    """Return ``(ts_hist, ts_fut, ts_fut_iso)`` for the forecast window.

    Only ``ts_fut_iso`` (ISO8601 strings for the horizon, or ``None``) is used by
    this univariate backend, but the full triple mirrors the Chronos-2 example so
    the timestamp logic reads the same across backends. ``None`` when the series
    has no real datetime axis (integer-index case) — the SDK contract keeps
    ``timestamps`` optional.
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
