"""A custom **forecast** backend wrapping Amazon **Chronos-2**.

Given the history of a single time series (plus optional covariates), this
backend predicts the next ``horizon`` steps as a median point forecast and a set
of quantiles. It wraps ``Chronos2Pipeline`` (``amazon/chronos-2``): a ~120M
parameter, zero-shot foundation model for time-series forecasting that runs on a
recent GPU (NVIDIA CUDA or AMD ROCm) or on CPU, on standard ``torch`` — no
custom kernels. The representative use case is financial series (a stock, index,
VIX, volume, calendar) where past/known-future covariates matter.

Contract shape — this backend implements the SDK's typed **forecast** contract
(:class:`ForecastRequest` in, :class:`ForecastResult` out), *not* the free-dict
``process`` shape. See ``inferencekey.backend`` for the dataclasses.

* input  — ``{"id": "f1", "target": [...], "horizon": 24, "freq": "D",
  "quantile_levels": [0.1, 0.5, 0.9], "past_covariates": {...},
  "future_covariates": {...}}`` (only ``id``/``target``/``horizon`` required).
* output — ``{"forecast": {"median": [...], "timestamps": [...],
  "quantiles": {"0.1": [...], "0.9": [...]}}}``.

The model weights are **not** shipped with this example: ``setup()`` downloads
``amazon/chronos-2`` from Hugging Face once (cached), then loads the pipeline a
single time. ``forecast()`` reuses that loaded pipeline for every request.

Run it (from the SDK repo root, with this folder importable)::

    python -m inferencekey.backend.serve \\
        --port 8099 \\
        --backend backend:Chronos2Backend \\
        --config-json '{"device": "auto"}'

``process()`` is implemented as a required stub — this backend only serves
``/forecast``. See ``README.md``.

Attribution: Chronos-2 by Amazon (``amazon/chronos-2``,
https://github.com/amazon-science/chronos-forecasting).
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any, List, Optional

from inferencekey.backend import (
    BackendContext,
    CustomBackend,
    ForecastRequest,
    ForecastResult,
    Job,
    Result,
    pick_device,
)

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime here
    import pandas as pd

#: The Hugging Face model id Chronos-2 loads its weights from.
_MODEL_NAME = "amazon/chronos-2"
#: The single-series id we tag the history with (Chronos-2's ``predict_df`` keys
#: rows by an id column; this example forecasts one series per request).
_SERIES_ID = "series"


class Chronos2Backend(CustomBackend):
    """Chronos-2 forecasting backend (series history in, quantile forecast out)."""

    # Declarative metadata exposed via GET /meta.
    name = "chronos-2"
    version = "0.1.0"
    task_type = "forecast"
    requirements = "requirements.txt"
    # chronos-forecasting (the chronos-2 line, >= 2.0.0rc1) requires CPython
    # 3.10+. Declaring it here makes the worker provision a matching interpreter
    # for the backend's venv regardless of the node's default Python.
    requires_python = ">=3.10"

    def setup(self, ctx: BackendContext) -> None:
        # Device selection mirrors the other examples: "auto" probes for an
        # accelerator (CUDA/ROCm -> "cuda", Apple -> "mps") and falls back to
        # CPU. pick_device validates an explicit {"device": ...} and degrades to
        # CPU with a warning rather than crashing on a node that can't use it.
        # Chronos-2 is small, so CPU is viable; a GPU just makes it faster.
        self.device = pick_device(str(ctx.config.get("device", "auto")))

        # Imported here (not at module top) so the SDK's static packaging path,
        # which must run without the heavy deps, never imports chronos/torch.
        from chronos import Chronos2Pipeline

        self.pipe = Chronos2Pipeline.from_pretrained(
            _MODEL_NAME, device_map=self.device
        )
        print("model loaded", file=sys.stderr, flush=True)

    def process(self, job: Job) -> Result:
        """Required stub — this backend only serves ``/forecast``.

        ``process`` is abstract on :class:`CustomBackend`, so a forecast-only
        backend must still implement it. We return a clear, non-crashing message
        pointing the caller at the right endpoint instead of pretending to run a
        text/classification job.
        """
        return Result(
            output={
                "error": (
                    "chronos-2 is a forecast backend: POST /forecast with a "
                    "ForecastRequest body, not /process"
                )
            }
        )

    def forecast(self, request: ForecastRequest) -> ForecastResult:
        """Predict ``request.horizon`` future steps of ``request.target``.

        Maps the SDK's :class:`ForecastRequest` onto Chronos-2's ``predict_df``
        (a pandas-DataFrame API) and maps the returned quantile DataFrame back
        onto a :class:`ForecastResult`.
        """
        import pandas as pd

        # --- history timestamps ------------------------------------------------
        # Three cases, in priority order:
        #   1. explicit ISO timestamps supplied by the caller -> parse them.
        #   2. a pandas freq (e.g. "D", "H") -> synthesise a date range ending
        #      "now" (anchor is irrelevant to a zero-shot model; only spacing and
        #      the continuation into the future matter).
        #   3. neither -> a plain integer RangeIndex 0..n-1 as the timestamp.
        n = len(request.target)
        ts_hist, ts_fut, ts_fut_iso = self._build_timestamps(
            pd, request.timestamps, request.freq, n, request.horizon
        )

        # --- context (history) DataFrame --------------------------------------
        # Columns: id, timestamp, target, plus one column per past covariate.
        context_cols: dict[str, Any] = {
            "id": _SERIES_ID,
            "timestamp": ts_hist,
            "target": request.target,
        }
        if request.past_covariates:
            for name, values in request.past_covariates.items():
                context_cols[name] = values
        context_df = pd.DataFrame(context_cols)

        # --- future DataFrame (only if there are known-future covariates) -----
        # Chronos-2 wants the future covariate values (and their timestamps) but
        # NOT a target column for the horizon. With no future covariates, pass
        # future_df=None and let the pipeline date the window itself.
        future_df: Optional["pd.DataFrame"] = None
        if request.future_covariates:
            future_cols: dict[str, Any] = {"id": _SERIES_ID, "timestamp": ts_fut}
            for name, values in request.future_covariates.items():
                future_cols[name] = values
            future_df = pd.DataFrame(future_cols)

        # --- run the model ----------------------------------------------------
        # Always ask the model for 0.5 (the median) even if the caller didn't,
        # so ForecastResult.median is a genuine central estimate rather than a
        # degraded fall-back to whichever quantile happens to sit nearest 0.5.
        # We only *report* the levels the caller asked for (see _extract_forecast).
        model_levels = list(request.quantile_levels)
        if not any(abs(q - 0.5) < 1e-9 for q in model_levels):
            model_levels.append(0.5)
        pred = self.pipe.predict_df(
            context_df,
            future_df=future_df,
            prediction_length=request.horizon,
            quantile_levels=model_levels,
            id_column="id",
            timestamp_column="timestamp",
            target="target",
        )

        # --- map the prediction DataFrame back onto ForecastResult ------------
        median, quantiles = self._extract_forecast(
            pred, request.quantile_levels, request.horizon
        )
        return ForecastResult(
            median=median, quantiles=quantiles, timestamps=ts_fut_iso
        )

    # -- helpers ----------------------------------------------------------------

    @staticmethod
    def _build_timestamps(
        pd: Any,
        timestamps: Optional[List[str]],
        freq: Optional[str],
        n: int,
        horizon: int,
    ):
        """Return ``(ts_hist, ts_fut, ts_fut_iso)`` for context/future/result.

        * ``ts_hist`` — timestamps (or integer index) for the ``n`` history rows.
        * ``ts_fut`` — timestamps for the ``horizon`` future rows (for the
          future-covariates DataFrame). Continues ``ts_hist``'s spacing.
        * ``ts_fut_iso`` — ISO8601 strings for the forecast window to echo back
          in the result, or ``None`` when the series has no real datetime axis
          (the integer-index case) — the SDK contract keeps ``timestamps``
          optional, so ``None`` is valid.
        """
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
                ts_fut = pd.DatetimeIndex(
                    [ts_hist[-1]] * horizon
                )
            return ts_hist, ts_fut, [t.isoformat() for t in ts_fut]

        if freq:
            # Anchor the window at "now"; only spacing/continuation matter to a
            # zero-shot model. periods=n ending now, then continue for horizon.
            # A freq alias pandas can't parse (e.g. a deprecated/upper-case one on
            # newer pandas) falls through to the integer-index case below rather
            # than sinking the forecast.
            try:
                ts_hist = pd.date_range(end=pd.Timestamp.now().normalize(),
                                        periods=n, freq=freq)
                ts_fut = pd.date_range(
                    start=ts_hist[-1], periods=horizon + 1, freq=freq
                )[1:]
                return ts_hist, ts_fut, [t.isoformat() for t in ts_fut]
            except (ValueError, TypeError):
                pass

        # No datetime axis: use an integer RangeIndex. Chronos-2 treats these as
        # ordered steps. No real timestamps to echo -> ts_fut_iso is None.
        ts_hist = list(range(n))
        ts_fut = list(range(n, n + horizon))
        return ts_hist, ts_fut, None

    @staticmethod
    def _extract_forecast(pred: Any, quantile_levels: List[float], horizon: int):
        """Pull ``(median, quantiles)`` out of Chronos-2's prediction DataFrame.

        Assumption about ``predict_df``'s output (documented so it's easy to
        adjust if the library changes): the returned DataFrame has one row per
        forecast step and one column per requested quantile level, named by the
        numeric level. We resolve each requested level to its column tolerantly —
        matching the column whose float value equals the level (covers columns
        named ``0.5`` as a float, ``"0.5"`` as a string, or ``"0.50"``) — so we
        don't hard-code an exact spelling. The median is the 0.5 column — we
        always request 0.5 from the model (see ``forecast``), so it is present;
        the remaining fall-backs only matter if the library changes its output.
        Structural columns (``id``/``timestamp``/``target``) that ``predict_df``
        may echo back are excluded from every numeric pool, so an integer
        ``timestamp`` (the RangeIndex case) can never be mistaken for a forecast.
        """
        # Structural columns predict_df may carry through; never a forecast.
        _NON_FORECAST_COLS = {"id", "timestamp", "target"}
        columns = [c for c in pred.columns if c not in _NON_FORECAST_COLS]

        def _col_as_float(col: Any) -> Optional[float]:
            try:
                return float(col)
            except (TypeError, ValueError):
                return None

        # Map each forecast column to its float value once (structural columns
        # already excluded above).
        numeric_cols = {c: _col_as_float(c) for c in columns}

        def _find_quantile_col(level: float) -> Optional[Any]:
            for col, val in numeric_cols.items():
                if val is not None and abs(val - level) < 1e-9:
                    return col
            return None

        def _series_to_list(col: Any) -> List[float]:
            return [float(x) for x in pred[col].tolist()][:horizon]

        quantiles: dict[str, List[float]] = {}
        for q in quantile_levels:
            col = _find_quantile_col(q)
            if col is not None:
                quantiles[str(q)] = _series_to_list(col)

        # Median: the 0.5 column if present, else a named point column, else the
        # quantile closest to 0.5, else (last resort) the first numeric column.
        median: List[float]
        median_col = _find_quantile_col(0.5)
        if median_col is not None:
            median = _series_to_list(median_col)
        else:
            named = next(
                (c for c in ("predictions", "mean", "median", "0.5") if c in columns),
                None,
            )
            if named is not None:
                median = _series_to_list(named)
            elif quantiles:
                nearest = min(quantile_levels, key=lambda q: abs(q - 0.5))
                median = list(quantiles[str(nearest)])
            else:
                numeric_only = [c for c, v in numeric_cols.items() if v is not None]
                if not numeric_only:
                    raise ValueError(
                        "chronos-2 predict_df returned no numeric forecast "
                        f"columns; got columns {columns!r}"
                    )
                median = _series_to_list(numeric_only[0])

        if len(median) != horizon:
            raise ValueError(
                f"chronos-2 produced {len(median)} forecast steps but horizon "
                f"is {horizon} (columns {columns!r})"
            )

        return median, (quantiles or None)
