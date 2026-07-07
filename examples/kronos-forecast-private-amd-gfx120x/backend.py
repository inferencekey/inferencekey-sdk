"""A custom **forecast** backend wrapping **Kronos** (K-line / OHLCV).

Kronos is a foundation model for financial candlestick (OHLCV) series: given the
recent history of a security's open/high/low/close/volume, it forecasts the next
``horizon`` candles. This backend maps the SDK's forecast contract onto Kronos:

* ``target``          → the **close** price history (the series we forecast).
* ``past_covariates`` → the other OHLCV columns for the same history, i.e.
  ``{"open": [...], "high": [...], "low": [...], "volume": [...]}`` (each the
  same length as ``target``). If they're omitted, we degrade to a flat candle
  (open=high=low=close, volume=0) so a bare ``target`` still forecasts.
* output ``median``/``quantiles`` → the forecast **close**, summarised across
  stochastic sample paths (see below).

**Quantiles by sampling.** Kronos is generative and its ``predict`` collapses
its internal samples to a mean — it does not return quantiles directly. To get a
median and quantiles we run ``predict`` ``n_paths`` times (each a distinct
stochastic path, via temperature/nucleus sampling), stack the close forecasts,
and take per-step percentiles. More paths → smoother quantiles but slower;
``n_paths`` is configurable (default 20).

Contract shape — this backend implements the SDK's typed **forecast** contract
(:class:`ForecastRequest` in, :class:`ForecastResult` out). See
``inferencekey.backend`` for the dataclasses.

The model weights are **not** shipped: ``setup()`` downloads the Kronos tokenizer
and model from Hugging Face once (cached). The Kronos *code* is vendored in
``kronos_model/`` (an MIT-licensed subset of https://github.com/shiyu-coder/Kronos;
see ``kronos_model/LICENSE``), so this example needs no GitHub install at runtime.

Run it (from the SDK repo root, with this folder importable)::

    python -m inferencekey.backend.serve \\
        --port 8099 \\
        --backend backend:KronosBackend \\
        --config-json '{"device": "auto", "n_paths": 20}'

``process()`` is implemented as a required stub — this backend only serves
``/forecast``. See ``README.md``.

Attribution: Kronos by Shiyu Chen et al. (https://github.com/shiyu-coder/Kronos,
MIT). The ``kronos_model/`` package here is a vendored subset of that project.
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

#: Hugging Face ids for the Kronos tokenizer and model weights.
_TOKENIZER_NAME = "NeoQuasar/Kronos-Tokenizer-base"
_MODEL_NAME = "NeoQuasar/Kronos-small"
#: OHLCV columns Kronos consumes (open/high/low/close required; volume optional).
_OHLC = ("open", "high", "low", "close")


class KronosBackend(CustomBackend):
    """Kronos OHLCV forecasting backend (close history in, close forecast out)."""

    # Declarative metadata exposed via GET /meta.
    name = "kronos"
    version = "0.1.0"
    task_type = "forecast"
    requirements = "requirements.txt"
    requires_python = ">=3.10"

    def setup(self, ctx: BackendContext) -> None:
        self.device = pick_device(str(ctx.config.get("device", "auto")))
        self.max_context = int(ctx.config.get("max_context", 512))
        # How many stochastic paths to average into quantiles (see module docs).
        self.n_paths = max(1, int(ctx.config.get("n_paths", 20)))

        # Imported here (not at module top) so the SDK's static packaging path,
        # which must run without the heavy deps, never imports torch/kronos.
        from kronos_model import Kronos, KronosPredictor, KronosTokenizer

        tokenizer = KronosTokenizer.from_pretrained(_TOKENIZER_NAME)
        model = Kronos.from_pretrained(_MODEL_NAME)
        # Kronos wants a concrete torch device string ("cuda:0"/"cpu"), whereas
        # pick_device returns "cuda"/"mps"/"cpu"; map "cuda" -> "cuda:0".
        device = "cuda:0" if self.device == "cuda" else self.device
        self.predictor = KronosPredictor(
            model, tokenizer, device=device, max_context=self.max_context
        )
        print("model loaded", file=sys.stderr, flush=True)

    def process(self, job: Job) -> Result:
        """Required stub — this backend only serves ``/forecast``."""
        return Result(
            output={
                "error": (
                    "kronos is a forecast backend: POST /forecast with a "
                    "ForecastRequest body (target = close history), not /process"
                )
            }
        )

    def forecast(self, request: ForecastRequest) -> ForecastResult:
        """Forecast the next ``request.horizon`` close prices from OHLCV history.

        Builds an OHLCV DataFrame from ``target`` (close) + ``past_covariates``
        (open/high/low/volume, degrading to a flat candle when absent), runs
        Kronos ``n_paths`` times, and turns the stacked close forecasts into a
        median + the requested quantiles.
        """
        import numpy as np
        import pandas as pd

        n = len(request.target)
        close = np.asarray(request.target, dtype=np.float64)

        # --- OHLCV history DataFrame ------------------------------------------
        cov = request.past_covariates or {}
        df = pd.DataFrame(
            {
                # open/high/low default to the close (a flat candle) when the
                # caller sends only `target`; volume defaults to 0.
                "open": np.asarray(cov.get("open", close), dtype=np.float64),
                "high": np.asarray(cov.get("high", close), dtype=np.float64),
                "low": np.asarray(cov.get("low", close), dtype=np.float64),
                "close": close,
                "volume": np.asarray(cov.get("volume", np.zeros(n)), dtype=np.float64),
            }
        )

        # --- timestamps -------------------------------------------------------
        # Kronos uses the datetime axis as a temporal feature, so it needs both a
        # history (x) and a future (y) timestamp series. Build them from the
        # request's timestamps/freq, synthesising a daily axis if none is given.
        x_ts, y_ts, y_iso = _build_timestamps(
            request.timestamps, request.freq, n, request.horizon
        )

        # --- run n_paths stochastic samples and collect the close forecasts ---
        paths: List[Any] = []
        for _ in range(self.n_paths):
            pred_df = self.predictor.predict(
                df=df,
                x_timestamp=x_ts,
                y_timestamp=y_ts,
                pred_len=request.horizon,
                sample_count=1,
                verbose=False,
            )
            paths.append(np.asarray(pred_df["close"], dtype=np.float64))

        stacked = np.stack(paths)  # (n_paths, horizon)
        if stacked.shape[1] != request.horizon:
            raise ValueError(
                f"kronos produced {stacked.shape[1]} forecast steps but horizon "
                f"is {request.horizon}"
            )

        median = [float(x) for x in np.quantile(stacked, 0.5, axis=0)]
        quantiles = {
            str(q): [float(x) for x in np.quantile(stacked, q, axis=0)]
            for q in request.quantile_levels
        }
        return ForecastResult(median=median, quantiles=(quantiles or None), timestamps=y_iso)


def _build_timestamps(
    timestamps: Optional[List[str]],
    freq: Optional[str],
    n: int,
    horizon: int,
):
    """Return ``(x_timestamp, y_timestamp, y_iso)`` as pandas Series for Kronos.

    Unlike the other forecast backends, Kronos *requires* a datetime axis, so we
    always produce one — synthesising a daily range when the request has neither
    ``timestamps`` nor a usable ``freq``. ``y_iso`` is the ISO8601 form of the
    future window to echo back in the result.
    """
    import pandas as pd

    if timestamps:
        x = pd.to_datetime(pd.Series(timestamps))
        inferred = pd.infer_freq(x) if len(x) >= 3 else None
        last = x.iloc[-1]
        if inferred is not None:
            y = pd.date_range(start=last, periods=horizon + 1, freq=inferred)[1:]
        elif len(x) >= 2:
            step = last - x.iloc[-2]
            y = pd.DatetimeIndex([last + step * (i + 1) for i in range(horizon)])
        else:
            y = pd.date_range(start=last, periods=horizon + 1, freq="D")[1:]
        y = pd.Series(y)
        return x, y, [t.isoformat() for t in y]

    # No explicit timestamps: build a range from freq (default daily), falling
    # back to daily if the freq alias doesn't parse on this pandas version.
    use_freq = freq or "D"
    try:
        x_idx = pd.date_range(end=pd.Timestamp.now().normalize(), periods=n, freq=use_freq)
        y_idx = pd.date_range(start=x_idx[-1], periods=horizon + 1, freq=use_freq)[1:]
    except (ValueError, TypeError):
        x_idx = pd.date_range(end=pd.Timestamp.now().normalize(), periods=n, freq="D")
        y_idx = pd.date_range(start=x_idx[-1], periods=horizon + 1, freq="D")[1:]
    x, y = pd.Series(x_idx), pd.Series(y_idx)
    return x, y, [t.isoformat() for t in y]
