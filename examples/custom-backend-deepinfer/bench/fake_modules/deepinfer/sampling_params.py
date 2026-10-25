"""Stand-in for ``deepinfer.sampling_params.SamplingParams``: only ``max_tokens`` matters here."""

from __future__ import annotations

from typing import Any, Optional


class SamplingParams:
    def __init__(self, max_tokens: int = 4096, temperature: float = 0.6, top_p: float = 1.0,
                 seq_seed: Optional[int] = None, **_: Any) -> None:
        self.max_tokens = int(max_tokens)
        self.temperature = temperature
        self.top_p = top_p
        self.seq_seed = seq_seed
