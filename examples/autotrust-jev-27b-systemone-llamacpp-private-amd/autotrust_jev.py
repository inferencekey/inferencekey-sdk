"""llama.cpp converter module for AutoTrust-JEV-27B v1.1 (llama.cpp b11815).

This is the file we used on 2026-10-22 to make our Q6_K GGUF. Upstream
llama.cpp b11815 does not know this checkpoint: its `decision_config.json` has
`"format_id": "jev-code-readout-v1"` and `"prompt": "jev"`. The module maps it
onto llama.cpp's existing `vllm-d3` decision model (d3 is the model Jev builds
on, per the repo's NOTICE.base; GGUF architecture `qwen35`), which writes
`decision.type = vllm-d3` and `attention.causal = false` into the GGUF. It
also keeps the checkpoint's full lm_head (the readout rows), because the
checkpoint ships no readout.safetensors.

Install: copy this file to `<llama.cpp>/conversion/autotrust_jev.py` and add
`"AutoTrustJevModel": "autotrust_jev",` after the `"VllmD3Model": "vllm_d3",`
line of `<llama.cpp>/conversion/__init__.py` (README, step 1).
"""

from __future__ import annotations

import json

from pathlib import Path
from typing import Any, Iterable, TYPE_CHECKING

from .base import ModelBase, gguf, logger
from .qwen import Qwen3_5TextModel
from .vllm_d3 import VllmD3Model

if TYPE_CHECKING:
    from torch import Tensor


def _is_autotrust_jev_checkpoint(dir_model: Path) -> bool:
    if not (dir_model / "decision_config.json").is_file():
        return False
    with open(dir_model / "decision_config.json", encoding="utf-8") as f:
        return json.load(f).get("format_id") == "jev-code-readout-v1"


@ModelBase.register_hparams_loader(_is_autotrust_jev_checkpoint)
def _load_autotrust_jev_hparams(dir_model: Path) -> dict[str, Any]:
    logger.info("gguf: detected AutoTrust JEV code-readout checkpoint")
    hparams = ModelBase.load_hparams(dir_model, False, guess=False)
    hparams["architectures"] = ["AutoTrustJevModel"]
    with open(dir_model / "decision_config.json", encoding="utf-8") as f:
        decision = json.load(f)
    # jev_format.py of the model repo is the d3 prompt byte for byte, with the same 255 codes
    if decision.get("prompt") != "jev":
        raise ValueError("unsupported AutoTrust JEV configuration")
    decision["prompt"] = "d3"
    hparams["decision"] = decision
    return hparams


@ModelBase.register("AutoTrustJevModel")
@ModelBase.example("autotrust/AutoTrust-JEV-27B-v1.1")
class AutoTrustJevModel(VllmD3Model):
    model_arch = gguf.MODEL_ARCH.QWEN35

    def generate_extra_tensors(self) -> Iterable[tuple[str, Tensor]]:
        # the checkpoint ships the readout as a full lm_head (zero outside the code rows), no readout.safetensors
        yield from Qwen3_5TextModel.generate_extra_tensors(self)
