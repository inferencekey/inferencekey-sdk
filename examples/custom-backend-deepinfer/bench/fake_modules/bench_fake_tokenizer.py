"""Fallback tokenizer for the benchmark when no real tokenizer is available offline.

It costs about what the real one costs, which is what the benchmark measures:
``decode`` spins for ``DECODE_US_PER_TOKEN`` microseconds per token while
holding the GIL (measured: transformers 4.50 ``PreTrainedTokenizerFast.decode``
with the DeepSeek-V4.1 ``tokenizer.json`` takes about 4.4 us per token, linear
in the length), and ``encode`` spins ``ENCODE_US_PER_TOKEN`` per token (about
5.6 us measured). The text is made of synthetic words, one per token.
"""

from __future__ import annotations

import time
import zlib
from typing import Any, List

DECODE_US_PER_TOKEN = 4.4
ENCODE_US_PER_TOKEN = 5.6
VOCAB_SIZE = 32768
_VOCAB = [f" w{i}" for i in range(VOCAB_SIZE)]


def _spin(seconds: float) -> None:
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        pass


class FakeTokenizer:
    def encode(self, text: str, add_special_tokens: bool = True, **_: Any) -> List[int]:
        # About four characters per token, like English text in a BPE vocabulary.
        ids = [zlib.crc32(text[i : i + 4].encode()) % VOCAB_SIZE for i in range(0, len(text), 4)]
        _spin(len(ids) * ENCODE_US_PER_TOKEN * 1e-6)
        return ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = False, **_: Any) -> str:
        _spin(len(token_ids) * DECODE_US_PER_TOKEN * 1e-6)
        return "".join(_VOCAB[i % VOCAB_SIZE] for i in token_ids)

    def __len__(self) -> int:
        return VOCAB_SIZE


class AutoTokenizer:
    """``transformers.AutoTokenizer`` look-alike, installed by ``sim_server.py --fake-tokenizer``."""

    @staticmethod
    def from_pretrained(*_: Any, **__: Any) -> FakeTokenizer:
        return FakeTokenizer()
