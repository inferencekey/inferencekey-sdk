"""A custom **audio2text** backend wrapping LP-MusicCaps.

Given a musical audio clip, this backend generates a natural-language
description of the song. It wraps the LP-MusicCaps ``BartCaptionModel``
(``seungheondoh/lp-music-caps``): the audio is resampled to 16 kHz mono, split
into 10-second chunks, and each chunk is captioned by a BART decoder conditioned
on a mel-spectrogram audio encoder.

Contract shape (free dicts, **not** OpenAI-compatible):

* input  — ``{"audio_b64": "<base64 of a .wav/.mp3 file>", "num_beams": 5}``
  (``num_beams`` optional, default 5).
* output — ``{"description": "<joined text>", "chunks": [{"text": ..., "time": "0:00-10:00"}, ...]}``

The model weights are **not** shipped with this example: ``setup()`` downloads
``transfer.pth`` from Hugging Face once and caches it locally, then loads the
model a single time. ``process()`` reuses that loaded model for every job.

Run it (from the SDK repo root, with this folder importable)::

    python -m inferencekey.backend.serve \\
        --port 8099 \\
        --backend backend:MusicCaptionBackend \\
        --config-json '{"device": "cpu"}'

Attribution: LP-MusicCaps by SeungHeon Doh et al.
(https://github.com/seungheondoh/lp-music-caps, model
``seungheondoh/lp-music-caps``). See ``README.md`` for the license note. The
``lpmc/`` package here is a vendored inference-only subset of that project.
"""

from __future__ import annotations

import base64
import binascii
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import torch

from inferencekey.backend import BackendContext, CustomBackend, Job, Result

from lpmc.music_captioning.model.bart import BartCaptionModel
from lpmc.utils.audio_utils import STR_CH_FIRST, load_audio
from lpmc.utils.eval_utils import load_pretrained

#: LP-MusicCaps transfer checkpoint on the Hugging Face hub.
_WEIGHTS_URL = "https://huggingface.co/seungheondoh/lp-music-caps/resolve/main/transfer.pth"
#: Default cache location for the downloaded weights.
_DEFAULT_WEIGHTS_DIR = Path.home() / ".cache" / "inferencekey" / "lp-music-caps"
#: From the original hparams.yaml (only these two values are actually used).
_MAX_LENGTH = 128
#: 10 s at 16 kHz — one captioning chunk.
_TARGET_SR = 16000
_CHUNK_DURATION = 10
_N_SAMPLES = _CHUNK_DURATION * _TARGET_SR  # 160000


def _download_weights(weights_dir: Path) -> Path:
    """Return the path to ``transfer.pth``, downloading it once if missing."""
    weights_dir.mkdir(parents=True, exist_ok=True)
    weights_path = weights_dir / "transfer.pth"
    if weights_path.exists() and weights_path.stat().st_size > 0:
        return weights_path
    tmp_path = weights_path.with_suffix(".pth.partial")
    urllib.request.urlretrieve(_WEIGHTS_URL, tmp_path)
    tmp_path.replace(weights_path)
    return weights_path


def _get_audio(audio_path: str) -> torch.Tensor:
    """Load an audio file and split it into 10 s / 160000-sample chunks.

    Replicates ``get_audio`` from the original Flask app: resample to 16 kHz
    mono, zero-pad clips shorter than one chunk, then stack whole chunks into a
    float32 tensor of shape ``[n_chunks, 160000]``.
    """
    # Decode via librosa (a declared dependency) rather than the module's
    # default ffmpeg subprocess: librosa reads wav/mp3/flac/ogg without needing
    # a system ffmpeg on PATH, so the backend is portable out of the box.
    audio, _ = load_audio(
        path=audio_path,
        ch_format=STR_CH_FIRST,
        sample_rate=_TARGET_SR,
        downmix_to_mono=True,
        resample_by="librosa",
    )
    if len(audio.shape) == 2:
        audio = audio.mean(0, False)  # to mono
    input_size = int(_N_SAMPLES)
    if audio.shape[-1] < input_size:  # pad sequence
        pad = np.zeros(input_size)
        pad[: audio.shape[-1]] = audio
        audio = pad
    ceil = int(audio.shape[-1] // _N_SAMPLES)
    audio = torch.from_numpy(
        np.stack(np.split(audio[: ceil * _N_SAMPLES], ceil)).astype("float32")
    )
    return audio


class MusicCaptionBackend(CustomBackend):
    """LP-MusicCaps music-captioning backend (audio in, description out)."""

    # Declarative metadata exposed via GET /meta.
    name = "lp-music-caps"
    version = "0.1.0"
    task_type = "audio2text"
    requirements = "requirements.txt"

    def setup(self, ctx: BackendContext) -> None:
        # Device is read explicitly from config; default CPU, no GPU
        # autodetection and no forced .cuda().
        self.device = str(ctx.config.get("device", "cpu"))
        weights_dir = Path(
            ctx.config.get("weights_dir", str(_DEFAULT_WEIGHTS_DIR))
        )

        weights_path = _download_weights(weights_dir)
        model = BartCaptionModel(max_length=_MAX_LENGTH)
        # transfer.pth has no 'module.' prefix -> mdp=False.
        self.model, _ = load_pretrained(
            str(weights_path), model, device=self.device, mdp=False
        )
        self.model.eval()
        print("model loaded", file=sys.stderr, flush=True)

    def process(self, job: Job) -> Result:
        audio_b64 = job.input.get("audio_b64")
        if not isinstance(audio_b64, str) or not audio_b64:
            raise ValueError("input must carry a base64 string 'audio_b64'")
        try:
            audio_bytes = base64.b64decode(audio_b64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError(f"input.audio_b64 is not valid base64: {exc}")

        num_beams = int(job.input.get("num_beams", 5))

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                suffix=".audio", delete=False
            ) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name

            try:
                audio_tensor = _get_audio(tmp_path)
            except Exception as exc:
                raise ValueError(f"could not decode audio: {exc}")

            audio_tensor = audio_tensor.to(self.device)
            with torch.no_grad():
                output = self.model.generate(
                    samples=audio_tensor,
                    num_beams=num_beams,
                )
        finally:
            if tmp_path is not None and os.path.exists(tmp_path):
                os.remove(tmp_path)

        chunks = []
        for idx, text in enumerate(output):
            time = f"{idx * 10}:00-{(idx + 1) * 10}:00"
            chunks.append({"text": text, "time": time})
        description = " ".join(chunk["text"] for chunk in chunks)
        return Result(output={"description": description, "chunks": chunks})
