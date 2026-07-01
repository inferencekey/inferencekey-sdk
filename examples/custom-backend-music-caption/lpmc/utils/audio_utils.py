### Vendored (inference-only subset) from LP-MusicCaps
### (seungheondoh/lp-music-caps). Only the audio loading helpers used by the
### music-captioning inference path are kept; the noise-generator utilities from
### the original module are omitted.

STR_CLIP_ID = 'clip_id'
STR_AUDIO_SIGNAL = 'audio_signal'
STR_TARGET_VECTOR = 'target_vector'

STR_CH_FIRST = 'channels_first'
STR_CH_LAST = 'channels_last'

import io
import os
import subprocess
from typing import Tuple
from pathlib import Path

import numpy as np
import soundfile as sf


def _resample_load_ffmpeg(path: str, sample_rate: int, downmix_to_mono: bool) -> Tuple[np.ndarray, int]:
    """
    Decoding, downmixing, and downsampling by ffmpeg.
    Returns a channel-first audio signal.
    """

    def _decode_resample_by_ffmpeg(filename, sr):
        """decode, downmix, and resample audio file"""
        channel_cmd = '-ac 1 ' if downmix_to_mono else ''  # downmixing option
        resampling_cmd = f'-ar {str(sr)}' if sr else ''  # downsampling option
        cmd = f"ffmpeg -i \"{filename}\" {channel_cmd} {resampling_cmd} -f wav -"
        p = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()
        return out

    src, sr = sf.read(io.BytesIO(_decode_resample_by_ffmpeg(path, sr=sample_rate)))
    return src.T, sr


def _resample_load_librosa(path: str, sample_rate: int, downmix_to_mono: bool, **kwargs) -> Tuple[np.ndarray, int]:
    """
    Decoding, downmixing, and downsampling by librosa.
    Returns a channel-first audio signal.
    """
    import librosa
    src, sr = librosa.load(path, sr=sample_rate, mono=downmix_to_mono, **kwargs)
    return src, sr


def load_audio(
    path,
    ch_format: str,
    sample_rate: int = None,
    downmix_to_mono: bool = False,
    resample_by: str = 'ffmpeg',
    **kwargs,
) -> Tuple[np.ndarray, int]:
    """Load an audio file, optionally downmixing to mono and resampling.

    The audio decoding is done by ``ffmpeg`` (default) or ``librosa``; both can
    handle common formats including mp3 and wav.

    Args:
        path: audio file path
        ch_format: one of 'channels_first' or 'channels_last'
        sample_rate: target sampling rate. if None, use the rate of the audio file
        downmix_to_mono:
        resample_by (str): 'librosa' or 'ffmpeg'. it decides backend for audio decoding and resampling.
        **kwargs: keyword args for librosa.load - offset, duration, dtype, res_type.

    Returns:
        (audio, sr) tuple
    """
    if ch_format not in (STR_CH_FIRST, STR_CH_LAST):
        raise ValueError(f'ch_format is wrong here -> {ch_format}')

    if os.stat(path).st_size > 8000:
        if resample_by == 'librosa':
            src, sr = _resample_load_librosa(path, sample_rate, downmix_to_mono, **kwargs)
        elif resample_by == 'ffmpeg':
            src, sr = _resample_load_ffmpeg(path, sample_rate, downmix_to_mono)
        else:
            raise NotImplementedError(f'resample_by: "{resample_by}" is not supported yet')
    else:
        raise ValueError('Given audio is too short!')
    return src, sr
