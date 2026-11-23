"""Run llama.cpp's convert_hf_to_gguf.py with retries on every HTTP read.

`--remote` streams the safetensors with plain `requests.get` range requests
and no retry, so a single broken connection kills a 30 minute conversion.
"""
import runpy
import sys
import time

import requests

_get = requests.get


def get_with_retry(*args, **kwargs):
    # Without a timeout a stalled connection hangs the whole conversion forever.
    kwargs.setdefault("timeout", (20, 60))
    for attempt in range(1, 11):
        try:
            response = _get(*args, **kwargs)
            response.raise_for_status()
            return response
        except (requests.exceptions.ChunkedEncodingError, requests.exceptions.ConnectionError,
                requests.exceptions.HTTPError, requests.exceptions.Timeout) as error:
            if attempt == 10:
                raise
            print(f"[retry {attempt}] {error}", file=sys.stderr, flush=True)
            time.sleep(min(60, 2 ** attempt))


requests.get = get_with_retry
script = sys.argv[1]
sys.argv = sys.argv[1:]
sys.path.insert(0, str(__import__("pathlib").Path(script).parent))
runpy.run_path(script, run_name="__main__")
