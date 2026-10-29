"""Optional: the same clips through **Qwen3-ASR-1.7B** on the built-in `sglang` backend, for a side-by-side.

Qwen3-ASR-1.7B is the model the Bark 3.0 Pro authors compare against (4.43 %
vs 4.29 % average word error on the Open ASR Leaderboard's English sets, their
table). It runs on InferenceKey's built-in sglang backend, no custom code.
Run main.py first with IK_KEEP_ALIVE=1 if you want both on the same GPU
(Bark needs about 2 GB, so a 24 GB card holds both). Results land in
./outputs/qwen3-asr; the script prints both word error rates when Bark's
summary is there.

Run:  python compare_qwen3_asr.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import requests

from asr_eval import run
from inferencekey import Backend, ExecutionPolicy, InferenceKeyError, ManagementClient, WorkloadSpec
from main import OUT_DIR as BARK_OUT, _load_dotenv

SLUG = "qwen3-asr-sglang"
MODEL = "Qwen/Qwen3-ASR-1.7B"
# Same launch command as examples/sglang-qwen3-asr-private. 0.45 of the card
# is plenty for one stream of short clips and leaves room for Bark.
COMMAND = (
    "sglang serve"
    f" --model-path {MODEL}"
    " --served-model-name qwen3-asr"
    " --trust-remote-code"
    f" --mem-fraction-static {os.environ.get('QWEN_MEM_FRACTION', '0.45')}"
)
OUT_DIR = Path(__file__).resolve().parent / "outputs" / "qwen3-asr"


def main() -> int:
    _load_dotenv()
    audio = os.environ.get("IK_AUDIO", "sample.wav")
    mgmt = ManagementClient.from_env()
    spec = WorkloadSpec(
        name="Qwen3-ASR 1.7B (sglang)",
        slug=SLUG,
        model=MODEL,
        backend=Backend.SGLANG,
        command=COMMAND,
        task_type="audio2text",
        worker_id=os.environ["IK_WORKER_ID"],
        config={"sglang_version": "0.5.14"},
        execution_policy=ExecutionPolicy.FIXED,
    )
    try:
        t0 = time.time()
        ref = mgmt.ensure(spec)
        print(f"ensured {ref.project_slug}/{ref.workload_slug}; waiting (first start installs sglang, ~4.7 GB of weights)...")
        mgmt.wait_until_ready(ref.workload_slug, timeout=2400)
        print(f"ready after {time.time() - t0:.0f} s")
        base = os.environ.get("INFERENCEKEY_BASE_URL", "https://cloud.inferencekey.com").rstrip("/")
        url = f"{base}/endpoint/{os.environ['INFERENCEKEY_PROJECT']}/{ref.workload_slug}/v1/audio/transcriptions"
        qwen = run(url, os.environ["INFERENCEKEY_API_KEY"], audio, OUT_DIR, model="qwen3-asr")
        bark_file = BARK_OUT / "summary.json"
        if bark_file.is_file() and qwen.get("overall"):
            bark = json.loads(bark_file.read_text())["summary"]
            print(f"\nword error rate on the same {qwen['clips']} clips: Bark 3.0 Pro {bark['overall']['wer']} %, "
                  f"Qwen3-ASR-1.7B {qwen['overall']['wer']} %")
            for s in qwen.get("by_set", {}):
                print(f"  {s:20s} Bark {bark['by_set'].get(s, {}).get('wer')} %  Qwen {qwen['by_set'][s]['wer']} %")
    except InferenceKeyError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except requests.HTTPError as e:
        print(f"request failed: {e}: {e.response.text[:300]}", file=sys.stderr)
        return 1
    finally:
        if os.environ.get("IK_KEEP_ALIVE") == "1":
            print(f"IK_KEEP_ALIVE=1: keeping {SLUG}; delete it with mgmt.delete('{SLUG}')")
        else:
            print(f"deleted={mgmt.delete(SLUG)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
