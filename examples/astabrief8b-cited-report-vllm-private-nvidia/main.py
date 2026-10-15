"""Serve Ai2's AstaBrief-8B with the vLLM backend on a private NVIDIA GPU and
turn a question plus your own retrieved excerpts into a cited report.

The canonical SDK shape: ensure() -> wait_until_ready() -> one call. The axis
this example varies: the model is a report writer, not a chat model. You give
it the research question and the excerpts YOUR retrieval step found, in the
exact prompt it was fine-tuned on, and it answers in one pass with sections
("SECTION;"), a two-sentence "TLDR;" per section and inline [ID] citations.

AstaBrief-8B is a Qwen3-8B fine-tune (Apache-2.0), 16.4 GB in BF16. With
--max-model-len 16384 it fits one 24 GB card (RTX 4090, RTX A5000, L4 ...).

Run:  cp .env.example .env   # fill in real values, then
      python main.py                         # sample excerpts
      python main.py my_references.json      # your own excerpts
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

from inferencekey import (
    Backend,
    DataClient,
    ExecutionPolicy,
    InferenceKeyError,
    ManagementClient,
    WorkloadSpec,
)

MODEL = "allenai/AstaBrief_8B"

# vLLM launch command, run verbatim by the `vllm` backend.
#   --max-model-len 16384 -> the length AstaBrief was trained with (16,000).
#                            Room for about 15 excerpts plus a 4K-token report,
#                            and the KV cache still fits next to the 16.4 GB
#                            of BF16 weights on a 24 GB card.
#   --dtype bfloat16      -> the checkpoint's own precision.
# vLLM downloads only the *.safetensors shards (the repo also ships duplicate
# pytorch_model-*.bin files, which would double the download), and it applies
# the model's generation_config.json (top_p 0.95, top_k 20) by default.
# The worker injects host and port; do not hard-code --port.
COMMAND = f"vllm serve {MODEL} --max-model-len 16384 --dtype bfloat16"

SLUG = "astabrief-8b"

# The prompt AstaBrief was fine-tuned on. Ai2 publishes it in a Hugging Face
# dataset under CC-BY-NC-4.0, so it is downloaded at run time (pinned
# revision) instead of being copied into this Apache-2.0 repository.
PROMPT_URL = (
    "https://huggingface.co/datasets/allenai/AstaBrief_prompts/resolve/"
    "355b7a260fce95820a2bdf359ca1ddec920ee974/sft_prompt.txt"
)

HERE = Path(__file__).resolve().parent


def build_prompt(query: str, references: list[dict]) -> str:
    """Fill the official template. Each reference becomes one key/value pair:
    key "[<id> | <first author> et al. | <year> | Citations: <n>]", value the
    quoted excerpt. The model cites with the same [<id> ...] keys."""
    template = urllib.request.urlopen(PROMPT_URL, timeout=30).read().decode()
    section = {
        f"[{r['id']} | {r['author']} et al. | {r['year']} | Citations: {r['citations']}]":
        f"\n{r['quote']} (Citation count: {r['citations']})"
        for r in references
    }
    return template.replace("[QUERY]", query).replace(
        "[SECTION_REFERENCES]", json.dumps(section, indent=4)
    )


def main() -> int:
    refs_path = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "sample_references.json"
    sample = json.loads(refs_path.read_text())
    query, references = sample["query"], sample["references"]

    # Control plane (ik_sdk_): reads INFERENCEKEY_SDK_TOKEN / _PROJECT / _BASE_URL.
    mgmt = ManagementClient.from_env()
    spec = WorkloadSpec(
        name="AstaBrief-8B cited reports (vLLM)",
        slug=SLUG,
        model=MODEL,
        backend=Backend.VLLM,
        command=COMMAND,
        task_type="text2text",
        # Pin to your NVIDIA worker (24 GB or more). Never hard-code the id.
        worker_id=os.environ["IK_WORKER_ID"],
        execution_policy=ExecutionPolicy.FIXED,
        execution_policy_config={"replicas": 1},
    )

    try:
        # 1) Provision / reconcile. Idempotent by slug.
        ref = mgmt.ensure(spec)
        print(f"ensured {ref.project_slug}/{ref.workload_slug}")

        # 2) Wait until vLLM serves. A cold worker installs vLLM and downloads
        #    16.4 GB of weights first, so give it time.
        t0 = time.monotonic()
        mgmt.wait_until_ready(ref.workload_slug, timeout=1800)
        print(f"ready after {time.monotonic() - t0:.1f} s")

        # 3) One report through the OpenAI-compatible endpoint (ik_live_).
        #    Settings from the model card: temperature 0.7, up to 4096 tokens.
        ep = DataClient.from_env().endpoint(ref.workload_slug)
        t1 = time.monotonic()
        out = ep.generate_text(
            messages=[{"role": "user", "content": build_prompt(query, references)}],
            temperature=0.7,
            max_tokens=4096,
        )
        seconds = time.monotonic() - t1
        usage = (out.raw or {}).get("usage", {})
        report = out.text.strip()
        sections = re.findall(r"^SECTION;\s*(.+)$", report, flags=re.M)
        cited = sorted(set(re.findall(r"\[([^\]|]+?) \|", report)))
        print(f"\n{report}\n")
        print(
            f"{seconds:.1f} s, {usage.get('prompt_tokens')} prompt tokens, "
            f"{usage.get('completion_tokens')} report tokens, {len(sections)} sections, "
            f"{len(cited)} of {len(references)} references cited"
        )
        outputs = HERE / "outputs"
        outputs.mkdir(exist_ok=True)
        (outputs / "report.md").write_text(report + "\n")
        print(f"saved {outputs / 'report.md'}")
    except InferenceKeyError as e:
        # Don't print tokens; surface a clean message.
        print(f"error: {e}", file=sys.stderr)
        return 1
    finally:
        # 4) Tear down so the reserved GPU stops billing, unless you want to
        #    keep calling it (IK_KEEP_ALIVE=1).
        if os.environ.get("IK_KEEP_ALIVE") != "1":
            print(f"deleted={mgmt.delete(SLUG)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
