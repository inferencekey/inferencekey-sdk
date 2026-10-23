"""Serve the DeepInfer backend over the real SDK runtime with simulated ranks.

Everything on the parent side is the production code: ``DeepInferBackend``,
``DeepInferEngine`` (router, request and result queues, reader thread,
detokenization), and the HTTP server of ``inferencekey.backend.serve``.
Even the rank loop is the real ``_rank_main``; only what it imports is
simulated: ``bench/fake_modules`` holds a stand-in ``torch`` and ``deepinfer``
(see ``fake_modules/deepinfer/entrypoints/llm.py`` for the rank model), and it
goes at the END of ``sys.path`` just before the ranks start, so it is only
found where no real torch or DeepInfer is installed. The production code has no
benchmark hook; the simulation knobs travel in the existing ``llm_kwargs``
config key, which the rank loop passes to ``LLM(...)``.

    python sim_server.py --port 8199 --tokenizer-dir /path/to/DeepSeek-V4.1-Flash

``load.py`` starts this script itself; run it by hand only to poke at it.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import pathlib
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
EXAMPLE_DIR = HERE.parent
FAKE_MODULES = HERE / "fake_modules"


def _model_dir(tokenizer_dir: str) -> str:
    """A checkpoint-shaped directory: a config.json plus the tokenizer files."""
    path = tempfile.mkdtemp(prefix="deepinfer-sim-model-")
    with open(os.path.join(path, "config.json"), "w", encoding="utf-8") as fh:
        json.dump({"model_type": "deepseek_v41_sim"}, fh)
    if tokenizer_dir:
        for name in ("tokenizer.json", "tokenizer_config.json"):
            src = os.path.join(tokenizer_dir, name)
            if os.path.exists(src):
                os.symlink(os.path.abspath(src), os.path.join(path, name))
    return path


def build_config(args: argparse.Namespace) -> dict:
    ctx = mp.get_context("spawn")
    sim = {
        "step_rate": args.step_rate,
        "tokens_per_step": args.tokens_per_step,
        "prefill_tokens_per_s": args.prefill_rate,
        "kv_blocks": args.kv_blocks,
        "decode_ctx_beta": args.decode_ctx_beta,
        "tokenizer": "fake" if args.fake_tokenizer else "hf",
        "stats_path": args.stats_path,
    }
    if not args.no_lockstep and args.dp > 1:
        sim["barrier"] = ctx.Barrier(args.dp)
        sim["flags"] = ctx.Array("b", args.dp, lock=False)
    return {
        "engine": "deepinfer",
        "model_dir": _model_dir("" if args.fake_tokenizer else args.tokenizer_dir),
        "ep": args.dp,
        "dp": args.dp,
        "cp": 1,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "thinking_mode": args.thinking_mode,
        "llm_kwargs": {"bench_sim": sim},
        "load_timeout_seconds": 120,
        **json.loads(args.backend_config or "{}"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--dp", type=int, default=4, help="simulated data-parallel ranks")
    parser.add_argument("--step-rate", type=float, default=70.0, help="decode steps per second per rank")
    parser.add_argument("--tokens-per-step", type=int, default=1, help="tokens per running request per step")
    parser.add_argument("--max-num-seqs", type=int, default=128)
    parser.add_argument("--max-num-batched-tokens", type=int, default=32768)
    parser.add_argument("--prefill-rate", type=float, default=88000.0,
                        help="prompt tokens per second per rank (0: prefill is free)")
    parser.add_argument("--kv-blocks", type=int, default=0,
                        help="KV cache blocks of 128 tokens per rank (0: unlimited)")
    parser.add_argument("--decode-ctx-beta", type=float, default=0.0,
                        help="extra seconds per decode step per token of context in the batch")
    parser.add_argument("--tokenizer-dir", default=os.environ.get("BENCH_TOKENIZER_DIR", ""),
                        help="directory with the DeepSeek-V4.1 tokenizer.json and tokenizer_config.json")
    parser.add_argument("--fake-tokenizer", action="store_true",
                        help="use bench_fake_tokenizer (cost-matched) instead of transformers")
    parser.add_argument("--thinking-mode", default="thinking")
    parser.add_argument("--stats-path", default="", help="rank counters go to <path>.rank<N>")
    parser.add_argument("--no-lockstep", action="store_true", help="ranks step on independent clocks")
    parser.add_argument("--backend-config", default="",
                        help="JSON object merged into the backend config, e.g. '{\"stream_interval_ms\": 0}'")
    args = parser.parse_args()
    if not args.fake_tokenizer and not args.tokenizer_dir:
        parser.error("pass --tokenizer-dir (or BENCH_TOKENIZER_DIR) or --fake-tokenizer")

    sys.path.insert(0, str(EXAMPLE_DIR))
    if args.fake_tokenizer:
        sys.path.insert(0, str(FAKE_MODULES))
        import bench_fake_tokenizer  # noqa: PLC0415

        sys.modules["transformers"] = bench_fake_tokenizer  # type: ignore[assignment]
    else:
        import transformers  # noqa: F401,PLC0415 (the real one, before the fakes join sys.path)
    # The ranks are spawned with this sys.path; the parent never imports torch or deepinfer.
    sys.path.append(str(FAKE_MODULES))

    from deepinfer_backend import DeepInferBackend  # noqa: PLC0415
    from null_backend import allow_ptrace  # noqa: PLC0415

    allow_ptrace()  # so load.py's py-spy (a sibling, not an ancestor) may attach

    from inferencekey.backend.serve import serve_backend  # noqa: PLC0415

    serve_backend(DeepInferBackend, args.port, build_config(args))


if __name__ == "__main__":
    main()
