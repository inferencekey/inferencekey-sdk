# SenseNova 6.8 Code2Video on llama.cpp, driven by Claude Code (private NVIDIA, 24 GB)

[SenseNova 6.8 Code2Video](https://huggingface.co/sensenova/SenseNova-6.8-Code2Video) (SenseTime, Apache-2.0) is a
Qwen3.8-27B fine-tune that makes short videos by writing code (Canvas, SVG, Three.js...), rendering it and reviewing
the frames. The authors run it inside Claude Code with their
[`code-video` skill](https://github.com/OpenSenseNova/SenseNova-6.8-Code2Video/tree/main/skills/code-video).

This example deploys the community 4-bit GGUF
([bartowski](https://huggingface.co/bartowski/sensenova_SenseNova-6.8-Code2Video-GGUF), Q4_K_M, 17.2 GB) plus the bf16
vision projector (0.93 GB, so the model can look at its own frames) with the built-in `llamacpp` backend on one
private 24 GB NVIDIA GPU, and makes one call through the Anthropic-compatible `/v1/messages` route.
`run_claude_code.sh` then points Claude Code at the same endpoint.

## Files

| File | What it does |
| --- | --- |
| `main.py` | `ensure()` the workload, `wait_until_ready()`, one streaming `/v1/messages` call that asks for a 6-second animation page; saves `outputs/clip.html` and `outputs/answer.json` |
| `run_claude_code.sh` | runs Claude Code headless in a task folder with the authors' skill against your endpoint |
| `claude-settings.example.json` | Claude Code settings (endpoint, context, output cap); copy to `claude-settings.json` |
| `.env.example` | placeholders for tokens and the worker id |

## Run

```bash
cp .env.example .env            # fill in real values
pip install -r requirements.txt # plus the inferencekey binding (see requirements.txt)
IK_KEEP_ALIVE=1 python main.py
cp claude-settings.example.json claude-settings.json   # set your project slug in ANTHROPIC_BASE_URL
mkdir -p ~/c2v-task && cp your_dense_prompt.md ~/c2v-task/dense_prompt.md
./run_claude_code.sh ~/c2v-task
```

Run Claude Code in an isolated folder or container: with `--permission-mode acceptEdits` and Bash allowed, the
agent runs shell commands without asking. It needs Node.js 18+, Python 3, ffmpeg and Playwright Chromium.

## What we measured (one RTX 4090 24 GB, 2026-11-11)

- **Ready in 25.2 s** from `ensure()` on a fresh worker: llama.cpp b12167 prebuilt CUDA 13.4 tarball, GGUF and
  projector pre-downloaded to the local disk (26 s and 3 s).
- **Decode 47.7 tok/s** on a fresh context, 44.5 at ~30K tokens, 39.4 at ~66K. Peak VRAM **21,425 MiB** with a
  98,304-token q8_0 context (only 16 of the 64 layers keep a KV cache).
- **Claude Code + code-video skill, 12-second brief, 40-minute cap**: SPEC.md after 5 min, a complete 12 s render
  (1280x720, 30 fps, passes the skill's `check_video.sh`) after 10 min, 61 tool calls and 21 frames read back as
  images. The picture was wrong (the two dot staircases met in 5 x 7 cells with a gap instead of a 5 x 6
  rectangle). Then Claude Code's auto-compaction asked the model for a summary three times; each answer ran to
  the 16,384-token output cap, which used the last 21 minutes. No `final.mp4` within the cap.
- Prompt caching works: the 44 requests spent 75 s on prefill in total and 1,915 s on decode (78,039 tokens).

Smaller cards (direct llama-server on the same 4090, 32K q8_0 context unless noted, one sample each):

| File | Peak VRAM | Decode | Our test image read right? | One-shot animation page |
| --- | --- | --- | --- | --- |
| Q4_K_M 17.2 GB | 19,233 MiB | 47.6 tok/s | yes | 8,192 tokens of thinking, no answer |
| IQ3_XXS 12.1 GB | 13,994 MiB | 63.5 tok/s | yes | renders; bug: one dot at a time |
| IQ2_XXS 8.6 GB (16K) | 10,252 MiB | 78.7 tok/s | yes | 8,192 tokens, page cut off |

These are memory peaks on the 4090, not runs on real 16 or 12 GB cards.

## Known issues

- **Long thinking.** The model thinks before every step (thousands of tokens). Use streaming calls; a
  non-streaming call that thinks past 120 s is cut by the worker's default backend timeout
  (`INFERENCEKEY_BACKEND_REQUEST_TIMEOUT_SECS`).
- **Compaction.** With a 98K context, Claude Code compacts at 80 %; the model's summary hit the output cap three
  times. A bigger context (the model supports 512K; it needs more VRAM) or a lower output cap for compaction may help.
- **Images through `/v1/messages`.** The code-video skill reviews its frames by reading PNG files, which Claude Code
  sends as image blocks. On 2026-11-11 the InferenceKey Anthropic route dropped image blocks and turned image tool
  results into text; we ran this test with a local fix that forwards them to llama.cpp as `image_url` parts (it
  worked: 21 frames reviewed). Until that fix is deployed, the model works blind on the hosted platform.
- The `model` field must be the workload slug (`sensenova-code2video`) or `default`.
