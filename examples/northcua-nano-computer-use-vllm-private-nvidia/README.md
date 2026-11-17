# NorthCUA-Nano: a computer-use model on one GPU (vLLM)

**NorthCUA-Nano** from Beipo AI is a 9B dense **computer-use** model built on
Qwen3.5-9B (32 layers, three linear-attention layers for every full-attention
layer, a vision encoder), released under **Apache-2.0** with 18.8 GB of BF16
weights. It takes a request, a screenshot and the history of what it already
did, and answers with its reasoning, a short status line and **GUI tool calls**
(`click`, `type`, `keypress`, `scroll`, `drag`...) whose coordinates live in a
**0 to 1000 space on both axes**, not in pixels. The authors report 78.9 % on
OSWorld-Verified and 72.9 % on ScreenSpot-Pro (their numbers, not reproduced
here). The family also has a 35B-A3B Mini and a 397B-A17B Pro.
[beipo-ai/NorthCUA-Nano](https://huggingface.co/beipo-ai/NorthCUA-Nano)

This example serves it with the built-in **`vllm`** backend on one private
NVIDIA GPU and talks to it the way the official
[NorthCUA Harness](https://github.com/beipo-ai/NorthCUA) does: same system
prompt (rendered for `normalized_1000`), same task and observation messages,
same native desktop tool schemas (copied in `prompts/`, Apache-2.0).

| File | What it does |
| --- | --- |
| `main.py` | `ensure()` the workload on your worker, wait until ready, send one screenshot of a mail client (`samples/mail.png`) with the task "Open the email from Priya Shah about the Q3 invoice", print the tool call, map it to pixels and draw it (`outputs/click.png`), `delete()` unless `IK_KEEP_ALIVE=1`. |
| `browser_agent.py` | A small agent loop on a real Chromium page (Playwright): screenshot, call the endpoint, run the tool calls, repeat until the model answers in text. Saves every step with its clicks drawn on it. A stand-in for the Harness, enough to watch the model work. |
| `northcua_step.py` | Builds the Harness-style request and maps `0..1000` to pixels. |
| `prompts/` | The Harness Desktop prompt templates and the 10 desktop tool schemas. |
| `samples/` | Three small local web apps (mail, expense form, settings dialog) and a screenshot. |
| `.env.example`, `requirements.txt` | Variables and client-side dependencies. |

```bash
cp .env.example .env            # fill in real values
python main.py                  # deploy + one step on a screenshot
IK_KEEP_ALIVE=1 python main.py  # keep the endpoint, then:
python -m playwright install chromium
python browser_agent.py samples/settings.html \
  "Turn on dark mode, set the language to Español and save the changes."
```

## Use a 1920x1080 screen

The one setting that mattered most in our test: **give the model 1920x1080
screenshots**. At 1280x800 its reasoning named the right control in pixels
(for example "the toggle is at (976, 278)") and then sent `click(508, 257)`:
976/1920 and 278/1080, so it normalised against 1920x1080 even though the
prompt states 1280 x 800. The same settings task failed in 12 steps (209 s)
at 1280x800 and passed in 5 steps (29 s) at 1920x1080. `browser_agent.py`
uses 1920x1080 by default (`--width`, `--height`).

## Compatibility

- **vLLM 0.33** (`vllm_version="0.33.0"`) serves `Qwen3_5ForConditionalGeneration`
  natively; it installs torch built for **CUDA 13** (driver 580 or newer).
- Flags: `--reasoning-parser qwen3` (thinking apart from the answer),
  `--enable-auto-tool-choice --tool-call-parser qwen3_coder` (the model's
  `<tool_call><function=...>` format becomes OpenAI `tool_calls`),
  `--limit-mm-per-prompt '{"image": 4, "video": 0}'`, 32K context.
- **BF16 (default)**: 17.6 GiB of weights after load; a **24 GB card**.
- **FP8 (`NORTHCUA_FP8=1`)**: vLLM's online FP8 quantization, 10.7 GiB of
  weights, for **16 GB** cards. On Ampere it is weight-only FP8 (Marlin).

## Measured (2026-11-17)

All runs through the InferenceKey platform (`ensure()` + the OpenAI-compatible
endpoint), vLLM 0.33.0, 32K context, the Harness prompt and tools, local web
apps rendered by Playwright. "Targets hit" is one try per target, temperature
0: the first click lands inside the button or field. Agent runs use
temperature 0.7.

| Card | Weights | Ready (cold) | Peak GPU memory | Targets hit, 1920x1080 | Agent tasks at 1920x1080 |
| --- | --- | --- | --- | --- | --- |
| A40 48 GB | BF16, util 0.90 | 470 s | 40.2 GB | 15 of 18 (11 of 18 at 1280x800) | settings 5 steps 28.7 s |
| 24 GB budget on the A40 (emulated) | BF16, util 0.52 | 87 s (warm) | 22.7 GB | 12 of 13 (mail + settings) | settings 8 steps 101.4 s |
| **RTX PRO 4000 Blackwell 24 GB** | BF16, util 0.90 | 635 s | 22.1 GB | 12 of 13 | settings 5 steps 32.6 s, expenses 5 steps 36.6 s |
| 16 GB budget on the A40 (emulated) | FP8 (Marlin, weight-only), util 0.35 | 194 s (warm) | 14.8 GB serving, **18.1 GB while loading** | 12 of 13 | settings 5 steps 28.4 s |
| **RTX 5080 16 GB** | FP8 (native), util 0.90 | 726 s | 15.2 GB | 12 of 13 | expenses 5 steps 25.3 s; settings saved at step 4 |

| What | Value |
| --- | --- |
| One agent step | median 3.5 to 5.4 s at 1920x1080 (screenshot in, action out); the first step of a task 7 to 14 s |
| Decode speed, one request | about 28 to 33 tokens/s BF16 (A40, RTX PRO 4000), about 50 FP8 on the A40, about 70 FP8 on the RTX 5080 |
| Prompt size | about 6,150 tokens for the system prompt, tools and one screenshot |
| KV cache at 32K context | A40 BF16: plenty; RTX PRO 4000 BF16: 43,690 tokens; RTX 5080 FP8: 33,760 tokens (1.03x of 32K) |
| Same task, 1280x800 vs 1920x1080 (A40) | settings dialog: failed in 12 steps (209 s) vs done in 5 steps (29 s) |

The 16 GB emulation on the A40 is not enough on its own: online FP8 quantization
on Ampere peaked at 18.1 GB while loading. On the real RTX 5080 (native FP8) the
load stayed under 16 GB.

## Notes

- Non-streaming responses through the InferenceKey endpoint currently drop the
  model's thinking (`reasoning`), and a request that thinks for more than
  120 s gets a 503 from the gateway; tool calls and answers are intact.
  Streaming carries the thinking as `reasoning_content`.
- The model sometimes thinks for a long time on simple form pages (1,300 to
  3,700 tokens, 40 to 116 s in our test). Cap `max_tokens` and keep a step
  budget.
- The Harness adds what this example leaves out: sandboxed desktops, context
  compaction, retries and outcome checks. Use it for real desktops, and keep
  credentials away from what the model can see.
