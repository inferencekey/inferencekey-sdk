# Haru-4B full-duplex voice on a custom backend (private NVIDIA GPU)

**What it deploys:** [Haru-4B](https://huggingface.co/higurashi-ai/Haru-4B) by
higurashi (Apache-2.0), a **full-duplex** spoken dialogue model on a
Qwen3.5-4B backbone: it listens and speaks at the same time, one 133 ms frame
at a time (7.5 Hz), and decides by itself when to talk, when to say "mm-hm"
and when to stay quiet. A text character card sets who it is and a 3 s to
3 min clip of one speaker sets its voice. Nine main languages: English,
Spanish, Portuguese, German, French, Chinese, Japanese, Russian and Korean.
One 10.5 GB bf16 file. It is a **pretrained base model**: it chats naturally,
but it is not an assistant and not a reliable source of facts (the authors
say so, and our test agrees).

The backend (`backend.py`, slug `haru-4b-full-duplex`) wraps the authors'
PyTorch package (`haru-speech` 0.1.1 on PyPI, from
[github.com/higurashi-ai/Haru](https://github.com/higurashi-ai/Haru)) and its
`DuplexSession`, with the authors' Triton audio codec when the GPU supports it.

**How a request works:** the platform takes HTTP requests, so a call carries
one recorded caller clip (one turn or several). The backend feeds it to the
model frame by frame, exactly like a live line, keeps the "microphone" open
for `tail_seconds` of silence (8 by default) and returns the model's side of
the call, a two-channel copy (caller left, model right, one clock), the text
the model wrote with its timing, how long each 133 ms frame took on the GPU,
and when the model started to talk after the caller's last word. It is not a
live microphone stream: the platform has no WebSocket passthrough yet.

Step-by-step guide, self-hosted or on InferenceKey:
[How to deploy Haru-4B](https://docs.inferencekey.com/en/deploy/haru-4b/).

## What we measured (2026-11-23)

Inputs: synthetic callers (English: Pocket TTS stock voice `anna`; Spanish: a
Gemini TTS prebuilt voice), voices given to the model from synthetic clips
(6.3 s English, 7.4 s Spanish). Timings are measured by the backend on the GPU.

| Card | Time per 133 ms frame (median) | Real-time margin | Peak on the card | Ready after `ensure()` |
| --- | --- | --- | --- | --- |
| RTX 4090 24 GB | 44 to 48 ms | 2.8 to 3.1x | 11,248 MiB | 76.2 s (fresh worker) |
| RTX 2000 Ada 16 GB (70 W) | 57 ms | 2.3x | 10,920 MiB | 84.7 s (fresh worker) |

- No frame took longer than 133 ms in any of 41 calls (worst single frame:
  74.6 ms with the Triton codec, 87.8 ms without).
  With `pace: realtime` (one frame every 133 ms, like a live call) there were
  0 late frames on both cards.
- **12 GB is enough:** with PyTorch capped at 11 GiB (to emulate a 12 GB card;
  an emulated cap, not a real 12 GB card) the model gave the same words with
  the same seed at the same speed on both cards. Capped at 9 GiB (a 10 GB card)
  it runs out of memory while loading: the weights alone are 9.92 GiB.
- Without the authors' Triton codec (`HARU_CODEC_BACKEND=torch`) a frame takes
  64 to 67 ms on the RTX 4090.
- After a greeting or a question, the model started talking 0.12 to 1.13 s
  after the caller's last word. After a 16 s story it waited 1.8 to 4.9 s and
  said little. Spanish replies were natural and on topic. Asked for the capital of
  Australia it stayed silent, changed the subject, or said "Sydney"; once, in
  a longer call, "is it Canberra?".

## Run

```bash
cp .env.example .env          # fill in real values
# the SDK: local build of this repo, see ../CONTRIBUTING.md; the client also needs requests
python publish.py             # once: uploads backend.py as a custom backend
python main.py caller.wav my-voice.wav                 # English card
python main.py llamada.wav mi-voz.wav --language es    # Spanish card
```

`caller.wav` is what the other person says (any rate, mono or stereo).
`my-voice.wav` is 3 s to 3 min of ONE speaker: the voice Haru speaks with.
Haru ships without voices. Only use a voice you have the right to use.

Output in `./outputs`: `call-<lang>.wav` (the model, 24 kHz),
`call-<lang>.conversation.wav` (caller left, model right) and a `.json` with
the text, the per-frame timing and the GPU memory.

## The request

`POST {base}/endpoint/{project}/haru-4b/v1/chat/completions` with the
data-plane key (the chat route of a `text2audio` workload; `main.py` turns it
on with a dashboard token, or turn on "Chat" in Endpoint settings):

```json
{"model": "default",
 "messages": [{"role": "user", "content": [{"type": "input_audio",
   "input_audio": {"data": "<base64 WAV of the caller>", "format": "wav"}}]}],
 "extra_body": {"voice_clip_b64": "<base64 WAV, 3 s to 3 min of one speaker>",
                "voice": "leo", "language": "es",
                "character": {"persona": "...", "name": "Marcos", "gender": "male",
                              "other_name": "Lucía", "relationship": "...", "situation": "..."},
                "tail_seconds": 8, "seed": 7}}
```

The answer's message content is a JSON string: `audio_b64`,
`conversation_b64`, `format`, `text`, `words` (`[{"t": seconds, "text"}]`),
`timings` (`frame_ms_median`, `frame_ms_p95`, `frame_ms_max`,
`realtime_factor`, `frames_over_133ms`, `model_starts_after_caller_s`,
`model_talks_while_caller_talks_s`, ...), `gpu_used_mib_peak`.

Options: `voice` (keep the clip under a name; later calls can send just the
name), `language` (card language), `character` or `system_prompt` (a ready
card), `tail_seconds` (0 to 30), `seed`, `temperature` (0.7), `top_k` (25),
`guidance_scale` (1 to 2, 1.5), `eop_bias` (> 0 makes it start talking sooner),
`speak_first`, `pace` (`offline` default, or `realtime`), `audio_format`
(`wav` or `flac`).

**Long calls:** a result travels in one platform message of at most 8 MB. A
bigger one is dropped and the caller sees a 504 `inference_timeout` (we hit
it with a 38 s clip plus 8 s of tail). The backend therefore switches to FLAC
when the WAVs would not fit, leaves out the two-channel copy if they still do
not, and returns a clear error if the reply alone is too big.

## Settings (`HARU_*` environment variables, passed through by `main.py`)

| Variable | Default | What it does |
| --- | --- | --- |
| `HARU_CODEC_BACKEND` | `auto` | `auto` (Triton codec when usable), `triton` or `torch` |
| `HARU_GPU_MEMORY_GB` | `0` (no cap) | Caps the GiB PyTorch may allocate, to check a smaller card |
| `HARU_HF_REVISION` | `8d6e83bc...` | Hugging Face revision of the weights |
| `HARU_CACHE_DIR` | `~/.cache/haru-backend` | Where the 10.5 GB checkpoint goes (keep it on a local disk) |

## Notes

- One conversation at a time per worker (the backend holds a lock); a second
  request waits for the first.
- transformers warns that its Gated DeltaNet kernels fall back to PyTorch
  (`flash-linear-attention` and `causal_conv1d` not installed). With
  `flash-linear-attention` 0.6.0 added by hand to the backend's venv, a frame
  took 38 to 41 ms instead of 44 to 48 ms on the RTX 4090 (430 MiB more on the
  card). It is not in `requirements.txt`.
- The authors also ship a vLLM-Omni plugin (`haru-speech-vllm`) for many
  conversations at once; this example does not use it.
