# Ming-3.0-tts-0.5B speech (custom backend, private NVIDIA)

Serves [Ming-3.0-tts-0.5B](https://huggingface.co/inclusionAI/Ming-3.0-tts-0.5B)
from inclusionAI (Ant Group, MIT), a dense 0.5B text-to-speech model (Qwen2.5-0.5B
backbone, a DiT patch head and an audio VAE, 44.1 kHz). One checkpoint of about
3.0 GB does zero-shot voice cloning, **voice design from a text description**,
free-form delivery instructions, built-in voices, emotion / speed / pitch /
volume, **14 inline paralinguistic tags** (`[Laughter]`, `[Sigh]`, `[Cough]`,
`[Uhm]`, `[Question-huh]`...) and seven languages (Chinese, English, German,
Spanish, French, Korean, Japanese). No built-in backend serves it, so this
example ships a **custom Python backend** that drives the authors' own Python
API (`ming_tts.MingTTS`, source fetched at a pinned commit), publishes it to your
tenant and deploys it on your NVIDIA worker.

## Compatibility
- SDK: local build in this repo (see [CONTRIBUTING](../CONTRIBUTING.md#4-depending-on-the-sdk))
- Language: Python >= 3.9 for the client scripts; the backend runs on CPython
  3.10 to 3.12 (the worker provisions it with `uv`; torch 2.6.0, transformers 4.46.3)
- Placement: private NVIDIA, driver for CUDA 12.4+. About 2.9 GB of GPU memory at
  peak; it still ran with PyTorch capped at 3 GiB (and failed to load at 2.75 GiB)
- Backend: custom (`ming-3-tts-05b`)   Policy: `fixed`

## What we measured (2026-10-28)
| | RTX 2000 Ada 16 GB | RTX 4090 24 GB |
| --- | --- | --- |
| Ready after `ensure()`, empty caches | 60 s | 47 s |
| GPU memory, peak allocated | 2.9 GB | 2.9 GB |
| Speed (seconds of audio per second of compute) | 0.61x to 0.68x | 0.75x to 0.77x |
| 20.7 s English paragraph | 31 to 32 s | 27 s |

The model is generated step by step with a small diffusion head per step, and
on both cards the GPU sat at about 20 to 30 % utilisation: the Python decode
loop, not the card, sets the speed, so a bigger GPU barely helps and it runs
slower than real time on both. Use it for offline narration and voice-overs,
not for live voice agents. Outputs were identical (same file, byte for byte)
with PyTorch capped at 8, 4 and 3 GiB.

## Prerequisites
- An SDK token (`ik_sdk_`) and an API key (`ik_live_`) for the project, see
  [Tokens](../README.md#tokens).
- A registered NVIDIA worker assigned to the project (`IK_WORKER_ID`).
- Your tenant id and a dashboard session token: `publish.py` uploads the backend
  with them, and `main.py` uses the session token to turn on the workload's
  `chat` endpoint. Without them, upload `dist/*.tar.gz` from Dashboard -> Custom
  backends and flip "Chat" in the workload's Endpoint settings.

## Run
```bash
cp .env.example .env        # fill in real values
python -m venv .venv && . .venv/bin/activate
pip install maturin requests
( cd ../../bindings/python && maturin develop --release )
python publish.py           # once: packages backend.py and uploads it
python main.py              # ensure -> wait until ready -> 4 WAV files in ./outputs
```

`main.py` needs no audio of your own: a built-in voice (`UK_George`) reads a
line, a new voice is designed from a text description, that voice is cloned to
say a line with `[Laughter]` and `[Sigh]`, and the same cloned voice speaks
Spanish.

## Request shape
The backend answers the OpenAI chat route, `/v1/chat/completions`. The last user
message is the text to say; the options go in `extra_body`:

```json
{"model": "default",
 "messages": [{"role": "user", "content": "Ahoy there! The storm passed hours ago."}],
 "extra_body": {"task": "VOICE_DESIGN",
                "style": "A gravelly old sea captain in his seventies, amused and a little hoarse",
                "seed": 1895}}
```

To clone a voice, send the reference as an `input_audio` part
(`{"type": "input_audio", "input_audio": {"data": "<base64 WAV>", "format": "wav"}}`)
and its transcript as `prompt_text`. For Spanish, German, French, Korean or
Japanese use `"task": "MULTILINGUAL"` and `"language_id": "es"` (and so on).

| `task` | Options |
| --- | --- |
| `TTS` | reference audio + `prompt_text` (cloning); inline tags in the text |
| `MULTILINGUAL` | `language_id` (`es`, `de`, `fr`, `ko`, `ja`) + reference audio + `prompt_text` |
| `VOICE_DESIGN` | `style`: a description of the voice |
| `FREEFORM` | `instruction`: how to say it ("Whisper, tense and frightened") |
| `IP` | `voice`: a built-in voice (`UK_George`, `US_Emma`, `UK_Charlotte`...) |
| `EMOTION`, `BASIC`, `DIALECT` | the authors' controls (`emotion`, `speed`/`pitch`/`volume`, `dialect`), see their [cookbook](https://github.com/inclusionAI/Ming-3.0-tts/blob/main/cookbooks/instruction_en.md) |

When `task` is missing the backend picks it from the other fields. The message
content of the answer is the backend's JSON (as a string): `audio_b64` (16-bit
WAV, 44.1 kHz mono), `seconds`, `task` and `timings` (`gpu_s`,
`realtime_factor`, `peak_vram_mib`).

## Things we hit
- **Numbers in English**: the authors' text normalizer turns the hyphen in
  "twenty-one" into a pause, so "4521" was spoken as "four thousand five
  hundred twenty ... one", and "555-0142" as "five hundred fifty-five minus one
  hundred forty-two". Spanish text skips the normalizer and "1.249,99 euros"
  came out garbled. Write important numbers as you want them read.
- **Paralinguistic tags** made audible sounds (the clips with tags were 1.3 to
  1.7 s longer than the same lines without them; Whisper heard the `[Uhm]` and
  the `[Question-huh]`), but there is no objective check for a laugh.
- The authors' Chinese voice-design example took 50 s of compute for 5 s of
  audio on the small card: one text fragment never hit the stop token and ran
  to the decode limit before the silence was trimmed.
- `/v1/audio/speech` does not carry a custom backend's output yet, so the
  example uses the chat route.

## Licence and responsible use
Code and weights are MIT. Only clone voices you have permission to use, and
label synthetic speech as synthetic where it could be mistaken for a real
recording (the authors' usage disclaimer).
