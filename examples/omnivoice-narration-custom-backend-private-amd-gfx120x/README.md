# OmniVoice narration (private, AMD gfx120x): voice-over with saved voices through a custom backend

A **text2audio** custom backend for the InferenceKey SDK that wraps
[k2-fsa/OmniVoice](https://github.com/k2-fsa/OmniVoice), a zero-shot TTS model
for 600+ languages (Apache-2.0, ~0.8B parameters). It narrates text with a
**saved voice** shipped inside the package, so every request sounds like the
same narrator, and it can also design a voice on the fly from a short
description. The one axis it varies vs the other examples: a TTS custom backend
pinned to a private **AMD R9700 (gfx120x)** box.

## Compatibility
- SDK: local build in this repo (or the published `inferencekey` wheel)
- Language: Python >= 3.9 for the `main.py` client. The **backend** declares
  `requires_python=">=3.10,<3.13"`, so the worker provisions a matching interpreter.
- Placement: private (AMD R9700 / gfx120x; NVIDIA works too)
- Backend: custom (`omnivoice-narration`), text2audio   Policy: fixed

## Prerequisites
- An SDK token (`ik_sdk_`), a data-plane API key (`ik_live_`), your project slug
  and your worker's id (`IK_WORKER_ID`). See [tokens & placement](../README.md#tokens).
- About **4 GB of free VRAM** (measured: 2.2 GB after loading in float16,
  3.4 GB while generating 30 s of audio).
- The backend **published** to your tenant: `python publish.py` (needs
  `INFERENCEKEY_TENANT_ID` and a dashboard session token), or upload
  `dist/omnivoice-narration-<version>.tar.gz` from Dashboard -> Custom backends.

## Run
```bash
cp .env.example .env        # fill in real values
# install the local SDK, see ../CONTRIBUTING.md#4-depending-on-the-sdk
python publish.py           # once per backend version
python main.py "Text to narrate"
```

## What it does
1. **`ensure()`** reconciles a private workload pinned to `IK_WORKER_ID`, with
   `backend="omnivoice-narration"` (the custom backend's slug) and
   `task_type="text2audio"`.
2. It turns on the workload's **chat** endpoint (text2audio workloads start
   with `audio_speech` and `jobs` only).
3. **`wait_until_ready()`** blocks while the cold worker builds the backend's
   venv (ROCm `torch` + `omnivoice`), downloads ~3 GB of weights at a pinned
   revision and loads the model once.
4. **Call**: a POST to `…/endpoint/{project}/{slug}/v1/chat/completions` whose
   last user message is the text. The reply's message content is the backend's
   JSON; `main.py` decodes `audio_b64` into `outputs/narration.mp3`.

### Request options (in `extra_body`)

| Field | Default | Meaning |
|---|---|---|
| `voice` (or `voz`) | `voz1` | a saved voice, a file in `backend/voices/`: `voz1` or `voz2` (two deep narrator voices: `voz1` has the epic movie-trailer tone, `voz2` suits dynamic texts such as demos and explainers; `narrador_trailer` is an alias of `voz1`) |
| `instruct` | none | design a voice instead, e.g. `"male, elderly, very low pitch"` |
| `speed` | `0.85` | speaking rate (`< 1` slower) |
| `language` | `es` | language of the text |
| `response_format` | `wav` | `wav`, `flac` or `mp3` |
| `pause_ms` | `350` | silence between sentence chunks |
| `num_step` | `32` | diffusion steps (16 is faster) |
| `seed` | random | reproducible output |

Long texts are split into sentence chunks (about 280 characters each) and
joined with a short pause. One request accepts up to 6000 characters, and the
encoded audio must stay under ~5.5 MB because results travel back over NATS:
WAV fits about a minute, **use `mp3` for longer narrations**.

### Adding a voice
A saved voice is an OmniVoice `VoiceClonePrompt` (a few KB of audio tokens):
```python
prompt = model.create_voice_clone_prompt(ref_audio="ref.wav", ref_text="What ref.wav says.")
prompt.save("backend/voices/my_voice.pt")
```
Bump `VERSION` in `publish.py` and `version` in `backend.py`, publish again and pass `"voice": "my_voice"`.
A running workload keeps the version it started with: re-create its assignment (or restart the worker) so it loads the new package.
Only clone voices you have the right to use.

## Why the chat route and not `/v1/audio/speech`
The worker relays a custom backend's result as JSON. `/v1/audio/speech` expects
raw audio bytes, which custom backends cannot return yet, so it answers 500.
Use the chat route (or `/jobs`) and decode `audio_b64`.

## Cost & cleanup
The workload holds **one fixed replica** on your GPU until you delete it.
`main.py` deletes it in a `finally` block; set `IK_KEEP_ALIVE=1` to keep it.
