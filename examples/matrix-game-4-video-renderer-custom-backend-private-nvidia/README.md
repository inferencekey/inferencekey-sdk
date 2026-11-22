# Matrix-Game 4.0 video renderer on a custom backend (private NVIDIA GPU)

**What it deploys:** [Matrix-Game 4.0](https://huggingface.co/Skywork/Matrix-Game-4.0)
by Skywork AI (Apache-2.0), a causal video renderer. A real game engine runs
the game (physics, input, rules) and draws plain low-poly frames; the model
repaints that stream, four frames per chunk, into detailed 480 x 864 video.
Each scene has its own 5B-parameter BF16 checkpoint built on
Wan2.2-Fun-5B-Control. This example serves the **Medieval Castle** scene with
the authors' own inference source (not Diffusers), wrapped in a **custom
Python backend** (`backend.py`, slug `matrix-game-4`).

**How a request works:** the client uploads a clean low-poly video (no HUD) in
1 MB parts, then asks for a render with a preset. The answer is the rendered
MP4 (in 2 MB parts with a SHA-256), the middle frame as JPEG, timings
(model frames per second, per-chunk milliseconds) and peak GPU memory.

Presets (the authors' names): `realtime` (TAEHV codec, one denoising step,
fused DiT, CUDA graphs), `rtx5090` (the same with FP8 linears, for Blackwell
cards), `rtx5080` / `rtx5070ti` (prompt encoded first and the 11 GB text
encoder freed before the DiT loads, for 16 GB cards), and `quality` (Wan2.2
VAE, two denoising steps). Switching preset reloads the engine (about 55 to
70 s).

## What we measured (2026-11-22, one RTX 5090 32 GB)

Input: the authors' own scripted 10 s castle test take (480 frames at
48 fps, recorded with their game, `--offline-test`). Numbers are model time
for the whole clip, measured by the backend.

| Preset | Frames per second | Median chunk (4 frames) | Peak on the card |
| --- | --- | --- | --- |
| `rtx5080` (FP8 linears, text encoder freed) | 128.0 | 31.2 ms | 9,900 MiB (11,632 while loading) |
| `rtx5090` (FP8 linears) | 100.9 | 39.6 ms | 12,006 MiB (21,690 while loading) |
| `realtime` (BF16, fused DiT) | 38.0 first run, 46.6 and 47.2 later | 85 to 105 ms | 14,408 MiB (21,558 while loading) |
| `quality` (Wan VAE, 2 steps) | 10.2 | 396 ms | 22,437 MiB |

- **Ready 116 s after `ensure()`** on a fresh worker: venv about 25 s, then
  `setup()` 91 s (source 1.3 s, 24.2 GB of resources downloaded and
  SHA-256-checked in 22.8 s, engine load and warm-up 63.7 s).
- A 10 s render through the endpoint: 22 s round trip with `realtime`
  (upload not included). From the dashboard's image tester, `realtime 2s`
  answered in 4.4 s.
- **Memory caps on the same card** (PyTorch `set_per_process_memory_fraction`,
  so these emulate smaller cards, they are not real ones): `realtime` and
  `quality` both work under a 22.5 GiB cap (a 24 GB card); `rtx5080` works under
  14.5 GiB (a 16 GB card) at the same speed; `realtime` under 14.5 GiB and
  `rtx5080` under 10.5 GiB (a 12 GB card) run out of memory while loading (the
  umT5 text encoder alone is 11 GB in BF16).

## Run

```bash
cp .env.example .env          # fill in real values
# the SDK: local build of this repo, see ../CONTRIBUTING.md; the client also needs requests
python publish.py             # once: uploads backend.py as a custom backend
python main.py --input reference.mkv --preset realtime
```

`--input` is a clean low-poly RGB video of the castle scene. Record one with
the authors' game (github.com/SkyworkAI/Matrix-Game, `Matrix-Game-4/`):
`bash scripts/build_game.sh && bash scripts/play.sh --scene castle`, F9 starts
and stops a take, and `reference.mkv` in the take folder is the clean stream
(not `ui.mkv`). Headless, `MedievalCastleBBoxMap --offline-test --scene castle`
records the authors' scripted 10 s test take on Xvfb with software OpenGL in
about 25 s.

Output in `./outputs`: `<input>-<preset>.mp4` (864 x 480, same frame rate as
the input) and a `.json` with the timings and memory.

## The request

`POST {base}/endpoint/{project}/matrix-game-castle/v1/images/generations` with the data-plane key:

```json
{"model": "default", "upload_id": "any-id", "part": 0, "parts": 10, "chunk_b64": "<1 MB of the video>"}
{"model": "default", "input_id": "c3dc2cd99355f2ac", "preset": "realtime", "seed": 0, "seconds": 10}
```

- The last upload part answers `{"status": "uploaded", "input_id"}`.
- Render options: `preset`, `seed`, `seconds` or `max_frames`,
  `vram_limit_gib` (cap memory, weights included, to emulate a smaller card),
  `wait_s`. From the dashboard's image tester, type words as the prompt, for
  example `realtime 2s`: it renders the last uploaded input.
- Answer: `{"status": "running", "clip_id"}` if it takes longer than about
  270 s; poll with `{"clip_id": "..."}`. When done: `status: "done"`, the MP4 in
  2 MB parts (`mp4_b64` is part 0, then `{"clip_id", "part": n}`; check
  `mp4_sha256`), `data[0].b64_json` (the middle frame), `timings`,
  `peak_reserved_mib`, `card_peak_mib`.

## Requirements

- An NVIDIA GPU with a driver for **CUDA 13.2 or newer** (PyTorch 2.12.0+cu132,
  the authors' baseline). The `rtx5090` / `rtx5080` presets need FP8-capable
  tensor cores; we ran them on an RTX 5090.
- About 35 GB of local disk (25 GB of resources plus the venv). Keep
  `MG4_HOME` on a local disk, not a network volume.
- `git` on the worker (the backend fetches `Matrix-Game-4/` at a pinned
  commit with a sparse checkout; without git it falls back to the GitHub
  tarball).

## Notes

- Request bodies are limited to 2 MB, so the client uploads in 1 MB parts
  (base64 adds a third). Parts of 2 MB were refused with HTTP 413.
- One render at a time per worker; a second request while one runs gets
  HTTP 400 `busy` (poll the first with `clip_id`).
- The model is specialized to the low-poly look of the released scenes:
  unseen geometry or another game's frames are out of distribution.
- Licenses: Matrix-Game 4.0 weights and code Apache-2.0 (Skywork AI); umT5 and
  Wan2.2 VAE Apache-2.0 (via Wan2.2-Fun-5B-Control); TAEHV MIT (Ollin Boer Bohan).
