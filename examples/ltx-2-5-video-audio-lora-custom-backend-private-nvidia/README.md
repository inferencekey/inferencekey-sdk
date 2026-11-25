# LTX-2.5 video with sound, official LoRAs and a LoRA training check, on a custom backend (private NVIDIA GPU)

**What it deploys:** [LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5) by
Lightricks, a model that makes a video clip **and its soundtrack** (speech,
foley, ambience) in one pass, from a prompt or from a prompt plus a first
image. This example runs the authors' own `DistilledPipeline`
([github.com/Lightricks/LTX-2](https://github.com/Lightricks/LTX-2),
`ltx_pipelines.distilled`, Apache-2.0 code) inside a **custom Python backend**
(`backend.py`, slug `ltx-2-5`), one subprocess per request, so an
out-of-memory error ends that request and never the backend.

Files it loads (about 66 GiB, 71.3 GB): the 22B distilled transformer in bf16
(cast to FP8 on the fly with `--quantization fp8-cast`), the Gemma 4 12B text
encoder fine-tuned by Lightricks, the convolutional video VAE (no `natten`
needed), the audio VAE, the x2 latent spatial upscaler, and the official
**Cinemagraph LoRA**. It can also load a LoRA you trained, and start a short
training run with the authors' `ltx-trainer` to check that fine-tuning works on
your card.

## What we measured (2026-11-25)

One prompt, 1536 x 1024, 121 frames (5 s at 24 fps) with AAC audio, all model
loads included (the pipeline loads and frees each part per request):

| Card | Setting | Seconds per clip | Peak on the card |
| --- | --- | --- | --- |
| L40S 48 GB | `fp8-cast` | 62.8 to 66.2 | 25,722 MiB |
| L40S under a 31.5 GiB memory cap (emulated 32 GB card) | `fp8-cast` | 64.1 | 25,722 MiB |
| L40S under a 23.5 GiB cap (emulated 24 GB card) | `fp8-cast` | out of memory loading the bf16 text encoder | - |
| L40S under 23.5 / 15.5 / 11.5 GiB caps | `fp8-cast` + `--offload cpu` | 111.2 to 112.2 | 8,978 MiB |
| L40S under a 7.5 GiB cap | `fp8-cast` + `--offload cpu` | out of memory | - |
| **real RTX 4090 24 GB** | `fp8-cast` + `--offload cpu` | **113.2** | **8,895 MiB** |

- Capped runs on the L40S gave bit-identical videos (ffmpeg PSNR inf); the
  RTX 4090 gave a different clip from the same seed (other GPU numerics), same
  scene and the same spoken line.
- **Ready 87 to 110 s after `ensure()`** on a fresh worker: venv about 26 s,
  then 71.3 GB downloaded and SHA-256-checked in 58 to 78 s.
- A 768 x 512, 49-frame preview: 28 s.
- Official Cinemagraph LoRA (image-to-video, strength 1.1): 64 s at
  1536 x 1024. On a cup close-up the frame holds (mean pixel change vs frame 0:
  8.1 at frame 120) where the same prompt without the LoRA drifts (36.3).
  Moving people are out of its scope (the authors say so, and it moved our
  barista).
- `ltx-trainer` smoke test (`{"train": ...}`), on 4 of our clips at
  576 x 384 x 49 under the 31.5 GiB cap, with the authors'
  `t2v_lora_low_vram.yaml` (INT8 transformer, 8-bit Gemma, rank 16): trainer
  install 8 s, preprocessing 85 s, 30 steps at 0.43 steps/s (about 2.3 s per
  step), trainer-reported peak 25.42 GB. The checkpoint loads back as a LoRA.
  This checks that training starts on a 32 GB budget; 30 steps is not a
  finished LoRA.

## Run

```bash
cp .env.example .env          # fill in real values
# the SDK: local build of this repo, see ../CONTRIBUTING.md; the client also needs requests (and Pillow for --image)
python publish.py             # once: uploads backend.py as a custom backend
python main.py                # text to video with sound, 1536 x 1024, 5 s
python main.py --offload cpu  # a 24 GB card (or 12 to 16 GB): about 1.7x slower
python main.py --image shot.jpg --lora cinemagraph --lora-strength 1.1 \
  --prompt "CINEMAGRAPH_MOTION, tripod locked-off static camera, zero camera movement, ... only the steam moves ..., seamless natural loop"
```

Output in `./outputs`: `<name>.mp4` (H.264 + AAC 48 kHz stereo) and a `.json`
with the timings, memory and the last lines of the pipeline log.

## The request

`POST {base}/endpoint/{project}/ltx-2-5-video/v1/images/generations` with the data-plane key:

```json
{"model": "default", "prompt": "A barista ... says: 'Fresh roast, every morning.' We hear ...",
 "width": 1536, "height": 1024, "num_frames": 121, "seed": 42,
 "quantization": "fp8-cast", "offload": "none"}
```

- Options: `image_b64` (first frame, PNG/JPEG), `image_strength`, `lora`
  (`cinemagraph`, or a file name in `LTX_HOME/loras`), `lora_strength`,
  `frame_rate`, `vram_limit_gib` (caps PyTorch memory to emulate a smaller
  card), `wait_s`. `width`/`height` multiples of 64, `num_frames` 8k+1.
- Put the dialogue and the sounds in the prompt: the audio is generated from it.
- Answer: `{"status": "running", "clip_id"}` if it takes longer than about
  270 s; poll with `{"clip_id": "..."}`. When done: `status: "done"`, the MP4
  in 2 MB parts (`mp4_b64` is part 0, then `{"clip_id", "part": n}`; check
  `mp4_sha256`), `data[0].b64_json` (the middle frame), `wall_s`,
  `peak_reserved_mib`, `card_peak_mib`, `audio`.
- Training check: `{"train": {"clip_ids": ["<ids of clips this backend
  made>"], "steps": 30, "width": 576, "height": 384, "frames": 49}}`; their
  prompts are the captions.
- From the dashboard's image tester, a plain prompt renders a 768 x 512,
  49-frame preview (`LTX_DASH_*` to change it).

## Requirements

- An NVIDIA GPU with FP8 (Ada or newer: RTX 4090, L40S, RTX 5090...) and a
  driver for **CUDA 13.2 or newer** (PyTorch 2.13.0+cu132).
- About 85 GB of local disk (71.3 GB of weights, the venv). Keep `LTX_HOME`
  on a local disk, not a network volume. With `--offload cpu` the transformer
  waits in system RAM: we used pods with 124 and 755 GB.
- `git` on the worker only for the training check.

## Notes

- **The weights are gated on Hugging Face.** Accept the terms on
  [Lightricks/LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5) and the
  Cinemagraph repo, and pass `HF_TOKEN` (a read token) in `.env`: the backend
  then downloads from `Lightricks/*`. Without a token it fetches the same files
  from public copies on Hugging Face (`FILES` in `backend.py`), and either way
  it checks every file against the SHA-256 Lightricks publishes, so a modified
  copy is refused. We ran without a token, from the copies.
- torch 2.13.0 pins cuDNN 9.20.0.48; the LTX-2 authors override it to
  9.24.0.43 (a system cuDNN in `/usr/lib`, 9.12 on our pods, otherwise clashes).
  `setup()` applies that override because a `requirements.txt` cannot.
  `torchaudio` 2.11.1 needs torch 2.13 (with 2.12.1 it fails with
  `libtorchaudio.so: undefined symbol`).
- Request bodies are limited to 2 MB: `main.py` re-encodes a big first frame
  as JPEG (a 2.0 MB PNG was refused with HTTP 413).
- One task at a time per worker; a second request while one runs gets HTTP
  400 `busy` (poll the first with `clip_id`).
- **License.** The code here and the LTX-2 code are Apache-2.0. The LTX-2.5
  weights, the official LoRAs and any LoRA you train on them are under the
  [LTX-2.x Community License](https://github.com/Lightricks/LTX-2/blob/main/LICENSE-2_x):
  free for entities under 10 million USD of annual revenue (affiliates
  counted); above that, any production use, including fine-tuning for
  commercial use, needs a paid license. Lightricks claims no rights in your
  outputs; keep their watermarking/provenance features and the AI disclosure
  rules. Read it before you use the outputs commercially.
