# Idol: one picture to a 3D Gaussian splat (custom backend, private NVIDIA GPU)

[Idol](https://huggingface.co/speridlabs/idol) (Speridlabs, Apache-2.0) turns one
picture of an object (a photo, a render or a generated image) into a 3D Gaussian
splat: it removes the background, frames the object and generates 32,768 to
1,048,576 Gaussians, written as a standard 3DGS `.ply`, including the sides the
picture never showed. [Idol Turbo](https://huggingface.co/speridlabs/idol-turbo)
adds two distilled phase LoRAs that do it in 5 steps without guidance instead of
50 guided steps.

No built-in backend serves it, so this example ships a **custom Python backend**
(`backend.py`) that wraps the authors' `IdolPipeline`
([idol-sdk](https://github.com/speridlabs/idol-sdk) at commit `8db7003`), loads
both modes into one process, and answers on the workload's OpenAI images route.

## What it deploys

| | |
| --- | --- |
| Model | `speridlabs/idol` (denoiser, octree decoder, Gaussian decoder, BiRefNet: about 3.1 GB) + `speridlabs/idol-turbo` LoRAs (117 MB) |
| Image encoders | DINOv3 ViT-H+/16 and the FLUX.2 VAE. The authors read them from two **gated** repos. By default (`IDOL_ENCODERS=open`) the backend takes them from Apache-2.0 repos instead, so no Hugging Face access request is needed: DINOv3 bf16 weights from `Comfy-Org/Idol` (with the model config from `onnx-community`), the FLUX.2 VAE from `black-forest-labs/FLUX.2-klein-4B`. `IDOL_ENCODERS=gated` plus `HF_TOKEN` uses the authors' repos. |
| Backend | custom, `idol-image-to-3d` (`task_type` `text2image`) |
| Policy | `fixed`, one replica on your worker (`IK_WORKER_ID`) |
| Downloads | about 5 GB in all, revisions pinned in `backend.py` |

## What we measured (2026-11-20, one request at a time, 262,144 Gaussians, seed 42)

| GPU | Ready after `ensure()` | 50 guided steps | Idol Turbo, 5 steps | Peak on the card |
| --- | --- | --- | --- | --- |
| RTX 3090 24 GB | 117 s (first start, compile included) | 40.7 s | 3.5 s | 7,308 MiB |
| RTX 3070 8 GB (real card) | 179 s | 66.9 s | 5.6 s | 7,189 MiB |
| 6 GB, emulated by a memory cap on the 3090 | | 40.7 s | 4.3 s | 5,699 MiB, only with `vae_tiling` |

- Times are the model's own (`seconds`); the round trip adds 0.5 to 2 s, and
  fetching the 17.8 MB `.ply` in 9 parts added 2.4 s on our network.
- 1,048,576 Gaussians costs about 2 s more (decode) and the same memory; that
  `.ply` is 71.3 MB (36 parts).
- Outputs are bit-identical from run to run on the same card. The 512 px VAE
  tiling that makes 6 GB fit changes the object very slightly.
- Turbo is 11x faster here; on our picks it ran darker and invented detail on the
  hidden back side more often than the 50-step model.

## Run it

```bash
# the SDK: local build of this repo, see ../CONTRIBUTING.md; the client also needs requests and Pillow
cp .env.example .env    # fill in tokens, project, worker id
python publish.py       # once: uploads backend.py to your tenant as a custom backend
python main.py                        # the authors' backpack example, 50 steps and Turbo
python main.py my-photo.jpg --real   # your own photograph
```

Outputs land in `./outputs`: `<name>-base.ply`, `<name>-turbo.ply` and the image
the model saw. Open the `.ply` files in any Gaussian splat viewer (for example
SuperSplat).

## The request

`POST {base}/endpoint/{project}/idol/v1/images/generations` with the data-plane key:

```json
{"model": "default", "image": "<base64 or data URL>", "mode": "real", "turbo": true, "seed": 42}
```

| Field | Default | Meaning |
| --- | --- | --- |
| `image` / `example` | | your picture, or the name of an example in idol-sdk `demo/examples` |
| `mode` | `synthetic` | `synthetic` for renders and generated images, `real` for photographs (looser crop) |
| `turbo` | `false` | Idol Turbo: 5 steps, no guidance |
| `steps`, `cfg_strength`, `rescale_t` | 50, 3.0, 3.0 | sampling (Turbo: 5 steps, guidance off) |
| `num_gaussians` | 262,144 | 32,768 to 1,048,576 |
| `vae_tiling` | `false` | encode the picture in 512 px tiles: what makes a 6 GB card fit |
| `offload_encoders` | `false` | keep BiRefNet, DINOv3 and the VAE in system RAM between requests (did not lower the peak in our test) |
| `vram_limit_gib` | | cap GPU memory for this request, to check a smaller card |

The answer carries `data[0].b64_json` (the image the model saw), the predicted
`camera`, `seconds` and `stage_seconds`, the memory peaks, and the `.ply` as
`ply_b64` part 0 of `parts`; fetch the rest with `{"ply_id": ..., "part": n}`
and check `ply_sha256`. `main.py` does all of that.

From the dashboard's image tester, type an example name as the prompt, plus
`turbo`, `real` or `tiling` (for example `obj-backpack turbo`); the tester
shows the image the model saw.

## Settings (`config.env` of the workload, passed by `main.py` from your environment)

`IDOL_ENCODERS` (`open` or `gated`), `IDOL_TURBO` (`1`), `IDOL_OPTIMIZE` (`1`:
torch.compile and batched guidance, as the authors ship it), `IDOL_VAE_TILING`,
`IDOL_OFFLOAD_ENCODERS`, `IDOL_VRAM_LIMIT_GIB`, `IDOL_WARMUP` (`1`: compiles both
modes during setup so the first request is fast). Changing only these settings
of a running workload does not restart the backend: `delete()` it, then run
`main.py` again.

## Licences

Idol and Idol Turbo: Apache-2.0. DINOv3 weights: Meta's DINOv3 License (read it
before commercial use). FLUX.2 [klein] VAE: Apache-2.0. BiRefNet: MIT.
