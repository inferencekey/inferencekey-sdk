# SANA 1.5 — custom **text2image** backend

Generates an image from a text prompt with `SanaPipeline` (diffusers), packaged
as an InferenceKey custom Python backend.

## Why this backend exists

On AMD RDNA4 nodes (gfx1201, e.g. Radeon AI PRO R9700) the two runtimes the
platform normally uses for `text2image` are both unavailable:

- **sglang** cannot be installed at all. Every published version pins
  `torch==2.9.1` or `torch==2.11.0` — never the `2.10.x+rocm` a ROCm node runs —
  and from 0.5.13 onwards it also requires `flashinfer_python[cu13]`, a CUDA 13
  package with no ROCm equivalent.
- **vllm-omni** does support SANA/diffusion and installs cleanly against ROCm
  torch, but its API needs vLLM ≥ 0.24, while the only vLLM build shipping ROCm
  kernels for gfx1201 (`_rocm_C.abi3.so`) is 0.19.1.

A custom backend driving diffusers directly sidesteps both problems: it only
needs `torch` + `diffusers`, which do have working ROCm wheels.

## Contract

Free-form dicts (**not** OpenAI-compatible):

**Input**

| field | required | default | notes |
|---|---|---|---|
| `prompt` | yes | — | non-empty string |
| `negative_prompt` | no | — | omitted entirely when not supplied |
| `steps` | no | 20 | clamped to 1–100 |
| `height` / `width` | no | 1024 | clamped to 256–2048; 1024 is SANA's native resolution |
| `guidance_scale` | no | 4.5 | |
| `seed` | no | — | makes the job reproducible |

**Output**

```json
{"image_b64": "<base64 PNG>", "width": 1024, "height": 1024,
 "steps": 20, "seed": 42, "model": "Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers"}
```

Note the job envelope requires a non-empty `id`, like every custom backend:
`{"id": "job-1", "input": {...}}`.

## Run it

```sh
python -m inferencekey.backend.serve \
    --port 8099 \
    --backend backend:SanaText2ImageBackend
```

Pick the larger checkpoint (needs more host RAM, see below):

```sh
python -m inferencekey.backend.serve \
    --port 8099 \
    --backend backend:SanaText2ImageBackend \
    --config-json '{"model_id": "Efficient-Large-Model/SANA1.5_4.8B_1024px_diffusers"}'
```

## Measured on an RDNA4 node (R9700 32 GB, host with 6 GB RAM)

Both variants were run on a node that was **also** serving a Qwen3-8B embedding
and a Qwen3-0.6B reranker on the same GPU.

| | 1.6B (default) | 4.8B |
|---|---|---|
| Weights on disk | 9.7 GB | 16.0 GB |
| Load time | ~100 s | 392 s |
| VRAM with all three models | 25.3 GB / 32 GB | 30.3 GB / 32 GB |
| First generation (1024², 20 steps) | **40.7 s** | 219 s |
| Warm generation | **5.6 s** | — |
| Host RAM headroom afterwards | ~2.2 GB | ~0 GB |

Both produce correct images. The 1.6B is the default because the 4.8B leaves the
host with no RAM headroom: on a small-RAM node, loading it a second time while
the previous process is still releasing memory is enough to OOM the VM.

## Host RAM is the real constraint, not VRAM

Diffusion checkpoints are staged through host RAM on their way to the GPU, and
the `amdgpu` driver needs those pages **resident** (pinned) for the DMA. When
they don't fit you get, in `dmesg`:

```
amdgpu: SVM mapping failed, exceeds resident system memory limit
```

and the process dies with a segfault mid-denoising.

Two consequences worth knowing:

- **Adding swap does not help.** A swapped-out page cannot be pinned, so the
  resident limit is unchanged.
- **Do not enable `enable_model_cpu_offload()` on a small-RAM node.** It keeps
  weights in host RAM by design, which is exactly what triggers the failure
  above. This backend deliberately loads straight to VRAM instead.

What matters is the size of the **largest shard**, not the total: diffusers
loads shard by shard. Both SANA variants shard at ~5 GB, which is why they work
where a model with a ~10 GB shard does not.
