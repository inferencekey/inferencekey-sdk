# wevisone-vl-4b-perception-custom-backend-private-nvidia

Tencent WeChat Vision's **WeVisOne-VL-4B** (Apache-2.0) on your own NVIDIA GPU:
one model that does open-vocabulary detection, referring (and says `None` when
the target is not there), counting, pointing, OCR, referring segmentation and
alpha matting. Masks and mattes come from a 9.6M-parameter pixel decoder that
ships with the model, not from SAM. The example deploys it and runs eight
requests, one per task, saving the annotated images, the RGBA cut-outs and the
structured answers to `./outputs`.

| | |
| --- | --- |
| Model | [`tencent/WeVisOne-VL-4B`](https://huggingface.co/tencent/WeVisOne-VL-4B) (Qwen3-VL based, 19.3 GB of F32 weights, loaded in BF16) plus `mask_decoder.pt`; [`tencent/WeVisOne-VL-2B`](https://huggingface.co/tencent/WeVisOne-VL-2B) with `WEVIS_MODEL` |
| Backend | **custom backend** `wevisone-vl-perception` (`backend.py`) that drives the authors' [inference package](https://github.com/Tencent/WeVisOne-VL) at a pinned commit (`40f735b8`) |
| API | `POST .../v1/images/generations` with `{"task", "query", "image"}` (https URL or base64); answer = annotated image (+ RGBA cut-out) in `data[]` plus boxes, points, count, `rejected`, mattes and timings |
| Placement | Private, `fixed` policy, pinned to `IK_WORKER_ID` |
| Needs | NVIDIA driver for CUDA 13.0 (580+), Python 3.12 on the worker; 16 GB of VRAM for every task, 24 GB for the faster vLLM engine |

## Why a custom backend

The text half of WeVisOne-VL is a Qwen3-VL model that vLLM can serve, but the
pixel half is not: for segmentation and matting the model writes an `<OBJ>`
token after each box, and the authors' decoder needs that token's hidden state
plus four layers of ViT features to draw the mask. So `backend.py` runs the
authors' own `WeVisOneVL` class:

- **`WEVIS_ENGINE=vllm`**: vLLM writes the answer (boxes, points, `<OBJ>`), then
  a Transformers copy of the model is teacher-forced on it to feed the decoder.
  Two copies of the weights: about 20 GB.
- **`WEVIS_ENGINE=transformers`**: one copy does everything. 2 to 3 times slower
  on object-level tasks, half the memory.
- **`auto`** (default): `vllm` on cards of 22 GiB or more, `transformers` below.

Two fixes in `backend.py` that you would otherwise hit:

1. The authors' `LLM(...)` call keeps Qwen3-VL's multimodal defaults, so vLLM
   profiles a 16,384-token image and on a 24 GB card shared with the second
   copy finds no room for its KV cache (`Available KV cache memory: -0.87 GiB`).
   The backend sizes vLLM for one image at the input resolution cap
   (`limit_mm_per_prompt`, `mm_processor_kwargs`), `max_model_len` 3072 and
   `gpu_memory_utilization` 0.465 on a 24 GB card.
2. Results travel in an 8 MB message and the worker base64-encodes them once
   more; two 1600 px PNGs of a photo did not fit (`message size of 9145132`),
   so the annotated image goes back as JPEG and the cut-out as a 1280 px PNG.

`task_type` is `text2image` because the SDK packager has no `image2image` yet;
it only decides which route reaches the backend (`/v1/images/generations`).

## What we measured (InferenceKey deploy test, 2026-10-25)

GPU time measured inside the backend, greedy decoding, one request at a time.

| Task (image) | RTX 4090, vLLM engine | RTX 4090, Transformers engine | RTX 5080 16 GB, Transformers |
| --- | --- | --- | --- |
| Detection, 5 labels, 32-33 boxes (COCO kitchen) | 1.9 s warm (14.5 to 15.5 s as the first request after a start) | 7.4 s | 8.3 s |
| Counting people (COCO, 12) | 0.78 s | 2.7 s | 2.8 s |
| Referring / rejection ("the dog" in a photo of cats) | 0.21 s / 0.11 s, `None` | 0.55 s / 0.36 s, `None` | 0.68 s / 0.47 s, `None` |
| OCR, whole street sign | 3.7 s, 29 words | 9.1 s, 21 words | 9.5 s, 22 words |
| Segmentation (640 px bicycle) | 1.2 s | 0.89 s | 1.2 s |
| Alpha matting (1620 x 1080 portrait) | 0.97 to 1.1 s | 1.1 s | 2.1 s |
| Peak card memory, all tasks | 23,539 MiB | 12,546 MiB | 13,148 MiB |
| Ready after `ensure()` | 49 to 82 s warm (vLLM compile and graphs) | 14 s warm | 87 s on a fresh pod (download 38 s) |

- Same answers across engines on most tasks, not all: counting the white chess
  pieces in the authors' example (16 on the board) gave **15 with vLLM** (it
  missed a knight) and **16 with Transformers**, on the same card.
- **Smaller cards** (emulated with a PyTorch memory cap on the 4090): with the
  4B capped at 11.5 GiB every object-level task and the small mattes worked, but
  the 1620 x 1080 matte ran out of memory; at 10.8 GiB (a strict 12 GB card)
  even a 640 px segmentation did (the decoder allocates 768 MiB at once). The
  2B capped at 7.2 GiB (8 GB card) passed 12 of 14 requests: the large matte ran
  out of memory and one OCR grounding answer came back malformed.

## Run

```bash
cp .env.example .env      # fill in real values
python -m venv .venv && . .venv/bin/activate
pip install maturin requests
( cd ../../bindings/python && maturin develop --release )
python publish.py          # once: uploads backend.py to your tenant
python main.py             # ensure -> wait until ready -> 8 requests -> delete
```

`IK_KEEP_ALIVE=1` keeps the workload for more requests. The dashboard image
tester works too: type `matting: person https://.../photo.jpg` as the prompt
(task, colon, query, image URL).

Example request:

```bash
curl -s "$INFERENCEKEY_BASE_URL/endpoint/$INFERENCEKEY_PROJECT/wevisone-vl-4b/v1/images/generations" \
  -H "Authorization: Bearer $INFERENCEKEY_API_KEY" -H "Content-Type: application/json" \
  -d '{"task": "count_box", "query": "people", "image": "http://images.cocodataset.org/val2017/000000001000.jpg"}'
```

## Files

- `backend.py`: the custom backend (setup downloads the authors' code and the pinned model revision).
- `requirements.txt`: the backend's venv (torch 2.11, vLLM 0.23.0, Transformers 5.12.1, the authors' pins).
- `publish.py`: packages and uploads the backend.
- `main.py`: deploys with `ensure()` and runs the eight requests.

Attribution: WeVisOne-VL by WeChat Vision, Tencent Inc., Apache-2.0. Test
images: the authors' examples and COCO val2017 photos (Flickr, their own licences).
