# unlimited-ocr-2-pdf-parsing-custom-backend-private-nvidia

Baidu's **Unlimited OCR 2** (MIT license) on one private NVIDIA GPU: a whole
PDF goes in **one request** and comes back as one Markdown answer. It is a 3B
mixture-of-experts document parser (about 0.5B active parameters) that reads
every page in order, with a `<PAGE>` marker per page, layout blocks with their
position on the page, HTML tables, and blocks that link a paragraph or a table
across a page break (`cross_text` / `append_text`, `append_table`).

`main.py` deploys it behind the OpenAI-compatible chat endpoint and parses
one PDF (a link, or a small local file).

| | |
| --- | --- |
| Model | [`baidu/Unlimited-OCR-2`](https://huggingface.co/baidu/Unlimited-OCR-2) at revision `c14071b5` (6.7 GB of BF16 weights) |
| Backend | **custom backend** `unlimited-ocr-2` (`backend.py`): installs the authors' SGLang build (`wheel/sglang-0.5.20.dev13+g75b9b06b02` from the model repository) with uv in a Python 3.10 environment, starts `sglang.launch_server` with the authors' flags and page-window settings, turns the PDF into page images and sends them as one multi-page request with the authors' n-gram repetition guard |
| API | `POST .../v1/chat/completions` (`"stream": true` recommended). The user message carries a link to a PDF or an image in its text, or a PDF as an OpenAI `file` part, or page images as `image_url` parts. Options go in the text: `pages 1-20`, `crop_mode=false`, `dpi=200` |
| Answer | Markdown with `<PAGE>` per page and `<BLOCK>type[x1, y1, x2, y2]<FORK>text` per layout block (coordinates on a 1000 x 1000 grid), tables as HTML |
| Placement | Private, `fixed` policy, pinned to `IK_WORKER_ID` |
| Needs | An NVIDIA GPU with 16 GB or more, a driver for CUDA 13 (the build uses torch 2.13 + cu130); we used a 100 GB container disk |

## Run it

```bash
cp .env.example .env          # fill in the tokens, the project and IK_WORKER_ID
pip install -r requirements.txt
python publish.py             # once: uploads backend.py to your tenant
IK_PDF=https://www.irs.gov/pub/irs-pdf/p15.pdf python main.py
```

Outputs land in `outputs/<name>.md` (the Markdown) and `outputs/<name>.json`
(pages, seconds, pages per minute, time to the first text).

**Two things the platform does that shape the request:**

- The chat route accepts bodies up to 8 MiB, but in our test a request with a
  1.7 MB PDF inline (2.2 MB of JSON) never reached the worker (it is being
  looked at). Send a **link** to the PDF instead: the backend downloads it.
- The platform forwards the standard chat fields only (messages, max_tokens,
  temperature...), so the backend reads its options from the message text:
  `"Parse this document: https://example.org/report.pdf pages 10-30 crop_mode=false"`.

**SDK version on the worker.** The backend implements `chat()` /
`chat_stream()`, which needs the SDK from this repository's `main`; the
`inferencekey` package on PyPI (0.260725.739) predates it and the backend
fails with `cannot import name 'chat_completion'`. Until the next PyPI
release, point the worker at a checkout of `bindings/python/python`
(`INFERENCEKEY_SDK_PYTHON_DIR`), which is what we did.

## Which mode: page tiles or one view per page

Each page can be encoded as one 1024-pixel global view plus tiles
(`crop_mode=true`, the authors' default, "gundam") or as the global view only
(`crop_mode=false`, "base"). On our test document the base mode was both
faster and more faithful:

| 59-page PDF, RTX 4090 | Page tiles (default) | One view per page (`crop_mode=false`) |
| --- | --- | --- |
| Whole PDF, round trip | 404.9 s, 8.7 pages/min | 223.8 s, 15.8 pages/min |
| Tokens in / out | 97,621 / 160,254 | 18,277 / 89,477 |
| Edit distance to the PDF's own text | 38.1 % (2.4 % without pages 46-54) | 4.2 % |
| The 9-page table (pages 46-54) | broke: 8 rows in one cell, one sentence repeated 1,508 times on page 46, pages 47, 49 and 52 empty | intact: 7 HTML tables, page 46 within 0.7 % |

Start with `crop_mode=false` for long documents and keep the tiles for small
print, where the extra resolution helps.

## Page windows

`UOCR_PAGE_WINDOW` (default 10) feeds the document to the model ten pages at
a time and reuses the KV tail of the previous window, as the authors
recommend. With one window for the whole document (`UOCR_PAGE_WINDOW=0`) our
59-page PDF hit the model's 131,072-token context (89,331 tokens of page
images alone in tile mode) and the answer stopped on page 27, with the first
text after 50 s instead of 10.5 s. On the pages both runs wrote, the text was
the same.

## What we measured (InferenceKey deploy test, 2026-11-02)

Document: [IRS Publication 15 (2026)](https://www.irs.gov/pub/irs-pdf/p15.pdf),
59 pages, public domain, rendered at 300 dpi. One request per run, greedy.

| | RTX 4090 24 GB |
| --- | --- |
| First start, cold | weights 10 s, SGLang environment 60 s, server start with CUDA graphs about 105 s |
| Warm start | ready about 60 s after `ensure()` |
| Decode speed | about 430 to 445 tokens/s, flat across the document |
| Peak memory on the card | 19.4 GB with the default memory share (0.7); 14.7 GB (tiles) and 15.2 GB (one view per page) with `UOCR_MEM_FRACTION_STATIC=0.5` |
| Two pages from the panel tester | first text 3.3 s, done in 10 s |

### Which GPU

| GPU or cap | Configuration | Peak on the card | Whole 59-page PDF | Output |
| --- | --- | --- | --- | --- |
| RTX 4090 24 GB (real) | defaults, tiles | 19.4 GB | 8.7 pages/min | reference |
| RTX 4090 24 GB (real) | `crop_mode=false`, `UOCR_MEM_FRACTION_STATIC=0.5` | 15.2 GB | 15.8 pages/min | 4.2 % edit distance |
| 16 GB cap on the 4090 (emulated) | `UOCR_MEM_FRACTION_STATIC=0.5`, tiles | 14.7 GB | 8.7 pages/min | same text as the 24 GB run |
| RTX 2000 Ada 16 GB, 70 W (real) | defaults, `crop_mode=false` | 13.8 GB | 5.6 pages/min | same quality (4.2 % edit distance) |
| RTX 2000 Ada 16 GB (real) | defaults, tiles, pages 1-10 | 13.8 GB | 5.0 pages/min | same text as the 4090 on those pages |

A 16 GB card runs it with the defaults; the RTX 2000 Ada was about 2.8 times
slower than the 4090 (about 147 against 440 tokens/s). We did not try 12 GB.

## Files

- `backend.py`: the custom backend (setup, PDF to page images, streaming).
- `publish.py`: packages and uploads the backend (`api="openai-chat"`).
- `main.py`: ensure, wait, parse one PDF, save the Markdown.

Model by Baidu ([model card](https://huggingface.co/baidu/Unlimited-OCR-2),
[code](https://github.com/baidu/Unlimited-OCR)), MIT license.
