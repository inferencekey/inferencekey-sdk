# Contributing an example

Examples are often the first thing someone sees, so we hold them to one bar:
**a new user can clone, set a few environment variables, run one command, and
get a working result** — without reading the SDK internals. This document is the
contract every example must meet.

If anything here is unclear or seems wrong for your example, open an issue rather
than quietly diverging — the consistency *is* the value.

## 1. Folder & naming convention

Examples live **flat** under `examples/`, one folder per example. There is no
`cloud/` vs `private/` nesting: the folder **name** carries that information, and
the [catalogue table](./README.md#catalogue) is how people browse.

Name a folder:

```
<topic>-<placement>[-<hardware>]
```

- **`topic`** — what it demonstrates, in kebab-case: `hello-world`,
  `chat-streaming`, `embeddings`, `batch`.
- **`placement`** — `cloud` (platform schedules it) or `private` (pinned to a
  worker you registered).
- **`hardware`** — **required for `private`, omitted for `cloud`.** Identifies
  the GPU vendor/arch the example expects, so a reader knows which box they need:
  `nvidia`, `amd-gfx120x` (the AMD R9700 / RDNA4), etc. Cloud examples have no
  hardware suffix because the platform picks the GPU.

Examples:

```
hello-world-cloud
chat-streaming-cloud
chat-streaming-private-nvidia
chat-streaming-private-amd-gfx120x
embeddings-cloud
```

A numeric prefix is **not** used — ordering and "where do I start" live in the
catalogue table and the `hello-world-cloud` entry, not in the filesystem.

## 2. One example, one language (usually)

Keep each example folder focused on **one language** so it stays small and
copy-pasteable. The catalogue's _Lang_ column and the example's README say which
one. If a scenario genuinely benefits from showing both Python and TypeScript,
put them in sibling files inside the same folder (`main.py`, `main.ts`) and a
single README that covers both — but prefer one language per folder.

Supported languages and floors:

| Language | Minimum | Notes |
| --- | --- | --- |
| Python | **3.9** | matches the SDK's `requires-python = ">=3.9"`. |
| TypeScript / Node | **18** | run with `tsx`; `"type": "module"`. |

## 3. Required files

Every example folder contains:

| File | Purpose |
| --- | --- |
| `README.md` | The example's own page — see the [template](#5-readme-template). |
| `.env.example` | Every env var the example reads, with **placeholder values only**. |
| the code | `main.py` **or** `chat_streaming.py` / `main.ts` etc. — one clear entry point. |
| `package.json` / `requirements.txt` | Declares the dependency on the SDK (see below). |

The `examples/.gitignore` already covers `.env`, `node_modules`,
`package-lock.json`, `__pycache__`, and `.venv` — do not re-add those per folder.

## 4. Depending on the SDK

The packages **are not published yet** (`inferencekey` on PyPI,
`@inferencekey/sdk` on npm). Until they are, examples depend on the **local build
in this repo**. From `examples/<your-example>/` the SDK lives two directories up,
under `bindings/`.

### Node / TypeScript

Depend on the local binding by path in `package.json`:

```jsonc
{
  "type": "module",
  "dependencies": {
    // local, unpublished build (this repo). examples/<x>/ → ../../bindings/node
    "@inferencekey/sdk": "file:../../bindings/node"
  },
  "devDependencies": {
    "@types/node": "^20",
    "tsx": "^4",
    "typescript": "^6"
  },
  "scripts": { "demo": "tsx main.ts", "typecheck": "tsc --noEmit" }
}
```

`npm install` resolves `@inferencekey/sdk` to whatever is currently compiled in
`bindings/node`. If you changed the SDK's Rust or TS, rebuild it first:

```bash
( cd ../../bindings/node && npm install && npm run build )   # napi build --platform --release && tsc
```

> **When the package ships,** swap the `file:` line for a normal semver range
> (`"@inferencekey/sdk": "^0.1.0"`) and the example keeps working unchanged.

### Python

The Python package is a maturin-built native extension. For a local checkout,
install it editable into the example's virtualenv with **maturin**:

```bash
python -m venv .venv && . .venv/bin/activate
pip install maturin
( cd ../../bindings/python && maturin develop --release )    # builds & installs `inferencekey` into this venv
```

After that, `from inferencekey import ManagementClient, DataClient, WorkloadSpec`
resolves to the local build. Pin it in `requirements.txt` as a comment for now:

```
# local, unpublished build — see README: `maturin develop` from ../../bindings/python
# inferencekey>=0.1.0   # uncomment once published to PyPI
```

> **When the package ships,** drop the maturin step and `pip install inferencekey`.

## 5. README template

Each example's `README.md` follows this skeleton — same order every time, so
readers build muscle memory:

```markdown
# <Example title> — <one-line what it shows>

<2–3 sentences: the scenario, and the one axis it varies vs the others.>

## Compatibility
- SDK: local build in this repo (or >= x.y once published)
- Language: Python >= 3.9  (or Node >= 18)
- Placement: cloud | private (<vendor/arch>)
- Backend: vllm | sglang | ...   Policy: autoscaling | fixed

## Prerequisites
- The tokens/ids this example needs (link to ../README.md#tokens).
- Runtime version; any account setup (a registered worker, for private).

## Run
\`\`\`bash
cp .env.example .env        # fill in real values
# install the local SDK — see ../CONTRIBUTING.md#4-depending-on-the-sdk
<the single run command>
\`\`\`

## What it does
The three steps — ensure() → wait until ready → call — in this example's terms.

## Cost & cleanup
What gets provisioned, the policy, and the teardown (this example calls
delete() on exit; how to clean up if you Ctrl-C).

## Troubleshooting
The 2–4 errors a first-timer is most likely to hit, and the fix for each.

---
See [CONTRIBUTING.md](../CONTRIBUTING.md) and the
[tokens & placement overview](../README.md#tokens).
```

Keep the **Run** block copy-pasteable: the commands must work as-is in a clean
terminal, in order, with no hidden steps.

## 6. The canonical shape — reuse, don't reinvent

Every example is a variation on the same three steps. Use these exact APIs rather
than rolling your own flow:

| Step | Python | TypeScript |
| --- | --- | --- |
| Provision / reconcile (control plane, `ik_sdk_`) | `mgmt = ManagementClient.from_env(project=...)`<br>`ref = mgmt.ensure(WorkloadSpec(...))` | `const mgmt = ManagementClient.fromEnv()`<br>`const ref = await mgmt.ensure(spec)` |
| Wait for the cold worker to serve | `mgmt.wait_until_ready(slug, timeout=600)` | `await mgmt.waitUntilReady(slug, { timeoutMs: 600_000 })` |
| Call the endpoint (data plane, `ik_live_`) | `data.endpoint(slug).generate_text_stream(...)` | `data.endpoint(slug).generateTextStream(...)` |
| Tear down | `mgmt.delete(slug)` | `await mgmt.delete(slug)` |

`ensure()` is **idempotent by slug**: re-running with the same spec is a no-op;
a changed spec reconciles in place (`on_drift` defaults to reconcile). A cold
worker can take minutes to pull its image and boot, so the readiness timeout
defaults to **600 s / 600_000 ms** — raise it for large models.

### Placement is the only real difference

- **Cloud:** leave `worker_id` / `gpu_resource_id` unset; set
  `execution_policy` to `autoscaling` so it can scale to zero.
- **Private:** set `worker_id` (`wrk_…`) **and** `gpu_resource_id` (`gpu_…`) from
  env. These pin the workload to your box.

The spec **does not carry the GPU vendor or architecture** — that is inferred
from the worker you pin to. The `command` is the same `vllm serve <model> …` on
NVIDIA and AMD.

### Writing the AMD (gfx120x / RDNA4) example

The AMD R9700 path differs from NVIDIA only in the worker environment and a
couple of serve flags, **not** in the SDK call:

- Backend must be **`vllm`** — **`sglang` is not supported** on AMD ROCm gfx120x
  (RDNA4); the worker rejects it as platform-unsupported.
- The serve command needs RDNA4-appropriate flags, e.g.
  `vllm serve <model> --enforce-eager --gpu-memory-utilization 0.8` (lower it if
  you hit OOM).
- The worker must be running the ROCm gfx120x base image; that is a worker-side
  concern, but the example's README should say so and link to the worker's ROCm
  setup docs.
- Pick a model that **fits** the R9700's VRAM and state the requirement in the
  README so nobody OOMs on first run.

## 7. Secrets & cost hygiene (hard rules)

These are non-negotiable — a PR that breaks one will be asked to change:

1. **No hard-coded secrets or ids.** Tokens (`ik_sdk_`, `ik_live_`) and ids
   (`wrk_`, `gpu_`) are read from the environment or a local `.env`. The `.env`
   is git-ignored; only `.env.example` (placeholders) is committed.
2. **`.env.example` holds placeholders only** — `ik_sdk_…`, `wrk_…`, never a real
   value, never a value from your own account.
3. **Never log a token.** Don't print `INFERENCEKEY_SDK_TOKEN` /
   `INFERENCEKEY_API_KEY` (or any `ik_…` value) to stdout/stderr, even when
   debugging.
4. **Every example that provisions GPU must tear down.** Call `delete()` on exit
   (incl. on Ctrl-C / error), and document the teardown in the README's
   _Cost & cleanup_ section. Prefer `autoscaling` (scale-to-zero) for cloud;
   if a private example uses `fixed`, the README must call out that the GPU stays
   reserved until deleted.

## Checklist before you open a PR

- [ ] Folder named `<topic>-<placement>[-<hardware>]`; flat under `examples/`.
- [ ] Added a row to the [catalogue table](./README.md#catalogue).
- [ ] `README.md` follows the [template](#5-readme-template); **Run** block works
      verbatim in a clean terminal.
- [ ] `.env.example` lists every var, placeholders only; no real values committed.
- [ ] Depends on the SDK via the local path / maturin; documented in the README.
- [ ] Uses `ensure → wait_until_ready → endpoint` and `delete()`s on exit.
- [ ] No token ever printed; no secret or id hard-coded.
- [ ] Private/AMD examples: `vllm` backend, RDNA4 flags, VRAM requirement stated.
