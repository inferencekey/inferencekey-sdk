# InferenceKey SDK

The official, multi-language SDK for the **InferenceKey** platform: declare AI
workloads in code, ensure they exist on the platform, and call the resulting
OpenAI-compatible endpoints.

> Status: **early development.** A single Rust **core** plus a stable **C ABI**
> and thin **native bindings** per language. Python and Node ship first; Go and
> Java follow over the same C ABI.

## Architecture — one core, many bindings

```
inferencekey-sdk/
├── core/            inferencekey-core — all logic + transport (reqwest/SSE).
│                    Pure domain (enums, spec, drift, wire, sse) + pipelines
│                    (ensure / generate_text / embed). One source of truth.
├── capi/            inferencekey-capi — stable C ABI (extern "C") over the core,
│                    for FFI consumers (cgo, JNI/FFM, …). cbindgen → inferencekey.h.
└── bindings/
    ├── python/      pyo3 + maturin → the `inferencekey` PyPI package.
    └── node/        napi-rs → the `@inferencekey/sdk` npm package.
```

Behaviour lives in the Rust core, so every language behaves identically; the
bindings are thin shells that marshal types and map errors to each language's
idioms.

## Two tokens, two clients (least privilege)

- **`ik_live_…`** — *consume* workloads. Data plane, passed **per workload** so one
  app can call several workloads each with its own key. Used by the data client.
- **`ik_sdk_…`** — *create / reconcile* workloads. Control plane, **scoped to one
  project**. Held by the management client. Cannot call inference.

A data client can't provision; a management client can't call inference —
enforced server-side, and again client-side with fast, typed wrong-token errors.

## Quickstart

### Python

```python
from inferencekey import ManagementClient, DataClient, WorkloadSpec, Backend

mgmt = ManagementClient.from_env(project="acme")          # INFERENCEKEY_SDK_TOKEN
ref = mgmt.ensure(WorkloadSpec(
    name="support-bot", slug="support-bot",
    model="meta-llama/Llama-3.1-8B-Instruct", backend=Backend.VLLM,
    command="vllm serve meta-llama/Llama-3.1-8B-Instruct --max-model-len 8192",
))  # on_drift defaults to RECONCILE

data = DataClient.from_env(project="acme")
out = data.endpoint(ref.workload_slug, api_key="ik_live_...").generate_text(prompt="Hola")
print(out.text)
```

### Node / TypeScript

```ts
import { ManagementClient, DataClient, Backend } from "@inferencekey/sdk";

const mgmt = ManagementClient.fromEnv({ project: "acme" });
const ref = await mgmt.ensure({
  name: "support-bot", slug: "support-bot",
  model: "meta-llama/Llama-3.1-8B-Instruct", backend: Backend.Vllm,
  command: "vllm serve meta-llama/Llama-3.1-8B-Instruct --max-model-len 8192",
});

const data = DataClient.fromEnv({ project: "acme" });
const ep = data.endpoint(ref.workloadSlug, { apiKey: process.env.SUPPORT_IK_LIVE! });
const res = await ep.generateText({ prompt: "Hola", temperature: 0.2, maxTokens: 300 });
```

Full docs at **docs.inferencekey.com**.

## Building

```bash
cargo build            # core + capi + bindings (Rust side)
cargo test             # core unit tests
# Python wheel:  (cd bindings/python && maturin build --release)
# Node addon:    (cd bindings/node   && npx napi build --release)
# C header:      (cd capi && cbindgen --config cbindgen.toml --output include/inferencekey.h)
```

## License

Apache-2.0.
