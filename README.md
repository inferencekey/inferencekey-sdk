# InferenceKey SDK

The official, multi-language SDK for the **InferenceKey** platform: declare AI
workloads in code, ensure they exist on the platform, and call the resulting
OpenAI-compatible endpoints.

> Status: **early development.** TypeScript first (this repo's `typescript/`),
> Python next. Other languages are tracked in the docs as "Coming soon".

## Two tokens, two clients (least privilege)

- **`ik_live_…`** — *consume* workloads. Data plane, passed **per workload** so one
  app can call several workloads each with its own key. Used by `DataClient`.
- **`ik_sdk_…`** — *create / reconcile* workloads. Control plane, **scoped to one
  project**. Held by `ManagementClient`. Cannot call inference.

A `DataClient` can't provision; a `ManagementClient` can't call inference —
enforced server-side (separate credentials, the auth extractor, and fast
wrong-token errors).

## TypeScript quickstart

```ts
import { ManagementClient, DataClient, Backends } from "@inferencekey/sdk";

// 1. Provision (CI / infra) — uses INFERENCEKEY_SDK_TOKEN
const mgmt = ManagementClient.fromEnv({ project: "acme" });
const ref = await mgmt
  .workload({
    name: "support-bot",
    slug: "support-bot",
    model: "meta-llama/Llama-3.1-8B-Instruct",
    backend: Backends.VLLM,
    command: "vllm serve meta-llama/Llama-3.1-8B-Instruct --max-model-len 8192",
  })
  .ensure(); // onDrift defaults to RECONCILE

// 2. Consume (app) — per-workload ik_live_ key
const client = DataClient.fromEnv({ project: "acme" });
const support = client.endpoint(ref.workloadSlug, { apiKey: process.env.SUPPORT_IK_LIVE! });
const res = await support.generateText({ prompt: "Hola", temperature: 0.2, maxTokens: 300 });
```

See [`typescript/`](typescript/) for the package. Full docs at
**docs.inferencekey.com**.

## Layout

```
inferencekey-sdk/
├── typescript/   # @inferencekey/sdk  (this MVP)
├── python/       # inferencekey        (next)
└── …             # go / java — coming soon
```

## License

Apache-2.0.
