# inferencekey (Python)

Official Python SDK for the **InferenceKey** platform — a thin, typed layer over
the shared Rust core (`inferencekey-core`).

Declare AI workloads in code, ensure they exist on the platform, and call the
resulting OpenAI-compatible endpoints.

```python
from inferencekey import ManagementClient, DataClient, WorkloadSpec, Backend

mgmt = ManagementClient.from_env(project="acme")          # INFERENCEKEY_SDK_TOKEN
ref = mgmt.ensure(WorkloadSpec(
    name="support-bot", slug="support-bot",
    model="meta-llama/Llama-3.1-8B-Instruct", backend=Backend.VLLM,
    command="vllm serve meta-llama/Llama-3.1-8B-Instruct --max-model-len 8192",
))

data = DataClient.from_env(project="acme")
out = data.endpoint(ref.workload_slug, api_key="ik_live_...").generate_text(prompt="Hola")
print(out.text)
```

Two tokens, two clients (least privilege): `ik_sdk_` provisions (control plane);
`ik_live_` calls inference (data plane, per workload). Full docs at
**docs.inferencekey.com**.

License: Apache-2.0.
