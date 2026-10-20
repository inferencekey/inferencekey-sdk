# custom-backend-openai-chat

A custom backend that speaks **OpenAI chat**: the template for serving your own
LLM engine on InferenceKey (DeepInfer, a custom PyTorch loop, anything that
generates text).

It implements `chat()` on top of the usual `setup()` / `process()`. With that,
the SDK runtime serves `POST /v1/chat/completions` (server-sent events when
`"stream": true`) and the platform routes chat to it the way it routes chat to
vLLM: tool calls, finish reason and token usage reach the caller, and the
workload's Anthropic `/v1/messages` endpoint works for Claude Code.

No third-party dependency: only the standard library.

## Two modes

| Mode | When | What it does |
| --- | --- | --- |
| upstream | `config.upstream_base_url` is set | Forwards every request to an OpenAI-compatible server and streams its answer token by token. Replace `_upstream_chat` / `_upstream_stream` with your engine. |
| scripted | default | A deterministic stand-in, no model. With tools, it reads `demo_file`, then edits it (`demo_edit_old` to `demo_edit_new`), then reports. Enough for Claude Code to run two real tool calls end to end. |

Config keys (the workload `config`, which the backend receives as `ctx.config`):

| Key | Mode | Meaning |
| --- | --- | --- |
| `upstream_base_url` | upstream | Base URL of the OpenAI-compatible server (without `/v1`). |
| `upstream_model` | upstream | Model name to send upstream (optional). |
| `upstream_api_key_env` | upstream | Name of an environment variable holding the upstream key; set it with `config.env`. |
| `upstream_timeout_secs` | upstream | Per-request timeout towards the upstream (default 600). |
| `demo_file`, `demo_edit_old`, `demo_edit_new` | scripted | The file Claude Code reads and the edit it makes. |
| `demo_setup_seconds` | both | Sleep in `setup()`, to exercise `ready_timeout_secs`. |
| `demo_stream_seconds` | scripted | Stream the answer over this many seconds, to exercise `request_timeout_secs`. |

## Run it locally

From the SDK repo root:

```bash
export PYTHONPATH="$PWD/bindings/python/python:$PWD/examples/custom-backend-openai-chat"
python -m inferencekey.backend.serve --port 8099 --backend backend:OpenAIChatBackend \
  --config-json '{"demo_file": "/tmp/demo.txt", "demo_edit_old": "hello", "demo_edit_new": "bye"}'

curl -s localhost:8099/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"read it"}],"tools":[{"type":"function","function":{"name":"Read"}}]}'
```

The answer carries a `Read` tool call with `"finish_reason": "tool_calls"`.

## Package, publish and use it

```bash
python -m inferencekey.backend.package \
  --src examples/custom-backend-openai-chat/backend.py \
  --entrypoint backend:OpenAIChatBackend \
  --requirements examples/custom-backend-openai-chat/requirements.txt \
  --name openai-chat-example --version 0.1.0 --task-type text2text \
  --api openai-chat --out dist
```

Publish `dist/openai-chat-example-0.1.0.tar.gz` (SDK `publish_custom_backend` or
the dashboard), create a `text2text` workload with `backend="openai-chat-example"`
and point Claude Code at it:

```bash
export ANTHROPIC_BASE_URL="https://<manager>/endpoint/<project>/<workload>"
export ANTHROPIC_API_KEY="ik_live_..."
export ANTHROPIC_MODEL="openai-chat-example"
claude -p "Read demo.txt and replace 'hello' with 'bye'." --allowedTools "Read Edit"
```

See the **Custom backends** guide, section "Serve an LLM", for the contract and
the workload config keys (`env`, `ready_timeout_secs`, `request_timeout_secs`).
