# Changelog

Notable changes to the InferenceKey SDK. Python releases are date-versioned
on PyPI; entries are listed newest first.

## Unreleased

### Custom backend runtime (`inferencekey.backend.serve`)

- **Forecast `id` is optional.** `ForecastRequest.from_wire` no longer answers
  `400` when `id` is missing, empty or not a string: it uses `series_id` when
  that is a non-empty string, else a generated UUID4. Send `series_id` behind a
  gateway such as LiteLLM, whose pass-through strips a top-level `id`.
- **Native HTTP/SSE transport (opt-in).** The Python wheel now contains a Rust
  HTTP server for custom backends (new crate `server/`,
  `inferencekey-backend-server`). It accepts connections, parses HTTP, answers
  `/healthz`, `/meta` and `/v1/models`, frames SSE and writes every byte
  without the GIL; Python only runs the backend's methods. Select it with
  `IK_BACKEND_HTTP=auto` (native when available, else stdlib) or
  `IK_BACKEND_HTTP=native` (exit code 2 if unavailable). **The default stays
  `stdlib` in this release**: in CPU load tests the native transport keeps
  `/healthz` under 50 ms at p99 and delivers the full 800-stream load, but the
  added time to first token at a simultaneous burst and the GIL share at 800
  streams are still above target. One `http transport: ...` line on stderr names
  the transport in use. Optional settings: `IK_BACKEND_MAX_INFLIGHT`,
  `IK_BACKEND_IO_THREADS`, `IK_BACKEND_BACKLOG`, `IK_BACKEND_STREAM_MAX_BYTES`,
  `IK_SSE_CONNECTION_CLOSE` and `IK_BACKEND_FLUSH_TICK_US`.
- **Wire differences on `native`** (status codes, error bodies and SSE `data:`
  lines are identical): streaming responses to HTTP/1.1 clients use chunked
  transfer encoding (HTTP/1.0 clients and `IK_SSE_CONNECTION_CLOSE=1` keep the
  close-delimited stream); JSON routes answer `HTTP/1.1` with keep-alive; a
  non-numeric `Content-Length` gets the HTTP library's bodyless `400`; a
  request still running at SIGTERM gets up to 2 s before the process exits.
- **Return type of `serve()` and `serve_backend()` on `native`.** They return
  an object with `server_address`, `server_port`, `serve_forever()`,
  `shutdown()` and `server_close()`, not a `ThreadingHTTPServer`. Code that
  checks `isinstance(..., ThreadingHTTPServer)` or uses `RequestHandlerClass`
  or `socket` should set `IK_BACKEND_HTTP=stdlib`.
- **Generator close on disconnect (both transports).** When a streaming client
  leaves, the runtime closes the `chat_stream()` generator at its next `yield`
  (`GeneratorExit`), so engine-abort code in a `try`/`finally` runs. A backend
  that catches `BaseException` around a `yield` must re-raise `GeneratorExit`.
- **Pre-serialized chunks (both transports).** `chat_stream()` may yield a
  `str` or `bytes` holding one line of chunk JSON; it is written as is. A chunk
  with a raw line break ends the stream with an error event and `[DONE]`.
- **Keepalives (both transports).** `chat_stream()` may yield `None`; the
  runtime writes an SSE `: keepalive` comment, which also detects a client that
  left during an idle period.
- **stdlib transport:** listen backlog raised from 5 to 1024, so a burst of new
  connections no longer gets resets.
- **Request body over 16 MiB (both transports).** The `400 {"error": "request
  body too large"}` now goes out after the body is read and dropped (up to
  64 MiB, for at most 10 s), with `Connection: close`. Before, the connection
  closed with the body unread and the kernel reset it, so a client sending the
  body whole (the worker) saw a send error instead of the `400`.
