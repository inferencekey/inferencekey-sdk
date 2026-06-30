"""A minimal, self-contained custom backend: an in-memory linear model.

No weights are downloaded. ``setup()`` builds a tiny ``nn.Linear`` (identity
weights) on the device read from config; ``process()`` runs a forward pass over
``input["values"]`` and echoes the resulting vector back in ``output["values"]``.
It demonstrates the contract end to end: load once, process many.

Run it (from the SDK repo root, with this folder importable)::

    python -m inferencekey.backend.serve \\
        --port 8099 \\
        --backend backend:EchoLinearBackend \\
        --config-json '{"device": "cpu", "size": 4}'
"""

from __future__ import annotations

import torch
from torch import nn

from inferencekey.backend import BackendContext, CustomBackend, Job, Result


class EchoLinearBackend(CustomBackend):
    """Identity-initialized ``nn.Linear`` that echoes its input vector."""

    def setup(self, ctx: BackendContext) -> None:
        # Device is read explicitly from config (default "cpu"); no GPU
        # autodetection. No weights are fetched from disk or network.
        self.device = str(ctx.config.get("device", "cpu"))
        size = int(ctx.config.get("size", 4))
        model = nn.Linear(size, size, bias=False)
        with torch.no_grad():
            model.weight.copy_(torch.eye(size))  # identity -> echo
        self.size = size
        self.model = model.to(self.device).eval()

    @staticmethod
    def _extract_text(payload: dict):
        """Pull user text from a chat-style or prompt-style job input."""
        prompt = payload.get("prompt")
        if isinstance(prompt, str) and prompt:
            return prompt
        messages = payload.get("messages")
        if isinstance(messages, list) and messages:
            for msg in reversed(messages):
                if isinstance(msg, dict) and msg.get("role") == "user":
                    content = msg.get("content")
                    if isinstance(content, str) and content:
                        return content
            last = messages[-1]
            if isinstance(last, dict) and isinstance(last.get("content"), str):
                return last["content"]
        return None

    def process(self, job: Job) -> Result:
        # A real inference job from the platform arrives as a chat body
        # (`messages`) or a `prompt`; echo the user's text back through a
        # trivial forward pass so the example handles the product's dominant
        # shape end to end, not just a raw numeric vector.
        text = self._extract_text(job.input)
        if text is not None:
            # Map chars to floats, run the identity Linear, map back: a
            # PyTorch forward pass that round-trips the text (the "echo").
            codes = [float(ord(c)) for c in text][: self.size] or [0.0]
            codes += [0.0] * (self.size - len(codes))
            with torch.no_grad():
                out = self.model(torch.tensor(codes, dtype=torch.float32, device=self.device))
            echoed = "".join(chr(int(round(v))) for v in out.tolist()[: len(text)])
            return Result(output={"text": echoed, "device": self.device})

        values = job.input.get("values")
        if not isinstance(values, list):
            # Neither a chat/prompt job nor a numeric vector — raising here
            # exercises the runtime's 500 path while keeping the server alive.
            raise ValueError("input must carry 'messages', 'prompt', or numeric 'values'")
        x = torch.tensor(values, dtype=torch.float32, device=self.device)
        if x.shape[-1] != self.size:
            raise ValueError(
                f"input.values must have length {self.size}, got {x.shape[-1]}"
            )
        with torch.no_grad():
            y = self.model(x)
        return Result(output={"values": y.tolist(), "device": self.device})
