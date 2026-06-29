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

    def process(self, job: Job) -> Result:
        values = job.input.get("values")
        if not isinstance(values, list):
            # A contract violation in the *input* — raising here exercises the
            # runtime's 500 path while keeping the server alive.
            raise ValueError("input.values must be a list of numbers")
        x = torch.tensor(values, dtype=torch.float32, device=self.device)
        if x.shape[-1] != self.size:
            raise ValueError(
                f"input.values must have length {self.size}, got {x.shape[-1]}"
            )
        with torch.no_grad():
            y = self.model(x)
        return Result(output={"values": y.tolist(), "device": self.device})
