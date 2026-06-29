"""A self-contained **text** custom backend: a tiny sentiment classifier.

This is the representative *text* example for the SDK contract (the product's
dominant case). No weights are downloaded: ``setup()`` builds a minuscule
``nn.Embedding`` + ``nn.Linear`` model in memory over a small fixed vocabulary,
seeded deterministically so a couple of obvious positive/negative words steer
the prediction. ``process()`` tokenizes ``input["text"]`` (or ``input["prompt"]``)
by whitespace, runs one forward pass, and returns a label.

It demonstrates that the contract models real text jobs:

* input  — ``{"text": "..."}`` or ``{"prompt": "..."}``
* output — ``{"label": "positive"|"negative", "score": float}``

Run it (from the SDK repo root, with this folder importable)::

    python -m inferencekey.backend.serve \\
        --port 8099 \\
        --backend backend:SentimentBackend \\
        --config-json '{"device": "cpu"}'
"""

from __future__ import annotations

import torch
from torch import nn

from inferencekey.backend import BackendContext, CustomBackend, Job, Result

#: A small fixed vocabulary; everything else maps to the <unk> slot (index 0).
_VOCAB = (
    "<unk>",
    "good", "great", "love", "excellent", "happy", "wonderful", "best",
    "bad", "terrible", "hate", "awful", "sad", "worst", "horrible",
)
_POSITIVE = {"good", "great", "love", "excellent", "happy", "wonderful", "best"}
_NEGATIVE = {"bad", "terrible", "hate", "awful", "sad", "worst", "horrible"}


class SentimentBackend(CustomBackend):
    """A tiny in-memory sentiment classifier over a fixed vocabulary."""

    # Declarative metadata exposed via GET /meta (C-1).
    name = "tiny-sentiment"
    version = "0.1.0"
    task_type = "classification"
    requirements = "requirements.txt"

    def setup(self, ctx: BackendContext) -> None:
        # Read device explicitly from config (default "cpu"); no GPU
        # autodetection, no weights fetched from disk or network.
        self.device = str(ctx.config.get("device", "cpu"))
        self.stoi = {tok: i for i, tok in enumerate(_VOCAB)}

        embed_dim = 8
        embedding = nn.Embedding(len(_VOCAB), embed_dim)
        linear = nn.Linear(embed_dim, 2)  # logits: [negative, positive]

        # Seed the weights so known words push toward the right class. We hand
        # each vocab word a one-hot-ish embedding and let the linear layer route
        # positive/negative words to their class. This is a real forward pass,
        # not a lookup table.
        with torch.no_grad():
            embedding.weight.zero_()
            linear.weight.zero_()
            linear.bias.zero_()
            for tok, idx in self.stoi.items():
                if tok in _POSITIVE:
                    embedding.weight[idx, 0] = 1.0
                    linear.weight[1, 0] = 1.0  # positive logit
                elif tok in _NEGATIVE:
                    embedding.weight[idx, 1] = 1.0
                    linear.weight[0, 1] = 1.0  # negative logit

        self.embedding = embedding.to(self.device).eval()
        self.linear = linear.to(self.device).eval()

    def process(self, job: Job) -> Result:
        text = job.input.get("text", job.input.get("prompt"))
        if not isinstance(text, str):
            # Contract violation in the input -> exercises the runtime's 500
            # path while the server stays alive.
            raise ValueError("input must carry a string 'text' or 'prompt'")

        token_ids = [self.stoi.get(tok, 0) for tok in text.lower().split()]
        if not token_ids:
            token_ids = [0]
        ids = torch.tensor(token_ids, dtype=torch.long, device=self.device)
        with torch.no_grad():
            pooled = self.embedding(ids).mean(dim=0)  # mean over tokens
            logits = self.linear(pooled)
            probs = torch.softmax(logits, dim=-1)
        positive = bool(probs[1] >= probs[0])
        return Result(
            output={
                "label": "positive" if positive else "negative",
                "score": float(probs[1] if positive else probs[0]),
            }
        )
