"""Stand-in for DeepInfer's native ``RequestOutput``."""

from __future__ import annotations


class RequestOutput:
    """One generated token of one request, as one engine step reports it."""

    __slots__ = ("request_id", "token_id", "finished", "prompt_len", "finish_reason")

    def __init__(self, request_id: str, token_id: int, finished: bool, prompt_len: int,
                 finish_reason: str = "") -> None:
        self.request_id = request_id
        self.token_id = token_id
        self.finished = finished
        self.prompt_len = prompt_len
        self.finish_reason = finish_reason

    @staticmethod
    def clone(raw: "RequestOutput") -> "RequestOutput":
        return RequestOutput(raw.request_id, raw.token_id, raw.finished, raw.prompt_len, raw.finish_reason)
