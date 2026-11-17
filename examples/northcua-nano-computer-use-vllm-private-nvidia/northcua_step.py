"""One NorthCUA step: screenshot + task in, reasoning + GUI tool calls out.

Builds the request the way the official NorthCUA Harness does for its Desktop
agent (Chat Completions): the Harness system prompt rendered for the
`normalized_1000` coordinate policy, the persistent task message, an
observation message with the current screenshot, and the Harness's native
desktop tools (click, double_click, drag, move, scroll, type, keypress, wait,
screenshot, note). The prompt templates and tool schemas in `prompts/` are
copied from beipo-ai/NorthCUA (Apache-2.0).

Coordinates come back in the 0..1000 space on both axes; `to_pixels()` maps
them to the screenshot once, at the execution boundary, as the model card asks.
"""

from __future__ import annotations

import base64
import json
import time
from pathlib import Path

import requests
from jinja2 import Environment, StrictUndefined

HERE = Path(__file__).resolve().parent
PROMPTS = HERE / "prompts"
TOOLS = json.loads((PROMPTS / "desktop_tools.json").read_text())

_env = Environment(autoescape=False, trim_blocks=True, lstrip_blocks=True, undefined=StrictUndefined)


def _render(name: str, **values) -> str:
    return _env.from_string((PROMPTS / name).read_text()).render(**values).strip()


def system_prompt(width: int, height: int, environment: str = "linux", step_limit: int | None = None) -> str:
    return _render(
        "computer_using_agent_system.j2",
        screenshot_width=width,
        screenshot_height=height,
        desktop_environment=environment,
        home_directory="",
        coordinate_space="normalized_1000",
        coordinate_instruction="Return x,y normalized to [0,1000] relative to the freshest grounding screenshot.",
        coordinate_is_normalized_1000=True,
        coordinate_is_image_pixels=False,
        step_budget={"available": bool(step_limit), "round_limit": step_limit},
    )


def task_prompt(task: str) -> str:
    return _render("computer_using_agent_persistent_user.j2", task=task)


def observation(png: bytes, image_count: int = 1) -> dict:
    text = _render(
        "computer_using_agent_observation_user.j2",
        platform_name="linux",
        total_image_count=image_count,
        history_image_count=image_count - 1,
        has_current_image=True,
        current_image_role="current",
        observation_metadata={},
    )
    url = "data:image/png;base64," + base64.b64encode(png).decode()
    return {"role": "user", "content": [{"type": "text", "text": text}, {"type": "image_url", "image_url": {"url": url}}]}


def call(url: str, api_key: str, model: str, messages: list, *, max_tokens: int = 4096,
         temperature: float = 0.7, top_p: float = 0.95, seed: int | None = None, timeout: int = 600) -> dict:
    """One Chat Completions call with the Harness tools. Returns the message
    plus timing and usage."""
    body = {"model": model, "messages": messages, "tools": TOOLS, "tool_choice": "auto",
            "max_tokens": max_tokens, "temperature": temperature, "top_p": top_p}
    if seed is not None:
        body["seed"] = seed
    t0 = time.monotonic()
    r = requests.post(url, headers={"Authorization": f"Bearer {api_key}"}, json=body, timeout=timeout)
    if r.status_code >= 400:
        raise requests.HTTPError(f"{r.status_code}: {r.text[:500]}", response=r)
    data = r.json()
    msg = data["choices"][0]["message"]
    return {
        "content": msg.get("content") or "",
        "reasoning": msg.get("reasoning_content") or msg.get("reasoning") or "",
        "tool_calls": msg.get("tool_calls") or [],
        "finish_reason": data["choices"][0].get("finish_reason"),
        "usage": data.get("usage") or {},
        "seconds": round(time.monotonic() - t0, 2),
    }


def to_pixels(x: float, y: float, width: int, height: int) -> tuple[int, int]:
    """normalized_1000 -> screenshot pixels (clamped to the image)."""
    px = round(min(max(float(x), 0.0), 1000.0) / 1000.0 * (width - 1))
    py = round(min(max(float(y), 0.0), 1000.0) / 1000.0 * (height - 1))
    return px, py


def args(tool_call: dict) -> dict:
    raw = tool_call.get("function", {}).get("arguments") or "{}"
    return json.loads(raw) if isinstance(raw, str) else raw
