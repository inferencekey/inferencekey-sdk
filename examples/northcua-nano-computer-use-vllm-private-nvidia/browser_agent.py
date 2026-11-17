"""A small NorthCUA agent loop that drives a real Chromium page with Playwright.

Each step: take a screenshot, send it with the task and the step history to
the NorthCUA-Nano endpoint (Harness prompt + desktop tools), execute the tool
calls it returns (click, double_click, type, keypress, scroll, drag, move,
wait), and repeat until the model answers with plain text. Coordinates come
back in 0..1000 and are mapped to the viewport once, here.

This is a stand-in for the official NorthCUA Harness, which you should use for
real desktops (it adds sandboxing, context compaction, retries and outcome
checks). It is enough to watch the model work on a web page.

Usage (after main.py has the endpoint ready, IK_KEEP_ALIVE=1):
  python browser_agent.py samples/expenses.html \
      "Add a new expense: 'Train to Lyon', 42.50 EUR, category Travel, and save it."
Frames with the model's clicks drawn on them go to outputs/<run>/.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import time
from pathlib import Path

from PIL import Image, ImageDraw
from playwright.sync_api import sync_playwright

import northcua_step as ncua

try:  # read .env if python-dotenv is installed
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

MODEL = "beipo-ai/NorthCUA-Nano"
SLUG = os.environ.get("NORTHCUA_SLUG", "northcua-nano")
KEYS = {"enter": "Enter", "return": "Enter", "tab": "Tab", "escape": "Escape", "esc": "Escape",
        "backspace": "Backspace", "delete": "Delete", "space": " ", "up": "ArrowUp", "down": "ArrowDown",
        "left": "ArrowLeft", "right": "ArrowRight", "home": "Home", "end": "End", "pageup": "PageUp",
        "pagedown": "PageDown", "ctrl": "Control", "control": "Control", "shift": "Shift", "alt": "Alt",
        "cmd": "Meta", "win": "Meta", "meta": "Meta", "super": "Meta"}


def chord(unit: str) -> str:
    return "+".join(KEYS.get(p.strip().lower(), p.strip() if len(p.strip()) > 1 else p.strip()) for p in unit.split("+"))


def execute(page, name: str, a: dict, w: int, h: int, marks: list) -> str:
    """Run one tool call on the page; return the text result the model sees."""
    if name in ("click", "double_click", "move"):
        x, y = ncua.to_pixels(a["x"], a["y"], w, h)
        marks.append((name, x, y))
        if name == "move":
            page.mouse.move(x, y)
        else:
            page.mouse.click(x, y, button=a.get("button", "left") or "left", click_count=2 if name == "double_click" else 1)
        return f"{name} at ({a['x']}, {a['y']}) done."
    if name == "type":
        page.keyboard.type(a.get("text", ""), delay=15)
        return f"Typed {len(a.get('text', ''))} characters."
    if name == "keypress":
        units = a.get("keys") or [a.get("key")]
        for u in units:
            if u:
                page.keyboard.press(chord(u))
        return f"Pressed {units}."
    if name == "scroll":
        if a.get("x") is not None and a.get("y") is not None:
            page.mouse.move(*ncua.to_pixels(a["x"], a["y"], w, h))
        page.mouse.wheel(-float(a.get("delta_x") or 0) * 100, -float(a.get("delta_y") or 0) * 100)
        return "Scrolled."
    if name == "drag":
        pts = [ncua.to_pixels(p["x"], p["y"], w, h) for p in a.get("path", [])]
        if len(pts) >= 2:
            page.mouse.move(*pts[0]); page.mouse.down()
            for p in pts[1:]:
                page.mouse.move(*p, steps=8)
            page.mouse.up()
        return "Dragged."
    if name == "wait":
        time.sleep(min(float(a.get("seconds", a.get("duration", 1)) or 1), 5))
        return "Waited."
    if name == "note":
        return f"[{a.get('kind', 'fact')}] {a.get('text', '')}"
    if name == "screenshot":
        return "Fresh screenshot attached."
    return f"Unknown tool {name}."


def run(page_url: str, task: str, *, max_steps: int, out: Path, url: str, key: str, width: int = 1280,
        height: int = 800, check: str | None = None, temperature: float = 0.7, seed: int | None = 7) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    log = {"task": task, "page": page_url, "steps": []}
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": width, "height": height})
        page.goto(page_url)
        messages = [{"role": "system", "content": ncua.system_prompt(width, height, step_limit=max_steps)},
                    {"role": "user", "content": ncua.task_prompt(task)}]
        t_start = time.monotonic()
        for step in range(1, max_steps + 1):
            time.sleep(0.4)
            png = page.screenshot()
            # Keep only the newest two screenshots in the prompt (older ones become text).
            with_img = [m for m in messages if m["role"] == "user" and isinstance(m["content"], list)
                        and any(c["type"] == "image_url" for c in m["content"])]
            for m in with_img[:-1]:
                m["content"] = [c for c in m["content"] if c["type"] == "text"] + [
                    {"type": "text", "text": "[older screenshot omitted]"}]
            messages.append(ncua.observation(png, image_count=2 if step > 1 else 1))
            r = ncua.call(url, key, MODEL, messages, temperature=temperature, seed=seed)
            marks: list = []
            entry = {"step": step, "seconds": r["seconds"], "usage": r["usage"], "content": r["content"],
                     "reasoning_chars": len(r["reasoning"]), "reasoning": r["reasoning"], "actions": []}
            assistant = {"role": "assistant", "content": r["content"], "tool_calls": r["tool_calls"]}
            if not r["tool_calls"]:
                assistant.pop("tool_calls")
            messages.append(assistant)
            for tc in r["tool_calls"]:
                name, a = tc["function"]["name"], ncua.args(tc)
                try:
                    res = execute(page, name, a, width, height, marks)
                except Exception as e:  # report the failure to the model, like a harness would
                    res = f"Error: {e}"
                entry["actions"].append({"tool": name, "args": a, "result": res})
                messages.append({"role": "tool", "tool_call_id": tc.get("id", ""), "content": res})
            # Save the frame the model looked at, with its actions drawn on it.
            img = Image.open(io.BytesIO(png)).convert("RGB")
            d = ImageDraw.Draw(img)
            for _, x, y in marks:
                d.ellipse([x - 16, y - 16, x + 16, y + 16], outline=(230, 40, 40), width=4)
            img.save(out / f"step-{step:02d}.png")
            print(f"step {step} ({r['seconds']} s): {r['content'].strip()[:160]}")
            for act in entry["actions"]:
                print(f"   {act['tool']}({json.dumps(act['args'])})")
            log["steps"].append(entry)
            if not r["tool_calls"]:
                break
        time.sleep(0.4)
        page.screenshot(path=str(out / "final.png"))
        log["seconds_total"] = round(time.monotonic() - t_start, 1)
        if check:
            log["check"] = page.evaluate(check)
        log["state"] = page.evaluate("() => window.appState || null")
        browser.close()
    (out / "run.json").write_text(json.dumps(log, indent=2, ensure_ascii=False) + "\n")
    return log


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("page", help="URL or local HTML file")
    ap.add_argument("task")
    ap.add_argument("--max-steps", type=int, default=15)
    ap.add_argument("--out", default="outputs/browser-run")
    ap.add_argument("--check", default=None, help="JS expression evaluated at the end (your success test)")
    # 1920x1080 by default: in our test the model converted its pixel guesses to
    # 0..1000 as if every screenshot were 1920x1080; at 1280x800 its clicks drifted.
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    a = ap.parse_args()
    page_url = a.page if "://" in a.page else Path(a.page).resolve().as_uri()
    base = os.environ.get("INFERENCEKEY_BASE_URL", "https://cloud.inferencekey.com").rstrip("/")
    url = f"{base}/endpoint/{os.environ['INFERENCEKEY_PROJECT']}/{SLUG}/v1/chat/completions"
    log = run(page_url, a.task, max_steps=a.max_steps, out=Path(a.out), url=url,
              key=os.environ["INFERENCEKEY_API_KEY"], check=a.check, width=a.width, height=a.height)
    print(f"\n{len(log['steps'])} steps, {log['seconds_total']} s; check: {log.get('check')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
