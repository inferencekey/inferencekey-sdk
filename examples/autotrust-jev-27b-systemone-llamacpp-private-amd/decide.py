"""Ask AutoTrust-JEV-27B ("Jev") one driving decision through its System One API.

Jev does not generate text: it reads a state and a question with options and
returns a probability for every option. This script sends one real request from
our Crazy Taxi test (a taxi with a passenger, at 37 yd/s, the game's guide arrow
27 degrees to the right) and prints the chosen option, the probabilities and the
latency measured on the client, end to end.

    python decide.py                 # the game's own request: 2 options
    python decide.py --options all   # all 5 full-throttle steering options
    python decide.py --runs 5        # repeat it, then print min / median / max

It calls `{base}/endpoint/{project}/{workload}/v1/systemone` with your ik_live_
key, or the URL in JEV_URL (for example your own llama-server, no key needed).
Results go to outputs/decide.json.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent

# The state the game loop sent at tick 37 of one 10 minute game (2026-11-23):
# a passenger on board, the destination 439 yd away, the route arrow slightly
# right. Distances are in yards, angles in degrees (negative is left).
STATE = {
    "passenger_onboard": True,
    "destination_direction": "slightly_right",
    "destination_angle_deg": 32,
    "customer_visible": False,
    "customer_direction": None,
    "customer_distance": None,
    "customer_fare": None,
    "stuck": False,
    "destination_distance_yd": 439,
    "at_destination": False,
    "route_arrow": "slightly_right",
    "route_arrow_angle_deg": 27,
    "speed": "very_fast",
    "speed_yd_s": 37,
    "hit_wall": False,
    "time_left_s": 587.3,
    "passenger_time_left_s": 39.0,
    "previous_action": "accelerate_straight",
    "map_distance_yd": 531,
    "target_side": "right",
    "target_angle_deg": 27,
}

INSTRUCTIONS = (
    "You drive a taxi in Crazy Taxi. Choose the best driving action for the next half second. "
    "Without a passenger, reach customers quickly, preferring long fares. With a passenger, "
    "reach the destination as fast as possible. "
    "Navigation: route_arrow is the game's guide arrow and shows which street to take; it leads. "
    "destination_direction is the straight line to the destination, through buildings if need "
    "be: use it only when there is no route_arrow, and never steer into a building to follow it. "
    "Steering: at speed (speed_yd_s high) prefer small corrections and brake before a sharp turn; "
    "turn hard only for real turns. Keep previous_action when it still fits: do not zigzag "
    "between left and right on small differences. "
    "Pickup: approach the customer's circle precisely, without overshooting; after a failed stop "
    "make small moves toward the customer."
)

STEERING = {
    "accelerate_straight": "Full throttle, wheel centred.",
    "accelerate_slight_left": "Full throttle with a gentle correction to the left.",
    "accelerate_slight_right": "Full throttle with a gentle correction to the right.",
    "accelerate_left": "Full throttle while turning left.",
    "accelerate_right": "Full throttle while turning right.",
}
# The game's rules offered only the turns that fit a 27 degree target at speed;
# Jev chose between them. `--options all` offers every full-throttle option.
GAME_OPTIONS = ["accelerate_slight_right", "accelerate_straight"]


def _load_dotenv() -> None:
    env = HERE / ".env"
    if env.is_file():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())


def endpoint_url() -> str:
    if os.environ.get("JEV_URL"):
        return os.environ["JEV_URL"]
    base = os.environ.get("INFERENCEKEY_BASE_URL", "https://cloud.inferencekey.com").rstrip("/")
    project = os.environ["INFERENCEKEY_PROJECT"]
    workload = os.environ.get("JEV_WORKLOAD", "autotrust-jev-27b-llamacpp-amd")
    return f"{base}/endpoint/{project}/{workload}/v1/systemone"


def build_request(options: list) -> dict:
    """One `choice` question. The keys of `criteria` are the option names Jev
    returns; the values describe each option to the model."""
    return {
        "state": STATE,
        "questions": {
            "action": {
                "type": "choice",
                "instructions": INSTRUCTIONS,
                "criteria": {name: STEERING[name] for name in options},
            }
        },
    }


def ask(session: requests.Session, url: str, body: dict) -> tuple:
    start = time.perf_counter()
    response = session.post(url, json=body, timeout=30)
    response.raise_for_status()
    return response.json(), time.perf_counter() - start


def main() -> int:
    _load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--options", choices=["game", "all"], default="game")
    parser.add_argument("--runs", type=int, default=1)
    args = parser.parse_args()

    url = endpoint_url()
    session = requests.Session()
    if os.environ.get("INFERENCEKEY_API_KEY"):
        session.headers["Authorization"] = f"Bearer {os.environ['INFERENCEKEY_API_KEY']}"
    body = build_request(GAME_OPTIONS if args.options == "game" else list(STEERING))

    calls = []
    try:
        for n in range(1, args.runs + 1):
            reply, seconds = ask(session, url, body)
            answer = reply["answers"]["action"]
            probabilities = dict(sorted(answer["probabilities"].items(), key=lambda kv: -kv[1]))
            print(f"call {n}: {answer['choice']}  ({seconds * 1000:.0f} ms, "
                  f"{reply.get('usage', {}).get('input_tokens')} input tokens)")
            for name, p in probabilities.items():
                print(f"    {p:6.3f}  {name}")
            calls.append({"choice": answer["choice"], "probabilities": probabilities,
                          "confidence": answer.get("confidence"), "seconds": round(seconds, 3),
                          "usage": reply.get("usage")})
    except (requests.RequestException, KeyError) as e:
        # The URL can carry no secret; the key lives in a header and is never printed.
        print(f"error: {e}", file=sys.stderr)
        return 1

    if len(calls) > 1:
        s = [c["seconds"] for c in calls]
        print(f"latency over {len(s)} calls: min {min(s):.3f} s, "
              f"median {statistics.median(s):.3f} s, max {max(s):.3f} s")
    out = HERE / "outputs"
    out.mkdir(exist_ok=True)
    (out / "decide.json").write_text(json.dumps({"request": body, "calls": calls}, indent=2) + "\n")
    print(f"saved {out / 'decide.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
