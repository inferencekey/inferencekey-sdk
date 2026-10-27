"""Shared helpers: call an OpenAI-compatible transcription endpoint and score word error rate."""

from __future__ import annotations

import json
import time
from pathlib import Path

import requests


def transcribe(url: str, api_key: str, path: Path, model: str = "default", timeout: int = 900) -> dict:
    """POST one audio file as multipart (what openai.audio.transcriptions.create sends)."""
    t = time.time()
    with open(path, "rb") as f:
        resp = requests.post(
            url,
            headers={"Authorization": f"Bearer {api_key}"},
            files={"file": (path.name, f, "audio/wav")},
            data={"model": model},
            timeout=timeout,
        )
    resp.raise_for_status()
    out = resp.json()
    out["round_trip_s"] = round(time.time() - t, 3)
    return out


def load_clips(audio: str) -> tuple[list[Path], dict]:
    """A file, or a folder of audio files; references come from <folder>/manifest.json when present
    (a list of {"id": "<file stem>", "ref": "<reference transcript>"})."""
    p = Path(audio)
    if p.is_file():
        return [p], {}
    files = sorted(x for x in p.iterdir() if x.suffix.lower() in (".wav", ".flac", ".mp3", ".ogg", ".m4a"))
    refs = {}
    if (p / "manifest.json").is_file():
        refs = {m["id"]: m for m in json.loads((p / "manifest.json").read_text())}
    return files, refs


def wer(refs: list[str], hyps: list[str]) -> dict:
    """Corpus word error rate after the Whisper English normalizer (what the Open ASR Leaderboard uses)."""
    import jiwer
    from whisper_normalizer.english import EnglishTextNormalizer

    norm = EnglishTextNormalizer()
    pairs = [(norm(r), norm(h)) for r, h in zip(refs, hyps)]
    pairs = [(r, h) for r, h in pairs if r.strip()]
    if not pairs:
        return {}
    o = jiwer.process_words([r for r, _ in pairs], [h for _, h in pairs])
    words = sum(len(r.split()) for r, _ in pairs)
    return {"wer": round(100 * o.wer, 2), "words": words, "substitutions": o.substitutions,
            "deletions": o.deletions, "insertions": o.insertions}


def run(url: str, api_key: str, audio: str, out_dir: Path, model: str = "default") -> dict:
    """Transcribe every clip, save one JSON per clip, print a line each, return the summary."""
    out_dir.mkdir(parents=True, exist_ok=True)
    files, refs = load_clips(audio)
    rows = []
    for f in files:
        try:
            out = transcribe(url, api_key, f, model=model)
        except requests.HTTPError as e:
            # e.g. 413: the transcription route takes request bodies up to 2 MiB
            # (about 65 s of 16-bit 16 kHz WAV); send long audio as Opus or MP3.
            print(f"{f.stem:28s} FAILED {e.response.status_code}: {e.response.text[:160]}")
            rows.append({"id": f.stem, "error": f"{e.response.status_code}: {e.response.text[:300]}"})
            continue
        ref = refs.get(f.stem, {})
        row = {"id": f.stem, "set": ref.get("set"), "audio_s": ref.get("seconds") or out.get("duration"),
               "text": out.get("text", ""), "ref": ref.get("ref"), "gpu_s": out.get("seconds"),
               "round_trip_s": out["round_trip_s"], "vram_peak_gib": out.get("vram_peak_gib")}
        if row["ref"]:
            row.update({k: v for k, v in wer([row["ref"]], [row["text"]]).items() if k == "wer"})
        (out_dir / f"{f.stem}.json").write_text(json.dumps(row, indent=1, ensure_ascii=False))
        print(f"{f.stem:28s} {row['audio_s'] or 0:7.1f} s audio  round trip {row['round_trip_s']:6.2f} s"
              + (f"  WER {row['wer']:5.1f} %" if "wer" in row else "") + f"  {row['text'][:70]}")
        rows.append(row)
    failed = [r for r in rows if "error" in r]
    rows = [r for r in rows if "error" not in r]
    summary = {"clips": len(rows), "failed": len(failed), "audio_s": round(sum(r["audio_s"] or 0 for r in rows), 1),
               "round_trip_s": round(sum(r["round_trip_s"] for r in rows), 2)}
    scored = [r for r in rows if r.get("ref")]
    if scored:
        summary["overall"] = wer([r["ref"] for r in scored], [r["text"] for r in scored])
        by_set = {}
        for r in scored:
            by_set.setdefault(r["set"] or "all", []).append(r)
        summary["by_set"] = {s: wer([r["ref"] for r in rs], [r["text"] for r in rs]) for s, rs in by_set.items()}
    (out_dir / "summary.json").write_text(json.dumps({"summary": summary, "rows": rows, "failed": failed}, indent=1, ensure_ascii=False))
    print(json.dumps(summary, indent=1))
    return summary
