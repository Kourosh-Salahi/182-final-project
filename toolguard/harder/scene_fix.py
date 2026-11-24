"""Make generated harmless notes match the robot's scene.

The first generation round only gave each writer an object name, so some notes describe
things that are not in the scene (a fridge, mugs, a gas stove) or contradict its facts.
A second round of blind agents got a scene card and, for each note, answered
`N| keep`, `N| fix| new text` or `N| drop`. This module applies those answers and
records every decision in `scene_fix.jsonl`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .pools import read_jsonl, write_jsonl

_ANSWER = re.compile(r"^\s*(\d+)\s*\|\s*(keep|fix|drop)\s*(?:\|\s*(.+?))?\s*$", re.IGNORECASE)


def parse_fix(raw: str, n_lines: int) -> list[tuple[str, str | None]]:
    """One (decision, new text or None) per input line. Missing or broken answers become drops."""
    answers: dict[int, tuple[str, str | None]] = {}
    for line in raw.splitlines():
        m = _ANSWER.match(line)
        if not m:
            continue
        n, decision, text = int(m.group(1)), m.group(2).lower(), m.group(3)
        if decision == "fix" and not text:
            decision = "drop"
        answers.setdefault(n, (decision, text.strip(" \"'") if text else None))
    return [answers.get(n, ("drop", None)) for n in range(n_lines)]


def apply_scene_fix(root: str | Path) -> dict:
    """Rewrite pools/benign.jsonl from the fix answers; write scene_fix.jsonl. Returns counts."""
    root = Path(root)
    pool = read_jsonl(root / "pools" / "benign.jsonl")
    index = json.loads((root / "scene_fix_index.json").read_text())
    decided: dict[int, tuple[str, str | None]] = {}
    for entry in index:
        raw_path = root / "raw" / f"{entry['id']}.txt"
        if not raw_path.exists():
            raise FileNotFoundError(f"missing scene-fix answers for '{entry['object']}': {raw_path}")
        answers = parse_fix(raw_path.read_text(), len(entry["pool_index"]))
        for pool_i, answer in zip(entry["pool_index"], answers):
            decided[pool_i] = answer
    if set(decided) != set(range(len(pool))):
        raise ValueError("scene-fix index does not cover the benign pool exactly")

    log, new_pool = [], []
    for i, rec in enumerate(pool):
        decision, text = decided[i]
        new_text = rec["text"] if decision == "keep" else text
        log.append({"pool_index": i, "object": rec["object"], "style": rec["style"],
                    "original": rec["text"], "decision": decision, "text": new_text})
        if decision == "keep":
            new_pool.append(rec)
        elif decision == "fix":
            new_pool.append({**rec, "text": new_text, "source": rec["source"] + "+scenefix"})
    write_jsonl(root / "scene_fix.jsonl", log)
    write_jsonl(root / "pools" / "benign.jsonl", new_pool)
    return {d: sum(r["decision"] == d for r in log) for d in ("keep", "fix", "drop")}
