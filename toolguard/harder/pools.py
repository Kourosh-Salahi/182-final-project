"""Briefs for the blind generators, parsing of their raw replies, and pool record I/O.

A brief is one request to a fresh subagent. Its reply is saved verbatim; `parse_raw`
turns it into pool records. Labels that matter (the scene object, the attack family)
come from the brief, never from the generated text.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

KINDS = ("benign", "attack", "contrast")

_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")
_TAGGED = re.compile(r"^([a-z_ ]+)\|\s*(.+)$")
_QUOTES = "\"'“”‘’"


@dataclass(frozen=True)
class Brief:
    id: str
    kind: str  # "benign" | "attack" | "contrast"
    target: str  # scene object name, "workspace", attack family name, or "contrast"


def load_briefs(path: str | Path) -> list[Brief]:
    briefs = [Brief(**row) for row in read_jsonl(path)]
    bad = [b.id for b in briefs if b.kind not in KINDS]
    if bad:
        raise ValueError(f"briefs with unknown kind: {bad}")
    return briefs


def _clean(line: str) -> str:
    return _BULLET.sub("", line).strip().strip(_QUOTES).strip()


def _record(brief: Brief, tag: str | None, text: str) -> dict:
    if brief.kind == "benign":
        return {"text": text, "object": brief.target, "style": tag, "source": brief.id}
    if brief.kind == "attack":
        return {"template": text, "family": brief.target, "shape": tag, "source": brief.id}
    return {"template": text, "source": brief.id}


def parse_raw(brief: Brief, raw: str) -> tuple[list[dict], list[dict]]:
    """(records, skipped lines with a reason) from one generator reply."""
    records: list[dict] = []
    skipped: list[dict] = []
    for line in raw.splitlines():
        cleaned = _clean(line)
        if not cleaned:
            continue
        if brief.kind == "contrast":
            records.append(_record(brief, None, cleaned))
            continue
        m = _TAGGED.match(cleaned)
        text = _clean(m.group(2)) if m else ""
        if not m or not text:
            skipped.append({"line": line, "reason": "no tag", "source": brief.id})
            continue
        records.append(_record(brief, m.group(1).strip(), text))
    return records, skipped


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
