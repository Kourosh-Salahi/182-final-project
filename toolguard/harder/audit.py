"""Filtering raw generator output into pools, and the hand-audit sheet.

`run_filter` parses every raw reply, filters each kind, and writes the pools, the rejected
lines (with reasons) and `audit_sheet.md`. A person ticks bad items on the sheet
(`- [x] benign:12 | ...`); `apply_audit` removes them from the pools.
"""

from __future__ import annotations

import random
import re
from collections import Counter
from pathlib import Path

from .filters import Embedder, filter_pool
from .pools import KINDS, load_briefs, parse_raw, read_jsonl, write_jsonl

POOL_FILES = {"benign": "benign.jsonl", "attack": "attacks.jsonl", "contrast": "contrast_wordings.jsonl"}
AUDIT_SAMPLE = 50
_TICKED = re.compile(r"^- \[[xX]\] (benign|attack|contrast):(\d+) \|")


def _text(rec: dict) -> str:
    return rec["text"] if "text" in rec else rec["template"]


def _group_key(rec: dict) -> str:
    return rec.get("object") or rec.get("family") or "contrast"


def run_filter(root: str | Path, embed: Embedder | None = None) -> dict:
    """Raw replies → pools, rejected.jsonl and audit_sheet.md under `root`. Returns counts."""
    root = Path(root)
    briefs = load_briefs(root / "briefs" / "index.jsonl")
    records: dict[str, list[dict]] = {k: [] for k in KINDS}
    rejected: list[dict] = []
    for brief in briefs:
        raw_path = root / "raw" / f"{brief.id}.txt"
        if not raw_path.exists():
            raise FileNotFoundError(f"missing raw output for brief '{brief.id}': {raw_path}")
        recs, skipped = parse_raw(brief, raw_path.read_text())
        records[brief.kind] += recs
        rejected += [{**s, "reasons": [s["reason"]]} for s in skipped]

    counts: dict = {}
    sheet = ["# Hand audit", "",
             "Tick `[x]` on any item that is mislabelled, a harmless line that pushes the robot to do a task,",
             "an attack that does not push the robot to act now, or a reworded copy of an existing line.", ""]
    for kind in KINDS:
        accepted, rej, near = filter_pool(records[kind], kind, embed)
        write_jsonl(root / "pools" / POOL_FILES[kind], accepted)
        rejected += rej
        counts[kind] = {
            "raw": len(records[kind]),
            "accepted": len(accepted),
            "by_group": dict(Counter(_group_key(r) for r in accepted)),
            "reject_reasons": dict(Counter(reason for r in rej for reason in r["reasons"])),
        }
        index = {id(r): i for i, r in enumerate(accepted)}
        sample = random.Random(0).sample(accepted, min(AUDIT_SAMPLE, len(accepted)))
        near_idx = {_text(n) for n in near}
        flagged = [r for r in accepted if _text(r) in near_idx and r not in sample]
        sheet += [f"## {kind} (random sample of {len(sample)}, then {len(flagged)} most similar to existing data)", ""]
        sheet += [f"- [ ] {kind}:{index[id(r)]} | {_text(r)}" for r in sample + flagged]
        sheet.append("")
    write_jsonl(root / "rejected.jsonl", rejected)
    (root / "audit_sheet.md").write_text("\n".join(sheet))
    return counts


def apply_audit(root: str | Path, sheet: str = "audit_sheet.md", report: str = "audit.md") -> list[dict]:
    """Remove ticked items from the pools; log them as rejected with reason "audit"."""
    root = Path(root)
    ticked: dict[str, set[int]] = {k: set() for k in KINDS}
    for line in (root / sheet).read_text().splitlines():
        m = _TICKED.match(line.strip())
        if m:
            ticked[m.group(1)].add(int(m.group(2)))

    removed: list[dict] = []
    for kind in KINDS:
        path = root / "pools" / POOL_FILES[kind]
        pool = read_jsonl(path)
        bad = [i for i in ticked[kind] if i >= len(pool)]
        if bad:
            raise ValueError(f"audit sheet points past the end of the {kind} pool: {sorted(bad)}")
        removed += [{**r, "reasons": ["audit"]} for i, r in enumerate(pool) if i in ticked[kind]]
        write_jsonl(path, [r for i, r in enumerate(pool) if i not in ticked[kind]])

    write_jsonl(root / "rejected.jsonl", read_jsonl(root / "rejected.jsonl") + removed)
    lines = ["# Hand audit result", "", f"Removed {len(removed)} items after reading {sheet}.", ""]
    lines += [f"- {k}: {len(ticked[k])} removed" for k in KINDS] + [""]
    lines += [f"- {_text(r)}" for r in removed]
    (root / report).write_text("\n".join(lines) + "\n")
    return removed


def refilter_benign(root: str | Path, embed: Embedder | None, sheet: str) -> dict:
    """Re-run the harmless-line filters on pools/benign.jsonl and write a review sheet of edited lines."""
    root = Path(root)
    pool = read_jsonl(root / "pools" / POOL_FILES["benign"])
    accepted, rej, near = filter_pool(pool, "benign", embed)
    write_jsonl(root / "pools" / POOL_FILES["benign"], accepted)
    tagged = [{**r, "round": "scenefix"} for r in rej]
    write_jsonl(root / "rejected.jsonl", read_jsonl(root / "rejected.jsonl") + tagged)
    edited = [i for i, r in enumerate(accepted) if r["source"].endswith("+scenefix")]
    sample = sorted(random.Random(0).sample(edited, min(AUDIT_SAMPLE, len(edited))))
    near_ids = {_text(n) for n in near}
    flagged = [i for i, r in enumerate(accepted) if _text(r) in near_ids and i not in sample]
    lines = ["# Hand audit, round 2 (scene fix)", "",
             "Tick `[x]` on any edited note that still mentions something outside the scene, contradicts it,",
             "pushes the robot to do a task, or reads as a reworded copy of an existing line.", "",
             f"## benign (random sample of {len(sample)} edited lines, "
             f"then {len(flagged)} most similar to existing data)", ""]
    lines += [f"- [ ] benign:{i} | {accepted[i]['text']}" for i in sample + flagged]
    (root / sheet).write_text("\n".join(lines) + "\n")
    return {"accepted": len(accepted), "rejected": dict(Counter(x for r in rej for x in r["reasons"])),
            "edited": len(edited), "on_sheet": len(sample) + len(flagged)}
