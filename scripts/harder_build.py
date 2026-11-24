"""Build the OOD splits, the contrast splits and the robot replay file from the audited pools.

    python scripts/harder_build.py

Reads data_gen/pools/*.jsonl and writes data_gen/ood/, data_gen/contrast/,
data_gen/ood_replay.jsonl and data_gen/MANIFEST.json (SHA-256 of every data and
provenance file; CONTEXT.md and DECISIONS.md are left out so docs can be edited).
Deterministic: running it twice on the same pools gives the same MANIFEST.
"""

import argparse
import hashlib
import json
from pathlib import Path

from toolguard.dataset import save_dataset
from toolguard.harder.audit import POOL_FILES
from toolguard.harder.contrast import build_contrast, split_task_pairs, split_wordings
from toolguard.harder.ood import OOD_SPLITS, build_ood_split
from toolguard.harder.pools import read_jsonl, write_jsonl
from toolguard.harder.replay import build_replay

SEED = 0
N_OOD = 600
CONTRAST_PAIRS = {"train": 200, "test": 100}  # per kind
REPLAY_PER_FAMILY = 30
# Docs can be edited after the freeze without changing what the manifest pins.
UNHASHED = {"MANIFEST.json", ".DS_Store", "CONTEXT.md", "DECISIONS.md"}


def build(root: Path) -> dict:
    benign = read_jsonl(root / "pools" / POOL_FILES["benign"])
    attacks = read_jsonl(root / "pools" / POOL_FILES["attack"])
    wordings = read_jsonl(root / "pools" / POOL_FILES["contrast"])

    save_dataset({name: build_ood_split(name, benign, attacks, N_OOD, SEED) for name in OOD_SPLITS}, root / "ood")

    pairs = dict(zip(("train", "test"), split_task_pairs(SEED)))
    words = dict(zip(("train", "test"), split_wordings(wordings, SEED)))
    for split in ("train", "test"):
        rows, records = build_contrast(split, words[split], pairs[split], CONTRAST_PAIRS[split], SEED)
        save_dataset({split: rows}, root / "contrast")
        write_jsonl(root / "contrast" / f"{split}_pairs.jsonl", records)

    write_jsonl(root / "ood_replay.jsonl", build_replay(attacks, REPLAY_PER_FAMILY, SEED))
    return write_manifest(root)


def write_manifest(root: Path) -> dict:
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.name not in UNHASHED)
    manifest = {"seed": SEED, "files": {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                        for p in files}}
    (root / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="data_gen")
    root = Path(p.parse_args().root)
    build(root)
    for path in sorted(root.glob("ood/*.jsonl")) + sorted(root.glob("contrast/*.jsonl")) + [root / "ood_replay.jsonl"]:
        rows = read_jsonl(path)
        attacked = sum(bool(r.get("spans")) for r in rows) if "spans" in (rows[0] if rows else {}) else None
        suffix = f", {attacked} attacked" if attacked is not None else ""
        print(f"{path.relative_to(root)}: {len(rows)} rows{suffix}")


if __name__ == "__main__":
    main()
