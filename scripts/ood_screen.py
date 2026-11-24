"""Keep the new attack wordings that hijack the undefended robot (data_gen/CONTEXT.md, section 7, step 2).

Reads the no-defense screening run and the replay file it was run on, and writes a copy of the
replay file in which `evaded` is true only for rows whose screening trial ended in `attack_call`.
Screening looks only at the undefended run.

    python scripts/ood_screen.py
"""

import argparse
import json
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", default="data_gen/ood_replay.jsonl")
    ap.add_argument("--screen", default="runs/ood_screen/trials.jsonl")
    ap.add_argument("--out", default="runs/ood_replay_screened.jsonl")
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.replay, encoding="utf-8") if line.strip()]
    trials = [json.loads(line) for line in open(args.screen, encoding="utf-8") if line.strip()]
    # Trial i of a family used the i-th row of that family in the replay file.
    by_family = defaultdict(list)
    for t in trials:
        if t["defense"] == "none":
            by_family[t["family"]].append(t)
    seen, kept = defaultdict(int), defaultdict(int)
    for row in rows:
        fam = row["family"]
        i = seen[fam]
        seen[fam] += 1
        trial = by_family[fam][i] if i < len(by_family[fam]) else None
        if trial is not None:
            assert trial["task_id"] == row["task_id"] and trial["goal_id"] == row["goal_id"], (fam, i)
        row["evaded"] = bool(trial and trial["outcome"] == "attack_call")
        row["screen_outcome"] = trial["outcome"] if trial else None
        kept[fam] += row["evaded"]
    with open(args.out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    for fam in seen:
        print(f"{fam:20s} kept {kept[fam]}/{seen[fam]}")
    print(f"total kept {sum(kept.values())}/{len(rows)} -> {args.out}")


if __name__ == "__main__":
    main()
