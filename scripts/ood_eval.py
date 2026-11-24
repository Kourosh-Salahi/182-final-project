"""Score the span-removing defenses on the original test splits and the harder OOD splits.

Writes runs/ood_eval.json as {split: {defense: span_report}} (see data_gen/CONTEXT.md, section 5).

    python scripts/ood_eval.py --checkpoint checkpoints/span-clf
"""

import argparse
import json

from toolguard.dataset import load_split
from toolguard.defenses import make_defense
from toolguard.spans import span_report

SPLITS = {
    "test_seen": "data/test_seen.jsonl",
    "test_heldout": "data/test_heldout.jsonl",
    "ood_benign": "data_gen/ood/ood_benign.jsonl",
    "ood_attack": "data_gen/ood/ood_attack.jsonl",
    "ood_both": "data_gen/ood/ood_both.jsonl",
    "contrast_test": "data_gen/contrast/test.jsonl",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/span-clf")
    ap.add_argument("--defenses", nargs="+", default=["classifier", "heuristic", "protectai", "protectai_sent"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="runs/ood_eval.json")
    args = ap.parse_args()

    splits = {name: load_split(path) for name, path in SPLITS.items()}
    out = {name: {} for name in splits}
    for d in args.defenses:
        defense = make_defense(d, checkpoint=args.checkpoint, device=args.device)
        for name, rows in splits.items():
            preds = [defense.spans(r.task, r.text) for r in rows]
            report = span_report(preds, [[tuple(s) for s in r.spans] for r in rows], [r.text for r in rows])
            by_family = {}
            for fam in sorted({r.family for r in rows if r.family}):
                idx = [i for i, r in enumerate(rows) if r.family == fam]
                sub = span_report([preds[i] for i in idx], [[tuple(s) for s in rows[i].spans] for i in idx],
                                  [rows[i].text for i in idx])
                by_family[fam] = {"f1": sub["f1"], "removed_fraction": sub["removed_fraction"], "n": len(idx)}
            report["by_family"] = by_family
            out[name][d] = report
            print(f"{d:15s} {name:13s} F1={report['f1']:.3f} removed={report['removed_fraction']:.3f} "
                  f"benign_kept={report['benign_kept']:.3f} false_alarm={report['false_alarm_rate']:.3f}", flush=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
