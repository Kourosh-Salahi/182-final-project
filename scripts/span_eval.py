"""Span-removal metrics of every span-removing defense on the two test splits, and a threshold
sweep for the classifier. Writes runs/span_eval.json. Runs on CPU in a few minutes.

    python scripts/span_eval.py
"""

import argparse
import json

from toolguard.dataset import load_split
from toolguard.defenses import make_defense
from toolguard.spans import span_report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--checkpoint", default="checkpoints/span-clf")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="runs/span_eval.json")
    args = ap.parse_args()

    splits = {name: load_split(f"{args.data}/{name}.jsonl") for name in ("val", "test_seen", "test_heldout")}
    golds = {name: [[tuple(s) for s in e.spans] for e in ex] for name, ex in splits.items()}
    out = {"defenses": {}, "threshold_sweep": []}
    for name in ("heuristic", "protectai", "protectai_sent", "classifier"):
        defense = make_defense(name, checkpoint=args.checkpoint, device=args.device)
        out["defenses"][name] = {}
        for split in ("test_seen", "test_heldout"):
            ex = splits[split]
            preds = [defense.spans(e.task, e.text) for e in ex]
            out["defenses"][name][split] = span_report(preds, golds[split], [e.text for e in ex])
            print(name, split, {k: round(v, 3) for k, v in out["defenses"][name][split].items()}, flush=True)
        if name == "classifier":
            for thr in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
                defense.clf.threshold = thr
                row = {"threshold": thr}
                for split in ("val", "test_heldout"):
                    ex = splits[split]
                    preds = [defense.spans(e.task, e.text) for e in ex]
                    r = span_report(preds, golds[split], [e.text for e in ex])
                    row[split] = {k: r[k] for k in ("precision", "recall", "f1", "false_alarm_rate")}
                out["threshold_sweep"].append(row)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
