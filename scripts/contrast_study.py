"""Task-contrast training study (data_gen/CONTEXT.md, section 6).

Four setups, several seeds each, same settings as scripts/classifier_study.py:

    A  train on data/train.jsonl                          task shown
    B  train on data/train.jsonl + contrast/train.jsonl   task shown
    C  as A, task hidden (replaced by the word "task" in training and testing)
    D  as B, task hidden

Each model is scored on test_seen, test_heldout, the three OOD splits and the contrast test set.
On the contrast test set it also reports, per pair kind (swap, near_miss, add_on):
    harmless_kept   nothing removed that overlaps the restated command in the harmless row
    attack_removed  at least half of the attack row's labelled span removed
    pair_accuracy   both of the above for the same pair
Writes runs/contrast_study.json. No checkpoints are kept.

    python scripts/contrast_study.py --seeds 0 1 2
"""

import argparse
import json
import random
from dataclasses import replace

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, get_linear_schedule_with_warmup

from toolguard.classifier import DEFAULT_MODEL, SpanClassifier, _collate, encode
from toolguard.dataset import load_split
from toolguard.spans import span_counts, span_report

NO_TASK = "task"
EVAL_SPLITS = {
    "test_seen": "data/test_seen.jsonl",
    "test_heldout": "data/test_heldout.jsonl",
    "ood_benign": "data_gen/ood/ood_benign.jsonl",
    "ood_attack": "data_gen/ood/ood_attack.jsonl",
    "ood_both": "data_gen/ood/ood_both.jsonl",
    "contrast_test": "data_gen/contrast/test.jsonl",
}
SETUPS = {"A": (False, True), "B": (True, True), "C": (False, False), "D": (True, False)}  # (contrast data, task shown)


def overlaps(spans, target):
    return any(s < target[1] and e > target[0] for s, e in spans)


def pair_metrics(preds, rows, pairs):
    out = {}
    for kind in sorted({p["kind"] for p in pairs}):
        kept = removed = both = n = 0
        for p in (p for p in pairs if p["kind"] == kind):
            h, a = p["harmless_row"], p["attack_row"]
            ok_h = not overlaps(preds[h], p["harmless_span"])
            gold = [tuple(s) for s in rows[a].spans]
            tp, _, fn = span_counts(preds[a], gold, len(rows[a].text))
            ok_a = (tp + fn) > 0 and tp / (tp + fn) >= 0.5
            kept, removed, both, n = kept + ok_h, removed + ok_a, both + (ok_h and ok_a), n + 1
        out[kind] = {"harmless_kept": kept / n, "attack_removed": removed / n, "pair_accuracy": both / n, "n": n}
    n = len(pairs)
    out["all"] = {k: sum(out[kind][k] * out[kind]["n"] for kind in out) / n
                  for k in ("harmless_kept", "attack_removed", "pair_accuracy")}
    out["all"]["n"] = n
    return out


def run(setup, seed, args, device, evals, pairs):
    use_contrast, show_task = SETUPS[setup]
    random.seed(seed)
    torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForTokenClassification.from_pretrained(args.model, num_labels=2).to(device)
    train = load_split("data/train.jsonl") + (load_split("data_gen/contrast/train.jsonl") if use_contrast else [])
    task_of = (lambda e: e.task) if show_task else (lambda e: NO_TASK)
    rows = [encode(tokenizer, task_of(e), e.text, e.spans) for e in train]

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * ((len(rows) + args.batch_size - 1) // args.batch_size)
    sched = get_linear_schedule_with_warmup(optim, int(0.06 * steps), steps)
    curve = []
    model.train()
    for _ in range(args.epochs):
        random.shuffle(rows)
        for i in range(0, len(rows), args.batch_size):
            batch = {k: v.to(device) for k, v in _collate(tokenizer, rows[i : i + args.batch_size]).items()}
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            optim.zero_grad()
            curve.append(loss.item())

    clf = SpanClassifier(model, tokenizer, device)
    metrics = {}
    for name, ex in evals.items():
        shown = ex if show_task else [replace(e, task=NO_TASK) for e in ex]
        preds = clf.predict_spans([e.task for e in shown], [e.text for e in shown])
        report = span_report(preds, [[tuple(s) for s in e.spans] for e in ex], [e.text for e in ex])
        metrics[name] = {k: report[k] for k in ("precision", "recall", "f1", "removed_fraction", "benign_kept",
                                                "false_alarm_rate")}
        if name == "contrast_test":
            metrics[name]["pairs"] = pair_metrics(preds, ex, pairs)
    c = metrics["contrast_test"]["pairs"]
    print(f"setup={setup} seed={seed} n_train={len(train)} seen_f1={metrics['test_seen']['f1']:.3f} "
          f"heldout_f1={metrics['test_heldout']['f1']:.3f} ood_both_f1={metrics['ood_both']['f1']:.3f} "
          f"pair_acc swap={c['swap']['pair_accuracy']:.2f} near_miss={c['near_miss']['pair_accuracy']:.2f} "
          f"add_on={c['add_on']['pair_accuracy']:.2f}", flush=True)
    return {"setup": setup, "seed": seed, "use_contrast": use_contrast, "task_shown": show_task,
            "n_train": len(train), "train_loss": curve, "metrics": metrics}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--setups", nargs="+", default=list(SETUPS), choices=list(SETUPS))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="runs/contrast_study.json")
    args = ap.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    evals = {name: load_split(path) for name, path in EVAL_SPLITS.items()}
    pairs = [json.loads(line) for line in open("data_gen/contrast/test_pairs.jsonl", encoding="utf-8") if line.strip()]
    runs = []
    for setup in args.setups:
        for seed in args.seeds:
            runs.append(run(setup, seed, args, device, evals, pairs))
            with open(args.out, "w") as f:  # rewritten after every run, so a crash loses at most one
                json.dump({"config": vars(args), "runs": runs}, f)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
