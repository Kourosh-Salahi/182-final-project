"""Training curves, seed variance and the task-conditioning ablation for the span classifier.

Trains DistilBERT with and without the trusted task as input, for several seeds, logging the
training loss every step and the validation / held-out-family loss every `--eval-every` steps.
Writes runs/classifier_study.json (read by scripts/make_figures.py). No checkpoints are kept.

    python scripts/classifier_study.py --seeds 0 1 2
"""

import argparse
import json
import random

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, get_linear_schedule_with_warmup

from toolguard.classifier import DEFAULT_MODEL, SpanClassifier, _collate, encode
from toolguard.dataset import load_split

NO_TASK = "task"  # constant placeholder fed in place of the user's task in the ablation


def mean_loss(model, tokenizer, rows, device, batch_size=32):
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(rows), batch_size):
            batch = {k: v.to(device) for k, v in _collate(tokenizer, rows[i : i + batch_size]).items()}
            total += model(**batch).loss.item() * len(batch["input_ids"])
            n += len(batch["input_ids"])
    model.train()
    return total / n


def run(seed, use_task, args, device):
    random.seed(seed)
    torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForTokenClassification.from_pretrained(args.model, num_labels=2).to(device)
    task_of = (lambda e: e.task) if use_task else (lambda e: NO_TASK)
    splits = {name: load_split(f"{args.data}/{name}.jsonl") for name in ("train", "val", "test_seen", "test_heldout")}
    rows = {name: [encode(tokenizer, task_of(e), e.text, e.spans) for e in ex] for name, ex in splits.items()}

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * ((len(rows["train"]) + args.batch_size - 1) // args.batch_size)
    sched = get_linear_schedule_with_warmup(optim, int(0.06 * steps), steps)
    curve, evals, step = [], [], 0
    model.train()
    for _ in range(args.epochs):
        train = rows["train"][:]
        random.shuffle(train)
        for i in range(0, len(train), args.batch_size):
            if step % args.eval_every == 0:
                evals.append({"step": step,
                              "val_loss": mean_loss(model, tokenizer, rows["val"], device),
                              "heldout_loss": mean_loss(model, tokenizer, rows["test_heldout"][:300], device)})
            batch = {k: v.to(device) for k, v in _collate(tokenizer, train[i : i + args.batch_size]).items()}
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            optim.zero_grad()
            curve.append(loss.item())
            step += 1
    evals.append({"step": step, "val_loss": mean_loss(model, tokenizer, rows["val"], device),
                  "heldout_loss": mean_loss(model, tokenizer, rows["test_heldout"][:300], device)})

    clf = SpanClassifier(model, tokenizer, device)
    metrics = {}
    for name in ("test_seen", "test_heldout"):
        ex = splits[name]
        if not use_task:
            ex = [type(e)(**{**e.__dict__, "task": NO_TASK}) for e in ex]
        report = clf.evaluate(ex)
        metrics[name] = {k: report[k] for k in ("precision", "recall", "f1", "removed_fraction", "benign_kept",
                                                "false_alarm_rate")}
        metrics[name]["by_family"] = report["by_family"]
    print(f"seed={seed} use_task={use_task} seen_f1={metrics['test_seen']['f1']:.4f} "
          f"heldout_f1={metrics['test_heldout']['f1']:.4f}", flush=True)
    return {"seed": seed, "use_task": use_task, "train_loss": curve, "evals": evals, "metrics": metrics}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="runs/classifier_study.json")
    args = ap.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    runs = [run(seed, use_task, args, device) for use_task in (True, False) for seed in args.seeds]
    with open(args.out, "w") as f:
        json.dump({"config": vars(args), "runs": runs}, f)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
