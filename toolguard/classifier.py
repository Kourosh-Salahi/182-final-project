"""Task-conditioned span classifier.

A small encoder reads `[CLS] trusted task [SEP] tool output [SEP]` and tags
each tool-output token as injected (1) or benign (0). Task tokens are never
labelled, they are only context: the same sentence can be benign when it
restates the user's task and an injection when it names another goal.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from .dataset import Example, load_split
from .spans import Span, merge_spans, span_report

DEFAULT_MODEL = "distilbert-base-uncased"  # or "answerdotai/ModernBERT-base"


def pick_device(device: str | None = None) -> str:
    import torch

    if device:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def encode(tokenizer, task: str, text: str, spans: list | None = None, max_length: int = 512) -> dict:
    """Tokenize one (task, tool output) pair; label tool-output tokens when spans are given."""
    enc = tokenizer(
        task,
        text,
        truncation="only_second",
        max_length=max_length,
        return_offsets_mapping=True,
    )
    seq_ids = enc.sequence_ids()
    enc["in_output"] = [sid == 1 for sid in seq_ids]
    if spans is not None:
        labels = []
        for sid, (s, e) in zip(seq_ids, enc["offset_mapping"]):
            if sid != 1 or e <= s:
                labels.append(-100)
            else:
                labels.append(int(any(s < ge and e > gs for gs, ge in spans)))
        enc["labels"] = labels
    return enc


def _collate(tokenizer, rows: list[dict]):
    import torch

    width = max(len(r["input_ids"]) for r in rows)
    pad = tokenizer.pad_token_id
    batch = {
        "input_ids": torch.tensor([r["input_ids"] + [pad] * (width - len(r["input_ids"])) for r in rows]),
        "attention_mask": torch.tensor([r["attention_mask"] + [0] * (width - len(r["input_ids"])) for r in rows]),
    }
    if "labels" in rows[0]:
        batch["labels"] = torch.tensor([r["labels"] + [-100] * (width - len(r["labels"])) for r in rows])
    return batch


class SpanClassifier:
    def __init__(self, model, tokenizer, device: str | None = None, threshold: float = 0.5, max_length: int = 512):
        self.device = pick_device(device)
        self.model = model.to(self.device).eval()
        self.tokenizer = tokenizer
        self.threshold = threshold
        self.max_length = max_length

    @classmethod
    def load(cls, path: str | Path, device: str | None = None, threshold: float = 0.5) -> "SpanClassifier":
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        return cls(
            AutoModelForTokenClassification.from_pretrained(path),
            AutoTokenizer.from_pretrained(path),
            device=device,
            threshold=threshold,
        )

    def predict_spans(self, tasks: list[str], texts: list[str], batch_size: int = 32) -> list[list[Span]]:
        """Predicted injected character spans for each (task, tool output) pair."""
        import torch

        out: list[list[Span]] = []
        for i in range(0, len(texts), batch_size):
            rows = [
                encode(self.tokenizer, t, x, max_length=self.max_length)
                for t, x in zip(tasks[i : i + batch_size], texts[i : i + batch_size])
            ]
            batch = {k: v.to(self.device) for k, v in _collate(self.tokenizer, rows).items()}
            with torch.no_grad():
                probs = self.model(**batch).logits.softmax(-1)[..., 1].cpu()
            for j, row in enumerate(rows):
                text = texts[i + j]
                spans = [
                    (s, e)
                    for k, (s, e) in enumerate(row["offset_mapping"])
                    if row["in_output"][k] and e > s and probs[j, k] >= self.threshold
                ]
                # Join adjacent flagged tokens across whitespace and punctuation.
                out.append(merge_spans(spans, text, max_gap=3))
        return out

    def evaluate(self, examples: list[Example]) -> dict:
        preds = self.predict_spans([e.task for e in examples], [e.text for e in examples])
        golds = [[tuple(s) for s in e.spans] for e in examples]
        texts = [e.text for e in examples]
        report = span_report(preds, golds, texts)
        by_family = {}
        for fam in sorted({e.family for e in examples if e.family}):
            idx = [i for i, e in enumerate(examples) if e.family == fam]
            sub = span_report([preds[i] for i in idx], [golds[i] for i in idx], [texts[i] for i in idx])
            by_family[fam] = {"f1": sub["f1"], "removed_fraction": sub["removed_fraction"], "n": len(idx)}
        report["by_family"] = by_family
        return report


def train(
    train_path: str | Path,
    val_path: str | Path | None,
    out_dir: str | Path,
    model_name: str = DEFAULT_MODEL,
    epochs: int = 3,
    lr: float = 5e-5,
    batch_size: int = 16,
    max_length: int = 512,
    seed: int = 0,
    device: str | None = None,
    limit: int | None = None,
    log=print,
) -> SpanClassifier:
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer, get_linear_schedule_with_warmup

    random.seed(seed)
    torch.manual_seed(seed)
    device = pick_device(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name, num_labels=2, id2label={0: "BENIGN", 1: "INJECTED"}, label2id={"BENIGN": 0, "INJECTED": 1}
    ).to(device)

    examples = load_split(train_path)[:limit]
    rows = [encode(tokenizer, e.task, e.text, e.spans, max_length) for e in examples]
    val = load_split(val_path)[:limit] if val_path else []

    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * ((len(rows) + batch_size - 1) // batch_size)
    sched = get_linear_schedule_with_warmup(optim, int(0.06 * steps), steps)

    out_dir = Path(out_dir)
    best_f1, history = -1.0, []
    log(f"training {model_name} on {len(rows)} examples for {epochs} epochs ({device})")
    for epoch in range(epochs):
        model.train()
        random.shuffle(rows)
        total = 0.0
        for i in range(0, len(rows), batch_size):
            batch = {k: v.to(device) for k, v in _collate(tokenizer, rows[i : i + batch_size]).items()}
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            optim.zero_grad()
            total += loss.item() * len(batch["input_ids"])
        record = {"epoch": epoch + 1, "train_loss": total / len(rows)}
        clf = SpanClassifier(model, tokenizer, device, max_length=max_length)
        if val:
            report = clf.evaluate(val)
            record.update(val_f1=report["f1"], val_false_alarm_rate=report["false_alarm_rate"])
        log(json.dumps(record))
        history.append(record)
        # Keep the best epoch by validation span F1 (the last one when there is no val set).
        f1 = record.get("val_f1", epoch)
        if f1 > best_f1:
            best_f1 = f1
            model.save_pretrained(out_dir)
            tokenizer.save_pretrained(out_dir)
    (out_dir / "train_history.json").write_text(json.dumps(history, indent=2))
    return SpanClassifier.load(out_dir, device)
