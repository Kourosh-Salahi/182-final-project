"""Character-span helpers and span-level metrics."""

from __future__ import annotations

import re

Span = tuple[int, int]


def merge_spans(spans: list[Span], text: str = "", max_gap: int = 0) -> list[Span]:
    """Merge overlapping spans, and spans separated only by a short non-alphanumeric gap."""
    out: list[Span] = []
    for s, e in sorted((int(s), int(e)) for s, e in spans if e > s):
        if out:
            ps, pe = out[-1]
            if s <= pe or (s - pe <= max_gap and not re.search(r"[A-Za-z0-9]", text[pe:s])):
                out[-1] = (ps, max(pe, e))
                continue
        out.append((s, e))
    return out


def remove_spans(text: str, spans: list[Span]) -> str:
    """Delete the spans and tidy the whitespace left behind."""
    kept, pos = [], 0
    for s, e in merge_spans(spans):
        kept.append(text[pos:s])
        pos = e
    kept.append(text[pos:])
    out = "".join(kept)
    out = re.sub(r"[ \t]{2,}", " ", out)
    out = re.sub(r"\n- (?=\n|$)", "", out)  # empty bullet left in key-value output
    return out.strip()


def char_mask(spans: list[Span], n: int) -> list[bool]:
    mask = [False] * n
    for s, e in spans:
        for i in range(max(0, s), min(n, e)):
            mask[i] = True
    return mask


def span_counts(pred: list[Span], gold: list[Span], n: int) -> tuple[int, int, int]:
    """Character-level (true positives, false positives, false negatives)."""
    p, g = char_mask(pred, n), char_mask(gold, n)
    tp = sum(a and b for a, b in zip(p, g))
    return tp, sum(p) - tp, sum(g) - tp


def prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def span_report(preds: list[list[Span]], golds: list[list[Span]], texts: list[str]) -> dict[str, float]:
    """Micro-averaged character span P/R/F1 plus example-level rates.

    removed_fraction: mean share of injected characters removed (injected examples)
    benign_kept:      mean share of benign characters kept (all examples)
    false_alarm_rate: share of clean examples where anything was removed
    """
    tp = fp = fn = 0
    removed, kept, alarms, n_inj, n_clean = [], [], 0, 0, 0
    for pred, gold, text in zip(preds, golds, texts):
        a, b, c = span_counts(pred, gold, len(text))
        tp, fp, fn = tp + a, fp + b, fn + c
        gold_chars = a + c
        benign_chars = len(text) - gold_chars
        if benign_chars:
            kept.append(1 - b / benign_chars)
        if gold_chars:
            n_inj += 1
            removed.append(a / gold_chars)
        else:
            n_clean += 1
            alarms += bool(pred)
    report = prf(tp, fp, fn)
    report.update(
        removed_fraction=sum(removed) / len(removed) if removed else float("nan"),
        benign_kept=sum(kept) / len(kept) if kept else float("nan"),
        false_alarm_rate=alarms / n_clean if n_clean else float("nan"),
        n_injected=n_inj,
        n_clean=n_clean,
    )
    return report
