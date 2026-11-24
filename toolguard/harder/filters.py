"""Decontamination and rule filters for generated pool records.

Generated text must not copy or reword what the classifier was trained on. Each record is
checked against the existing data (every harmless sentence the current generator can emit,
and every attack template), by word 8-gram overlap, edit ratio and, when an embedder is
given, embedding cosine. Rule checks catch broken slots, the old "first ... then" attack
shape, and harmless lines that name a robot task.
"""

from __future__ import annotations

import difflib
import math
import re
from typing import Callable

from ..attacks import TEMPLATES
from ..tasks import LIBERO_GOAL_TASKS, SCENE_OBJECTS, normalize
from ..tools import _CARE_NOTES, _HISTORY_NOTES, _SCENE_NOTES, _TASK_HINTS

NGRAM = 8
EDIT_RATIO = 0.8
COSINE = 0.85
WITHIN_POOL_RATIO = 0.9
# Accepted items at or above these scores go on the hand-audit sheet (review only, never auto-rejected).
REVIEW_EDIT = 0.6
REVIEW_COS = 0.7
SEQUENCING = re.compile(r"\b(first|then|before|after that|continue|proceed)\b", re.IGNORECASE)

_SLOTS = {"benign": (), "attack": ("{goal}", "{Goal}"), "contrast": ("{task}", "{Task}")}
_ANY_SLOT = re.compile(r"\{(goal|Goal|task|Task)\}")
_AFTER_SLOT_AND = re.compile(r"\{[Tt]ask\}[^.?!]*\band\b")
# Things that are not in the robot's scene (one table, seven objects, an electric stove).
OFF_SCENE = re.compile(
    r"\b(fridge|refrigerat\w*|cellar|basement|hallway|bedroom|garage|pantry|dining|mugs?|basil|carafe|"
    r"bananas?|apples?|placemats?|trivet|cutting board|knife|wine glass(es)?|butane|canister|gas|"
    r"melamine|bamboo|cats?|dogs?|sink|counter(top)?|cupboard|shelf|shelves|magnums?)\b",
    re.IGNORECASE,
)
_EXTRA_ASK = re.compile(r"\b(let me know|come back|report back|tell me)\b", re.IGNORECASE)

Embedder = Callable[[list[str]], list[list[float]]]


def existing_benign_sentences() -> list[str]:
    """Every harmless sentence the current generator (`toolguard.tools`) can emit."""
    out: list[str] = []
    for obj in SCENE_OBJECTS.values():
        out += list(obj.facts)
        out.append(f"It is located {obj.location}.")
        out += [h.format(obj=obj.name, task=t.instruction) for h in _TASK_HINTS for t in LIBERO_GOAL_TASKS]
    out += list(_CARE_NOTES) + list(_SCENE_NOTES)
    out += [h.format(other=t.instruction) for h in _HISTORY_NOTES for t in LIBERO_GOAL_TASKS]
    return out


def existing_templates() -> list[str]:
    return [t for fam in TEMPLATES.values() for split in fam.values() for t in split]


def slotless(text: str) -> str:
    return normalize(_ANY_SLOT.sub("X", text))


def _text(rec: dict) -> str:
    return rec["text"] if "text" in rec else rec["template"]


def _ngrams(words: list[str], n: int = NGRAM) -> set[tuple[str, ...]]:
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def max_edit_ratio(text: str, reference: list[str]) -> float:
    a = slotless(text)
    best = 0.0
    for ref in reference:
        m = difflib.SequenceMatcher(None, a, slotless(ref))
        if m.real_quick_ratio() > best and m.quick_ratio() > best:
            best = max(best, m.ratio())
    return best


def lexical_reasons(text: str, reference: list[str]) -> list[str]:
    reasons = []
    grams = _ngrams(slotless(text).split())
    if grams and any(grams & _ngrams(slotless(ref).split()) for ref in reference):
        reasons.append("ngram8")
    if max_edit_ratio(text, reference) >= EDIT_RATIO:
        reasons.append("edit")
    return reasons


def rule_reasons(rec: dict, kind: str) -> list[str]:
    text = _text(rec)
    reasons = []
    leftover = text
    for slot in _SLOTS[kind]:
        leftover = leftover.replace(slot, "")
    if "{" in leftover or "}" in leftover:
        reasons.append("stray_brace")
    n_slots = sum(text.count(s) for s in _SLOTS[kind])
    if kind == "attack":
        if n_slots == 0:
            reasons.append("no_slot")
        if SEQUENCING.search(text):
            reasons.append("sequencing")
    elif kind == "contrast":
        if n_slots != 1:
            reasons.append("no_slot")
        # A wording must only restate the task: no second instruction, no sequencing.
        if SEQUENCING.search(text) or _AFTER_SLOT_AND.search(text) or _EXTRA_ASK.search(text):
            reasons.append("extra_instruction")
    else:
        if any(normalize(t.instruction) in normalize(text) for t in LIBERO_GOAL_TASKS):
            reasons.append("names_task")
        if OFF_SCENE.search(text):
            reasons.append("off_scene")
    return reasons


def _unit(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def _max_cos(vec: list[float], refs: list[list[float]]) -> float:
    return max((sum(a * b for a, b in zip(vec, r)) for r in refs), default=0.0)


def filter_pool(
    records: list[dict], kind: str, embed: Embedder | None = None
) -> tuple[list[dict], list[dict], list[dict]]:
    """(accepted, rejected with reasons, accepted records similar enough to review by hand, with scores)."""
    reference = existing_benign_sentences() if kind == "benign" else existing_templates()
    ref_vecs = [_unit(v) for v in embed([slotless(r) for r in reference])] if embed else []
    rec_vecs = [_unit(v) for v in embed([slotless(_text(r)) for r in records])] if embed and records else []

    accepted: list[dict] = []
    rejected: list[dict] = []
    near: list[dict] = []
    for i, rec in enumerate(records):
        text = _text(rec)
        reasons = rule_reasons(rec, kind) + lexical_reasons(text, reference)
        edit = max_edit_ratio(text, reference)
        cos = _max_cos(rec_vecs[i], ref_vecs) if embed else 0.0
        if embed and cos >= COSINE:
            reasons.append("semantic")
        if not reasons and any(
            difflib.SequenceMatcher(None, slotless(text), slotless(_text(a))).ratio() >= WITHIN_POOL_RATIO
            for a in accepted
        ):
            reasons.append("duplicate")
        if reasons:
            rejected.append({**rec, "reasons": reasons})
            continue
        accepted.append(rec)
        if edit >= REVIEW_EDIT or cos >= REVIEW_COS:
            near.append({**rec, "max_edit": round(edit, 3), "max_cos": round(cos, 3)})
    return accepted, rejected, near
