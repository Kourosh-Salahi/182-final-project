"""Paraphrase the seed attack templates with an open LLM.

Output is a JSON file {family: [template, ...]} where each template keeps a
{goal} slot. `build-dataset --extra-templates` mixes them into the training
pool of their family; eval templates are never paraphrased, so the unseen-
wording test split stays unseen.
"""

from __future__ import annotations

import json
from pathlib import Path

from .attacks import FAMILIES, TEMPLATES
from .backends import Backend

PROMPT = (
    "You are helping build a dataset for training a prompt-injection detector for a robot assistant. "
    "Below is one example of injected text of the type '{family}'. The token {{goal}} stands for the "
    "action the attacker wants the robot to perform.\n\n"
    "Example:\n{template}\n\n"
    "Write {k} varied rewrites that keep the same intent but change wording, tone, length and formatting. "
    "Each rewrite must contain {{goal}} exactly once. Return only a JSON list of strings."
)


def _valid(template: str) -> bool:
    if template.count("{goal}") + template.count("{Goal}") != 1:
        return False
    try:
        template.format(goal="x", Goal="X")
    except (KeyError, IndexError, ValueError):
        return False
    return True


def parse_rewrites(reply: str) -> list[str]:
    start, end = reply.find("["), reply.rfind("]")
    if start < 0 or end <= start:
        return []
    try:
        items = json.loads(reply[start : end + 1])
    except json.JSONDecodeError:
        return []
    out = []
    for item in items:
        if not isinstance(item, str):
            continue
        # Literal braces (e.g. fake JSON tool calls) must be doubled to survive str.format.
        t = item.replace("{", "{{").replace("}", "}}").replace("{{goal}}", "{goal}").replace("{{Goal}}", "{Goal}")
        if _valid(t):
            out.append(t)
    return out


def augment(backend: Backend, k: int = 5, families: tuple[str, ...] = FAMILIES, out_path: str | Path | None = None, log=print):
    extra: dict[str, list[str]] = {}
    for family in families:
        seen = set(TEMPLATES[family]["train"]) | set(TEMPLATES[family]["eval"])
        for template in TEMPLATES[family]["train"]:
            shown = template.replace("{{", "{").replace("}}", "}")
            prompt = PROMPT.format(family=family.replace("_", " "), template=shown, k=k)
            reply = backend.complete([{"role": "user", "content": prompt}], [])
            new = [t for t in parse_rewrites(reply.get("content", "")) if t not in seen]
            seen.update(new)
            extra.setdefault(family, []).extend(new)
        log(f"{family}: {len(extra.get(family, []))} paraphrases")
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(extra, indent=2))
    return extra
