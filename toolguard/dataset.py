"""Span-labelled dataset of benign and injected tool responses.

Splits:
  train / val   seen families, train templates
  test_seen     seen families, eval templates (unseen wording)
  test_heldout  held-out families (all their templates)
Each split is about half benign, half injected.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from .attacks import FAMILIES, HELD_OUT_FAMILIES, TEMPLATES, Injection, make_injection
from .tasks import LIBERO_GOAL_TASKS, SCENE_OBJECTS, Task
from .tools import ToolResponse, build_response


@dataclass
class Example:
    task: str  # trusted user instruction the classifier conditions on
    task_id: int
    tool: str
    args: dict
    text: str  # tool output as the orchestrator would see it undefended
    spans: list[list[int]]  # injected character spans [start, end)
    family: str | None = None
    goal: str | None = None
    goal_id: int | None = None
    template_split: str | None = None


def sample_tool_call(task: Task, rng: random.Random) -> tuple[str, dict]:
    if rng.random() < 0.2:
        return "get_scene_notes", {}
    # Mostly objects the task involves, sometimes another one.
    name = rng.choice(task.objects) if rng.random() < 0.8 else rng.choice(list(SCENE_OBJECTS))
    return "lookup_object", {"name": name}


def sample_goal(task: Task, rng: random.Random) -> Task:
    return rng.choice([t for t in LIBERO_GOAL_TASKS if t.id != task.id])


def make_example(
    rng: random.Random,
    families: tuple[str, ...] | None,
    template_split: str = "train",
    extra_templates: dict[str, list[str]] | None = None,
) -> Example:
    """One example; benign when `families` is None."""
    task = rng.choice(LIBERO_GOAL_TASKS)
    tool, args = sample_tool_call(task, rng)
    inj: Injection | None = None
    if families:
        inj = make_injection(rng.choice(families), sample_goal(task, rng), rng, template_split, extra_templates)
    resp: ToolResponse = build_response(tool, args, task, rng, injection=inj.text if inj else None)
    return Example(
        task=task.instruction,
        task_id=task.id,
        tool=tool,
        args=args,
        text=resp.text,
        spans=[list(s) for s in resp.spans],
        family=inj.family if inj else None,
        goal=inj.goal.instruction if inj else None,
        goal_id=inj.goal.id if inj else None,
        template_split=inj.template_split if inj else None,
    )


def build_dataset(
    n_train: int = 3000,
    n_val: int = 400,
    n_test: int = 600,
    held_out: tuple[str, ...] = HELD_OUT_FAMILIES,
    p_inject: float = 0.5,
    seed: int = 0,
    extra_templates: dict[str, list[str]] | None = None,
) -> dict[str, list[Example]]:
    unknown = set(held_out) - set(FAMILIES)
    if unknown:
        raise ValueError(f"unknown families: {sorted(unknown)}")
    seen = tuple(f for f in FAMILIES if f not in held_out)
    rng = random.Random(seed)

    def split(n: int, families: tuple[str, ...], template_split: str) -> list[Example]:
        out = []
        for _ in range(n):
            fams = families if rng.random() < p_inject else None
            ts = template_split
            if fams and template_split == "any":
                # Weight by pool size so every held-out template is equally likely.
                ts = rng.choices(("train", "eval"), weights=[
                    sum(len(TEMPLATES[f]["train"]) for f in fams),
                    sum(len(TEMPLATES[f]["eval"]) for f in fams),
                ])[0]
            out.append(make_example(rng, fams, ts, extra_templates if ts == "train" else None))
        return out

    data = {
        "train": split(n_train, seen, "train"),
        "val": split(n_val, seen, "train"),
        "test_seen": split(n_test, seen, "eval"),
    }
    if held_out:
        data["test_heldout"] = split(n_test, tuple(held_out), "any")
    return data


def save_dataset(data: dict[str, list[Example]], out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, examples in data.items():
        with open(out / f"{name}.jsonl", "w") as f:
            for ex in examples:
                f.write(json.dumps(asdict(ex)) + "\n")


def load_split(path: str | Path) -> list[Example]:
    with open(path) as f:
        return [Example(**json.loads(line)) for line in f if line.strip()]
