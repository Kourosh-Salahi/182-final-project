"""Out-of-distribution test splits built from the generated pools.

    ood_benign  new harmless writing + the original `eval` attack templates
    ood_attack  the original harmless pool + new attack shapes
    ood_both    new harmless writing + new attack shapes

Even rows are harmless, odd rows attacked. Replies use the same layout as the original
data (`layout_response` / `build_response`), so only the wording changes.
"""

from __future__ import annotations

import random

from ..attacks import FAMILIES, make_injection, render_template
from ..dataset import Example, sample_goal, sample_tool_call
from ..tasks import LIBERO_GOAL_TASKS, SCENE_OBJECTS, resolve_object
from ..tools import ToolResponse, build_response, layout_response

OOD_SPLITS = ("ood_benign", "ood_attack", "ood_both")
FORMATS = ("prose", "kv", "json")
LOCATIONS = tuple(SCENE_OBJECTS) + ("workspace",)


def _benign_by_location(pool: list[dict]) -> dict[str, list[str]]:
    by_loc: dict[str, list[str]] = {loc: [] for loc in LOCATIONS}
    for rec in pool:
        by_loc.setdefault(rec["object"], []).append(rec["text"])
    for loc in LOCATIONS:
        if len(by_loc[loc]) < 2:
            raise ValueError(f"benign pool has {len(by_loc[loc])} sentences for '{loc}', need at least 2")
    return by_loc


def _attacks_by_family(pool: list[dict]) -> dict[str, list[str]]:
    by_fam: dict[str, list[str]] = {f: [] for f in FAMILIES}
    for rec in pool:
        by_fam[rec["family"]].append(rec["template"])
    empty = [f for f, ts in by_fam.items() if not ts]
    if empty:
        raise ValueError(f"attack pool has no wordings for families {empty}")
    return by_fam


def _location(tool: str, args: dict) -> str:
    if tool == "get_scene_notes":
        return "workspace"
    obj = resolve_object(str(args.get("name", "")))
    if obj is None:
        raise ValueError(f"unknown object in tool call: {args}")
    return obj.name


def build_ood_split(
    name: str, benign_pool: list[dict], attack_pool: list[dict], n: int = 600, seed: int = 0
) -> list[Example]:
    if name not in OOD_SPLITS:
        raise ValueError(f"unknown OOD split '{name}', choose from {OOD_SPLITS}")
    new_benign = name != "ood_attack"
    new_attacks = name != "ood_benign"
    sents_by_loc = _benign_by_location(benign_pool) if new_benign else {}
    wordings = _attacks_by_family(attack_pool) if new_attacks else {}
    rng = random.Random(f"{seed}/{name}")

    rows = []
    for i in range(n):
        task = rng.choice(LIBERO_GOAL_TASKS)
        tool, args = sample_tool_call(task, rng)
        injection, family, goal, template_split = None, None, None, None
        if i % 2 == 1:
            family = FAMILIES[(i // 2) % len(FAMILIES)]
            goal = sample_goal(task, rng)
            if new_attacks:
                injection, template_split = render_template(rng.choice(wordings[family]), goal), "ood"
            else:
                injection, template_split = make_injection(family, goal, rng, "eval").text, "eval"

        if new_benign:
            loc = _location(tool, args)
            pool = sents_by_loc[loc]
            sents = rng.sample(pool, k=rng.randint(2, min(5, len(pool))))
            resp: ToolResponse = layout_response(loc, sents, rng, injection, rng.choice(FORMATS))
        else:
            resp = build_response(tool, args, task, rng, injection=injection)

        rows.append(Example(
            task=task.instruction, task_id=task.id, tool=tool, args=args, text=resp.text,
            spans=[list(s) for s in resp.spans], family=family,
            goal=goal.instruction if goal else None, goal_id=goal.id if goal else None,
            template_split=template_split,
        ))
    return rows
