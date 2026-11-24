"""New attack wordings as a robot replay file (`toolguard run --replay`).

Rows follow `toolguard.adaptive.load_replay`: task_id, goal_id, family, text, evaded. All
rows are marked evaded=True so the replay uses every one of them.
"""

from __future__ import annotations

import random

from ..attacks import FAMILIES, render_template
from ..dataset import sample_goal
from ..tasks import LIBERO_GOAL_TASKS


def build_replay(attack_pool: list[dict], per_family: int = 30, seed: int = 0) -> list[dict]:
    rng = random.Random(f"{seed}/replay")
    rows = []
    for family in FAMILIES:
        wordings = [r["template"] for r in attack_pool if r["family"] == family]
        if not wordings:
            raise ValueError(f"attack pool has no wordings for family '{family}'")
        order = rng.sample(wordings, k=len(wordings))
        for i in range(per_family):
            template = order[i % len(order)]
            task = rng.choice(LIBERO_GOAL_TASKS)
            goal = sample_goal(task, rng)
            rows.append({"task_id": task.id, "goal_id": goal.id, "family": family,
                         "text": render_template(template, goal), "template": template, "evaded": True})
    return rows
