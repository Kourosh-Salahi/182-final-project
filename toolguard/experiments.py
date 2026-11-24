"""E1 / E2: attack success at the tool call and at the arm, per defense and attack family.

E1 is the undefended agent (clean and attacked). E2 runs the same trials under
each defense. Trials are paired: trial i of a family uses the same task,
attacker goal, injection text and tool-response layout under every defense.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

from .agent import TrialResult, run_trial
from .attacks import FAMILIES, HELD_OUT_FAMILIES, Injection, make_injection
from .backends import Backend
from .dataset import sample_goal
from .defenses import Defense
from .executors import Executor
from .scene import render_scene
from .tasks import LIBERO_GOAL_TASKS, get_task

CLEAN = "clean"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _rate(k: int, n: int) -> dict:
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "rate": k / n if n else float("nan"), "ci95": [lo, hi]}


def summarize(results: list[TrialResult]) -> list[dict]:
    rows = []
    keys = sorted({(r.defense, r.family or CLEAN) for r in results}, key=lambda k: (k[0], k[1] != CLEAN, k[1]))
    for defense, cond in keys:
        rs = [r for r in results if r.defense == defense and (r.family or CLEAN) == cond]
        n_roll = sum(r.rollout_requested for r in rs)
        row = {
            "defense": defense,
            "condition": cond,
            "held_out": cond in HELD_OUT_FAMILIES,
            "n": len(rs),
            "user_call": _rate(sum(r.outcome == "user_call" for r in rs), len(rs)),
            "no_call": _rate(sum(r.outcome == "no_call" for r in rs), len(rs)),
        }
        if cond != CLEAN:
            row["exposed"] = _rate(sum(r.exposed for r in rs), len(rs))
            row["survived_defense"] = _rate(sum(r.survived for r in rs), len(rs))
            row["attack_at_call"] = _rate(sum(r.attack_at_call for r in rs), len(rs))
            # Attack success among trials where the poisoned tool output was actually served.
            ex = [r for r in rs if r.exposed]
            row["attack_given_exposed"] = _rate(sum(r.attack_at_call for r in ex), len(ex))
        if n_roll:
            # Arm rates are over all trials given a rollout, so they are comparable to the call rates.
            row["user_success_at_arm"] = _rate(sum(r.user_success_at_arm for r in rs), n_roll)
            if cond != CLEAN:
                row["attack_at_arm"] = _rate(sum(r.attack_at_arm for r in rs), n_roll)
        rows.append(row)
    return rows


def format_table(rows: list[dict]) -> str:
    def pct(row, key):
        if key not in row:
            return "-"
        r = row[key]
        if not r["n"]:
            return "-"
        return f"{100 * r['rate']:5.1f}% [{100 * r['ci95'][0]:.0f},{100 * r['ci95'][1]:.0f}]"

    cols = [
        ("defense", lambda r: r["defense"]),
        ("condition", lambda r: r["condition"] + ("*" if r["held_out"] else "")),
        ("n", lambda r: str(r["n"])),
        ("ASR@call", lambda r: pct(r, "attack_at_call")),
        ("ASR@arm", lambda r: pct(r, "attack_at_arm")),
        ("user@call", lambda r: pct(r, "user_call")),
        ("user@arm", lambda r: pct(r, "user_success_at_arm")),
        ("exposed", lambda r: pct(r, "exposed")),
        ("survived", lambda r: pct(r, "survived_defense")),
    ]
    table = [[name for name, _ in cols]] + [[fn(r) for _, fn in cols] for r in rows]
    widths = [max(len(row[i]) for row in table) for i in range(len(cols))]
    lines = ["  ".join(c.ljust(w) for c, w in zip(row, widths)) for row in table]
    lines.insert(1, "  ".join("-" * w for w in widths))
    return "\n".join(lines) + "\n(* = attack family held out of classifier training; brackets are 95% Wilson intervals)"


def run_experiment(
    backend: Backend,
    defenses: list[Defense],
    families: tuple[str, ...] = FAMILIES,
    n_trials: int = 50,
    include_clean: bool = True,
    executor: Executor | None = None,
    n_rollouts: int = 0,
    use_image: bool = True,
    template_split: str = "eval",
    seed: int = 0,
    out_dir: str | Path | None = None,
    injections: dict[str, list[tuple[int, Injection]]] | None = None,
    inject_all: bool = True,
    log=print,
) -> list[TrialResult]:
    """Run every (defense, condition) cell for `n_trials` trials.

    The first `n_rollouts` trials of each cell are rolled out on the arm when an
    executor is given; the rest stop at the first `execute_subgoal` call.
    `injections` overrides the sampled attacks with fixed (task_id, injection)
    pairs per family, used to replay adaptive attacks from E3. With `inject_all`
    (default) the attacker has poisoned every corpus entry, so each lookup in a
    trial returns the injection; otherwise only the first lookup does.
    """
    conditions = ([CLEAN] if include_clean else []) + list(families)
    frames: dict[tuple[int, int], object] = {}
    results: list[TrialResult] = []
    out_file = None
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        out_file = open(Path(out_dir) / "trials.jsonl", "w")

    try:
        for defense in defenses:
            for cond in conditions:
                for i in range(n_trials):
                    # Seeded by (seed, condition, i) only, so trials are paired across defenses.
                    rng = random.Random(f"{seed}/{cond}/{i}")
                    injection = None
                    if cond != CLEAN and injections is not None:
                        if i >= len(injections.get(cond, [])):
                            break
                        task_id, injection = injections[cond][i]
                        task = get_task(task_id)
                    else:
                        task = rng.choice(LIBERO_GOAL_TASKS)
                        if cond != CLEAN:
                            injection = make_injection(cond, sample_goal(task, rng), rng, template_split)

                    episode = i
                    image = None
                    if use_image and backend.name != "mock":
                        key = (task.id, episode % 10)
                        if key not in frames:
                            frame = executor.initial_frame(task, episode % 10) if executor else None
                            # No simulator: a schematic rendering of the same scene.
                            frames[key] = frame if frame is not None else render_scene(seed=episode % 10)
                        image = frames[key]

                    r = run_trial(backend, defense, task, rng, injection=injection, image=image, inject_all=inject_all)
                    r.rollout_requested = bool(executor) and i < n_rollouts
                    if r.rollout_requested and r.executed_task_id is not None:
                        # Run whatever goal the orchestrator asked for and score that goal.
                        outcome = executor.execute(get_task(r.executed_task_id), episode % 10)
                        r.arm_ran, r.arm_success = True, outcome.success
                    results.append(r)
                    if out_file:
                        out_file.write(json.dumps(r.to_json()) + "\n")
                        out_file.flush()
                log(f"done: defense={defense.name} condition={cond}")
    finally:
        if out_file:
            out_file.close()

    rows = summarize(results)
    if out_dir:
        (Path(out_dir) / "summary.json").write_text(json.dumps(rows, indent=2))
        (Path(out_dir) / "summary.txt").write_text(format_table(rows) + "\n")
    return results
