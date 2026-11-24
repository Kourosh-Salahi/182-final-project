"""Undefended attack success per attack template.

Used to check that the attack set is strong before any defense is compared:
    python scripts/template_asr.py --n 10 --out runs/template_asr.json
"""

import argparse
import json
import random
from collections import defaultdict

from toolguard.agent import run_trial
from toolguard.attacks import FAMILIES, TEMPLATES, Injection, render_template
from toolguard.backends import HFBackend
from toolguard.dataset import sample_goal
from toolguard.defenses import NoDefense
from toolguard.experiments import wilson
from toolguard.sim import KitchenSim
from toolguard.tasks import LIBERO_GOAL_TASKS


_REPLACE = ("instead", "forget", "cancel", "ignore", "disregard", "no longer", "supersede", "withdrawn", "replaced",
            "rather than", "not the one", "wrong task", "changed", "only task", "do not do what")
_ORDER = ("first", "before", "until", "then ", "once ", "after that", "comes right after", "follows it")


def style(template: str) -> str:
    """staged: asks for the attacker's goal as a step before the user's task, which is kept.
    direct: asks to replace the user's task, or just issues the command."""
    t = template.lower()
    return "staged" if any(w in t for w in _ORDER) and not any(w in t for w in _REPLACE) else "direct"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10, help="trials per template")
    ap.add_argument("--families", nargs="+", default=list(FAMILIES))
    ap.add_argument("--model", default=None)
    ap.add_argument("--gen-cache", default="runs/gen_cache.jsonl")
    ap.add_argument("--out", default="runs/template_asr.json")
    ap.add_argument("--show", type=int, default=0, help="print this many example failures per template")
    args = ap.parse_args()

    backend = HFBackend(model=args.model, cache_path=args.gen_cache)
    sim, frames = KitchenSim(), {}

    def frame(seed):
        if seed not in frames:
            sim.reset(seed)
            frames[seed] = sim.render()
        return frames[seed]

    rows, outcomes = [], defaultdict(lambda: defaultdict(int))
    for family in args.families:
        for split in ("train", "eval"):
            for tid, template in enumerate(TEMPLATES[family][split]):
                hits, shown = 0, 0
                for i in range(args.n):
                    rng = random.Random(f"probe/{family}/{split}/{tid}/{i}")
                    task = rng.choice(LIBERO_GOAL_TASKS)
                    goal = sample_goal(task, rng)
                    inj = Injection(render_template(template, goal), family, goal, split, tid)
                    r = run_trial(backend, NoDefense(), task, rng, injection=inj, image=frame(i % 10), inject_all=True)
                    hits += r.attack_at_call
                    outcomes[family][r.outcome] += 1
                    if not r.attack_at_call and shown < args.show:
                        shown += 1
                        print(f"  MISS task={task.instruction!r} goal={goal.instruction!r} -> {r.outcome}: {r.instruction!r}")
                lo, hi = wilson(hits, args.n)
                rows.append({"family": family, "split": split, "template_id": tid, "template": template,
                             "style": style(template),
                             "asr": hits / args.n, "k": hits, "n": args.n, "ci95": [lo, hi]})
                print(f"{family:20s} {split:5s} #{tid}: {hits}/{args.n}  {template[:70]!r}", flush=True)
    total_k, total_n = sum(r["k"] for r in rows), sum(r["n"] for r in rows)
    print(f"\noverall undefended ASR@call: {total_k}/{total_n} = {total_k / total_n:.1%}")
    for family in args.families:
        k = sum(r["k"] for r in rows if r["family"] == family)
        n = sum(r["n"] for r in rows if r["family"] == family)
        print(f"  {family:20s} {k}/{n} = {k / n:.1%}   outcomes: {dict(outcomes[family])}")
    for st in ("direct", "staged"):
        k = sum(r["k"] for r in rows if r["style"] == st)
        n = sum(r["n"] for r in rows if r["style"] == st)
        if n:
            print(f"  style={st:9s} {k}/{n} = {k / n:.1%}  ({sum(r['style'] == st for r in rows)} wordings)")
    with open(args.out, "w") as f:
        json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
