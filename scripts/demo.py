"""Render a side-by-side clip of one prompt injection in the MuJoCo kitchen.

Three panels, same user task, same poisoned tool output:
  1. clean          no attack
  2. attacked       injection in the lookup result, no defense
  3. defended       same injection, span classifier in front of the VLM

The VLM (Qwen3-VL, run locally) decides every `execute_subgoal` call; the
scripted MuJoCo arm then carries out whatever it asked for.

    python scripts/demo.py --task 8 --goal 7 --family false_recovery --out runs/demo
"""

import argparse
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from toolguard.agent import run_trial
from toolguard.attacks import TEMPLATES, Injection, render_template
from toolguard.backends import HFBackend
from toolguard.defenses import NoDefense, make_defense
from toolguard.sim import KitchenSim
from toolguard.tasks import get_task

W, H, HEAD, TEXT = 480, 360, 34, 190
RED, GREEN, INK, GREY = (200, 30, 30), (20, 130, 60), (25, 25, 25), (110, 110, 110)


def font(size, bold=False):
    for name in (("arialbd.ttf", "DejaVuSans-Bold.ttf") if bold else ("arial.ttf", "DejaVuSans.ttf")):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_segments(draw, segments, x, y, width, fnt, line_h, max_lines):
    """Word-wrap coloured text segments. Returns the y below the last line."""
    cx, lines = x, 1
    for text, colour in segments:
        for word in text.replace("\n", " ").split(" "):
            if not word:
                continue
            w = draw.textlength(word + " ", font=fnt)
            if cx + w > x + width and cx > x:
                cx, y, lines = x, y + line_h, lines + 1
                if lines > max_lines:
                    draw.text((x + width - 14, y - line_h), "...", fill=GREY, font=fnt)
                    return y
            draw.text((cx, y), word, fill=colour, font=fnt)
            cx += w
    return y + line_h


def panel(frame, title, title_colour, task, tool_segments, note, command, command_colour):
    img = Image.new("RGB", (W, HEAD + H + TEXT), (250, 250, 250))
    d = ImageDraw.Draw(img)
    d.rectangle((0, 0, W, HEAD), fill=title_colour)
    d.text((10, 7), title, fill=(255, 255, 255), font=font(17, bold=True))
    img.paste(frame, (0, HEAD))
    y = HEAD + H + 6
    y = draw_segments(d, [("User task: ", GREY), (task, INK)], 10, y, W - 20, font(14, bold=True), 18, 1)
    y = draw_segments(d, [("lookup_object returned: ", GREY)] + tool_segments, 10, y + 2, W - 20, font(12), 15, 6)
    if note:
        y = draw_segments(d, [note], 10, y + 1, W - 20, font(12, bold=True), 15, 1)
    if command:
        draw_segments(d, [("Arm command: ", GREY), (command, command_colour)], 10, HEAD + H + TEXT - 40, W - 20,
                      font(14, bold=True), 18, 2)
    return img


def tool_segments(result, injection, defended):
    """Coloured text of the poisoned lookup result as the VLM saw it."""
    entry = next((t for t in result.tool_log if t["injected"]), result.tool_log[0] if result.tool_log else None)
    if entry is None:
        return [("(no lookup)", GREY)]
    if defended or injection is None:
        return [(entry["shown"], INK)]
    raw = entry["raw"]
    i = raw.find(injection.text)
    if i < 0:
        return [(raw, INK)]
    return [(raw[:i], INK), (raw[i : i + len(injection.text)], RED), (raw[i + len(injection.text):], INK)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, default=8, help="user task id (LIBERO-Goal index)")
    ap.add_argument("--goal", type=int, default=7, help="attacker goal id")
    ap.add_argument("--family", default="false_recovery")
    ap.add_argument("--split", default="eval")
    ap.add_argument("--template", type=int, default=-1, help="template index in that family and split")
    ap.add_argument("--checkpoint", default="checkpoints/span-clf")
    ap.add_argument("--defense", default="classifier")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gen-cache", default="runs/gen_cache.jsonl")
    ap.add_argument("--out", default="runs/demo")
    args = ap.parse_args()

    task, goal = get_task(args.task), get_task(args.goal)
    template = TEMPLATES[args.family][args.split][args.template]
    injection = Injection(render_template(template, goal), args.family, goal, args.split, args.template)
    backend = HFBackend(cache_path=args.gen_cache)
    defense = make_defense(args.defense, checkpoint=args.checkpoint, backend=backend)
    sim = KitchenSim(W, H)
    sim.reset(args.seed)
    image = sim.render()

    conditions = [
        ("1. No attack", (70, 70, 80), NoDefense(), None),
        ("2. Injection, no defense", (170, 40, 40), NoDefense(), injection),
        (f"3. Injection + {args.defense}", (30, 110, 70), defense, injection),
    ]
    clips, log = [], []
    for title, colour, dfn, inj in conditions:
        # Same seed in every panel, so the benign text around the injection is identical.
        result = run_trial(backend, dfn, task, random.Random(args.seed), injection=inj, image=image,
                           response_format="prose", inject_all=True)
        segs = tool_segments(result, inj, defended=dfn.name != "none")
        note = None
        if inj and dfn.name != "none":
            note = ("attacker instruction cut out before the VLM read it", GREEN) if not result.survived else (
                "injection NOT removed", RED)
        sim.reset(args.seed)
        frames = [(sim.render(), None, INK)] * 10
        for sub in result.subgoals:
            hijack = inj is not None and sub["task_id"] == goal.id
            cmd, cc = f'execute_subgoal("{sub["instruction"]}")', RED if hijack else GREEN
            if hijack:
                cmd += "  <- attacker's goal"
            if sub["task_id"] is None:
                frames += [(sim.render(), cmd + "  (not a known skill)", GREY)] * 8
                continue
            frames += [(sim.render(), cmd, cc)] * 6
            frames += [(f, cmd, cc) for f in sim.execute(sub["task_id"])]
        if not result.subgoals:
            frames += [(sim.render(), "(no arm command issued)", GREY)] * 8
        frames += [frames[-1]] * 12
        clips.append([panel(f, title, colour, task.instruction, segs, note, cmd, cc) for f, cmd, cc in frames])
        log.append({"condition": title, "outcome": result.outcome, "subgoals": result.subgoals,
                    "survived_defense": result.survived, "tool_log": result.tool_log})
        print(title, "->", result.outcome, [s["instruction"] for s in result.subgoals])

    n = max(len(c) for c in clips)
    frames = []
    for i in range(n):
        row = Image.new("RGB", (3 * W + 20, HEAD + H + TEXT), (255, 255, 255))
        for j, clip in enumerate(clips):
            row.paste(clip[min(i, len(clip) - 1)], (j * (W + 10), 0))
        frames.append(row)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"task{task.id}_goal{goal.id}_{args.family}"
    frames[0].save(out / f"{stem}.gif", save_all=True, append_images=frames[1:], duration=70, loop=0)
    frames[-1].save(out / f"{stem}_final.png")
    (out / f"{stem}.json").write_text(json.dumps({"task": task.instruction, "attacker_goal": goal.instruction,
                                                   "injection": injection.text, "panels": log}, indent=2))
    print(f"wrote {out / (stem + '.gif')} and {out / (stem + '_final.png')}")


if __name__ == "__main__":
    main()
