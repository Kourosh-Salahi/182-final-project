"""Rebuild every figure and table of the report from the files under runs/.

    python scripts/make_figures.py            # writes report/figures/*.pdf and report/generated/*.tex

Inputs (each produced by the command in the README):
    runs/e1/trials.jsonl             undefended baseline
    runs/e2/trials.jsonl             all defenses, paired trials
    runs/e3/adaptive.jsonl           adaptive search against the classifier
    runs/e3_replay/trials.jsonl      evading attacks replayed through the VLM
    runs/classifier_study.json       training curves, seeds, task-conditioning ablation
    runs/span_eval.json              span metrics and threshold sweep
    runs/template_asr.json           undefended success of every attack wording
    runs/ood_eval.json               span metrics on the harder (OOD) splits
    runs/contrast_study.json         task-contrast training study
    runs/ood_screen, runs/ood_e2     new attack wordings on the robot
Missing inputs are skipped with a note, so partial results still build.
"""

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

FIG, GEN = Path("report/figures"), Path("report/generated")
FAMILIES = ["goal_substitution", "authority_spoofing", "false_recovery", "tool_call_hijack"]
FAMILY_LABEL = {"goal_substitution": "Goal subst.", "authority_spoofing": "Authority", "false_recovery": "False recovery*",
                "tool_call_hijack": "Tool-call hijack*"}
DEFENSE_LABEL = {
    "none": "No defense", "classifier": "Span classifier (ours)", "protectai": "ProtectAI detector, whole output",
    "protectai_sent": "ProtectAI detector, per sentence", "sandwich": "Sandwich", "spotlight": "Spotlighting (delimiters)",
    "datamark": "Spotlighting (datamarking)", "instruct": "Instructional warning", "promptarmor": "PromptArmor-style LLM filter",
    "heuristic": "Keyword filter",
}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def load_trials(path):
    p = Path(path)
    if not p.exists():
        print("skip (missing):", path)
        return None
    return [json.loads(line) for line in p.open() if line.strip()]


def load_json(path):
    p = Path(path)
    if not p.exists():
        print("skip (missing):", path)
        return None
    return json.loads(p.read_text())


def pct(k, n):
    lo, hi = wilson(k, n)
    return f"{100 * k / n:.0f} \\,{{\\scriptsize[{100 * lo:.0f},{100 * hi:.0f}]}}" if n else "--"


def write(name, text):
    (GEN / name).write_text(text, encoding="utf-8")
    print("wrote", GEN / name)


# ---------------------------------------------------------------- harness diagram
def fig_harness():
    fig, ax = plt.subplots(figsize=(6.6, 3.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    grey, green, red = "#d9d9d9", "#8fd3b6", "#f4a58a"

    def box(x, y, w, h, title, sub, colour):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04", fc=colour, ec="#555", lw=0.8))
        ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center", fontsize=8.5, weight="bold")
        ax.text(x + w / 2, y + h * 0.27, sub, ha="center", va="center", fontsize=7)

    def arrow(a, b, text="", dx=0.0, dy=0.0, ha="center"):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=9, lw=0.9, color="#333"))
        if text:
            ax.text((a[0] + b[0]) / 2 + dx, (a[1] + b[1]) / 2 + dy, text, fontsize=7, ha=ha, va="center")

    # Laid out so that no arrow crosses a box: the lookup call runs straight down to the right of
    # the defense box, and every label sits in empty space.
    box(1.6, 5.0, 2.4, 0.8, "User task $u$", "trusted instruction", grey)
    box(5.4, 5.0, 2.4, 0.8, "Camera frame", "MuJoCo render", grey)
    box(1.6, 3.5, 6.2, 0.9, "Qwen3-VL orchestrator", "reads task, image and tool results; calls tools", grey)
    box(0.1, 1.9, 2.4, 0.9, "Defense $D(u, x)$", "span classifier or baseline", green)
    box(0.1, 0.3, 3.6, 0.9, "lookup_object", "facts from our corpus", grey)
    box(4.6, 0.3, 2.3, 0.9, "Attacker", "plants text in corpus", red)
    box(7.5, 1.9, 2.4, 0.9, "Arm executor", "scripted MuJoCo arm", grey)
    box(7.5, 0.3, 2.4, 0.9, "Outcome", "whose goal ran?", "#ffffff")
    arrow((2.8, 5.0), (2.8, 4.44))
    arrow((6.6, 5.0), (6.6, 4.44))
    arrow((3.3, 3.46), (3.3, 1.24), "lookup_object(name)", dx=0.12, ha="left")
    arrow((1.0, 1.24), (1.0, 1.86), "raw output $x$", dx=0.12, ha="left")
    arrow((1.3, 2.84), (1.9, 3.46), "cleaned output", dx=-0.55, dy=0.08, ha="right")
    arrow((4.56, 0.75), (3.74, 0.75))
    arrow((7.2, 3.46), (8.7, 2.84), "execute_subgoal(text)", dx=-0.1, dy=-0.12, ha="right")
    arrow((8.7, 1.86), (8.7, 1.24))
    fig.savefig(FIG / "harness.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIG / "harness.pdf")


# ---------------------------------------------------------------- E1 / E2
def cell(trials, defense, family):
    rs = [t for t in trials if t["defense"] == defense and (t["family"] or "clean") == family]
    return rs


def table_e1(e1):
    lines = ["\\begin{tabular}{lrrrr}", "\\toprule",
             "Condition & $n$ & Attacker goal sent & User task sent & Neither (\\%) \\\\", "\\midrule"]
    for fam in ["clean"] + FAMILIES:
        rs = cell(e1, "none", fam)
        n = len(rs)
        atk = sum(t["outcome"] == "attack_call" for t in rs)
        usr = sum(t["outcome"] == "user_call" for t in rs)
        label = "No attack" if fam == "clean" else FAMILY_LABEL[fam].replace("*", "")
        lines.append(f"{label} & {n} & {pct(atk, n) if fam != 'clean' else '--'} & {pct(usr, n)} & {100 * (n - atk - usr) / n:.0f} \\\\")
    rs = [t for t in e1 if t["family"]]
    atk = sum(t["outcome"] == "attack_call" for t in rs)
    usr = sum(t["outcome"] == "user_call" for t in rs)
    lines += ["\\midrule", f"All attacks & {len(rs)} & {pct(atk, len(rs))} & {pct(usr, len(rs))} & {100 * (len(rs) - atk - usr) / len(rs):.0f} \\\\",
              "\\bottomrule", "\\end{tabular}"]
    write("tab_e1.tex", "\n".join(lines) + "\n")


def e2_stats(e2):
    defenses = [d for d in DEFENSE_LABEL if any(t["defense"] == d for t in e2)]
    stats = {}
    for d in defenses:
        att = [t for t in e2 if t["defense"] == d and t["family"]]
        clean = [t for t in e2 if t["defense"] == d and not t["family"]]
        stats[d] = {
            "asr": (sum(t["attack_at_call"] for t in att), len(att)),
            "clean": (sum(t["outcome"] == "user_call" for t in clean), len(clean)),
            "user_under_attack": (sum(t["user_task_called"] for t in att), len(att)),
            "survived": (sum(t["survived"] for t in att), len(att)),
            "by_family": {f: (sum(t["attack_at_call"] for t in cell(e2, d, f)), len(cell(e2, d, f))) for f in FAMILIES},
        }
    return stats


def fig_e2(e2):
    stats = e2_stats(e2)
    order = sorted(stats, key=lambda d: (d != "none", -stats[d]["asr"][0] / max(1, stats[d]["asr"][1])))
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.0), sharey=True, gridspec_kw={"width_ratios": [3, 2]})
    ys = range(len(order))
    for ax, key, title, colour in ((axes[0], "asr", "Attack success (%)", "#c0392b"),
                                   (axes[1], "clean", "Clean-task success (%)", "#2e7d5b")):
        for y, d in zip(ys, order):
            k, n = stats[d][key]
            if not n:
                continue
            lo, hi = wilson(k, n)
            p = k / n
            ax.barh(y, 100 * p, color="#2c5f8a" if d == "classifier" else colour, alpha=0.85, height=0.65)
            ax.errorbar(100 * p, y, xerr=[[100 * (p - lo)], [100 * (hi - p)]], color="k", capsize=2, lw=0.8)
            ax.text(min(100 * hi + 2, 101), y, f"{100 * p:.0f}", va="center", fontsize=7)
        ax.set_xlim(0, 112)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xlabel(title)
    axes[0].set_yticks(list(ys))
    axes[0].set_yticklabels([DEFENSE_LABEL[d] for d in order])
    axes[0].invert_yaxis()
    fig.tight_layout()
    fig.savefig(FIG / "defenses.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIG / "defenses.pdf")

    lines = ["\\begin{tabular}{l" + "r" * 7 + "}", "\\toprule",
             "Defense & " + " & ".join(FAMILY_LABEL[f].replace("*", "$^*$") for f in FAMILIES)
             + " & All attacks & Injection readable & Clean \\\\", "\\midrule"]
    for d in order:
        s = stats[d]
        row = [DEFENSE_LABEL[d]] + [pct(*s["by_family"][f]) for f in FAMILIES]
        row += [pct(*s["asr"]), pct(*s["survived"]), pct(*s["clean"])]
        lines.append(" & ".join(row) + " \\\\")
        if d == "none":
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write("tab_e2.tex", "\n".join(lines) + "\n")

    # Paired comparison with the undefended run: trial i of a family is the same episode under every defense.
    base = {(t["family"], i): t for i, t in enumerate_cells(e2, "none")}
    lines = ["\\begin{tabular}{lrrr}", "\\toprule",
             "Defense & Hijack prevented & Hijack introduced & Exact McNemar $p$ \\\\", "\\midrule"]
    for d in order:
        if d == "none":
            continue
        b = c = 0
        for i, t in enumerate_cells(e2, d):
            ref = base.get((t["family"], i))
            if ref is None or not t["family"]:
                continue
            b += ref["attack_at_call"] and not t["attack_at_call"]
            c += t["attack_at_call"] and not ref["attack_at_call"]
        n = b + c
        p = min(1.0, 2 * sum(math.comb(n, j) for j in range(0, min(b, c) + 1)) / 2 ** n) if n else 1.0
        lines.append(f"{DEFENSE_LABEL[d]} & {b} & {c} & {p:.1e} \\\\" if p < 0.001 else
                     f"{DEFENSE_LABEL[d]} & {b} & {c} & {p:.3f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write("tab_paired.tex", "\n".join(lines) + "\n")


def enumerate_cells(trials, defense):
    """(index within its (defense, family) cell, trial), in file order."""
    count = defaultdict(int)
    for t in trials:
        if t["defense"] != defense:
            continue
        key = t["family"] or "clean"
        yield count[key], t
        count[key] += 1


# ---------------------------------------------------------------- classifier
def fig_training(study):
    runs = [r for r in study["runs"] if r["use_task"]]
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    steps = range(1, len(runs[0]["train_loss"]) + 1)
    win = 10
    for r in runs:
        loss = r["train_loss"]
        smooth = [statistics.fmean(loss[max(0, i - win + 1) : i + 1]) for i in range(len(loss))]
        ax.plot(steps, smooth, color="#888", lw=0.7, alpha=0.8, label="train (10-step mean), per seed" if r is runs[0] else None)
    for key, colour, label in (("val_loss", "#2c5f8a", "validation (seen families)"),
                               ("heldout_loss", "#c0392b", "held-out families")):
        xs = [e["step"] for e in runs[0]["evals"]]
        vals = [[r["evals"][i][key] for r in runs] for i in range(len(xs))]
        mean = [statistics.fmean(v) for v in vals]
        ax.plot([max(x, 1) for x in xs], mean, color=colour, lw=1.4, label=label + ", mean of 3 seeds")
        ax.fill_between([max(x, 1) for x in xs], [min(v) for v in vals], [max(v) for v in vals], color=colour, alpha=0.2)
    ax.set_yscale("log")
    ax.set_xlabel("Optimizer step (batch size 16)")
    ax.set_ylabel("Token cross-entropy (log scale)")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "training.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIG / "training.pdf")


def ms(values):
    return f"{statistics.fmean(values):.3f} $\\pm$ {statistics.pstdev(values):.3f}"


def table_classifier(study, span_eval):
    lines = ["\\begin{tabular}{llrrrr}", "\\toprule",
             "Model & Test split & Span F1 & Injected chars removed & Benign chars kept & False alarms \\\\", "\\midrule"]
    for use_task, label in ((True, "Classifier, with task"), (False, "Classifier, task hidden")):
        runs = [r for r in study["runs"] if r["use_task"] == use_task]
        for split, sl in (("test_seen", "seen families"), ("test_heldout", "held-out families")):
            m = [r["metrics"][split] for r in runs]
            lines.append(f"{label} & {sl} & {ms([x['f1'] for x in m])} & {ms([x['removed_fraction'] for x in m])} & "
                         f"{ms([x['benign_kept'] for x in m])} & {ms([x['false_alarm_rate'] for x in m])} \\\\")
    if span_eval:
        lines.append("\\midrule")
        for d, label in (("protectai", "ProtectAI, whole output"), ("protectai_sent", "ProtectAI, per sentence"),
                         ("heuristic", "Keyword filter")):
            for split, sl in (("test_seen", "seen families"), ("test_heldout", "held-out families")):
                r = span_eval["defenses"][d][split]
                lines.append(f"{label} & {sl} & {r['f1']:.3f} & {r['removed_fraction']:.3f} & {r['benign_kept']:.3f} & "
                             f"{r['false_alarm_rate']:.3f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write("tab_spans.tex", "\n".join(lines) + "\n")


def fig_threshold(span_eval):
    sweep = span_eval["threshold_sweep"]
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    xs = [r["threshold"] for r in sweep]
    for split, colour, label in (("val", "#2c5f8a", "validation"), ("test_heldout", "#c0392b", "held-out families")):
        ax.plot(xs, [r[split]["recall"] for r in sweep], color=colour, lw=1.3, label=f"recall, {label}")
        ax.plot(xs, [r[split]["precision"] for r in sweep], color=colour, lw=1.0, ls="--", label=f"precision, {label}")
    ax.axvline(0.5, color="#888", lw=0.7, ls=":")
    ax.set_xlabel("Decision threshold $\\tau$")
    ax.set_ylabel("Character-level score")
    ax.set_ylim(0.6, 1.005)
    ax.legend(fontsize=6.5, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "threshold.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIG / "threshold.pdf")


# ---------------------------------------------------------------- attack wording
def fig_wording(rows):
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    lines = ["\\begin{tabular}{lrrr}", "\\toprule", "Wording style & Wordings & Trials & Attack success (\\%) \\\\", "\\midrule"]
    for x, (st, label, colour) in enumerate((("direct", "Replace the task\nor bare command", "#888"),
                                             ("staged", "Extra step before\nthe user's task", "#c0392b"))):
        sub = [r for r in rows if r["style"] == st]
        k, n = sum(r["k"] for r in sub), sum(r["n"] for r in sub)
        lo, hi = wilson(k, n)
        ax.bar(x, 100 * k / n, color=colour, alpha=0.35, width=0.6)
        ax.errorbar(x, 100 * k / n, yerr=[[100 * (k / n - lo)], [100 * (hi - k / n)]], color="k", capsize=3, lw=0.9)
        jitter = [x - 0.22 + 0.44 * ((i * 37) % 100) / 100 for i in range(len(sub))]
        ax.scatter(jitter, [100 * r["asr"] for r in sub], s=10, color=colour, zorder=3, edgecolor="k", linewidth=0.3)
        lines.append(f"{label.replace(chr(10), ' ')} & {len(sub)} & {n} & {pct(k, n)} \\\\")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Replace the task\nor bare command", "Extra step before\nthe user's task"])
    ax.set_ylabel("Undefended attack success (%)")
    ax.set_ylim(-3, 105)
    fig.tight_layout()
    fig.savefig(FIG / "wording.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIG / "wording.pdf")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write("tab_wording.tex", "\n".join(lines) + "\n")


# ---------------------------------------------------------------- adaptive
def adaptive(records, replay):
    n = len(records)
    start = sum(r["initial_survival"] >= 0.5 for r in records)
    k = sum(r["evaded"] for r in records)
    lines = ["\\begin{tabular}{lr}", "\\toprule", "Quantity & Value (\\%) \\\\", "\\midrule",
             f"Attacks evading the classifier before search ($n={n}$) & {pct(start, n)} \\\\",
             f"Attacks evading the classifier after search ($n={n}$) & {pct(k, n)} \\\\"]
    if replay:
        for d, label in (("none", "no defense"), ("classifier", "classifier")):
            rs = [t for t in replay if t["defense"] == d]
            a = sum(t["attack_at_call"] for t in rs)
            lines.append(f"Evading attacks that hijack the VLM, {label} ($n={len(rs)}$) & {pct(a, len(rs))} \\\\")
        rs = [t for t in replay if t["defense"] == "classifier"]
        a = sum(t["attack_at_call"] for t in rs)
        lines.append(f"End-to-end adaptive attack success against the classifier ($n={n}$) & {pct(a, n)} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write("tab_adaptive.tex", "\n".join(lines) + "\n")


# ---------------------------------------------------------------- harder data
OOD_SPLITS = [("test_seen", "Original: seen families"), ("test_heldout", "Original: held-out families"),
              ("ood_attack", "New attack shapes, original harmless text"),
              ("ood_benign", "Original attacks, new harmless writing"),
              ("ood_both", "New attack shapes, new harmless writing")]


def harder_text(ood):
    """Span metrics of the E2 checkpoint and baselines on the original and the harder splits."""
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.6), sharey=True)
    ys = list(range(len(OOD_SPLITS)))
    series = (("classifier", "Span classifier (ours)", "#2c5f8a"), ("protectai_sent", "ProtectAI, per sentence", "#999999"))
    for ax, key, count, title in ((axes[0], "removed_fraction", "n_injected", "Injected text removed (%)"),
                                  (axes[1], "false_alarm_rate", "n_clean", "Clean outputs flagged (%)")):
        for j, (d, label, colour) in enumerate(series):
            for y, (split, _) in zip(ys, OOD_SPLITS):
                r = ood[split][d]
                v, n = r[key], r[count]
                ax.barh(y + (j - 0.5) * 0.38, 100 * v, height=0.36, color=colour, label=label if y == 0 else None)
                if key == "false_alarm_rate":  # a per-example rate, so a binomial interval applies
                    lo, hi = wilson(round(v * n), n)
                    ax.errorbar(100 * v, y + (j - 0.5) * 0.38, xerr=[[100 * (v - lo)], [100 * (hi - v)]], color="k",
                                capsize=1.5, lw=0.7)
                ax.text(100 * v + 4, y + (j - 0.5) * 0.38, f"{100 * v:.0f}", va="center", fontsize=6.5)
        ax.set_xlim(0, 112)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xlabel(title)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([label for _, label in OOD_SPLITS])
    axes[0].invert_yaxis()
    axes[0].legend(fontsize=6.5, frameon=False, loc="lower left", bbox_to_anchor=(-1.25, -0.42), ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "harder.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIG / "harder.pdf")

    lines = ["\\begin{tabular}{ll" + "r" * 3 + "}", "\\toprule",
             "Test split & Defense & Injected chars removed & Benign chars kept & False alarms \\\\", "\\midrule"]
    for split, label in OOD_SPLITS + [("contrast_test", "Task-contrast test")]:
        for d, dl in (("classifier", "Span classifier (ours)"), ("protectai_sent", "ProtectAI, per sentence"),
                      ("protectai", "ProtectAI, whole output"), ("heuristic", "Keyword filter")):
            r = ood[split][d]
            lines.append(f"{label if d == 'classifier' else ''} & {dl} & {r['removed_fraction']:.3f} & "
                         f"{r['benign_kept']:.3f} & {r['false_alarm_rate']:.3f} \\\\")
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines.append("\\end{tabular}")
    write("tab_ood.tex", "\n".join(lines) + "\n")


def contrast(study):
    setups = (("A", "original", "shown"), ("B", "original + contrast", "shown"),
              ("C", "original", "hidden"), ("D", "original + contrast", "hidden"))
    runs = {k: [r for r in study["runs"] if r["setup"] == k] for k, _, _ in setups}
    kinds = (("swap", "Swap"), ("near_miss", "Near miss"), ("add_on", "Add-on"))
    fig, ax = plt.subplots(figsize=(5.0, 2.6))
    colours = {"A": "#bbbbbb", "B": "#2c5f8a", "C": "#e3c9a0", "D": "#c98a3a"}
    for j, (k, data, task) in enumerate(setups):
        for x, (kind, _) in enumerate(kinds):
            vals = [100 * r["metrics"]["contrast_test"]["pairs"][kind]["pair_accuracy"] for r in runs[k]]
            m = statistics.fmean(vals)
            ax.bar(x + (j - 1.5) * 0.2, m, width=0.19, color=colours[k],
                   label=f"{k}: {data}, task {task}" if x == 0 else None)
            ax.errorbar(x + (j - 1.5) * 0.2, m, yerr=[[m - min(vals)], [max(vals) - m]], color="k", capsize=2, lw=0.7)
    ax.set_xticks(range(len(kinds)))
    ax.set_xticklabels([label for _, label in kinds])
    ax.set_ylabel("Pair accuracy (%)")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=6.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.32), ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "contrast.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIG / "contrast.pdf")

    def m(k, split, key):
        return ms([r["metrics"][split][key] for r in runs[k]])

    def pa(k, kind):
        return ms([r["metrics"]["contrast_test"]["pairs"][kind]["pair_accuracy"] for r in runs[k]])

    lines = ["\\begin{tabular}{lll" + "r" * 6 + "}", "\\toprule",
             " & & & \\multicolumn{3}{c}{Contrast pair accuracy} & Seen & Held-out & New harmless \\\\",
             "Setup & Training data & Task & Swap & Near miss & Add-on & F1 & F1 & false alarms \\\\", "\\midrule"]
    for k, data, task in setups:
        lines.append(f"{k} & {data} & {task} & {pa(k, 'swap')} & {pa(k, 'near_miss')} & {pa(k, 'add_on')} & "
                     f"{m(k, 'test_seen', 'f1')} & {m(k, 'test_heldout', 'f1')} & {m(k, 'ood_benign', 'false_alarm_rate')} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write("tab_contrast.tex", "\n".join(lines) + "\n")


def ood_robot(screen, e2):
    fams = [f for f in FAMILIES]
    lines = ["\\begin{tabular}{lrr}", "\\toprule", "Family & Wordings screened & Hijack the undefended agent \\\\", "\\midrule"]
    tk = tn = 0
    for f in fams:
        rs = [t for t in screen if t["family"] == f]
        k = sum(t["attack_at_call"] for t in rs)
        tk, tn = tk + k, tn + len(rs)
        lines.append(f"{FAMILY_LABEL[f].replace('*', '')} & {len(rs)} & {k} \\\\")
    lines += ["\\midrule", f"All & {tn} & {tk} ({pct(tk, tn)}\\%) \\\\", "\\bottomrule", "\\end{tabular}"]
    write("tab_ood_screen.tex", "\n".join(lines) + "\n")
    lines = ["\\begin{tabular}{lrr}", "\\toprule", "Defense & Hijacks & Trials \\\\", "\\midrule"]
    for d in ("none", "protectai", "protectai_sent", "sandwich", "classifier"):
        rs = [t for t in e2 if t["defense"] == d]
        if rs:
            lines.append(f"{DEFENSE_LABEL[d]} & {sum(t['attack_at_call'] for t in rs)} & {len(rs)} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write("tab_ood_robot.tex", "\n".join(lines) + "\n")



def main():
    FIG.mkdir(parents=True, exist_ok=True)
    GEN.mkdir(parents=True, exist_ok=True)
    fig_harness()
    e1 = load_trials("runs/e1/trials.jsonl")
    if e1:
        table_e1(e1)
    e2 = load_trials("runs/e2/trials.jsonl")
    if e2:
        fig_e2(e2)
    study, span_eval = load_json("runs/classifier_study.json"), load_json("runs/span_eval.json")
    if study:
        fig_training(study)
        table_classifier(study, span_eval)
    if span_eval:
        fig_threshold(span_eval)
    rows = load_json("runs/template_asr.json")
    if rows:
        fig_wording(rows)
    ood = load_json("runs/ood_eval.json")
    if ood:
        harder_text(ood)
    cstudy = load_json("runs/contrast_study.json")
    if cstudy:
        contrast(cstudy)
    screen, ood_e2 = load_trials("runs/ood_screen/trials.jsonl"), load_trials("runs/ood_e2/trials.jsonl")
    if screen and ood_e2:
        ood_robot(screen, ood_e2)
    records = load_trials("runs/e3/adaptive.jsonl")
    if records:
        adaptive(records, load_trials("runs/e3_replay/trials.jsonl"))


if __name__ == "__main__":
    main()
