"""Command line entry point: `python -m toolguard <command>` or `toolguard <command>`."""

from __future__ import annotations

import argparse
import json

from .attacks import FAMILIES, HELD_OUT_FAMILIES


def _backend(args):
    from .backends import MockBackend, OpenAICompatBackend

    if args.backend == "mock":
        return MockBackend(seed=args.seed)
    return OpenAICompatBackend(model=args.model, base_url=args.base_url)


def _add_backend_args(p):
    p.add_argument("--backend", choices=("mock", "openai"), default="mock",
                   help="'openai' is any OpenAI-compatible server, e.g. vLLM serving Qwen3-VL")
    p.add_argument("--model", default="Qwen/Qwen3-VL-32B-Instruct")
    p.add_argument("--base-url", default=None, help="default: $OPENAI_BASE_URL or http://localhost:8000/v1")


def _load_extra(path):
    if not path:
        return None
    with open(path) as f:
        return json.load(f)


def cmd_build_dataset(args):
    from .dataset import build_dataset, save_dataset

    data = build_dataset(
        n_train=args.n_train, n_val=args.n_val, n_test=args.n_test, held_out=tuple(args.held_out),
        seed=args.seed, extra_templates=_load_extra(args.extra_templates),
    )
    save_dataset(data, args.out)
    for name, examples in data.items():
        injected = sum(bool(e.spans) for e in examples)
        print(f"{name}: {len(examples)} examples ({injected} injected) -> {args.out}/{name}.jsonl")


def cmd_llm_augment(args):
    from .augment import augment

    extra = augment(_backend(args), k=args.k, out_path=args.out)
    print(f"wrote {sum(map(len, extra.values()))} paraphrased templates to {args.out}")


def cmd_train(args):
    from .classifier import train
    from .dataset import load_split

    clf = train(
        f"{args.data}/train.jsonl", f"{args.data}/val.jsonl", args.out, model_name=args.model,
        epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, seed=args.seed, device=args.device,
        limit=args.limit,
    )
    print(f"saved best checkpoint to {args.out}")
    for split in ("test_seen", "test_heldout"):
        try:
            examples = load_split(f"{args.data}/{split}.jsonl")
        except FileNotFoundError:
            continue
        print(split, json.dumps(clf.evaluate(examples[: args.limit]), indent=2))


def cmd_eval_spans(args):
    from .dataset import load_split
    from .defenses import make_defense
    from .spans import span_report

    examples = load_split(args.split)
    defense = make_defense(args.defense, checkpoint=args.checkpoint, device=args.device)
    preds = [defense.spans(e.task, e.text) for e in examples]
    report = span_report(preds, [[tuple(s) for s in e.spans] for e in examples], [e.text for e in examples])
    print(json.dumps({"defense": defense.name, "split": args.split, **report}, indent=2))


def cmd_run(args):
    from .adaptive import load_replay
    from .defenses import make_defense
    from .executors import make_executor
    from .experiments import format_table, run_experiment, summarize

    defenses = [make_defense(d, checkpoint=args.checkpoint, device=args.device) for d in args.defenses]
    executor = make_executor(args.executor)
    results = run_experiment(
        _backend(args), defenses, families=tuple(args.families), n_trials=args.n,
        include_clean=not args.no_clean, executor=executor, n_rollouts=args.rollouts,
        use_image=not args.no_image, seed=args.seed, out_dir=args.out,
        injections=load_replay(args.replay) if args.replay else None,
    )
    print(format_table(summarize(results)))
    if args.backend == "mock" or args.executor == "mock":
        print("\nNOTE: mock components were used. These numbers exercise the pipeline; they are not results.")
    if args.out:
        print(f"\ntrials and summary written to {args.out}/")


def cmd_adaptive(args):
    from .adaptive import run_adaptive
    from .defenses import make_defense

    defense = make_defense(args.defense, checkpoint=args.checkpoint, device=args.device)
    out = run_adaptive(
        defense, n=args.n, budget=args.budget, mode=args.mode, seed=args.seed,
        extra_templates=_load_extra(args.extra_templates), out_dir=args.out,
    )
    print(json.dumps(out["summary"], indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(prog="toolguard", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("build-dataset", help="generate the span-labelled dataset")
    p.add_argument("--out", default="data")
    p.add_argument("--n-train", type=int, default=3000)
    p.add_argument("--n-val", type=int, default=400)
    p.add_argument("--n-test", type=int, default=600)
    p.add_argument("--held-out", nargs="*", default=list(HELD_OUT_FAMILIES), choices=FAMILIES)
    p.add_argument("--extra-templates", help="JSON from `llm-augment`")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_build_dataset)

    p = sub.add_parser("llm-augment", help="paraphrase attack templates with an LLM")
    _add_backend_args(p)
    p.add_argument("--k", type=int, default=5, help="rewrites requested per template")
    p.add_argument("--out", default="data/extra_templates.json")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_llm_augment)

    p = sub.add_parser("train", help="fine-tune the span classifier")
    p.add_argument("--data", default="data")
    p.add_argument("--out", default="checkpoints/span-clf")
    p.add_argument("--model", default="distilbert-base-uncased")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--limit", type=int, default=None, help="use only the first N examples (smoke tests)")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_train)

    p = sub.add_parser("eval-spans", help="span metrics of a removing defense on a dataset split")
    p.add_argument("--split", default="data/test_seen.jsonl")
    p.add_argument("--defense", default="classifier", choices=("heuristic", "prompt_guard", "classifier"))
    p.add_argument("--checkpoint", default="checkpoints/span-clf")
    p.add_argument("--device", default=None)
    p.set_defaults(fn=cmd_eval_spans)

    p = sub.add_parser("run", help="E1/E2: attack success per defense and attack family")
    _add_backend_args(p)
    p.add_argument("--defenses", nargs="+", default=["none"],
                   help="E1: none. E2: none delimiting prompt_guard classifier")
    p.add_argument("--families", nargs="+", default=list(FAMILIES), choices=FAMILIES)
    p.add_argument("--n", type=int, default=50, help="trials per (defense, condition)")
    p.add_argument("--no-clean", action="store_true", help="skip the no-attack condition")
    p.add_argument("--executor", choices=("none", "mock", "openvla"), default="none")
    p.add_argument("--rollouts", type=int, default=0, help="full arm rollouts per cell (headline: 120)")
    p.add_argument("--no-image", action="store_true", help="do not send the camera frame")
    p.add_argument("--replay", help="adaptive.jsonl from `adaptive`; replays the evading attacks")
    p.add_argument("--checkpoint", default="checkpoints/span-clf")
    p.add_argument("--device", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("adaptive", help="E3: token and paraphrase search against a defense")
    p.add_argument("--defense", default="classifier", choices=("heuristic", "prompt_guard", "classifier"))
    p.add_argument("--checkpoint", default="checkpoints/span-clf")
    p.add_argument("--mode", choices=("token", "paraphrase", "both"), default="both")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--budget", type=int, default=60, help="defense queries per attack")
    p.add_argument("--extra-templates", help="JSON from `llm-augment`, added to the paraphrase pool")
    p.add_argument("--device", default=None)
    p.add_argument("--out", default="runs/e3")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_adaptive)

    args = parser.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
