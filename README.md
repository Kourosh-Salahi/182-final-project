# Classifying Tool-Output Prompt Injections in an Open-Weight Robot Agent

A VLM orchestrator (Qwen3-VL) receives a user task and a camera frame and calls tools:
information tools that return text, and `execute_task`, which hands an instruction to a VLA
(OpenVLA) that moves the arm in LIBERO-Goal. The attacker controls only the text an
information tool returns. A small task-conditioned span classifier sits between each
information tool and the orchestrator and removes injected instruction spans.

## Status

| Piece | State |
| --- | --- |
| LIBERO-Goal tasks, scene objects, information tools | done |
| Four attack families, seed templates, span-labelled dataset | done (template-based; LLM paraphrasing implemented, not yet run) |
| Span classifier (DistilBERT / ModernBERT) training and evaluation | done, runs locally |
| Baselines: none, delimiting, keyword heuristic, Llama Prompt Guard 2 | done (Prompt Guard not yet run: gated model) |
| Tool-calling harness and E1/E2 runner | done, verified with the mock orchestrator only |
| Qwen3-VL backend (OpenAI-compatible, e.g. vLLM) | written, not yet run against a server |
| OpenVLA + LIBERO executor | written from the OpenVLA eval script, not yet run |
| E3 adaptive attack (token + paraphrase search) | done for defense evasion; replay through the agent supported |

Everything except the last three "not yet run" items works on a laptop.

## Quick start (laptop, no GPU)

```bash
pip install -e ".[train,dev]"
pytest -m "not slow"

python -m toolguard build-dataset --out data
python -m toolguard train --data data --out checkpoints/span-clf
python -m toolguard eval-spans --split data/test_heldout.jsonl

# E1/E2 with the scripted mock orchestrator and a coin-flip arm (pipeline check only)
python -m toolguard run --defenses none delimiting heuristic classifier --n 40 \
    --executor mock --rollouts 20 --out runs/dryrun

# E3: adaptive search against the classifier, then replay the evading attacks
python -m toolguard adaptive --defense classifier --out runs/e3
python -m toolguard run --defenses classifier --replay runs/e3/adaptive.jsonl --no-clean
```

The mock orchestrator follows any other-goal instruction it reads with a fixed probability.
Its numbers exercise the pipeline and are not results.

## Real runs (GPU box)

1. Serve the orchestrator with tool calling enabled, for example with vLLM
   (check the tool-call parser name for your vLLM version):

   ```bash
   vllm serve Qwen/Qwen3-VL-32B-Instruct --enable-auto-tool-choice --tool-call-parser hermes
   ```

2. Install LIBERO from source and the OpenVLA dependencies (`pip install -e ".[sim]"`).
3. Run:

   ```bash
   # E1: clean and undefended
   python -m toolguard run --backend openai --defenses none --n 200 \
       --executor openvla --rollouts 120 --out runs/e1
   # E2: defenses on seen and held-out families
   python -m toolguard run --backend openai --defenses none delimiting prompt_guard classifier \
       --n 200 --executor openvla --rollouts 120 --out runs/e2
   ```

`--rollouts N` rolls out the first N trials of each cell on the arm; the rest stop at the
first `execute_task` call. Fallback models: `--model Qwen/Qwen3-VL-8B-Instruct`; a SmolVLA
executor is not implemented yet.

To grow the dataset with LLM paraphrases of the attack templates:

```bash
python -m toolguard llm-augment --backend openai --model <open LLM> --out data/extra_templates.json
python -m toolguard build-dataset --extra-templates data/extra_templates.json
```

## Layout

| File | Purpose |
| --- | --- |
| `toolguard/tasks.py` | the ten LIBERO-Goal tasks, scene objects, instruction-to-task matching |
| `toolguard/tools.py` | tool schemas and benign tool responses with exact injected spans |
| `toolguard/attacks.py` | attack families and templates; `HELD_OUT_FAMILIES` |
| `toolguard/dataset.py` | train / val / test_seen / test_heldout splits |
| `toolguard/classifier.py` | span classifier: encoding, training, inference |
| `toolguard/defenses.py` | none, delimiting, heuristic, Prompt Guard 2, classifier |
| `toolguard/backends.py` | Qwen3-VL via OpenAI-compatible API; mock orchestrator |
| `toolguard/agent.py` | one trial of the harness, outcome labelling |
| `toolguard/executors.py` | OpenVLA on LIBERO; mock executor |
| `toolguard/experiments.py` | E1/E2 runner, paired trials, Wilson intervals |
| `toolguard/adaptive.py` | E3 token and paraphrase search |
| `toolguard/augment.py` | LLM paraphrasing of attack templates |

## Definitions

- **Attack success at the call**: the first `execute_task` call maps to the attacker's goal.
- **Attack success at the arm**: that call's rollout also passes LIBERO's success check for the attacker's goal.
- **Held-out families** (default): `false_recovery`, `tool_call_hijack`. Seen families also keep
  separate eval templates, so `test_seen` measures unseen wording.
- **Span F1**: character-level, micro-averaged over tool-output characters.

## Known gaps

- The dataset is template-based, and seen-family span F1 is about 1.0, so it is too easy as is.
  It needs the LLM paraphrases and more benign variety before the numbers mean much.
- Adaptive replay re-samples the benign text around the injection, so an attack that evaded
  in search can be caught on replay.
- Camera-borne injections are out of scope, as in the proposal.
