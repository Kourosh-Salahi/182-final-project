# How VLM agent harnesses are built, and how that compares to ours and to VLA harnesses

Short answer: our tool-calling loop is built the same way standard VLM/LLM agent harnesses are.
A VLA harness is a different kind of loop, and has no text channel for a tool-output injection
to enter. The injection problem, and our defense, live entirely on the VLM side.

## 1. VLM / LLM agent harnesses

All of the common ones are the same loop:

1. Build a chat: system prompt, user task (text plus images for a VLM), tool schemas.
2. The model replies with text or a structured tool call.
3. The harness runs the tool and appends the result as a `tool` message.
4. Repeat until the model stops calling tools or a step limit is hit.

| Harness | How tools reach the model | How results come back | Where a defense plugs in |
| --- | --- | --- | --- |
| Qwen function calling (Qwen-Agent, vLLM, transformers) | JSON Schema list, rendered into the system turn by the model's chat template (Hermes style) | `tool` role message (`function` role in Qwen-Agent) | Not provided; you wrap the tool executor |
| AgentDojo | `AgentPipeline([SystemMessage, InitQuery, LLM, ToolsExecutionLoop([ToolsExecutor, LLM])])` | `ToolsExecutor` appends tool messages | Pipeline elements inside the loop, after `ToolsExecutor` |
| OpenAI-compatible servers (vLLM) | `tools=` on the chat endpoint, parsed back with a tool-call parser (`hermes` for Qwen) | `tool` message with `tool_call_id` | Client side, before the result is appended |

Notes that matter for us:

- Qwen's docs recommend Hermes-style tool use and say the template that implements it ships in
  the tokenizer's chat template. The model writes `<tool_call>{"name": ..., "arguments": ...}</tool_call>`
  and tool results are wrapped in `<tool_response>` tags inside a user turn. Our local backend
  (`HFBackend`) renders with that same template and parses the same tags, so it is equivalent to
  serving the model with `vllm serve ... --enable-auto-tool-choice --tool-call-parser hermes`.
- AgentDojo ships four defenses, all text-level and all placed where our classifier sits:
  `tool_filter`, `transformers_pi_detector` (ProtectAI DeBERTa-v3 prompt-injection-v2, scoring the
  whole tool message), `spotlighting_with_delimiting` (wraps outputs in `<<` `>>` plus a system
  sentence), and `repeat_user_prompt` (re-adds the user query after each tool run). We reimplement
  the last three as baselines with the same wording.
- A VLM adds one thing to this loop: images in the user turn (and sometimes in tool results, for
  example screenshots in computer-use agents). That opens a second injection channel (text in the
  image). The proposal puts that channel out of scope; our attacker only controls tool text.

## 2. Our harness

`toolguard/agent.py` is the loop above with one extra stage:

```
user task + camera frame -> Qwen3-VL -> lookup_object(name) -> corpus (attacker may poison)
                                                             -> defense (classifier or baseline)
                              Qwen3-VL <- cleaned output  <-
                              Qwen3-VL -> execute_subgoal(text) -> [VLA policy -> arm]
```

Same as a standard VLM harness: OpenAI-format messages, JSON Schema tools, the model's native
tool-call format, one `tool` message per result, a step limit.

Different by design:

- The defense is a first-class stage between the tool executor and the model, and it is given the
  trusted user task. AgentDojo's detector slot is in the same place but its detector never sees
  the task.
- We stop the episode at the first `execute_subgoal` call and label it (user's goal, attacker's
  goal, something else). AgentDojo instead runs to completion and checks environment state.
- Only one tool has side effects, so "was the agent hijacked" is a single unambiguous call.

## 3. VLA harnesses

A VLA evaluation harness is a closed control loop, not a chat loop:

| | VLM agent harness | VLA harness (OpenVLA on LIBERO, vla-eval) |
| --- | --- | --- |
| Model interface | chat messages in, text or tool call out | one `predict(observation) -> action` |
| Step rate | one model call per decision, seconds apart | every control step (VLA and primitives at 15 Hz in VoLo) |
| Input | task, images, growing text history incl. tool results | current camera frame(s) and one fixed instruction string |
| Output | discrete, named tool calls with arguments | continuous 7-DoF action (or a short action chunk) |
| Environment side | tool executor | `reset / step / make_obs / get_step_result` (vla-eval) |
| Success | task-specific check on calls or state | simulator predicate at episode end |
| History | whole conversation | none or a few frames |

vla-eval makes the split explicit: the model implements a single `predict()`, the benchmark
implements four methods, and the two talk over a socket. Nothing in that loop reads retrieved text,
so a tool-output injection cannot reach a VLA directly. It can only arrive as the instruction
string, which is whatever the orchestrator passed to `execute_subgoal`.

## 4. Where the two meet: VLA as a tool

VoLo (the paper our proposal builds on) runs a VLM agent with the VLA as one tool next to
`grasp`, `place` and perception tools. The VLA call is asynchronous and interruptible: the arm
keeps moving at 15 Hz while the VLM watches a front camera at 0.2 Hz and can switch tools.
It reports that a weak open orchestrator helps much less (Qwen3-VL-8B: 19.95% vs 41.80% for the
full system), with several times more VLM-level failures.

Our `execute_subgoal` is the simplest version of that pattern: synchronous, one call, no
monitoring. That is enough for the security question, because the hijack is decided at the moment
the VLM chooses the argument of `execute_subgoal`. Whether the arm then succeeds is a property of
the VLA, not of the injection or the defense.

## 5. What this means for the project

- The thing being attacked and defended is the VLM's tool loop. The headline metric is attack
  success at the `execute_subgoal` call. Arm-level success is an optional downstream measurement.
- Because our loop matches standard harnesses, text-only injection defenses transfer directly and
  are fair baselines: spotlighting, datamarking, sandwich, instructional prevention, the ProtectAI
  detector, Prompt Guard 2, and a PromptArmor-style LLM filter.
- Closest prior work to our classifier is filtering of tool outputs: PromptArmor (an LLM quotes the
  injection and it is removed) and DataFilter (a fine-tuned Llama-3.1-8B that rewrites the data
  given the user prompt; reported 2.2% average attack success vs 5.9% for PromptArmor and 22.8% for
  sandwich). Ours differs in being a small encoder that tags spans, about 100x smaller, and in
  being evaluated inside a multimodal robot agent.

## Sources

- Qwen function calling: https://qwen.readthedocs.io/en/latest/framework/function_call.html
- AgentDojo pipeline and defenses: https://agentdojo.spylab.ai/concepts/agent_pipeline/ ,
  https://agentdojo.spylab.ai/api/agent_pipeline/defenses/
- VoLo: https://huggingface.co/papers/2606.07723
- vla-eval: https://www.alphaxiv.org/abs/2603.13966
- DataFilter: https://arxiv.org/abs/2510.19207
- PromptArmor: https://arxiv.org/abs/2507.15219
- MELON: https://arxiv.org/abs/2502.05174
