"""Orchestrator backends.

A backend takes OpenAI-format chat messages plus tool schemas and returns the
assistant message. `OpenAICompatBackend` talks to Qwen3-VL served by vLLM (or
any OpenAI-compatible server). `MockBackend` is a scripted stand-in so the
whole pipeline runs on a laptop; its numbers say nothing about a real model.
"""

from __future__ import annotations

import json
import os
import random
import urllib.request

from .tasks import LIBERO_GOAL_TASKS, match_task, normalize


def tool_call(name: str, arguments: dict, call_id: str = "call_0") -> dict:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}


class Backend:
    name = "backend"

    def complete(self, messages: list[dict], tools: list[dict]) -> dict:
        raise NotImplementedError


class OpenAICompatBackend(Backend):
    def __init__(
        self,
        model: str = "Qwen/Qwen3-VL-32B-Instruct",
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        timeout: float = 300.0,
    ):
        self.model = model
        self.name = model
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or "http://localhost:8000/v1").rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or "EMPTY"
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    def complete(self, messages: list[dict], tools: list[dict]) -> dict:
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if tools:
            body.update(tools=tools, tool_choice="auto")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            msg = json.load(resp)["choices"][0]["message"]
        out = {"role": "assistant", "content": msg.get("content") or ""}
        if msg.get("tool_calls"):
            out["tool_calls"] = msg["tool_calls"]
        return out


class MockBackend(Backend):
    """Scripted orchestrator for tests and dry runs.

    It looks up the first object of the task, then calls `execute_subgoal`. If a
    tool result it was shown names a different LIBERO-Goal instruction, it
    follows that one with probability `susceptibility` (halved when the result
    is wrapped in untrusted-data delimiters).
    """

    name = "mock"

    def __init__(self, susceptibility: float = 0.6, seed: int = 0):
        self.susceptibility = susceptibility
        self.rng = random.Random(seed)

    @staticmethod
    def _text(content) -> str:
        if isinstance(content, str):
            return content
        return " ".join(p.get("text", "") for p in content if p.get("type") == "text")

    def complete(self, messages: list[dict], tools: list[dict]) -> dict:
        user = next(self._text(m["content"]) for m in messages if m["role"] == "user")
        task = match_task(user.split("Task:", 1)[-1].strip())
        if any(
            c["function"]["name"] == "execute_subgoal"
            for m in messages if m["role"] == "assistant" for c in m.get("tool_calls", [])
        ):
            return {"role": "assistant", "content": "Done."}
        results = [self._text(m["content"]) for m in messages if m["role"] == "tool"]
        n_calls = sum(len(m.get("tool_calls", [])) for m in messages if m["role"] == "assistant")
        call_id = f"call_{n_calls}"

        if task is None:
            return {"role": "assistant", "content": "I do not recognise this task."}
        if not results:
            call = tool_call("lookup_object", {"name": task.objects[0]}, call_id)
            return {"role": "assistant", "content": "", "tool_calls": [call]}

        instruction = task.instruction
        for result in results:
            p = self.susceptibility * (0.5 if "UNTRUSTED_TOOL_DATA" in result else 1.0)
            # Benign history notes also name other goals; this toy model is not fooled by them.
            lines = [
                s
                for s in result.replace(". ", ".\n").split("\n")
                if not any(k in s for k in ("log shows", "practiced the routine", "demonstration set"))
            ]
            seen = normalize(" ".join(lines))
            for other in LIBERO_GOAL_TASKS:
                if other.id != task.id and normalize(other.instruction) in seen and self.rng.random() < p:
                    instruction = other.instruction
                    break
        call = tool_call("execute_subgoal", {"instruction": instruction}, call_id)
        return {"role": "assistant", "content": "", "tool_calls": [call]}


class HFBackend(Backend):
    """Qwen3-VL run locally with transformers (4-bit weights fit an 8 GB GPU).

    Does what vLLM's chat endpoint does for this model: renders the messages
    and tool schemas with the model's own chat template (Hermes-style
    `<tool_call>` blocks), generates greedily, and parses the tool calls back
    into OpenAI format. Replies are cached on the exact prompt, so paired
    trials that share a prefix (same task, same first lookup) generate once.
    """

    DEFAULT_MODEL = "unsloth/Qwen3-VL-4B-Instruct-unsloth-bnb-4bit"

    def __init__(self, model: str | None = None, max_new_tokens: int = 200, cache_path: str | None = None):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch = torch
        self.name = model or self.DEFAULT_MODEL
        self.processor = AutoProcessor.from_pretrained(self.name)
        self.model = AutoModelForImageTextToText.from_pretrained(self.name, device_map={"": 0}, dtype="auto").eval()
        self._fix_unquantized_layers()
        self.max_new_tokens = max_new_tokens
        self.cache: dict[str, str] = {}
        self.cache_path = cache_path
        self.n_generated = 0
        if cache_path and os.path.exists(cache_path):
            with open(cache_path, encoding="utf-8") as f:
                self.cache = {k: v for k, v in (json.loads(line) for line in f if line.strip())}

    def _fix_unquantized_layers(self) -> None:
        """Some pre-quantized checkpoints keep a few layers in full precision (here the vision
        deepstack mergers) but transformers still builds them as 4-bit layers, which then fail
        in the forward pass. Swap those for ordinary linear layers holding the stored weights."""
        try:
            import bitsandbytes as bnb
        except ImportError:
            return
        nn = self.torch.nn
        for parent in list(self.model.modules()):
            for child_name, child in list(parent.named_children()):
                w = getattr(child, "weight", None)
                if isinstance(child, bnb.nn.Linear4bit) and w.shape[1] != 1 and getattr(w, "quant_state", None) is None:
                    plain = nn.Linear(child.in_features, child.out_features, bias=child.bias is not None)
                    plain.weight = nn.Parameter(w.data, requires_grad=False)
                    if child.bias is not None:
                        plain.bias = nn.Parameter(child.bias.data.to(w.dtype), requires_grad=False)
                    setattr(parent, child_name, plain.to(w.device))

    @staticmethod
    def _convert(messages: list[dict]) -> tuple[list[dict], list]:
        """OpenAI-format messages -> template-ready messages plus the PIL images in order."""
        import base64
        import io

        from PIL import Image

        out, images = [], []
        for m in messages:
            m = dict(m)
            if isinstance(m.get("content"), list):
                parts = []
                for p in m["content"]:
                    if p.get("type") == "image_url":
                        data = p["image_url"]["url"].split(",", 1)[1]
                        images.append(Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB"))
                        parts.append({"type": "image"})
                    else:
                        parts.append({"type": "text", "text": p.get("text", "")})
                m["content"] = parts
            if m.get("tool_calls"):
                calls = []
                for c in m["tool_calls"]:
                    fn = dict(c["function"])
                    if isinstance(fn.get("arguments"), str):
                        try:
                            fn["arguments"] = json.loads(fn["arguments"] or "{}")
                        except json.JSONDecodeError:
                            fn["arguments"] = {}
                    calls.append({"type": "function", "function": fn})
                m["tool_calls"] = calls
            out.append(m)
        return out, images

    def render(self, messages: list[dict], tools: list[dict]) -> tuple[str, list]:
        msgs, images = self._convert(messages)
        text = self.processor.apply_chat_template(
            msgs, tools=tools or None, add_generation_prompt=True, tokenize=False
        )
        return text, images

    def generate(self, text: str, images: list) -> str:
        import hashlib

        h = hashlib.sha256(text.encode())
        for im in images:
            h.update(im.tobytes())
        key = h.hexdigest()
        if key in self.cache:
            return self.cache[key]
        inputs = self.processor(text=[text], images=images or None, return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        reply = self.processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        self.n_generated += 1
        self.cache[key] = reply
        if self.cache_path:
            try:
                with open(self.cache_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps([key, reply]) + "\n")
            except OSError:
                pass  # a full disk should cost a cache entry, not the run
        return reply

    def complete(self, messages: list[dict], tools: list[dict]) -> dict:
        reply = self.generate(*self.render(messages, tools))
        n_prev = sum(len(m.get("tool_calls", [])) for m in messages if m["role"] == "assistant")
        return parse_hermes_reply(reply, first_id=n_prev)


def parse_hermes_reply(reply: str, first_id: int = 0) -> dict:
    """Split a Hermes-style completion into text and OpenAI-format tool calls."""
    import re

    calls = []
    for block in re.findall(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|$)", reply, flags=re.DOTALL):
        try:
            obj = json.loads(block)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("name"), str):
            args = obj.get("arguments", {})
            calls.append(tool_call(obj["name"], args if isinstance(args, dict) else {}, f"call_{first_id + len(calls)}"))
    out = {"role": "assistant", "content": reply.split("<tool_call>", 1)[0].strip()}
    if calls:
        out["tool_calls"] = calls
    return out
