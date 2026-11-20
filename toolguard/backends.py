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

    It looks up the first object of the task, then calls `execute_task`. If a
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
        call = tool_call("execute_task", {"instruction": instruction}, call_id)
        return {"role": "assistant", "content": "", "tool_calls": [call]}
