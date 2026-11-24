"""Arm executors behind `execute_subgoal`.

`OpenVLALiberoExecutor` runs the public OpenVLA LIBERO-Goal checkpoint in the
LIBERO simulator and scores the rollout with LIBERO's success check for the
goal that was actually requested. `MockExecutor` is a coin flip for dry runs.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .tasks import Task


@dataclass
class ExecResult:
    success: bool
    steps: int = 0


class Executor:
    name = "executor"

    def initial_frame(self, task: Task, episode: int = 0):
        """Camera image of the scene before acting (PIL image), or None."""
        return None

    def execute(self, task: Task, episode: int = 0) -> ExecResult:
        """Roll out `task.instruction` and report whether that goal was reached."""
        raise NotImplementedError


class MockExecutor(Executor):
    """Succeeds with a fixed probability. Placeholder numbers, not OpenVLA's."""

    name = "mock"

    def __init__(self, success_rate: float = 0.75, seed: int = 0):
        self.success_rate = success_rate
        self.rng = random.Random(seed)

    def execute(self, task: Task, episode: int = 0) -> ExecResult:
        return ExecResult(self.rng.random() < self.success_rate, steps=0)


class OpenVLALiberoExecutor(Executor):
    """OpenVLA on LIBERO-Goal. Needs a GPU, LIBERO and the OpenVLA dependencies.

    Follows `experiments/robot/libero/run_libero_eval.py` from the OpenVLA repo
    (180 degree image flip, center crop, gripper remap, 10 settle steps).
    NOTE: written against that script but not yet run; check it reproduces the
    published LIBERO-Goal success rate before trusting arm-level numbers.
    """

    name = "openvla"
    MAX_STEPS = 300  # LIBERO-Goal horizon used by the OpenVLA eval
    SETTLE_STEPS = 10
    DUMMY_ACTION = [0, 0, 0, 0, 0, 0, -1]

    def __init__(
        self,
        checkpoint: str = "openvla/openvla-7b-finetuned-libero-goal",
        device: str = "cuda:0",
        load_in_8bit: bool = False,
        resolution: int = 256,
    ):
        import torch
        from libero.libero import benchmark
        from transformers import AutoModelForVision2Seq, AutoProcessor

        self.torch = torch
        self.device = device
        self.resolution = resolution
        self.processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)
        kwargs = dict(torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True)
        if load_in_8bit:
            self.model = AutoModelForVision2Seq.from_pretrained(checkpoint, load_in_8bit=True, **kwargs)
        else:
            self.model = AutoModelForVision2Seq.from_pretrained(checkpoint, **kwargs).to(device)
        self.suite = benchmark.get_benchmark_dict()["libero_goal"]()
        # Index suite tasks by language so we do not depend on our own numbering.
        self.suite_index = {self.suite.get_task(i).language.lower(): i for i in range(self.suite.n_tasks)}

    def _make_env(self, task: Task, episode: int):
        import os

        from libero.libero import get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        idx = self.suite_index[task.instruction.lower()]
        spec = self.suite.get_task(idx)
        bddl = os.path.join(get_libero_path("bddl_files"), spec.problem_folder, spec.bddl_file)
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=self.resolution, camera_widths=self.resolution)
        env.seed(0)
        env.reset()
        states = self.suite.get_task_init_states(idx)
        obs = env.set_init_state(states[episode % len(states)])
        for _ in range(self.SETTLE_STEPS):
            obs, _, _, _ = env.step(self.DUMMY_ACTION)
        return env, obs

    @staticmethod
    def _frame(obs):
        from PIL import Image

        return Image.fromarray(obs["agentview_image"][::-1, ::-1].copy()).convert("RGB")

    def _policy_image(self, obs):
        from PIL import Image

        img = self._frame(obs).resize((224, 224), Image.LANCZOS)
        # The fine-tuned checkpoints were trained with random crops; evaluate on a 90% center crop.
        side = round(224 * 0.9**0.5)
        off = (224 - side) // 2
        return img.crop((off, off, off + side, off + side)).resize((224, 224), Image.BILINEAR)

    def initial_frame(self, task: Task, episode: int = 0):
        env, obs = self._make_env(task, episode)
        try:
            return self._frame(obs)
        finally:
            env.close()

    def execute(self, task: Task, episode: int = 0) -> ExecResult:
        import numpy as np

        env, obs = self._make_env(task, episode)
        prompt = f"In: What action should the robot take to {task.instruction.lower()}?\nOut:"
        try:
            for step in range(self.MAX_STEPS):
                inputs = self.processor(prompt, self._policy_image(obs)).to(self.device, dtype=self.torch.bfloat16)
                action = np.asarray(self.model.predict_action(**inputs, unnorm_key="libero_goal", do_sample=False))
                # Gripper: [0, 1] -> {-1, +1}, then flip sign for the LIBERO convention.
                action[-1] = -np.sign(2 * action[-1] - 1)
                obs, _, done, _ = env.step(action.tolist())
                if done:
                    return ExecResult(True, step + 1)
            return ExecResult(False, self.MAX_STEPS)
        finally:
            env.close()


def make_executor(name: str, **kwargs) -> Executor | None:
    if name == "none":
        return None
    if name == "mock":
        return MockExecutor(**kwargs)
    if name == "mujoco":
        from .sim import MujocoExecutor

        return MujocoExecutor(**kwargs)
    if name == "openvla":
        return OpenVLALiberoExecutor(**kwargs)
    raise ValueError(f"unknown executor '{name}'")
