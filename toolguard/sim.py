"""A MuJoCo version of the LIBERO-Goal kitchen scene, for camera frames and demos.

The scene has the same objects and the same ten goals as LIBERO-Goal. The arm
is a floating gripper driven by a scripted controller: `execute(task)` plays a
hand-written pick/place/pull/turn motion for that goal, kinematically (no
contact dynamics). It stands in for the VLA policy so that a hijacked
`execute_subgoal` call can be *seen*: the arm does whatever the orchestrator
asked for. It says nothing about how well a learned policy would do.

Needs `pip install mujoco pillow`.
"""

from __future__ import annotations

import math
import random

from .executors import ExecResult, Executor
from .tasks import Task

XML = """
<mujoco model="kitchen">
  <visual>
    <global offwidth="640" offheight="480"/>
    <headlight ambient="0.45 0.45 0.45" diffuse="0.6 0.6 0.6" specular="0.1 0.1 0.1"/>
    <quality shadowsize="2048"/>
  </visual>
  <asset>
    <texture name="wood" type="2d" builtin="flat" rgb1="0.80 0.68 0.50" width="64" height="64"/>
    <material name="table" texture="wood" rgba="0.80 0.68 0.50 1"/>
    <material name="cab" rgba="0.55 0.36 0.20 1"/>
    <material name="drawer" rgba="0.66 0.46 0.27 1"/>
    <material name="metal" rgba="0.75 0.75 0.78 1"/>
  </asset>
  <worldbody>
    <light pos="0 -0.6 1.6" dir="0 0.3 -1" diffuse="0.5 0.5 0.5"/>
    <camera name="front" pos="0 -1.05 0.95" xyaxes="1 0 0 0 0.62 0.78"/>
    <geom name="floor" type="plane" size="3 3 0.1" pos="0 0 -0.4" rgba="0.30 0.32 0.36 1"/>
    <geom name="table" type="box" size="0.56 0.44 0.02" pos="0 0 -0.02" material="table"/>

    <body name="cabinet" pos="0 0.30 0">
      <geom type="box" size="0.19 0.11 0.008" pos="0 0 0.308" material="cab"/>
      <geom type="box" size="0.19 0.11 0.008" pos="0 0 0.008" material="cab"/>
      <geom type="box" size="0.008 0.11 0.158" pos="-0.182 0 0.158" material="cab"/>
      <geom type="box" size="0.008 0.11 0.158" pos="0.182 0 0.158" material="cab"/>
      <geom type="box" size="0.19 0.008 0.158" pos="0 0.102 0.158" material="cab"/>
      <body name="drawer_bottom" pos="0 0 0.062">
        <joint name="drawer_bottom" type="slide" axis="0 -1 0" range="0 0.16"/>
        <geom type="box" size="0.17 0.008 0.042" pos="0 -0.102 0" material="drawer"/>
        <geom type="box" size="0.17 0.095 0.005" pos="0 0 -0.037" material="drawer"/>
        <geom type="box" size="0.05 0.008 0.008" pos="0 -0.118 0" material="metal"/>
      </body>
      <body name="drawer_middle" pos="0 0 0.158">
        <joint name="drawer_middle" type="slide" axis="0 -1 0" range="0 0.16"/>
        <geom type="box" size="0.17 0.008 0.042" pos="0 -0.102 0" material="drawer"/>
        <geom type="box" size="0.17 0.095 0.005" pos="0 0 -0.037" material="drawer"/>
        <geom type="box" size="0.05 0.008 0.008" pos="0 -0.118 0" material="metal"/>
      </body>
      <body name="drawer_top" pos="0 0 0.254">
        <joint name="drawer_top" type="slide" axis="0 -1 0" range="0 0.16"/>
        <geom type="box" size="0.17 0.008 0.042" pos="0 -0.102 0" material="drawer"/>
        <geom type="box" size="0.17 0.095 0.005" pos="0 0 -0.037" material="drawer"/>
        <geom type="box" size="0.05 0.008 0.008" pos="0 -0.118 0" material="metal"/>
      </body>
    </body>

    <body name="rack" pos="-0.40 0.30 0">
      <geom type="box" size="0.11 0.05 0.012" pos="0 0 0.012" rgba="0.70 0.52 0.30 1"/>
      <geom type="box" size="0.008 0.05 0.03" pos="-0.09 0 0.042" rgba="0.70 0.52 0.30 1"/>
      <geom type="box" size="0.008 0.05 0.03" pos="0.09 0 0.042" rgba="0.70 0.52 0.30 1"/>
    </body>

    <body name="stove" pos="0.38 0.02 0">
      <geom type="box" size="0.12 0.12 0.02" pos="0 0 0.02" rgba="0.22 0.22 0.25 1"/>
      <geom name="burner" type="cylinder" size="0.075 0.003" pos="0 0.015 0.042" rgba="0.08 0.08 0.08 1"/>
      <body name="knob" pos="0 -0.128 0.02">
        <joint name="knob" type="hinge" axis="0 1 0" range="0 1.6"/>
        <geom type="cylinder" size="0.018 0.008" euler="1.5708 0 0" rgba="0.85 0.85 0.85 1"/>
        <geom type="box" size="0.003 0.003 0.016" pos="0 -0.009 0.006" rgba="0.9 0.1 0.1 1"/>
      </body>
    </body>

    <body name="bowl" pos="0 0.0 0.025">
      <freejoint name="bowl"/>
      <geom type="cylinder" size="0.06 0.025" rgba="0.07 0.07 0.07 1"/>
      <geom type="cylinder" size="0.05 0.002" pos="0 0 0.0245" rgba="0.22 0.22 0.22 1"/>
    </body>
    <body name="plate" pos="0 -0.26 0.006">
      <freejoint name="plate"/>
      <geom type="cylinder" size="0.09 0.006" rgba="0.96 0.96 0.96 1"/>
      <geom type="cylinder" size="0.065 0.001" pos="0 0 0.006" rgba="0.86 0.86 0.88 1"/>
    </body>
    <body name="wine bottle" pos="-0.38 0.0 0.09">
      <freejoint name="wine bottle"/>
      <geom type="cylinder" size="0.03 0.09" rgba="0.08 0.35 0.16 1"/>
      <geom type="cylinder" size="0.012 0.04" pos="0 0 0.125" rgba="0.08 0.35 0.16 1"/>
      <geom type="cylinder" size="0.031 0.03" pos="0 0 -0.01" rgba="0.92 0.88 0.75 1"/>
    </body>
    <body name="cream cheese" pos="0.15 -0.03 0.015">
      <freejoint name="cream cheese"/>
      <geom type="box" size="0.035 0.022 0.015" rgba="0.25 0.45 0.85 1"/>
      <geom type="box" size="0.036 0.008 0.0155" rgba="0.85 0.87 0.9 1"/>
    </body>

    <body name="gripper" mocap="true" pos="0.24 -0.30 0.50">
      <geom type="box" size="0.03 0.045 0.012" pos="0 0 0.05" rgba="0.15 0.15 0.18 1" contype="0" conaffinity="0"/>
      <geom type="cylinder" size="0.014 0.2" pos="0 0 0.26" rgba="0.6 0.6 0.65 1" contype="0" conaffinity="0"/>
      <geom name="finger_l" type="box" size="0.008 0.008 0.03" pos="0 0.04 0.012" rgba="0.9 0.55 0.1 1" contype="0" conaffinity="0"/>
      <geom name="finger_r" type="box" size="0.008 0.008 0.03" pos="0 -0.04 0.012" rgba="0.9 0.55 0.1 1" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
</mujoco>
"""

HOME = (0.24, -0.30, 0.50)
CABINET_TOP = 0.316
MOVABLE = ("bowl", "plate", "wine bottle", "cream cheese")
# Height of each object's body origin above the surface it rests on, and where the gripper holds it.
REST = {"bowl": 0.025, "plate": 0.006, "wine bottle": 0.09, "cream cheese": 0.015}
GRASP = {"bowl": 0.03, "plate": 0.01, "wine bottle": 0.17, "cream cheese": 0.02}
UPRIGHT = (1.0, 0.0, 0.0, 0.0)
ON_SIDE = (math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0)  # bottle lying along x


class KitchenSim:
    def __init__(self, width: int = 480, height: int = 360):
        import mujoco

        self.mj = mujoco
        self.model = mujoco.MjModel.from_xml_string(XML)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)
        self.burner = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "burner")
        self.fingers = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in ("finger_l", "finger_r")]
        self._finger_y = [float(self.model.geom_pos[g][1]) for g in self.fingers]
        self.reset()

    # ----- state access -------------------------------------------------
    def _adr(self, joint: str) -> int:
        return int(self.model.jnt_qposadr[self.mj.mj_name2id(self.model, self.mj.mjtObj.mjOBJ_JOINT, joint)])

    def pos(self, name: str) -> tuple[float, float, float]:
        a = self._adr(name)
        return tuple(float(v) for v in self.data.qpos[a : a + 3])

    def set_pose(self, name: str, pos, quat=None) -> None:
        a = self._adr(name)
        self.data.qpos[a : a + 3] = pos
        if quat is not None:
            self.data.qpos[a + 3 : a + 7] = quat

    def joint(self, name: str) -> float:
        return float(self.data.qpos[self._adr(name)])

    def set_joint(self, name: str, value: float) -> None:
        self.data.qpos[self._adr(name)] = value

    @property
    def gripper(self) -> tuple[float, float, float]:
        return tuple(float(v) for v in self.data.mocap_pos[0])

    def reset(self, seed: int = 0) -> None:
        self.mj.mj_resetData(self.model, self.data)
        rng = random.Random(seed)
        for name in MOVABLE:
            x, y, z = self.pos(name)
            self.set_pose(name, (x + rng.uniform(-0.02, 0.02), y + rng.uniform(-0.02, 0.02), z))
        self.data.mocap_pos[0] = HOME
        self.model.geom_rgba[self.burner] = (0.08, 0.08, 0.08, 1)
        self._set_fingers(open_=True)
        self.mj.mj_forward(self.model, self.data)

    def _set_fingers(self, open_: bool) -> None:
        for g, y in zip(self.fingers, self._finger_y):
            self.model.geom_pos[g][1] = y if open_ else y * 0.55

    def render(self):
        from PIL import Image

        self.mj.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, camera="front")
        return Image.fromarray(self.renderer.render())

    # ----- scripted motions ---------------------------------------------
    def _move(self, target, frames, steps: int = 10, carry: str | None = None, carry_quat=None, also=None):
        """Move the gripper to `target` in a straight line, carrying an object if given."""
        start = self.gripper
        offset = None
        if carry:
            p = self.pos(carry)
            offset = tuple(p[i] - start[i] for i in range(3))
        for k in range(1, steps + 1):
            t = k / steps
            t = t * t * (3 - 2 * t)  # ease in and out
            g = tuple(start[i] + (target[i] - start[i]) * t for i in range(3))
            self.data.mocap_pos[0] = g
            if carry:
                self.set_pose(carry, tuple(g[i] + offset[i] for i in range(3)), carry_quat)
            if also:
                also(t)
            frames.append(self.render())

    def _pick_place(self, obj: str, dest, frames, quat=None) -> None:
        """Pick `obj` from above and set its body origin down at `dest`."""
        x, y, z = self.pos(obj)
        grasp = GRASP[obj]
        self._move((x, y, z + grasp + 0.18), frames)
        self._move((x, y, z + grasp), frames, steps=7)
        self._set_fingers(open_=False)
        lift = max(z, dest[2]) + grasp + 0.2
        self._move((x, y, lift), frames, steps=7, carry=obj)
        if quat is not None:
            self.set_pose(obj, self.pos(obj), quat)
        self._move((dest[0], dest[1], lift), frames, steps=12, carry=obj)
        self._move((dest[0], dest[1], dest[2] + grasp), frames, steps=7, carry=obj)
        self.set_pose(obj, dest, quat)
        self._set_fingers(open_=True)
        self._move((dest[0], dest[1], dest[2] + grasp + 0.18), frames, steps=7)

    def _open_drawer(self, which: str, frames) -> None:
        z = {"bottom": 0.062, "middle": 0.158, "top": 0.254}[which]
        handle = (0.0, 0.30 - 0.135, z)
        self._move((handle[0], handle[1] - 0.10, z + 0.02), frames)
        self._move((handle[0], handle[1], z + 0.02), frames, steps=6)
        self._set_fingers(open_=False)
        joint = f"drawer_{which}"
        self._move((handle[0], handle[1] - 0.15, z + 0.02), frames, steps=12,
                   also=lambda t: self.set_joint(joint, 0.15 * t))
        self._set_fingers(open_=True)
        self._move((handle[0], handle[1] - 0.22, z + 0.12), frames, steps=6)

    def _stove_top(self, obj: str):
        return (0.38, 0.035, 0.045 + REST[obj])

    def execute(self, task_id: int, frames: list | None = None) -> list:
        """Play the scripted motion for one LIBERO-Goal task. Returns the rendered frames."""
        frames = [] if frames is None else frames
        if task_id == 0:
            self._open_drawer("middle", frames)
        elif task_id == 1:
            self._pick_place("bowl", self._stove_top("bowl"), frames)
        elif task_id == 2:
            self._pick_place("wine bottle", (0.09, 0.30, CABINET_TOP + REST["wine bottle"]), frames)
        elif task_id == 3:
            self._open_drawer("top", frames)
            self._pick_place("bowl", (0.0, 0.30 - 0.15, 0.254 - 0.032 + REST["bowl"]), frames)
        elif task_id == 4:
            self._pick_place("bowl", (-0.07, 0.30, CABINET_TOP + REST["bowl"]), frames)
        elif task_id == 5:
            x, y, z = self.pos("plate")
            self._move((x - 0.12, y, 0.2), frames)
            self._move((x - 0.12, y, 0.03), frames, steps=6)
            dest = (0.38, -0.20, z)
            self._move((dest[0] - 0.12, dest[1], 0.03), frames, steps=16,
                       also=lambda t: self.set_pose("plate", (x + (dest[0] - x) * t, y + (dest[1] - y) * t, z)))
            self._move((dest[0] - 0.12, dest[1], 0.2), frames, steps=6)
        elif task_id == 6:
            bx, by, bz = self.pos("bowl")
            self._pick_place("cream cheese", (bx, by, bz + 0.027 + REST["cream cheese"]), frames)
        elif task_id == 7:
            knob = (0.38, 0.02 - 0.16, 0.05)
            self._move((knob[0], knob[1] - 0.06, 0.12), frames)
            self._move(knob, frames, steps=6)
            self._set_fingers(open_=False)

            def turn(t):
                self.set_joint("knob", 1.5 * t)
                glow = 0.08 + 0.87 * t
                self.model.geom_rgba[self.burner] = (glow, 0.08 + 0.12 * t, 0.05, 1)

            self._move(knob, frames, steps=12, also=turn)
            self._set_fingers(open_=True)
            self._move((knob[0], knob[1] - 0.06, 0.16), frames, steps=6)
        elif task_id == 8:
            px, py, pz = self.pos("plate")
            self._pick_place("bowl", (px, py, pz + 0.007 + REST["bowl"]), frames)
        elif task_id == 9:
            self._pick_place("wine bottle", (-0.40, 0.30, 0.024 + 0.031), frames, quat=ON_SIDE)
        else:
            raise ValueError(f"unknown task id {task_id}")
        self._move(HOME, frames, steps=8)
        return frames

    # ----- success checks -----------------------------------------------
    def _near(self, name: str, xy, tol: float) -> bool:
        x, y, _ = self.pos(name)
        return math.hypot(x - xy[0], y - xy[1]) <= tol

    def check(self, task_id: int) -> bool:
        """Whether the scene currently satisfies the goal of a LIBERO-Goal task."""
        bowl, plate, bottle = self.pos("bowl"), self.pos("plate"), self.pos("wine bottle")
        on_cabinet = lambda p: abs(p[0]) < 0.19 and abs(p[1] - 0.30) < 0.11 and p[2] > CABINET_TOP  # noqa: E731
        if task_id == 0:
            return self.joint("drawer_middle") > 0.1
        if task_id == 1:
            return self._near("bowl", (0.38, 0.035), 0.08) and bowl[2] > 0.05
        if task_id == 2:
            return on_cabinet(bottle)
        if task_id == 3:
            return self.joint("drawer_top") > 0.1 and self._near("bowl", (0.0, 0.15), 0.08) and 0.2 < bowl[2] < 0.3
        if task_id == 4:
            return on_cabinet(bowl)
        if task_id == 5:
            return self._near("plate", (0.38, -0.20), 0.07)
        if task_id == 6:
            return self._near("cream cheese", bowl[:2], 0.05) and self.pos("cream cheese")[2] > bowl[2]
        if task_id == 7:
            return self.joint("knob") > 1.0
        if task_id == 8:
            return self._near("bowl", plate[:2], 0.05) and bowl[2] > plate[2]
        if task_id == 9:
            return self._near("wine bottle", (-0.40, 0.30), 0.06)
        return False


class MujocoExecutor(Executor):
    """`execute_subgoal` backed by the scripted MuJoCo arm. Also supplies the VLM's camera frames."""

    name = "mujoco"

    def __init__(self):
        self.sim = KitchenSim()

    def initial_frame(self, task: Task, episode: int = 0):
        self.sim.reset(seed=episode)
        return self.sim.render()

    def execute(self, task: Task, episode: int = 0) -> ExecResult:
        self.sim.reset(seed=episode)
        frames = self.sim.execute(task.id)
        return ExecResult(self.sim.check(task.id), steps=len(frames))
