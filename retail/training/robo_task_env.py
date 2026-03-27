"""
robo_task_env.py
v2.2: The "Sailing-Ready" Version
- Fix: Korrekte G1-Aktuatoren (22-28) via _apply_action()
- Fix: Friction-Range [0.6, 1.0] gegen NaN-Instability
- Feature: Domain Randomization (Pos ±20cm, Masse ±30%)
- Feature: Bin-Approach Reward + Success Radius 0.25m
- Feature: Relative Pfade via os.path.dirname(__file__)
- Observation: Hand(3) + Obj(3) + Target(3) + Qpos[:20] = 29
"""

import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
import os


# ── Reward-Funktion v1.3 (mit Bin-Approach) ───────────────────────────
def compute_reward(hand_pos, obj_pos, target_pos, is_grasped):
    dist_hand_obj   = np.linalg.norm(hand_pos - obj_pos)
    dist_obj_target = np.linalg.norm(obj_pos - target_pos)

    # 1. Reach (deaktiviert wenn gegriffen)
    reward_reach = 0.0 if is_grasped else 2.0 * np.exp(-5.0 * dist_hand_obj)

    # 2. Grasp & Lift
    reward_grasp = 2.0 if is_grasped else 0.0
    reward_lift = 0.0
    if is_grasped and obj_pos[2] > 0.1:
        reward_lift = 1.0 + (obj_pos[2] * 2.0)

    # 3. Bin Approach (führt Arm zum Ziel)
    reward_approach = 0.0
    if is_grasped:
        reward_approach = (1.0 - np.clip(dist_obj_target / 0.5, 0, 1)) * 0.5
        reward_approach += 5.0 * np.exp(-3.0 * dist_obj_target)

    return reward_reach + reward_grasp + reward_lift + reward_approach


class RoboTaskEnv(gym.Env):
    """
    Gymnasium Environment für RoboTask Platform v2.2.
    Echter G1-Arm, Domain Randomization, Bin-Approach Reward.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.max_steps = 500
        self.success_radius = 0.25  # Toleranz für DR-Training

        self._step_count = 0
        self._success_count = 0
        self._episode_count = 0
        self._gripper_actuator = 28

        self._load_model()

        # Observation: Hand(3) + Obj(3) + Target(3) + Qpos[:20] = 29
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(29,), dtype=np.float32)
        # Action: x, y, z, gripper
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        self._renderer = None

    def _load_model(self):
        base_dir = os.path.dirname(__file__)
        xml_path = os.path.join(
            base_dir, "../../unitree_robots/g1/scene_training.xml")

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data  = mujoco.MjData(self.model)

        self._obj_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "flasche_r1")
        self._ee_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link")
        self._base_mass = self.model.body_mass[self._obj_body_id].copy()

        self._target_pos_base = np.array([0.4, 0.2, 0.4])

        print(f"✅ RoboTaskEnv v2.2: {self.model.nbody} Bodies")
        print(f"   Scene: scene_training.xml")
        print(f"   Objekt id={self._obj_body_id} | EE id={self._ee_body_id}")

    def _apply_action(self, action):
        """Korrekte G1-Arm Aktuator-Mappings (bewährt aus v2.0)"""
        scale = 0.05
        self.data.ctrl[22] += action[2] * scale * 0.3    # shoulder pitch: hoch/runter
        self.data.ctrl[23] += action[1] * scale * 0.2    # shoulder roll: links/rechts
        self.data.ctrl[24] += action[0] * scale * 0.2    # shoulder yaw: vor/zurück
        self.data.ctrl[25] += action[2] * scale * (-0.2) # elbow: gegenbewegung
        self.data.ctrl[27] += action[2] * scale * 0.1    # wrist pitch
        self.data.ctrl[28]  = np.clip(action[3], -1.0, 1.0)  # gripper

        np.clip(self.data.ctrl,
                self.model.actuator_ctrlrange[:, 0],
                self.model.actuator_ctrlrange[:, 1],
                out=self.data.ctrl)

    def _get_obs(self):
        hand_pos   = self.data.xpos[self._ee_body_id].copy()
        obj_pos    = self.data.xpos[self._obj_body_id].copy()
        target_pos = self._current_target_pos.copy()
        qpos_slice = self.data.qpos[:20].copy()

        return np.concatenate([
            hand_pos, obj_pos, target_pos, qpos_slice
        ]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self._step_count = 0
        self._episode_count += 1

        # 1. Flaschenposition ±20cm
        obj_jnt = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "flasche_r1")
        if obj_jnt >= 0:
            qadr = self.model.jnt_qposadr[obj_jnt]
            noise = np.random.uniform(-0.2, 0.2, 2)
            self.data.qpos[qadr:qadr+2] = np.array([0.4, 0.0]) + noise

        # 2. Target Position ±10cm (Z stabil)
        t_noise = np.random.uniform(-0.1, 0.1, 3)
        t_noise[2] = 0.0
        self._current_target_pos = self._target_pos_base + t_noise

        # 3. Masse ±30%
        self.model.body_mass[self._obj_body_id] = (
            self._base_mass * np.random.uniform(0.7, 1.3))

        # 4. Friction [0.6, 1.0] — fix gegen NaN-Instability
        for i in range(self.model.ngeom):
            geom_name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
            if "bottle" in geom_name:
                self.model.geom_friction[i, 0] = np.random.uniform(0.6, 1.0)

        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def step(self, action):
        self._step_count += 1
        self._apply_action(action)

        for _ in range(5):
            mujoco.mj_step(self.model, self.data)

        obs        = self._get_obs()
        hand_pos   = self.data.xpos[self._ee_body_id].copy()
        obj_pos    = self.data.xpos[self._obj_body_id].copy()
        target_pos = self._current_target_pos

        is_grasped = np.linalg.norm(hand_pos - obj_pos) < 0.05
        reward = compute_reward(hand_pos, obj_pos, target_pos, is_grasped)

        dist_target = np.linalg.norm(obj_pos - target_pos)
        success = dist_target < self.success_radius

        if success:
            reward += 100.0
            self._success_count += 1

        terminated = bool(success)
        truncated  = self._step_count >= self.max_steps

        return obs, reward, terminated, truncated, {"success": success}

    def render(self):
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(
                    self.model, height=480, width=640)
            self._renderer.update_scene(self.data)
            return self._renderer.render()
        return None

    def close(self):
        if self._renderer:
            self._renderer.close()
            self._renderer = None

    def success_rate(self):
        if self._episode_count == 0:
            return 0.0
        return self._success_count / self._episode_count


# ── Quick Test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🤖 RoboTaskEnv v2.2 Test")
    print("=" * 40)

    env = RoboTaskEnv()
    obs, _ = env.reset()

    hand_pos = obs[:3]
    obj_pos  = obs[3:6]
    dist = np.linalg.norm(hand_pos - obj_pos)

    print(f"\n📍 Hand:    {hand_pos.round(3)}")
    print(f"📍 Flasche: {obj_pos.round(3)}")
    print(f"📏 Distanz: {dist:.3f}m")
    print(f"📊 Obs shape: {obs.shape}")

    if dist < 0.5:
        print("✅ Flasche in Reichweite!")
    else:
        print(f"⚠️  Flasche zu weit: {dist:.2f}m")

    print(f"\n✅ RoboTaskEnv v2.2 bereit!")
    env.close()
