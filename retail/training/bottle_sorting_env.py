import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
import os

class BottleSortingEnv(gym.Env):
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        scene_path = os.path.join(os.path.dirname(__file__), "../scenes/bottle_sorting.xml")
        self.model = mujoco.MjModel.from_xml_path(scene_path)
        self.data = mujoco.MjData(self.model)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(22,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        self.objs = ["bottle_glass", "bottle_pet", "can_aluminium"]
        self.bins = ["glass_bin", "pet_bin", "can_bin"]
        self.sorted_mask = [False, False, False]
        self.step_count = 0

    def _get_obs(self):
        try:
            hand_pos = self.data.site("hand_site").xpos
        except:
            hand_pos = np.zeros(3)
        obj_positions = []
        for obj in self.objs:
            obj_positions.extend(self.data.body(obj).xpos)
        bin_positions = []
        for b in self.bins:
            bin_positions.extend(self.data.body(b).xpos)
        gripper_state = [self.data.ctrl[0]] if len(self.data.ctrl) > 0 else [0.0]
        return np.concatenate([hand_pos, obj_positions, bin_positions, gripper_state]).astype(np.float32)

    def step(self, action):
        self.step_count += 1
        mujoco.mj_step(self.model, self.data)
        obs = self._get_obs()
        reward = -0.01
        for i, (obj, b) in enumerate(zip(self.objs, self.bins)):
            if not self.sorted_mask[i]:
                dist = np.linalg.norm(self.data.body(obj).xpos - self.data.body(b).xpos)
                if dist < 0.1:
                    self.sorted_mask[i] = True
                    reward += 2.0
        terminated = all(self.sorted_mask)
        if terminated:
            reward += 10.0
        truncated = self.step_count >= 500
        return obs, reward, terminated, truncated, {"bodies": self.model.nbody}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        for obj in self.objs:
            qpos_adr = self.model.body(obj).jntadr[0]
            self.data.qpos[qpos_adr:qpos_adr+2] += np.random.uniform(-0.2, 0.2, size=2)
        self.sorted_mask = [False, False, False]
        self.step_count = 0
        return self._get_obs(), {}
