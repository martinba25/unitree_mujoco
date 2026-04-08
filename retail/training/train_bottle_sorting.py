import os
import sys
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

sys.path.insert(0, os.path.dirname(__file__))
from bottle_sorting_env import BottleSortingEnv

POLICY_DIR   = "/home/martinba/unitree_mujoco/retail/policies/bottle_sorting_ppo"
LOG_DIR      = "/home/martinba/unitree_mujoco/retail/policies/bottle_sorting_logs"
os.makedirs(POLICY_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

if __name__ == "__main__":
    print("🚀 Starte RoboTask Training: Bottle Sorting")
    env = make_vec_env(BottleSortingEnv, n_envs=4, monitor_dir=LOG_DIR)
    eval_callback = EvalCallback(BottleSortingEnv(), best_model_save_path=POLICY_DIR, log_path=LOG_DIR, eval_freq=2500, deterministic=True, render=False)
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=LOG_DIR, batch_size=256, learning_rate=3e-4, ent_coef=0.01)
    model.learn(total_timesteps=1_000_000, callback=eval_callback, progress_bar=True)
    model.save(f"{POLICY_DIR}/final_bottle_sorting_model")
