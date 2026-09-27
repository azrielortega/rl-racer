"""Train a PPO agent on the qualifying env with parallel envs, saving checkpoints and TensorBoard logs.

Usage (from the repo root): python -m agent.train [--steps 1000000] [--envs 8] [--name ppo]
Watch: tensorboard --logdir runs
"""

import argparse
from collections import deque
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from agent.env import RacerEnv

RUNS = Path(__file__).resolve().parent.parent / "runs"


class LapStats(BaseCallback):
    """Logs lap rate and lap times to TensorBoard, over the last 100 finished episodes."""

    def __init__(self):
        super().__init__()
        self.laps = deque(maxlen=100)
        self.times = deque(maxlen=100)
        self.best = None

    def _on_step(self):
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done:
                self.laps.append(info["lap"])
                if info["lap"]:
                    self.times.append(info["lap_time"])
                    self.best = min(self.best or info["lap_time"], info["lap_time"])
        return True

    def _on_rollout_end(self):
        if self.laps:
            self.logger.record("laps/lap_rate", sum(self.laps) / len(self.laps))
        if self.times:
            self.logger.record("laps/lap_time_mean", sum(self.times) / len(self.times))
            self.logger.record("laps/lap_time_best", self.best)


def main():
    parser = argparse.ArgumentParser(description="Train PPO on the qualifying env.")
    parser.add_argument("--steps", type=int, default=1_000_000, help="total agent steps across all envs")
    parser.add_argument("--envs", type=int, default=8, help="parallel envs, one process each")
    parser.add_argument("--name", default="ppo", help="run folder under runs/")
    args = parser.parse_args()

    out = RUNS / args.name
    env = make_vec_env(RacerEnv, n_envs=args.envs, vec_env_cls=SubprocVecEnv)
    model = PPO("MlpPolicy", env, device="cpu", tensorboard_log=str(out), verbose=1)
    # save_freq counts calls per env, so divide to save every ~100k total steps.
    checkpoints = CheckpointCallback(save_freq=max(100_000 // args.envs, 1), save_path=str(out / "checkpoints"))
    model.learn(total_timesteps=args.steps, callback=[checkpoints, LapStats()], tb_log_name="tb")
    model.save(out / "final")
    env.close()
    print(f"saved {out / 'final.zip'}")


if __name__ == "__main__":
    main()
