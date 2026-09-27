"""Train a PPO agent on the qualifying env with parallel envs, saving checkpoints and TensorBoard logs.

Usage (from the repo root): python -m agent.train [--steps 1000000] [--envs 8] [--name ppo]
Resume: python -m agent.train --resume runs/ppo/final.zip --steps 1000000   (trains 1M more steps)
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
    parser.add_argument("--resume", help="saved model .zip to keep training (a checkpoint or final.zip)")
    args = parser.parse_args()

    out = RUNS / args.name
    env = make_vec_env(RacerEnv, n_envs=args.envs, vec_env_cls=SubprocVecEnv)
    if args.resume:
        model = PPO.load(args.resume, env=env, device="cpu", tensorboard_log=str(out), verbose=1)
    else:
        model = PPO("MlpPolicy", env, device="cpu", tensorboard_log=str(out), verbose=1)
    # save_freq counts calls per env, so divide to save every ~50k total steps.
    checkpoints = CheckpointCallback(save_freq=max(50_000 // args.envs, 1), save_path=str(out / "checkpoints"))
    # When resuming, keep counting steps from the saved model so checkpoint names and graphs continue on.
    try:
        model.learn(
            total_timesteps=args.steps,
            callback=[checkpoints, LapStats()],
            tb_log_name="tb",
            reset_num_timesteps=not args.resume,
        )
    except KeyboardInterrupt:
        print("\nstopped with Ctrl+C, saving the agent as it is now")
    model.save(out / "final")
    print(f"saved {out / 'final.zip'} at {model.num_timesteps} steps")
    try:
        env.close()
    except (BrokenPipeError, EOFError, ConnectionResetError):
        pass  # Ctrl+C also reaches the env worker processes, so they may already be gone


if __name__ == "__main__":
    main()
