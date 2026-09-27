"""Gymnasium env"""

import gymnasium as gym
import numpy as np

from agent.sensors import OBSERVATION_SIZE, observe
from sim.config import load_config, load_track
from sim.physics import Car
from sim.race import Progress, race_step

CHECKPOINT_REWARD = 1.0
LAP_REWARD = 10.0
STEP_PENALTY = -0.01
OUT_OF_BOUNDS_PENALTY = -5.0
STALL_STEPS = 75  # agent steps (about 5 s) without a new checkpoint before the episode is cut off


def decode_action(action):
    """Map a Discrete(9) action to its (throttle, steer) pair (spec 3).

    Inputs:  action (int) - 0..8

    Outputs: (int, int) - throttle and steer, each -1/0/+1
    """
    return action // 3 - 1, action % 3 - 1


class RacerEnv(gym.Env):
    """One lap from a standing start: ends on the finish line (terminated), going out of bounds (terminated) or a stall (truncated)."""

    def __init__(self, cfg=None, track=None):
        self.cfg = cfg or load_config()
        self.track = track or load_track(self.cfg.track)
        self.observation_space = gym.spaces.Box(-1.0, 1.0, (OBSERVATION_SIZE,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(9)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.car = Car.at(self.track.start_pose)
        self.progress = Progress()
        self.physics_steps = 0
        self.steps_since_checkpoint = 0
        return self._observe(), {}

    def step(self, action):
        throttle, steer = decode_action(int(action))
        reward, terminated = STEP_PENALTY, False
        info = {"lap": False, "out_of_bounds": False}
        self.steps_since_checkpoint += 1
        # Frame skip (spec 2): the action is held for FRAME_SKIP physics steps and their rewards are summed.
        for _ in range(self.cfg.frame_skip):
            result = race_step(self.car, self.progress, throttle, steer, self.cfg, self.track)
            self.physics_steps += 1
            if result.checkpoints:
                reward += CHECKPOINT_REWARD * result.checkpoints
                self.steps_since_checkpoint = 0
            if result.went_out:
                # Qualifying (spec 6): leaving the road voids the lap, so stop at once.
                reward += OUT_OF_BOUNDS_PENALTY
                info["out_of_bounds"] = terminated = True
                break
            if result.lap:
                reward += LAP_REWARD
                info["lap"] = terminated = True
                info["lap_time"] = self.physics_steps * self.cfg.dt
                break
        truncated = not terminated and self.steps_since_checkpoint >= STALL_STEPS
        return self._observe(), reward, terminated, truncated, info

    def _observe(self):
        return np.array(observe(self.car, self.progress, self.track, self.cfg), dtype=np.float32)
