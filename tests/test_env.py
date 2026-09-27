import math

import pytest
from gymnasium.utils.env_checker import check_env

from agent.env import LAP_REWARD, OUT_OF_BOUNDS_PENALTY, STALL_STEPS, STEP_PENALTY, RacerEnv, decode_action

FULL_THROTTLE, COAST = 7, 4


def run(env, action):
    """Hold one action until the episode ends; returns (total reward, agent steps, terminated, truncated, info)."""
    total, steps = 0.0, 0
    while True:
        _, reward, terminated, truncated, info = env.step(action)
        total, steps = total + reward, steps + 1
        if terminated or truncated:
            return total, steps, terminated, truncated, info


def test_passes_gymnasium_checks():
    check_env(RacerEnv())


def test_actions_match_the_spec_table():
    assert [decode_action(a) for a in range(9)] == [(t, s) for t in (-1, 0, 1) for s in (-1, 0, 1)]


def test_reset_starts_stationary_on_the_start_line():
    env = RacerEnv()
    obs, _ = env.reset(seed=0)
    assert obs.shape == (12,)
    assert (env.car.x, env.car.y, env.car.heading) == (env.track.start_pose.x, env.track.start_pose.y, env.track.start_pose.heading)
    assert env.car.speed == 0 and obs[9] == 0 and obs[11] == 0


def test_full_throttle_goes_out_at_the_first_corner_and_ends():
    env = RacerEnv()
    env.reset(seed=0)
    total, steps, terminated, truncated, info = run(env, FULL_THROTTLE)
    assert terminated and not truncated and info["out_of_bounds"] and not info["lap"]
    checkpoints = env.progress.next_checkpoint
    assert checkpoints > 0  # it earned the straight before crashing out
    assert total == pytest.approx(checkpoints + steps * STEP_PENALTY + OUT_OF_BOUNDS_PENALTY)


def test_standing_still_is_cut_off_as_a_stall():
    env = RacerEnv()
    env.reset(seed=0)
    total, steps, terminated, truncated, _ = run(env, COAST)
    assert truncated and not terminated
    assert steps == STALL_STEPS
    assert total == pytest.approx(STALL_STEPS * STEP_PENALTY)


def test_crossing_the_finish_line_ends_with_the_lap_bonus():
    env = RacerEnv()
    env.reset(seed=0)
    # Just behind the finish line with only it left to cross; the start pose sits on it.
    env.progress.next_checkpoint = len(env.track.checkpoints) - 1
    env.car.x -= math.cos(env.car.heading) * 5
    env.car.y -= math.sin(env.car.heading) * 5
    env.car.speed = 200  # covers about 13 units in one agent step
    _, reward, terminated, truncated, info = env.step(FULL_THROTTLE)
    assert terminated and not truncated and info["lap"]
    assert reward == pytest.approx(1 + LAP_REWARD + STEP_PENALTY)
    assert info["lap_time"] == pytest.approx(env.physics_steps * env.cfg.dt)


def test_reset_clears_the_previous_episode():
    env = RacerEnv()
    env.reset(seed=0)
    run(env, FULL_THROTTLE)
    env.reset(seed=0)
    assert env.progress.next_checkpoint == 0 and not env.progress.out_of_bounds
    assert env.physics_steps == 0 and env.steps_since_checkpoint == 0
