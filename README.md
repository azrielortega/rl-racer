# RL Racer

A top-down qualifying game: the player tries to beat the lap time of a PPO agent. The agent is trained here in Python (Gymnasium + Stable-Baselines3) and exported to ONNX so it can drive live in the browser game, which runs the same physics in TypeScript. The full design is in [spec.md](spec.md).

## Setup

Needs Python 3.11+ (developed on 3.13). Run everything from the repo root.

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Install the CPU build of PyTorch first. The policy is a small network, so a GPU doesn't speed up training, and the CPU build is about 200MB instead of 2GB+. With conda, activate your env and run the same two `pip install` lines.

Check it works:

```bash
python -m pytest
```

## Drive

```bash
python -m tools.drive
```

| Key | Action |
|---|---|
| W / Up | Throttle |
| S / Down | Brake, then reverse |
| A, D / Left, Right | Steer |
| V | Show or hide the sensor rays |
| R | Restart |
| Esc | Quit |

The road edges don't block the car. Leaving the road turns the car red and adds to the out-of-bounds count.

## Train

```bash
python -m agent.train                                  # 1M steps, 8 parallel envs, saved in runs/ppo/
python -m agent.train --steps 3000000 --name agent_v2  # longer run in its own folder
```

| Flag | Default | Meaning |
|---|---|---|
| `--steps` | 1000000 | Agent steps to train (more steps when resuming) |
| `--envs` | 8 | Parallel envs, one process each. About the number of physical CPU cores. |
| `--name` | `ppo` | Run folder under `runs/` |
| `--resume` | | A saved `.zip` to keep training from |

1M steps takes about 12–15 minutes on an 8-core CPU. Each run saves to `runs/<name>/`:

- `checkpoints/rl_model_<steps>_steps.zip`: a snapshot every 50k steps
- `final.zip`: the agent at the end
- `tb_1/`: TensorBoard logs

**Stop early** with Ctrl+C. It saves `final.zip` before exiting.

**Resume** from any checkpoint or `final.zip`. Step counts, checkpoint names and graphs carry on from where they stopped:

```bash
python -m agent.train --name agent_v2 --resume runs/agent_v2/final.zip --steps 1000000
```

**Monitor** in a second terminal, then open http://localhost:6006:

```bash
tensorboard --logdir runs
```

| Graph | What to look for |
|---|---|
| `rollout/ep_rew_mean` | Should rise. About 36 means it completes full laps. |
| `laps/lap_rate` | Share of the last 100 episodes that finished a lap. Should reach close to 1. |
| `laps/lap_time_mean`, `laps/lap_time_best` | Should fall as it gets faster. |

The rewards are set in [agent/env.py](agent/env.py): +1 per checkpoint, +10 for the lap, -0.01 per agent step and -5 for leaving the road (which also ends the episode). If lap times stop improving, a stronger `STEP_PENALTY` (e.g. -0.05) pushes the agent to go faster.

## Watch the agent

Any checkpoint or `final.zip` can drive in the same window:

```bash
python -m tools.drive --model runs/agent_v2/final.zip
python -m tools.drive --model runs/agent_v2/checkpoints/rl_model_500000_steps.zip
```

The agent picks its best action every 4 physics steps, exactly as in training. Each run ends like a training episode (lap, out of bounds, or about 5 s without a new checkpoint) and restarts. The HUD shows how the last run ended. You can watch checkpoints while training is still running.

## Export to ONNX

```bash
python -m tools.export_onnx runs/agent_v2/final.zip -o web/agent
python -m tools.export_onnx runs/agent_v2/checkpoints/rl_model_250000_steps.zip -o web/agent_250000
```

This writes two files:

- `<out>.onnx`: the policy, about 32KB. Input `obs` is float32 `[1, 12]` (the observation in spec 4); output `logits` is float32 `[1, 9]`. The action is the index of the largest logit, where `throttle = action // 3 - 1` and `steer = action % 3 - 1`.
- `<out>.parity.json`: one deterministic run with the car state, observation, logits and action at every agent step. Use it to check the TypeScript sensors and physics: from each recorded state, the TS observation should match `obs` to about 1e-4, and the model should pick the same `action`.

The script checks that the ONNX model makes the same choice as the trained agent at every step, then prints how the run ended (e.g. `lap in 15.65s`). Without `-o` it writes `web/agent` and overwrites the previous export. Different checkpoints of one run make natural difficulty levels, since later checkpoints lap faster.

## Tracks

Tracks are authored as a centerline plus a road width, and `tools/gen_track.py` generates the edges, checkpoints and start pose (spec 8):

```bash
python tools/gen_track.py tracks/circuit.centerline.json -o tracks/circuit.json --svg tracks/circuit.svg
```

The track in use and all physics constants are set in [config.json](config.json), which the TypeScript game loads too. An agent only knows the track it was trained on, so changing the track means retraining.

## Limitations and future work

**Limitations**

- **One hand-made track.** The only track is inspired by one of the circuits in Formula 1. Its centerline points in `tracks/circuit.centerline.json` were placed with help from AI to trace that circuit's layout, so it follows the real track's shape only approximately.
- **The agent only knows the track it was trained on.** Its sensors aren't tied to one track, but it has only ever practised on this one circuit, so a new track needs its own training run.

**Future work**

- **Track creator.** Let users build their own tracks by placing centerline dots in the browser and setting a road width. `tools/gen_track.py` already turns a list of dots into a full track (edges, checkpoints, start pose) and warns about corners that are too tight, so the creator mainly needs an editor on top of it and a way to train or adapt an agent for each new track.

## Layout

| Path | Contents |
|---|---|
| `sim/` | Physics, geometry, track loading, checkpoints and out-of-bounds (the logic ported to TypeScript) |
| `agent/` | Sensors (observation), Gymnasium env, PPO training |
| `tools/` | Keyboard/agent viewer, track generator, ONNX export |
| `tracks/` | Authored centerlines and generated tracks |
| `tests/` | pytest suite |
| `runs/` | Training output (git-ignored) |
| `web/` | ONNX exports for the website |
