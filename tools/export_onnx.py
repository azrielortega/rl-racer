"""Export a trained PPO agent to ONNX for the browser, plus a parity file to check the TypeScript port against.

Usage (from the repo root): python -m tools.export_onnx runs/ppo/final.zip [-o web/agent]
Writes <out>.onnx (input "obs" float32 [1, 12], output "logits" float32 [1, 9]; the action is the argmax)
and <out>.parity.json (one deterministic run: car state, observation, logits and action for every agent step).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from stable_baselines3 import PPO

from agent.env import RacerEnv
from agent.sensors import OBSERVATION_SIZE


class Actor(torch.nn.Module):
    """The policy's action head only: observation in, one score per action out (no value head, no sampling)."""

    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, obs):
        latent = self.policy.mlp_extractor.forward_actor(self.policy.pi_features_extractor(obs))
        return self.policy.action_net(latent)


def export(model, path):
    """Write the actor network to an ONNX file.

    Inputs:  model (PPO) - trained agent; path (Path) - .onnx file to write

    Outputs: None
    """
    actor = Actor(model.policy).eval()
    torch.onnx.export(
        actor, torch.zeros(1, OBSERVATION_SIZE), str(path), input_names=["obs"], output_names=["logits"],
        opset_version=18, external_data=False,  # one self-contained file for the browser
    )


def record_run(model, session):
    """Drive one deterministic episode, checking every ONNX action against SB3's own choice.

    Inputs:  model (PPO); session (ort.InferenceSession) - the exported model

    Outputs: (list[dict], dict) - per-agent-step records, and the final step's info (lap, lap_time, out_of_bounds)
    """
    env = RacerEnv()
    obs, _ = env.reset(seed=0)
    steps, done, info = [], False, {}
    while not done:
        car, progress = env.car, env.progress
        logits = session.run(None, {"obs": obs[None]})[0][0]
        action = int(np.argmax(logits))
        expected = int(model.predict(obs, deterministic=True)[0])
        if action != expected:
            raise SystemExit(f"ONNX chose {action} but SB3 chose {expected} at agent step {len(steps)}")
        steps.append({
            "car": {"x": car.x, "y": car.y, "heading": car.heading, "speed": car.speed},
            "next_checkpoint": progress.next_checkpoint,
            "out_of_bounds": progress.out_of_bounds,
            "obs": obs.tolist(),
            "logits": logits.tolist(),
            "action": action,
        })
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    return steps, info


def main():
    parser = argparse.ArgumentParser(description="Export a trained agent to ONNX with a parity file.")
    parser.add_argument("model", help="trained PPO .zip (a checkpoint or final.zip)")
    parser.add_argument("-o", "--out", default="web/agent", help="output path without extension")
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    model = PPO.load(args.model, device="cpu")
    export(model, out.with_suffix(".onnx"))

    session = ort.InferenceSession(str(out.with_suffix(".onnx")))
    steps, info = record_run(model, session)
    ended = f"lap in {info['lap_time']:.2f}s" if info.get("lap") else "out of bounds" if info.get("out_of_bounds") else "stalled"
    with open(out.with_suffix(".parity.json"), "w") as f:
        json.dump({"model": args.model, "ended": ended, "steps": steps}, f)
    print(f"wrote {out.with_suffix('.onnx')} and {out.with_suffix('.parity.json')}")
    print(f"ONNX matched SB3 on all {len(steps)} agent steps; the run ended: {ended}")


if __name__ == "__main__":
    main()
