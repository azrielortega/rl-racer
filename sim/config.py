"""Load the shared physics constants (config.json) and the track it points to."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

Point = tuple[float, float]
Segment = tuple[Point, Point]


@dataclass(frozen=True)
class Config:
    dt: float
    frame_skip: int
    accel: float
    drag: float
    max_speed: float
    max_reverse: float
    min_turn_radius: float
    grip: float  # max sideways acceleration; sets the turning radius at speed
    car_length: float
    car_width: float
    ray_max: float
    track: str


@dataclass(frozen=True)
class Pose:
    x: float
    y: float
    heading: float


@dataclass(frozen=True)
class Track:
    width: float
    length: float
    # numpy (N, 2, 2) arrays of endpoint pairs, so the rays and bounds check test every segment at once.
    walls: np.ndarray  # road edges: drawn and used by the sensor rays, they don't block the car
    centerline: np.ndarray  # closed loop; the car is out of bounds beyond width / 2 from it
    checkpoints: list[Segment]  # in driving order; the last one is the finish line
    start_pose: Pose  # one qualifying car; the ghost passes through, so no second slot


def load_config(path=ROOT / "config.json"):
    """Read config.json, mapping its UPPER_CASE keys onto Config fields.

    Inputs:  path (str | Path) - config file, defaults to the repo's config.json

    Outputs: Config - immutable constants; unknown or missing keys raise TypeError
    """
    with open(path) as f:
        raw = json.load(f)
    return Config(**{k.lower(): v for k, v in raw.items()})


def load_track(path):
    """Read a generated track JSON (from tools/gen_track.py).

    Inputs:  path (str | Path) - track file; relative paths resolve from the repo root

    Outputs: Track - walls and centerline as (N, 2, 2) arrays, checkpoints as ((x1, y1), (x2, y2)) tuples, start pose
    """
    with open(ROOT / path) as f:
        raw = json.load(f)
    return Track(
        width=raw["width"],
        length=raw["length"],
        walls=np.array(raw["walls"], dtype=float),
        centerline=np.array(list(zip(raw["centerline"], raw["centerline"][1:] + raw["centerline"][:1])), dtype=float),
        checkpoints=[_segment(s) for s in raw["checkpoints"]],
        start_pose=Pose(raw["start_pose"]["x"], raw["start_pose"]["y"], raw["start_pose"]["heading"]),
    )


def _segment(s):
    (x1, y1), (x2, y2) = s
    return (float(x1), float(y1)), (float(x2), float(y2))


if __name__ == "__main__":
    cfg = load_config()
    track = load_track(cfg.track)
    print(cfg)
    print(
        f"{cfg.track}: {len(track.walls)} wall segments, {len(track.checkpoints)} checkpoints, "
        f"start {track.start_pose}, width {track.width}, length {track.length}"
    )
