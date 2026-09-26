"""Load the shared physics constants (config.json) and the track it points to."""

import json
from dataclasses import dataclass
from pathlib import Path

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
    turn_rate: float
    car_radius: float
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
    walls: list[Segment]
    checkpoints: list[Segment]  # in driving order; the last one is the finish line
    start_poses: list[Pose]


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

    Outputs: Track - walls and checkpoints as ((x1, y1), (x2, y2)) tuples, plus start poses
    """
    with open(ROOT / path) as f:
        raw = json.load(f)
    return Track(
        width=raw["width"],
        length=raw["length"],
        walls=[_segment(s) for s in raw["walls"]],
        checkpoints=[_segment(s) for s in raw["checkpoints"]],
        start_poses=[Pose(p["x"], p["y"], p["heading"]) for p in raw["start_poses"]],
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
        f"{len(track.start_poses)} start poses, width {track.width}, length {track.length}"
    )
