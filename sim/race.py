import math
from dataclasses import dataclass

from sim.geometry import closest_point, segments_cross
from sim.physics import step

BOUNCE = -0.3  # spec 6: speed multiplier when the car hits a wall in game mode


@dataclass
class Progress:
    next_checkpoint: int = 0
    laps: int = 0


@dataclass
class StepResult:
    checkpoints: int = 0  # checkpoints passed in order this step
    lap: bool = False  # crossed the finish line (last checkpoint) this step
    crashed: bool = False  # touched a wall this step


def update_progress(progress, before, after, checkpoints):
    """Advance the checkpoint counter if the move before->after crosses the next checkpoint in order.

    Inputs:  progress (Progress); before, after ((x, y)) - car position around one step; checkpoints (list[segment])

    Outputs: (int, bool) - checkpoints passed this step and whether a lap was completed
    """
    passed, lap = 0, False
    # Only the next checkpoint, crossed forwards, counts: reversing, re-crossing or skipping ahead earns nothing.
    while _crosses_forward(before, after, checkpoints[progress.next_checkpoint]):
        passed += 1
        if progress.next_checkpoint == len(checkpoints) - 1:
            progress.laps += 1
            lap = True
        progress.next_checkpoint = (progress.next_checkpoint + 1) % len(checkpoints)
        if passed == len(checkpoints):
            break
    return passed, lap


def _crosses_forward(before, after, checkpoint):
    (x1, y1), (x2, y2) = checkpoint
    # gen_track draws every checkpoint with the same orientation relative to the driving direction,
    # so a forward move always satisfies (checkpoint vector) x (move vector) > 0.
    forward = (x2 - x1) * (after[1] - before[1]) - (y2 - y1) * (after[0] - before[0]) > 0
    return forward and segments_cross(before, after, *checkpoint)


def hits_wall(car, walls, radius):
    """Check whether the car's circle overlaps any wall (training: this ends the episode).

    Inputs:  car (Car); walls (list[segment]); radius (float) - CAR_RADIUS

    Outputs: bool - True if any wall is closer than radius
    """
    p = (car.x, car.y)
    return any(math.dist(p, closest_point(p, a, b)) < radius for a, b in walls)


def bounce_off_walls(car, walls, radius):
    """Push the car out of any walls it overlaps and bounce its speed (game mode, spec 6).

    Inputs:  car (Car); walls (list[segment]); radius (float) - CAR_RADIUS

    Outputs: bool - True if the car touched a wall (car.x, car.y, car.speed are updated in place)
    """
    hit = False
    for a, b in walls:
        c = closest_point((car.x, car.y), a, b)
        d = math.dist((car.x, car.y), c)
        if d >= radius:
            continue
        hit = True
        if d == 0:
            # Centre exactly on the wall: no normal to push along, so back out the way the car came.
            nx, ny, d = -math.cos(car.heading), -math.sin(car.heading), 0.0
        else:
            nx, ny = (car.x - c[0]) / d, (car.y - c[1]) / d
        car.x += nx * (radius - d)
        car.y += ny * (radius - d)
    if hit:
        car.speed *= BOUNCE  # once per step, even when touching two walls in a corner
    return hit


def race_step(car, progress, throttle, steer, cfg, track, bounce):
    """One physics step with collisions and checkpoints, in the same order for training and the game.

    Inputs:  car (Car); progress (Progress); throttle, steer (int) - -1/0/+1; cfg (Config); track (Track);
             bounce (bool) - True for game mode (push out + bounce), False for training (crash only)

    Outputs: StepResult - checkpoints passed, lap completed, wall touched
    """
    before = (car.x, car.y)
    step(car, throttle, steer, cfg)
    if bounce:
        crashed = bounce_off_walls(car, track.walls, cfg.car_radius)
    else:
        crashed = hits_wall(car, track.walls, cfg.car_radius)
    passed, lap = update_progress(progress, before, (car.x, car.y), track.checkpoints)
    return StepResult(passed, lap, crashed)
