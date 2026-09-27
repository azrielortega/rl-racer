from dataclasses import dataclass

from sim.geometry import min_segment_distances, segments_cross
from sim.physics import corners, step


@dataclass
class Progress:
    next_checkpoint: int = 0
    laps: int = 0
    out_of_bounds: bool = False  # state after the last step, to detect the moment the car leaves the road


@dataclass
class StepResult:
    checkpoints: int = 0  # checkpoints passed in order this step
    lap: bool = False  # crossed the finish line (last checkpoint) this step
    out_of_bounds: bool = False  # car is off the road after this step
    went_out: bool = False  # car left the road this step (was on it the step before)


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


def is_out_of_bounds(car, track, cfg):
    """Check whether any corner of the car's rectangle is off the road (farther than width / 2 from the centerline).

    Inputs:  car (Car); track (Track); cfg (Config) - car size

    Outputs: bool - True if out of bounds
    """
    # The road edges are the centerline offset by width / 2, so this matches the drawn edges exactly.
    return bool((min_segment_distances(corners(car, cfg), track.centerline) > track.width / 2).any())


def race_step(car, progress, throttle, steer, cfg, track):
    """One physics step, then the bounds and checkpoint checks, in the same order for training and the game.

    Inputs:  car (Car); progress (Progress); throttle, steer (int) - -1/0/+1; cfg (Config); track (Track)

    Outputs: StepResult - checkpoints passed, lap completed, out-of-bounds state and whether it just went out
    """
    before = (car.x, car.y)
    step(car, throttle, steer, cfg)
    out = is_out_of_bounds(car, track, cfg)
    went_out = out and not progress.out_of_bounds
    progress.out_of_bounds = out
    passed, lap = update_progress(progress, before, (car.x, car.y), track.checkpoints)
    return StepResult(passed, lap, out, went_out)
