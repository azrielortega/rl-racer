import math

import pytest

from sim.config import load_config, load_track
from sim.geometry import point_segment_distance
from sim.physics import Car
from sim.race import Progress, bounce_off_walls, hits_wall, race_step, update_progress

CFG = load_config()
TRACK = load_track(CFG.track)
R = CFG.car_radius

# Three checkpoints across a corridor driven toward +x: x = 100, 200, 300 (the last is the finish).
# Oriented like gen_track's checkpoints (first end at +y for travel toward +x), so +x counts as forward.
CPS = [((x, 40), (x, -40)) for x in (100, 200, 300)]


def midpoint(seg):
    (x1, y1), (x2, y2) = seg
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def test_checkpoints_count_only_in_order():
    progress = Progress()
    assert update_progress(progress, (50, 0), (150, 0), CPS) == (1, False)
    assert update_progress(progress, (150, 0), (250, 0), CPS) == (1, False)
    assert progress.next_checkpoint == 2


def test_skipping_ahead_counts_nothing():
    progress = Progress()
    assert update_progress(progress, (150, 0), (250, 0), CPS) == (0, False)  # cp 1 while 0 is next
    assert progress.next_checkpoint == 0


def test_reversing_over_passed_checkpoint_counts_nothing():
    progress = Progress()
    update_progress(progress, (50, 0), (150, 0), CPS)
    assert update_progress(progress, (150, 0), (50, 0), CPS) == (0, False)
    assert update_progress(progress, (50, 0), (150, 0), CPS) == (0, False)  # cp 0 again: not next anymore


def test_crossing_next_checkpoint_backwards_counts_nothing():
    progress = Progress()
    assert update_progress(progress, (150, 0), (50, 0), CPS) == (0, False)
    assert progress.next_checkpoint == 0


def test_finish_line_completes_lap_and_wraps():
    progress = Progress(next_checkpoint=2)
    assert update_progress(progress, (250, 0), (350, 0), CPS) == (1, True)
    assert progress.laps == 1
    assert progress.next_checkpoint == 0


def just_past(i, direction):
    """Point 10% of the way from checkpoint i's middle toward its neighbour (direction +1 ahead, -1 behind)."""
    cps = TRACK.checkpoints
    (ax, ay), (bx, by) = midpoint(cps[i]), midpoint(cps[(i + direction) % len(cps)])
    return (ax + 0.1 * (bx - ax), ay + 0.1 * (by - ay))


def test_full_monza_lap_counts_every_checkpoint_once():
    progress = Progress()
    pos = (TRACK.start_poses[0].x, TRACK.start_poses[0].y)
    for i in range(len(TRACK.checkpoints)):
        target = just_past(i, +1)
        assert update_progress(progress, pos, target, TRACK.checkpoints) == (1, i == len(TRACK.checkpoints) - 1)
        pos = target
    assert progress.laps == 1


def test_monza_backwards_counts_nothing():
    progress = Progress()
    pos = (TRACK.start_poses[0].x, TRACK.start_poses[0].y)
    for i in reversed(range(len(TRACK.checkpoints) - 1)):
        target = just_past(i, -1)
        assert update_progress(progress, pos, target, TRACK.checkpoints) == (0, False)
        pos = target


def test_start_poses_are_clear_of_walls():
    for pose in TRACK.start_poses:
        assert not hits_wall(Car.at(pose), TRACK.walls, R)


def test_bounce_pushes_car_out_and_reverses_speed():
    wall = ((0, 0), (100, 0))
    car = Car(50, 6, -math.pi / 2, speed=200)  # 6 from the wall, radius 10, heading into it
    assert bounce_off_walls(car, [wall], R)
    assert point_segment_distance((car.x, car.y), *wall) == pytest.approx(R)
    assert car.speed == pytest.approx(-60)


def test_bounce_out_of_corner_touching_two_walls():
    walls = [((0, 0), (100, 0)), ((0, 0), (0, 100))]
    car = Car(5, 5, -3 * math.pi / 4, speed=100)
    assert bounce_off_walls(car, walls, R)
    for wall in walls:
        assert point_segment_distance((car.x, car.y), *wall) >= R - 1e-9
    assert car.speed == pytest.approx(-30)  # bounced once, not once per wall


def test_training_mode_crashes_without_moving_car():
    wall = ((0, 0), (100, 0))
    car = Car(50, 5, 0)
    assert hits_wall(car, [wall], R)
    assert (car.x, car.y) == (50, 5)
    assert not hits_wall(Car(50, 11, 0), [wall], R)


def test_race_step_full_throttle_from_start_crashes_eventually_in_both_modes():
    # Monza starts on the main straight, so flat out with no steering reaches the first corner's wall.
    for bounce in (False, True):
        car, progress = Car.at(TRACK.start_poses[0]), Progress()
        passed = 0
        for _ in range(600):
            result = race_step(car, progress, 1, 0, CFG, TRACK, bounce)
            passed += result.checkpoints
            if result.crashed:
                break
        assert result.crashed
        assert passed >= 1
        if bounce:
            assert car.speed < 0
