import math

from sim.config import load_config, load_track
from sim.physics import Car
from sim.race import Progress, is_out_of_bounds, race_step, update_progress

CFG = load_config()
TRACK = load_track(CFG.track)

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


def test_full_lap_counts_every_checkpoint_once():
    progress = Progress()
    pos = (TRACK.start_pose.x, TRACK.start_pose.y)
    for i in range(len(TRACK.checkpoints)):
        target = just_past(i, +1)
        assert update_progress(progress, pos, target, TRACK.checkpoints) == (1, i == len(TRACK.checkpoints) - 1)
        pos = target
    assert progress.laps == 1


def test_driving_backwards_counts_nothing():
    progress = Progress()
    pos = (TRACK.start_pose.x, TRACK.start_pose.y)
    for i in reversed(range(len(TRACK.checkpoints) - 1)):
        target = just_past(i, -1)
        assert update_progress(progress, pos, target, TRACK.checkpoints) == (0, False)
        pos = target


def test_start_pose_is_in_bounds():
    assert not is_out_of_bounds(Car.at(TRACK.start_pose), TRACK, CFG)


def car_beside_centerline(offset, turn=0.0):
    """Car next to the middle of the first centerline segment, `offset` to the side, facing along it plus `turn`."""
    (x1, y1), (x2, y2) = TRACK.centerline[0]
    length = math.dist((x1, y1), (x2, y2))
    nx, ny = -(y2 - y1) / length, (x2 - x1) / length
    heading = math.atan2(y2 - y1, x2 - x1) + turn
    return Car((x1 + x2) / 2 + nx * offset, (y1 + y2) / 2 + ny * offset, heading)


def test_any_corner_over_the_edge_is_out_of_bounds():
    half = TRACK.width / 2
    side = CFG.car_width / 2  # facing along the road, the corners stick out half the car width sideways
    for sign in (1, -1):
        assert not is_out_of_bounds(car_beside_centerline(sign * (half - side - 1)), TRACK, CFG)
        assert is_out_of_bounds(car_beside_centerline(sign * (half - side + 1)), TRACK, CFG)


def test_car_across_the_road_uses_its_length():
    half = TRACK.width / 2
    reach = CFG.car_length / 2  # turned 90 degrees, the corners stick out half the car length sideways
    assert not is_out_of_bounds(car_beside_centerline(half - reach - 1, math.pi / 2), TRACK, CFG)
    assert is_out_of_bounds(car_beside_centerline(half - reach + 1, math.pi / 2), TRACK, CFG)


def test_full_throttle_from_start_leaves_track_once_and_keeps_going():
    # The track starts on the main straight, so flat out with no steering runs off at the first corner.
    car, progress = Car.at(TRACK.start_pose), Progress()
    passed, went_out, first_out = 0, 0, None
    for n in range(600):
        result = race_step(car, progress, 1, 0, CFG, TRACK)
        passed += result.checkpoints
        went_out += result.went_out
        if result.out_of_bounds and first_out is None:
            first_out = n
    assert first_out is not None
    assert passed >= 1
    assert went_out == 1  # counted once when leaving, not on every step spent off the road
    assert result.out_of_bounds
    assert car.speed == CFG.max_speed  # the road edge doesn't stop or bounce the car


def test_going_out_and_back_in_counts_each_exit():
    progress = Progress()
    (x1, y1), (x2, y2) = TRACK.centerline[0]
    length = math.dist((x1, y1), (x2, y2))
    nx, ny = -(y2 - y1) / length, (x2 - x1) / length
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    exits = 0  # offsets 0 and 60: fully on the road, and fully off it
    for offset in (0, 60, 60, 0, 60, 0):  # on, out, still out, back on, out again, back on
        car = Car(mx + nx * offset, my + ny * offset, 0)
        exits += race_step(car, progress, 0, 0, CFG, TRACK).went_out
    assert exits == 2
