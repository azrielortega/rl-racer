import math

import pytest

from agent.sensors import OBSERVATION_SIZE, RAY_OFFSETS, angle_to, observe, ray_distances
from sim.config import load_config, load_track
from sim.physics import Car
from sim.race import Progress, race_step

CFG = load_config()
TRACK = load_track(CFG.track)
LEFT, AHEAD, RIGHT = 0, 4, 8  # indexes of the -90, 0 and +90 degree rays


def start_car():
    return Car.at(TRACK.start_poses[0])


def test_observation_shape_and_ranges():
    obs = observe(start_car(), Progress(), TRACK, CFG)
    assert len(obs) == OBSERVATION_SIZE == 12
    assert all(-1 <= v <= 1 for v in obs)


def test_side_rays_span_the_road_width_on_a_straight():
    # Monza starts on the main straight, so straight left + straight right = road width.
    rays = ray_distances(start_car(), TRACK, CFG)
    assert rays[LEFT] + rays[RIGHT] == pytest.approx(TRACK.width, abs=0.5)


def test_ray_ahead_is_capped_on_a_long_straight():
    assert observe(start_car(), Progress(), TRACK, CFG)[AHEAD] == 1.0


def test_rays_turn_with_the_car():
    car = start_car()
    before = ray_distances(car, TRACK, CFG)
    car.heading += math.pi  # facing backwards: left and right swap
    after = ray_distances(car, TRACK, CFG)
    assert after[LEFT] == pytest.approx(before[RIGHT], abs=0.5)
    assert after[RIGHT] == pytest.approx(before[LEFT], abs=0.5)


def test_speed_is_normalised():
    car = start_car()
    car.speed = CFG.max_speed / 2
    assert observe(car, Progress(), TRACK, CFG)[9] == pytest.approx(0.5)
    car.speed = -CFG.max_reverse
    assert observe(car, Progress(), TRACK, CFG)[9] == pytest.approx(-CFG.max_reverse / CFG.max_speed)


def test_angle_to_point():
    car = Car(0, 0, 0)
    assert angle_to(car, (100, 0)) == pytest.approx(0)
    assert angle_to(car, (0, 100)) == pytest.approx(math.pi / 2)  # below on screen = to the right
    assert angle_to(car, (0, -100)) == pytest.approx(-math.pi / 2)
    assert abs(angle_to(car, (-100, 1e-9))) == pytest.approx(math.pi)


def test_next_checkpoint_is_ahead_at_start_and_behind_when_turned_around():
    car = start_car()
    assert abs(observe(car, Progress(), TRACK, CFG)[10]) < 0.1
    car.heading += math.pi
    assert abs(observe(car, Progress(), TRACK, CFG)[10]) > 0.9


def test_angle_follows_the_next_checkpoint():
    progress = Progress(next_checkpoint=5)
    (x1, y1), (x2, y2) = TRACK.checkpoints[5]
    car = start_car()
    expected = angle_to(car, ((x1 + x2) / 2, (y1 + y2) / 2)) / math.pi
    assert observe(car, progress, TRACK, CFG)[10] == pytest.approx(expected)


def test_out_of_bounds_flag():
    car, progress = start_car(), Progress()
    assert observe(car, progress, TRACK, CFG)[11] == 0.0
    car.y += TRACK.width  # sideways off the start straight
    race_step(car, progress, 0, 0, CFG, TRACK)
    assert observe(car, progress, TRACK, CFG)[11] == 1.0


def test_nine_rays_from_minus_to_plus_ninety():
    assert [round(math.degrees(o), 1) for o in RAY_OFFSETS] == [-90, -67.5, -45, -22.5, 0, 22.5, 45, 67.5, 90]
