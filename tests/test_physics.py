import math

import pytest

from sim.config import Pose, load_config, load_track
from sim.physics import Car, step

CFG = load_config()


def seconds_until(car, throttle, done, limit=6000):
    for n in range(limit):
        if done(car):
            return n * CFG.dt
        step(car, throttle, 0, CFG)
    raise AssertionError("condition never reached")


def test_loads_track_from_config():
    track = load_track(CFG.track)
    assert len(track.walls) > 0
    assert len(track.checkpoints) > 2
    assert len(track.start_poses) == 2


def test_reaches_top_speed_in_under_two_seconds():
    car = Car(0, 0, 0)
    assert seconds_until(car, 1, lambda c: c.speed >= CFG.max_speed) == pytest.approx(1.87, abs=0.05)


def test_full_brake_stops_in_about_one_second():
    car = Car(0, 0, 0, CFG.max_speed)
    assert seconds_until(car, -1, lambda c: c.speed <= 0) == pytest.approx(0.95, abs=0.05)


def test_speed_is_clamped():
    car = Car(0, 0, 0)
    for _ in range(600):
        step(car, -1, 0, CFG)
    assert car.speed == -CFG.max_reverse


def test_stopped_car_cannot_spin():
    car = Car(0, 0, 0)
    for _ in range(60):
        step(car, 0, 1, CFG)
    assert car.heading == 0


def test_moves_along_heading_with_y_down():
    car = Car.at(Pose(0, 0, math.pi / 2))
    for _ in range(60):
        step(car, 1, 0, CFG)
    assert car.x == pytest.approx(0, abs=1e-9)
    assert car.y > 0


def test_turn_radius_at_top_speed():
    car = Car(0, 0, 0, CFG.max_speed)
    # Hold throttle so drag doesn't slow the car; one full circle should take 2π / TURN_RATE seconds.
    for _ in range(round(2 * math.pi / CFG.turn_rate / CFG.dt)):
        step(car, 1, 1, CFG)
    assert math.hypot(car.x, car.y) < 5
