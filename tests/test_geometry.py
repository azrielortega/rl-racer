import math

import pytest

from sim.geometry import cast_ray, closest_point, point_segment_distance, segments_cross

WALL = ((0, 0), (100, 0))


def test_point_distance_to_wall_middle():
    assert point_segment_distance((50, 5), *WALL) == pytest.approx(5)


def test_point_past_wall_end_measures_to_endpoint():
    assert point_segment_distance((103, 4), *WALL) == pytest.approx(5)
    assert closest_point((103, 4), *WALL) == (100, 0)


def test_zero_length_segment():
    assert point_segment_distance((3, 4), (0, 0), (0, 0)) == pytest.approx(5)


def test_move_crossing_checkpoint():
    checkpoint = ((0, -10), (0, 10))
    assert segments_cross((-5, 0), (5, 0), *checkpoint)
    assert segments_cross((5, 0), (-5, 0), *checkpoint)  # direction doesn't matter; race logic handles order


def test_move_missing_checkpoint():
    checkpoint = ((0, -10), (0, 10))
    assert not segments_cross((-5, 20), (5, 20), *checkpoint)  # passes beyond its end
    assert not segments_cross((-5, 0), (-1, 0), *checkpoint)  # stops short
    assert not segments_cross((-5, 0), (-5, 5), *checkpoint)  # parallel


def test_landing_on_checkpoint_counts_exactly_once():
    checkpoint = ((0, -10), (0, 10))
    assert segments_cross((-5, 0), (0, 0), *checkpoint)  # step ends on the line
    assert not segments_cross((0, 0), (5, 0), *checkpoint)  # next step starts on it


def test_ray_straight_at_wall_100_away():
    wall = ((100, -50), (100, 50))
    assert cast_ray((0, 0), 0, [wall], 300) == pytest.approx(100)


def test_ray_at_angle():
    wall = ((100, -500), (100, 500))
    assert cast_ray((0, 0), math.radians(60), [wall], 300) == pytest.approx(200)


def test_ray_misses_or_is_capped():
    wall = ((100, -50), (100, 50))
    assert cast_ray((0, 0), math.pi, [wall], 300) == 300  # pointing away
    assert cast_ray((0, 0), math.pi / 2, [wall], 300) == 300  # parallel
    assert cast_ray((0, 0), 0, [((400, -50), (400, 50))], 300) == 300  # beyond max range


def test_ray_returns_nearest_wall():
    walls = [((200, -50), (200, 50)), ((80, -50), (80, 50)), ((150, -50), (150, 50))]
    assert cast_ray((0, 0), 0, walls, 300) == pytest.approx(80)
