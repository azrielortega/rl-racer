import math

import pytest

from sim.geometry import cast_rays, closest_point, min_segment_distances, point_segment_distance, segments_cross

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


def cast_ray(origin, angle, walls, max_dist):
    return float(cast_rays(origin, [angle], walls, max_dist)[0])


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


def test_several_rays_at_once():
    walls = [((100, -50), (100, 50)), ((-50, 60), (50, 60))]
    assert cast_rays((0, 0), [0, math.pi / 2, math.pi], walls, 300).tolist() == pytest.approx([100, 60, 300])


def test_min_segment_distances_match_the_scalar_version():
    segments = [WALL, ((0, 0), (0, 0)), ((200, -50), (200, 50))]
    points = [(50, 5), (103, 4), (3, 4), (190, 70), (-20, -20)]
    expected = [min(point_segment_distance(p, a, b) for a, b in segments) for p in points]
    assert min_segment_distances(points, segments).tolist() == pytest.approx(expected)
