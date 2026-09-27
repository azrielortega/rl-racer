import math

import numpy as np


def _cross(ax, ay, bx, by):
    return ax * by - ay * bx


def closest_point(p, a, b):
    """Find the point on segment a-b nearest to p.

    Inputs:  p, a, b ((x, y)) - query point and segment endpoints

    Outputs: (x, y) - nearest point on the segment (an endpoint if p lies beyond it)
    """
    ex, ey = b[0] - a[0], b[1] - a[1]
    length_sq = ex * ex + ey * ey
    if length_sq == 0:
        return a
    t = ((p[0] - a[0]) * ex + (p[1] - a[1]) * ey) / length_sq
    t = min(max(t, 0.0), 1.0)
    return (a[0] + t * ex, a[1] + t * ey)


def point_segment_distance(p, a, b):
    """Shortest distance from p to segment a-b.

    Inputs:  p, a, b ((x, y)) - query point and segment endpoints

    Outputs: float - distance
    """
    return math.dist(p, closest_point(p, a, b))


def segments_cross(p1, p2, q1, q2):
    """Check whether the movement p1->p2 crosses the line segment q1-q2 (e.g. a checkpoint).

    Inputs:  p1, p2 ((x, y)) - car position before and after a step; q1, q2 ((x, y)) - checkpoint ends

    Outputs: bool - True if the move crosses or ends exactly on the segment
    """
    qx, qy = q2[0] - q1[0], q2[1] - q1[1]
    side1 = _cross(qx, qy, p1[0] - q1[0], p1[1] - q1[1])
    side2 = _cross(qx, qy, p2[0] - q1[0], p2[1] - q1[1])
    # Half-open test (> 0 vs <= 0): a car that lands exactly on the line counts once, not again next step.
    if (side1 > 0) == (side2 > 0):
        return False
    px, py = p2[0] - p1[0], p2[1] - p1[1]
    end1 = _cross(px, py, q1[0] - p1[0], q1[1] - p1[1])
    end2 = _cross(px, py, q2[0] - p1[0], q2[1] - p1[1])
    return end1 * end2 <= 0


def min_segment_distances(points, segments):
    """Distance from each point to its nearest segment, all segments at once (vectorised point_segment_distance).

    Inputs:  points (array-like (P, 2)); segments (array-like (S, 2, 2)) - endpoint pairs

    Outputs: np.ndarray (P,) - shortest distance per point
    """
    # Flat x/y arrays of shape (P, S) instead of (P, S, 2): about 3x faster at this size.
    p = np.asarray(points, dtype=float)
    segments = np.asarray(segments, dtype=float).reshape(-1, 2, 2)
    ax, ay = segments[:, 0, 0], segments[:, 0, 1]
    ex, ey = segments[:, 1, 0] - ax, segments[:, 1, 1] - ay
    length_sq = ex * ex + ey * ey
    dx, dy = p[:, 0:1] - ax, p[:, 1:2] - ay
    # Zero-length segments: e is 0, so the closest point is `a` whatever t is.
    t = np.clip((dx * ex + dy * ey) / np.where(length_sq == 0, 1, length_sq), 0, 1)
    rx, ry = dx - t * ex, dy - t * ey
    return np.sqrt((rx * rx + ry * ry).min(axis=1))


def cast_rays(origin, angles, walls, max_dist):
    """Distance along each ray to the nearest wall, capped at max_dist (the distance sensors, spec 4).

    Inputs:  origin ((x, y)); angles (array-like) - world-space radians; walls (array-like (W, 2, 2)); max_dist (float)

    Outputs: np.ndarray - one distance per angle, max_dist where no wall is closer
    """
    walls = np.asarray(walls, dtype=float).reshape(-1, 2, 2)
    a = walls[:, 0] - origin  # (W, 2), wall start relative to the ray origin
    e = walls[:, 1] - walls[:, 0]  # (W, 2)
    dx, dy = np.cos(angles)[:, None], np.sin(angles)[:, None]  # (R, 1)
    denom = dx * e[:, 1] - dy * e[:, 0]  # (R, W); 0 means the ray is parallel to the wall
    safe = np.where(denom == 0, 1, denom)
    t = (a[:, 0] * e[:, 1] - a[:, 1] * e[:, 0]) / safe  # distance along the ray
    u = (a[:, 0] * dy - a[:, 1] * dx) / safe  # position along the wall, 0..1
    hit = (denom != 0) & (t >= 0) & (u >= 0) & (u <= 1)
    return np.minimum(np.where(hit, t, max_dist).min(axis=1), max_dist)
