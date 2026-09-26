import math


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


def cast_ray(origin, angle, walls, max_dist):
    """Distance along a ray to the nearest wall, capped at max_dist (one distance sensor, spec 4).

    Inputs:  origin ((x, y)); angle (float) - world-space radians; walls (list[((x, y), (x, y))]); max_dist (float)

    Outputs: float - distance to the first wall hit, or max_dist if none is closer
    """
    ox, oy = origin
    dx, dy = math.cos(angle), math.sin(angle)
    best = max_dist
    for a, b in walls:
        ex, ey = b[0] - a[0], b[1] - a[1]
        denom = _cross(dx, dy, ex, ey)
        if denom == 0:
            continue  # parallel to the wall
        ax, ay = a[0] - ox, a[1] - oy
        t = _cross(ax, ay, ex, ey) / denom  # distance along the ray
        u = _cross(ax, ay, dx, dy) / denom  # position along the wall, 0..1
        if 0 <= t < best and 0 <= u <= 1:
            best = t
    return best
