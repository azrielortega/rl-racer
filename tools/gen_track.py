"""Generate a full track (walls, checkpoints, start poses) from a hand-authored centerline.

Usage: python tools/gen_track.py tracks/oval.centerline.json -o tracks/oval.json --svg oval.svg
"""

import argparse
import json
import math
import sys

DEFAULTS = {"segment_length": 10.0, "checkpoint_spacing": 100.0}


def catmull_rom_closed(points, samples_per_segment=20):
    """Sample a closed centripetal Catmull-Rom spline through the control points.

    Inputs:  points (list[(x, y)]) - control points in drive order; samples_per_segment (int) - density

    Outputs: list[(x, y)] - dense polyline around the loop (not repeating the first point)
    """
    n = len(points)
    out = []
    for i in range(n):
        p0, p1, p2, p3 = points[i - 1], points[i], points[(i + 1) % n], points[(i + 2) % n]
        # Centripetal parameterization (alpha = 0.5) avoids cusps and loops on uneven point spacing.
        t0 = 0.0
        t1 = t0 + math.dist(p0, p1) ** 0.5
        t2 = t1 + math.dist(p1, p2) ** 0.5
        t3 = t2 + math.dist(p2, p3) ** 0.5
        for k in range(samples_per_segment):
            t = t1 + (t2 - t1) * k / samples_per_segment
            a1 = _lerp(p0, p1, (t - t0) / (t1 - t0))
            a2 = _lerp(p1, p2, (t - t1) / (t2 - t1))
            a3 = _lerp(p2, p3, (t - t2) / (t3 - t2))
            b1 = _lerp(a1, a2, (t - t0) / (t2 - t0))
            b2 = _lerp(a2, a3, (t - t1) / (t3 - t1))
            out.append(_lerp(b1, b2, (t - t1) / (t2 - t1)))
    return out


def _lerp(a, b, u):
    return (a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u)


def resample_closed(points, step):
    """Resample a closed polyline into evenly spaced points by arc length.

    Inputs:  points (list[(x, y)]) - closed polyline; step (float) - target spacing

    Outputs: (list[(x, y)], float) - evenly spaced points and total loop length
    """
    loop = points + [points[0]]
    cum = [0.0]
    for a, b in zip(loop, loop[1:]):
        cum.append(cum[-1] + math.dist(a, b))
    total = cum[-1]
    count = max(3, round(total / step))
    out, j = [], 0
    for i in range(count):
        s = total * i / count
        while cum[j + 1] < s:
            j += 1
        seg = cum[j + 1] - cum[j]
        out.append(_lerp(loop[j], loop[j + 1], (s - cum[j]) / seg if seg else 0.0))
    return out, total


def unit_tangents(points):
    """Compute unit tangents of a closed polyline using central differences.

    Inputs:  points (list[(x, y)]) - evenly spaced closed polyline

    Outputs: list[(tx, ty)] - unit tangent at each point
    """
    n = len(points)
    out = []
    for i in range(n):
        dx = points[(i + 1) % n][0] - points[i - 1][0]
        dy = points[(i + 1) % n][1] - points[i - 1][1]
        d = math.hypot(dx, dy)
        out.append((dx / d, dy / d))
    return out


def min_turn_radius(points):
    """Find the tightest corner on a closed polyline via the circumradius of each point triple.

    Inputs:  points (list[(x, y)]) - evenly spaced closed polyline

    Outputs: (float, int) - smallest radius and the index where it occurs
    """
    best, best_i = math.inf, 0
    n = len(points)
    for i in range(n):
        a, b, c = points[i - 1], points[i], points[(i + 1) % n]
        cross = abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
        if cross > 1e-9:
            r = math.dist(a, b) * math.dist(b, c) * math.dist(a, c) / (2 * cross)
            if r < best:
                best, best_i = r, i
    return best, best_i


def _segments_cross(p, q, r, s):
    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    d1, d2 = orient(r, s, p), orient(r, s, q)
    d3, d4 = orient(p, q, r), orient(p, q, s)
    return d1 * d2 < 0 and d3 * d4 < 0


def find_wall_crossings(walls, per_wall):
    """Find pairs of wall segments that intersect (folded inner wall, walls touching, etc.).

    Inputs:  walls (list[((x, y), (x, y))]) - all wall segments; per_wall (int) - segments per wall loop

    Outputs: list[(x, y)] - approximate location of each crossing
    """
    hits = []
    for i in range(len(walls)):
        for j in range(i + 1, len(walls)):
            same_wall = i // per_wall == j // per_wall
            gap = (j - i) % per_wall
            if same_wall and gap in (1, per_wall - 1):
                continue
            if _segments_cross(*walls[i], *walls[j]):
                hits.append(walls[i][0])
    return hits


def generate(spec):
    """Build the full track dict from a centerline spec.

    Inputs:  spec (dict) - {"width", "centerline", optional "segment_length", "checkpoint_spacing"}

    Outputs: (dict, list[str]) - track JSON (walls, checkpoints, start_pose, centerline) and warnings
    """
    width = float(spec["width"])
    step = float(spec.get("segment_length", DEFAULTS["segment_length"]))
    spacing = float(spec.get("checkpoint_spacing", DEFAULTS["checkpoint_spacing"]))
    ctrl = [tuple(map(float, p)) for p in spec["centerline"]]
    if len(ctrl) < 3:
        raise ValueError("centerline needs at least 3 points")
    for a, b in zip(ctrl, ctrl[1:] + ctrl[:1]):
        if math.dist(a, b) < 1e-6:
            raise ValueError(f"duplicate consecutive centerline point {a}")

    center, length = resample_closed(catmull_rom_closed(ctrl), step)
    tangents = unit_tangents(center)
    half = width / 2
    left = [(x - ty * half, y + tx * half) for (x, y), (tx, ty) in zip(center, tangents)]
    right = [(x + ty * half, y - tx * half) for (x, y), (tx, ty) in zip(center, tangents)]

    n = len(center)
    walls = [(w[i], w[(i + 1) % n]) for w in (left, right) for i in range(n)]

    # Evenly spread checkpoints; the last one lands on index 0, so the finish line is the start line.
    cp_count = max(3, round(length / spacing))
    cp_idx = [round((k + 1) * n / cp_count) % n for k in range(cp_count)]
    checkpoints = [(left[i], right[i]) for i in cp_idx]

    (sx, sy), (tx, ty) = center[0], tangents[0]
    start_pose = {"x": sx, "y": sy, "heading": math.atan2(ty, tx)}

    warnings = []
    radius, r_idx = min_turn_radius(center)
    if radius <= half:
        warnings.append(
            f"corner radius {radius:.1f} at {_fmt(center[r_idx])} is <= width/2 ({half:.1f}): inner wall folds"
        )
    # A wall segment pointing against the centerline means the offset curve folded back on itself (cusp).
    backwards = [
        center[i]
        for w in (left, right)
        for i in range(n)
        if (w[(i + 1) % n][0] - w[i][0]) * tangents[i][0] + (w[(i + 1) % n][1] - w[i][1]) * tangents[i][1] <= 0
    ]
    if backwards:
        warnings.append(f"wall folds back near {_fmt(backwards[0])} ({len(backwards)} segment(s))")
    crossings = find_wall_crossings(walls, n)
    if crossings:
        spots = ", ".join(_fmt(p) for p in crossings[:5])
        warnings.append(f"{len(crossings)} wall crossing(s) near {spots}")

    track = {
        "width": width,
        "length": round(length, 2),
        "min_corner_radius": round(radius, 2),
        "walls": [[_pt(a), _pt(b)] for a, b in walls],
        "checkpoints": [[_pt(a), _pt(b)] for a, b in checkpoints],
        "start_pose": {k: round(v, 4) for k, v in start_pose.items()},
        "centerline": [_pt(p) for p in center],
    }
    return track, warnings


def _pt(p):
    return [round(p[0], 2), round(p[1], 2)]


def _fmt(p):
    return f"({p[0]:.0f}, {p[1]:.0f})"


def write_svg(track, ctrl, path):
    """Write an SVG preview: walls, checkpoints (finish in red), start pose and control points.

    Inputs:  track (dict) - output of generate(); ctrl (list) - control points; path (str) - output file

    Outputs: None - writes the SVG file
    """
    xs = [p[0] for seg in track["walls"] for p in seg]
    ys = [p[1] for seg in track["walls"] for p in seg]
    pad = 20
    x0, y0 = min(xs) - pad, min(ys) - pad
    w, h = max(xs) - x0 + pad, max(ys) - y0 + pad

    def line(a, b, color, width, extra=""):
        return f'<line x1="{a[0]}" y1="{a[1]}" x2="{b[0]}" y2="{b[1]}" stroke="{color}" stroke-width="{width}" {extra}/>'

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{x0} {y0} {w} {h}" width="{w}" height="{h}">',
             f'<rect x="{x0}" y="{y0}" width="{w}" height="{h}" fill="#2b2b2b"/>']
    cps = track["checkpoints"]
    for i, (a, b) in enumerate(cps):
        color = "#e33" if i == len(cps) - 1 else "#3a8"
        parts.append(line(a, b, color, 2))
    c = track["centerline"]
    pts = " ".join(f"{x},{y}" for x, y in c + c[:1])
    parts.append(f'<polyline points="{pts}" fill="none" stroke="#bbb" stroke-width="1" stroke-dasharray="8 8"/>')
    for a, b in track["walls"]:
        parts.append(line(a, b, "#fff", 2, 'stroke-linecap="round"'))
    for x, y in ctrl:
        parts.append(f'<circle cx="{x}" cy="{y}" r="4" fill="#fc3"/>')
    p = track["start_pose"]
    tip = (p["x"] + 20 * math.cos(p["heading"]), p["y"] + 20 * math.sin(p["heading"]))
    parts.append(f'<circle cx="{p["x"]}" cy="{p["y"]}" r="6" fill="#39f"/>')
    parts.append(line((p["x"], p["y"]), tip, "#39f", 3))
    parts.append("</svg>")
    with open(path, "w") as f:
        f.write("\n".join(parts))


def main():
    parser = argparse.ArgumentParser(description="Generate track walls/checkpoints/start pose from a centerline.")
    parser.add_argument("input", help="centerline JSON: {width, centerline, [segment_length], [checkpoint_spacing]}")
    parser.add_argument("-o", "--output", help="track JSON output path (default: stdout)")
    parser.add_argument("--svg", help="also write an SVG preview to this path")
    args = parser.parse_args()
    with open(args.input) as f:
        spec = json.load(f)
    track, warnings = generate(spec)

    text = json.dumps(track, indent=1)
    if args.output:
        with open(args.output, "w") as f:
            f.write(text + "\n")
    else:
        print(text)
    if args.svg:
        write_svg(track, spec["centerline"], args.svg)

    print(
        f"length {track['length']:.0f}, {len(track['walls'])} wall segments, "
        f"{len(track['checkpoints'])} checkpoints, min corner radius {track['min_corner_radius']:.1f}",
        file=sys.stderr,
    )
    for w in warnings:
        print(f"WARNING: {w}", file=sys.stderr)
    sys.exit(1 if warnings else 0)


if __name__ == "__main__":
    main()
