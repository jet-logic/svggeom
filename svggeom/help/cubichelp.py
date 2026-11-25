"""Cubic bezier helpers.

All coordinates are plain `Tuple[float, float]`; boxes are `((xmin, xmax), (ymin, ymax))`.
"""

from typing import Sequence, Tuple
from math import fabs, sqrt, hypot

Pt = Sequence[float]
Box = Tuple[Pt, Pt]


def cubic_extrema(py0: float, py1: float, py2: float, py3: float) -> Pt:
    """Returns the extreme value, given a set of bezier coordinates."""
    atol = 1e-9
    cmin, cmax = min(py0, py3), max(py0, py3)
    pd1 = py1 - py0
    pd2 = py2 - py1
    pd3 = py3 - py2

    def _is_bigger(point):
        if 0 < point < 1:
            pyx = (
                py0 * (1 - point) * (1 - point) * (1 - point)
                + 3 * py1 * point * (1 - point) * (1 - point)
                + 3 * py2 * point * point * (1 - point)
                + py3 * point * point * point
            )
            return min(cmin, pyx), max(cmax, pyx)
        return cmin, cmax

    if fabs(pd1 - 2 * pd2 + pd3) > atol:
        if pd2 * pd2 > pd1 * pd3:
            pds = sqrt(pd2 * pd2 - pd1 * pd3)
            cmin, cmax = _is_bigger((pd1 - pd2 + pds) / (pd1 - 2 * pd2 + pd3))
            cmin, cmax = _is_bigger((pd1 - pd2 - pds) / (pd1 - 2 * pd2 + pd3))
    elif fabs(pd2 - pd1) > atol:
        cmin, cmax = _is_bigger(-pd1 / (2 * (pd2 - pd1)))

    return cmin, cmax


def split_at_scalar(z: float, from_val: float, a: float, b: float, to: float) -> Tuple[
    Tuple[float, float, float, float],
    Tuple[float, float, float, float],
]:
    t = (
        z**3 * to
        - 3 * z**2 * (z - 1) * b
        + 3 * z * (z - 1) ** 2 * a
        - (z - 1) ** 3 * from_val
    )
    part1 = (
        from_val,
        z * a - (z - 1) * from_val,
        z**2 * b - 2 * z * (z - 1) * a + (z - 1) ** 2 * from_val,
        t,
    )
    part2 = (
        t,
        z**2 * to - 2 * z * (z - 1) * b + (z - 1) ** 2 * a,
        z * to - (z - 1) * b,
        to,
    )
    return part1, part2


def cubic_box(points: Sequence[Pt]) -> Box:
    """Bounding box as ((xmin, xmax), (ymin, ymax))."""
    (sx, sy), (x1, y1), (x2, y2), (ex, ey) = points
    x_extrema = cubic_extrema(sx, x1, x2, ex)
    y_extrema = cubic_extrema(sy, y1, y2, ey)
    return (x_extrema, y_extrema)


def cubic_flatness(points: Sequence[Pt]) -> float:
    (sx, sy), (x1, y1), (x2, y2), (ex, ey) = points
    ux = (3 * x1 - 2 * sx - ex) ** 2
    uy = (3 * y1 - 2 * sy - ey) ** 2
    vx = (3 * x2 - 2 * ex - sx) ** 2
    vy = (3 * y2 - 2 * ey - sy) ** 2
    if ux < vx:
        ux = vx
    if uy < vy:
        uy = vy
    return ux + uy


def cubic_point_at(points: Sequence[Pt], t: float) -> Pt:
    (sx, sy), (x1, y1), (x2, y2), (ex, ey) = points
    F = 1 - t
    x = F**3 * sx + 3 * F**2 * t * x1 + 3 * F * t**2 * x2 + t**3 * ex
    y = F**3 * sy + 3 * F**2 * t * y1 + 3 * F * t**2 * y2 + t**3 * ey
    return x, y


def cubic_split_at(points: Sequence[Pt], z: float):
    (sx, sy), (x1, y1), (x2, y2), (ex, ey) = points
    x_parts = split_at_scalar(z, sx, x1, x2, ex)
    y_parts = split_at_scalar(z, sy, y1, y2, ey)
    curve1 = [(x_parts[0][i], y_parts[0][i]) for i in range(4)]
    curve2 = [(x_parts[1][i], y_parts[1][i]) for i in range(4)]
    return [curve1, curve2]


# --- local tuple arithmetic (replaces Vector ops) --------------------------


def _sub(a: Pt, b: Pt) -> Pt:
    return (a[0] - b[0], a[1] - b[1])


def _add(a: Pt, b: Pt) -> Pt:
    return (a[0] + b[0], a[1] + b[1])


def _mul(a: Pt, s: float) -> Pt:
    return (a[0] * s, a[1] * s)


def _close(a: Pt, b: Pt, eps: float = 1e-12) -> bool:
    # Mirrors Vector.close_to(p): True iff |delta x| < eps and |delta y| < eps.
    return abs(a[0] - b[0]) < eps and abs(a[1] - b[1]) < eps


def cubic_slope_at(points: Sequence[Pt], t: float) -> Pt:
    from_pt, c1, c2, to = points
    if t <= 0:
        if _close(from_pt, c1):
            return _sub(c2, from_pt)
        return _sub(c1, from_pt)
    elif t >= 1:
        return _sub(to, c2)

    if _close(from_pt, c1):
        if _close(to, c2):
            return _sub(to, from_pt)
        # original had a dead `if t <= 0` branch here (t > 0 already true);
        # folded into the sole remaining arm for parity.
        a = _mul(_sub(c2, from_pt), 2 * (1 - t))
        b = _mul(_sub(to, c2), t)
        return _add(a, b)
    elif _close(to, c2):
        a = _mul(_sub(c1, from_pt), 2 * (1 - t))
        b = _mul(_sub(to, c1), t)
        return _add(a, b)
    else:
        a = _mul(_sub(c1, from_pt), 3 * (1 - t) ** 2)
        b = _mul(_sub(c2, c1), 6 * (1 - t) * t)
        c = _mul(_sub(to, c2), 3 * t**2)
        return _add(_add(a, b), c)


def cubic_length(points: Sequence[Pt]) -> float:
    if cubic_flatness(points) > 0.15:
        a, b = cubic_split_at(points, 0.5)
        return cubic_length(a) + cubic_length(b)
    else:
        (sx, sy), _, _, (ex, ey) = points
        return hypot(ex - sx, ey - sy)
