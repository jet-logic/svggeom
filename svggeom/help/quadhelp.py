"""Quadratic bezier helpers.

Self-contained: no `..vector` / `..bbox` imports. All coordinates are
plain `Tuple[float, float]`; boxes are `((xmin, xmax), (ymin, ymax))`.
"""

from math import fabs, hypot, log, sqrt
from typing import Sequence, Tuple

Pt = Sequence[float]
Box = Tuple[Pt, Pt]


def quadratic_extrema(py0: float, py1: float, py2: float) -> Pt:
    """Returns the extreme value, given a set of quadratic bezier coordinates."""
    atol = 1e-9
    cmin, cmax = min(py0, py2), max(py0, py2)

    def _is_bigger(point):
        if 0 < point < 1:
            pyx = (
                py0 * (1 - point) * (1 - point)
                + 2 * py1 * point * (1 - point)
                + py2 * point * point
            )
            return min(cmin, pyx), max(cmax, pyx)
        return cmin, cmax

    if fabs(py0 + py2 - 2 * py1) > atol:
        cmin, cmax = _is_bigger((py0 - py1) / (py0 + py2 - 2 * py1))

    return cmin, cmax


def quad_length(points: Sequence[Pt], t: float = 1.0) -> float:
    (x0, y0), (x1, y1), (x2, y2) = points

    ax = x0 - 2 * x1 + x2
    ay = y0 - 2 * y1 + y2
    bx = 2 * x1 - 2 * x0
    by = 2 * y1 - 2 * y0

    A = 4 * (ax * ax + ay * ay)
    B = 4 * (ax * bx + ay * by)
    C = bx * bx + by * by

    if A == 0:
        return t * hypot(x2 - x0, y2 - y0)

    b = B / (2 * A)
    c = C / A
    u = t + b
    k = c - b * b

    uuk = sqrt(u * u + k) if u * u + k > 0 else 0.0
    bbk = sqrt(b * b + k) if b * b + k > 0 else 0.0

    denominator = b + bbk
    if abs(denominator) > 1e-12:
        term = k * log(abs((u + uuk) / denominator))
    else:
        term = 0.0

    return (sqrt(A) / 2) * (u * uuk - b * bbk + term)


def quad_split_at(points: Sequence[Pt], t: float):
    x1, y1 = points[0]
    cx, cy = points[1]
    x2, y2 = points[2]

    mx1 = (1 - t) * x1 + t * cx
    mx2 = (1 - t) * cx + t * x2
    mxt = (1 - t) * mx1 + t * mx2

    my1 = (1 - t) * y1 + t * cy
    my2 = (1 - t) * cy + t * y2
    myt = (1 - t) * my1 + t * my2

    return [
        [(x1, y1), (mx1, my1), (mxt, myt)],
        [(mxt, myt), (mx2, my2), (x2, y2)],
    ]


def quad_point_at(points: Sequence[Pt], t: float) -> Pt:
    x1, y1 = points[0]
    cx, cy = points[1]
    x2, y2 = points[2]

    v = 1 - t
    x = v * v * x1 + 2 * v * t * cx + t * t * x2
    y = v * v * y1 + 2 * v * t * cy + t * t * y2
    return (x, y)


# --- local tuple arithmetic (replaces Vector ops) --------------------------


def _sub(a: Pt, b: Pt) -> Pt:
    return (a[0] - b[0], a[1] - b[1])


def _add(a: Pt, b: Pt) -> Pt:
    return (a[0] + b[0], a[1] + b[1])


def _mul(a: Pt, s: float) -> Pt:
    return (a[0] * s, a[1] * s)


def _div(a: Pt, s: float) -> Pt:
    return (a[0] / s, a[1] / s)


def quad_slope_at(points: Sequence[Pt], t: float) -> Pt:
    from_pt, c, to = points

    if c == from_pt or c == to:
        d = _sub(to, from_pt)
        length = hypot(d[0], d[1])
        if length == 0:
            return (0, 0)
        return _div(d, length)

    if t >= 1:
        return _sub(to, c)
    elif t <= 0:
        return _sub(c, from_pt)

    a = _mul(_sub(c, from_pt), 1 - t)
    b = _mul(_sub(to, c), t)
    return _mul(_add(a, b), 2)


def quad_bbox(points: Sequence[Pt]) -> Box:
    """Bounding box as ((xmin, xmax), (ymin, ymax))."""
    x1, y1 = points[0]
    x2, y2 = points[1]
    x3, y3 = points[2]

    x_extrema = quadratic_extrema(x1, x2, x3)
    y_extrema = quadratic_extrema(y1, y2, y3)
    return (x_extrema, y_extrema)
