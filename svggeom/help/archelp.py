from math import ceil, fabs, pi, sin, cos, acos, sqrt, tan, radians


def approximate_unit_arc(theta1: float, delta_theta: float):
    alpha = (4 / 3) * tan(delta_theta / 4)

    x1 = cos(theta1)
    y1 = sin(theta1)
    x2 = cos(theta1 + delta_theta)
    y2 = sin(theta1 + delta_theta)

    return [
        x1,
        y1,
        x1 - y1 * alpha,
        y1 + x1 * alpha,
        x2 + y2 * alpha,
        y2 - x2 * alpha,
        x2,
        y2,
    ]


def arc_to_curve(
    rx: float,
    ry: float,
    cx: float,
    cy: float,
    sin_phi: float,
    cos_phi: float,
    theta1: float,
    delta_theta: float,
):
    result: list[list[float]] = []

    # Split an arc to multiple segments, so each segment
    # will be less than TAU/4 (= 90deg)
    segments = max(ceil(abs(delta_theta) / (2 * pi / 12)), 1)
    delta_theta /= segments

    for i in range(segments):
        result.append(approximate_unit_arc(theta1, delta_theta))
        theta1 += delta_theta

    # We have a bezier approximation of a unit circle,
    # now need to transform back to the original ellipse
    # return [
    #     transform_curve(curve, rx, ry, cx, cy, sin_phi, cos_phi) for curve in result
    # ]
    for curve in result:
        assert 8 == len(curve)
        for i in range(0, len(curve), 2):
            x = curve[i]
            y = curve[i + 1]

            # scale
            x *= rx
            y *= ry

            # rotate
            xp = cos_phi * x - sin_phi * y
            yp = sin_phi * x + cos_phi * y

            # translate
            curve[i] = xp + cx
            curve[i + 1] = yp + cy
        yield (curve[0:2], curve[2:4], curve[4:6], curve[6:8])


def transform_curve(curve: list[float], rx, ry, cx, cy, sin_phi, cos_phi):
    for i in range(0, len(curve), 2):
        x = curve[i]
        y = curve[i + 1]

        # scale
        x *= rx
        y *= ry

        # rotate
        xp = cos_phi * x - sin_phi * y
        yp = sin_phi * x + cos_phi * y

        # translate
        curve[i] = xp + cx
        curve[i + 1] = yp + cy

    return curve


TAU = pi * 2


def unit_vector_angle(ux: float, uy: float, vx: float, vy: float):
    sign = -1 if ux * vy - uy * vx < 0 else 1
    dot = ux * vx + uy * vy

    if dot > 1.0:
        dot = 1.0
    elif dot < -1.0:
        dot = -1.0
    return sign * acos(dot)


def cossin(a: float):
    return cos(a), sin(a)


def arc_params(
    x1: float,
    y1: float,
    rx: float,
    ry: float,
    phi: float,
    arc: float,
    sweep: float,
    x2: float,
    y2: float,
):
    # SVG path arc rotation (x-axis-rotation) is expressed in degrees.
    phi = radians(phi)
    cosP, sinP = cossin(phi)

    x1_ = (cosP * (x1 - x2) + sinP * (y1 - y2)) / 2
    y1_ = (-sinP * (x1 - x2) + cosP * (y1 - y2)) / 2

    if True:
        if not rx or not ry:
            raise ValueError("Not an ellipse")
        else:
            rx = fabs(rx)
            ry = fabs(ry)

    rxSq = rx**2
    rySq = ry**2
    y1_Sq = y1_**2
    x1_Sq = x1_**2

    if True:
        L = x1_**2 / rxSq + y1_**2 / rySq
        if L > 1:
            m = sqrt(L)
            rx *= m
            ry *= m

    s1 = rxSq * y1_Sq
    s2 = rySq * x1_Sq
    v1 = (rxSq * rySq - s1 - s2) / (s1 + s2)
    m1 = 0 if v1 <= 0 else sqrt(v1) * (-1 if arc == sweep else 1)
    cx_ = (rx * y1_ * m1) / ry
    cy_ = (-ry * x1_ * m1) / rx

    cx = cosP * cx_ - sinP * cy_ + (x1 + x2) / 2
    cy = sinP * cx_ + cosP * cy_ + (y1 + y2) / 2
    v1x = (x1_ - cx_) / rx
    v1y = (y1_ - cy_) / ry
    v2x = (-x1_ - cx_) / rx
    v2y = (-y1_ - cy_) / ry

    theta1 = unit_vector_angle(1, 0, v1x, v1y)
    delta_theta = unit_vector_angle(v1x, v1y, v2x, v2y)

    return (
        phi,
        rx,
        ry,
        sinP,
        cosP,
        cx,
        cy,
        theta1,
        delta_theta
        + (
            TAU
            if sweep and delta_theta < 0
            else 0 if sweep else -TAU if delta_theta > 0 else 0
        ),
    )


LENGTH_MIN_DEPTH = 17
LENGTH_ERROR = 1e-12


def _dist(ax, ay, bx, by):
    return sqrt((bx - ax) ** 2 + (by - ay) ** 2)


def arc_length(
    curve,  # anything with .point_at(t) -> (x, y) or Vector
    start: float,
    end: float,
    start_point,
    end_point,
    error: float = LENGTH_ERROR,
    min_depth: int = LENGTH_MIN_DEPTH,
    depth: int = 0,
) -> float:
    mid = (start + end) / 2
    mid_point = curve.point_at(mid)

    length = _dist(*start_point, *end_point)
    first_half = _dist(*start_point, *mid_point)
    second_half = _dist(*mid_point, *end_point)
    length2 = first_half + second_half

    if (length2 - length > error) or (depth < min_depth):
        depth += 1
        return arc_length(
            curve, start, mid, start_point, mid_point, error, min_depth, depth
        ) + arc_length(curve, mid, end, mid_point, end_point, error, min_depth, depth)
    return length2
