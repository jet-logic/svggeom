from math import isclose
from typing import Optional, Sequence
from .matrix import Matrix
from .help.cubichelp import cubic_extrema
from .help.archelp import arc_params, arc_to_curve

FF = Sequence[float]


def _as_ff(p: Sequence[float]) -> FF:
    return (float(p[0]), float(p[1]))


def _sub(a: Sequence[float], b: Sequence[float]) -> FF:
    return (a[0] - b[0], a[1] - b[1])


# i >>> v >>> o
class Bezier(list[tuple[FF, FF, FF]]):

    def close(self):
        if not self:
            raise ValueError("close requires a non-empty Bezier")
        self.line_to(self[0][0])

    def is_closed(self, epsilon=1e-9) -> bool:
        if len(self) < 2:
            return False
        first_v = self[0][0]
        last_v = self[-1][0]
        return isclose(first_v[0], last_v[0], abs_tol=epsilon) and isclose(
            first_v[1], last_v[1], abs_tol=epsilon
        )

    def insert_point(
        self, index: Optional[int], v: FF = (0, 0), i: FF = (0, 0), o: FF = (0, 0)
    ):
        if index is None:
            self.append((v, i, o))
        else:
            self.insert(index, (v, i, o))
        return self

    def add_point(self, v: FF = (0, 0), i: FF = (0, 0), o: FF = (0, 0)):
        return self.insert_point(None, v, i, o)

    def add_smooth_point(self, i: FF = (0, 0)):
        return self.add_point(i, (i[0], i[1]), (-i[0], -i[1]))

    def bounding_box(self):
        """Return ((xmin, xmax), (ymin, ymax)) or None if empty."""
        if not self:
            return None
        if len(self) == 1:
            x, y = self[0][0]
            return ((x, x), (y, y))

        xmin = ymin = float("inf")
        xmax = ymax = float("-inf")

        def pairs():
            a = None
            for b in self:
                if a is not None:
                    yield a, b
                a = b

        for a, b in pairs():
            x4, y4 = b[0]
            x1, y1 = a[0]
            # recover absolute controls (same convention as curve_to / transform)
            x3, y3 = x4 - b[1][0], y4 - b[1][1]
            x2, y2 = x1 - a[2][0], y1 - a[2][1]

            cx0, cx1 = cubic_extrema(x1, x2, x3, x4)
            cy0, cy1 = cubic_extrema(y1, y2, y3, y4)
            xmin = min(xmin, cx0, cx1)
            xmax = max(xmax, cx0, cx1)
            ymin = min(ymin, cy0, cy1)
            ymax = max(ymax, cy0, cy1)

        if xmin > xmax or ymin > ymax:
            return None
        return ((xmin, xmax), (ymin, ymax))

    def move_to(self, p: FF = (0, 0)):
        pts = self
        if len(pts) > 1:
            raise RuntimeError(f"Moved already: {pts!r}")
        elif len(pts) == 1:
            pts[0] = (p, (0, 0), (0, 0))
        else:
            assert len(pts) == 0
            self.insert_point(None, v=p)
        return self

    def line_to(self, p: FF):
        return self.add_point(v=p)

    def curve_to(self, c1: Sequence[float], c2: Sequence[float], p2: Sequence[float]):
        if not self:
            raise ValueError("curve_to requires a preceding move_to()")
        pts = self
        pv, pi, po = pts[-1]
        pts[-1] = (pv, pi, _sub(pv, c1))
        return self.add_point(v=_as_ff(p2), i=_sub(p2, c2))

    def arc_to(
        self,
        rx: float,
        ry: float,
        rotation: float,
        large_arc_flag: bool,
        sweep_flag: bool,
        p2: FF,
    ):
        if not self:
            raise ValueError("requires a preceding move_to()")
        x1, y1 = self[-1][0]
        x2, y2 = p2

        phi, rx, ry, sinP, cosP, cx, cy, rtheta, rdelta = arc_params(
            x1, y1, rx, ry, rotation, large_arc_flag, sweep_flag, x2, y2
        )
        for p1, c1, c2, p2 in arc_to_curve(rx, ry, cx, cy, sinP, cosP, rtheta, rdelta):
            self.curve_to(c1, c2, p2)
        return self

    def quad_to(self, c2: FF, p2: FF):
        if not self:
            raise ValueError("requires a preceding move_to()")
        p1 = self[-1][0]
        x1 = 1.0 / 3 * p1[0] + 2.0 / 3 * c2[0]
        x2 = 2.0 / 3 * c2[0] + 1.0 / 3 * p2[0]
        y1 = 1.0 / 3 * p1[1] + 2.0 / 3 * c2[1]
        y2 = 2.0 / 3 * c2[1] + 1.0 / 3 * p2[1]
        return self.curve_to((x1, y1), (x2, y2), p2)

    def transform(self, tm: "Matrix"):
        pts: list[tuple[FF, FF, FF]] = []
        for v_pt, i_tan, o_tan in self:
            # recover absolute control points (vertex - tangent)
            c_in = _sub(v_pt, i_tan)
            c_out = _sub(v_pt, o_tan)
            v_t = _as_ff(tm.apply_to_point(v_pt))
            c_in_t = _as_ff(tm.apply_to_point(c_in))
            c_out_t = _as_ff(tm.apply_to_point(c_out))
            # re-encode as tangents relative to the transformed vertex
            pts.append((v_t, _sub(v_t, c_in_t), _sub(v_t, c_out_t)))

        self[:] = pts
        return self

    def reverse_dir(self):
        """
        Reverse the direction of the bezier curve.
        The shape remains exactly the same, only direction changes.

        Returns:
            Self for chaining
        """
        if len(self) == 0:
            return self

        # Reverse the points list
        reversed_points = []

        for i in range(len(self) - 1, -1, -1):
            vertex, in_tangent, out_tangent = self[i]

            # Swap in and out tangents when reversing
            reversed_points.append((vertex, out_tangent, in_tangent))
        self[:] = reversed_points
        return self
