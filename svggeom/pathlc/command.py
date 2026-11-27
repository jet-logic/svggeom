from math import acos, atan2, cos, hypot, isclose, pi, sin, sqrt, tan, tau
from typing import Any, Optional, Sequence, Tuple
from ..help.archelp import arc_params, arc_to_curve
from ..help.cubichelp import (
    cubic_extrema,
    cubic_length,
    cubic_point_at,
    cubic_slope_at,
    cubic_split_at,
)
from ..help.quadhelp import (
    quad_length,
    quad_point_at,
    quad_slope_at,
    quad_split_at,
    quadratic_extrema,
)

# Plain 2-D point arithmetic, replacing the dependency on ..vector.
# Mirrors the local tuple helpers already used in help/cubichelp.py & quadhelp.py.
Pt = Sequence[float]
FF = Tuple[float, float]
BBox = tuple[tuple[float, float], tuple[float, float]]


def _pt2(p: Pt) -> FF:
    """Normalize any point-like sequence to a plain ``(x, y)`` tuple."""
    return (float(p[0]), float(p[1]))


def _add(a: Pt, b: Pt) -> FF:
    return (a[0] + b[0], a[1] + b[1])


def _sub(a: Pt, b: Pt) -> FF:
    return (a[0] - b[0], a[1] - b[1])


def _mul(a: Pt, s: float) -> FF:
    return (a[0] * s, a[1] * s)


def _div(a: Pt, s: float) -> FF:
    return (a[0] / s, a[1] / s)


def _len(a: Pt) -> float:
    return hypot(a[0], a[1])


def _close(a: Pt, b: Pt, eps: float = 1e-12) -> bool:
    """Equivalent of ``Vector.close_to``."""
    return abs(a[0] - b[0]) < eps and abs(a[1] - b[1]) < eps


def _reflect(p: Pt, at: Pt) -> FF:
    """Equivalent of ``Vector.reflect_at``: mirror ``p`` around point ``at``."""
    return _add(at, _sub(at, p))


def _transform(m: Sequence[float], p: Pt) -> FF:
    """Equivalent of ``Vector.transform``: apply an SVG hexad matrix to a point."""
    a, b, c, d, e, f = m
    return (a * p[0] + c * p[1] + e, b * p[0] + d * p[1] + f)


def _with_x(p: Pt, n: float) -> FF:
    return (n, p[1])


def _with_y(p: Pt, n: float) -> FF:
    return (p[0], n)


def _shift_x(p: Pt, d: float) -> FF:
    return (p[0] + d, p[1])


def _shift_y(p: Pt, d: float) -> FF:
    return (p[0], p[1] + d)


def check_t(t: float):
    # assert t >= 0 and t <= 1, f"t={t!r}"
    if t > 1:
        return 1
    elif t < 0:
        return 0
    return t


def arc_tangent_to(
    cur: "BaseLC | None",
    p1: Sequence[float],
    p2: Sequence[float],
    r: float,
    epsilon=1e-12,
):
    x1, y1 = p1
    x2, y2 = p2
    x0, y0 = cur._to if cur else (0, 0)
    x21 = x2 - x1
    y21 = y2 - y1
    x01 = x0 - x1
    y01 = y0 - y1
    l01_2 = x01 * x01 + y01 * y01

    # Is the radius negative? Error.
    if r < 0:
        raise ValueError("negative radius: " + str(r))
    elif not cur:
        # Is this path empty? Move to (x1,y1).
        cur = MoveLC(None, (x1, y1))
    elif not (l01_2 > epsilon):
        # Or, is (x1,y1) coincident with (x0,y0)? Do nothing.
        pass
    elif not (abs(y01 * x21 - y21 * x01) > epsilon) or not r:
        # Or, are (x0,y0), (x1,y1) and (x2,y2) collinear?
        # Equivalently, is (x1,y1) coincident with (x2,y2)?
        # Or, is the radius zero? Line to (x1,y1).
        cur = cur.line_to((x1, y1))
    else:

        # Otherwise, draw an arc!
        x20 = x2 - x0
        y20 = y2 - y0
        l21_2 = x21 * x21 + y21 * y21
        l20_2 = x20 * x20 + y20 * y20
        l21 = sqrt(l21_2)
        l01 = sqrt(l01_2)
        l = r * tan((pi - acos((l21_2 + l01_2 - l20_2) / (2 * l21 * l01))) / 2)
        t01 = l / l01
        t21 = l / l21

        # If the start tangent is not coincident with (x0,y0), line to.
        if abs(t01 - 1) > epsilon:
            cur = cur.L((x1 + t01 * x01, y1 + t01 * y01))
        assert cur
        cur = cur.arc_to(
            r,
            r,
            0,
            0,
            1 if y01 * x20 > x01 * y20 else 0,
            [x1 + t21 * x21, y1 + t21 * y21],
        )

    return cur


class BaseLC:
    "Base Linked command"

    def __init__(self, prev: Optional["BaseLC"], to: Sequence[float]):
        self._prev = prev
        self._to: FF = _pt2(to)

    def _bbox(self) -> BBox | None:
        """Compute this command's box, or return None when it has no geometry."""
        return None

    def bbox(self, *, default: BBox | None = None) -> BBox:
        """Return this command's box, using ``default`` when no box exists.

        Raise ValueError when no box can be computed and ``default`` is None.
        """
        result = self._bbox()
        if result is not None:
            return result
        if default is not None:
            return default
        raise ValueError(f"{type(self).__name__} has no computable bounding box")

    def to(self):
        return self._to

    def first(self) -> "BaseLC|None":
        cur: BaseLC | None = self
        while cur:
            p: BaseLC | None = cur._prev
            if p:
                cur = p
            else:
                break
        return cur

    def _each(self):
        cur: BaseLC | None = self
        while cur:
            p: BaseLC | None = cur._prev
            yield cur
            if p:
                cur = p
            else:
                break

    def last_move(self) -> "MoveLC|None":
        cur: BaseLC | None = self
        while cur:
            if isinstance(cur, MoveLC):
                return cur
            cur = cur._prev
        return None

    def _from(self):
        prev = self._prev
        assert prev
        return prev._to

    def start(self):
        prev = self._prev
        assert prev
        return prev._to

    def end(self):
        return self._to

    def req_prev(self):
        prev = self._prev
        assert isinstance(prev, BaseLC)
        return prev

    def path_len(self):
        total_len = 0.0
        cur: "BaseLC | None" = self
        while cur is not None:
            seg_len = cur.length()
            if seg_len > 0:
                total_len += seg_len
            cur = cur._prev
        return total_len

    ####
    def with_far_prev(self, far_prev: "BaseLC", new_prev: "BaseLC") -> "BaseLC":
        """Recursively replace `far_prev` with `new_prev` in the chain."""
        if far_prev is self:
            return new_prev
        elif self._prev is not None:
            updated_prev = self._prev.with_far_prev(far_prev, new_prev)
            return self.with_prev(updated_prev)
        else:
            raise ValueError("No prev")

    def with_far_prev_3(
        self, far_prev: "BaseLC", new_prev: Optional["BaseLC"]
    ) -> Optional["BaseLC"]:
        """Like with_far_prev, but allows new_prev to be None and returns Optional."""
        if far_prev is self:
            return self.with_prev(new_prev)
        elif self._prev is not None:
            updated_prev = self._prev.with_far_prev_3(far_prev, new_prev)
            return self.with_prev(updated_prev)
        else:
            raise ValueError("No prev")

    def describe(self, fnum=None, **kwargs) -> str:
        if fnum is None:
            s = "{0:." + str(kwargs.get("precision", 6)) + "f}"

            def fnum(n=0.0):
                f = s.format(n)
                return f.rstrip("0").rstrip(".") if "." in f else f

        # Collect parts iteratively from tail to head
        parts = []
        cur: "BaseLC | None" = self
        while cur is not None:
            cmd, *args = cur.term(**kwargs)
            parts.append(f"{cmd}" + kwargs.get("sep", ",").join(map(fnum, args)))
            cur = cur._prev

        # Reverse to get head-to-tail order and join
        return "".join(reversed(parts))

    def terms(self, **kwargs):
        # Collect term sequences iteratively from tail to head
        parts = []
        cur: "BaseLC | None" = self
        while cur is not None:
            parts.append(cur.term(**kwargs))
            cur = cur._prev

        # Reverse to head-to-tail and flatten into a single list
        result = []
        for part in reversed(parts):
            result.extend(part)
        return result

    ###
    def move_to(self, p: Sequence[float]):
        return MoveLC(self, p)

    def line_to(self, p: Sequence[float]):
        return LineLC(self, p)

    def close(self):
        m = self.last_move()
        return CloseLC(self, m._to) if m else self

    def curve_to(
        self, c1: Sequence[float], c2: Sequence[float], p2: Sequence[float]
    ) -> "CubicLC":
        return CubicLC(self, c1, (c2), (p2))

    def quad_to(self, c: Sequence[float], p: Sequence[float]):
        return QuadLC(self, (c), (p))

    def arc_to(
        self,
        rx: float,
        ry: float,
        phi: float,
        big_arc: bool | float,
        sweep: bool | float,
        p: Sequence[float],
    ):
        return ArcLC(self, rx, ry, phi, big_arc, sweep, (p))

    def arc_centered_at(
        self,
        c: Sequence[float],
        radius: float,
        startAngle: float,
        endAngle: float,
        counterclockwise=False,
    ):
        return arc_centered_at(self, c, radius, startAngle, endAngle, counterclockwise)

    def arc_tangent_to(self, p1: Sequence[float], p2: Sequence[float], r: float):
        return arc_tangent_to(self, p1, p2, r)

    ####
    @staticmethod
    def parse(s: str):
        from .parse import parse_each

        return parse_each(s)

    ####
    def M(self, p: Sequence[float]):
        return self.move_to((p))

    def m(self, p: Sequence[float]):
        return self.move_to(_add(self._to, p))

    def Z(self):
        return self.close()

    def z(self):
        return self.Z()

    def L(self, p: Sequence[float]):
        return self.line_to(p)

    def l(self, p: Sequence[float]):
        return self.line_to(_add(self._to, p))

    def H(self, n: float):
        return self.line_to(_with_x(self._to, n))

    def h(self, n: float):
        return self.line_to(_shift_x(self._to, n))

    def V(self, n: float):
        return self.line_to(_with_y(self._to, n))

    def v(self, n: float):
        return self.line_to(_shift_y(self._to, n))

    def Q(self, c: Sequence[float], p: Sequence[float]):
        return self.quad_to(c, p)

    def q(self, c: Sequence[float], p: Sequence[float]):
        rel = self._to
        return self.quad_to(_add(rel, c), _add(rel, p))

    def C(self, c1: Sequence[float], c2: Sequence[float], p2: Sequence[float]):
        return self.curve_to(c1, c2, p2)

    def c(self, c1: Sequence[float], c2: Sequence[float], p2: Sequence[float]):
        to = self._to
        return self.curve_to(_add(to, c1), _add(to, c2), _add(to, p2))

    def S(self, c: Sequence[float], p: Sequence[float]):
        if isinstance(self, CubicLC):
            return self.curve_to(_reflect(self.c2, self._to), (c), p)
        else:
            return self.curve_to(self._to, (c), p)

    def s(self, c: Sequence[float], p: Sequence[float]):
        to = self.to()
        if isinstance(self, CubicLC):
            return self.curve_to(_reflect(self.c2, to), _add(to, c), _add(to, p))
        else:
            return self.curve_to(to, _add(to, c), _add(to, p))

    def T(self, p: Sequence[float]):
        to = self._to
        if isinstance(self, QuadLC):
            return self.quad_to(_reflect(self.p, to), p)
        else:
            return self.quad_to(to, p)

    def t(self, p: Sequence[float]):
        to = self._to
        if isinstance(self, QuadLC):
            return self.quad_to(_reflect(self.p, to), _add(to, p))
        else:
            return self.quad_to(to, _add(to, p))

    def A(
        self,
        rx: float,
        ry: float,
        phi: float,
        big_arc: bool | float,
        sweep: bool | float,
        p: Sequence[float],
    ):
        return self.arc_to(rx, ry, phi, big_arc, sweep, p)

    def a(
        self,
        rx: float,
        ry: float,
        phi: float,
        big_arc: bool | float,
        sweep: bool | float,
        p: Sequence[float],
    ):
        return self.arc_to(rx, ry, phi, big_arc, sweep, _add(self._to, p))

    ###

    def length(self):
        return _len(_sub(self.end(), self.start()))

    def point_at(self, t: float):
        a, b = self._from(), self._to
        return _add(a, _mul(_sub(b, a), check_t(t)))

    def slope_at(self, t: float) -> FF:
        raise NotImplementedError(f"Not in {type(self)!r}")

    def tangent_at(self, t: float):
        s = self.slope_at(t)
        return _div(s, _len(s))

    def split_at(self, t: float) -> tuple["BaseLC", "BaseLC"]:
        raise NotImplementedError()

    def transform(self, m: Sequence[float]) -> "BaseLC":
        raise NotImplementedError(f"{self.__class__.__name__}")

    def reversed(self, next: "BaseLC|None") -> "BaseLC|None":
        raise NotImplementedError()

    def with_prev(self, prev: "BaseLC|None") -> "BaseLC":
        raise NotImplementedError()

    def term(self, **kwargs) -> Sequence[str | float | Any]:
        raise NotImplementedError()

    def as_curve(self) -> "MoveLC|CubicLC":
        raise NotImplementedError()

    def __repr__(self):
        raise NotImplementedError()


class LineLC(BaseLC):
    def _bbox(self) -> BBox | None:
        x2, y2, *_ = self._to
        prev = self._prev
        if prev:
            x1, y1, *_ = prev._to
            return ((min(x1, x2), max(x1, x2)), (min(y1, y2), max(y1, y2)))
        return None

    def with_prev(self, prev: "BaseLC|None") -> "BaseLC":
        return LineLC(prev, self._to)

    def split_at(self, t: float) -> tuple["BaseLC", "BaseLC"]:
        c = self.point_at(t)
        return LineLC(self._prev, c), LineLC(MoveLC(None, c), self._to)

    def reversed(self, next: "BaseLC|None") -> "BaseLC|None":
        if not next:
            next = MoveLC(None, self._to)
        prev = self._prev or MoveLC(None, (0, 0))
        rev = LineLC(next, prev._to)
        return prev.reversed(rev) or rev

    def as_curve(self) -> "MoveLC|CubicLC":
        prev = self._prev or MoveLC(None, (0, 0))
        cur = prev.as_curve()
        assert isinstance(cur, (CubicLC, MoveLC))
        c1 = cur.to()
        c2 = self.to()
        return cur.curve_to(c1, c2, c2)

    def term(self, **kwargs) -> Sequence[str | float | Any]:
        x, y, *_ = self._to
        prev = self._prev
        if prev is None:
            # No predecessor: best we can do is an absolute L
            return "L", x, y
        sx, sy, *_ = prev._to
        dx = x - sx
        dy = y - sy
        relative = kwargs.get("relative")
        short = kwargs.get("short")

        if relative:
            if short:
                if dx == 0:
                    return "v", dy
                if dy == 0:
                    return "h", dx
                return "l", dx, dy
            return "l", dx, dy
        else:
            if short:
                if sx == x:
                    return "V", y
                if sy == y:
                    return "H", x
                return "L", x, y
            return "L", x, y

    def transform(self, m: Sequence[float]) -> "BaseLC":
        prev = self._prev
        return self.__class__(
            prev.transform(m) if prev else None, _transform(m, self._to)
        )

    def length(self):
        return _len(_sub(self.to(), self.req_prev().to()))

    def __repr__(self):
        return f"{self.__class__.__name__}({[*self.to()]!r}, {self._prev!r})"

    def slope_at(self, t: float):
        # 't' not needed
        d = _sub(self.to(), self.req_prev().to())
        return _div(d, _len(d))


class MoveLC(LineLC):
    def length(self):
        return 0

    def term(self, **kwargs) -> Sequence[str | float | Any]:
        x, y = self.to()
        prev = self._prev
        if prev and kwargs.get("relative"):
            sx, sy = prev._to
            return "m", x - sx, y - sy
        return "M", x, y

    def split_at(self, t: float) -> tuple["BaseLC", "BaseLC"]:
        c = self.point_at(t)
        return MoveLC(self._prev, c), MoveLC(MoveLC(None, c), self._to)

    def with_prev(self, prev: BaseLC | None):
        return MoveLC(prev, self._to)

    def reversed(self, next: BaseLC | None) -> BaseLC | None:
        prev = self._prev
        if prev:
            seg = MoveLC(next, prev._to)
            return prev.reversed(seg) or seg
        return next

    def as_curve(self) -> "MoveLC|CubicLC":
        prev = self._prev
        return prev.as_curve().move_to(self.to()) if prev else MoveLC(None, self.to())

    def _bbox(self) -> BBox | None:
        # A MoveLC has no geometry; it must not contribute the "jump"
        # rectangle that LineLC.bbox() would otherwise produce.
        return None


class CloseLC(LineLC):
    def split_at(self, t: float) -> tuple["BaseLC", "BaseLC"]:
        c = self.point_at(t)
        return LineLC(self._prev, c), CloseLC(MoveLC(None, c), self._to)

    def term(self, **kwargs) -> Sequence[str | float | Any]:
        if kwargs.get("close") is False:
            return super().term(**kwargs)
        if kwargs.get("relative"):
            return "z"
        return "Z"

    def reversed(self, next: "BaseLC|None") -> "BaseLC|None":
        if not next:
            next = MoveLC(None, self._to)
        prev = self._prev
        if prev:
            rev = LineLC(next, prev._to)
            return prev.reversed(rev) or rev
        return next

    def with_prev(self, prev: BaseLC | None):
        return CloseLC(prev, self._to)


class QuadLC(BaseLC):
    p: FF

    def __init__(self, prev: BaseLC | None, p: Sequence[float], to: Sequence[float]):
        super().__init__(prev, to)
        self.p = _pt2(p)

    def _pts(
        self,
    ) -> tuple[Sequence[float], Sequence[float], Sequence[float]]:
        return (self._from(), self.p, self._to)

    def _bbox(self) -> BBox:
        (x1, y1), (x2, y2), (x3, y3) = self._pts()
        return (quadratic_extrema(x1, x2, x3), quadratic_extrema(y1, y2, y3))

    def transform(self, m: Sequence[float]):
        prev = self._prev
        return QuadLC(
            prev.transform(m) if prev else None,
            _transform(m, self.p),
            _transform(m, self._to),
        )

    def term(self, **kwargs) -> Sequence[str | float | Any]:
        p = self.p
        ex, ey, *_ = self._to
        x1, y1, *_ = p
        prev = self._prev
        if prev and kwargs:
            sx, sy, *_ = prev._to
            if kwargs.get("smooth"):
                ref = (
                    _reflect(prev.p, prev._to) if isinstance(prev, QuadLC) else prev._to
                )
                if _close(ref, p):
                    if kwargs.get("relative"):
                        return ["t", ex - sx, ey - sy]
                    return ["T", ex, ey]
            if kwargs.get("relative"):
                return ["q", x1 - sx, y1 - sy, ex - sx, ey - sy]
        return ["Q", x1, y1, ex, ey]

    def as_curve(self) -> "MoveLC|CubicLC":
        cur = (self._prev or MoveLC(None, (0, 0))).as_curve()
        assert isinstance(cur, (CubicLC, MoveLC))
        p0, p1, p2 = cur.to(), self.p, self.to()
        c1 = p0 if _close(p0, p1) else _add(p0, _mul(_sub(p1, p0), 2 / 3))
        c2 = p2 if _close(p2, p1) else _add(p2, _mul(_sub(p1, p2), 2 / 3))
        return cur.curve_to(c1, c2, p2)

    def length(self) -> float:
        return quad_length(self._pts())

    def slope_at(self, t: float):
        return quad_slope_at(self._pts(), check_t(t))

    def point_at(self, t: float):
        return quad_point_at(self._pts(), check_t(t))

    def split_at(self, t: float):

        t = check_t(t)
        a, b = quad_split_at(self._pts(), t)

        # Left: [from, a[1], a[2]] -> QuadLC(prev, a[1], a[2])
        left = QuadLC(prev=self._prev, p=a[1], to=a[2])

        # Right: starts at b[0]; use MoveLC to anchor it
        move = MoveLC(prev=None, to=b[0])
        right = QuadLC(prev=move, p=b[1], to=b[2])

        return left, right

    def reversed(self, next: Optional["BaseLC"] = None) -> Optional["BaseLC"]:
        if next is None:
            next = MoveLC(prev=None, to=self.to())

        if self._prev is not None:
            rev = QuadLC(prev=next, p=self.p, to=self._prev.to())
            result = self._prev.reversed(rev)
            return result if result is not None else rev
        else:
            return next

    def with_prev(self, prev: Optional[BaseLC]):
        return QuadLC(prev=prev, p=self.p, to=self.to())


class CubicLC(BaseLC):
    c1: FF
    c2: FF

    def __init__(
        self,
        prev: BaseLC | None,
        c1: Sequence[float],
        c2: Sequence[float],
        to: Sequence[float],
    ):
        super().__init__(prev, to)
        self.c1 = _pt2(c1)
        self.c2 = _pt2(c2)

    def _pts(
        self,
    ) -> Sequence[Sequence[float]]:
        return (self._from(), self.c1, self.c2, self._to)

    def _bbox(self) -> BBox:
        (x1, y1), (x2, y2), (x3, y3), (x4, y4) = self._pts()
        return (cubic_extrema(x1, x2, x3, x4), cubic_extrema(y1, y2, y3, y4))

    def term(self, **kwargs) -> Sequence[str | float | Any]:
        p1, c1, c2, p2 = self._pts()
        ex, ey = p2
        x1, y1 = c1
        x2, y2 = c2
        prev = self._prev
        if prev and kwargs:
            sx, sy, *_ = prev._to
            if kwargs.get("smooth"):
                ref = (
                    _reflect(prev.c2, prev._to)
                    if isinstance(prev, CubicLC)
                    else prev._to
                )
                if _close(ref, c1):
                    if kwargs.get("relative"):
                        return ["s", x2 - sx, y2 - sy, ex - sx, ey - sy]
                    return ["S", x2, y2, ex, ey]
            if kwargs.get("relative"):
                return ["c", x1 - sx, y1 - sy, x2 - sx, y2 - sy, ex - sx, ey - sy]
        return ["C", x1, y1, x2, y2, ex, ey]

    def transform(self, m: Sequence[float]):
        prev = self._prev
        return CubicLC(
            prev.transform(m) if prev else None,
            _transform(m, self.c1),
            _transform(m, self.c2),
            _transform(m, self._to),
        )

    def as_curve(self) -> "MoveLC|CubicLC":
        prev = self._prev.as_curve() if self._prev else MoveLC(None, (0, 0))
        assert isinstance(prev, (CubicLC, MoveLC))
        return prev.curve_to(self.c1, self.c2, self.to())

    def length(self):
        return cubic_length(self._pts())

    def point_at(self, t=0.0):
        return cubic_point_at(self._pts(), t)

    def slope_at(self, t: float):
        return cubic_slope_at(self._pts(), check_t(t))

    def split_at(self, t: float) -> tuple["BaseLC", "BaseLC"]:
        t = check_t(t)
        a, b = cubic_split_at(self._pts(), t)
        left = self.req_prev().curve_to(a[1], a[2], a[3])
        right = MoveLC(None, b[0]).curve_to(b[1], b[2], b[3])
        return left, right

    def reversed(self, next: Optional["BaseLC"] = None) -> Optional["BaseLC"]:
        if next is None:
            next = MoveLC(prev=None, to=self.to())

        if self._prev is not None:
            rev = CubicLC(next, c1=self.c2, c2=self.c1, to=self._prev.to())
            result = self._prev.reversed(rev)
            return result if result is not None else rev
        else:
            return next

    def with_prev(self, prev: Optional["BaseLC"]) -> "CubicLC":
        return CubicLC(prev=prev, c1=self.c1, c2=self.c2, to=self.to())


class ArcLC(BaseLC):
    def __init__(
        self,
        prev: BaseLC | None,
        rx: float,
        ry: float,
        PHI: float,
        big_arc: bool | float,
        sweep: bool | float,
        to: Sequence[float],
    ):
        super().__init__(prev, to)
        # Retain the flags for path serialization as well as for arc conversion.
        self.big_arc = bool(big_arc)
        self.sweep = bool(sweep)
        x1, y1 = self._from()  # needs prev
        x2, y2 = self._to

        # SVG: identical endpoints -> omit the arc (treat as zero-length / no-op)
        if abs(x1 - x2) < 1e-12 and abs(y1 - y2) < 1e-12:
            # Make a degenerate but valid ArcLC that never touches arc_params.
            # Length, point_at, etc. will see start == end and return 0 / the point.
            self.rx = abs(rx)
            self.ry = abs(ry)
            self._phi = PHI
            self.phi = PHI
            self.sin_phi = 0.0
            self.cos_phi = 1.0
            self.cx = x1
            self.cy = y1
            self.rtheta = 0.0
            self.rdelta = 0.0
            return

        # original call
        (
            self.phi,
            self.rx,
            self.ry,
            self.sin_phi,
            self.cos_phi,
            self.cx,
            self.cy,
            self.rtheta,
            self.rdelta,
        ) = arc_params(x1, y1, rx, ry, PHI, big_arc, sweep, x2, y2)
        self._phi = PHI

    def as_curve(self) -> "MoveLC|CubicLC":
        cur = self._prev.as_curve() if self._prev else MoveLC(None, (0, 0))
        assert isinstance(cur, (CubicLC, MoveLC))
        p1 = cur.to()
        p2 = self.to()
        for p1, c1, c2, p2 in arc_to_curve(
            self.rx,
            self.ry,
            self.cx,
            self.cy,
            self.sin_phi,
            self.cos_phi,
            self.rtheta,
            self.rdelta,
        ):
            cur = cur.curve_to(c1, c2, p2)
        return cur

    def _bbox(self) -> BBox | None:
        """Return the tight axis-aligned bounds of this elliptical arc.

        The cubic conversion in :meth:`as_curve` is an approximation, which
        is useful for drawing and interpolation but can slightly under- or
        overshoot the true arc extrema.  An SVG ellipse's extrema occur at
        four analytically known parameter angles; include those that lie on
        this arc, together with both endpoints.
        """
        start = self._from()
        end = self._to
        if _close(start, end):
            return None

        rx, ry = self.rx, self.ry
        cp, sp = self.cos_phi, self.sin_phi
        theta0, delta = self.rtheta, self.rdelta

        def on_arc(theta: float) -> bool:
            if delta >= 0:
                distance = (theta - theta0) % tau
                return distance <= delta + 1e-12
            distance = (theta0 - theta) % tau
            return distance <= -delta + 1e-12

        def point(theta: float):
            ct, st = cos(theta), sin(theta)
            return (
                self.cx + rx * cp * ct - ry * sp * st,
                self.cy + rx * sp * ct + ry * cp * st,
            )

        # For A*cos(theta) + B*sin(theta), the maximum is at atan2(B, A)
        # and the minimum is half a turn later.
        x_extreme = atan2(-ry * sp, rx * cp)
        y_extreme = atan2(ry * cp, rx * sp)
        candidates = [start, end]
        for theta in (x_extreme, x_extreme + pi, y_extreme, y_extreme + pi):
            if on_arc(theta):
                candidates.append(point(theta))

        xs = [p[0] for p in candidates]
        ys = [p[1] for p in candidates]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)

        # Trigonometric evaluation can put an endpoint extremum a few ulps
        # outside the exact endpoint coordinate (e.g. cos(pi / 2)).  Snap
        # only those tiny excursions back to the endpoint values.
        scale = max(1.0, rx, ry, *(abs(v) for p in (start, end) for v in p))
        epsilon = 1e-12 * scale
        endpoint_xmin, endpoint_xmax = sorted((start[0], end[0]))
        endpoint_ymin, endpoint_ymax = sorted((start[1], end[1]))
        if endpoint_xmin - epsilon <= xmin < endpoint_xmin:
            xmin = endpoint_xmin
        if endpoint_xmax < xmax <= endpoint_xmax + epsilon:
            xmax = endpoint_xmax
        if endpoint_ymin - epsilon <= ymin < endpoint_ymin:
            ymin = endpoint_ymin
        if endpoint_ymax < ymax <= endpoint_ymax + epsilon:
            ymax = endpoint_ymax

        return ((xmin, xmax), (ymin, ymax))

    def transform(self, m: Sequence[float]):
        # pylint: disable=invalid-name, too-many-locals
        prev = self._prev
        a0, b0, c0, d0, e0, f0 = m
        x, y = self._to
        newend = (a0 * x + c0 * y + e0, b0 * x + d0 * y + f0)

        if self._phi != 0:
            phi_rad = self._phi * pi / 180.0
            cp = cos(phi_rad)
            sp = sin(phi_rad)
            a_T = a0 * cp + c0 * sp
            b_T = b0 * cp + d0 * sp
            c_T = -a0 * sp + c0 * cp
            d_T = -b0 * sp + d0 * cp
        else:
            a_T, b_T, c_T, d_T = a0, b0, c0, d0

        a, c, b, d = a_T, b_T, c_T, d_T

        detT = a * d - b * c
        detT2 = detT**2

        rx = float(self.rx)
        ry = float(self.ry)

        def get_degen():
            # Degenerate: zero radius, or a singular transform collapsed the
            # arc to a point/line. SVG spec says treat this as line-to.
            prev_t = prev.transform(m) if prev else None
            if prev_t is None:
                return MoveLC(None, newend)
            return prev_t.line_to(newend)

        if rx == 0.0 or ry == 0.0 or detT2 == 0.0:
            # degenerate arc
            # transform only last point
            return get_degen()
        A = (d**2 / rx**2 + c**2 / ry**2) / detT2
        B = -(d * b / rx**2 + c * a / ry**2) / detT2
        D = (b**2 / rx**2 + a**2 / ry**2) / detT2

        theta = atan2(-2 * B, D - A) / 2
        theta_deg = theta * 180.0 / pi
        DA = D - A
        l2 = 4 * B**2 + DA**2

        if l2 == 0:
            delta = 0.0
        else:
            delta = 0.5 * (-(DA**2) - 4 * B**2) / sqrt(l2)

        half = (A + D) / 2

        try:
            rx_ = 1.0 / sqrt(half + delta)
            ry_ = 1.0 / sqrt(half - delta)

            if detT > 0:
                sweep = self.sweep
            else:
                sweep = not self.sweep > 0
            return ArcLC(
                prev.transform(m) if prev else None,
                rx_,
                ry_,
                theta_deg,
                self.big_arc,
                sweep,
                newend,
            )
            # return Arc(rx_ + 1j * ry_, theta_deg, self.large_arc, sweep, newend)
        except ZeroDivisionError:
            return get_degen()

    def term(self, **kwargs) -> Sequence[str | float | Any]:
        x, y, *_ = self._to
        prev = self._prev
        if prev and kwargs and kwargs.get("relative"):
            sx, sy, *_ = prev._to
            return [
                "a",
                self.rx,
                self.ry,
                self._phi,
                self.big_arc and 1 or 0,
                self.sweep and 1 or 0,
                x - sx,
                y - sy,
            ]
        return [
            "A",
            self.rx,
            self.ry,
            self._phi,
            self.big_arc and 1 or 0,
            self.sweep and 1 or 0,
            x,
            y,
        ]

    def point_at(self, t=0.0):
        a = self.req_prev().to()
        b = self.to()
        if _close(a, b):
            return a
        elif t <= 0:
            return a
        elif t >= 1:
            return b

        rx = self.rx
        ry = self.ry
        cosp = self.cos_phi
        sinp = self.sin_phi
        rtheta = self.rtheta
        rdelta = self.rdelta
        cx = self.cx
        cy = self.cy
        th = rtheta + rdelta * t
        sint = sin(th)
        cost = cos(th)

        x = rx * cosp * cost - ry * sinp * sint + cx
        y = rx * sinp * cost + ry * cosp * sint + cy
        return (x, y)

    def slope_at(self, t: float):

        rx = self.rx
        ry = self.ry
        cp = self.cos_phi
        sp = self.sin_phi
        rdelta = self.rdelta
        rtheta = self.rtheta

        th = rtheta + t * rdelta
        st = sin(th)
        ct = cos(th)
        k = rdelta  # scale factor = dth/dt

        dx = -rx * cp * st * k - ry * sp * ct * k
        dy = -rx * sp * st * k + ry * cp * ct * k

        return (dx, dy)

    def split_at(self, t: float) -> tuple["BaseLC", "BaseLC"]:
        t = check_t(t)
        mid = self.point_at(t)
        deltaA = abs(self.rdelta)
        t_flag = deltaA * t > pi
        one_minus_t_flag = deltaA * (1 - t) > pi
        left = ArcLC(
            prev=self._prev,
            rx=self.rx,
            ry=self.ry,
            PHI=self.phi,
            big_arc=t_flag,
            sweep=self.sweep,
            to=mid,
        )
        move = MoveLC(prev=None, to=mid)
        right = ArcLC(
            prev=move,
            rx=self.rx,
            ry=self.ry,
            PHI=self.phi,
            big_arc=one_minus_t_flag,
            sweep=self.sweep,
            to=self.to(),
        )
        return left, right

    def reversed(self, next: Optional[BaseLC] = None):
        if next is None:
            next = MoveLC(prev=None, to=self.to())

        if self._prev is not None:
            rev = ArcLC(
                prev=next,
                rx=self.rx,
                ry=self.ry,
                PHI=self.phi,
                big_arc=self.big_arc,
                sweep=not self.sweep,  # flip sweep direction
                to=self._prev.to(),
            )
            result = self._prev.reversed(rev)
            return result if result is not None else rev
        else:
            return next

    def length(self) -> float:
        b = self._to
        a = self._from()
        if _close(a, b):
            return 0
        return self.arc_length(0, 1, a, b, error=0.000005)

    def with_prev(self, prev: Optional[BaseLC]):
        return ArcLC(
            prev=prev,
            rx=self.rx,
            ry=self.ry,
            PHI=self.phi,
            big_arc=self.big_arc,
            sweep=self.sweep,
            to=self.to(),
        )

    LENGTH_MIN_DEPTH = 17
    LENGTH_ERROR = 1e-6

    def arc_length(
        self,
        start: float,
        end: float,
        start_point: Pt,
        end_point: Pt,
        error: float = LENGTH_ERROR,
        min_depth: int = LENGTH_MIN_DEPTH,
        depth: int = 0,
    ) -> float:
        mid = (start + end) / 2
        mid_point = self.point_at(mid)

        length = _len(_sub(end_point, start_point))
        first_half = _len(_sub(mid_point, start_point))
        second_half = _len(_sub(end_point, mid_point))
        length2 = first_half + second_half
        if isclose(length2, length, rel_tol=error) or (depth >= min_depth):
            return length2
        depth += 1
        return self.arc_length(
            start, mid, start_point, mid_point, error, min_depth, depth
        ) + self.arc_length(mid, end, mid_point, end_point, error, min_depth, depth)


def arc_centered_at(
    cur: BaseLC | None,
    c: Sequence[float],
    radius: float,
    start_angle: float,
    end_angle: float,
    counterclockwise=False,
    epsilon=1e-12,
):
    x, y = c
    a0 = start_angle
    a1 = end_angle
    cw = not counterclockwise
    r = radius
    dx = r * cos(a0)
    dy = r * sin(a0)
    x0 = x + dx
    y0 = y + dy

    if r < 0:
        raise ValueError("negative radius: " + str(r))
    elif cur is None:
        cur = MoveLC(None, (x0, y0))
    elif not _close(cur._to, (x0, y0), epsilon):
        cur = cur.line_to((x0, y0))

    da = a1 - a0 if cw else a0 - a1

    if r == 0:
        return cur
    elif da < 0:
        da = (da % tau) + tau

    assert cur
    if da > (tau - epsilon):
        return cur.arc_to(r, r, 0, 1, cw, [x - dx, y - dy]).arc_to(
            r, r, 0, 1, cw, [x0, y0]
        )
    elif da > epsilon:
        return cur.arc_to(r, r, 0, da >= pi, cw, [x + r * cos(a1), y + r * sin(a1)])

    return cur
