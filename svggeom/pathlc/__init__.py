import re
from typing import Any, Optional, Sequence
from weakref import WeakKeyDictionary

# from ...numeric import to_float
from .command import BBox, BaseLC, CloseLC, MoveLC


class PathLC:
    """Path linked command"""

    _NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")
    _SEPARATOR = re.compile(r"(?:\s*,\s*|\s+)")

    @staticmethod
    def _points(value: str) -> list[tuple[float, float]]:
        """Parse an SVG ``points`` attribute into coordinate pairs."""
        matches = list(PathLC._NUMBER.finditer(value))
        if not matches:
            if value.strip():
                raise ValueError(f"Invalid SVG points value: {value!r}")
            return []

        values = []
        if value[: matches[0].start()].strip():
            raise ValueError(f"Invalid SVG points value: {value!r}")

        for index, match in enumerate(matches):
            if index:
                gap = value[matches[index - 1].end() : match.start()]
                # SVG permits a sign to separate adjacent numbers (e.g. ``10-5``).
                if gap and not PathLC._SEPARATOR.fullmatch(gap):
                    raise ValueError(f"Invalid SVG points value: {value!r}")
                if not gap and match.group()[0] not in "+-":
                    raise ValueError(f"Invalid SVG points value: {value!r}")
            number = float(match.group())
            values.append(0.0 if number == 0 else number)

        if value[matches[-1].end() :].strip():
            raise ValueError(f"Invalid SVG points value: {value!r}")
        if len(values) % 2:
            raise ValueError(
                f"SVG points needs coordinate pairs, got {len(values)} values"
            )

        return list(zip(values[::2], values[1::2]))

    @classmethod
    def _path_from_points(cls, value: str, *, close: bool = False) -> "PathLC | None":
        """Build a path from an SVG ``points`` attribute, or return ``None``."""
        points = cls._points(value)
        if not points:
            return None
        path = cls.at(points[0])
        for point in points[1:]:
            path.line_to(point)
        if close:
            path.close()
        return path

    # @classmethod
    # def _describe_points(cls, value: str, *, close: bool = False) -> str:
    #     """Describe an SVG ``points`` attribute as path data."""
    #     path = cls._path_from_points(value, close=close)
    #     return "" if path is None else path.describe()

    def __init__(self, tail: BaseLC):
        assert isinstance(tail, BaseLC)
        self._tail = tail

    @staticmethod
    def parse(d: str) -> "PathLC":
        return PathLC(BaseLC.parse(d))

    @staticmethod
    def at(p: Sequence[float]):
        return PathLC(MoveLC(None, p))

    def transform(self, M: Any):
        return PathLC(self._tail.transform(M))

    def _bbox(self) -> BBox | None:
        """Compute the union of command boxes, or None if the path has no geometry."""
        b: BBox | None = None
        cur: BaseLC | None = self._tail
        while cur:
            cb = cur._bbox()
            if cb is not None:
                if b is None:
                    b = cb
                else:
                    (xmin, xmax), (ymin, ymax) = cb
                    (bxmin, bxmax), (bymin, bymax) = b
                    b = (
                        (min(bxmin, xmin), max(bxmax, xmax)),
                        (min(bymin, ymin), max(bymax, ymax)),
                    )
            cur = cur._prev
        return b

    def bbox(self, *, default: BBox | None = None) -> BBox:
        """Return the path box, using ``default`` when the path has no geometry.

        Raise ValueError when no box can be computed and ``default`` is None.
        """
        result = self._bbox()
        if result is not None:
            return result
        if default is not None:
            return default
        raise ValueError("PathLC has no computable bounding box")

    def length(self):
        cur: BaseLC | None = self._tail
        if cur:
            return path_length(cur)
        return 0.0

    # str
    def __str__(self) -> str:
        return self.describe()

    def describe(self, **kwargs):
        return self._tail.describe(**kwargs)

    # build
    def move_to(self, p: Sequence[float]):
        self._tail = self._tail.move_to(p)
        return self

    def line_to(self, p: Sequence[float]):
        self._tail = self._tail.line_to(p)
        return self

    def arc_to(
        self,
        rx: float,
        ry: float,
        phi: float,
        big_arc: bool | float,
        sweep: bool | float,
        p: Sequence[float],
    ):
        """Append an SVG elliptical arc ending at ``p``."""
        self._tail = self._tail.arc_to(rx, ry, phi, big_arc, sweep, p)
        return self

    def close(self):
        self._tail = self._tail.close()
        return self

    def quad_to(self, c: Sequence[float], p: Sequence[float]):
        self._tail = self._tail.quad_to(c, p)
        return self

    def curve_to(self, c1: Sequence[float], c2: Sequence[float], p2: Sequence[float]):
        self._tail = self._tail.curve_to(c1, c2, p2)
        return self

    def rect(self, x: float, y: float, w: float, h: float):
        self._tail = self._tail.M([x, y]).h(w).v(h).h(-w).z()
        return self

    def arc_tangent_to(self, p1: Sequence[float], p2: Sequence[float], r: float):
        self._tail = self._tail.arc_tangent_to(p1, p2, r)
        return self

    def arc_centered_at(
        self,
        c: Sequence[float],
        radius: float,
        startAngle: float,
        endAngle: float,
        counterclockwise=False,
    ):
        self._tail = self._tail.arc_centered_at(
            c, radius, startAngle, endAngle, counterclockwise
        )
        return self

    def as_curve(self):
        self._tail = self._tail.as_curve()
        return self

    def segment_at_length(self, n: float, clamp=False):
        cur: BaseLC | None = self._tail
        if cur:
            return segment_at_length(cur, n, path_length(cur), clamp)
        return (None, float("nan"), float("nan"))

    def segment_at(self, T: float, clamp=False):
        cur: BaseLC | None = self._tail
        if cur:
            len = path_length(cur)
            seg, n, N = segment_at_length(cur, T * len, len, clamp)
            return seg, (0 if N == 0 else n / N)
        return (None, float("nan"))

    def tangent_at(self, T: float):
        seg, t = self.segment_at(T)
        if seg is not None:
            return seg.tangent_at(t)
        return None

    def slope_at(self, T: float):
        seg, t = self.segment_at(T)
        if seg is not None:
            return seg.slope_at(t)
        return None

    def point_at(self, T: float):
        seg, t = self.segment_at(T)
        if seg is not None:
            return seg.point_at(t)
        # Empty path (only a MoveLC): the point is the tail's endpoint.
        if self._tail is not None:
            return self._tail.to()
        return None

    def point_at_length(self, L: float, clamp: bool = False):
        seg, n, N = self.segment_at_length(L, clamp)
        if seg is not None:
            if N != 0:
                return seg.point_at(n / N)
            # Zero-length segment (or whole path of length 0)
            return seg.point_at(0)  # or seg.to() / seg.end()
        if self._tail is not None:
            return self._tail.to()
        return None

    def terms(self, **kwargs):
        return [] if self._tail is None else self._tail.terms(**kwargs)

    def reversed(self, _next: BaseLC | None = None):
        """Return a path whose drawable subpaths run in reverse order.

        Subpaths are reversed independently and their order is reversed too;
        move-only subpaths are retained. Closed subpaths remain closed.
        """
        commands = list(reversed(list(self._tail._each())))
        subpaths: list[list[BaseLC]] = []
        for command in commands:
            if isinstance(command, MoveLC):
                subpaths.append([command])
            elif subpaths:
                subpaths[-1].append(command)
            else:
                subpaths.append([command])

        result: BaseLC | None = _next
        for subpath in reversed(subpaths):
            # Detach this subpath from earlier subpaths before invoking the
            # command-level recursive reversal implementation.
            isolated: BaseLC | None = None
            for command in subpath:
                isolated = command.with_prev(isolated)
            assert isolated is not None
            original_tail = isolated
            reversed_tail = original_tail.reversed(None)
            if reversed_tail is None:
                # A move-only subpath reverses to the same move.
                first = isolated.first()
                assert first is not None
                reversed_tail = MoveLC(None, first.to())
            if isinstance(original_tail, CloseLC) and not isinstance(
                reversed_tail, MoveLC
            ):
                reversed_tail = CloseLC(reversed_tail._prev, reversed_tail.to())

            first_move = reversed_tail
            while first_move._prev is not None:
                first_move = first_move._prev
            if result is not None:
                reversed_tail = reversed_tail.with_far_prev_3(first_move, result)
            result = reversed_tail

        if result is None:
            raise RuntimeError("Can not reverse empty path")
        return PathLC(result)

    def split_at(self, T: float):
        _tail = self._tail
        if _tail is not None:
            seg, t = self.segment_at(T)
            if seg is not None:
                if t == 0.0:
                    prev = seg._prev
                    to_point = prev.to() if prev is not None else None
                    move_seg = MoveLC(None, to_point)  # type: ignore
                    left = PathLC(prev)
                    right = PathLC(_tail.with_far_prev_3(seg, move_seg))  # type: ignore
                    return left, right

                elif t == 1.0:
                    move_seg = MoveLC(None, seg.to())  # type: ignore
                    left = PathLC(seg)
                    right = PathLC(_tail.with_far_prev(seg, move_seg))  # type: ignore
                    return left, right

                if t < 0.0 or t > 1.0:
                    raise ValueError("Parameter t must be in [0, 1]")

                a, b = seg.split_at(t)  # type: ignore
                if seg is _tail:
                    return PathLC(a), PathLC(b)
                else:
                    return PathLC(a), PathLC(_tail.with_far_prev(seg, b))  # type: ignore

        # Default fallback
        raise RuntimeError("Cant split")
        # return PathLC(None), PathLC(None)

    def cut_at(self, T: float) -> "PathLC":
        """Return the portion of the path from start up to parameter T.

        - T in [0, 1):  prefix [0, T]
        - T >= 1:       the full path
        - T < 0:        wraps; returns the *suffix* [(T % 1), 1]
        """
        if T >= 1.0:
            return PathLC(self._tail)
        if T < 0:
            return self.split_at(T % 1.0)[1]
        return self.split_at(T)[0]

    def crop_at(self, T0: float, T1: float = 1.0) -> "PathLC":
        """Return the sub-path from normalized parameter T0 to T1.

        Handles wrapping, ordering, and edge cases.
        """
        # in crop_at, before normalize:
        if T1 >= 1.0:
            T1 = 1.0  # keep as "end"; do not normalize to 0.0
        else:
            T1 = normalize_t(T1)
        T0 = normalize_t(T0)
        # T1 = normalize_t(T1)

        # Case: Full path
        if T0 <= 0.0 and T1 >= 1.0:
            # TODO: Return a clone instead of self if mutability is a concern
            return self

        # Case: Crop from start to T1
        if T0 <= 0.0 and T1 > 0.0:
            return self.cut_at(T1)

        # Case: Crop from T0 to end
        if T0 < 1.0 and T1 >= 1.0:
            return self.cut_at(T0 - 1.0)

        # Case: Proper sub-interval within [0, 1)
        if T0 < 1.0 and T0 < T1:
            # Step 1: Cut at T0 -> keep suffix (from T0 to 1)
            suffix = self.cut_at(T0 - 1.0)  # T0 - 1 is negative -> returns [T0, 1]
            # Step 2: On that suffix, cut at relative position (T1 - T0) / (1 - T0)
            relative_t = (T1 - T0) / (1.0 - T0)
            return suffix.cut_at(relative_t)

        # Case: T0 > T1 -> swap and retry (handles wrap-around like [0.8, 0.2])
        if T0 > T1:
            return self.crop_at(T1, T0)

        # Fallback: degenerate case (T0 == T1 or out of bounds)
        raise RuntimeError("Cant crop")
        # return PathLC(None)  # or PathLC() if constructor allows


unit_len_map: WeakKeyDictionary[BaseLC, float] = WeakKeyDictionary()
path_len_map: WeakKeyDictionary[BaseLC, float] = WeakKeyDictionary()


def segment_length(cur: BaseLC) -> float:
    n = unit_len_map.get(cur)
    if n is None:
        n = unit_len_map[cur] = cur.length()
    return n


def path_length(cur: BaseLC):
    n = path_len_map.get(cur)
    if n is None:
        n = path_len_map[cur] = cur.path_len()
    return n


def segment_at_length(
    cur: Optional[BaseLC], n: float, LEN: float, clamp: bool = False
) -> tuple[Optional[BaseLC], float, float]:
    assert LEN >= 0
    if cur is not None:
        # A move-only path (or another zero-length path) has no drawable
        # segment. Avoid modulo by zero and let point_at_length use the tail.
        if LEN == 0:
            return (None, float("nan"), float("nan"))
        if n < 0:
            if clamp:
                n = 0.0
            else:
                n = n % LEN
        if n == 0:
            last = None
            temp_cur: BaseLC | None = cur
            while temp_cur is not None:
                if not isinstance(temp_cur, MoveLC):
                    last = temp_cur
                temp_cur = temp_cur._prev
            if last is not None:
                return (last, 0.0, segment_length(last))
        elif n > LEN:
            if clamp:
                n = LEN
            else:
                n = n % LEN
                if n == 0:
                    n = LEN

        to = LEN
        temp_cur = cur
        while temp_cur is not None:
            if not isinstance(temp_cur, MoveLC):
                lenS = segment_length(temp_cur)
                assert lenS >= 0
                if lenS >= 0:
                    to -= lenS
                    lenT = n - to
                    if lenT >= 0:
                        return (temp_cur, lenT, lenS)
            temp_cur = temp_cur._prev

    return (None, float("nan"), float("nan"))


def normalize_t(t: float) -> float:
    return t % 1


__all__ = "PathLC"
