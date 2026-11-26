from typing import Optional, Sequence
from math import atan2, cos, radians, sin, sqrt, tau
from .vector import Vector, vec


class RayIntersection(Vector):
    """A ray-pair intersection point with parameters on both rays.

    It behaves like a :class:`Vector`; ``t_self`` and ``t_other`` locate the
    point in ``self.pos + t_self * self.dir`` and
    ``other.pos + t_other * other.dir`` respectively. Parameters are not
    distances unless the corresponding direction is unit length.
    """

    def __new__(cls, point: Sequence[float], t_self: float, t_other: float):
        result = super().__new__(cls, *point)
        object.__setattr__(result, "_t_self", float(t_self))
        object.__setattr__(result, "_t_other", float(t_other))
        return result

    def __setattr__(self, name, value):
        raise AttributeError(f"{type(self).__name__} is immutable")

    @property
    def t_self(self) -> float:
        return self._t_self

    @property
    def t_other(self) -> float:
        return self._t_other

    @property
    def is_forward(self) -> bool:
        """Whether the intersection is at or ahead of both ray origins."""
        return self.t_self >= 0 and self.t_other >= 0

    def __reduce__(self):
        return (type(self), (Vector(*self), self.t_self, self.t_other))


class VecRay:
    """A ray: an origin ``pos`` and a direction ``dir``"""

    def __init__(self, pos: Sequence[float] = [0, 0], aim: Sequence[float] = [1, 0]):
        self._pos = vec(pos)
        self._dir = vec(aim)
        if not any(self._dir):
            raise ValueError(
                f"Ray direction must not be the zero vector: {tuple(self._dir)}"
            )
        if len(self._pos) != len(self._dir):
            raise ValueError("pos and dir must have the same dimension")

    # Query
    @property
    def x(self):
        return self.pos[0]

    @property
    def y(self):
        return self.pos[1]

    @property
    def z(self):
        if len(self.pos) < 3:
            raise AttributeError("2D ray has no z coordinate")
        return self.pos[2]

    @property
    def dx(self):
        return self.dir[0]

    @property
    def dy(self):
        return self.dir[1]

    @property
    def pos(self):
        return self._pos

    @property
    def dir(self):
        return self._dir

    def __iter__(self):
        """Iterate over the coordinates of the origin (not the direction)."""
        yield from self.pos

    def at(self):
        return self.pos

    def distance_to(self, p: Sequence[float]):
        """Straight-line distance from the origin to the point ``p``."""
        return self.vector_to(vec(p)).length()

    def point_at_distance(self, d):
        """The point ``d`` units from the origin along the direction.

        The length of the direction is ignored; a negative ``d`` goes backwards.
        This is computed with trigonometry, so it can differ from
        ``Ray.advanced(d).pos`` in the last digit. 2-D only."""
        # return self.pos + Vector.polar(d, self.dir.radians)
        return self.pos + self.dir.normalized() * d

    def vector_to(self, p: Sequence[float]):
        """The vector from the origin to the point ``p`` (``p - pos``)."""
        return vec(p) - self.pos

    def side_of(self, p: Sequence[float]):
        """Which side of the ray's line the point ``p`` lies on.
        Returns ``1`` if ``p`` is to the left of the direction of travel
        (counter-clockwise, y axis pointing up), ``-1`` if it is to the right,
        and ``0`` if it is exactly on the line. The test uses the infinite line
        through the ray, so points behind the origin count too, and the
        comparison with zero is exact (floating point)."""
        Ax, Ay = self.pos
        dx, dy = self.dir  # not (pos + dir) - pos: that loses dir at large |pos|
        X, Y = vec(p)
        d = dx * (Y - Ay) - dy * (X - Ax)
        return 1 if d > 0 else -1 if d < 0 else 0

    # Calc
    def distance_to_line(self, a: Sequence[float], b: Sequence[float]):
        """Perpendicular distance from the origin to the infinite line through
        ``a`` and ``b``.

        The line is not clamped to the segment ``a``-``b``; use
        :meth:`closest_point_on_segment` for segment distances. Returns ``nan``
        when ``a == b`` (no line is defined). 2-D only."""
        x, y = self.pos
        x1, y1 = a
        x2, y2 = b
        dx, dy = x2 - x1, y2 - y1

        if dx and dy:
            return abs(dx * (y1 - y) - dy * (x1 - x)) / sqrt(dx**2 + dy**2)
        elif dy:
            return abs(x1 - x)  # dx == 0
        elif dx:
            return abs(y1 - y)  # dy == 0
        return float("nan")

    def closest_point_on_segment(self, a: Sequence[float], b: Sequence[float]):
        """Closest point to the origin on the segment from ``a`` to ``b``.
        Clamped to the segment's end points"""
        return self.pos.closest_point_on_segment(a, b)

    def intersection_with_line(self, a, b) -> Optional[Vector]:
        """Intersection with the infinite line through ``a`` and ``b``.

        Returns ``None`` when the lines are parallel or coincident. Note that
        both the ray and the other line are currently treated as infinite, so
        the result can lie behind the ray's origin. 2-D only."""
        x1, y1 = a
        x2, y2 = b
        x3, y3 = self.pos
        x4, y4 = self.pos + self.dir
        e1 = x1 * y2 - y1 * x2
        e2 = x3 * y4 - y3 * x4
        dx = [x1 - x2, x3 - x4]
        dy = [y1 - y2, y3 - y4]
        d = dx[0] * dy[1] - dy[0] * dx[1]
        if d == 0:
            return None
        return Vector((e1 * dx[1] - dx[0] * e2) / d, (e1 * dy[1] - dy[0] * e2) / d)

    def intersection_with_ray(self, r: "VecRay") -> Optional[RayIntersection]:
        """Intersection of the supporting lines of this ray and ``r``.

        Returns ``None`` when the lines are parallel or coincident. Otherwise
        returns a :class:`RayIntersection`, which is also a :class:`Vector`.
        Its ``t_self`` and ``t_other`` parameters describe the intersection
        relative to each ray; ``is_forward`` is true only when it lies at or
        ahead of both origins. This reports behind-origin intersections too,
        rather than discarding their useful geometric information. 2-D only.
        """
        if len(self.pos) != 2 or len(r.pos) != 2:
            raise ValueError("ray intersection is 2-D only")

        delta = r.pos - self.pos
        cross = lambda a, b: a[0] * b[1] - a[1] * b[0]
        denominator = cross(self.dir, r.dir)
        if denominator == 0:
            return None

        t_self = cross(delta, r.dir) / denominator
        t_other = cross(delta, self.dir) / denominator
        point = self.pos + self.dir * t_self
        return RayIntersection(point, t_self, t_other)

    def closest_point_to(self, p: Sequence[float]):
        """Closest point on this ray (a half-line) to the point ``p``.

        This is the foot of the perpendicular from ``p`` when it lies ahead of
        the origin, and the origin itself when ``p`` is behind it."""
        p_vec = vec(p)
        d2 = self.dir.length_squared()
        if d2 == 0:
            return self.pos
        t = (p_vec - self.pos).dot(self.dir) / d2
        if t < 0:
            return self.pos
        return self.pos + self.dir * t


class Ray(VecRay):

    def clone(self):
        """A new ray with the same origin and direction."""
        pos, dir = self._pos, self._dir
        return type(self)(pos, dir)

    def new_pos(self, v: Vector):
        """A new ray with origin ``v`` and the same direction."""
        return type(self)(v, self.dir)

    def new_dir(self, v: Vector):
        """A new ray with the same origin and direction ``v``.

        Raises:
            ValueError: if ``v`` is the zero vector.
        """
        return type(self)(self.pos, v)

    def new_ray(self, p: Vector, a: Vector):
        """A new ray of this class with origin ``p`` and direction ``a``.

        Does not use this ray's own origin or direction.
        """
        return type(self)(p, a)

    def moved_to(self, p: Sequence[float]):
        """A new ray with its origin at ``p``."""
        return self.new_pos(vec(p))

    def pointing(self, rad: float | Sequence[float]):
        """A new ray with its direction *set* to ``rad``.
        Args:
            rad: an angle in radians, counter-clockwise from +x, which gives a
                unit-length direction (the old direction's length is
                discarded); or a vector, which is used as given.
        """
        if isinstance(rad, (int, float)):
            return self.new_dir(Vector.from_polar(1, rad))
        else:
            return self.new_dir(vec(rad))

    def oriented(self, deg: float):
        """Like :meth:`pointing`, with the angle given in degrees"""
        return self.pointing((deg * tau) / 360)

    def shifted(
        self,
        pos_delta: Sequence[float],
        dir_delta: Sequence[float],
    ):
        """A new ray with its origin and direction offset by the given vectors.

        The original ray is unchanged. The direction must remain non-zero.
        """
        return type(self)(self._pos + vec(pos_delta), self._dir + vec(dir_delta))

    def moved_by(self, p: Sequence[float]):
        """A new ray with its origin offset by ``p``"""
        return self.new_pos(self._pos + vec(p))

    def advanced(self, d: float):
        """A new ray whose origin has moved ``d`` units along the direction.

        The direction's length is ignored (it is normalized first); a negative
        ``d`` moves backwards."""
        return self.new_pos(self._pos + self._dir.normalized() * d)

    def lerp_to(self, p: Sequence[float], t: float):
        """A new ray whose origin is interpolated from here (``t=0``) to ``p``
        (``t=1``). ``t`` outside ``[0, 1]`` extrapolates. The direction is kept"""
        return self.new_pos(self._pos + (vec(p) - self._pos) * t)

    def rotated(self, rotation: float | Sequence[float]):
        """A new 2-D ray with its direction rotated counter-clockwise.

        A numeric ``rotation`` is an angle in degrees. A 2-component sequence
        is a rotation vector: its direction gives the rotation angle (via
        ``atan2(y, x)``); its length is ignored. The original direction's
        length and the ray's origin are preserved."""
        if isinstance(rotation, (int, float)):
            angle = radians(rotation)
        else:
            rotation_vec = vec(rotation)
            if len(rotation_vec) != 2:
                raise ValueError("rotation vector must be 2-D")
            if not any(rotation_vec):
                raise ValueError("rotation vector must not be zero")
            angle = atan2(rotation_vec[1], rotation_vec[0])

        if len(self.dir) != 2:
            raise ValueError("rotation is only supported for 2-D rays")

        c, s = cos(angle), sin(angle)
        dx, dy = self.dir
        return self.new_dir(Vector(dx * c - dy * s, dx * s + dy * c))
