"""Immutable N-dimensional vectors (a tuple subclass)."""

from math import atan2, cos, hypot, pi, sin, tau
from typing import Sequence


class Vector(tuple[float, ...]):
    """An immutable vector of ``int``/``float`` components"""

    def __new__(cls, *U):
        """Build a vector from components: ``Vector(x, y[, z, ...])``"""
        if not all(isinstance(x, (int, float)) for x in U):
            raise TypeError("Vector components must be numeric")
        return super(Vector, cls).__new__(cls, U)

    def __reduce__(self):
        # tuple's default hands __new__ a single tuple argument, which our
        # variadic __new__ rejects; rebuild from the components instead so that
        # copy, deepcopy and pickle all work.
        return (type(self), tuple(self))

    ##
    @property
    def x(self):
        return self[0]

    @property
    def y(self):
        return self[1]

    @property
    def z(self):
        if len(self) <= 2:
            raise AttributeError("2D vector has no z coordinate")
        return self[2]

    @property
    def radians(self):
        """Angle of the ``(x, y)`` part in radians, in ``[0, tau)``"""
        x, y, *_ = self
        r = atan2(y, x)
        if r < 0:
            r += tau
            # A tiny negative angle (e.g. -1e-20) rounds to exactly tau; wrap it.
            if r >= tau:
                r = 0.0
        return r

    @property
    def degrees(self):
        """:attr:`radians` converted to degrees, in ``[0, 360)``"""
        return (self.radians * 180) / pi

    @property
    def gradians(self):
        """:attr:`radians` converted to gradians (400 per turn), in ``[0, 400)``"""
        return (self.degrees * 10) / 9

    ##

    def __add__(self, other: Sequence[float]):  # type: ignore
        """Component-wise sum"""
        if len(self) != len(other):
            raise ValueError(
                f"Vector dimensions don't match: {len(self)} vs {len(other)}"
            )
        return Vector(*map((lambda a, b: a + b), self, other))

    def __sub__(self, other: Sequence[float]):
        """Component-wise difference"""
        if len(self) != len(other):
            raise ValueError(
                f"Vector dimensions don't match: {len(self)} vs {len(other)}"
            )
        return Vector(*map((lambda a, b: a - b), self, other))

    def __radd__(self, other: Sequence[float]):
        """Component-wise sum with a plain sequence on the left"""
        # Without this, `(1, 2) + Vector(3, 4)` silently falls back to tuple
        # concatenation and yields (1, 2, 3, 4).
        try:
            return vec(other) + self
        except TypeError:
            return NotImplemented

    def __rsub__(self, other: Sequence[float]):
        """Component-wise difference with a plain sequence on the left."""
        try:
            return vec(other) - self
        except TypeError:
            return NotImplemented

    def __mul__(self, scalar: float):  # type: ignore[override]
        """Multiply every component by a scalar (not tuple repetition)."""
        return Vector(*(x * scalar for x in self))

    def __rmul__(self, scalar: float):  # type: ignore
        """Scalar multiplication with the scalar on the left."""
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float):
        """Divide every component by a scalar."""
        return Vector(*(x / scalar for x in self))

    def __neg__(self):
        """Negate every component."""
        return Vector(*(-x for x in self))

    def lerp(self, other: Sequence[float], t: float):
        """Linear interpolation from this vector (``t=0``) to ``other`` (``t=1``)"""
        return self * (1 - t) + vec(other) * t

    ##
    def abs_quad(self):
        """Squared length (sum of squared components)"""
        return sum(n * n for n in self)

    def abs(self):
        """Euclidean length, computed with ``math.hypot`` so it cannot overflow
        or underflow in the intermediate squares."""
        return hypot(*self)  # avoids overflow/underflow of sum-of-squares

    def dot(self, other: Sequence[float]) -> float:
        """Compute the dot product of this vector with another vector"""
        if len(other) != len(self):
            raise ValueError(
                f"Vector dimensions don't match: {len(self)} vs {len(other)}"
            )
        return sum(a * b for a, b in zip(self, other))

    def normalized(self, epsilon=1e-12):
        """Return the unit-length vector pointing the same way"""
        a = self.abs()
        if abs(a) < epsilon:
            raise ValueError(f"Can't normalize vector of zero length [{self}]")
        return self / a

    ###
    def _padded(self, size: int):
        return (*self, *(0.0 for _ in range(max(0, size - len(self)))))

    def with_x(self, n: float):
        return Vector(*(n if i == 0 else v for i, v in enumerate(self._padded(1))))

    def with_y(self, n: float):
        return Vector(*(n if i == 1 else v for i, v in enumerate(self._padded(2))))

    def with_z(self, n: float):
        return Vector(*(n if i == 2 else v for i, v in enumerate(self._padded(3))))

    ###
    def shift_x(self, d: float):
        return Vector(*(d + v if i == 0 else v for i, v in enumerate(self._padded(1))))

    def shift_y(self, d: float):
        return Vector(*(d + v if i == 1 else v for i, v in enumerate(self._padded(2))))

    def shift_z(self, d: float):
        return Vector(*(d + v if i == 2 else v for i, v in enumerate(self._padded(3))))

    ###
    def reflect_through(self, p: Sequence[float]):
        """Point reflection: this vector mirrored through ``p`` (``2p - self``)"""
        P = vec(p)
        return P + (P - self)

    def apply_svg_matrix(self, m: Sequence[float]):
        """Transform ``(x, y)`` by an SVG hexad matrix ``[a, b, c, d, e, f]``"""
        a, b, c, d, e, f = m
        x, y, *_ = self
        return Vector(a * x + c * y + e, b * x + d * y + f, *_)

    def is_close(self, other: Sequence[float], abs_tol=1e-12):
        """Whether ``other`` has the same dimension and every component differs
        from this vector's by less than ``abs_tol``.

        A NaN component is never close to anything; equal infinities are close;
        two empty vectors are close."""
        # `a == b` first so equal infinities count as close (inf - inf is nan);
        # `< abs_tol` so that a NaN component is never close to anything.
        return len(self) == len(other) and all(
            a == b or abs(a - b) < abs_tol for a, b in zip(self, other)
        )

    @staticmethod
    def from_polar(radius: float = 1, phi: float = 0):
        # 2-D polar coordinates: return a 2-D vector.
        if radius:
            return Vector(radius * cos(phi), radius * sin(phi))
        return Vector(0, 0)

    @staticmethod
    def from_spherical(radius: float = 1, phi: float = 0, theta: float = 0):
        # http://www.kwon3d.com/theory/crdsys/polar.html
        sin_t = sin(theta)
        if radius:
            return Vector(
                radius * cos(phi) * sin_t,
                radius * sin(phi) * sin_t,
                radius * cos(theta),
            )
        return Vector(0, 0, 0)

    def closest_point_on_segment(self, a, b):
        """Closest point to this vector on the segment from ``a`` to ``b``.

        The result is clamped to the segment, so it is ``a`` or ``b`` when this
        point projects beyond an end. If ``a == b`` the result is ``a``."""
        a_vec, b_vec = vec(a), vec(b)
        a_to_b = b_vec - a_vec
        denom = a_to_b.abs_quad()
        if denom == 0:
            return a_vec
        t = (self - a_vec).dot(a_to_b) / denom
        t = max(0.0, min(1.0, t))
        return a_vec + (a_to_b * t)


def vec(v: Sequence[float]):
    """Return ``v`` itself if it is already a :class:`Vector`, else convert it."""
    return v if isinstance(v, Vector) else Vector(*v)
