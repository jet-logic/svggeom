from typing import Optional, Sequence
from math import atan2, cos, fabs, pi, radians, sin, tan


class Matrix:
    arg1: complex = 1 + 0j
    arg2: complex = 0 + 0j
    arg3: complex = 0 + 0j
    absolute_tolerance: float = 1e-05

    @property
    def a(self):
        return (self.arg1 + self.arg2).real

    @property
    def b(self):
        return (self.arg1 + self.arg2).imag

    @property
    def c(self):
        return (self.arg2 - self.arg1).imag

    @property
    def d(self):
        return (self.arg1 - self.arg2).real

    @property
    def e(self):
        return self.arg3.real

    @property
    def f(self):
        return self.arg3.imag

    def __init__(
        self, c1: complex = 1 + 0j, c2: complex = 0 + 0j, c3: complex = 0 + 0j
    ) -> None:
        self.arg1, self.arg2, self.arg3 = c1, c2, c3

    def to_hexad(self):
        """Returns the transform as a hexad matrix (used in svg)"""
        return (self.a, self.b, self.c, self.d, self.e, self.f)

    def __fastmatmul(self, oa1: complex, oa2: complex, oa3: complex):
        """Matrix multiplication without instance checks or transform istantiation"""
        return (
            self.arg1 * oa1 + self.arg2 * oa2.conjugate(),
            self.arg1 * oa2 + self.arg2 * oa1.conjugate(),
            self.arg1 * oa3 + self.arg2 * oa3.conjugate() + self.arg3,
        )

    def __matmul__(self, matrix: "Matrix"):
        """Combine this transform's internal matrix with the given matrix"""
        # Conform the input to a known quantity (and convert if needed)
        # Return a transformation as the combined result
        oa1, oa2, oa3 = matrix.arg1, matrix.arg2, matrix.arg3  # type: ignore
        return Matrix(*self.__fastmatmul(oa1, oa2, oa3))

    def __imatmul__(self, matrix: "Matrix"):
        """In place multiplication of transform matrices"""
        tmp = self @ matrix
        self.arg1, self.arg2, self.arg3 = tmp.arg1, tmp.arg2, tmp.arg3
        return self

    def __neg__(self):
        """Returns an inverted transformation"""
        det = (self.arg1 * self.arg1.conjugate()) - (self.arg2 * self.arg2.conjugate())

        # invert the rotation/scaling part
        na1 = self.arg1.conjugate() / det
        na2 = -(self.arg2 / det)
        # invert the translational part
        na3 = -na1 * self.arg3 - na2 * self.arg3.conjugate()
        return Matrix(na1, na2, na3)

    def _is_URT(self, exactly=False):
        """
        Checks that transformation can be decomposed into product of
        Uniform scale (U), Rotation around origin (R) and translation (T)

        :return: decomposition as U*R*T is possible
        """
        tol = self.absolute_tolerance if not exactly else 0.0
        return (fabs(self.a - self.d) <= tol) and (fabs(self.b + self.c) <= tol)

    def is_identity(self):
        return self.arg1 == 1 + 0j and self.arg2 == 0 + 0j and self.arg3 == 0 + 0j

    def is_translate(self, exactly: bool = False):
        """Returns True if this transformation is ONLY translate"""
        tol = self.absolute_tolerance if not exactly else 0.0
        return (
            fabs(self.a - 1) <= tol
            and abs(self.d - 1) <= tol
            and fabs(self.b) <= tol
            and fabs(self.c) <= tol
        )

    def is_scale(self, exactly=False):
        """Returns True if this transformation is ONLY scale"""
        tol = self.absolute_tolerance if not exactly else 0.0
        return (
            fabs(self.e) <= tol
            and fabs(self.f) <= tol
            and fabs(self.b) <= tol
            and fabs(self.c) <= tol
        )

    def is_rotate(self, exactly=False):
        """Returns True if this transformation is ONLY rotate"""
        tol = self.absolute_tolerance if not exactly else 0.0
        return (
            self._is_URT(exactly=exactly)
            and fabs(self.e) <= tol
            and fabs(self.f) <= tol
            and fabs(self.a**2 + self.b**2 - 1) <= tol
        )

    def rotation_degrees(self):
        # type: () -> float
        """Return the amount of rotation in this transform"""
        if not self._is_URT(exactly=False):
            raise ValueError(
                "Rotation angle is undefined for non-uniformly scaled or skewed "
                "matrices"
            )
        return atan2(self.b, self.a) * 180 / pi

    def __str__(self):
        """Format the given matrix into a string representation for svg"""
        hexad = self.to_hexad()
        if self.is_translate():
            if not self:
                return ""
            return f"translate({self.e:.6g}, {self.f:.6g})"
        if self.is_scale():
            return f"scale({self.a:.6g}, {self.d:.6g})"
        if self.is_rotate():
            return f"rotate({self.rotation_degrees():.6g})"
        return f"matrix({' '.join(f'{var:.6g}' for var in hexad)})"

    def add_rotate(
        self, deg, center_x: float | tuple[float, float] = 0, center_y: float = 0
    ):
        """Add rotation to this transformation"""
        if isinstance(center_x, (tuple, list)):
            center_x, center_y = center_x
            c = complex(center_x, center_y)
        else:
            c = complex(center_x, center_y)
        _cos, _sin = cos(radians(deg)), sin(radians(deg))
        self.arg1, self.arg2, self.arg3 = self.__fastmatmul(_cos + 1j * _sin, 0, c)
        self.arg1, self.arg2, self.arg3 = self.__fastmatmul(1 + 0j, 0 + 0j, -c)
        return self

    def add_skewx(self, deg: float):
        """Add skew x to this transformation, and return it"""
        ttan = tan(radians(deg)) / 2
        self.arg1, self.arg2, self.arg3 = self.__fastmatmul(
            1 - ttan * 1j, 0 + ttan * 1j, 0 + 0j
        )
        return self

    def add_skewy(self, deg: float):
        """Add skew y to this transformation, and return it"""
        ttan = tan(radians(deg)) / 2
        self.arg1, self.arg2, self.arg3 = self.__fastmatmul(
            1 + ttan * 1j, 0 + ttan * 1j, 0 + 0j
        )
        return self

    def add_scale(self, sc_x: float, sc_y: Optional[float] = None):
        """Add scale to this transformation"""
        sc_x = float(sc_x)
        sc_y = sc_x if sc_y is None else float(sc_y)
        self.arg1, self.arg2, self.arg3 = self.__fastmatmul(
            (sc_x + sc_y) / 2, (sc_x - sc_y) / 2, 0
        )
        return self

    def add_translate(self, x: float | tuple[float, float], y: float = 0):
        """Add translate to this transformation"""
        if isinstance(x, (tuple, list)):
            x, y = x

        self.arg1, self.arg2, self.arg3 = self.__fastmatmul(1, 0, complex(x, y))

        return self

    def add_skew_from_axis(self, skew: float, axis: float):
        self.add_rotate(axis)
        self @= Matrix.hexad(c=tan(skew))
        self.add_rotate(-axis)
        return self

    def add_matrix(
        self,
        a: float = 1,
        b: float = 0,
        c: float = 0,
        d: float = 1,
        e: float = 0,
        f: float = 0,
    ):
        self @= Matrix.hexad(a, b, c, d, e, f)
        return self

    def capply_to_point(self, point: complex) -> complex:
        """Transform a tuple (X, Y), return a complex"""
        return self.arg1 * point + self.arg2 * point.conjugate() + self.arg3

    def apply_to_point(self, point: Sequence[float]) -> tuple[float, float]:
        """Transform a vector (X, Y)"""
        # x, y = point
        # return Vec(a * x + c * y + e, b * x + d * y + f)
        c = self.capply_to_point(complex(point[0], point[1]))
        return c.real, c.imag

    @staticmethod
    def hexad(
        a: float = 1,
        b: float = 0,
        c: float = 0,
        d: float = 1,
        e: float = 0,
        f: float = 0,
    ):
        return Matrix(
            (a + d) / 2 + 1j * (b - c) / 2, (a - d) / 2 + 1j * (b + c) / 2, e + f * 1j
        )
