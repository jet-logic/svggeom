from typing import Generator, Optional, Sequence, Tuple, Union, overload
from math import isfinite

IntervalArgs = Union[
    "Interval", Tuple[float, float], float
]  # pylint: disable=invalid-name


class Interval(tuple[float, float]):
    """A pair of numbers that represent the minimum and maximum values."""

    # We use __new__ instead of __init__ because tuples are immutable
    # and must be initialized at creation time.
    def __new__(
        cls,
        x: Union[Optional["Interval"], Tuple[float, float], float] = None,
        y: Optional[float] = None,
    ) -> "Interval":
        if y is not None:
            if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                # FIX: Apply min() and max() to ensure correct ordering
                min_val, max_val = min(float(x), float(y)), max(float(x), float(y))
            else:
                raise ValueError(
                    f"Not a number for scaling: {str((x, y))} "
                    f"({type(x).__name__},{type(y).__name__})"
                )
        else:
            value = x
            if value is None:
                # identity for addition, zero for intersection
                min_val, max_val = float("+inf"), float("-inf")
            elif isinstance(value, Interval):
                min_val, max_val = value.minimum, value.maximum
            elif isinstance(value, (tuple, list)) and len(value) == 2:
                min_val, max_val = min(value), max(value)
            elif isinstance(value, (int, float)):
                min_val = max_val = float(value)
            else:
                raise ValueError(
                    f"Not a number for scaling: {str(value)} ({type(value).__name__})"
                )

        # Initialize the underlying tuple
        return super().__new__(cls, (min_val, max_val))

    # Expose tuple indices as properties to maintain API compatibility
    @property
    def minimum(self) -> float:
        return self[0]

    @property
    def maximum(self) -> float:
        return self[1]

    def __bool__(self):
        return isfinite(self.minimum) and isfinite(self.maximum)

    def __neg__(self):
        return Interval((-1 * self.maximum, -1 * self.minimum))

    # --- Arithmetic Operators (Immutable) ---
    # Note: __iadd__, __iand__, and __imul__ are intentionally omitted.
    # Python will automatically use __add__, __and__, and __mul__ and
    # rebind the variable when +=, &=, or *= are used.

    def __add__(self, other):
        """Calculate the bounding interval that covers both given bounding intervals"""
        if other is None:
            return Interval(self)
        other = Interval(other)
        return Interval(
            (min(self.minimum, other.minimum), max(self.maximum, other.maximum))
        )

    def __radd__(self, other):
        if other is None:
            return Interval(self)
        return self + other

    def __and__(self, other: "Interval") -> "Interval":
        """Calculate the bounding interval where both given bounding intervals overlap"""
        if other is None:
            return Interval(self)
        other = Interval(other)
        min_val = max(self.minimum, other.minimum)
        max_val = min(self.maximum, other.maximum)
        if min_val > max_val:
            return Interval()
        return Interval((min_val, max_val))

    def __rand__(self, other):
        if other is None:
            return Interval(self)
        return self & other

    def __mul__(self, other) -> "Interval":
        if other is None:
            return Interval(self)
        return Interval((self.minimum * other, self.maximum * other))

    def __iter__(self) -> Generator[float, None, None]:
        yield self.minimum
        yield self.maximum

    def __eq__(self, other) -> bool:
        if isinstance(other, Interval):
            return super().__eq__(other)
        try:
            return super().__eq__(Interval(other))
        except ValueError:
            return False

    def __contains__(self, value) -> bool:
        return self.minimum <= value <= self.maximum

    def __repr__(self) -> str:
        return f"BoundingInterval({self.minimum}, {self.maximum})"

    @property
    def center(self) -> float:
        return self.minimum + ((self.maximum - self.minimum) / 2)

    @property
    def size(self) -> float:
        return self.maximum - self.minimum

    def equals(self, p: "Interval", epsilon=1e-12):
        if not p:
            return False
        elif p is self:
            return True
        return (
            abs(self.minimum - p.minimum) < epsilon
            and abs(self.maximum - p.maximum) < epsilon
        )


class BoundingBox:  # pylint: disable=too-few-public-methods
    """
    Some functions to compute a rough bbox of a given list of objects.

    BoundingBox(other)
    BoundingBox(x, y)
    BoundingBox((x1, x2), (y1, y2))
    """

    x: Interval
    y: Interval

    @property
    def width(self):
        return self.x.size

    @property
    def height(self):
        return self.y.size

    @property
    def top(self):
        return self.y.minimum

    @property
    def left(self):
        return self.x.minimum

    @property
    def bottom(self):
        return self.y.maximum

    @property
    def right(self):
        return self.x.maximum

    @property
    def center_x(self):
        return self.x.center

    @property
    def center_y(self):
        return self.y.center

    @property
    def min_y(self):
        return self.y.minimum

    @property
    def min_x(self):
        return self.x.minimum

    @property
    def max_y(self):
        return self.y.maximum

    @property
    def max_x(self):
        return self.x.maximum

    @property
    def diagonal_length(self):
        return (self.width**2 + self.height**2) ** (0.5)

    @overload
    def __init__(self, other=None):
        # type: (Optional[BoundingBox]) -> None
        pass

    @overload
    def __init__(self, x, y):
        # type: (IntervalArgs, IntervalArgs) -> None
        pass

    def __init__(self, x=None, y=None):  # type: ignore
        if y is None:
            if x is None:
                # identity for addition, zero for intersection
                pass
            elif isinstance(x, BoundingBox):
                x, y = x.x, x.y
            else:
                raise ValueError(
                    f"Not a number for scaling: {str(x)} ({type(x).__name__})"
                )
        self.x = Interval(x)
        self.y = Interval(y)

    @staticmethod
    def from_rect(x: float, y: float, width: float, height: float) -> "BoundingBox":
        """Create a bounding box using x, y, width and height"""
        return BoundingBox((x, x + width), (y, y + height))

    def __bool__(self):
        # type: () -> bool
        return bool(self.x) and bool(self.y)

    def __neg__(self):
        # type: () -> BoundingBox
        return BoundingBox(-self.x, -self.y)

    def __add__(self, other):
        # type: (Optional[BoundingBox]) -> BoundingBox
        """Calculate the bounding box that covers both given bounding boxes"""
        new = BoundingBox(self)
        new += BoundingBox(other)
        return new

    def __iadd__(self, other):
        # type: (Optional[BoundingBox]) -> BoundingBox
        other = BoundingBox(other)
        self.x += other.x
        self.y += other.y
        return self

    def __radd__(self, other):
        # type: (Optional[BoundingBox]) -> BoundingBox
        return self + other

    def __and__(self, other):
        # type: (Optional[BoundingBox]) -> BoundingBox
        """Calculate the bounding box where both given bounding boxes overlap"""
        new = BoundingBox(self)
        new &= BoundingBox(other)
        return new

    def __iand__(self, other: Optional["BoundingBox"]) -> "BoundingBox":
        other = BoundingBox(other)
        self.x = self.x & other.x
        self.y = self.y & other.y
        if not self.x or not self.y:
            self.x, self.y = Interval(), Interval()
        return self

    def __rand__(self, other):
        # type: (Optional[BoundingBox]) -> BoundingBox
        return self & other

    def __mul__(self, factor):
        # type: (float) -> BoundingBox
        new = BoundingBox(self)
        new *= factor
        return new

    def __imul__(self, factor):
        # type: (float) -> BoundingBox
        self.x *= factor
        self.y *= factor
        return self

    def __eq__(self, other):
        # type (object) -> bool
        if isinstance(other, BoundingBox):
            return tuple(self) == tuple(other)
        return False

    def __iter__(self) -> Generator["Interval", None, None]:
        yield self.x
        yield self.y

    @property
    def area(self):
        """Return area of the bounding box

        .. versionadded:: 1.2"""
        return self.width * self.height

    @property
    def minimum(self):
        """Return the minimum x,y coords"""
        return (self.x.minimum, self.y.minimum)

    @property
    def maximum(self):
        """Return the maximum x,y coords"""
        return (self.x.maximum, self.y.maximum)

    def __repr__(self):
        # type: () -> str
        return f"BoundingBox({tuple(self.x)},{tuple(self.y)})"

    @property
    def center(self):
        """Returns the middle of the bounding box"""
        return (self.x.center, self.y.center)

    @property
    def size(self):
        return (self.x.size, self.y.size)

    def resize(self, delta_x: float, delta_y: Optional[float] = None) -> "BoundingBox":
        """Enlarges / shrinks a bounding box by a constant value. If only delta_x
        is given, each side is moved by the same amount; if delta_y is given,
        different deltas are applied to horizontal and vertical intervals."""
        if delta_y is None:
            delta_y = delta_x
        return BoundingBox(
            (self.x.minimum - delta_x, self.x.maximum + delta_x),
            (self.y.minimum - delta_y, self.y.maximum + delta_y),
        )

    def equals(self, p: "BoundingBox", epsilon=1e-12):
        if not p:
            return False
        elif p is self:
            return True
        return self.x.equals(p.x, epsilon=epsilon) and self.y.equals(
            p.y, epsilon=epsilon
        )

    def dump_rect(self):
        x, y, w, h = self.left, self.top, self.width, self.height
        return (x, y, w, h)

    def transform(self, m: Sequence[float]):
        if not self:
            return BoundingBox()  # return empty
        xMin = yMin = float("inf")
        xMax = yMax = float("-inf")
        left, top, bottom, right = self.left, self.top, self.bottom, self.right
        a, b, c, d, e, f = m
        corners = (
            (left, top),
            (right, top),
            (left, bottom),
            (right, bottom),
        )
        for x, y in corners:
            nx = a * x + c * y + e
            ny = b * x + d * y + f
            xMin = min(xMin, nx)
            xMax = max(xMax, nx)
            yMin = min(yMin, ny)
            yMax = max(yMax, ny)
        return self.__class__((xMin, xMax), (yMin, yMax))
