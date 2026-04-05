"""
Module 01 – Object-Oriented Programming
========================================
Demonstrates classes, inheritance, polymorphism, abstract base classes,
dataclasses, properties, and dunder methods in Python 3.11+.

Run directly:  python 03_oop.py
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar


# ---------------------------------------------------------------------------
# 1. Abstract base class + polymorphism
# ---------------------------------------------------------------------------

class Shape(ABC):
    """Abstract base class for 2-D geometric shapes."""

    @abstractmethod
    def area(self) -> float:
        """Return the area of the shape."""

    @abstractmethod
    def perimeter(self) -> float:
        """Return the perimeter of the shape."""

    def describe(self) -> str:
        """Human-readable description (uses concrete implementations)."""
        return (
            f"{type(self).__name__}: "
            f"area={self.area():.4f}, perimeter={self.perimeter():.4f}"
        )

    def __repr__(self) -> str:
        return self.describe()


class Circle(Shape):
    """A circle with a given radius."""

    def __init__(self, radius: float) -> None:
        if radius <= 0:
            raise ValueError(f"Radius must be positive, got {radius}")
        self._radius = radius

    @property
    def radius(self) -> float:
        """Read-only radius."""
        return self._radius

    def area(self) -> float:
        return math.pi * self._radius ** 2

    def perimeter(self) -> float:
        return 2 * math.pi * self._radius


class Rectangle(Shape):
    """A rectangle with width and height."""

    def __init__(self, width: float, height: float) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("Dimensions must be positive")
        self._width = width
        self._height = height

    @property
    def width(self) -> float:
        return self._width

    @property
    def height(self) -> float:
        return self._height

    def area(self) -> float:
        return self._width * self._height

    def perimeter(self) -> float:
        return 2 * (self._width + self._height)


class Square(Rectangle):
    """A square is a rectangle with equal sides."""

    def __init__(self, side: float) -> None:
        super().__init__(side, side)

    def __repr__(self) -> str:
        return f"Square(side={self._width}, area={self.area():.4f})"


# ---------------------------------------------------------------------------
# 2. Dataclasses
# ---------------------------------------------------------------------------

@dataclass(order=True)
class Point:
    """An immutable 2-D point."""

    x: float
    y: float

    def distance_to(self, other: Point) -> float:
        """Euclidean distance between *self* and *other*."""
        return math.hypot(self.x - other.x, self.y - other.y)

    def __add__(self, other: Point) -> Point:
        return Point(self.x + other.x, self.y + other.y)


@dataclass
class Employee:
    """A company employee record."""

    name: str
    department: str
    salary: float
    _headcount: ClassVar[int] = 0

    def __post_init__(self) -> None:
        if self.salary < 0:
            raise ValueError("Salary cannot be negative")
        Employee._headcount += 1

    @classmethod
    def headcount(cls) -> int:
        """Return total number of Employee instances created."""
        return cls._headcount

    def give_raise(self, percent: float) -> None:
        """Increase salary by *percent* %."""
        if percent < 0:
            raise ValueError("Raise percentage cannot be negative")
        self.salary *= 1 + percent / 100

    def __repr__(self) -> str:
        return f"Employee({self.name!r}, {self.department!r}, ${self.salary:,.2f})"


# ---------------------------------------------------------------------------
# 3. Properties with validation
# ---------------------------------------------------------------------------

class Temperature:
    """Temperature with Celsius storage and Fahrenheit/Kelvin views."""

    def __init__(self, celsius: float = 0.0) -> None:
        self.celsius = celsius  # uses property setter

    @property
    def celsius(self) -> float:
        """Temperature in degrees Celsius."""
        return self._celsius

    @celsius.setter
    def celsius(self, value: float) -> None:
        if value < -273.15:
            raise ValueError(f"Temperature below absolute zero: {value}")
        self._celsius = value

    @property
    def fahrenheit(self) -> float:
        """Temperature in degrees Fahrenheit."""
        return self._celsius * 9 / 5 + 32

    @fahrenheit.setter
    def fahrenheit(self, value: float) -> None:
        self.celsius = (value - 32) * 5 / 9

    @property
    def kelvin(self) -> float:
        """Temperature in Kelvin."""
        return self._celsius + 273.15

    def __repr__(self) -> str:
        return f"Temperature({self._celsius:.2f}°C / {self.fahrenheit:.2f}°F / {self.kelvin:.2f}K)"


# ---------------------------------------------------------------------------
# 4. Mixin pattern
# ---------------------------------------------------------------------------

class JSONSerializableMixin:
    """Mixin that adds simple JSON serialisation to dataclasses."""

    def to_dict(self) -> dict[str, object]:
        """Return instance fields as a plain dict."""
        import dataclasses
        if dataclasses.is_dataclass(self):
            return dataclasses.asdict(self)  # type: ignore[arg-type]
        return vars(self)


@dataclass
class Product(JSONSerializableMixin):
    """A product in a catalogue."""

    sku: str
    name: str
    price: float
    tags: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 5. Slots for memory efficiency
# ---------------------------------------------------------------------------

class Vector3D:
    """A 3-D vector using __slots__ for lower memory overhead."""

    __slots__ = ("x", "y", "z")

    def __init__(self, x: float, y: float, z: float) -> None:
        self.x = x
        self.y = y
        self.z = z

    def __add__(self, other: Vector3D) -> Vector3D:
        return Vector3D(self.x + other.x, self.y + other.y, self.z + other.z)

    def __mul__(self, scalar: float) -> Vector3D:
        return Vector3D(self.x * scalar, self.y * scalar, self.z * scalar)

    def magnitude(self) -> float:
        """Return the Euclidean norm."""
        return math.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)

    def dot(self, other: Vector3D) -> float:
        """Dot product with *other*."""
        return self.x * other.x + self.y * other.y + self.z * other.z

    def __repr__(self) -> str:
        return f"Vector3D({self.x}, {self.y}, {self.z})"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run OOP demonstrations."""
    shapes: list[Shape] = [Circle(5), Rectangle(4, 6), Square(3)]
    print("=== Shapes ===")
    for s in shapes:
        print(" ", s)

    print("\n=== Dataclass: Point ===")
    p1, p2 = Point(0, 0), Point(3, 4)
    print(f"  {p1} + {p2} = {p1 + p2}")
    print(f"  Distance: {p1.distance_to(p2)}")

    print("\n=== Dataclass: Employee ===")
    emp = Employee("Alice", "Engineering", 90_000)
    emp.give_raise(10)
    print(" ", emp)
    print(f"  Headcount: {Employee.headcount()}")

    print("\n=== Temperature ===")
    t = Temperature(100)
    print(" ", t)
    t.fahrenheit = 32
    print(" ", t)

    print("\n=== Product + Mixin ===")
    prod = Product("SKU-001", "Wireless Mouse", 29.99, ["electronics", "peripherals"])
    print("  Dict:", prod.to_dict())

    print("\n=== Vector3D ===")
    v1, v2 = Vector3D(1, 2, 3), Vector3D(4, 5, 6)
    print(f"  {v1} + {v2} = {v1 + v2}")
    print(f"  Dot product: {v1.dot(v2)}")
    print(f"  |v1| = {v1.magnitude():.4f}")


if __name__ == "__main__":
    main()
