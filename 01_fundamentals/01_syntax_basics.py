"""
Module 01 – Python Syntax Basics
=================================
Demonstrates core Python 3.11+ syntax: variables, type annotations, control
flow, functions, generators, and comprehensions.

Run directly:  python 01_syntax_basics.py
"""

from __future__ import annotations

import math
from collections.abc import Generator, Iterator
from typing import Any


# ---------------------------------------------------------------------------
# 1. Variables and built-in types
# ---------------------------------------------------------------------------

def demonstrate_types() -> None:
    """Show Python's common built-in types with annotations."""
    name: str = "Alice"
    age: int = 30
    height: float = 1.75
    is_active: bool = True
    data: bytes = b"\x00\xff"
    nothing: None = None

    print(f"{name=}, {age=}, {height=}, {is_active=}, {data=}, {nothing=}")

    # Numeric operations
    print(f"Integer division: {17 // 5}")
    print(f"Modulo:           {17 %  5}")
    print(f"Power:            {2  ** 10}")
    print(f"Floor / ceil:     {math.floor(3.7)} / {math.ceil(3.2)}")


# ---------------------------------------------------------------------------
# 2. Strings
# ---------------------------------------------------------------------------

def demonstrate_strings() -> None:
    """Common string operations and formatting."""
    sentence = "  Hello, Python World!  "

    print(sentence.strip())
    print(sentence.lower().strip())
    print(sentence.strip().replace("World", "Universe"))
    print(", ".join(["alpha", "beta", "gamma"]))

    # f-string expressions
    pi = math.pi
    print(f"Pi to 4 decimals: {pi:.4f}")
    print(f"{'centred':^20}")

    # Multi-line strings
    haiku = """\
    An old silent pond
    A frog jumps into the pond
    Splash! Silence again
    """
    print(haiku)


# ---------------------------------------------------------------------------
# 3. Control flow
# ---------------------------------------------------------------------------

def grade(score: int) -> str:
    """Return letter grade for a numeric score using match statement."""
    match score // 10:
        case 10 | 9:
            return "A"
        case 8:
            return "B"
        case 7:
            return "C"
        case 6:
            return "D"
        case _:
            return "F"


def fizzbuzz(n: int) -> list[str]:
    """Classic FizzBuzz for numbers 1..n."""
    result: list[str] = []
    for i in range(1, n + 1):
        if i % 15 == 0:
            result.append("FizzBuzz")
        elif i % 3 == 0:
            result.append("Fizz")
        elif i % 5 == 0:
            result.append("Buzz")
        else:
            result.append(str(i))
    return result


# ---------------------------------------------------------------------------
# 4. Functions
# ---------------------------------------------------------------------------

def clamp(value: float, low: float, high: float) -> float:
    """Clamp *value* to [low, high] range."""
    return max(low, min(high, value))


def power(base: float, exp: int = 2) -> float:
    """Return base raised to exp (default: squared)."""
    return base ** exp


def multi_return(s: str) -> tuple[int, int, str]:
    """Return length, word-count, and uppercased version of *s*."""
    return len(s), len(s.split()), s.upper()


def variadic(*args: int, multiplier: int = 1) -> int:
    """Sum all positional args then multiply by *multiplier*."""
    return sum(args) * multiplier


# ---------------------------------------------------------------------------
# 5. Comprehensions
# ---------------------------------------------------------------------------

def demonstrate_comprehensions() -> None:
    """List, dict, set, and generator comprehensions."""
    squares = [x ** 2 for x in range(10)]
    evens   = [x for x in range(20) if x % 2 == 0]
    nested  = [x * y for x in range(1, 4) for y in range(1, 4)]

    word_lengths: dict[str, int] = {w: len(w) for w in ["apple", "banana", "cherry"]}
    unique_chars: set[str]       = {c.lower() for c in "Hello World" if c.isalpha()}

    # Generator expression (lazy)
    total = sum(x ** 2 for x in range(1_000_000))

    print(f"Squares:    {squares}")
    print(f"Evens:      {evens}")
    print(f"Nested:     {nested}")
    print(f"Word len:   {word_lengths}")
    print(f"Unique:     {sorted(unique_chars)}")
    print(f"Sum 0..1M²: {total}")


# ---------------------------------------------------------------------------
# 6. Generators and iterators
# ---------------------------------------------------------------------------

def fibonacci() -> Generator[int, None, None]:
    """Infinite Fibonacci sequence generator."""
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b


def take(n: int, it: Iterator[Any]) -> list[Any]:
    """Return the first *n* items from iterator *it*."""
    return [next(it) for _ in range(n)]


def countdown(start: int) -> Generator[int, None, None]:
    """Count down from *start* to 0 inclusive."""
    while start >= 0:
        yield start
        start -= 1


# ---------------------------------------------------------------------------
# 7. Walrus operator and other modern syntax
# ---------------------------------------------------------------------------

def demonstrate_modern_syntax() -> None:
    """Python 3.8+ walrus operator and other idioms."""
    data = [1, 4, 9, 16, 25, 36]

    # Walrus: assign and test in one expression
    if (n := len(data)) > 5:
        print(f"List has {n} elements (more than 5)")

    # Starred assignment
    first, *middle, last = data
    print(f"first={first}, middle={middle}, last={last}")

    # Dictionary merge (3.9+)
    defaults: dict[str, Any] = {"color": "blue", "size": 10}
    overrides: dict[str, Any] = {"size": 20, "weight": 1.5}
    merged = defaults | overrides
    print(f"Merged dict: {merged}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run all demonstrations."""
    print("=== Types ===")
    demonstrate_types()

    print("\n=== Strings ===")
    demonstrate_strings()

    print("\n=== Grades ===")
    for score in (95, 85, 75, 65, 55):
        print(f"  {score} -> {grade(score)}")

    print("\n=== FizzBuzz (1-20) ===")
    print(" ".join(fizzbuzz(20)))

    print("\n=== Comprehensions ===")
    demonstrate_comprehensions()

    print("\n=== Fibonacci (first 10) ===")
    fib_gen = fibonacci()
    print(take(10, fib_gen))

    print("\n=== Modern Syntax ===")
    demonstrate_modern_syntax()


if __name__ == "__main__":
    main()
