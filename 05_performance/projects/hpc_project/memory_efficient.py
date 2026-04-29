"""
memory_efficient.py
===================
Demonstrates memory-efficient Python patterns that matter at HPC scale.

Topics
------
1. ``__slots__`` vs regular ``__dict__`` classes
2. Generator pipelines vs materialised lists
3. ``array.array`` vs ``list`` for homogeneous numeric data
4. ``struct`` / ``ctypes.Structure`` for packed records
5. ``sys.getsizeof`` / ``tracemalloc`` inspection helpers

Run::

    python memory_efficient.py
"""

from __future__ import annotations

import array
import ctypes
import struct
import sys
import tracemalloc
from collections.abc import Generator, Iterator
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# 1. __slots__ vs __dict__
# ---------------------------------------------------------------------------

POINT_COUNT = 200_000


class PointDict:
    """Regular class – stores attributes in a per-instance ``__dict__``."""

    def __init__(self, x: float, y: float, z: float) -> None:
        self.x = x
        self.y = y
        self.z = z

    def distance_sq(self) -> float:
        """Return squared distance from origin."""
        return self.x**2 + self.y**2 + self.z**2


class PointSlots:
    """
    Slots class – replaces ``__dict__`` with a fixed C-level array.

    Benefits
    --------
    * ~40–50 % lower memory per instance (no hash-table overhead).
    * Slightly faster attribute access.
    """

    __slots__ = ("x", "y", "z")

    def __init__(self, x: float, y: float, z: float) -> None:
        self.x = x
        self.y = y
        self.z = z

    def distance_sq(self) -> float:
        """Return squared distance from origin."""
        return self.x**2 + self.y**2 + self.z**2


def measure_slots_vs_dict(n: int = POINT_COUNT) -> None:
    """Compare heap usage of PointDict vs PointSlots."""
    print("\n=== __slots__ vs __dict__ ===")

    tracemalloc.start()
    dict_points = [PointDict(float(i), float(i), float(i)) for i in range(n)]
    _, dict_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    _ = dict_points  # keep reference alive during measurement

    tracemalloc.start()
    slot_points = [PointSlots(float(i), float(i), float(i)) for i in range(n)]
    _, slot_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    _ = slot_points

    print(f"  PointDict  peak: {dict_peak / 1024 / 1024:.2f} MB")
    print(f"  PointSlots peak: {slot_peak / 1024 / 1024:.2f} MB")
    saving = (dict_peak - slot_peak) / dict_peak * 100
    print(f"  Saving:   ~{saving:.0f}%")

    # Per-instance size estimate
    d = PointDict(1.0, 2.0, 3.0)
    s = PointSlots(1.0, 2.0, 3.0)
    print(f"  sizeof(PointDict instance):  {sys.getsizeof(d)} bytes")
    print(f"  sizeof(PointSlots instance): {sys.getsizeof(s)} bytes")


# ---------------------------------------------------------------------------
# 2. Generator pipelines
# ---------------------------------------------------------------------------


def read_numbers(n: int) -> Iterator[int]:
    """Simulate reading numbers from disk – yields one at a time."""
    yield from range(n)


def square_gen(it: Iterator[int]) -> Generator[int, None, None]:
    """Square each value (generator stage)."""
    for x in it:
        yield x * x


def filter_even_gen(it: Iterator[int]) -> Generator[int, None, None]:
    """Pass through only even values (generator stage)."""
    for x in it:
        if x % 2 == 0:
            yield x


def pipeline_generator(n: int) -> int:
    """Run a three-stage generator pipeline; return the final count."""
    return sum(filter_even_gen(square_gen(read_numbers(n))))


def pipeline_list(n: int) -> int:
    """Equivalent computation materialising each stage into a list."""
    numbers = list(range(n))
    squared = [x * x for x in numbers]
    evens = [x for x in squared if x % 2 == 0]
    return sum(evens)


def measure_generator_vs_list(n: int = 500_000) -> None:
    """Compare peak memory of generator pipeline vs materialised lists."""
    print("\n=== Generator pipeline vs materialised lists ===")

    tracemalloc.start()
    _ = pipeline_list(n)
    _, list_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    tracemalloc.start()
    _ = pipeline_generator(n)
    _, gen_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print(f"  pipeline_list      peak: {list_peak / 1024:.1f} KB")
    print(f"  pipeline_generator peak: {gen_peak / 1024:.1f} KB")
    print(f"  Memory reduction:  {(list_peak - gen_peak) / list_peak * 100:.0f}%")


# ---------------------------------------------------------------------------
# 3. array.array vs list
# ---------------------------------------------------------------------------


def measure_array_vs_list(n: int = 1_000_000) -> None:
    """Show raw memory difference between list[int] and array.array('l')."""
    print("\n=== array.array vs list (1 million integers) ===")

    int_list = list(range(n))
    int_arr: array.array = array.array("l", range(n))  # type: ignore[type-arg]

    # sys.getsizeof only counts the container header for lists; itemsize gives true cost
    list_bytes = sys.getsizeof(int_list)
    arr_bytes = sys.getsizeof(int_arr)
    print(f"  list size:         {list_bytes / 1024 / 1024:.2f} MB (header only; objects extra)")
    print(f"  array.array size:  {arr_bytes / 1024 / 1024:.2f} MB (all data inlined)")
    print(f"  Per-element array: {int_arr.itemsize} bytes vs ~28 bytes (CPython int object)")


# ---------------------------------------------------------------------------
# 4. struct-packed records
# ---------------------------------------------------------------------------


_PARTICLE_FMT = "fff"  # 3 × float32 = 12 bytes
_PARTICLE_SIZE = struct.calcsize(_PARTICLE_FMT)


def pack_particles(positions: list[tuple[float, float, float]]) -> bytes:
    """Pack particle positions into a tight binary buffer."""
    buf = bytearray(len(positions) * _PARTICLE_SIZE)
    offset = 0
    for x, y, z in positions:
        struct.pack_into(_PARTICLE_FMT, buf, offset, x, y, z)
        offset += _PARTICLE_SIZE
    return bytes(buf)


def unpack_particles(buf: bytes) -> list[tuple[float, float, float]]:
    """Unpack a binary buffer produced by :func:`pack_particles`."""
    n = len(buf) // _PARTICLE_SIZE
    return [
        struct.unpack_from(_PARTICLE_FMT, buf, i * _PARTICLE_SIZE)  # type: ignore[misc]
        for i in range(n)
    ]


# 5. ctypes Structure – zero-overhead interop record
class Vec3(ctypes.Structure):
    """
    A compact 3-component float vector backed by a ctypes Structure.

    Memory layout is identical to a C ``struct { float x, y, z; }``,
    making it safe to pass directly to C extensions or shared-memory regions.
    """

    _fields_ = [("x", ctypes.c_float), ("y", ctypes.c_float), ("z", ctypes.c_float)]

    def __repr__(self) -> str:
        return f"Vec3(x={self.x:.3f}, y={self.y:.3f}, z={self.z:.3f})"

    def length(self) -> float:
        """Euclidean length."""
        return (self.x**2 + self.y**2 + self.z**2) ** 0.5


def demo_struct_records(n: int = 5) -> None:
    """Show struct packing and ctypes.Structure usage."""
    print("\n=== struct-packed records ===")
    positions = [(float(i), float(i) * 0.5, float(i) * 0.1) for i in range(n)]
    buf = pack_particles(positions)
    recovered = unpack_particles(buf)
    print(f"  Packed {n} particles into {len(buf)} bytes ({_PARTICLE_SIZE} bytes each)")
    print(f"  First record: {recovered[0]}")

    print("\n=== ctypes.Structure (Vec3) ===")
    v = Vec3(1.0, 2.0, 3.0)
    print(f"  {v}  |length| = {v.length():.4f}  sizeof = {ctypes.sizeof(v)} bytes")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run all memory-efficiency demonstrations."""
    print("=" * 60)
    print("  Memory-Efficient Patterns Demo")
    print("=" * 60)
    measure_slots_vs_dict()
    measure_generator_vs_list()
    measure_array_vs_list()
    demo_struct_records()


if __name__ == "__main__":
    main()
