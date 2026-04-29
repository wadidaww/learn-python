"""
caching.py
==========
Memoisation and caching strategies for expensive computations.

Patterns covered
----------------
1. ``functools.lru_cache``  – automatic LRU memoisation (pure functions)
2. ``functools.cache``      – unbounded memoisation (Python 3.9+)
3. Manual dict memoisation  – full control over eviction
4. ``TimedCache``           – TTL-based expiry (cache invalidation by age)
5. ``LFUCache``             – Least-Frequently-Used eviction policy

Run::

    python caching.py
"""

from __future__ import annotations

import functools
import time
from collections import defaultdict
from collections.abc import Callable
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

# ---------------------------------------------------------------------------
# 1. functools.lru_cache
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=256)
def fibonacci_lru(n: int) -> int:
    """
    Recursive Fibonacci with automatic LRU memoisation.

    Without caching this is O(2ⁿ); with caching it becomes O(n).
    ``lru_cache`` stores up to *maxsize* recent results and evicts the
    least-recently-used entry when the cache is full.
    """
    if n < 2:
        return n
    return fibonacci_lru(n - 1) + fibonacci_lru(n - 2)


@functools.cache  # unbounded – use when the input space is small and finite
def binomial(n: int, k: int) -> int:
    """Binomial coefficient C(n, k) with unbounded memoisation."""
    if k == 0 or k == n:
        return 1
    return binomial(n - 1, k - 1) + binomial(n - 1, k)


# ---------------------------------------------------------------------------
# 2. Manual dict memoisation (with eviction control)
# ---------------------------------------------------------------------------


class MemoDict:
    """
    Simple manual memoisation dictionary.

    Compared with ``lru_cache`` this gives full control: you can inspect
    the cache, purge individual entries, or plug in a custom eviction
    policy.
    """

    def __init__(self) -> None:
        self._store: dict[Any, Any] = {}
        self.hits = 0
        self.misses = 0

    def get(self, key: Any) -> tuple[bool, Any]:
        """Return ``(found, value)``."""
        if key in self._store:
            self.hits += 1
            return True, self._store[key]
        self.misses += 1
        return False, None

    def put(self, key: Any, value: Any) -> None:
        """Store *value* under *key*."""
        self._store[key] = value

    def invalidate(self, key: Any) -> None:
        """Remove *key* from the cache."""
        self._store.pop(key, None)

    def clear(self) -> None:
        """Clear the entire cache."""
        self._store.clear()
        self.hits = self.misses = 0

    @property
    def size(self) -> int:
        """Number of cached entries."""
        return len(self._store)

    def hit_ratio(self) -> float:
        """Fraction of lookups that were cache hits."""
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


_prime_factors_cache: MemoDict = MemoDict()


def expensive_prime_factors(n: int, _cache: MemoDict = _prime_factors_cache) -> list[int]:
    """
    Return the prime factors of *n*, using a shared MemoDict.

    The default-argument trick keeps one cache instance per function.
    """
    hit, cached = _cache.get(n)
    if hit:
        return cached  # type: ignore[return-value]

    factors: list[int] = []
    d = 2
    remaining = n
    while d * d <= remaining:
        while remaining % d == 0:
            factors.append(d)
            remaining //= d
        d += 1
    if remaining > 1:
        factors.append(remaining)

    _cache.put(n, factors)
    return factors


# ---------------------------------------------------------------------------
# 3. TimedCache – TTL-based expiry
# ---------------------------------------------------------------------------


class TimedCache:
    """
    Key-value cache where entries expire after *ttl* seconds.

    Useful for caching API responses, configuration, or any value that
    may become stale over time.
    """

    def __init__(self, ttl: float = 60.0) -> None:
        self.ttl = ttl
        self._store: dict[Any, tuple[Any, float]] = {}  # key -> (value, expires_at)

    def get(self, key: Any) -> tuple[bool, Any]:
        """Return ``(valid, value)``; ``valid`` is False when expired or absent."""
        entry = self._store.get(key)
        if entry is None:
            return False, None
        value, expires_at = entry
        if time.monotonic() > expires_at:
            del self._store[key]
            return False, None
        return True, value

    def put(self, key: Any, value: Any) -> None:
        """Store *value* under *key* with a TTL-based expiry timestamp."""
        self._store[key] = (value, time.monotonic() + self.ttl)

    def __len__(self) -> int:
        return len(self._store)


def timed_cache(ttl: float = 60.0) -> Callable[[F], F]:
    """
    Decorator that wraps a function with a :class:`TimedCache`.

    Usage::

        @timed_cache(ttl=5.0)
        def get_config(env: str) -> dict:
            ...
    """

    def decorator(fn: F) -> F:
        cache: TimedCache = TimedCache(ttl=ttl)

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            key = (args, tuple(sorted(kwargs.items())))
            valid, value = cache.get(key)
            if valid:
                return value
            result = fn(*args, **kwargs)
            cache.put(key, result)
            return result

        wrapper.cache = cache  # type: ignore[attr-defined]
        return wrapper  # type: ignore[return-value]

    return decorator


# ---------------------------------------------------------------------------
# 4. LFU Cache – Least-Frequently-Used eviction
# ---------------------------------------------------------------------------


class LFUCache:
    """
    Least-Frequently-Used cache with O(1) average get/put.

    When the cache reaches capacity the key that has been accessed the
    fewest times is evicted.  Ties are broken by insertion order (oldest
    evicted first).

    Reference: "An O(1) algorithm for implementing the LFU cache eviction
    scheme" – Shah et al., 2010.
    """

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        self.capacity = capacity
        self._key_to_val: dict[Any, Any] = {}
        self._key_to_freq: dict[Any, int] = {}
        self._freq_to_keys: defaultdict[int, list[Any]] = defaultdict(list)
        self._min_freq = 0

    def get(self, key: Any) -> Any:
        """Return cached value or raise ``KeyError`` on miss."""
        if key not in self._key_to_val:
            raise KeyError(key)
        self._increment_freq(key)
        return self._key_to_val[key]

    def put(self, key: Any, value: Any) -> None:
        """Insert or update *key* → *value*."""
        if key in self._key_to_val:
            self._key_to_val[key] = value
            self._increment_freq(key)
            return

        if len(self._key_to_val) >= self.capacity:
            self._evict()

        self._key_to_val[key] = value
        self._key_to_freq[key] = 1
        self._freq_to_keys[1].append(key)
        self._min_freq = 1

    def _increment_freq(self, key: Any) -> None:
        freq = self._key_to_freq[key]
        self._key_to_freq[key] = freq + 1
        self._freq_to_keys[freq].remove(key)
        if not self._freq_to_keys[freq] and freq == self._min_freq:
            self._min_freq += 1
        self._freq_to_keys[freq + 1].append(key)

    def _evict(self) -> None:
        keys_at_min = self._freq_to_keys[self._min_freq]
        evicted = keys_at_min.pop(0)  # oldest among least-frequent
        del self._key_to_val[evicted]
        del self._key_to_freq[evicted]

    def __len__(self) -> int:
        return len(self._key_to_val)


# ---------------------------------------------------------------------------
# Demo helpers
# ---------------------------------------------------------------------------


def _time_call(fn: Callable[[], Any], label: str) -> float:
    start = time.perf_counter()
    fn()
    elapsed = time.perf_counter() - start
    print(f"  {label:<40} {elapsed * 1000:.3f} ms")
    return elapsed


def demo_lru_cache() -> None:
    """Show lru_cache speedup on recursive Fibonacci."""
    print("\n=== functools.lru_cache – Fibonacci ===")
    fibonacci_lru.cache_clear()

    cold_time = _time_call(lambda: fibonacci_lru(35), "fibonacci_lru(35) – cold cache")
    warm_time = _time_call(lambda: fibonacci_lru(35), "fibonacci_lru(35) – warm cache")
    info = fibonacci_lru.cache_info()
    print(f"  Cache: {info}")
    print(f"  Speedup (warm / cold): {cold_time / warm_time:.0f}x")


def demo_binomial() -> None:
    """Show unbounded cache on binomial coefficients."""
    print("\n=== functools.cache – Binomial coefficients ===")
    results = [binomial(20, k) for k in range(21)]
    print(f"  Row 20 of Pascal's triangle: {results}")
    print(f"  C(20,10) = {binomial(20, 10)}")


def demo_timed_cache() -> None:
    """Demonstrate TTL-based cache with a short expiry window."""
    print("\n=== TimedCache (TTL = 0.1 s) ===")

    call_count = 0

    @timed_cache(ttl=0.1)
    def slow_fn(x: int) -> int:
        nonlocal call_count
        call_count += 1
        time.sleep(0.02)  # simulate I/O
        return x * x

    slow_fn(5)
    slow_fn(5)  # hit
    slow_fn(5)  # hit
    print(f"  3 calls with TTL=0.1s → underlying fn called {call_count}× (expected 1)")

    time.sleep(0.15)  # let the TTL expire
    slow_fn(5)  # miss – TTL elapsed
    print(f"  After TTL expiry       → underlying fn called {call_count}× (expected 2)")


def demo_lfu_cache() -> None:
    """Demonstrate LFU eviction order."""
    print("\n=== LFUCache (capacity=3) ===")
    cache: LFUCache = LFUCache(capacity=3)
    for i in range(3):
        cache.put(i, i * 10)

    cache.get(0)  # freq[0]=2
    cache.get(0)  # freq[0]=3
    cache.get(1)  # freq[1]=2
    # freq: {0:3, 1:2, 2:1}  → key 2 is LFU

    cache.put(3, 30)  # evicts key 2
    try:
        cache.get(2)
        print("  ERROR: key 2 should have been evicted")
    except KeyError:
        print("  Key 2 correctly evicted (LFU)")

    print(f"  cache[0]={cache.get(0)}, cache[1]={cache.get(1)}, cache[3]={cache.get(3)}")


def main() -> None:
    """Run all caching demonstrations."""
    print("=" * 60)
    print("  Caching Strategies Demo")
    print("=" * 60)
    demo_lru_cache()
    demo_binomial()
    demo_timed_cache()
    demo_lfu_cache()


if __name__ == "__main__":
    main()
