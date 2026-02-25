"""
creational/singleton.py
========================
Thread-safe Singleton pattern implemented two ways:

1. ``SingletonMeta`` – metaclass approach (most Pythonic)
2. ``SingletonDecorator`` – decorator approach
"""

from __future__ import annotations

import threading


# ---------------------------------------------------------------------------
# 1. Metaclass Singleton
# ---------------------------------------------------------------------------

class SingletonMeta(type):
    """
    Thread-safe Singleton metaclass.

    Classes using this metaclass will only ever have one instance.

    Example::

        class AppConfig(metaclass=SingletonMeta):
            def __init__(self) -> None:
                self.debug = False

        a = AppConfig()
        b = AppConfig()
        assert a is b
    """

    _instances: dict[type, object] = {}
    _lock: threading.Lock = threading.Lock()

    def __call__(cls, *args: object, **kwargs: object) -> object:
        with cls._lock:
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
        return cls._instances[cls]


class ApplicationConfig(metaclass=SingletonMeta):
    """
    Global application configuration (Singleton).

    All parts of the application share the same config object.
    """

    def __init__(self) -> None:
        self.debug: bool = False
        self.database_url: str = "sqlite:///:memory:"
        self.max_connections: int = 10
        self._settings: dict[str, object] = {}

    def set(self, key: str, value: object) -> None:
        """Store a configuration value."""
        self._settings[key] = value

    def get(self, key: str, default: object = None) -> object:
        """Retrieve a configuration value."""
        return self._settings.get(key, default)

    def __repr__(self) -> str:
        return (
            f"ApplicationConfig(debug={self.debug}, "
            f"db={self.database_url!r}, settings={self._settings})"
        )


# ---------------------------------------------------------------------------
# 2. Decorator Singleton
# ---------------------------------------------------------------------------

def singleton(cls: type) -> type:
    """
    Class decorator that turns *cls* into a Singleton.

    Usage::

        @singleton
        class Logger:
            ...
    """
    instances: dict[type, object] = {}
    lock = threading.Lock()

    def get_instance(*args: object, **kwargs: object) -> object:
        with lock:
            if cls not in instances:
                instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    get_instance._cls = cls  # type: ignore[attr-defined]
    return get_instance  # type: ignore[return-value]


@singleton
class DatabasePool:
    """Simulated database connection pool (Singleton via decorator)."""

    def __init__(self, max_size: int = 5) -> None:
        self.max_size = max_size
        self._pool: list[str] = [f"conn-{i}" for i in range(max_size)]

    def acquire(self) -> str | None:
        """Acquire a connection from the pool."""
        return self._pool.pop() if self._pool else None

    def release(self, conn: str) -> None:
        """Return a connection to the pool."""
        self._pool.append(conn)

    def available(self) -> int:
        """Number of available connections."""
        return len(self._pool)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def main() -> None:
    """Demonstrate Singleton patterns."""
    # Metaclass singleton
    cfg1 = ApplicationConfig()
    cfg2 = ApplicationConfig()
    assert cfg1 is cfg2, "Should be the same instance"
    cfg1.debug = True
    print(f"cfg2.debug = {cfg2.debug}")  # True – shared state
    print(f"Same instance: {cfg1 is cfg2}")

    # Thread-safety test
    results: list[int] = []

    def create_config() -> None:
        results.append(id(ApplicationConfig()))

    threads = [threading.Thread(target=create_config) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print(f"All {len(set(results))} unique id(s) from 10 threads")

    # Decorator singleton
    pool1 = DatabasePool()
    pool2 = DatabasePool()
    assert pool1 is pool2
    conn = pool1.acquire()
    print(f"Acquired: {conn}, available: {pool2.available()}")


if __name__ == "__main__":
    main()
