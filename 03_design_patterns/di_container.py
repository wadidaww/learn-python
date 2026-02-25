"""
di_container.py
================
Lightweight dependency injection (IoC) container.

Supports:
  - Transient registrations (new instance per resolve)
  - Singleton registrations (same instance per container)
  - Factory registrations (callable)
  - Auto-wiring via type hints
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")


class DIError(Exception):
    """Raised for DI container errors (missing registration, circular deps)."""


class Container:
    """
    Simple IoC container with auto-wiring.

    Example::

        container = Container()
        container.register_singleton(Database, PostgresDatabase)
        container.register_transient(UserService, UserService)

        svc = container.resolve(UserService)
    """

    def __init__(self) -> None:
        self._singletons: dict[type, Any] = {}
        self._transient_factories: dict[type, Callable[[], Any]] = {}
        self._singleton_factories: dict[type, Callable[[], Any]] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_singleton(
        self,
        interface: type[T],
        implementation: type[T] | None = None,
        instance: T | None = None,
    ) -> None:
        """
        Register a singleton.

        Pass either an *implementation* class (auto-wired) or a ready-made *instance*.
        """
        if instance is not None:
            self._singletons[interface] = instance
        elif implementation is not None:
            self._singleton_factories[interface] = lambda: self._build(implementation)
        else:
            self._singleton_factories[interface] = lambda: self._build(interface)

    def register_transient(
        self,
        interface: type[T],
        implementation: type[T] | None = None,
    ) -> None:
        """
        Register a transient – new instance on every resolve.
        """
        impl = implementation or interface
        self._transient_factories[interface] = lambda: self._build(impl)

    def register_factory(
        self,
        interface: type[T],
        factory: Callable[[], T],
    ) -> None:
        """Register a custom factory callable."""
        self._transient_factories[interface] = factory

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    def resolve(self, interface: type[T]) -> T:
        """
        Resolve a registered type.

        Raises:
            DIError: If the type is not registered.
        """
        # Already-built singleton?
        if interface in self._singletons:
            return self._singletons[interface]

        # Singleton factory?
        if interface in self._singleton_factories:
            instance = self._singleton_factories[interface]()
            self._singletons[interface] = instance
            return instance

        # Transient?
        if interface in self._transient_factories:
            return self._transient_factories[interface]()

        raise DIError(
            f"No registration for {interface.__name__!r}. "
            "Did you forget to register it?"
        )

    # ------------------------------------------------------------------
    # Auto-wiring
    # ------------------------------------------------------------------

    def _build(self, cls: type[T]) -> T:
        """
        Construct *cls* by resolving its __init__ type hints from this container.
        """
        sig = inspect.signature(cls.__init__)
        hints = {}
        try:
            import typing
            hints = typing.get_type_hints(cls.__init__)
        except Exception:
            pass

        kwargs: dict[str, Any] = {}
        for name, param in sig.parameters.items():
            if name == "self":
                continue
            # Skip *args and **kwargs parameters
            if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue
            if param.annotation is inspect.Parameter.empty:
                if param.default is inspect.Parameter.empty:
                    raise DIError(
                        f"Cannot auto-wire {cls.__name__}.{name}: "
                        "no type annotation and no default"
                    )
                continue
            dep_type = hints.get(name, param.annotation)
            try:
                kwargs[name] = self.resolve(dep_type)
            except DIError:
                if param.default is not inspect.Parameter.empty:
                    pass  # use default
                else:
                    raise

        return cls(**kwargs)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Example domain classes
# ---------------------------------------------------------------------------

class Logger:
    """Simple console logger."""

    def log(self, message: str) -> None:
        print(f"[LOG] {message}")


class Database:
    """Abstract database interface."""

    def query(self, sql: str) -> list[dict[str, Any]]:
        raise NotImplementedError


class InMemoryDatabase(Database):
    """In-memory database implementation."""

    def __init__(self) -> None:
        self._store: list[dict[str, Any]] = []

    def query(self, sql: str) -> list[dict[str, Any]]:
        return self._store

    def insert(self, record: dict[str, Any]) -> None:
        self._store.append(record)


class UserRepository:
    """User data access object."""

    def __init__(self, db: Database, logger: Logger) -> None:
        self._db = db
        self._logger = logger

    def find_all(self) -> list[dict[str, Any]]:
        self._logger.log("UserRepository.find_all()")
        return self._db.query("SELECT * FROM users")


class UserService:
    """High-level user service."""

    def __init__(self, repo: UserRepository) -> None:
        self._repo = repo

    def list_users(self) -> list[dict[str, Any]]:
        return self._repo.find_all()


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def main() -> None:
    """Demonstrate the DI container."""
    container = Container()

    # Register components
    container.register_singleton(Logger)
    container.register_singleton(Database, InMemoryDatabase)
    container.register_transient(UserRepository)
    container.register_transient(UserService)

    # Seed some data
    db = container.resolve(Database)
    assert isinstance(db, InMemoryDatabase)
    db.insert({"id": 1, "name": "Alice"})
    db.insert({"id": 2, "name": "Bob"})

    # Resolve high-level service (auto-wires all deps)
    svc = container.resolve(UserService)
    users = svc.list_users()
    print("Users:", users)

    # Singleton: same logger instance every time
    logger1 = container.resolve(Logger)
    logger2 = container.resolve(Logger)
    assert logger1 is logger2
    print(f"Singleton logger: {logger1 is logger2}")


if __name__ == "__main__":
    main()
