"""
behavioral/observer.py
=======================
Observer / Event-bus pattern.

Provides both a classic Observer (typed) and a lightweight EventBus.
"""

from __future__ import annotations

import weakref
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable
from typing import Any, Generic, TypeVar

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Classic Observer
# ---------------------------------------------------------------------------

class Observer(ABC, Generic[T]):
    """Abstract observer that receives typed updates."""

    @abstractmethod
    def update(self, event: str, data: T) -> None:
        """Called when the observed subject emits *event* with *data*."""


class Subject(Generic[T]):
    """
    Observable subject that notifies registered observers.

    Observers are stored as weak references so they don't prevent GC.
    """

    def __init__(self) -> None:
        self._observers: list[weakref.ref[Observer[T]]] = []

    def subscribe(self, observer: Observer[T]) -> None:
        """Register *observer*."""
        self._observers.append(weakref.ref(observer))

    def unsubscribe(self, observer: Observer[T]) -> None:
        """Deregister *observer*."""
        self._observers = [
            ref for ref in self._observers
            if ref() is not None and ref() is not observer
        ]

    def notify(self, event: str, data: T) -> None:
        """Notify all live observers with *event* and *data*."""
        dead: list[weakref.ref[Observer[T]]] = []
        for ref in self._observers:
            obs = ref()
            if obs is None:
                dead.append(ref)
            else:
                obs.update(event, data)
        for ref in dead:
            self._observers.remove(ref)


# ---------------------------------------------------------------------------
# Domain example: Stock price monitor
# ---------------------------------------------------------------------------

class StockSubject(Subject[float]):
    """Tracks a stock's price and notifies observers on change."""

    def __init__(self, ticker: str, price: float) -> None:
        super().__init__()
        self.ticker = ticker
        self._price = price

    @property
    def price(self) -> float:
        return self._price

    @price.setter
    def price(self, value: float) -> None:
        old = self._price
        self._price = value
        change_pct = (value - old) / old * 100
        self.notify("price_change", value)
        if abs(change_pct) >= 5:
            self.notify("large_move", value)


class AlertObserver(Observer[float]):
    """Prints an alert when a price change occurs."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.events: list[tuple[str, float]] = []

    def update(self, event: str, data: float) -> None:
        self.events.append((event, data))
        print(f"  [{self.name}] {event}: ${data:.2f}")


# ---------------------------------------------------------------------------
# Lightweight EventBus (publish-subscribe)
# ---------------------------------------------------------------------------

Handler = Callable[..., None]


class EventBus:
    """
    Simple synchronous publish-subscribe event bus.

    Handlers are regular callables (functions or methods).

    Example::

        bus = EventBus()

        @bus.on("user.created")
        def welcome(user_id: int) -> None:
            print(f"Welcome, user {user_id}!")

        bus.emit("user.created", user_id=42)
    """

    def __init__(self) -> None:
        self._handlers: defaultdict[str, list[Handler]] = defaultdict(list)

    def on(self, event: str) -> Callable[[Handler], Handler]:
        """Decorator to register a handler for *event*."""
        def decorator(fn: Handler) -> Handler:
            self._handlers[event].append(fn)
            return fn
        return decorator

    def subscribe(self, event: str, handler: Handler) -> None:
        """Register *handler* for *event*."""
        self._handlers[event].append(handler)

    def unsubscribe(self, event: str, handler: Handler) -> None:
        """Remove *handler* from *event*."""
        try:
            self._handlers[event].remove(handler)
        except ValueError:
            pass

    def emit(self, event: str, **kwargs: Any) -> None:
        """Dispatch *event* to all registered handlers."""
        for handler in list(self._handlers.get(event, [])):
            handler(**kwargs)

    def once(self, event: str, handler: Handler) -> None:
        """Register *handler* to fire only once for *event*."""
        def wrapper(**kwargs: Any) -> None:
            self.unsubscribe(event, wrapper)
            handler(**kwargs)
        self.subscribe(event, wrapper)

    def handler_count(self, event: str) -> int:
        """Return the number of handlers registered for *event*."""
        return len(self._handlers.get(event, []))


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def main() -> None:
    """Demonstrate observer patterns."""
    print("=== Classic Observer ===")
    stock = StockSubject("AAPL", 150.0)
    trader = AlertObserver("TraderBot")
    logger = AlertObserver("AuditLog")
    stock.subscribe(trader)
    stock.subscribe(logger)

    stock.price = 155.0  # +3.3% – only price_change
    stock.price = 140.0  # -9.7% – price_change + large_move

    stock.unsubscribe(logger)
    stock.price = 145.0  # only TraderBot notified
    print(f"  Trader events: {len(trader.events)}")

    print("\n=== EventBus ===")
    bus = EventBus()

    received: list[str] = []

    @bus.on("order.placed")
    def handle_order(order_id: int, amount: float) -> None:
        received.append(f"order {order_id}")
        print(f"  Order placed: #{order_id} for ${amount:.2f}")

    @bus.on("order.placed")
    def send_email(order_id: int, amount: float) -> None:
        print(f"  Sending confirmation email for order #{order_id}")

    bus.emit("order.placed", order_id=1001, amount=49.99)
    bus.emit("order.placed", order_id=1002, amount=9.99)
    print(f"  Received {len(received)} order events")


if __name__ == "__main__":
    main()
