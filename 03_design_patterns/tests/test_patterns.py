"""
tests/test_patterns.py
=======================
pytest tests for design pattern implementations.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import pytest

# Ensure 03_design_patterns is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from creational.singleton import ApplicationConfig, DatabasePool, SingletonMeta
from creational.factory import (
    EmailNotification,
    Message,
    SMSNotification,
    create_notification,
    render_login_form,
    WebUIFactory,
    MobileUIFactory,
)
from behavioral.observer import AlertObserver, EventBus, StockSubject
from behavioral.strategy import (
    CreditCardPayment,
    DataSorter,
    PercentageDiscount,
    FlatDiscount,
    NoDiscount,
    QuickSortStrategy,
    ShoppingCart,
)
from structural.decorator import (
    CachingDecorator,
    ConcreteComponent,
    LoggingDecorator,
    PricingDecorator,
    memoize,
    timer,
    validate_positive,
)
from di_container import Container, DIError, InMemoryDatabase, Logger, UserService


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

class TestSingleton:
    def setup_method(self) -> None:
        # Clear existing singleton instances for isolated tests
        ApplicationConfig._instances.clear()  # type: ignore[attr-defined]

    def test_same_instance(self) -> None:
        a = ApplicationConfig()
        b = ApplicationConfig()
        assert a is b

    def test_shared_state(self) -> None:
        a = ApplicationConfig()
        a.debug = True
        b = ApplicationConfig()
        assert b.debug is True

    def test_thread_safe(self) -> None:
        results: list[int] = []

        def create() -> None:
            results.append(id(ApplicationConfig()))

        threads = [threading.Thread(target=create) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(set(results)) == 1


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class TestFactory:
    def test_create_email(self) -> None:
        n = create_notification("email")
        assert isinstance(n, EmailNotification)
        assert n.channel() == "email"

    def test_create_sms(self) -> None:
        n = create_notification("sms")
        assert isinstance(n, SMSNotification)

    def test_unknown_channel(self) -> None:
        with pytest.raises(ValueError, match="Unknown channel"):
            create_notification("fax")

    def test_send_returns_true(self) -> None:
        n = create_notification("email")
        msg = Message("to@example.com", "Hi", "Body")
        assert n.send(msg) is True

    def test_abstract_factory_web(self) -> None:
        widgets = render_login_form(WebUIFactory())
        assert any("<input" in w for w in widgets)

    def test_abstract_factory_mobile(self) -> None:
        widgets = render_login_form(MobileUIFactory())
        assert any("UIButton" in w for w in widgets)


# ---------------------------------------------------------------------------
# Observer
# ---------------------------------------------------------------------------

class TestObserver:
    def test_notified_on_price_change(self) -> None:
        stock = StockSubject("TEST", 100.0)
        obs = AlertObserver("test")
        stock.subscribe(obs)
        stock.price = 110.0
        assert len(obs.events) >= 1

    def test_large_move_event(self) -> None:
        stock = StockSubject("TEST", 100.0)
        obs = AlertObserver("test")
        stock.subscribe(obs)
        stock.price = 50.0  # -50%
        events = [e[0] for e in obs.events]
        assert "large_move" in events

    def test_unsubscribe(self) -> None:
        stock = StockSubject("TEST", 100.0)
        obs = AlertObserver("test")
        stock.subscribe(obs)
        stock.unsubscribe(obs)
        stock.price = 200.0
        assert len(obs.events) == 0


class TestEventBus:
    def test_emit_calls_handler(self) -> None:
        bus = EventBus()
        calls: list[int] = []
        bus.subscribe("evt", lambda x: calls.append(x))
        bus.emit("evt", x=42)
        assert calls == [42]

    def test_unsubscribe(self) -> None:
        bus = EventBus()
        calls: list[int] = []
        handler = lambda: calls.append(1)  # noqa: E731
        bus.subscribe("evt", handler)
        bus.unsubscribe("evt", handler)
        bus.emit("evt")
        assert calls == []

    def test_once(self) -> None:
        bus = EventBus()
        calls: list[int] = []
        bus.once("evt", lambda: calls.append(1))
        bus.emit("evt")
        bus.emit("evt")
        assert calls == [1]


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

class TestStrategy:
    def test_credit_card_payment(self) -> None:
        p = CreditCardPayment("4111111111111234", "12/26", "123")
        cart = ShoppingCart()
        cart.add_item("thing", 50.0)
        cart.set_payment_strategy(p)
        result = cart.checkout()
        assert result.success

    def test_discount_percentage(self) -> None:
        d = PercentageDiscount(20)
        assert d.apply(100.0) == 80.0

    def test_discount_flat(self) -> None:
        d = FlatDiscount(15)
        assert d.apply(100.0) == 85.0

    def test_discount_no_negative(self) -> None:
        d = FlatDiscount(200)
        assert d.apply(50.0) == 0.0

    def test_sorting_strategy(self) -> None:
        sorter = DataSorter(QuickSortStrategy())
        assert sorter.sort([3, 1, 2]) == [1, 2, 3]


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------

class TestDecorator:
    def test_memoize_caches(self) -> None:
        call_count = 0

        @memoize
        def expensive(n: int) -> int:
            nonlocal call_count
            call_count += 1
            return n * 2

        expensive(5)
        expensive(5)
        expensive(5)
        assert call_count == 1

    def test_validate_positive(self) -> None:
        @validate_positive("x")
        def f(x: float) -> float:
            return x

        assert f(5) == 5
        with pytest.raises(ValueError):
            f(-1)

    def test_class_decorator_logging(self) -> None:
        log: list[str] = []
        comp = LoggingDecorator(ConcreteComponent(), log)
        comp.operation()
        assert len(log) == 1

    def test_pricing_decorator(self) -> None:
        comp = PricingDecorator(ConcreteComponent(), surcharge=5.0)
        assert comp.cost() == 15.0

    def test_caching_decorator(self) -> None:
        comp = CachingDecorator(ConcreteComponent())
        r1 = comp.operation()
        r2 = comp.operation()
        assert r1 == r2


# ---------------------------------------------------------------------------
# DI Container
# ---------------------------------------------------------------------------

class TestDIContainer:
    def test_resolve_singleton(self) -> None:
        c = Container()
        c.register_singleton(Logger)
        l1 = c.resolve(Logger)
        l2 = c.resolve(Logger)
        assert l1 is l2

    def test_resolve_transient(self) -> None:
        from di_container import Database, UserRepository
        c = Container()
        c.register_singleton(Logger)
        c.register_singleton(Database, InMemoryDatabase)
        c.register_transient(UserRepository)
        r1 = c.resolve(UserRepository)
        r2 = c.resolve(UserRepository)
        assert r1 is not r2

    def test_auto_wire(self) -> None:
        from di_container import Database, UserRepository
        c = Container()
        c.register_singleton(Logger)
        c.register_singleton(Database, InMemoryDatabase)
        c.register_transient(UserRepository)
        c.register_transient(UserService)
        svc = c.resolve(UserService)
        assert isinstance(svc, UserService)

    def test_missing_registration(self) -> None:
        c = Container()
        with pytest.raises(DIError):
            c.resolve(Logger)
