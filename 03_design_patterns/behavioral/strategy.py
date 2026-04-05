"""
behavioral/strategy.py
=======================
Strategy pattern: encapsulate interchangeable algorithms behind a common interface.

Examples:
  - Payment processing (CreditCard, PayPal, Crypto)
  - Sorting strategies
  - Discount calculation strategies
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Payment strategies
# ---------------------------------------------------------------------------

@dataclass
class PaymentResult:
    """Result of a payment attempt."""

    success: bool
    transaction_id: str
    message: str


class PaymentStrategy(ABC):
    """Abstract payment strategy."""

    @abstractmethod
    def pay(self, amount: float, currency: str = "USD") -> PaymentResult:
        """Execute a payment of *amount* in *currency*."""

    @abstractmethod
    def name(self) -> str:
        """Return the strategy's display name."""


class CreditCardPayment(PaymentStrategy):
    """Pay using a credit card."""

    def __init__(self, card_number: str, expiry: str, cvv: str) -> None:
        # Store only last 4 digits
        self._last4 = card_number[-4:]
        self._expiry = expiry
        self._cvv = cvv  # would be hashed in production

    def pay(self, amount: float, currency: str = "USD") -> PaymentResult:
        # Simulate payment processing
        return PaymentResult(
            success=True,
            transaction_id=f"CC-{self._last4}-{int(amount * 100)}",
            message=f"Charged ${amount:.2f} {currency} to card ending {self._last4}",
        )

    def name(self) -> str:
        return f"CreditCard (*{self._last4})"


class PayPalPayment(PaymentStrategy):
    """Pay via PayPal."""

    def __init__(self, email: str) -> None:
        self._email = email

    def pay(self, amount: float, currency: str = "USD") -> PaymentResult:
        return PaymentResult(
            success=True,
            transaction_id=f"PP-{hash(self._email) % 100000:05d}",
            message=f"PayPal payment of ${amount:.2f} from {self._email}",
        )

    def name(self) -> str:
        return f"PayPal ({self._email})"


class CryptoPayment(PaymentStrategy):
    """Pay with cryptocurrency."""

    def __init__(self, wallet: str, coin: str = "BTC") -> None:
        self._wallet = wallet
        self._coin = coin

    def pay(self, amount: float, currency: str = "USD") -> PaymentResult:
        return PaymentResult(
            success=True,
            transaction_id=f"CRYPTO-{self._coin}-{abs(hash(self._wallet)) % 10**8:08d}",
            message=f"${amount:.2f} {currency} sent as {self._coin} to {self._wallet[:8]}…",
        )

    def name(self) -> str:
        return f"Crypto ({self._coin})"


@dataclass
class ShoppingCart:
    """Shopping cart that delegates payment to a strategy."""

    items: list[tuple[str, float]] = field(default_factory=list)
    _payment_strategy: PaymentStrategy | None = field(default=None, repr=False)

    def add_item(self, name: str, price: float) -> None:
        self.items.append((name, price))

    @property
    def total(self) -> float:
        return sum(price for _, price in self.items)

    def set_payment_strategy(self, strategy: PaymentStrategy) -> None:
        """Switch payment strategy at runtime."""
        self._payment_strategy = strategy

    def checkout(self) -> PaymentResult:
        """Process payment with the configured strategy."""
        if self._payment_strategy is None:
            raise RuntimeError("No payment strategy configured")
        return self._payment_strategy.pay(self.total)


# ---------------------------------------------------------------------------
# Sorting strategies
# ---------------------------------------------------------------------------

class SortStrategy(ABC):
    """Abstract sorting strategy."""

    @abstractmethod
    def sort(self, data: list[Any]) -> list[Any]:
        """Return a sorted copy of *data*."""


class BubbleSortStrategy(SortStrategy):
    """O(n²) bubble sort — good for educational purposes."""

    def sort(self, data: list[Any]) -> list[Any]:
        arr = list(data)
        n = len(arr)
        for i in range(n):
            for j in range(n - i - 1):
                if arr[j] > arr[j + 1]:
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]
        return arr


class QuickSortStrategy(SortStrategy):
    """O(n log n) average quicksort."""

    def sort(self, data: list[Any]) -> list[Any]:
        if len(data) <= 1:
            return list(data)
        pivot = data[len(data) // 2]
        left   = [x for x in data if x < pivot]
        middle = [x for x in data if x == pivot]
        right  = [x for x in data if x > pivot]
        return self.sort(left) + middle + self.sort(right)


class BuiltinSortStrategy(SortStrategy):
    """Timsort via Python's built-in sorted()."""

    def sort(self, data: list[Any]) -> list[Any]:
        return sorted(data)


class DataSorter:
    """Sorts data using the configured strategy."""

    def __init__(self, strategy: SortStrategy) -> None:
        self._strategy = strategy

    @property
    def strategy(self) -> SortStrategy:
        return self._strategy

    @strategy.setter
    def strategy(self, s: SortStrategy) -> None:
        self._strategy = s

    def sort(self, data: list[Any]) -> list[Any]:
        return self._strategy.sort(data)


# ---------------------------------------------------------------------------
# Discount strategies
# ---------------------------------------------------------------------------

class DiscountStrategy(ABC):
    @abstractmethod
    def apply(self, price: float) -> float:
        """Return discounted price."""


class NoDiscount(DiscountStrategy):
    def apply(self, price: float) -> float:
        return price


class PercentageDiscount(DiscountStrategy):
    def __init__(self, percent: float) -> None:
        self._percent = percent

    def apply(self, price: float) -> float:
        return round(price * (1 - self._percent / 100), 2)


class FlatDiscount(DiscountStrategy):
    def __init__(self, amount: float) -> None:
        self._amount = amount

    def apply(self, price: float) -> float:
        return max(0.0, round(price - self._amount, 2))


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def main() -> None:
    """Demonstrate strategy patterns."""
    print("=== Payment Strategies ===")
    cart = ShoppingCart()
    cart.add_item("Widget A", 29.99)
    cart.add_item("Widget B", 14.99)
    print(f"Total: ${cart.total:.2f}")

    for strategy in [
        CreditCardPayment("4111111111111234", "12/26", "123"),
        PayPalPayment("alice@example.com"),
        CryptoPayment("1A2b3C4d5E6f7G8h", "BTC"),
    ]:
        cart.set_payment_strategy(strategy)
        result = cart.checkout()
        print(f"  {strategy.name()}: {result.message}")

    print("\n=== Sorting Strategies ===")
    data = [5, 2, 8, 1, 9, 3]
    sorter = DataSorter(BubbleSortStrategy())
    for strat in [BubbleSortStrategy(), QuickSortStrategy(), BuiltinSortStrategy()]:
        sorter.strategy = strat
        print(f"  {type(strat).__name__}: {sorter.sort(data)}")

    print("\n=== Discount Strategies ===")
    price = 100.0
    for d in [NoDiscount(), PercentageDiscount(20), FlatDiscount(15)]:
        print(f"  {type(d).__name__}: ${d.apply(price):.2f}")


if __name__ == "__main__":
    main()
