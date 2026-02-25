"""
creational/factory.py
======================
Factory Method and Abstract Factory patterns.

Example domain: notification senders (Email, SMS, Push).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


# ---------------------------------------------------------------------------
# Product interfaces
# ---------------------------------------------------------------------------

@dataclass
class Message:
    """A message to be sent via some channel."""

    recipient: str
    subject: str
    body: str


class Notification(ABC):
    """Abstract product: a notification sender."""

    @abstractmethod
    def send(self, message: Message) -> bool:
        """Send *message*; return True on success."""

    @abstractmethod
    def channel(self) -> str:
        """Return the name of the notification channel."""


# ---------------------------------------------------------------------------
# Concrete products
# ---------------------------------------------------------------------------

class EmailNotification(Notification):
    """Sends notifications via email (simulated)."""

    def __init__(self, smtp_host: str = "localhost", port: int = 587) -> None:
        self._host = smtp_host
        self._port = port

    def send(self, message: Message) -> bool:
        print(
            f"[EMAIL] {self._host}:{self._port} → {message.recipient} | "
            f"{message.subject}"
        )
        return True

    def channel(self) -> str:
        return "email"


class SMSNotification(Notification):
    """Sends notifications via SMS (simulated)."""

    def __init__(self, api_key: str = "dummy-key") -> None:
        self._api_key = api_key

    def send(self, message: Message) -> bool:
        body_preview = message.body[:40]
        print(f"[SMS] → {message.recipient} | {body_preview!r}")
        return True

    def channel(self) -> str:
        return "sms"


class PushNotification(Notification):
    """Sends push notifications (simulated)."""

    def send(self, message: Message) -> bool:
        print(f"[PUSH] → {message.recipient} | {message.subject}")
        return True

    def channel(self) -> str:
        return "push"


# ---------------------------------------------------------------------------
# Factory Method
# ---------------------------------------------------------------------------

class NotificationFactory(ABC):
    """Abstract factory method for creating notifications."""

    @abstractmethod
    def create_notification(self) -> Notification:
        """Create and return a Notification instance."""

    def notify(self, message: Message) -> bool:
        """Template method: create → send."""
        notifier = self.create_notification()
        return notifier.send(message)


class EmailNotificationFactory(NotificationFactory):
    def __init__(self, smtp_host: str = "smtp.example.com") -> None:
        self._host = smtp_host

    def create_notification(self) -> Notification:
        return EmailNotification(self._host)


class SMSNotificationFactory(NotificationFactory):
    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    def create_notification(self) -> Notification:
        return SMSNotification(self._api_key)


# ---------------------------------------------------------------------------
# Registry-based Simple Factory
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, type[Notification]] = {
    "email": EmailNotification,
    "sms":   SMSNotification,
    "push":  PushNotification,
}


def create_notification(channel: str, **kwargs: Any) -> Notification:
    """
    Registry-based factory function.

    Args:
        channel: One of 'email', 'sms', 'push'.
        **kwargs: Passed to the concrete class constructor.

    Raises:
        ValueError: If *channel* is unknown.
    """
    cls = _REGISTRY.get(channel)
    if cls is None:
        raise ValueError(
            f"Unknown channel {channel!r}. Available: {sorted(_REGISTRY)}"
        )
    return cls(**kwargs)


def register_channel(name: str, cls: type[Notification]) -> None:
    """Register a new notification channel type."""
    _REGISTRY[name] = cls


# ---------------------------------------------------------------------------
# Abstract Factory – UI component family
# ---------------------------------------------------------------------------

class Button(ABC):
    @abstractmethod
    def render(self) -> str: ...

    @abstractmethod
    def on_click(self) -> str: ...


class TextInput(ABC):
    @abstractmethod
    def render(self) -> str: ...


class WebButton(Button):
    def render(self) -> str:
        return "<button class='btn'>Click me</button>"

    def on_click(self) -> str:
        return "window.handleClick()"


class WebTextInput(TextInput):
    def render(self) -> str:
        return "<input type='text' />"


class MobileButton(Button):
    def render(self) -> str:
        return "UIButton(frame: CGRect(...))"

    def on_click(self) -> str:
        return "handleTap()"


class MobileTextInput(TextInput):
    def render(self) -> str:
        return "UITextField(frame: CGRect(...))"


class UIFactory(ABC):
    """Abstract Factory for UI components."""

    @abstractmethod
    def create_button(self) -> Button: ...

    @abstractmethod
    def create_text_input(self) -> TextInput: ...


class WebUIFactory(UIFactory):
    def create_button(self) -> Button:
        return WebButton()

    def create_text_input(self) -> TextInput:
        return WebTextInput()


class MobileUIFactory(UIFactory):
    def create_button(self) -> Button:
        return MobileButton()

    def create_text_input(self) -> TextInput:
        return MobileTextInput()


def render_login_form(factory: UIFactory) -> list[str]:
    """Render a login form using the given UI factory."""
    button = factory.create_button()
    text_input = factory.create_text_input()
    return [text_input.render(), button.render()]


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def main() -> None:
    """Demonstrate factory patterns."""
    msg = Message("alice@example.com", "Hello", "Welcome to the platform!")

    print("=== Factory Method ===")
    for factory in [EmailNotificationFactory(), SMSNotificationFactory("key-123")]:
        factory.notify(msg)

    print("\n=== Registry Factory ===")
    for channel in ["email", "sms", "push"]:
        n = create_notification(channel)
        n.send(msg)

    print("\n=== Abstract Factory (UI) ===")
    for factory_cls in [WebUIFactory, MobileUIFactory]:
        factory = factory_cls()
        widgets = render_login_form(factory)
        print(f"\n{factory_cls.__name__}:")
        for w in widgets:
            print(f"  {w}")


if __name__ == "__main__":
    main()
