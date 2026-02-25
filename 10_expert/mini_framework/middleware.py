"""
mini_framework/middleware.py
=============================
Middleware chain implementation for the mini framework.

Middleware is a callable with the signature:
    async def middleware(request: Request, call_next: Handler) -> Response

The chain is built from outer (first applied) to inner (last applied).
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from typing import Any


# Type aliases defined here to avoid circular import with core.py
# We use forward references for Request/Response
Handler = Callable[..., Awaitable[Any]]
Middleware = Callable[..., Awaitable[Any]]


class MiddlewareChain:
    """
    Builds and executes a chain of middleware around a final handler.

    Middleware functions have the signature:
        async def mw(request, call_next) -> Response
    """

    def __init__(self) -> None:
        self._middleware: list[Middleware] = []

    def add(self, middleware: Middleware) -> None:
        """Append a middleware to the chain."""
        self._middleware.append(middleware)

    def wrap(self, handler: Handler) -> Handler:
        """
        Wrap *handler* with all registered middleware.

        The first middleware added is the outermost (called first).
        """
        for mw in reversed(self._middleware):
            _current_handler = handler  # closure capture

            async def make_next(request: Any, mw: Middleware = mw, h: Handler = _current_handler) -> Any:
                async def call_next(req: Any) -> Any:
                    return await h(req)
                return await mw(request, call_next)

            handler = make_next
        return handler


# ---------------------------------------------------------------------------
# Built-in middleware
# ---------------------------------------------------------------------------

async def timing_middleware(request: Any, call_next: Handler) -> Any:
    """Record request processing time in the response headers."""
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    if hasattr(response, "headers") and isinstance(response.headers, dict):
        response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
    return response


async def logging_middleware(request: Any, call_next: Handler) -> Any:
    """Log each request to stdout."""
    method = getattr(request, "method", "?")
    path   = getattr(request, "path", "?")
    print(f"  --> {method} {path}")
    response = await call_next(request)
    status = getattr(response, "status", "?")
    print(f"  <-- {method} {path} {status}")
    return response


def cors_middleware(
    allow_origins: list[str] | None = None,
    allow_methods: list[str] | None = None,
) -> Middleware:
    """
    Factory that returns a CORS middleware.

    Args:
        allow_origins: Allowed origins (default: ["*"]).
        allow_methods: Allowed HTTP methods.
    """
    origins = allow_origins or ["*"]
    methods = allow_methods or ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"]

    async def _cors(request: Any, call_next: Handler) -> Any:
        response = await call_next(request)
        if hasattr(response, "headers"):
            response.headers["Access-Control-Allow-Origin"]  = ", ".join(origins)
            response.headers["Access-Control-Allow-Methods"] = ", ".join(methods)
        return response

    return _cors


def rate_limit_middleware(max_requests: int, window_seconds: float) -> Middleware:
    """
    Simple in-memory rate limiter (IP-based).

    Args:
        max_requests:   Maximum requests per *window_seconds*.
        window_seconds: Rolling window size in seconds.
    """
    _counts: dict[str, list[float]] = {}

    async def _rate_limit(request: Any, call_next: Handler) -> Any:
        client_ip = getattr(request, "client_ip", "unknown")
        now = time.time()

        timestamps = _counts.get(client_ip, [])
        timestamps = [ts for ts in timestamps if now - ts < window_seconds]

        if len(timestamps) >= max_requests:
            # Return 429 without a real Response class (use a dict)
            return type("Response", (), {
                "status": 429,
                "body": b"Too Many Requests",
                "headers": {},
            })()

        timestamps.append(now)
        _counts[client_ip] = timestamps
        return await call_next(request)

    return _rate_limit
