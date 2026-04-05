"""
mini_framework/core.py
=======================
Async web framework core: Request, Response, and Application.

The framework is ASGI-compatible in spirit (uses async handlers) but
can also be driven directly without a real ASGI server for testing.
"""

from __future__ import annotations

import asyncio
import json as _json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from http import HTTPStatus
from typing import Any

from mini_framework.router import Router
from mini_framework.middleware import MiddlewareChain, timing_middleware, logging_middleware


# ---------------------------------------------------------------------------
# Request and Response
# ---------------------------------------------------------------------------

@dataclass
class Request:
    """Represents an incoming HTTP request."""

    method:    str
    path:      str
    headers:   dict[str, str] = field(default_factory=dict)
    body:      bytes          = field(default=b"")
    params:    dict[str, str] = field(default_factory=dict)
    query:     dict[str, str] = field(default_factory=dict)
    client_ip: str            = field(default="127.0.0.1")
    state:     dict[str, Any] = field(default_factory=dict)

    def json(self) -> Any:
        """Parse request body as JSON."""
        return _json.loads(self.body)

    def text(self) -> str:
        """Return request body as text."""
        return self.body.decode("utf-8")


@dataclass
class Response:
    """Represents an outgoing HTTP response."""

    status:  int              = 200
    body:    bytes            = field(default=b"")
    headers: dict[str, str]   = field(default_factory=dict)

    @classmethod
    def json(
        cls,
        data: Any,
        status: int = 200,
        headers: dict[str, str] | None = None,
    ) -> Response:
        """Convenience constructor for JSON responses."""
        body = _json.dumps(data, default=str).encode()
        h = {"Content-Type": "application/json", **(headers or {})}
        return cls(status=status, body=body, headers=h)

    @classmethod
    def text(
        cls,
        content: str,
        status: int = 200,
        headers: dict[str, str] | None = None,
    ) -> Response:
        """Convenience constructor for plain-text responses."""
        h = {"Content-Type": "text/plain; charset=utf-8", **(headers or {})}
        return cls(status=status, body=content.encode(), headers=h)

    @classmethod
    def html(
        cls,
        content: str,
        status: int = 200,
    ) -> Response:
        """Convenience constructor for HTML responses."""
        return cls(
            status=status,
            body=content.encode(),
            headers={"Content-Type": "text/html; charset=utf-8"},
        )

    @classmethod
    def redirect(cls, location: str, permanent: bool = False) -> Response:
        """Return a redirect response."""
        return cls(
            status=301 if permanent else 302,
            headers={"Location": location},
        )


# Type alias for async route handlers
Handler = Callable[[Request], Awaitable[Response]]


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

class Application:
    """
    The mini web framework application.

    Example::

        app = Application()

        @app.router.get("/")
        async def index(request: Request) -> Response:
            return Response.json({"hello": "world"})

        # Drive manually in tests:
        response = asyncio.run(app.handle(Request("GET", "/")))
    """

    def __init__(self) -> None:
        self.router = Router()
        self.middleware = MiddlewareChain()

    # ------------------------------------------------------------------
    # Built-in middleware shortcuts
    # ------------------------------------------------------------------

    def use_timing(self) -> None:
        self.middleware.add(timing_middleware)

    def use_logging(self) -> None:
        self.middleware.add(logging_middleware)

    # ------------------------------------------------------------------
    # Request handling
    # ------------------------------------------------------------------

    async def handle(self, request: Request) -> Response:
        """
        Process *request* through middleware and routing.
        """
        resolved = self.router.resolve(request.method, request.path)

        if resolved is None:
            # Check if path exists for a different method (405)
            async def not_found_handler(req: Request) -> Response:
                return Response.json(
                    {"error": f"Not Found: {req.path}"},
                    status=404,
                )
            core_handler: Handler = not_found_handler
        else:
            route_handler, path_params = resolved
            request.params = path_params

            async def bound_handler(req: Request) -> Response:
                result = await route_handler(req, **req.params)
                if isinstance(result, Response):
                    return result
                if isinstance(result, dict):
                    return Response.json(result)
                if isinstance(result, str):
                    return Response.text(result)
                return Response.json(result)

            core_handler = bound_handler

        # Wrap in middleware
        wrapped = self.middleware.wrap(core_handler)
        return await wrapped(request)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

async def demo() -> None:
    """Demonstrate the mini framework without a real server."""
    app = Application()
    app.use_logging()
    app.use_timing()

    @app.router.get("/")
    async def index(request: Request) -> Response:
        return Response.json({"message": "Hello from the mini framework!"})

    @app.router.get("/users/{user_id}")
    async def get_user(request: Request, user_id: str) -> Response:
        return Response.json({"user_id": user_id, "name": f"User {user_id}"})

    @app.router.post("/echo")
    async def echo(request: Request) -> Response:
        return Response.json({"echo": request.text()})

    print("=== Mini Framework Demo ===")
    requests = [
        Request("GET", "/"),
        Request("GET", "/users/42"),
        Request("GET", "/not-found"),
        Request("POST", "/echo", body=b"hello world"),
    ]

    for req in requests:
        resp = await app.handle(req)
        print(f"  {req.method} {req.path} → {resp.status}: {resp.body[:60].decode()!r}")


def main() -> None:
    asyncio.run(demo())


if __name__ == "__main__":
    main()
