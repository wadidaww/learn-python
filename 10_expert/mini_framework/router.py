"""
mini_framework/router.py
=========================
URL router with path parameter extraction and method matching.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Awaitable
from dataclasses import dataclass, field
from typing import Any


Handler = Callable[..., Awaitable[Any]]


@dataclass
class Route:
    """A single registered route."""

    method:  str
    pattern: re.Pattern[str]
    handler: Handler
    param_names: list[str]
    raw_path: str

    def match(self, method: str, path: str) -> dict[str, str] | None:
        """
        Return path parameters dict if *method* and *path* match, else None.
        """
        if self.method != method.upper() and self.method != "*":
            return None
        m = self.pattern.fullmatch(path)
        if m is None:
            return None
        return dict(zip(self.param_names, m.groups()))


class Router:
    """
    Path router that supports static paths and ``{param}`` placeholders.

    Example::

        router = Router()

        @router.get("/")
        async def index(request):
            ...

        @router.get("/users/{user_id}")
        async def get_user(request, user_id: str):
            ...
    """

    PARAM_RE = re.compile(r"\{(\w+)\}")

    def __init__(self) -> None:
        self._routes: list[Route] = []

    # ------------------------------------------------------------------
    # Registration helpers
    # ------------------------------------------------------------------

    def _compile(self, path: str) -> tuple[re.Pattern[str], list[str]]:
        """Convert a path template to a compiled regex + param name list."""
        param_names: list[str] = []
        parts = self.PARAM_RE.split(path)
        # PARAM_RE.split gives [literal, param1, literal, param2, ...]
        pattern_parts: list[str] = []
        for i, part in enumerate(parts):
            if i % 2 == 0:
                pattern_parts.append(re.escape(part))
            else:
                param_names.append(part)
                pattern_parts.append(r"([^/]+)")
        return re.compile("".join(pattern_parts)), param_names

    def add_route(self, method: str, path: str, handler: Handler) -> None:
        """Register *handler* for (*method*, *path*)."""
        pattern, param_names = self._compile(path)
        self._routes.append(
            Route(
                method=method.upper(),
                pattern=pattern,
                handler=handler,
                param_names=param_names,
                raw_path=path,
            )
        )

    def route(self, method: str, path: str) -> Callable[[Handler], Handler]:
        """Decorator to register a handler."""
        def decorator(fn: Handler) -> Handler:
            self.add_route(method, path, fn)
            return fn
        return decorator

    def get(self, path: str)    -> Callable[[Handler], Handler]: return self.route("GET",    path)
    def post(self, path: str)   -> Callable[[Handler], Handler]: return self.route("POST",   path)
    def put(self, path: str)    -> Callable[[Handler], Handler]: return self.route("PUT",    path)
    def delete(self, path: str) -> Callable[[Handler], Handler]: return self.route("DELETE", path)
    def patch(self, path: str)  -> Callable[[Handler], Handler]: return self.route("PATCH",  path)

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def resolve(
        self,
        method: str,
        path: str,
    ) -> tuple[Handler, dict[str, str]] | None:
        """
        Find the matching handler and extracted path params.

        Returns None if no route matches.
        """
        for route in self._routes:
            params = route.match(method, path)
            if params is not None:
                return route.handler, params
        return None
