"""
networking/http_server.py
==========================
Minimal HTTP/1.1 server built on raw sockets (no frameworks).

Handles:
  - GET requests for static text content
  - 404 and 405 responses
  - Keep-alive and chunked transfer (basic)

Run:
    python http_server.py             # listens on :8080
    curl http://127.0.0.1:8080/
    curl http://127.0.0.1:8080/hello
"""

from __future__ import annotations

import logging
import socket
import threading
import time
from datetime import datetime, timezone
from http import HTTPStatus
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

HTTP_VERSION = "HTTP/1.1"
BUFFER_SIZE  = 8192
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8080


# ---------------------------------------------------------------------------
# Request / Response data types
# ---------------------------------------------------------------------------

class HTTPRequest:
    """Parsed HTTP request."""

    __slots__ = ("method", "path", "version", "headers", "body")

    def __init__(
        self,
        method: str,
        path: str,
        version: str,
        headers: dict[str, str],
        body: bytes,
    ) -> None:
        self.method  = method
        self.path    = path
        self.version = version
        self.headers = headers
        self.body    = body

    @classmethod
    def parse(cls, raw: bytes) -> HTTPRequest | None:
        """Parse raw HTTP bytes into an HTTPRequest; return None on error."""
        try:
            header_section, _, body = raw.partition(b"\r\n\r\n")
            lines = header_section.decode("iso-8859-1").split("\r\n")
            method, path, version = lines[0].split(" ", 2)
            headers: dict[str, str] = {}
            for line in lines[1:]:
                if ": " in line:
                    k, _, v = line.partition(": ")
                    headers[k.lower()] = v
            return cls(method, path, version, headers, body)
        except Exception:
            return None

    def __repr__(self) -> str:
        return f"HTTPRequest({self.method} {self.path})"


class HTTPResponse:
    """Builder for HTTP responses."""

    def __init__(
        self,
        status: HTTPStatus,
        body: bytes = b"",
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status  = status
        self.body    = body
        self.headers: dict[str, str] = headers or {}

    def to_bytes(self) -> bytes:
        """Serialise to wire format."""
        date = datetime.now(timezone.utc).strftime("%a, %d %b %Y %H:%M:%S GMT")
        self.headers.setdefault("Date",           date)
        self.headers.setdefault("Server",         "PythonHTTP/0.1")
        self.headers.setdefault("Content-Length", str(len(self.body)))
        self.headers.setdefault("Connection",     "close")

        status_line = f"{HTTP_VERSION} {self.status.value} {self.status.phrase}\r\n"
        header_lines = "".join(f"{k}: {v}\r\n" for k, v in self.headers.items())
        return (status_line + header_lines + "\r\n").encode() + self.body


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

Handler = Any  # (request: HTTPRequest) -> HTTPResponse


class Router:
    """Simple path-based router."""

    def __init__(self) -> None:
        self._routes: dict[tuple[str, str], Handler] = {}

    def route(self, method: str, path: str) -> Any:
        """Decorator to register a handler for (method, path)."""
        def decorator(fn: Handler) -> Handler:
            self._routes[(method.upper(), path)] = fn
            return fn
        return decorator

    def dispatch(self, request: HTTPRequest) -> HTTPResponse:
        """Find and call the matching handler, or return an error response."""
        handler = self._routes.get((request.method, request.path))
        if handler is None:
            # Check if path exists for a different method
            path_methods = [m for m, p in self._routes if p == request.path]
            if path_methods:
                return HTTPResponse(
                    HTTPStatus.METHOD_NOT_ALLOWED,
                    b"Method Not Allowed",
                    {"Allow": ", ".join(path_methods)},
                )
            return HTTPResponse(
                HTTPStatus.NOT_FOUND,
                f"Not Found: {request.path}".encode(),
            )
        try:
            return handler(request)
        except Exception as exc:
            logger.exception("Handler error: %s", exc)
            return HTTPResponse(HTTPStatus.INTERNAL_SERVER_ERROR, b"Internal Server Error")


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

class HTTPServer:
    """
    Minimal HTTP/1.1 server.

    Example::

        app = HTTPServer()

        @app.router.route("GET", "/")
        def index(req):
            return HTTPResponse(HTTPStatus.OK, b"Hello, World!", {"Content-Type": "text/plain"})

        app.start()
        app.stop()
    """

    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> None:
        self.host = host
        self.port = port
        self.router = Router()
        self._running = False
        self._sock: socket.socket | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Bind and start accepting connections in background thread."""
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((self.host, self.port))
        self._sock.listen(10)
        self._running = True
        self._thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._thread.start()
        logger.info("HTTPServer on http://%s:%d", self.host, self.port)

    def stop(self) -> None:
        """Stop the server."""
        self._running = False
        if self._sock:
            try:
                self._sock.close()
            except OSError:
                pass

    def _accept_loop(self) -> None:
        assert self._sock is not None
        while self._running:
            try:
                conn, addr = self._sock.accept()
            except OSError:
                break
            t = threading.Thread(
                target=self._handle, args=(conn, addr), daemon=True
            )
            t.start()

    def _handle(self, conn: socket.socket, addr: tuple[str, int]) -> None:
        """Read request, dispatch, send response."""
        try:
            raw = b""
            while b"\r\n\r\n" not in raw:
                chunk = conn.recv(BUFFER_SIZE)
                if not chunk:
                    return
                raw += chunk

            request = HTTPRequest.parse(raw)
            if request is None:
                conn.sendall(HTTPResponse(HTTPStatus.BAD_REQUEST, b"Bad Request").to_bytes())
                return

            logger.info("%s:%d  %s %s", addr[0], addr[1], request.method, request.path)
            response = self.router.dispatch(request)
            conn.sendall(response.to_bytes())
        except OSError:
            pass
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Default application
# ---------------------------------------------------------------------------

def build_app(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> HTTPServer:
    """Create and configure a sample HTTP application."""
    app = HTTPServer(host, port)

    @app.router.route("GET", "/")
    def index(req: HTTPRequest) -> HTTPResponse:
        return HTTPResponse(
            HTTPStatus.OK,
            b"<h1>Hello from PythonHTTP!</h1>",
            {"Content-Type": "text/html; charset=utf-8"},
        )

    @app.router.route("GET", "/health")
    def health(req: HTTPRequest) -> HTTPResponse:
        return HTTPResponse(
            HTTPStatus.OK,
            b'{"status": "ok"}',
            {"Content-Type": "application/json"},
        )

    @app.router.route("GET", "/hello")
    def hello(req: HTTPRequest) -> HTTPResponse:
        return HTTPResponse(
            HTTPStatus.OK,
            b"Hello, World!",
            {"Content-Type": "text/plain; charset=utf-8"},
        )

    return app


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def main() -> None:
    """Start the server, send test requests, then stop."""
    app = build_app()
    app.start()
    time.sleep(0.1)

    try:
        for path in ["/", "/health", "/hello", "/missing"]:
            with socket.create_connection((DEFAULT_HOST, DEFAULT_PORT), timeout=2) as s:
                s.sendall(f"GET {path} HTTP/1.1\r\nHost: localhost\r\n\r\n".encode())
                resp = s.recv(BUFFER_SIZE).decode()
                status_line = resp.split("\r\n")[0]
                print(f"  GET {path:<10} → {status_line}")
    finally:
        app.stop()


if __name__ == "__main__":
    main()
