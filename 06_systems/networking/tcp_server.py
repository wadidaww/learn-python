"""
networking/tcp_server.py
=========================
A simple echo TCP server using Python's socket module.

Supports multiple concurrent clients via threading.

Run:
    # Server (terminal 1)
    python tcp_server.py

    # Client (terminal 2)
    python -c "import socket; s=socket.create_connection(('127.0.0.1',9000)); s.sendall(b'Hello'); print(s.recv(1024))"
"""

from __future__ import annotations

import logging
import socket
import threading
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 9000
BUFFER_SIZE  = 4096


class TCPServer:
    """
    Multi-threaded TCP echo server.

    Each accepted connection is handled in its own thread.
    The server can be stopped gracefully by calling :meth:`stop`.

    Example::

        server = TCPServer("127.0.0.1", 9000)
        server.start()          # non-blocking; spawns listener thread
        server.stop()
    """

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        backlog: int = 5,
    ) -> None:
        self.host = host
        self.port = port
        self.backlog = backlog
        self._running = False
        self._server_socket: socket.socket | None = None
        self._clients: list[socket.socket] = []
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Bind the server and start accepting connections in a background thread."""
        self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_socket.bind((self.host, self.port))
        self._server_socket.listen(self.backlog)
        self._running = True
        self._thread = threading.Thread(target=self._accept_loop, daemon=True)
        self._thread.start()
        logger.info("TCPServer listening on %s:%d", self.host, self.port)

    def stop(self) -> None:
        """Stop the server and close all connections."""
        self._running = False
        if self._server_socket:
            try:
                self._server_socket.close()
            except OSError:
                pass
        with self._lock:
            for client in list(self._clients):
                try:
                    client.close()
                except OSError:
                    pass
            self._clients.clear()
        logger.info("TCPServer stopped")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _accept_loop(self) -> None:
        """Main accept loop – runs in its own thread."""
        assert self._server_socket is not None
        while self._running:
            try:
                conn, addr = self._server_socket.accept()
            except OSError:
                break
            with self._lock:
                self._clients.append(conn)
            logger.info("Client connected: %s:%d", *addr)
            t = threading.Thread(
                target=self._handle_client,
                args=(conn, addr),
                daemon=True,
            )
            t.start()

    def _handle_client(
        self,
        conn: socket.socket,
        addr: tuple[str, int],
    ) -> None:
        """Echo all received bytes back to the client."""
        try:
            while self._running:
                data = conn.recv(BUFFER_SIZE)
                if not data:
                    break
                response = self._process(data)
                conn.sendall(response)
        except OSError:
            pass
        finally:
            with self._lock:
                try:
                    self._clients.remove(conn)
                except ValueError:
                    pass
            conn.close()
            logger.info("Client disconnected: %s:%d", *addr)

    def _process(self, data: bytes) -> bytes:
        """Process incoming data. Override in subclasses for custom behaviour."""
        return b"ECHO: " + data


class UpperCaseTCPServer(TCPServer):
    """A TCP server that upper-cases all incoming text."""

    def _process(self, data: bytes) -> bytes:
        return data.upper()


def simple_client(host: str, port: int, message: str) -> str:
    """Connect to *host*:*port*, send *message*, and return the response."""
    with socket.create_connection((host, port), timeout=5) as sock:
        sock.sendall(message.encode())
        return sock.recv(BUFFER_SIZE).decode()


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def main() -> None:
    """Start the server, send test messages, then shut down."""
    import time

    server = TCPServer(DEFAULT_HOST, DEFAULT_PORT)
    server.start()
    time.sleep(0.1)  # give the server a moment to bind

    try:
        for msg in ["hello", "world", "python"]:
            response = simple_client(DEFAULT_HOST, DEFAULT_PORT, msg)
            print(f"  sent={msg!r:<10}  recv={response!r}")
    finally:
        server.stop()


if __name__ == "__main__":
    main()
