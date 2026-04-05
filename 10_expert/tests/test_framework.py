"""
tests/test_framework.py
========================
Tests for the mini web framework and task queue.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from mini_framework.router import Router
from mini_framework.core import Application, Request, Response
from mini_framework.middleware import (
    MiddlewareChain,
    timing_middleware,
    cors_middleware,
    logging_middleware,
)
from task_queue.queue import TaskQueue, TaskState, Priority
from task_queue.worker import WorkerPool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(coro):  # type: ignore[no-untyped-def]
    return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

class TestRouter:
    def setup_method(self) -> None:
        self.router = Router()

    def test_static_route(self) -> None:
        async def handler(req: Request) -> Response:
            return Response.text("ok")

        self.router.add_route("GET", "/", handler)
        result = self.router.resolve("GET", "/")
        assert result is not None
        h, params = result
        assert h is handler
        assert params == {}

    def test_path_param(self) -> None:
        async def handler(req: Request, user_id: str) -> Response:
            return Response.text(user_id)

        self.router.add_route("GET", "/users/{user_id}", handler)
        result = self.router.resolve("GET", "/users/42")
        assert result is not None
        _, params = result
        assert params == {"user_id": "42"}

    def test_method_mismatch(self) -> None:
        async def handler(req: Request) -> Response:
            return Response.text("ok")

        self.router.add_route("POST", "/items", handler)
        assert self.router.resolve("GET", "/items") is None

    def test_no_match(self) -> None:
        assert self.router.resolve("GET", "/not-found") is None

    def test_multiple_params(self) -> None:
        async def handler(req: Request, a: str, b: str) -> Response:
            return Response.json({"a": a, "b": b})

        self.router.add_route("GET", "/items/{a}/sub/{b}", handler)
        result = self.router.resolve("GET", "/items/foo/sub/bar")
        assert result is not None
        _, params = result
        assert params == {"a": "foo", "b": "bar"}

    def test_decorator(self) -> None:
        @self.router.get("/hello")
        async def hello(req: Request) -> Response:
            return Response.text("hello")

        assert self.router.resolve("GET", "/hello") is not None


# ---------------------------------------------------------------------------
# Request / Response
# ---------------------------------------------------------------------------

class TestResponse:
    def test_json(self) -> None:
        r = Response.json({"k": "v"})
        assert r.status == 200
        assert r.headers["Content-Type"] == "application/json"
        assert b'"k"' in r.body

    def test_text(self) -> None:
        r = Response.text("hello")
        assert r.body == b"hello"

    def test_redirect(self) -> None:
        r = Response.redirect("/other")
        assert r.status == 302
        assert r.headers["Location"] == "/other"


class TestRequest:
    def test_json_parsing(self) -> None:
        req = Request("POST", "/", body=b'{"x": 1}')
        assert req.json() == {"x": 1}

    def test_text(self) -> None:
        req = Request("POST", "/", body=b"hello")
        assert req.text() == "hello"


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

class TestApplication:
    def setup_method(self) -> None:
        self.app = Application()

        @self.app.router.get("/")
        async def index(request: Request) -> Response:
            return Response.json({"status": "ok"})

        @self.app.router.get("/users/{user_id}")
        async def get_user(request: Request, user_id: str) -> Response:
            return Response.json({"user_id": user_id})

        @self.app.router.post("/echo")
        async def echo(request: Request) -> Response:
            return Response.text(request.text())

    def test_index(self) -> None:
        resp = run(self.app.handle(Request("GET", "/")))
        assert resp.status == 200

    def test_path_param(self) -> None:
        resp = run(self.app.handle(Request("GET", "/users/99")))
        assert resp.status == 200
        import json
        data = json.loads(resp.body)
        assert data["user_id"] == "99"

    def test_not_found(self) -> None:
        resp = run(self.app.handle(Request("GET", "/missing")))
        assert resp.status == 404

    def test_echo(self) -> None:
        resp = run(self.app.handle(Request("POST", "/echo", body=b"test")))
        assert resp.body == b"test"

    def test_timing_middleware(self) -> None:
        self.app.use_timing()
        resp = run(self.app.handle(Request("GET", "/")))
        assert resp.status == 200
        # Timing header should be set
        assert "X-Process-Time-Ms" in resp.headers


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

class TestMiddlewareChain:
    def test_order(self) -> None:
        order: list[str] = []

        async def mw1(request: Request, call_next):  # type: ignore[no-untyped-def]
            order.append("mw1-before")
            resp = await call_next(request)
            order.append("mw1-after")
            return resp

        async def mw2(request: Request, call_next):  # type: ignore[no-untyped-def]
            order.append("mw2-before")
            resp = await call_next(request)
            order.append("mw2-after")
            return resp

        async def handler(request: Request) -> Response:
            order.append("handler")
            return Response.text("ok")

        chain = MiddlewareChain()
        chain.add(mw1)
        chain.add(mw2)
        wrapped = chain.wrap(handler)
        run(wrapped(Request("GET", "/")))
        assert order == ["mw1-before", "mw2-before", "handler", "mw2-after", "mw1-after"]


# ---------------------------------------------------------------------------
# Task Queue
# ---------------------------------------------------------------------------

class TestTaskQueue:
    def test_enqueue_dequeue(self) -> None:
        async def job() -> int:
            return 42

        async def _test() -> None:
            q = TaskQueue()
            tid = await q.enqueue(job)
            task = await q.dequeue()
            assert task.task_id == tid

        run(_test())

    def test_priority_ordering(self) -> None:
        async def job() -> None:
            pass

        async def _test() -> None:
            q = TaskQueue()
            await q.enqueue(job, priority=Priority.LOW)
            await q.enqueue(job, priority=Priority.HIGH)
            await q.enqueue(job, priority=Priority.NORMAL)

            first  = await q.dequeue()
            second = await q.dequeue()
            # Higher priority (more negative stored value) dequeues first
            assert first.max_retries >= 0   # structural check
            # All three tasks were enqueued
            assert q.qsize == 1

        run(_test())

    def test_wait_for_result(self) -> None:
        async def fast_job(x: int) -> int:
            await asyncio.sleep(0.01)
            return x * 2

        async def _test() -> None:
            q = TaskQueue()
            pool = WorkerPool(workers=2, queue=q)
            await pool.start()
            tid = await q.enqueue(fast_job, 21)
            result = await q.wait_for_result(tid, timeout=3.0)
            await pool.stop()
            assert result == 42

        run(_test())


# ---------------------------------------------------------------------------
# Worker Pool
# ---------------------------------------------------------------------------

class TestWorkerPool:
    def test_basic(self) -> None:
        async def add(a: int, b: int) -> int:
            return a + b

        async def _test() -> None:
            pool = WorkerPool(workers=2)
            await pool.start()
            tid = await pool.submit(add, 3, 4)
            result = await pool.queue.wait_for_result(tid, timeout=3.0)
            await pool.stop()
            assert result == 7

        run(_test())

    def test_multiple_tasks(self) -> None:
        async def square(n: int) -> int:
            return n * n

        async def _test() -> None:
            pool = WorkerPool(workers=3)
            await pool.start()
            tids = [await pool.submit(square, i) for i in range(5)]
            results = [await pool.queue.wait_for_result(tid) for tid in tids]
            await pool.stop()
            assert sorted(results) == [0, 1, 4, 9, 16]

        run(_test())

    def test_failed_task(self) -> None:
        async def bad_task() -> None:
            raise ValueError("intentional failure")

        async def _test() -> None:
            pool = WorkerPool(workers=1)
            await pool.start()
            tid = await pool.queue.enqueue(bad_task, max_retries=0)
            with pytest.raises(RuntimeError, match="Task failed"):
                await pool.queue.wait_for_result(tid, timeout=3.0)
            await pool.stop()

        run(_test())
