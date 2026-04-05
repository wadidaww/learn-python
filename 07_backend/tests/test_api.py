"""
tests/test_api.py
==================
API tests using FastAPI TestClient.
Entire test module is skipped if FastAPI is not installed.
"""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi", reason="fastapi not installed")

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


class TestHealth:
    def test_health_ok(self) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"


class TestUsers:
    def setup_method(self) -> None:
        """Clear the in-memory DB between tests."""
        from app.routers import users as u_mod
        u_mod._db.clear()
        u_mod._next_id = 1

    def _create_user(self, username: str = "alice") -> dict:
        resp = client.post("/api/v1/users", json={
            "username":  username,
            "email":     f"{username}@example.com",
            "full_name": username.title(),
            "password":  "SecurePass1",
        })
        assert resp.status_code == 201
        return resp.json()

    def test_create_user(self) -> None:
        user = self._create_user("alice")
        assert user["username"] == "alice"
        assert user["email"] == "alice@example.com"
        assert "password" not in user

    def test_list_users_empty(self) -> None:
        resp = client.get("/api/v1/users")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_users(self) -> None:
        self._create_user("alice")
        self._create_user("bob")
        resp = client.get("/api/v1/users")
        assert resp.status_code == 200
        assert len(resp.json()) == 2

    def test_get_user(self) -> None:
        created = self._create_user("alice")
        resp = client.get(f"/api/v1/users/{created['id']}")
        assert resp.status_code == 200
        assert resp.json()["username"] == "alice"

    def test_get_user_not_found(self) -> None:
        resp = client.get("/api/v1/users/9999")
        assert resp.status_code == 404

    def test_update_user(self) -> None:
        created = self._create_user("alice")
        resp = client.patch(
            f"/api/v1/users/{created['id']}",
            json={"full_name": "Alice Smith"},
        )
        assert resp.status_code == 200
        assert resp.json()["full_name"] == "Alice Smith"

    def test_delete_user(self) -> None:
        created = self._create_user("alice")
        resp = client.delete(f"/api/v1/users/{created['id']}")
        assert resp.status_code == 204
        resp = client.get(f"/api/v1/users/{created['id']}")
        assert resp.status_code == 404

    def test_duplicate_username(self) -> None:
        self._create_user("alice")
        resp = client.post("/api/v1/users", json={
            "username":  "alice",
            "email":     "other@example.com",
            "full_name": "Alice 2",
            "password":  "SecurePass1",
        })
        assert resp.status_code == 409
