"""
app/routers/users.py
=====================
Users CRUD router.

Endpoints:
  GET    /users          – list all users
  POST   /users          – create user
  GET    /users/{id}     – get user by id
  PATCH  /users/{id}     – partial update
  DELETE /users/{id}     – delete user
"""

from __future__ import annotations

from datetime import datetime, timezone

try:
    from fastapi import APIRouter, HTTPException, Query, status
    from app.models import UserCreate, UserResponse, UserUpdate
    FASTAPI_AVAILABLE = True
except ImportError:  # pragma: no cover
    FASTAPI_AVAILABLE = False

if FASTAPI_AVAILABLE:
    router = APIRouter(prefix="/users", tags=["users"])

    # In-memory store (replace with a real DB in production)
    _db: dict[int, dict] = {}
    _next_id = 1


    def _next_user_id() -> int:
        global _next_id
        uid = _next_id
        _next_id += 1
        return uid


    @router.get("", response_model=list[UserResponse])
    async def list_users(
        skip:   int = Query(0, ge=0),
        limit:  int = Query(20, ge=1, le=100),
        active: bool | None = Query(None),
    ) -> list[dict]:
        """Return a paginated list of users, optionally filtered by active status."""
        users = list(_db.values())
        if active is not None:
            users = [u for u in users if u["is_active"] == active]
        return users[skip : skip + limit]


    @router.post("", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
    async def create_user(payload: UserCreate) -> dict:
        """Create a new user."""
        # Check uniqueness
        for user in _db.values():
            if user["username"] == payload.username:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"Username '{payload.username}' already taken",
                )
            if user["email"] == payload.email:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"Email '{payload.email}' already registered",
                )

        uid = _next_user_id()
        user_record: dict = {
            "id":         uid,
            "username":   payload.username,
            "email":      payload.email,
            "full_name":  payload.full_name,
            "is_active":  True,
            "created_at": datetime.now(timezone.utc),
            # Never store plaintext passwords in production; use bcrypt etc.
            "_password_hash": payload.password,
        }
        _db[uid] = user_record
        return user_record


    @router.get("/{user_id}", response_model=UserResponse)
    async def get_user(user_id: int) -> dict:
        """Return a single user by ID."""
        user = _db.get(user_id)
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User {user_id} not found",
            )
        return user


    @router.patch("/{user_id}", response_model=UserResponse)
    async def update_user(user_id: int, payload: UserUpdate) -> dict:
        """Partially update a user."""
        user = _db.get(user_id)
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User {user_id} not found",
            )
        update_data = payload.model_dump(exclude_none=True)
        user.update(update_data)
        return user


    @router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
    async def delete_user(user_id: int) -> None:
        """Delete a user."""
        if user_id not in _db:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User {user_id} not found",
            )
        del _db[user_id]
