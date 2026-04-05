"""
app/models.py
=============
Pydantic v2 models for request/response bodies.
"""

from __future__ import annotations

try:
    from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator
    PYDANTIC_AVAILABLE = True
except ImportError:  # pragma: no cover
    PYDANTIC_AVAILABLE = False

from datetime import datetime, timezone


if PYDANTIC_AVAILABLE:
    class UserCreate(BaseModel):
        """Payload for creating a new user."""

        model_config = ConfigDict(str_strip_whitespace=True)

        username:  str       = Field(..., min_length=3, max_length=50)
        email:     str       = Field(..., pattern=r"^[^@]+@[^@]+\.[^@]+$")
        full_name: str       = Field(..., min_length=1, max_length=200)
        password:  str       = Field(..., min_length=8)

        @field_validator("username")
        @classmethod
        def username_alphanumeric(cls, v: str) -> str:
            if not v.replace("_", "").replace("-", "").isalnum():
                raise ValueError("Username must be alphanumeric (- and _ allowed)")
            return v.lower()


    class UserResponse(BaseModel):
        """Public user representation (no password)."""

        model_config = ConfigDict(from_attributes=True)

        id:         int
        username:   str
        email:      str
        full_name:  str
        is_active:  bool
        created_at: datetime


    class UserUpdate(BaseModel):
        """Partial user update payload."""

        full_name: str | None = None
        email:     str | None = None
        is_active: bool | None = None


    class TokenResponse(BaseModel):
        """JWT token response."""

        access_token:  str
        token_type:    str = "bearer"
        expires_in:    int  # seconds


    class LoginRequest(BaseModel):
        """Login credentials."""

        username: str
        password: str


    class ErrorResponse(BaseModel):
        """Standard error envelope."""

        error:   str
        detail:  str | None = None
        status:  int
