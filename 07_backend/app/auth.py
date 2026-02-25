"""
app/auth.py
============
JWT-style authentication utilities using only stdlib (hmac + hashlib).

This is an educational implementation. For production use python-jose or PyJWT.

Token format:  base64url(header).base64url(payload).base64url(signature)
Signature:     HMAC-SHA256(header + "." + payload, secret)
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from typing import Any

# ── Secret key (load from env in production) ──────────────────────────────
import os
_SECRET_KEY = os.environ.get("SECRET_KEY", "change-me-in-production-use-env-var")
_ALGORITHM  = "HS256"
_DEFAULT_TTL = 3600  # 1 hour


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _b64url_encode(data: bytes) -> str:
    """Base64-URL encode bytes (no padding)."""
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _b64url_decode(s: str) -> bytes:
    """Base64-URL decode a string, re-adding padding as needed."""
    padding = "=" * (4 - len(s) % 4) if len(s) % 4 else ""
    return base64.urlsafe_b64decode(s + padding)


def _sign(message: str, secret: str) -> str:
    """Return HMAC-SHA256 signature as base64url string."""
    mac = hmac.new(secret.encode(), message.encode(), hashlib.sha256)
    return _b64url_encode(mac.digest())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_token(
    payload: dict[str, Any],
    secret: str = _SECRET_KEY,
    ttl: int = _DEFAULT_TTL,
) -> str:
    """
    Create a signed JWT-like token.

    Args:
        payload: Claims dictionary (do not include 'exp' or 'iat'; they are added).
        secret:  Signing secret.
        ttl:     Token lifetime in seconds.

    Returns:
        Dot-separated base64url token string.
    """
    now = int(time.time())
    header  = {"alg": _ALGORITHM, "typ": "JWT"}
    claims  = {**payload, "iat": now, "exp": now + ttl}

    header_encoded  = _b64url_encode(json.dumps(header,  separators=(",", ":")).encode())
    payload_encoded = _b64url_encode(json.dumps(claims, separators=(",", ":")).encode())
    signing_input   = f"{header_encoded}.{payload_encoded}"
    signature       = _sign(signing_input, secret)

    return f"{signing_input}.{signature}"


def verify_token(token: str, secret: str = _SECRET_KEY) -> dict[str, Any]:
    """
    Verify and decode a token.

    Args:
        token:  Token string as returned by :func:`create_token`.
        secret: Signing secret.

    Returns:
        Decoded claims dict.

    Raises:
        ValueError: If the token is malformed, expired, or the signature is invalid.
    """
    parts = token.split(".")
    if len(parts) != 3:
        raise ValueError("Malformed token: expected 3 dot-separated parts")

    header_enc, payload_enc, provided_sig = parts
    signing_input = f"{header_enc}.{payload_enc}"

    # Verify signature (constant-time comparison)
    expected_sig = _sign(signing_input, secret)
    if not hmac.compare_digest(expected_sig, provided_sig):
        raise ValueError("Invalid token signature")

    # Decode payload
    try:
        claims: dict[str, Any] = json.loads(_b64url_decode(payload_enc))
    except Exception as exc:
        raise ValueError(f"Cannot decode token payload: {exc}") from exc

    # Check expiry
    exp = claims.get("exp")
    if exp is not None and int(time.time()) > exp:
        raise ValueError("Token has expired")

    return claims


def hash_password(password: str, salt: str | None = None) -> str:
    """
    Hash a password with PBKDF2-HMAC-SHA256.

    Returns:
        A string of the form "salt$hash" (both base64url encoded).
    """
    import secrets
    _salt = salt.encode() if salt else secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), _salt, iterations=260_000)
    return f"{_b64url_encode(_salt)}${_b64url_encode(dk)}"


def verify_password(password: str, stored: str) -> bool:
    """
    Verify a password against a stored hash.

    Args:
        password: Plain-text password to verify.
        stored:   Hash string as returned by :func:`hash_password`.

    Returns:
        True if the password matches.
    """
    try:
        salt_b64, _ = stored.split("$", 1)
        salt = _b64url_decode(salt_b64).decode("latin-1")
        expected = hash_password(password, salt)
        return hmac.compare_digest(expected, stored)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def main() -> None:
    """Demonstrate token creation and verification."""
    print("=== JWT-like Tokens ===")
    token = create_token({"sub": "user_42", "role": "admin"}, ttl=60)
    print(f"Token: {token[:60]}…")

    claims = verify_token(token)
    print(f"Claims: {claims}")

    print("\n=== Password Hashing ===")
    raw = "s3cr3t_P@ssword"
    stored = hash_password(raw)
    print(f"Stored: {stored[:40]}…")
    print(f"Verify correct:   {verify_password(raw, stored)}")
    print(f"Verify incorrect: {verify_password('wrong', stored)}")


if __name__ == "__main__":
    main()
