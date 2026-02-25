# Module 07: Backend Production

Build production-grade REST APIs with FastAPI, Pydantic, and Docker.

## Prerequisites

```bash
pip install "fastapi[all]" uvicorn pydantic
```

## Running

```bash
uvicorn app.main:app --reload
```

## Topics Covered

| File | Concepts |
|------|----------|
| `app/main.py` | FastAPI app, CORS, lifespan events |
| `app/models.py` | Pydantic v2 models |
| `app/routers/users.py` | CRUD router with HTTP verbs |
| `app/auth.py` | JWT utilities (stdlib hmac/hashlib) |
| `Dockerfile` | Production container |
| `tests/test_api.py` | TestClient tests (skipped if FastAPI absent) |
