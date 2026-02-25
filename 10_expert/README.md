# Module 10: Expert Level

Build your own async web framework and task queue from scratch.

## Topics Covered

| File | Concepts |
|------|----------|
| `mini_framework/core.py` | Async ASGI-like core, request/response |
| `mini_framework/router.py` | URL routing with path parameters |
| `mini_framework/middleware.py` | Middleware chain |
| `task_queue/queue.py` | In-memory task queue |
| `task_queue/worker.py` | Worker pool |

## Running

```bash
python mini_framework/core.py  # starts the async framework demo
pytest tests/ -v
```
