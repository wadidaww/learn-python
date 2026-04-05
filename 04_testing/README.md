# Module 04: Testing & Code Quality

Write tests that give you genuine confidence in your code.

## Topics Covered

| File | Concepts |
|------|----------|
| `examples/calculator.py` | The system under test |
| `tests/test_calculator.py` | Unit tests, fixtures, parametrize, mocks |
| `tests/test_property_based.py` | Manual property-based testing |
| `conftest.py` | Shared fixtures |

## Running

```bash
pytest tests/ -v
pytest tests/ --cov=examples --cov-report=term-missing
```
