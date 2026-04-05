# Module 03: Design Patterns & Clean Architecture

Gang-of-Four patterns and beyond, in idiomatic Python 3.11+.

## Patterns Covered

### Creational
- **Singleton** (`creational/singleton.py`) – thread-safe, metaclass-based
- **Factory** (`creational/factory.py`) – Factory Method + Abstract Factory

### Behavioral
- **Observer** (`behavioral/observer.py`) – event bus / pub-sub
- **Strategy** (`behavioral/strategy.py`) – payment strategies, sort strategies

### Structural
- **Decorator** (`behavioral/decorator.py`) – function and class decorators

### Dependency Injection
- **DI Container** (`di_container.py`) – lightweight IoC container

## Running

```bash
python creational/singleton.py
pytest tests/ -v
```
