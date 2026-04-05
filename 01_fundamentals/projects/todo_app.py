"""
projects/todo_app.py
=====================
A complete CLI todo application with persistent JSON storage.

Usage:
    python todo_app.py add "Buy groceries"
    python todo_app.py list
    python todo_app.py complete 1
    python todo_app.py delete 1
    python todo_app.py --help
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_DB = Path.home() / ".todo_app" / "todos.json"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class TodoItem:
    """A single todo item."""

    id: int
    title: str
    done: bool = False
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    completed_at: str | None = None

    def complete(self) -> None:
        """Mark this item as done."""
        self.done = True
        self.completed_at = datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

class TodoStore:
    """JSON-backed persistent store for TodoItems."""

    def __init__(self, path: Path = DEFAULT_DB) -> None:
        self.path = path
        self._items: list[TodoItem] = []
        self._next_id: int = 1
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            self._items = [TodoItem(**item) for item in raw.get("items", [])]
            self._next_id = raw.get("next_id", 1)

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "next_id": self._next_id,
            "items": [asdict(item) for item in self._items],
        }
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def add(self, title: str) -> TodoItem:
        """Add a new todo item; returns the created item."""
        item = TodoItem(id=self._next_id, title=title)
        self._items.append(item)
        self._next_id += 1
        self._save()
        return item

    def get_all(self) -> list[TodoItem]:
        """Return all todo items."""
        return list(self._items)

    def get_by_id(self, item_id: int) -> TodoItem:
        """Return item by *item_id* or raise KeyError."""
        for item in self._items:
            if item.id == item_id:
                return item
        raise KeyError(f"No todo with id={item_id}")

    def complete(self, item_id: int) -> TodoItem:
        """Mark an item as complete; returns updated item."""
        item = self.get_by_id(item_id)
        item.complete()
        self._save()
        return item

    def delete(self, item_id: int) -> None:
        """Delete an item by id."""
        item = self.get_by_id(item_id)
        self._items.remove(item)
        self._save()


# ---------------------------------------------------------------------------
# CLI handlers
# ---------------------------------------------------------------------------

def cmd_add(store: TodoStore, args: argparse.Namespace) -> int:
    """Handle 'add' subcommand."""
    item = store.add(args.title)
    print(f"Added [{item.id}] {item.title}")
    return 0


def cmd_list(store: TodoStore, _args: argparse.Namespace) -> int:
    """Handle 'list' subcommand."""
    items = store.get_all()
    if not items:
        print("No todos yet. Add one with: todo_app.py add <title>")
        return 0
    for item in items:
        status = "✓" if item.done else " "
        print(f"  [{status}] {item.id:>3}. {item.title}")
    pending = sum(1 for i in items if not i.done)
    print(f"\n{len(items)} total, {pending} pending")
    return 0


def cmd_complete(store: TodoStore, args: argparse.Namespace) -> int:
    """Handle 'complete' subcommand."""
    try:
        item = store.complete(args.id)
        print(f"Completed [{item.id}] {item.title}")
        return 0
    except KeyError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_delete(store: TodoStore, args: argparse.Namespace) -> int:
    """Handle 'delete' subcommand."""
    try:
        store.delete(args.id)
        print(f"Deleted todo {args.id}")
        return 0
    except KeyError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Construct and return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="todo_app",
        description="A simple CLI todo application",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB,
        metavar="PATH",
        help="Path to the JSON database file (default: ~/.todo_app/todos.json)",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # add
    p_add = sub.add_parser("add", help="Add a new todo item")
    p_add.add_argument("title", help="Title of the todo item")

    # list
    sub.add_parser("list", help="List all todo items")

    # complete
    p_done = sub.add_parser("complete", help="Mark a todo item as done")
    p_done.add_argument("id", type=int, help="ID of the todo to complete")

    # delete
    p_del = sub.add_parser("delete", help="Delete a todo item")
    p_del.add_argument("id", type=int, help="ID of the todo to delete")

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    store = TodoStore(args.db)

    handlers = {
        "add": cmd_add,
        "list": cmd_list,
        "complete": cmd_complete,
        "delete": cmd_delete,
    }
    return handlers[args.command](store, args)


if __name__ == "__main__":
    sys.exit(main())
