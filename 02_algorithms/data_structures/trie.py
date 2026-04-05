"""
data_structures/trie.py
========================
A Trie (prefix tree) for efficient string operations:
    - insert:          O(m)  where m = word length
    - search:          O(m)
    - starts_with:     O(m)
    - delete:          O(m)
    - words_with_prefix: O(m + k)  where k = number of matching words
"""

from __future__ import annotations

from collections.abc import Iterator


class TrieNode:
    """A single node in the Trie."""

    __slots__ = ("children", "is_end", "count")

    def __init__(self) -> None:
        self.children: dict[str, TrieNode] = {}
        self.is_end: bool = False
        self.count: int = 0  # number of words that pass through this node


class Trie:
    """
    Prefix tree supporting insert, search, delete, and prefix enumeration.

    Example::

        t = Trie()
        t.insert("apple")
        t.insert("app")
        assert t.search("apple")
        assert not t.search("appl")
        assert t.starts_with("app")
        assert sorted(t.words_with_prefix("app")) == ["app", "apple"]
    """

    def __init__(self) -> None:
        self._root = TrieNode()
        self._word_count = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def insert(self, word: str) -> None:
        """Insert *word* into the trie."""
        if not word:
            return
        node = self._root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
            node.count += 1
        if not node.is_end:
            node.is_end = True
            self._word_count += 1

    def search(self, word: str) -> bool:
        """Return True if *word* is in the trie."""
        node = self._find_node(word)
        return node is not None and node.is_end

    def starts_with(self, prefix: str) -> bool:
        """Return True if any word in the trie starts with *prefix*."""
        return self._find_node(prefix) is not None

    def delete(self, word: str) -> bool:
        """
        Delete *word* from the trie.

        Returns True if the word was present and deleted, False otherwise.
        """
        if not self.search(word):
            return False
        self._delete_recursive(self._root, word, 0)
        self._word_count -= 1
        return True

    def words_with_prefix(self, prefix: str) -> list[str]:
        """Return all words that start with *prefix*."""
        node = self._find_node(prefix)
        if node is None:
            return []
        results: list[str] = []
        self._collect_words(node, prefix, results)
        return results

    def all_words(self) -> list[str]:
        """Return all words stored in the trie."""
        return self.words_with_prefix("")

    def __len__(self) -> int:
        return self._word_count

    def __contains__(self, word: object) -> bool:
        return isinstance(word, str) and self.search(word)

    def __iter__(self) -> Iterator[str]:
        return iter(self.all_words())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_node(self, prefix: str) -> TrieNode | None:
        """Return the node at the end of *prefix*, or None."""
        node = self._root
        for char in prefix:
            if char not in node.children:
                return None
            node = node.children[char]
        return node

    def _collect_words(self, node: TrieNode, current: str, results: list[str]) -> None:
        """DFS to collect all words reachable from *node*."""
        if node.is_end:
            results.append(current)
        for char, child in sorted(node.children.items()):
            self._collect_words(child, current + char, results)

    def _delete_recursive(self, node: TrieNode, word: str, depth: int) -> bool:
        """
        Recursively delete *word*; return True if the current node can be deleted.
        """
        if depth == len(word):
            node.is_end = False
            return len(node.children) == 0

        char = word[depth]
        child = node.children.get(char)
        if child is None:
            return False

        child.count -= 1
        should_delete = self._delete_recursive(child, word, depth + 1)
        if should_delete:
            del node.children[char]
        return not node.is_end and len(node.children) == 0


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def main() -> None:
    """Demonstrate Trie usage."""
    trie = Trie()
    words = ["apple", "app", "apply", "application", "apt", "bat", "bath", "bad"]
    for w in words:
        trie.insert(w)

    print(f"Words in trie: {len(trie)}")
    print(f"All words:     {trie.all_words()}")
    print(f"Prefix 'app':  {trie.words_with_prefix('app')}")
    print(f"Prefix 'ba':   {trie.words_with_prefix('ba')}")
    print(f"'apple' in trie: {'apple' in trie}")
    print(f"'appl' in trie:  {'appl' in trie}")

    trie.delete("apple")
    print(f"After deleting 'apple': {trie.words_with_prefix('app')}")


if __name__ == "__main__":
    main()
