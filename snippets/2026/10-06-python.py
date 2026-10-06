class TrieNode:
    __slots__ = ("children", "is_end")

    def __init__(self):
        self.children: dict[str, "TrieNode"] = {}
        self.is_end = False


class Trie:
    """Prefix tree supporting insert, search, and starts_with."""

    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        node = self.root
        for ch in word:
            node = node.children.setdefault(ch, TrieNode())
        node.is_end = True

    def search(self, word: str) -> bool:
        node = self._walk(word)
        return node is not None and node.is_end

    def starts_with(self, prefix: str) -> bool:
        return self._walk(prefix) is not None

    def _walk(self, s: str) -> TrieNode | None:
        node = self.root
        for ch in s:
            if ch not in node.children:
                return None
            node = node.children[ch]
        return node


if __name__ == "__main__":
    t = Trie()
    for w in ["apple", "app", "apex", "bat"]:
        t.insert(w)
    print(f"search('app')       = {t.search('app')}")
    print(f"starts_with('ap')   = {t.starts_with('ap')}")
    print(f"search('application') = {t.search('application')}")
