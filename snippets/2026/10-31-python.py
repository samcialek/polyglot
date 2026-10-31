from collections import OrderedDict


class LRUCache:
    """Least-recently-used cache with O(1) get and put."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._store: OrderedDict[str, object] = OrderedDict()

    def get(self, key: str) -> object | None:
        if key not in self._store:
            return None
        self._store.move_to_end(key)
        return self._store[key]

    def put(self, key: str, value: object) -> None:
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = value
        if len(self._store) > self.capacity:
            self._store.popitem(last=False)


if __name__ == "__main__":
    cache = LRUCache(3)
    for k, v in [("a", 1), ("b", 2), ("c", 3), ("d", 4)]:
        cache.put(k, v)
    print(f"get('a') = {cache.get('a')}")  # None (evicted)
    print(f"get('d') = {cache.get('d')}")  # 4
