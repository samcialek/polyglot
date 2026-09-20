import hashlib


class BloomFilter:
    """Space-efficient probabilistic set membership test."""

    def __init__(self, size: int = 1024, num_hashes: int = 3):
        self.size = size
        self.num_hashes = num_hashes
        self.bits = bytearray(size)

    def _hashes(self, item: str) -> list[int]:
        positions = []
        for i in range(self.num_hashes):
            digest = hashlib.sha256(f"{i}:{item}".encode()).hexdigest()
            positions.append(int(digest, 16) % self.size)
        return positions

    def add(self, item: str) -> None:
        for pos in self._hashes(item):
            self.bits[pos] = 1

    def might_contain(self, item: str) -> bool:
        return all(self.bits[pos] for pos in self._hashes(item))


if __name__ == "__main__":
    bf = BloomFilter(size=256, num_hashes=5)
    for word in ["hello", "world", "bloom", "filter"]:
        bf.add(word)
    print(f"might_contain('bloom')  = {bf.might_contain('bloom')}")
    print(f"might_contain('python') = {bf.might_contain('python')}")
