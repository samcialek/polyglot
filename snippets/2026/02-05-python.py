from typing import Iterator, Generator
import itertools


def fibonacci() -> Generator[int, None, None]:
    """Infinite Fibonacci sequence generator."""
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b


def sliding_window(iterable, size: int) -> Generator[tuple, None, None]:
    """Yield sliding windows of `size` over an iterable."""
    it = iter(iterable)
    window = tuple(itertools.islice(it, size))
    if len(window) == size:
        yield window
    for item in it:
        window = window[1:] + (item,)
        yield window


def flatten(nested) -> Generator:
    """Recursively flatten any nested iterable (except strings)."""
    for item in nested:
        if hasattr(item, "__iter__") and not isinstance(item, (str, bytes)):
            yield from flatten(item)
        else:
            yield item


def chunked(iterable, n: int) -> Generator[list, None, None]:
    """Split iterable into chunks of size n."""
    it = iter(iterable)
    while chunk := list(itertools.islice(it, n)):
        yield chunk


if __name__ == "__main__":
    # First 10 Fibonacci numbers
    fibs = list(itertools.islice(fibonacci(), 10))
    print(f"Fibonacci: {fibs}")

    # Sliding window
    windows = list(sliding_window(range(6), 3))
    print(f"Windows: {windows}")

    # Flatten nested structure
    nested = [1, [2, [3, 4], 5], [6, [7, [8]]]]
    print(f"Flattened: {list(flatten(nested))}")

    # Chunked
    chunks = list(chunked(range(10), 3))
    print(f"Chunks: {chunks}")
