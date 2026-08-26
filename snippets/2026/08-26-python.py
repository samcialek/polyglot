import time
from contextlib import contextmanager


class Timer:
    """Context manager that measures elapsed time."""

    def __init__(self, label: str = "Block"):
        self.label = label
        self.elapsed = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self._start
        print(f"{self.label}: {self.elapsed:.4f}s")
        return False


@contextmanager
def temp_list():
    """Yield a temporary list that gets cleared on exit."""
    data = []
    try:
        yield data
    finally:
        print(f"Cleaning up {len(data)} items")
        data.clear()


@contextmanager
def suppress(*exceptions):
    """Suppress specified exception types."""
    try:
        yield
    except exceptions:
        pass


if __name__ == "__main__":
    with Timer("Sum computation"):
        total = sum(range(1_000_000))
        print(f"Sum: {total}")

    with temp_list() as items:
        items.extend([1, 2, 3, 4, 5])
        print(f"Items: {items}")

    with suppress(ZeroDivisionError):
        result = 1 / 0  # silently suppressed
    print("Survived division by zero!")
