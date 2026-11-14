from dataclasses import dataclass, field, asdict
from typing import Optional
import json


@dataclass(frozen=True)
class Point:
    """Immutable 2D point."""
    x: float
    y: float

    def distance_to(self, other: "Point") -> float:
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5

    def __add__(self, other: "Point") -> "Point":
        return Point(self.x + other.x, self.y + other.y)


@dataclass
class Config:
    """Nested config with defaults and serialization."""
    host: str = "localhost"
    port: int = 8080
    debug: bool = False
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, s: str) -> "Config":
        return cls(**json.loads(s))


@dataclass(order=True)
class Priority:
    """Sortable priority wrapper."""
    priority: int
    label: str = field(compare=False)

    def __repr__(self):
        return f"P({self.priority}, {self.label!r})"


if __name__ == "__main__":
    p1, p2 = Point(0, 0), Point(3, 4)
    print(f"Distance: {p1.distance_to(p2)}")
    print(f"Sum: {p1 + p2}")

    cfg = Config(host="prod.example.com", port=443, tags=["api", "v2"])
    print(cfg.to_json())

    tasks = [Priority(3, "low"), Priority(1, "urgent"), Priority(2, "medium")]
    print(f"Sorted: {sorted(tasks)}")
