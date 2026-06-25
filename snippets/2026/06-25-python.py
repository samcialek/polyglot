from typing import Protocol, TypeVar, runtime_checkable
from dataclasses import dataclass


T = TypeVar("T")


@runtime_checkable
class Comparable(Protocol):
    def __lt__(self, other: "Comparable") -> bool: ...
    def __eq__(self, other: object) -> bool: ...


@runtime_checkable
class Serializable(Protocol):
    def to_dict(self) -> dict: ...
    @classmethod
    def from_dict(cls, data: dict) -> "Serializable": ...


@dataclass(order=True)
class Temperature:
    celsius: float

    @property
    def fahrenheit(self) -> float:
        return self.celsius * 9 / 5 + 32

    def to_dict(self) -> dict:
        return {"celsius": self.celsius}

    @classmethod
    def from_dict(cls, data: dict) -> "Temperature":
        return cls(celsius=data["celsius"])


def find_min(items: list[T]) -> T | None:
    """Find minimum using Protocol-based duck typing."""
    if not items:
        return None
    result = items[0]
    for item in items[1:]:
        if item < result:
            result = item
    return result


def serialize_all(items: list[Serializable]) -> list[dict]:
    return [item.to_dict() for item in items]


if __name__ == "__main__":
    temps = [Temperature(100), Temperature(0), Temperature(37), Temperature(-40)]
    print(f"Coldest: {find_min(temps)}")
    print(f"Is Comparable? {isinstance(temps[0], Comparable)}")
    print(f"Is Serializable? {isinstance(temps[0], Serializable)}")
    print(f"Serialized: {serialize_all(temps)}")
