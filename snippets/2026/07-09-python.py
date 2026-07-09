def binary_search(arr: list[int], target: int) -> int:
    """Return index of target in sorted array, or -1 if not found."""
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1


if __name__ == "__main__":
    data = [2, 5, 8, 12, 16, 23, 38, 56, 72, 91]
    for val in [23, 42]:
        idx = binary_search(data, val)
        print(f"binary_search({val}) -> {idx}")
