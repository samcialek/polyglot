import heapq
from collections import defaultdict


def dijkstra(graph: dict[str, list[tuple[str, int]]], start: str) -> dict[str, int]:
    """Return shortest distances from *start* to every reachable node."""
    dist: dict[str, int] = {start: 0}
    heap = [(0, start)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist.get(u, float("inf")):
            continue
        for v, w in graph.get(u, []):
            nd = d + w
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    return dist


if __name__ == "__main__":
    g = {
        "A": [("B", 1), ("C", 4)],
        "B": [("C", 2), ("D", 6)],
        "C": [("D", 3)],
    }
    print(dijkstra(g, "A"))  # {'A': 0, 'B': 1, 'C': 3, 'D': 6}
