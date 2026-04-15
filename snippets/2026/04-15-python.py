from collections import deque


def topological_sort(graph: dict[str, list[str]]) -> list[str]:
    """Return a topological ordering of a DAG using Kahn's algorithm."""
    in_deg: dict[str, int] = {n: 0 for n in graph}
    for u in graph:
        for v in graph[u]:
            in_deg.setdefault(v, 0)
            in_deg[v] += 1
    queue = deque(n for n, d in in_deg.items() if d == 0)
    order = []
    while queue:
        u = queue.popleft()
        order.append(u)
        for v in graph.get(u, []):
            in_deg[v] -= 1
            if in_deg[v] == 0:
                queue.append(v)
    if len(order) != len(in_deg):
        raise ValueError("cycle detected")
    return order


if __name__ == "__main__":
    dag = {"A": ["B", "C"], "B": ["D"], "C": ["D"], "D": []}
    print(topological_sort(dag))
