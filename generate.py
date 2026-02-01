#!/usr/bin/env python3
"""
Polyglot Snippet Generator
Generates meaningful code snippets in weighted programming languages.
Each snippet is a real algorithm, data structure, or utility implementation.

Language weights:
  Python 56% | R 20% | JS 5% | TypeScript 4% | Go 3% | Rust 3% | Ruby 3% | C 3% | Java 3%
"""

import random
import os
import sys
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Weighted language selection
# ---------------------------------------------------------------------------

WEIGHTS = {
    "python": 56,
    "r": 20,
    "javascript": 5,
    "typescript": 4,
    "go": 3,
    "rust": 3,
    "ruby": 3,
    "c": 3,
    "java": 3,
}

# ---------------------------------------------------------------------------
# Language definitions: (name, extension, snippets[])
# Each snippet: (title, description, code)
# ---------------------------------------------------------------------------

LANGUAGES = {
    "python": {
        "ext": "py",
        "snippets": [
            ("binary_search", "Binary search implementation", '''
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
'''),
            ("lru_cache", "LRU Cache using OrderedDict", '''
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
'''),
            ("merge_sort", "Merge sort implementation", '''
def merge_sort(arr: list[int]) -> list[int]:
    """Sort a list using the merge sort algorithm — O(n log n)."""
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return _merge(left, right)


def _merge(a: list[int], b: list[int]) -> list[int]:
    result = []
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i] <= b[j]:
            result.append(a[i]); i += 1
        else:
            result.append(b[j]); j += 1
    result.extend(a[i:])
    result.extend(b[j:])
    return result


if __name__ == "__main__":
    data = [38, 27, 43, 3, 9, 82, 10]
    print(f"Sorted: {merge_sort(data)}")
'''),
            ("trie", "Trie (prefix tree) data structure", '''
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
'''),
            ("dijkstra", "Dijkstra's shortest path algorithm", '''
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
'''),
            ("bloom_filter", "Probabilistic Bloom filter", '''
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
'''),
            ("matrix_multiply", "Matrix multiplication", '''
Matrix = list[list[float]]


def mat_mul(a: Matrix, b: Matrix) -> Matrix:
    """Multiply two matrices (naive O(n^3) implementation)."""
    rows_a, cols_a = len(a), len(a[0])
    rows_b, cols_b = len(b), len(b[0])
    assert cols_a == rows_b, "incompatible dimensions"
    result = [[0.0] * cols_b for _ in range(rows_a)]
    for i in range(rows_a):
        for j in range(cols_b):
            for k in range(cols_a):
                result[i][j] += a[i][k] * b[k][j]
    return result


if __name__ == "__main__":
    A = [[1, 2], [3, 4]]
    B = [[5, 6], [7, 8]]
    C = mat_mul(A, B)
    for row in C:
        print(row)  # [19, 22] and [43, 50]
'''),
            ("topological_sort", "Topological sort via Kahn's algorithm", '''
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
'''),
            ("decorator_patterns", "Decorator patterns showcase", '''
import functools
import time


def timer(func):
    """Measure execution time of a function."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"{func.__name__} took {elapsed:.4f}s")
        return result
    return wrapper


def memoize(func):
    """Cache function results based on arguments."""
    cache = {}
    @functools.wraps(func)
    def wrapper(*args):
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]
    return wrapper


def retry(max_attempts=3, delay=0.1):
    """Retry a function on exception."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts:
                        raise
                    print(f"Attempt {attempt} failed: {e}. Retrying...")
                    time.sleep(delay)
        return wrapper
    return decorator


@timer
@memoize
def fibonacci(n: int) -> int:
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)


if __name__ == "__main__":
    print(f"fib(30) = {fibonacci(30)}")
    print(f"fib(30) = {fibonacci(30)}")  # cached — instant
'''),
            ("context_managers", "Custom context managers", '''
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
'''),
            ("dataclass_patterns", "Dataclass patterns and tricks", '''
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
'''),
            ("generator_patterns", "Generator and iterator patterns", '''
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
'''),
            ("async_patterns", "Async/await patterns", '''
import asyncio
from typing import Any


async def fetch_data(url: str, delay: float) -> dict[str, Any]:
    """Simulate an async HTTP request."""
    print(f"Fetching {url}...")
    await asyncio.sleep(delay)
    return {"url": url, "status": 200, "data": f"Response from {url}"}


async def fetch_with_timeout(url: str, timeout: float) -> dict | None:
    """Fetch with timeout — returns None on timeout."""
    try:
        return await asyncio.wait_for(fetch_data(url, 2.0), timeout=timeout)
    except asyncio.TimeoutError:
        print(f"Timeout fetching {url}")
        return None


async def fetch_all_concurrent(urls: list[str]) -> list[dict]:
    """Fetch multiple URLs concurrently using gather."""
    tasks = [fetch_data(url, i * 0.3) for i, url in enumerate(urls)]
    return await asyncio.gather(*tasks)


async def producer_consumer():
    """Async producer-consumer pattern with a queue."""
    queue: asyncio.Queue[int] = asyncio.Queue(maxsize=5)

    async def producer():
        for i in range(8):
            await queue.put(i)
            print(f"Produced: {i}")
            await asyncio.sleep(0.1)
        await queue.put(-1)  # sentinel

    async def consumer():
        while True:
            item = await queue.get()
            if item == -1:
                break
            print(f"Consumed: {item}")
            await asyncio.sleep(0.2)

    await asyncio.gather(producer(), consumer())


async def main():
    urls = ["api.example.com/users", "api.example.com/posts", "api.example.com/comments"]
    results = await fetch_all_concurrent(urls)
    for r in results:
        print(f"  {r['url']} -> {r['status']}")

    print("\\nProducer-consumer:")
    await producer_consumer()


if __name__ == "__main__":
    asyncio.run(main())
'''),
            ("type_system_tricks", "Advanced type hints and protocols", '''
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
'''),
        ],
    },

    "r": {
        "ext": "R",
        "snippets": [
            ("statistical_summary", "Statistical summary functions", '''
# Custom statistical summary functions — beyond base R.

geometric_mean <- function(x) {
  exp(mean(log(x[x > 0])))
}

harmonic_mean <- function(x) {
  n <- length(x[x != 0])
  n / sum(1 / x[x != 0])
}

trimmed_summary <- function(x, trim = 0.1) {
  sorted <- sort(x)
  n <- length(sorted)
  lo <- floor(n * trim) + 1
  hi <- n - floor(n * trim)
  trimmed <- sorted[lo:hi]

  list(
    n = length(x),
    trimmed_n = length(trimmed),
    mean = mean(trimmed),
    sd = sd(trimmed),
    median = median(trimmed),
    iqr = IQR(trimmed),
    geometric_mean = geometric_mean(trimmed),
    harmonic_mean = harmonic_mean(trimmed)
  )
}

# Demo with simulated data
set.seed(42)
data <- c(rnorm(95, mean = 50, sd = 10), runif(5, 200, 500))  # 5 outliers

cat("=== Raw data summary ===\n")
print(summary(data))

cat("\n=== Trimmed (10%) summary ===\n")
result <- trimmed_summary(data, trim = 0.1)
for (name in names(result)) {
  cat(sprintf("  %-20s %s\n", paste0(name, ":"), round(result[[name]], 3)))
}
'''),
            ("bootstrap_ci", "Bootstrap confidence intervals", '''
# Bootstrap confidence intervals — nonparametric inference.

bootstrap_ci <- function(x, stat_fn = mean, n_boot = 10000, alpha = 0.05) {
  boot_stats <- replicate(n_boot, {
    sample_idx <- sample(seq_along(x), replace = TRUE)
    stat_fn(x[sample_idx])
  })

  list(
    estimate = stat_fn(x),
    ci_lower = quantile(boot_stats, alpha / 2),
    ci_upper = quantile(boot_stats, 1 - alpha / 2),
    se = sd(boot_stats),
    bias = mean(boot_stats) - stat_fn(x)
  )
}

# Demo: confidence interval for median income
set.seed(123)
incomes <- c(rlnorm(200, meanlog = 10.5, sdlog = 0.8))

cat("=== Bootstrap CI for Median Income ===\n")
result <- bootstrap_ci(incomes, stat_fn = median)
cat(sprintf("  Median:  $%.0f\n", result$estimate))
cat(sprintf("  95%% CI:  [$%.0f, $%.0f]\n", result$ci_lower, result$ci_upper))
cat(sprintf("  SE:      $%.0f\n", result$se))
cat(sprintf("  Bias:    $%.2f\n", result$bias))

# Compare mean vs trimmed mean
cat("\n=== Mean vs Trimmed Mean ===\n")
ci_mean <- bootstrap_ci(incomes, stat_fn = mean)
ci_trim <- bootstrap_ci(incomes, stat_fn = function(x) mean(x, trim = 0.1))
cat(sprintf("  Mean:         $%.0f [%.0f, %.0f]\n", ci_mean$estimate, ci_mean$ci_lower, ci_mean$ci_upper))
cat(sprintf("  Trimmed Mean: $%.0f [%.0f, %.0f]\n", ci_trim$estimate, ci_trim$ci_lower, ci_trim$ci_upper))
'''),
            ("bayesian_ab_test", "Bayesian A/B test analysis", '''
# Bayesian A/B testing with Beta-Binomial conjugate model.

bayesian_ab_test <- function(successes_a, trials_a, successes_b, trials_b,
                              prior_alpha = 1, prior_beta = 1, n_sim = 100000) {
  # Posterior distributions (Beta-Binomial conjugate)
  post_a <- rbeta(n_sim, prior_alpha + successes_a, prior_beta + trials_a - successes_a)
  post_b <- rbeta(n_sim, prior_alpha + successes_b, prior_beta + trials_b - successes_b)

  # P(B > A)
  prob_b_better <- mean(post_b > post_a)

  # Lift distribution
  lift <- (post_b - post_a) / post_a

  list(
    prob_b_better = prob_b_better,
    expected_rate_a = mean(post_a),
    expected_rate_b = mean(post_b),
    expected_lift = mean(lift),
    lift_ci = quantile(lift, c(0.025, 0.975)),
    risk_of_choosing_b = mean(pmax(post_a - post_b, 0))  # expected loss
  )
}

# Demo: website conversion test
set.seed(42)
# Control: 120 conversions out of 1000
# Variant: 145 conversions out of 1000
result <- bayesian_ab_test(120, 1000, 145, 1000)

cat("=== Bayesian A/B Test Results ===\n")
cat(sprintf("  P(B > A):        %.1f%%\n", result$prob_b_better * 100))
cat(sprintf("  Rate A:          %.2f%%\n", result$expected_rate_a * 100))
cat(sprintf("  Rate B:          %.2f%%\n", result$expected_rate_b * 100))
cat(sprintf("  Expected Lift:   %.1f%%\n", result$expected_lift * 100))
cat(sprintf("  Lift 95%% CI:     [%.1f%%, %.1f%%]\n",
            result$lift_ci[1] * 100, result$lift_ci[2] * 100))
cat(sprintf("  Risk (choose B): %.4f\n", result$risk_of_choosing_b))
'''),
            ("causal_inference", "Causal inference with propensity scores", '''
# Propensity score matching for causal inference — observational data.

simulate_treatment_data <- function(n = 500) {
  set.seed(42)
  age <- rnorm(n, 45, 12)
  income <- rnorm(n, 60000, 15000)

  # Treatment assignment depends on covariates (confounding)
  logit_p <- -2 + 0.03 * age + 0.00002 * income
  prob_treat <- plogis(logit_p)
  treatment <- rbinom(n, 1, prob_treat)

  # Outcome depends on treatment AND covariates
  outcome <- 100 + 5 * treatment + 0.5 * age + 0.0003 * income + rnorm(n, 0, 10)

  data.frame(age, income, treatment, outcome)
}

propensity_score_match <- function(df) {
  # Estimate propensity scores via logistic regression
  ps_model <- glm(treatment ~ age + income, data = df, family = binomial)
  df$pscore <- predict(ps_model, type = "response")

  # Naive estimate (biased)
  naive <- mean(df$outcome[df$treatment == 1]) - mean(df$outcome[df$treatment == 0])

  # IPW estimate (inverse propensity weighting)
  w1 <- df$treatment / df$pscore
  w0 <- (1 - df$treatment) / (1 - df$pscore)
  ipw_ate <- weighted.mean(df$outcome, w1) - weighted.mean(df$outcome, w0)

  # Regression adjustment
  reg_model <- lm(outcome ~ treatment + age + income, data = df)
  reg_ate <- coef(reg_model)["treatment"]

  list(
    naive_ate = naive,
    ipw_ate = ipw_ate,
    regression_ate = reg_ate,
    true_ate = 5  # we know this from simulation
  )
}

# Run
df <- simulate_treatment_data()
results <- propensity_score_match(df)

cat("=== Causal Inference: Average Treatment Effect ===\n")
cat(sprintf("  True ATE:       %.2f\n", results$true_ate))
cat(sprintf("  Naive estimate: %.2f (biased — ignores confounders)\n", results$naive_ate))
cat(sprintf("  IPW estimate:   %.2f\n", results$ipw_ate))
cat(sprintf("  Regression adj: %.2f\n", results$regression_ate))
'''),
            ("mcmc_sampler", "Metropolis-Hastings MCMC sampler", '''
# Metropolis-Hastings MCMC sampler for Bayesian posterior estimation.

metropolis_hastings <- function(log_target, init, n_iter = 10000,
                                 proposal_sd = 1, burn_in = 1000) {
  samples <- numeric(n_iter)
  current <- init
  accepted <- 0

  for (i in seq_len(n_iter)) {
    proposal <- rnorm(1, current, proposal_sd)
    log_ratio <- log_target(proposal) - log_target(current)

    if (log(runif(1)) < log_ratio) {
      current <- proposal
      accepted <- accepted + 1
    }
    samples[i] <- current
  }

  list(
    samples = samples[(burn_in + 1):n_iter],
    acceptance_rate = accepted / n_iter
  )
}

# Demo: Estimate posterior of normal mean with known variance
# Prior: mu ~ N(0, 10^2), Likelihood: x ~ N(mu, 2^2)
set.seed(42)
true_mu <- 3.5
data <- rnorm(50, mean = true_mu, sd = 2)

log_posterior <- function(mu) {
  log_prior <- dnorm(mu, mean = 0, sd = 10, log = TRUE)
  log_likelihood <- sum(dnorm(data, mean = mu, sd = 2, log = TRUE))
  log_prior + log_likelihood
}

result <- metropolis_hastings(log_posterior, init = 0, n_iter = 20000,
                               proposal_sd = 0.5, burn_in = 2000)

cat("=== MCMC Posterior Estimation ===\n")
cat(sprintf("  True mu:         %.2f\n", true_mu))
cat(sprintf("  Posterior mean:  %.2f\n", mean(result$samples)))
cat(sprintf("  Posterior SD:    %.2f\n", sd(result$samples)))
cat(sprintf("  95%% CI:          [%.2f, %.2f]\n",
            quantile(result$samples, 0.025), quantile(result$samples, 0.975)))
cat(sprintf("  Acceptance rate: %.1f%%\n", result$acceptance_rate * 100))
cat(sprintf("  MLE (x-bar):     %.2f\n", mean(data)))
'''),
            ("mixed_effects", "Linear mixed effects from scratch", '''
# Simplified mixed-effects model estimation via EM algorithm.
# Y_ij = beta0 + beta1*x_ij + b_i + epsilon_ij

simulate_mixed_data <- function(n_groups = 20, obs_per_group = 10) {
  set.seed(42)
  beta0 <- 5.0      # fixed intercept
  beta1 <- 2.0      # fixed slope
  sigma_b <- 1.5    # random effect SD
  sigma_e <- 1.0    # residual SD

  group <- rep(1:n_groups, each = obs_per_group)
  b <- rnorm(n_groups, 0, sigma_b)  # random intercepts
  x <- rnorm(n_groups * obs_per_group, 0, 1)
  y <- beta0 + beta1 * x + b[group] + rnorm(length(x), 0, sigma_e)

  data.frame(y = y, x = x, group = group)
}

estimate_mixed_model <- function(df, max_iter = 50, tol = 1e-6) {
  # Simple EM for random intercept model
  groups <- unique(df$group)
  n_g <- length(groups)

  # Initial estimates
  sigma2_b <- 1.0
  sigma2_e <- 1.0

  for (iter in seq_len(max_iter)) {
    # E-step: estimate random effects
    b_hat <- numeric(n_g)
    for (i in seq_along(groups)) {
      idx <- df$group == groups[i]
      n_i <- sum(idx)
      resid_i <- df$y[idx] - mean(df$y)  # simplified
      b_hat[i] <- (sigma2_b / (sigma2_b + sigma2_e / n_i)) * mean(resid_i)
    }

    # M-step: update fixed effects via OLS on residuals
    df$y_adj <- df$y - b_hat[df$group]
    fit <- lm(y_adj ~ x, data = df)

    # Update variance components
    resid <- df$y - predict(fit, df) - b_hat[df$group]
    sigma2_e_new <- mean(resid^2)
    sigma2_b_new <- max(var(b_hat) - sigma2_e / mean(table(df$group)), 0.01)

    if (abs(sigma2_e_new - sigma2_e) + abs(sigma2_b_new - sigma2_b) < tol) break
    sigma2_e <- sigma2_e_new
    sigma2_b <- sigma2_b_new
  }

  list(
    beta = coef(fit),
    sigma_b = sqrt(sigma2_b),
    sigma_e = sqrt(sigma2_e),
    random_effects = b_hat,
    iterations = iter
  )
}

# Run
df <- simulate_mixed_data()
result <- estimate_mixed_model(df)

cat("=== Mixed Effects Model (EM) ===\n")
cat(sprintf("  beta0 (true=5.0): %.2f\n", result$beta[1]))
cat(sprintf("  beta1 (true=2.0): %.2f\n", result$beta[2]))
cat(sprintf("  sigma_b (true=1.5): %.2f\n", result$sigma_b))
cat(sprintf("  sigma_e (true=1.0): %.2f\n", result$sigma_e))
cat(sprintf("  Converged in %d iterations\n", result$iterations))
'''),
            ("survival_analysis", "Kaplan-Meier survival estimator", '''
# Kaplan-Meier survival curve estimation from scratch.

kaplan_meier <- function(time, event) {
  # Sort by time
  ord <- order(time)
  time <- time[ord]
  event <- event[ord]

  unique_times <- sort(unique(time[event == 1]))
  n <- length(time)

  surv_prob <- 1.0
  results <- data.frame(
    time = numeric(0), n_risk = integer(0),
    n_event = integer(0), survival = numeric(0),
    se = numeric(0)
  )

  var_sum <- 0  # for Greenwood's formula
  for (t in unique_times) {
    n_risk <- sum(time >= t)
    n_event <- sum(time == t & event == 1)
    hazard <- n_event / n_risk
    surv_prob <- surv_prob * (1 - hazard)

    if (n_event > 0) {
      var_sum <- var_sum + n_event / (n_risk * (n_risk - n_event))
    }
    se <- surv_prob * sqrt(var_sum)

    results <- rbind(results, data.frame(
      time = t, n_risk = n_risk, n_event = n_event,
      survival = surv_prob, se = se
    ))
  }

  results$ci_lower <- pmax(results$survival - 1.96 * results$se, 0)
  results$ci_upper <- pmin(results$survival + 1.96 * results$se, 1)
  results
}

# Simulate clinical trial survival data
set.seed(42)
n <- 100
time <- rexp(n, rate = 0.1)              # true survival times
censor_time <- runif(n, 0, 30)           # censoring times
observed_time <- pmin(time, censor_time)  # what we observe
event <- as.integer(time <= censor_time)  # 1 = event, 0 = censored

km <- kaplan_meier(observed_time, event)

cat("=== Kaplan-Meier Survival Estimates ===\n")
cat(sprintf("  Events: %d / %d (%.0f%% censored)\n",
            sum(event), n, (1 - mean(event)) * 100))
cat("\n  Time    At Risk  Events  Survival  95% CI\n")
cat("  ", strrep("-", 55), "\n", sep = "")
for (i in seq_len(min(nrow(km), 12))) {
  cat(sprintf("  %5.1f   %4d     %4d    %.3f     [%.3f, %.3f]\n",
              km$time[i], km$n_risk[i], km$n_event[i],
              km$survival[i], km$ci_lower[i], km$ci_upper[i]))
}

# Median survival time
median_idx <- which(km$survival <= 0.5)[1]
if (!is.na(median_idx)) {
  cat(sprintf("\n  Median survival time: %.1f\n", km$time[median_idx]))
}
'''),
            ("ridge_regression", "Ridge regression with cross-validation", '''
# Ridge regression with k-fold cross-validation for lambda selection.

ridge_regression <- function(X, y, lambda) {
  # Closed-form: beta = (X'X + lambda*I)^{-1} X'y
  p <- ncol(X)
  XtX <- crossprod(X)
  Xty <- crossprod(X, y)
  beta <- solve(XtX + lambda * diag(p), Xty)
  as.vector(beta)
}

cv_ridge <- function(X, y, lambdas, k = 5) {
  n <- nrow(X)
  folds <- sample(rep(1:k, length.out = n))

  cv_errors <- sapply(lambdas, function(lam) {
    fold_errors <- sapply(1:k, function(fold) {
      train <- folds != fold
      test <- folds == fold
      beta <- ridge_regression(X[train, , drop = FALSE], y[train], lam)
      preds <- X[test, , drop = FALSE] %*% beta
      mean((y[test] - preds)^2)
    })
    mean(fold_errors)
  })

  list(
    lambdas = lambdas,
    cv_errors = cv_errors,
    best_lambda = lambdas[which.min(cv_errors)],
    min_error = min(cv_errors)
  )
}

# Demo: correlated predictors (where ridge excels)
set.seed(42)
n <- 200; p <- 10
Sigma <- outer(1:p, 1:p, function(i, j) 0.7^abs(i - j))
X <- MASS::mvrnorm(n, mu = rep(0, p), Sigma = Sigma)
true_beta <- c(3, -2, 0, 0, 1.5, 0, 0, -1, 0, 0.5)
y <- X %*% true_beta + rnorm(n, 0, 2)

# Cross-validate
lambdas <- 10^seq(-2, 3, length.out = 50)
cv_result <- cv_ridge(X, y, lambdas)

cat("=== Ridge Regression with CV ===\n")
cat(sprintf("  Best lambda: %.4f\n", cv_result$best_lambda))
cat(sprintf("  CV MSE:      %.4f\n", cv_result$min_error))

# Compare OLS vs Ridge
ols_beta <- solve(crossprod(X), crossprod(X, y))
ridge_beta <- ridge_regression(X, y, cv_result$best_lambda)

cat("\n  Coefficient comparison:\n")
cat(sprintf("  %-6s %8s %8s %8s\n", "Var", "True", "OLS", "Ridge"))
for (j in 1:p) {
  cat(sprintf("  X%-5d %8.2f %8.2f %8.2f\n", j, true_beta[j], ols_beta[j], ridge_beta[j]))
}
'''),
            ("time_series_decompose", "Time series seasonal decomposition", '''
# Classical time series decomposition: trend + seasonal + residual.

decompose_ts <- function(x, period) {
  n <- length(x)

  # Trend: centered moving average
  trend <- rep(NA, n)
  half <- period %/% 2
  for (i in (half + 1):(n - half)) {
    if (period %% 2 == 0) {
      trend[i] <- (0.5 * x[i - half] + sum(x[(i - half + 1):(i + half - 1)]) + 0.5 * x[i + half]) / period
    } else {
      trend[i] <- mean(x[(i - half):(i + half)])
    }
  }

  # Seasonal: average deviation from trend by position in cycle
  detrended <- x - trend
  seasonal <- numeric(n)
  for (s in 1:period) {
    idx <- seq(s, n, by = period)
    seasonal_mean <- mean(detrended[idx], na.rm = TRUE)
    seasonal[idx] <- seasonal_mean
  }
  # Center seasonal component
  seasonal <- seasonal - mean(seasonal, na.rm = TRUE)

  # Residual
  residual <- x - trend - seasonal

  list(
    observed = x,
    trend = trend,
    seasonal = seasonal,
    residual = residual,
    period = period
  )
}

# Demo: simulate monthly data with trend + seasonality
set.seed(42)
n <- 120  # 10 years of monthly data
t <- 1:n
trend <- 50 + 0.3 * t                        # linear trend
seasonal <- 10 * sin(2 * pi * t / 12)        # yearly cycle
noise <- rnorm(n, 0, 3)
y <- trend + seasonal + noise

result <- decompose_ts(y, period = 12)

cat("=== Time Series Decomposition ===\n")
cat(sprintf("  Observations: %d (%.0f years)\n", n, n / 12))
cat(sprintf("  Period: %d months\n", result$period))

# Show a few values
cat("\n  Month  Observed  Trend    Seasonal  Residual\n")
for (i in 13:24) {  # year 2
  cat(sprintf("  %3d    %6.1f   %6.1f    %6.1f    %6.1f\n",
              i, result$observed[i],
              ifelse(is.na(result$trend[i]), NA, result$trend[i]),
              result$seasonal[i],
              ifelse(is.na(result$residual[i]), NA, result$residual[i])))
}

cat(sprintf("\n  Residual SD: %.2f (noise was SD=3)\n",
            sd(result$residual, na.rm = TRUE)))
'''),
            ("pca_from_scratch", "PCA implementation from scratch", '''
# Principal Component Analysis from scratch using eigendecomposition.

pca <- function(X, n_components = NULL) {
  n <- nrow(X)
  p <- ncol(X)
  if (is.null(n_components)) n_components <- min(n, p)

  # Center the data
  means <- colMeans(X)
  X_centered <- sweep(X, 2, means)

  # Covariance matrix
  cov_matrix <- crossprod(X_centered) / (n - 1)

  # Eigendecomposition
  eig <- eigen(cov_matrix, symmetric = TRUE)
  values <- eig$values[1:n_components]
  vectors <- eig$vectors[, 1:n_components, drop = FALSE]

  # Project data
  scores <- X_centered %*% vectors

  total_var <- sum(eig$values)
  list(
    scores = scores,
    loadings = vectors,
    eigenvalues = values,
    variance_explained = values / total_var,
    cumulative_var = cumsum(values) / total_var,
    center = means
  )
}

# Demo: iris-like simulated data
set.seed(42)
n <- 150
# 3 clusters with correlated features
mu <- list(c(5, 3.5, 1.4, 0.2), c(6, 2.8, 4.5, 1.3), c(6.5, 3, 5.5, 2))
X <- do.call(rbind, lapply(1:3, function(k) {
  MASS::mvrnorm(n/3, mu[[k]], diag(4) * 0.3)
}))
colnames(X) <- paste0("V", 1:4)

result <- pca(X, n_components = 4)

cat("=== PCA Results ===\n")
cat("\nVariance explained:\n")
for (i in 1:4) {
  bar <- paste(rep("#", round(result$variance_explained[i] * 40)), collapse = "")
  cat(sprintf("  PC%d: %5.1f%% %s (cumulative: %.1f%%)\n",
              i, result$variance_explained[i] * 100, bar,
              result$cumulative_var[i] * 100))
}

cat("\nLoadings (first 2 PCs):\n")
cat(sprintf("  %-4s  %7s  %7s\n", "Var", "PC1", "PC2"))
for (j in 1:4) {
  cat(sprintf("  V%-3d  %7.3f  %7.3f\n", j, result$loadings[j, 1], result$loadings[j, 2]))
}
'''),
        ],
    },

    "javascript": {
        "ext": "js",
        "snippets": [
            ("debounce", "Debounce utility function", '''
/**
 * Creates a debounced version of a function that delays invocation
 * until after `wait` ms have elapsed since the last call.
 */
function debounce(fn, wait) {
  let timer = null;
  return function (...args) {
    clearTimeout(timer);
    timer = setTimeout(() => fn.apply(this, args), wait);
  };
}

// Demo
const log = debounce((msg) => console.log(`[debounced] ${msg}`), 300);
log("hello");
log("world"); // only this one fires after 300ms
'''),
            ("quick_sort", "Quicksort implementation", '''
/**
 * In-place quicksort using Lomuto partition scheme.
 */
function quickSort(arr, lo = 0, hi = arr.length - 1) {
  if (lo < hi) {
    const pivot = partition(arr, lo, hi);
    quickSort(arr, lo, pivot - 1);
    quickSort(arr, pivot + 1, hi);
  }
  return arr;
}

function partition(arr, lo, hi) {
  const pivot = arr[hi];
  let i = lo;
  for (let j = lo; j < hi; j++) {
    if (arr[j] <= pivot) {
      [arr[i], arr[j]] = [arr[j], arr[i]];
      i++;
    }
  }
  [arr[i], arr[hi]] = [arr[hi], arr[i]];
  return i;
}

console.log(quickSort([10, 7, 8, 9, 1, 5])); // [1, 5, 7, 8, 9, 10]
'''),
            ("event_emitter", "Minimal EventEmitter", '''
/**
 * Lightweight event emitter / pub-sub system.
 */
class EventEmitter {
  #listeners = new Map();

  on(event, fn) {
    if (!this.#listeners.has(event)) this.#listeners.set(event, []);
    this.#listeners.get(event).push(fn);
    return this;
  }

  off(event, fn) {
    const fns = this.#listeners.get(event);
    if (fns) this.#listeners.set(event, fns.filter((f) => f !== fn));
    return this;
  }

  emit(event, ...args) {
    for (const fn of this.#listeners.get(event) ?? []) fn(...args);
  }

  once(event, fn) {
    const wrapper = (...args) => { this.off(event, wrapper); fn(...args); };
    return this.on(event, wrapper);
  }
}

// Demo
const bus = new EventEmitter();
bus.on("greet", (name) => console.log(`Hello, ${name}!`));
bus.emit("greet", "world");
'''),
            ("promise_all", "Promise.all from scratch", '''
/**
 * Re-implementation of Promise.all for learning purposes.
 */
function promiseAll(promises) {
  return new Promise((resolve, reject) => {
    const results = new Array(promises.length);
    let remaining = promises.length;
    if (remaining === 0) return resolve(results);
    promises.forEach((p, i) => {
      Promise.resolve(p).then(
        (val) => {
          results[i] = val;
          if (--remaining === 0) resolve(results);
        },
        reject
      );
    });
  });
}

// Demo
promiseAll([
  Promise.resolve(1),
  new Promise((r) => setTimeout(() => r(2), 50)),
  Promise.resolve(3),
]).then(console.log); // [1, 2, 3]
'''),
            ("deep_clone", "Deep clone utility", '''
/**
 * Deep clone a value, handling objects, arrays, dates, maps, sets, and regexps.
 */
function deepClone(value) {
  if (value === null || typeof value !== "object") return value;
  if (value instanceof Date) return new Date(value.getTime());
  if (value instanceof RegExp) return new RegExp(value.source, value.flags);
  if (value instanceof Map) return new Map([...value].map(([k, v]) => [deepClone(k), deepClone(v)]));
  if (value instanceof Set) return new Set([...value].map(deepClone));
  if (Array.isArray(value)) return value.map(deepClone);
  const clone = Object.create(Object.getPrototypeOf(value));
  for (const key of Reflect.ownKeys(value)) {
    clone[key] = deepClone(value[key]);
  }
  return clone;
}

// Demo
const original = { a: [1, { b: 2 }], d: new Date(), s: new Set([3, 4]) };
const cloned = deepClone(original);
cloned.a[1].b = 99;
console.log(original.a[1].b); // 2 — unchanged
console.log(cloned.a[1].b);   // 99
'''),
        ],
    },

    "typescript": {
        "ext": "ts",
        "snippets": [
            ("result_type", "Rust-style Result type", '''
/**
 * Rust-inspired Result<T, E> type for explicit error handling in TypeScript.
 */
type Result<T, E> = { ok: true; value: T } | { ok: false; error: E };

function Ok<T>(value: T): Result<T, never> {
  return { ok: true, value };
}

function Err<E>(error: E): Result<never, E> {
  return { ok: false, error };
}

function divide(a: number, b: number): Result<number, string> {
  if (b === 0) return Err("division by zero");
  return Ok(a / b);
}

// Usage
const res = divide(10, 3);
if (res.ok) {
  console.log(`10 / 3 = ${res.value.toFixed(4)}`);
} else {
  console.error(res.error);
}
'''),
            ("binary_heap", "Generic min-heap", '''
/**
 * Generic min-heap (priority queue) implementation.
 */
class MinHeap<T> {
  private data: T[] = [];

  constructor(private compareFn: (a: T, b: T) => number = (a, b) => (a as any) - (b as any)) {}

  get size(): number { return this.data.length; }

  push(val: T): void {
    this.data.push(val);
    this.bubbleUp(this.data.length - 1);
  }

  pop(): T | undefined {
    if (this.data.length === 0) return undefined;
    const top = this.data[0];
    const last = this.data.pop()!;
    if (this.data.length > 0) {
      this.data[0] = last;
      this.sinkDown(0);
    }
    return top;
  }

  private bubbleUp(i: number): void {
    while (i > 0) {
      const parent = (i - 1) >> 1;
      if (this.compareFn(this.data[i], this.data[parent]) >= 0) break;
      [this.data[i], this.data[parent]] = [this.data[parent], this.data[i]];
      i = parent;
    }
  }

  private sinkDown(i: number): void {
    const n = this.data.length;
    while (true) {
      let smallest = i;
      const l = 2 * i + 1, r = 2 * i + 2;
      if (l < n && this.compareFn(this.data[l], this.data[smallest]) < 0) smallest = l;
      if (r < n && this.compareFn(this.data[r], this.data[smallest]) < 0) smallest = r;
      if (smallest === i) break;
      [this.data[i], this.data[smallest]] = [this.data[smallest], this.data[i]];
      i = smallest;
    }
  }
}

// Demo
const heap = new MinHeap<number>();
[5, 3, 8, 1, 2].forEach((n) => heap.push(n));
while (heap.size > 0) process.stdout.write(`${heap.pop()} `); // 1 2 3 5 8
'''),
            ("observer_pattern", "Type-safe Observer pattern", '''
/**
 * Type-safe Observer (pub/sub) pattern using TypeScript generics.
 */
type Listener<T> = (data: T) => void;

class Observable<EventMap extends Record<string, unknown>> {
  private listeners = new Map<keyof EventMap, Set<Listener<any>>>();

  on<K extends keyof EventMap>(event: K, listener: Listener<EventMap[K]>): () => void {
    if (!this.listeners.has(event)) this.listeners.set(event, new Set());
    this.listeners.get(event)!.add(listener);
    return () => this.listeners.get(event)?.delete(listener);
  }

  emit<K extends keyof EventMap>(event: K, data: EventMap[K]): void {
    this.listeners.get(event)?.forEach((fn) => fn(data));
  }
}

// Usage
interface AppEvents {
  login: { userId: string; timestamp: number };
  logout: { userId: string };
  error: { code: number; message: string };
}

const app = new Observable<AppEvents>();
const unsub = app.on("login", ({ userId, timestamp }) => {
  console.log(`User ${userId} logged in at ${new Date(timestamp).toISOString()}`);
});
app.emit("login", { userId: "sam", timestamp: Date.now() });
unsub();
'''),
            ("pipe_function", "Functional pipe and compose", '''
/**
 * Type-safe pipe utility for functional programming.
 */
type Fn<A, B> = (a: A) => B;

function pipe<A, B>(f: Fn<A, B>): Fn<A, B>;
function pipe<A, B, C>(f: Fn<A, B>, g: Fn<B, C>): Fn<A, C>;
function pipe<A, B, C, D>(f: Fn<A, B>, g: Fn<B, C>, h: Fn<C, D>): Fn<A, D>;
function pipe(...fns: Fn<any, any>[]): Fn<any, any> {
  return (x: any) => fns.reduce((acc, fn) => fn(acc), x);
}

// Demo
const slugify = pipe(
  (s: string) => s.toLowerCase(),
  (s: string) => s.replace(/[^a-z0-9]+/g, "-"),
  (s: string) => s.replace(/^-|-$/g, ""),
);

console.log(slugify("Hello, World! This is TypeScript")); // hello-world-this-is-typescript
'''),
        ],
    },

    "go": {
        "ext": "go",
        "snippets": [
            ("concurrent_worker_pool", "Concurrent worker pool", '''
package main

import (
\t"fmt"
\t"math/rand"
\t"sync"
\t"time"
)

func worker(id int, jobs <-chan int, results chan<- int, wg *sync.WaitGroup) {
\tdefer wg.Done()
\tfor j := range jobs {
\t\ttime.Sleep(time.Duration(rand.Intn(100)) * time.Millisecond)
\t\tfmt.Printf("Worker %d processed job %d\\n", id, j)
\t\tresults <- j * j
\t}
}

func main() {
\tconst numJobs = 10
\tconst numWorkers = 3

\tjobs := make(chan int, numJobs)
\tresults := make(chan int, numJobs)

\tvar wg sync.WaitGroup
\tfor w := 1; w <= numWorkers; w++ {
\t\twg.Add(1)
\t\tgo worker(w, jobs, results, &wg)
\t}

\tfor j := 1; j <= numJobs; j++ {
\t\tjobs <- j
\t}
\tclose(jobs)

\tgo func() {
\t\twg.Wait()
\t\tclose(results)
\t}()

\tsum := 0
\tfor r := range results {
\t\tsum += r
\t}
\tfmt.Printf("Sum of squares: %d\\n", sum)
}
'''),
            ("binary_search", "Generic binary search", '''
package main

import (
\t"cmp"
\t"fmt"
)

func BinarySearch[T cmp.Ordered](arr []T, target T) int {
\tlo, hi := 0, len(arr)-1
\tfor lo <= hi {
\t\tmid := lo + (hi-lo)/2
\t\tswitch {
\t\tcase arr[mid] == target:
\t\t\treturn mid
\t\tcase arr[mid] < target:
\t\t\tlo = mid + 1
\t\tdefault:
\t\t\thi = mid - 1
\t\t}
\t}
\treturn -1
}

func main() {
\tints := []int{2, 5, 8, 12, 16, 23, 38, 56, 72, 91}
\tfmt.Println("Index of 23:", BinarySearch(ints, 23))
\tfmt.Println("Index of 42:", BinarySearch(ints, 42))
}
'''),
            ("ring_buffer", "Lock-free ring buffer", '''
package main

import "fmt"

type RingBuffer[T any] struct {
\tbuf   []T
\tsize  int
\thead  int
\ttail  int
\tcount int
}

func NewRingBuffer[T any](capacity int) *RingBuffer[T] {
\treturn &RingBuffer[T]{buf: make([]T, capacity), size: capacity}
}

func (r *RingBuffer[T]) Push(v T) bool {
\tif r.count == r.size { return false }
\tr.buf[r.tail] = v
\tr.tail = (r.tail + 1) % r.size
\tr.count++
\treturn true
}

func (r *RingBuffer[T]) Pop() (T, bool) {
\tvar zero T
\tif r.count == 0 { return zero, false }
\tv := r.buf[r.head]
\tr.head = (r.head + 1) % r.size
\tr.count--
\treturn v, true
}

func main() {
\trb := NewRingBuffer[int](3)
\trb.Push(10); rb.Push(20); rb.Push(30)
\tv, _ := rb.Pop()
\tfmt.Println("Popped:", v)
\trb.Push(40)
\tfor rb.count > 0 {
\t\tv, _ := rb.Pop()
\t\tfmt.Println(v)
\t}
}
'''),
        ],
    },

    "rust": {
        "ext": "rs",
        "snippets": [
            ("binary_search", "Binary search implementation", '''
fn binary_search<T: Ord>(arr: &[T], target: &T) -> Option<usize> {
    let (mut lo, mut hi) = (0, arr.len());
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        match arr[mid].cmp(target) {
            std::cmp::Ordering::Equal => return Some(mid),
            std::cmp::Ordering::Less => lo = mid + 1,
            std::cmp::Ordering::Greater => hi = mid,
        }
    }
    None
}

fn main() {
    let data = vec![2, 5, 8, 12, 16, 23, 38, 56, 72, 91];
    println!("Search 23: {:?}", binary_search(&data, &23));
    println!("Search 42: {:?}", binary_search(&data, &42));
}
'''),
            ("iterator_combinators", "Custom iterator combinators", '''
struct Fibonacci { a: u64, b: u64 }

impl Fibonacci {
    fn new() -> Self { Fibonacci { a: 0, b: 1 } }
}

impl Iterator for Fibonacci {
    type Item = u64;
    fn next(&mut self) -> Option<Self::Item> {
        let val = self.a;
        (self.a, self.b) = (self.b, self.a + self.b);
        Some(val)
    }
}

fn main() {
    let fibs: Vec<u64> = Fibonacci::new().take(10).collect();
    println!("Fibonacci: {:?}", fibs);

    let sum: u64 = Fibonacci::new()
        .take_while(|&n| n < 4_000_000)
        .filter(|n| n % 2 == 0)
        .sum();
    println!("Sum of even fibs < 4M: {}", sum);
}
'''),
            ("hashmap_word_count", "Word frequency counter", '''
use std::collections::HashMap;

fn word_freq(text: &str) -> HashMap<String, usize> {
    let mut freq = HashMap::new();
    for word in text.split_whitespace() {
        let clean: String = word.chars()
            .filter(|c| c.is_alphanumeric())
            .collect::<String>()
            .to_lowercase();
        if !clean.is_empty() {
            *freq.entry(clean).or_insert(0) += 1;
        }
    }
    freq
}

fn main() {
    let text = "the quick brown fox jumps over the lazy dog the fox";
    let freq = word_freq(text);
    let mut pairs: Vec<_> = freq.iter().collect();
    pairs.sort_by(|a, b| b.1.cmp(a.1));
    for (word, count) in &pairs[..5.min(pairs.len())] {
        println!("{:>8}: {}", word, count);
    }
}
'''),
        ],
    },

    "ruby": {
        "ext": "rb",
        "snippets": [
            ("enumerable_extensions", "Enumerable method showcase", '''
# Ruby Enumerable methods with practical examples.

data = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]

evens, odds = data.partition(&:even?)
puts "Evens: #{evens}"
puts "Odds:  #{odds}"

grouped = data.group_by { |n| n < 5 ? :small : :large }
puts "Grouped: #{grouped}"

averages = data.each_with_object([]) do |n, acc|
  prev_sum = acc.empty? ? 0 : acc.last[:sum]
  prev_cnt = acc.empty? ? 0 : acc.last[:count]
  acc << { sum: prev_sum + n, count: prev_cnt + 1,
           avg: (prev_sum + n).to_f / (prev_cnt + 1) }
end
puts "Running averages: #{averages.map { |a| a[:avg].round(2) }}"

words = ["hello world", "hello ruby", "world class"]
freq = words.flat_map { |s| s.split }.tally
puts "Word freq: #{freq}"
'''),
            ("linked_list", "Linked list with Enumerable", '''
# Singly linked list that mixes in Enumerable.

class LinkedList
  include Enumerable
  Node = Struct.new(:value, :next_node)

  def initialize
    @head = nil
    @size = 0
  end

  attr_reader :size

  def push(value)
    @head = Node.new(value, @head)
    @size += 1
    self
  end

  def pop
    return nil if @head.nil?
    val = @head.value
    @head = @head.next_node
    @size -= 1
    val
  end

  def each
    node = @head
    while node
      yield node.value
      node = node.next_node
    end
  end

  def to_s
    "[#{to_a.join(" -> ")}]"
  end
end

list = LinkedList.new
[5, 4, 3, 2, 1].each { |n| list.push(n) }
puts list
puts "Sum:  #{list.sum}"
puts "Max:  #{list.max}"
puts "Even: #{list.select(&:even?)}"
'''),
        ],
    },

    "c": {
        "ext": "c",
        "snippets": [
            ("hash_table", "Hash table with chaining", '''
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TABLE_SIZE 64

typedef struct Entry {
    char *key;
    int value;
    struct Entry *next;
} Entry;

typedef struct { Entry *buckets[TABLE_SIZE]; } HashTable;

static unsigned int hash(const char *key) {
    unsigned int h = 5381;
    while (*key) h = ((h << 5) + h) + (unsigned char)*key++;
    return h % TABLE_SIZE;
}

HashTable *ht_create(void) { return calloc(1, sizeof(HashTable)); }

void ht_set(HashTable *ht, const char *key, int value) {
    unsigned int idx = hash(key);
    for (Entry *e = ht->buckets[idx]; e; e = e->next)
        if (strcmp(e->key, key) == 0) { e->value = value; return; }
    Entry *e = malloc(sizeof(Entry));
    e->key = strdup(key); e->value = value;
    e->next = ht->buckets[idx]; ht->buckets[idx] = e;
}

int ht_get(HashTable *ht, const char *key, int *out) {
    for (Entry *e = ht->buckets[hash(key)]; e; e = e->next)
        if (strcmp(e->key, key) == 0) { *out = e->value; return 1; }
    return 0;
}

int main(void) {
    HashTable *ht = ht_create();
    ht_set(ht, "alice", 42);
    ht_set(ht, "bob", 17);
    int val;
    if (ht_get(ht, "bob", &val)) printf("bob -> %d\\n", val);
    if (!ht_get(ht, "dave", &val)) printf("dave -> not found\\n");
    return 0;
}
'''),
            ("sieve_of_eratosthenes", "Sieve of Eratosthenes", '''
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int *sieve(int n, int *count) {
    char *is_prime = malloc(n + 1);
    memset(is_prime, 1, n + 1);
    is_prime[0] = is_prime[1] = 0;
    for (int i = 2; i <= (int)sqrt(n); i++)
        if (is_prime[i])
            for (int j = i * i; j <= n; j += i)
                is_prime[j] = 0;
    *count = 0;
    for (int i = 2; i <= n; i++) *count += is_prime[i];
    int *primes = malloc(*count * sizeof(int));
    int idx = 0;
    for (int i = 2; i <= n; i++)
        if (is_prime[i]) primes[idx++] = i;
    free(is_prime);
    return primes;
}

int main(void) {
    int count;
    int *primes = sieve(100, &count);
    printf("Primes up to 100 (%d total):\\n", count);
    for (int i = 0; i < count; i++) printf("%d ", primes[i]);
    printf("\\n");
    free(primes);
    return 0;
}
'''),
        ],
    },

    "java": {
        "ext": "java",
        "snippets": [
            ("generic_stack", "Generic stack implementation", '''
import java.util.Arrays;
import java.util.Iterator;
import java.util.NoSuchElementException;

public class GenericStack<T> implements Iterable<T> {
    @SuppressWarnings("unchecked")
    private T[] data = (T[]) new Object[4];
    private int size = 0;

    public void push(T item) {
        if (size == data.length) data = Arrays.copyOf(data, data.length * 2);
        data[size++] = item;
    }

    public T pop() {
        if (size == 0) throw new NoSuchElementException();
        T item = data[--size]; data[size] = null; return item;
    }

    public T peek() {
        if (size == 0) throw new NoSuchElementException();
        return data[size - 1];
    }

    public int size() { return size; }

    @Override
    public Iterator<T> iterator() {
        return new Iterator<>() {
            private int i = size - 1;
            public boolean hasNext() { return i >= 0; }
            public T next() { return data[i--]; }
        };
    }

    public static void main(String[] args) {
        GenericStack<Integer> stack = new GenericStack<>();
        for (int i = 1; i <= 5; i++) stack.push(i);
        System.out.println("Peek: " + stack.peek());
        for (int val : stack) System.out.print(val + " ");
        System.out.println();
    }
}
'''),
            ("binary_tree", "Binary search tree", '''
public class BinaryTree {
    private record Node(int value, Node left, Node right) {
        Node withLeft(Node l) { return new Node(value, l, right); }
        Node withRight(Node r) { return new Node(value, left, r); }
    }

    private Node root;

    public void insert(int value) { root = insert(root, value); }

    private Node insert(Node node, int value) {
        if (node == null) return new Node(value, null, null);
        if (value < node.value) return node.withLeft(insert(node.left, value));
        if (value > node.value) return node.withRight(insert(node.right, value));
        return node;
    }

    public boolean contains(int value) { return contains(root, value); }

    private boolean contains(Node node, int value) {
        if (node == null) return false;
        if (value == node.value) return true;
        return value < node.value ? contains(node.left, value) : contains(node.right, value);
    }

    public void inOrder() { inOrder(root); System.out.println(); }

    private void inOrder(Node node) {
        if (node == null) return;
        inOrder(node.left);
        System.out.print(node.value + " ");
        inOrder(node.right);
    }

    public static void main(String[] args) {
        BinaryTree tree = new BinaryTree();
        for (int v : new int[]{5, 3, 7, 1, 4, 6, 8}) tree.insert(v);
        tree.inOrder();
        System.out.println("Contains 4: " + tree.contains(4));
    }
}
'''),
        ],
    },
}

# ---------------------------------------------------------------------------
# Generator logic
# ---------------------------------------------------------------------------

PRETTY_NAMES = {
    "python": "Python",
    "r": "R",
    "javascript": "JavaScript",
    "typescript": "TypeScript",
    "go": "Go",
    "rust": "Rust",
    "ruby": "Ruby",
    "c": "C",
    "java": "Java",
}


def pick_language() -> str:
    """Weighted random language selection."""
    langs = list(WEIGHTS.keys())
    weights = [WEIGHTS[l] for l in langs]
    return random.choices(langs, weights=weights, k=1)[0]


def generate_snippet():
    """Pick a weighted-random language and snippet, write the file, return metadata."""
    now = datetime.now(timezone.utc)
    year = now.strftime("%Y")
    date_slug = now.strftime("%m-%d")

    lang_key = pick_language()
    lang = LANGUAGES[lang_key]
    ext = lang["ext"]

    title, description, code = random.choice(lang["snippets"])

    filename = f"{date_slug}-{lang_key}.{ext}"
    rel_path = f"snippets/{year}/{filename}"

    os.makedirs(os.path.dirname(rel_path), exist_ok=True)
    with open(rel_path, "w", encoding="utf-8") as f:
        f.write(code.strip() + "\n")

    pretty_lang = PRETTY_NAMES.get(lang_key, lang_key.capitalize())
    commit_msg = f"Add {description.lower()} in {pretty_lang}"

    return {
        "file": rel_path,
        "language": lang_key,
        "title": title,
        "description": description,
        "commit_message": commit_msg,
    }


if __name__ == "__main__":
    result = generate_snippet()
    for key, val in result.items():
        print(f"{key}={val}")
