#!/usr/bin/env python3
"""
Polyglot Snippet Generator
Generates meaningful code snippets in 20+ programming languages.
Each snippet is a real algorithm, data structure, or utility implementation.
"""

import random
import os
import sys
from datetime import datetime, timezone

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
 * @param {number[]} arr
 * @param {number} lo
 * @param {number} hi
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
 * Resolves when all promises resolve; rejects on first rejection.
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
            ("linked_list", "Singly linked list", '''
/**
 * Singly linked list with push, pop, reverse, and iterator support.
 */
class ListNode {
  constructor(value, next = null) {
    this.value = value;
    this.next = next;
  }
}

class LinkedList {
  #head = null;
  #size = 0;

  get length() { return this.#size; }

  push(value) {
    this.#head = new ListNode(value, this.#head);
    this.#size++;
    return this;
  }

  pop() {
    if (!this.#head) return undefined;
    const val = this.#head.value;
    this.#head = this.#head.next;
    this.#size--;
    return val;
  }

  reverse() {
    let prev = null, curr = this.#head;
    while (curr) {
      const next = curr.next;
      curr.next = prev;
      prev = curr;
      curr = next;
    }
    this.#head = prev;
    return this;
  }

  *[Symbol.iterator]() {
    let node = this.#head;
    while (node) { yield node.value; node = node.next; }
  }
}

const list = new LinkedList();
[1, 2, 3, 4, 5].forEach((v) => list.push(v));
console.log([...list]);            // [5, 4, 3, 2, 1]
console.log([...list.reverse()]);  // [1, 2, 3, 4, 5]
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

function safeSqrt(n: number): Result<number, string> {
  if (n < 0) return Err("cannot sqrt negative number");
  return Ok(Math.sqrt(n));
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

  peek(): T | undefined { return this.data[0]; }

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
console.log();
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

// Usage — events are fully typed
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
unsub(); // clean up
'''),
            ("pipe_function", "Functional pipe and compose", '''
/**
 * Type-safe pipe and compose utilities for functional programming.
 */
type Fn<A, B> = (a: A) => B;

function pipe<A, B>(f: Fn<A, B>): Fn<A, B>;
function pipe<A, B, C>(f: Fn<A, B>, g: Fn<B, C>): Fn<A, C>;
function pipe<A, B, C, D>(f: Fn<A, B>, g: Fn<B, C>, h: Fn<C, D>): Fn<A, D>;
function pipe(...fns: Fn<any, any>[]): Fn<any, any> {
  return (x: any) => fns.reduce((acc, fn) => fn(acc), x);
}

// Demo: transform a string through a pipeline
const slugify = pipe(
  (s: string) => s.toLowerCase(),
  (s: string) => s.replace(/[^a-z0-9]+/g, "-"),
  (s: string) => s.replace(/^-|-$/g, ""),
);

console.log(slugify("Hello, World! This is TypeScript")); // hello-world-this-is-typescript
console.log(slugify("  Functional  Programming  101 ")); // functional-programming-101
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

// worker processes jobs from the jobs channel and sends results to results.
func worker(id int, jobs <-chan int, results chan<- int, wg *sync.WaitGroup) {
\tdefer wg.Done()
\tfor j := range jobs {
\t\t// Simulate work
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

// BinarySearch returns the index of target in a sorted slice, or -1.
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
\tfmt.Println("Index of 23:", BinarySearch(ints, 23)) // 5
\tfmt.Println("Index of 42:", BinarySearch(ints, 42)) // -1

\tstrs := []string{"alpha", "beta", "delta", "gamma"}
\tfmt.Println("Index of delta:", BinarySearch(strs, "delta")) // 2
}
'''),
            ("ring_buffer", "Lock-free ring buffer", '''
package main

import "fmt"

// RingBuffer is a fixed-size circular queue.
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
\tif r.count == r.size {
\t\treturn false // full
\t}
\tr.buf[r.tail] = v
\tr.tail = (r.tail + 1) % r.size
\tr.count++
\treturn true
}

func (r *RingBuffer[T]) Pop() (T, bool) {
\tvar zero T
\tif r.count == 0 {
\t\treturn zero, false
\t}
\tv := r.buf[r.head]
\tr.head = (r.head + 1) % r.size
\tr.count--
\treturn v, true
}

func (r *RingBuffer[T]) Len() int { return r.count }

func main() {
\trb := NewRingBuffer[int](3)
\trb.Push(10)
\trb.Push(20)
\trb.Push(30)
\tfmt.Println("Full?", !rb.Push(40)) // true
\tv, _ := rb.Pop()
\tfmt.Println("Popped:", v) // 10
\trb.Push(40)
\tfor rb.Len() > 0 {
\t\tv, _ := rb.Pop()
\t\tfmt.Println(v) // 20, 30, 40
\t}
}
'''),
        ],
    },

    "rust": {
        "ext": "rs",
        "snippets": [
            ("binary_search", "Binary search implementation", '''
/// Binary search on a sorted slice. Returns Some(index) or None.
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
    println!("Search 23: {:?}", binary_search(&data, &23)); // Some(5)
    println!("Search 42: {:?}", binary_search(&data, &42)); // None
}
'''),
            ("iterator_combinators", "Custom iterator combinators", '''
/// Generates Fibonacci numbers lazily via an iterator.
struct Fibonacci {
    a: u64,
    b: u64,
}

impl Fibonacci {
    fn new() -> Self {
        Fibonacci { a: 0, b: 1 }
    }
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
    // First 10 Fibonacci numbers
    let fibs: Vec<u64> = Fibonacci::new().take(10).collect();
    println!("Fibonacci: {:?}", fibs);

    // Sum of even Fibonacci numbers below 4 million
    let sum: u64 = Fibonacci::new()
        .take_while(|&n| n < 4_000_000)
        .filter(|n| n % 2 == 0)
        .sum();
    println!("Sum of even fibs < 4M: {}", sum);
}
'''),
            ("hashmap_word_count", "Word frequency counter with HashMap", '''
use std::collections::HashMap;

/// Count word frequencies in a string, case-insensitive.
fn word_freq(text: &str) -> HashMap<String, usize> {
    let mut freq = HashMap::new();
    for word in text.split_whitespace() {
        let clean: String = word
            .chars()
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
# Demonstrates Ruby\'s powerful Enumerable methods with practical examples.

data = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]

# Partition into evens and odds
evens, odds = data.partition(&:even?)
puts "Evens: #{evens}"
puts "Odds:  #{odds}"

# Group by magnitude
grouped = data.group_by { |n| n < 5 ? :small : :large }
puts "Grouped: #{grouped}"

# Running average using each_with_object
averages = data.each_with_object([]) do |n, acc|
  prev_sum = acc.empty? ? 0 : acc.last[:sum]
  prev_cnt = acc.empty? ? 0 : acc.last[:count]
  acc << { sum: prev_sum + n, count: prev_cnt + 1,
           avg: (prev_sum + n).to_f / (prev_cnt + 1) }
end
puts "Running averages: #{averages.map { |a| a[:avg].round(2) }}"

# Flat map + tally (Ruby 2.7+)
words = ["hello world", "hello ruby", "world class"]
freq = words.flat_map { |s| s.split }.tally
puts "Word freq: #{freq}"
'''),
            ("linked_list", "Linked list with Enumerable", '''
# Singly linked list that mixes in Enumerable for free methods.

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
puts list                          # [1 -> 2 -> 3 -> 4 -> 5]
puts "Sum:  #{list.sum}"          # 15
puts "Max:  #{list.max}"          # 5
puts "Even: #{list.select(&:even?)}" # [2, 4]
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

HashTable *ht_create(void) {
    HashTable *ht = calloc(1, sizeof(HashTable));
    return ht;
}

void ht_set(HashTable *ht, const char *key, int value) {
    unsigned int idx = hash(key);
    for (Entry *e = ht->buckets[idx]; e; e = e->next) {
        if (strcmp(e->key, key) == 0) { e->value = value; return; }
    }
    Entry *e = malloc(sizeof(Entry));
    e->key = strdup(key);
    e->value = value;
    e->next = ht->buckets[idx];
    ht->buckets[idx] = e;
}

int ht_get(HashTable *ht, const char *key, int *out) {
    unsigned int idx = hash(key);
    for (Entry *e = ht->buckets[idx]; e; e = e->next) {
        if (strcmp(e->key, key) == 0) { *out = e->value; return 1; }
    }
    return 0;
}

void ht_free(HashTable *ht) {
    for (int i = 0; i < TABLE_SIZE; i++) {
        Entry *e = ht->buckets[i];
        while (e) { Entry *next = e->next; free(e->key); free(e); e = next; }
    }
    free(ht);
}

int main(void) {
    HashTable *ht = ht_create();
    ht_set(ht, "alice", 42);
    ht_set(ht, "bob", 17);
    ht_set(ht, "charlie", 99);

    int val;
    if (ht_get(ht, "bob", &val)) printf("bob -> %d\\n", val);
    if (!ht_get(ht, "dave", &val)) printf("dave -> not found\\n");

    ht_free(ht);
    return 0;
}
'''),
            ("sieve_of_eratosthenes", "Sieve of Eratosthenes", '''
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/**
 * Sieve of Eratosthenes — find all primes up to n.
 * Returns a heap-allocated array; caller must free it.
 */
int *sieve(int n, int *count) {
    char *is_prime = malloc(n + 1);
    memset(is_prime, 1, n + 1);
    is_prime[0] = is_prime[1] = 0;

    for (int i = 2; i <= (int)sqrt(n); i++) {
        if (is_prime[i]) {
            for (int j = i * i; j <= n; j += i)
                is_prime[j] = 0;
        }
    }

    *count = 0;
    for (int i = 2; i <= n; i++) *count += is_prime[i];

    int *primes = malloc(*count * sizeof(int));
    int idx = 0;
    for (int i = 2; i <= n; i++) {
        if (is_prime[i]) primes[idx++] = i;
    }
    free(is_prime);
    return primes;
}

int main(void) {
    int count;
    int *primes = sieve(100, &count);
    printf("Primes up to 100 (%d total):\\n", count);
    for (int i = 0; i < count; i++) {
        printf("%d ", primes[i]);
    }
    printf("\\n");
    free(primes);
    return 0;
}
'''),
        ],
    },

    "cpp": {
        "ext": "cpp",
        "snippets": [
            ("smart_pointer", "Simplified unique_ptr implementation", '''
#include <iostream>
#include <utility>

/**
 * Simplified unique_ptr — demonstrates RAII and move semantics.
 */
template <typename T>
class UniquePtr {
    T *ptr_;

public:
    explicit UniquePtr(T *p = nullptr) : ptr_(p) {}
    ~UniquePtr() { delete ptr_; }

    // Move constructor & assignment
    UniquePtr(UniquePtr &&other) noexcept : ptr_(other.ptr_) { other.ptr_ = nullptr; }
    UniquePtr &operator=(UniquePtr &&other) noexcept {
        if (this != &other) { delete ptr_; ptr_ = other.ptr_; other.ptr_ = nullptr; }
        return *this;
    }

    // No copies
    UniquePtr(const UniquePtr &) = delete;
    UniquePtr &operator=(const UniquePtr &) = delete;

    T &operator*() const { return *ptr_; }
    T *operator->() const { return ptr_; }
    T *get() const { return ptr_; }
    explicit operator bool() const { return ptr_ != nullptr; }

    T *release() { T *p = ptr_; ptr_ = nullptr; return p; }
    void reset(T *p = nullptr) { delete ptr_; ptr_ = p; }
};

struct Point { double x, y; };

int main() {
    UniquePtr<Point> p(new Point{3.0, 4.0});
    std::cout << "Point(" << p->x << ", " << p->y << ")\\n";

    UniquePtr<Point> q = std::move(p);
    std::cout << "Moved: p is " << (p ? "valid" : "null")
              << ", q = (" << q->x << ", " << q->y << ")\\n";
    return 0;
}
'''),
            ("graph_bfs", "BFS on adjacency list graph", '''
#include <iostream>
#include <vector>
#include <queue>
#include <unordered_map>
#include <unordered_set>

using Graph = std::unordered_map<int, std::vector<int>>;

/**
 * Breadth-first search returning the visit order.
 */
std::vector<int> bfs(const Graph &g, int start) {
    std::vector<int> order;
    std::unordered_set<int> visited;
    std::queue<int> q;

    q.push(start);
    visited.insert(start);

    while (!q.empty()) {
        int node = q.front(); q.pop();
        order.push_back(node);
        if (g.count(node)) {
            for (int neighbor : g.at(node)) {
                if (!visited.count(neighbor)) {
                    visited.insert(neighbor);
                    q.push(neighbor);
                }
            }
        }
    }
    return order;
}

int main() {
    Graph g = {
        {1, {2, 3}},
        {2, {4, 5}},
        {3, {5}},
        {4, {}},
        {5, {6}},
        {6, {}},
    };

    auto order = bfs(g, 1);
    std::cout << "BFS order: ";
    for (int n : order) std::cout << n << " ";
    std::cout << "\\n";
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

/**
 * Resizable generic stack backed by an array.
 */
public class GenericStack<T> implements Iterable<T> {
    @SuppressWarnings("unchecked")
    private T[] data = (T[]) new Object[4];
    private int size = 0;

    public void push(T item) {
        if (size == data.length) data = Arrays.copyOf(data, data.length * 2);
        data[size++] = item;
    }

    public T pop() {
        if (size == 0) throw new NoSuchElementException("Stack is empty");
        T item = data[--size];
        data[size] = null; // help GC
        return item;
    }

    public T peek() {
        if (size == 0) throw new NoSuchElementException("Stack is empty");
        return data[size - 1];
    }

    public int size() { return size; }
    public boolean isEmpty() { return size == 0; }

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
/**
 * Binary search tree with insert, search, and in-order traversal.
 */
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
        return node; // duplicate
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
        int[] values = {5, 3, 7, 1, 4, 6, 8};
        for (int v : values) tree.insert(v);
        tree.inOrder();  // 1 2 3 4 5 6 7 8
        System.out.println("Contains 4: " + tree.contains(4));
        System.out.println("Contains 9: " + tree.contains(9));
    }
}
'''),
        ],
    },

    "kotlin": {
        "ext": "kt",
        "snippets": [
            ("sealed_state_machine", "Sealed class state machine", '''
/**
 * Models a network request lifecycle using Kotlin sealed classes.
 */
sealed class NetworkState<out T> {
    data object Idle : NetworkState<Nothing>()
    data object Loading : NetworkState<Nothing>()
    data class Success<T>(val data: T) : NetworkState<T>()
    data class Error(val message: String, val code: Int = -1) : NetworkState<Nothing>()
}

fun <T> NetworkState<T>.fold(
    onIdle: () -> String = { "Idle" },
    onLoading: () -> String = { "Loading..." },
    onSuccess: (T) -> String,
    onError: (String, Int) -> String = { msg, code -> "Error($code): $msg" }
): String = when (this) {
    is NetworkState.Idle -> onIdle()
    is NetworkState.Loading -> onLoading()
    is NetworkState.Success -> onSuccess(data)
    is NetworkState.Error -> onError(message, code)
}

data class User(val name: String, val email: String)

fun main() {
    val states = listOf(
        NetworkState.Idle,
        NetworkState.Loading,
        NetworkState.Success(User("Sam", "sam@example.com")),
        NetworkState.Error("Not Found", 404)
    )

    states.forEach { state ->
        val display = state.fold(
            onSuccess = { user -> "Got user: ${user.name} (${user.email})" }
        )
        println(display)
    }
}
'''),
            ("coroutine_flow", "Kotlin Flow example", '''
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*

/**
 * Demonstrates Kotlin coroutine Flows — reactive data streams.
 */

// Simulate a sensor emitting temperature readings
fun temperatureSensor(): Flow<Double> = flow {
    val base = 20.0
    var i = 0
    while (true) {
        emit(base + (i % 10) * 0.5 + Math.random() * 2)
        i++
        delay(100)
    }
}

fun main() = runBlocking {
    println("Temperature monitoring (5 readings, smoothed):")

    temperatureSensor()
        .take(10)
        .windowed(3) // sliding window of 3
        .map { window -> window.average() } // moving average
        .collect { avg ->
            println("  Avg temp: ${"%.2f".format(avg)}°C")
        }
}

// Extension: sliding window on Flow
fun Flow<Double>.windowed(size: Int): Flow<List<Double>> = flow {
    val buffer = mutableListOf<Double>()
    collect { value ->
        buffer.add(value)
        if (buffer.size >= size) {
            emit(buffer.toList())
            buffer.removeAt(0)
        }
    }
}
'''),
        ],
    },

    "bash": {
        "ext": "sh",
        "snippets": [
            ("file_organizer", "File organizer by extension", '''
#!/usr/bin/env bash
# Organizes files in a directory into subdirectories by extension.
# Usage: ./file_organizer.sh [directory]

set -euo pipefail

DIR="${1:-.}"

if [[ ! -d "$DIR" ]]; then
    echo "Error: '$DIR' is not a directory" >&2
    exit 1
fi

declare -A count

while IFS= read -r -d '' file; do
    ext="${file##*.}"
    if [[ "$ext" == "$file" || -z "$ext" ]]; then
        ext="no_extension"
    fi
    ext="${ext,,}" # lowercase

    target="$DIR/$ext"
    mkdir -p "$target"
    mv "$file" "$target/"
    count[$ext]=$(( ${count[$ext]:-0} + 1 ))
done < <(find "$DIR" -maxdepth 1 -type f -print0)

echo "=== Organization complete ==="
for ext in "${!count[@]}"; do
    printf "  %-15s %d files\\n" "$ext" "${count[$ext]}"
done
'''),
            ("parallel_ping", "Parallel host availability checker", '''
#!/usr/bin/env bash
# Check availability of multiple hosts in parallel.
# Usage: ./parallel_ping.sh host1 host2 host3 ...

set -euo pipefail

TIMEOUT=2
MAX_PARALLEL=10

RED=\'\\033[0;31m\'
GREEN=\'\\033[0;32m\'
NC=\'\\033[0m\'

check_host() {
    local host="$1"
    if ping -c 1 -W "$TIMEOUT" "$host" &>/dev/null; then
        printf "${GREEN}%-30s UP${NC}\\n" "$host"
    else
        printf "${RED}%-30s DOWN${NC}\\n" "$host"
    fi
}

hosts=("${@:-google.com github.com example.com localhost}")

echo "Checking ${#hosts[@]} hosts (timeout: ${TIMEOUT}s)..."
echo "---"

pids=()
for host in "${hosts[@]}"; do
    check_host "$host" &
    pids+=($!)

    # Throttle parallelism
    if (( ${#pids[@]} >= MAX_PARALLEL )); then
        wait "${pids[0]}"
        pids=("${pids[@]:1}")
    fi
done

# Wait for remaining
for pid in "${pids[@]}"; do
    wait "$pid"
done

echo "---"
echo "Done."
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
        ],
    },

    "lua": {
        "ext": "lua",
        "snippets": [
            ("class_system", "OOP class system via metatables", '''
--- Simple OOP class system using Lua metatables.

local function class(base)
    local cls = {}
    cls.__index = cls
    if base then
        setmetatable(cls, { __index = base })
    end

    function cls:new(...)
        local instance = setmetatable({}, cls)
        if instance.init then instance:init(...) end
        return instance
    end

    return cls
end

-- Base class: Shape
local Shape = class()

function Shape:init(name)
    self.name = name or "shape"
end

function Shape:area()
    return 0
end

function Shape:__tostring()
    return string.format("%s(area=%.2f)", self.name, self:area())
end

-- Circle extends Shape
local Circle = class(Shape)

function Circle:init(radius)
    Shape.init(self, "Circle")
    self.radius = radius
end

function Circle:area()
    return math.pi * self.radius ^ 2
end

-- Rectangle extends Shape
local Rect = class(Shape)

function Rect:init(w, h)
    Shape.init(self, "Rectangle")
    self.w = w
    self.h = h
end

function Rect:area()
    return self.w * self.h
end

-- Demo
local shapes = { Circle:new(5), Rect:new(4, 6), Circle:new(2.5) }
for _, s in ipairs(shapes) do
    print(tostring(s))
end
'''),
        ],
    },

    "perl": {
        "ext": "pl",
        "snippets": [
            ("text_processor", "Text processing pipeline", r'''
#!/usr/bin/env perl
# Text processing pipeline: word frequency analysis with regex magic.
use strict;
use warnings;

my $text = <<'END';
To be or not to be that is the question
Whether tis nobler in the mind to suffer
The slings and arrows of outrageous fortune
Or to take arms against a sea of troubles
And by opposing end them to die to sleep
END

# Word frequency count (case-insensitive)
my %freq;
$freq{lc($_)}++ for ($text =~ /\b(\w+)\b/g);

# Sort by frequency descending, then alphabetically
my @sorted = sort { $freq{$b} <=> $freq{$a} || $a cmp $b } keys %freq;

printf "%-15s %s\n", "WORD", "COUNT";
printf "%-15s %s\n", "-" x 15, "-" x 5;

for my $word (@sorted[0..9]) {
    printf "%-15s %d\n", $word, $freq{$word};
}

# Bonus: find all words that are palindromes
my @palindromes = grep { $_ eq reverse $_ && length $_ > 1 } keys %freq;
print "\nPalindromes found: @palindromes\n" if @palindromes;
print "Unique words: " . scalar(keys %freq) . "\n";
'''),
        ],
    },

    "php": {
        "ext": "php",
        "snippets": [
            ("collection_pipeline", "Functional collection pipeline", '''
<?php
/**
 * Functional collection pipeline — chainable array operations.
 */
class Collection implements IteratorAggregate, Countable {
    private array $items;

    public function __construct(array $items = []) {
        $this->items = array_values($items);
    }

    public static function of(array $items): self {
        return new self($items);
    }

    public function map(callable $fn): self {
        return new self(array_map($fn, $this->items));
    }

    public function filter(callable $fn): self {
        return new self(array_filter($this->items, $fn));
    }

    public function reduce(callable $fn, mixed $initial = null): mixed {
        return array_reduce($this->items, $fn, $initial);
    }

    public function flatMap(callable $fn): self {
        return new self(array_merge(...array_map($fn, $this->items)));
    }

    public function take(int $n): self {
        return new self(array_slice($this->items, 0, $n));
    }

    public function sortBy(callable $fn): self {
        $items = $this->items;
        usort($items, $fn);
        return new self($items);
    }

    public function unique(): self {
        return new self(array_unique($this->items));
    }

    public function toArray(): array { return $this->items; }
    public function count(): int { return count($this->items); }
    public function getIterator(): ArrayIterator { return new ArrayIterator($this->items); }
}

// Demo: process a list of names
$result = Collection::of(["Alice", "Bob", "Charlie", "alice", "DAVE", "bob"])
    ->map(fn($name) => strtolower($name))
    ->unique()
    ->filter(fn($name) => strlen($name) > 3)
    ->sortBy(fn($a, $b) => strcmp($a, $b))
    ->toArray();

echo "Processed: " . implode(", ", $result) . PHP_EOL;
// Output: Processed: alice, charlie, dave
'''),
        ],
    },

    "haskell": {
        "ext": "hs",
        "snippets": [
            ("quicksort", "Quicksort with list comprehensions", '''
-- | Elegant quicksort using list comprehensions.
-- Demonstrates Haskell\'s declarative style.

module Main where

quicksort :: (Ord a) => [a] -> [a]
quicksort []     = []
quicksort (x:xs) = quicksort smaller ++ [x] ++ quicksort bigger
  where
    smaller = [y | y <- xs, y <= x]
    bigger  = [y | y <- xs, y > x]

-- | Merge sort for comparison — stable sort.
mergesort :: (Ord a) => [a] -> [a]
mergesort []  = []
mergesort [x] = [x]
mergesort xs  = merge (mergesort left) (mergesort right)
  where
    (left, right) = splitAt (length xs `div` 2) xs
    merge [] ys = ys
    merge xs [] = xs
    merge (x:xs) (y:ys)
      | x <= y   = x : merge xs (y:ys)
      | otherwise = y : merge (x:xs) ys

-- | Check if a list is sorted.
isSorted :: (Ord a) => [a] -> Bool
isSorted []       = True
isSorted [_]      = True
isSorted (x:y:xs) = x <= y && isSorted (y:xs)

main :: IO ()
main = do
    let xs = [3, 6, 1, 8, 2, 9, 4, 7, 5]
    putStrLn $ "Original:   " ++ show xs
    putStrLn $ "Quicksort:  " ++ show (quicksort xs)
    putStrLn $ "Mergesort:  " ++ show (mergesort xs)
    putStrLn $ "Is sorted?  " ++ show (isSorted (quicksort xs))
'''),
            ("maybe_monad", "Maybe monad and safe operations", '''
-- | Demonstrates the Maybe monad for safe, composable computations.

module Main where

import Data.Char (digitToInt, isDigit)

-- | Safe division that returns Nothing for division by zero.
safeDiv :: Double -> Double -> Maybe Double
safeDiv _ 0 = Nothing
safeDiv x y = Just (x / y)

-- | Safe square root for non-negative numbers.
safeSqrt :: Double -> Maybe Double
safeSqrt x
  | x < 0    = Nothing
  | otherwise = Just (sqrt x)

-- | Parse a string as a positive integer.
safeParseNat :: String -> Maybe Int
safeParseNat s
  | null s         = Nothing
  | all isDigit s  = Just (foldl (\\acc c -> acc * 10 + digitToInt c) 0 s)
  | otherwise      = Nothing

-- | Chain safe operations using do-notation (monadic bind).
compute :: Double -> Double -> Maybe Double
compute x y = do
    ratio  <- safeDiv x y
    result <- safeSqrt ratio
    safeDiv result 2.0

main :: IO ()
main = do
    putStrLn "Safe computations with Maybe:"
    print $ compute 100 25    -- Just 1.0
    print $ compute 100 0     -- Nothing (div by zero)
    print $ compute (-16) 1   -- Nothing (negative sqrt)
    putStrLn ""
    putStrLn "Safe parsing:"
    print $ safeParseNat "42"   -- Just 42
    print $ safeParseNat "-5"   -- Nothing
    print $ safeParseNat ""     -- Nothing
'''),
        ],
    },

    "scala": {
        "ext": "scala",
        "snippets": [
            ("pattern_matching", "Advanced pattern matching", '''
/**
 * Demonstrates Scala\'s powerful pattern matching with sealed traits,
 * extractors, and guards.
 */
object PatternMatching extends App {

  // Algebraic data type for arithmetic expressions
  sealed trait Expr
  case class Num(value: Double) extends Expr
  case class Add(left: Expr, right: Expr) extends Expr
  case class Mul(left: Expr, right: Expr) extends Expr
  case class Var(name: String) extends Expr

  type Env = Map[String, Double]

  def eval(expr: Expr, env: Env = Map.empty): Option[Double] = expr match {
    case Num(v) => Some(v)
    case Var(name) => env.get(name)
    case Add(l, r) =>
      for { a <- eval(l, env); b <- eval(r, env) } yield a + b
    case Mul(l, r) =>
      for { a <- eval(l, env); b <- eval(r, env) } yield a * b
  }

  def simplify(expr: Expr): Expr = expr match {
    case Add(Num(0), e) => simplify(e)
    case Add(e, Num(0)) => simplify(e)
    case Mul(Num(1), e) => simplify(e)
    case Mul(e, Num(1)) => simplify(e)
    case Mul(Num(0), _) => Num(0)
    case Mul(_, Num(0)) => Num(0)
    case Add(l, r)      => Add(simplify(l), simplify(r))
    case Mul(l, r)      => Mul(simplify(l), simplify(r))
    case other           => other
  }

  def prettyPrint(expr: Expr): String = expr match {
    case Num(v) => if (v == v.toInt) v.toInt.toString else v.toString
    case Var(n) => n
    case Add(l, r) => s"(${prettyPrint(l)} + ${prettyPrint(r)})"
    case Mul(l, r) => s"(${prettyPrint(l)} * ${prettyPrint(r)})"
  }

  // Demo
  val expr = Add(Mul(Num(2), Var("x")), Mul(Num(0), Var("y")))
  val env = Map("x" -> 5.0, "y" -> 3.0)

  println(s"Expression: ${prettyPrint(expr)}")
  println(s"Simplified: ${prettyPrint(simplify(expr))}")
  println(s"Evaluated:  ${eval(expr, env)}")
}
'''),
        ],
    },

    "swift": {
        "ext": "swift",
        "snippets": [
            ("protocol_oriented", "Protocol-oriented design", '''
import Foundation

/// Protocol-oriented design: composable, testable geometry.

protocol Shape {
    var area: Double { get }
    var perimeter: Double { get }
    var description: String { get }
}

extension Shape {
    var description: String {
        String(format: "%@ — area: %.2f, perimeter: %.2f",
               String(describing: type(of: self)), area, perimeter)
    }
}

struct Circle: Shape {
    let radius: Double
    var area: Double { .pi * radius * radius }
    var perimeter: Double { 2 * .pi * radius }
}

struct Rectangle: Shape {
    let width: Double
    let height: Double
    var area: Double { width * height }
    var perimeter: Double { 2 * (width + height) }
}

struct Triangle: Shape {
    let a: Double, b: Double, c: Double
    var perimeter: Double { a + b + c }
    var area: Double {
        let s = perimeter / 2
        return (s * (s - a) * (s - b) * (s - c)).squareRoot()
    }
}

// Generic function works with any Shape
func largest<S: Shape>(_ shapes: [S]) -> S? {
    shapes.max(by: { $0.area < $1.area })
}

// Demo
let shapes: [any Shape] = [
    Circle(radius: 5),
    Rectangle(width: 8, height: 6),
    Triangle(a: 3, b: 4, c: 5)
]

for shape in shapes {
    print(shape.description)
}
'''),
        ],
    },

    "zig": {
        "ext": "zig",
        "snippets": [
            ("array_list", "Dynamic array implementation", '''
const std = @import("std");
const Allocator = std.mem.Allocator;

/// A simplified dynamic array (ArrayList) implementation in Zig.
fn ArrayList(comptime T: type) type {
    return struct {
        const Self = @This();

        items: []T,
        capacity: usize,
        len: usize,
        allocator: Allocator,

        pub fn init(allocator: Allocator) Self {
            return .{
                .items = &[_]T{},
                .capacity = 0,
                .len = 0,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *Self) void {
            if (self.capacity > 0) {
                self.allocator.free(self.items.ptr[0..self.capacity]);
            }
        }

        pub fn append(self: *Self, item: T) !void {
            if (self.len >= self.capacity) {
                try self.grow();
            }
            self.items.ptr[self.len] = item;
            self.len += 1;
            self.items.len = self.len;
        }

        fn grow(self: *Self) !void {
            const new_cap = if (self.capacity == 0) 4 else self.capacity * 2;
            const new_mem = try self.allocator.alloc(T, new_cap);
            if (self.len > 0) {
                @memcpy(new_mem[0..self.len], self.items.ptr[0..self.len]);
            }
            if (self.capacity > 0) {
                self.allocator.free(self.items.ptr[0..self.capacity]);
            }
            self.items = new_mem[0..self.len];
            self.capacity = new_cap;
        }
    };
}

pub fn main() !void {
    const stdout = std.io.getStdOut().writer();
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var list = ArrayList(i32).init(allocator);
    defer list.deinit();

    for (0..10) |i| {
        try list.append(@intCast(i * i));
    }

    try stdout.print("Squares: ");
    for (list.items) |item| {
        try stdout.print("{} ", .{item});
    }
    try stdout.print("\\n", .{});
}
'''),
        ],
    },

    "elixir": {
        "ext": "ex",
        "snippets": [
            ("genserver_counter", "GenServer counter process", '''
defmodule Counter do
  @moduledoc """
  A simple counter implemented as a GenServer.
  Demonstrates Elixir\'s actor model and OTP patterns.
  """
  use GenServer

  # Client API

  def start_link(initial \\\\ 0) do
    GenServer.start_link(__MODULE__, initial, name: __MODULE__)
  end

  def increment(amount \\\\ 1), do: GenServer.call(__MODULE__, {:increment, amount})
  def decrement(amount \\\\ 1), do: GenServer.call(__MODULE__, {:decrement, amount})
  def value, do: GenServer.call(__MODULE__, :value)
  def reset, do: GenServer.cast(__MODULE__, :reset)

  # Server callbacks

  @impl true
  def init(initial), do: {:ok, initial}

  @impl true
  def handle_call({:increment, amount}, _from, state) do
    new_state = state + amount
    {:reply, new_state, new_state}
  end

  def handle_call({:decrement, amount}, _from, state) do
    new_state = state - amount
    {:reply, new_state, new_state}
  end

  def handle_call(:value, _from, state) do
    {:reply, state, state}
  end

  @impl true
  def handle_cast(:reset, _state) do
    {:noreply, 0}
  end
end

defmodule Pipeline do
  @moduledoc "Enum pipeline examples — Elixir\'s bread and butter."

  def word_frequencies(text) do
    text
    |> String.downcase()
    |> String.split(~r/[^a-z]+/, trim: true)
    |> Enum.frequencies()
    |> Enum.sort_by(fn {_word, count} -> -count end)
  end
end

# Demo (script mode)
text = "to be or not to be that is the question"
Pipeline.word_frequencies(text)
|> Enum.each(fn {word, count} ->
  IO.puts("  #{String.pad_trailing(word, 12)} #{count}")
end)
'''),
            ("pattern_matching", "Pattern matching showcase", '''
defmodule Shapes do
  @moduledoc """
  Pattern matching and guards in Elixir — computing areas
  of geometric shapes using multiple function clauses.
  """

  @type shape ::
    {:circle, number()} |
    {:rectangle, number(), number()} |
    {:triangle, number(), number(), number()}

  @spec area(shape()) :: {:ok, float()} | {:error, String.t()}
  def area({:circle, radius}) when radius > 0 do
    {:ok, :math.pi() * radius * radius}
  end

  def area({:rectangle, w, h}) when w > 0 and h > 0 do
    {:ok, w * h * 1.0}
  end

  def area({:triangle, a, b, c}) when a > 0 and b > 0 and c > 0 do
    s = (a + b + c) / 2
    val = s * (s - a) * (s - b) * (s - c)
    if val > 0, do: {:ok, :math.sqrt(val)}, else: {:error, "invalid triangle"}
  end

  def area(_), do: {:error, "invalid shape"}

  def describe(shape) do
    case area(shape) do
      {:ok, a} -> "#{inspect(shape)} has area #{Float.round(a, 2)}"
      {:error, msg} -> "#{inspect(shape)}: #{msg}"
    end
  end
end

shapes = [
  {:circle, 5},
  {:rectangle, 4, 6},
  {:triangle, 3, 4, 5},
  {:circle, -1}
]

Enum.each(shapes, fn s -> IO.puts(Shapes.describe(s)) end)
'''),
        ],
    },
}

# ---------------------------------------------------------------------------
# Generator logic
# ---------------------------------------------------------------------------

def generate_snippet():
    """Pick a random language and snippet, write the file, return metadata."""
    now = datetime.now(timezone.utc)
    year = now.strftime("%Y")
    date_slug = now.strftime("%m-%d")

    # Pick random language
    lang_key = random.choice(list(LANGUAGES.keys()))
    lang = LANGUAGES[lang_key]
    ext = lang["ext"]

    # Pick random snippet
    title, description, code = random.choice(lang["snippets"])

    # Build file path
    filename = f"{date_slug}-{lang_key}.{ext}"
    rel_path = f"snippets/{year}/{filename}"

    # Write file
    os.makedirs(os.path.dirname(rel_path), exist_ok=True)
    with open(rel_path, "w", encoding="utf-8") as f:
        f.write(code.strip() + "\n")

    # Build commit message
    pretty_lang = lang_key.replace("cpp", "C++").replace("bash", "Bash").replace("typescript", "TypeScript").replace("javascript", "JavaScript").replace("haskell", "Haskell").replace("kotlin", "Kotlin").replace("elixir", "Elixir").replace("python", "Python").replace("ruby", "Ruby").replace("rust", "Rust").replace("scala", "Scala").replace("swift", "Swift")
    pretty_lang = pretty_lang[0].upper() + pretty_lang[1:] if pretty_lang[0].islower() else pretty_lang
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
    # Output for GitHub Actions to consume
    for key, val in result.items():
        print(f"{key}={val}")
