# 🌍 Polyglot

**Daily code snippets in 20+ programming languages.**

A living collection of small, meaningful algorithm implementations, data structures, and design patterns — one per day, in a different language each time.

## Languages

Python · JavaScript · TypeScript · Go · Rust · Ruby · C · C++ · Java · Kotlin · Bash · R · Lua · Perl · PHP · Haskell · Scala · Swift · Zig · Elixir

## Structure

```
snippets/
  2026/
    01-15-rust.rs       # Binary search in Rust
    01-16-python.py     # LRU cache in Python
    01-17-haskell.hs    # Maybe monad in Haskell
    ...
```

## How It Works

A [GitHub Actions workflow](.github/workflows/daily-commit.yml) runs daily and:

1. Picks a random language from the rotation
2. Selects a meaningful snippet — real implementations, not boilerplate
3. Commits it to `snippets/YYYY/MM-DD-language.ext`
4. Pushes to main

The generator (`generate.py`) contains a curated library of ~50+ snippets across all languages, covering:

- **Algorithms** — binary search, merge sort, quicksort, Dijkstra's
- **Data structures** — linked lists, tries, heaps, hash tables, ring buffers
- **Design patterns** — observer, state machines, RAII
- **Utilities** — debounce, deep clone, LRU cache, bloom filters
- **Functional** — monads, pipes, pattern matching
- **Concurrency** — worker pools, coroutine flows, GenServers

## Running Locally

```bash
cd polyglot
python generate.py
# Outputs: file=snippets/2026/02-01-rust.rs
#          language=rust
#          title=binary_search
#          description=Binary search implementation
#          commit_message=Add binary search implementation in Rust
```

## License

Public domain. Steal anything you like.
