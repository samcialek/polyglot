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
