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
