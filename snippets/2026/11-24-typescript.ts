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
