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
