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
