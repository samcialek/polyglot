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
