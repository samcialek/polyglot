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
