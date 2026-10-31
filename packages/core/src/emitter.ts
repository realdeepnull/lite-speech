/**
 * A minimal, framework-neutral, typed event emitter.
 *
 * Core has no runtime dependency on a framework or on the browser-only
 * `EventTarget`, so the same code runs in Node unit tests and in a browser.
 * The emitter is generic over a map of event names to argument tuples, which
 * keeps call sites fully typed.
 *
 * Listeners are called in subscription order and `emit()` is synchronous, so a
 * throwing listener propagates to the caller — event delivery is not silently
 * swallowed.
 */
export type ListenerMap = Record<string, unknown[]>;

export interface Emitter<Events extends ListenerMap> {
  /**
   * Subscribes to an event and returns an unsubscribe function. The same
   * listener may be subscribed more than once and is then called once per
   * subscription.
   */
  on<K extends keyof Events>(event: K, listener: (...args: Events[K]) => void): () => void;

  /** Subscribes to the next emission only. */
  once<K extends keyof Events>(event: K, listener: (...args: Events[K]) => void): () => void;

  /** Removes one occurrence of a listener. */
  off<K extends keyof Events>(event: K, listener: (...args: Events[K]) => void): void;

  /** Emits an event to every current listener, in subscription order. */
  emit<K extends keyof Events>(event: K, ...args: Events[K]): void;

  /** Number of listeners for an event, or for all events when omitted. */
  count<K extends keyof Events>(event?: K): number;

  /** Removes all listeners of one event, or of every event. */
  clear<K extends keyof Events>(event?: K): void;
}

/**
 * Creates a typed emitter.
 *
 * @example
 * ```ts
 * const emitter = createEmitter<{ chunk: [sequence: number] }>();
 * const off = emitter.on("chunk", (sequence) => console.log(sequence));
 * emitter.emit("chunk", 0);
 * off();
 * ```
 */
export function createEmitter<Events extends ListenerMap>(): Emitter<Events> {
  const listeners = new Map<keyof Events, Array<(...args: unknown[]) => void>>();

  const off = <K extends keyof Events>(event: K, listener: (...args: Events[K]) => void): void => {
    const set = listeners.get(event);
    if (!set) {
      return;
    }
    const index = set.indexOf(listener as (...args: unknown[]) => void);
    if (index !== -1) {
      set.splice(index, 1);
    }
    if (set.length === 0) {
      listeners.delete(event);
    }
  };

  const on = <K extends keyof Events>(
    event: K,
    listener: (...args: Events[K]) => void,
  ): (() => void) => {
    const set = listeners.get(event);
    if (set) {
      set.push(listener as (...args: unknown[]) => void);
    } else {
      listeners.set(event, [listener as (...args: unknown[]) => void]);
    }
    return () => off(event, listener);
  };

  const once = <K extends keyof Events>(
    event: K,
    listener: (...args: Events[K]) => void,
  ): (() => void) => {
    const wrapper = (...args: Events[K]): void => {
      off(event, wrapper);
      listener(...args);
    };
    return on(event, wrapper);
  };

  const emit = <K extends keyof Events>(event: K, ...args: Events[K]): void => {
    const set = listeners.get(event);
    if (!set) {
      return;
    }
    // Snapshot: a listener may unsubscribe (or subscribe) during emission.
    for (const listener of [...set]) {
      (listener as (...value: unknown[]) => void)(...args);
    }
  };

  const count = <K extends keyof Events>(event?: K): number => {
    if (event !== undefined) {
      return listeners.get(event)?.length ?? 0;
    }
    let total = 0;
    for (const set of listeners.values()) {
      total += set.length;
    }
    return total;
  };

  const clear = <K extends keyof Events>(event?: K): void => {
    if (event === undefined) {
      listeners.clear();
    } else {
      listeners.delete(event);
    }
  };

  return { on, once, off, emit, count, clear };
}
