import { getDefaultEngines, type TtsEngine } from "./engine.js";
import { TTS_ERROR_CODES, TtsError } from "./errors.js";
import { createTts, type Tts } from "./tts.js";
import type { DebugCallback, OfflinePolicy, ProgressCallback } from "./types.js";

/** Options for {@link TtsPool}. */
export interface TtsPoolOptions {
  /**
   * Engine shared by every pooled instance, or a factory that creates one per
   * instance.
   *
   * `dispose()` on a `Tts` disposes its engine, so an engine instance must not
   * outlive the instance that owns it. Pass a factory
   * (`() => createOnnxEngine()`) to give every cached entry its own engine; pass
   * a single engine only when the pool is the sole owner and entries are always
   * disposed together. A factory may be asynchronous, e.g. for a dynamic import.
   */
  readonly engine?: TtsEngine | (() => TtsEngine | Promise<TtsEngine>);
  /** Candidate engines; the first one is used when `engine` is omitted. */
  readonly engines?: readonly TtsEngine[];
  /** Network policy for every created instance. Default `"strict"`. */
  readonly offline?: OfflinePolicy;
  /** Progress callback forwarded to every created instance. */
  readonly onProgress?: ProgressCallback;
  /** Debug callback forwarded to every created instance. */
  readonly onDebug?: DebugCallback;
  /**
   * Maximum number of cached instances. When exceeded, the least recently used
   * entry is disposed. Must be a positive integer; default: unlimited.
   */
  readonly maxSize?: number;
}

/** A pool key as accepted by {@link TtsPool.switchTo}. */
export interface TtsPoolRequest {
  readonly language: string;
  readonly voice?: string;
}

/**
 * Returns the cache key for a language + voice pair.
 *
 * The key is stable and readable (`"en:female-1"`); a pool never conflates two
 * voices of the same language.
 */
export function ttsPoolKey(request: TtsPoolRequest): string {
  return `${request.language}:${request.voice ?? ""}`;
}

/**
 * A reusable cache of `Tts` instances keyed by language + voice.
 *
 * `switchTo()` returns a cached instance immediately or creates one. Concurrent
 * calls for the same key share a single creation promise, so the pool never
 * spawns two workers or triggers two downloads for one voice. `maxSize` enables
 * LRU eviction by `Map` insertion order; eviction is fire-and-forget so
 * `switchTo()` never blocks on tearing down an unused model.
 *
 * @example
 * ```ts
 * const pool = new TtsPool({ engine: () => createOnnxEngine(), maxSize: 2 });
 * const tts = await pool.switchTo({ language: "en", voice: "female-1" });
 * console.log(pool.cachedKeys()); // ["en:female-1"]
 * await pool.dispose();
 * ```
 */
export class TtsPool {
  /** Insertion order doubles as LRU order. */
  readonly #cache = new Map<string, Tts>();
  /** In-flight creations, so concurrent callers share one `Tts`. */
  readonly #creating = new Map<string, Promise<Tts>>();
  readonly #engine: TtsEngine | (() => TtsEngine | Promise<TtsEngine>) | undefined;
  readonly #engines: readonly TtsEngine[] | undefined;
  readonly #offline: OfflinePolicy | undefined;
  readonly #onProgress: ProgressCallback | undefined;
  readonly #onDebug: DebugCallback | undefined;
  readonly #maxSize: number | undefined;
  #currentKey: string | undefined;
  #disposed = false;

  constructor(options: TtsPoolOptions = {}) {
    if (
      options.maxSize !== undefined &&
      (!Number.isInteger(options.maxSize) || options.maxSize < 1)
    ) {
      throw new TtsError(
        TTS_ERROR_CODES.INVALID_ARGUMENT,
        `maxSize must be a positive integer, got ${String(options.maxSize)}`,
        { details: { maxSize: options.maxSize } },
      );
    }
    this.#engine = options.engine;
    this.#engines = options.engines;
    this.#offline = options.offline;
    this.#onProgress = options.onProgress;
    this.#onDebug = options.onDebug;
    this.#maxSize = options.maxSize;
  }

  /**
   * Returns the instance for `request`, creating it on first use.
   *
   * Concurrent calls for the same key await the same promise; a failed creation
   * is not cached, so a later call retries.
   */
  async switchTo(request: TtsPoolRequest): Promise<Tts> {
    this.#assertNotDisposed();
    const key = ttsPoolKey(request);
    const cached = this.#cache.get(key);
    if (cached) {
      this.#touch(key, cached);
      return cached;
    }
    const inFlight = this.#creating.get(key);
    if (inFlight) {
      return inFlight;
    }
    const creation = this.#create(request, key);
    this.#creating.set(key, creation);
    try {
      return await creation;
    } finally {
      this.#creating.delete(key);
    }
  }

  /**
   * Returns the cached instance for `request`, or `undefined`.
   *
   * Does not create anything and does not refresh the LRU order.
   */
  get(request: TtsPoolRequest): Tts | undefined {
    return this.#cache.get(ttsPoolKey(request));
  }

  /** The most recently used instance, or `undefined` when the pool is empty. */
  current(): Tts | undefined {
    return this.#currentKey === undefined ? undefined : this.#cache.get(this.#currentKey);
  }

  /** All cached keys, oldest first. */
  cachedKeys(): string[] {
    return [...this.#cache.keys()];
  }

  /** Number of cached instances. */
  get size(): number {
    return this.#cache.size;
  }

  /**
   * Disposes one cached instance and removes it from the pool.
   *
   * The entry stays tracked when disposal throws, so the caller can retry
   * instead of leaking an engine. If a creation for `key` is still in flight, it
   * is awaited first — otherwise the eviction request would be silently lost and
   * the instance would be cached right afterwards.
   */
  async disposeKey(key: string): Promise<void> {
    const inFlight = this.#creating.get(key);
    if (inFlight) {
      await inFlight.catch(() => undefined);
    }
    const instance = this.#cache.get(key);
    if (!instance) {
      return;
    }
    await instance.dispose();
    this.#cache.delete(key);
    this.#clearCurrent(key);
  }

  /**
   * Disposes every cached instance. Instances created by an in-flight
   * `switchTo()` are disposed instead of being cached, so a racing creation
   * cannot leak. The first disposal failure is re-thrown after all attempts.
   */
  async dispose(): Promise<void> {
    this.#disposed = true;
    const instances = [...this.#cache.values()];
    this.#cache.clear();
    this.#currentKey = undefined;
    const results = await Promise.allSettled(instances.map((instance) => instance.dispose()));
    const failure = results.find(
      (result): result is PromiseRejectedResult => result.status === "rejected",
    );
    if (failure) {
      throw failure.reason;
    }
  }

  async #create(request: TtsPoolRequest, key: string): Promise<Tts> {
    const instance = createTts({
      ...(await this.#engineOptions()),
      language: request.language,
      ...(request.voice !== undefined && { voice: request.voice }),
      ...(this.#offline !== undefined && { offline: this.#offline }),
      ...(this.#onProgress !== undefined && { onProgress: this.#onProgress }),
      ...(this.#onDebug !== undefined && { onDebug: this.#onDebug }),
    });
    if (this.#disposed) {
      await instance.dispose();
      throw new TtsError(TTS_ERROR_CODES.DISPOSED, "TtsPool has been disposed");
    }
    this.#cache.set(key, instance);
    this.#currentKey = key;
    this.#enforceMaxSize();
    return instance;
  }

  async #engineOptions(): Promise<{ engine?: TtsEngine; engines?: readonly TtsEngine[] }> {
    const engine = this.#engine;
    if (typeof engine === "function") {
      return { engine: await engine() };
    }
    if (engine !== undefined) {
      return { engine };
    }
    if (this.#engines !== undefined) {
      return { engines: this.#engines };
    }
    return { engines: getDefaultEngines() };
  }

  #touch(key: string, instance: Tts): void {
    this.#cache.delete(key);
    this.#cache.set(key, instance);
    this.#currentKey = key;
  }

  #clearCurrent(key: string): void {
    if (this.#currentKey !== key) {
      return;
    }
    this.#currentKey = [...this.#cache.keys()].at(-1);
  }

  #enforceMaxSize(): void {
    const maxSize = this.#maxSize;
    if (maxSize === undefined) {
      return;
    }
    while (this.#cache.size > maxSize) {
      const oldestKey = this.#cache.keys().next().value;
      if (oldestKey === undefined) {
        return;
      }
      const instance = this.#cache.get(oldestKey);
      this.#cache.delete(oldestKey);
      this.#clearCurrent(oldestKey);
      // Eviction must not block switchTo(); a failing teardown is swallowed so
      // it cannot surface as an unhandled rejection in an unrelated call.
      void instance?.dispose().catch(() => {});
    }
  }

  #assertNotDisposed(): void {
    if (this.#disposed) {
      throw new TtsError(TTS_ERROR_CODES.DISPOSED, "TtsPool has been disposed");
    }
  }
}
