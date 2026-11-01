import {
  assertBudget,
  assertIntegrity,
  estimateDownload,
  type AssetPlan,
  type DownloadEstimate,
  type IntegrityReport,
} from "./assets.js";
import { assertTtsEngine, getDefaultEngines, type TtsEngine } from "./engine.js";
import { normalizeTtsError, TTS_ERROR_CODES, TtsError } from "./errors.js";
import { prepareText } from "./prepare.js";
import { createAudioStream } from "./stream.js";
import type {
  AudioChunk,
  AudioResult,
  DebugCallback,
  DebugEvent,
  OfflinePolicy,
  ProgressCallback,
  SynthesizeOptions,
  TtsCapabilities,
  TtsOptions,
  TtsRequest,
  VoiceInfo,
} from "./types.js";

/** Returns a high-resolution timestamp in milliseconds. */
function now(): number {
  return typeof performance === "undefined" ? Date.now() : performance.now();
}

/** Options for {@link Tts.prepare}. */
export interface PrepareAssetsOptions {
  /**
   * Whether provisioning may use the network. Default `false`: without an
   * explicit opt-in, `prepare()` only verifies what is already cached.
   */
  readonly allowNetwork?: boolean;
  /**
   * Download budget in decimal megabytes (1 MB = 1 000 000 bytes). Rejected
   * **before** anything is fetched. Omit for "no limit".
   */
  readonly maxDownloadMB?: number;
  /**
   * Whether to verify declared SHA-256 digests after provisioning.
   *
   * Default `false`: verifying a 68 MB profile costs a full read of every byte,
   * which is the wrong default for a warm cache. Pass `true` for a cold-machine
   * provisioning step or a build script, where a corrupt download must be caught
   * before it is relied on.
   */
  readonly verify?: boolean;
  /** Progress callback; defaults to the one passed to `createTts()`. */
  readonly onProgress?: ProgressCallback;
  /** Cancels the provisioning. */
  readonly signal?: AbortSignal;
}

/** Options for {@link Tts.verifyAssets}. */
export interface VerifyAssetsOptions {
  /** Cancels the verification. */
  readonly signal?: AbortSignal;
}

type Phase = "load" | "synthesize" | "stream";

/** Everything a `Tts` needs, with every optional field already resolved. */
interface TtsInit {
  readonly engine: TtsEngine;
  readonly request: TtsRequest;
  readonly offline: OfflinePolicy;
  readonly onProgress?: ProgressCallback;
  readonly onDebug?: DebugCallback;
}

/**
 * The application-facing handle: one engine, one language/voice, one network
 * policy, and the lifecycle that ties them together.
 *
 * Created through {@link createTts}, which is synchronous and never downloads
 * anything. Every operation normalizes errors into a `TtsError`, re-checks
 * disposal and abort after each `await`, and enforces the offline policy before
 * the engine is asked to synthesize.
 */
export class Tts {
  readonly #engine: TtsEngine;
  readonly #request: TtsRequest;
  readonly #offline: OfflinePolicy;
  readonly #onProgress: ProgressCallback | undefined;
  readonly #onDebug: DebugCallback | undefined;
  /** Internal abort handles of running streams, so `dispose()` can stop them. */
  readonly #activeStreams = new Set<AbortController>();
  #ready = false;
  #disposed = false;
  #disposePromise: Promise<void> | undefined;
  #loadPromise: Promise<void> | undefined;

  constructor(init: TtsInit) {
    this.#engine = init.engine;
    this.#request = init.request;
    this.#offline = init.offline;
    this.#onProgress = init.onProgress;
    this.#onDebug = init.onDebug;
  }

  /** Read-only copy of the request this instance synthesizes. */
  get request(): TtsRequest {
    return { ...this.#request };
  }

  /** Network policy this instance was created with. */
  get offline(): OfflinePolicy {
    return this.#offline;
  }

  /** Identifier of the bound engine, e.g. `"onnx"`. */
  get engineId(): string {
    return this.#engine.id;
  }

  /**
   * Whether the model is loaded and synthesis can start immediately.
   *
   * `false` does not mean "not cached": use {@link Tts.isCached} for that.
   */
  isReady(): boolean {
    return this.#ready && !this.#disposed;
  }

  /** Whether the instance has been disposed. */
  isDisposed(): boolean {
    return this.#disposed;
  }

  /** Voices the engine reports for its configured model. */
  voices(): readonly VoiceInfo[] {
    return this.#engine.voices();
  }

  /**
   * Resolved capabilities of the engine, or `undefined` when the engine has
   * none or the model is not loaded yet.
   */
  capabilities(): TtsCapabilities | undefined {
    return this.#engine.capabilities?.();
  }

  /**
   * Loads the model. Single-flight: concurrent calls share one engine load, and
   * a **rejected** load is not memoized, so a transient failure (offline blip,
   * aborted download) does not poison the instance.
   */
  /**
   * Loads the model. Single-flight: concurrent calls share one engine load, and
   * a **rejected** load is not memoized, so a transient failure (offline blip,
   * aborted download) does not poison the instance.
   *
   * In `offline: "strict"` mode this is the second of the two gates, not a way
   * around the first: the assets are checked exactly like a synthesis would, so
   * `await tts.preload()` cannot download what `synthesize()` would refuse.
   */
  async preload(): Promise<void> {
    this.#assertNotDisposed();
    if (this.#ready) {
      return;
    }
    if (!this.#loadPromise) {
      const start = now();
      this.#debug({ type: "load-start", timestamp: start, engine: this.#engine.id });
      const load = this.#assertOfflineUsable(undefined, "load")
        .then(() => this.#engine.load(this.#request, this.#onProgress, this.#onDebug))
        .then(() => {
          if (this.#disposed) {
            return;
          }
          this.#ready = true;
          this.#debug({
            type: "load-done",
            timestamp: now(),
            engine: this.#engine.id,
            durationMs: now() - start,
          });
        })
        .catch((error: unknown) => {
          // Deliberately un-memoized: a transient failure must not poison the
          // instance for its remaining lifetime.
          if (this.#loadPromise === load) {
            this.#loadPromise = undefined;
          }
          throw normalizeTtsError(
            error,
            TTS_ERROR_CODES.MODEL_LOAD_FAILED,
            "Failed to load the TTS model",
          );
        });
      this.#loadPromise = load;
    }
    await this.#loadPromise;
  }

  /**
   * Whether every asset for this instance's request is available locally.
   *
   * Prefers `engine.describeAssets()` and falls back to `engine.isCached()`.
   * Throws `NOT_SUPPORTED` when the engine offers neither: a silent `false`
   * would turn an engine limitation into a wrong offline diagnosis.
   */
  async isCached(): Promise<boolean> {
    this.#assertNotDisposed();
    const plan = await this.#describeAssets();
    if (plan) {
      return plan.complete;
    }
    const isCached = this.#engine.isCached;
    if (!isCached) {
      throw new TtsError(
        TTS_ERROR_CODES.NOT_SUPPORTED,
        `Engine ${this.#engine.id} cannot report its cache state`,
      );
    }
    try {
      return await isCached.call(this.#engine, this.#request);
    } catch (error) {
      throw normalizeTtsError(
        error,
        TTS_ERROR_CODES.OFFLINE_ASSETS_MISSING,
        "Failed to determine the local cache state",
      );
    }
  }

  /**
   * Per-file asset plan with cache state, sizes, and licenses.
   *
   * Throws `NOT_SUPPORTED` when the engine cannot describe its assets; the core
   * never fabricates an asset list.
   */
  async describeAssets(): Promise<AssetPlan> {
    this.#assertNotDisposed();
    const plan = await this.#describeAssets();
    if (!plan) {
      throw new TtsError(
        TTS_ERROR_CODES.NOT_SUPPORTED,
        `Engine ${this.#engine.id} does not implement describeAssets()`,
      );
    }
    return plan;
  }

  /**
   * What a consent dialog must show: how many bytes a download would cost and
   * which files are missing.
   *
   * Throws `NOT_SUPPORTED` when the engine cannot describe its assets.
   */
  async estimateDownload(): Promise<DownloadEstimate> {
    return estimateDownload(await this.describeAssets());
  }

  /**
   * Explicit, opt-in provisioning.
   *
   * 1. Describes the assets, so an engine that cannot report them is rejected
   *    before any I/O happens.
   * 2. Checks the budget **before** downloading.
   * 3. Without `allowNetwork`, only verifies that everything is already cached.
   * 4. Otherwise delegates to the engine's `prepare()`.
   *
   * Returns the asset plan that was checked, so callers can render the result.
   */
  async prepare(options: PrepareAssetsOptions = {}): Promise<AssetPlan> {
    this.#assertNotDisposed();
    this.#assertNotAborted(options.signal, "load");
    const plan = await this.describeAssets();
    try {
      assertBudget(plan, options.maxDownloadMB);
    } catch (error) {
      throw normalizeTtsError(
        error,
        TTS_ERROR_CODES.DOWNLOAD_BUDGET_EXCEEDED,
        "The download budget cannot be met",
      );
    }
    this.#assertNotDisposed();
    this.#assertNotAborted(options.signal, "load");

    if (!options.allowNetwork) {
      if (!plan.complete) {
        throw this.#offlineError(plan);
      }
      if (options.verify) {
        await this.#verify(options.signal);
      }
      return plan;
    }

    const prepare = this.#engine.prepare;
    if (!prepare) {
      // Without engine-side provisioning, the engine's own `load()` is the only
      // download path; it runs on the next preload()/synthesize().
      return plan;
    }
    const onProgress = options.onProgress ?? this.#onProgress;
    try {
      await prepare.call(this.#engine, this.#request, onProgress);
    } catch (error) {
      throw normalizeTtsError(
        error,
        TTS_ERROR_CODES.MODEL_LOAD_FAILED,
        "Failed to prepare the TTS assets",
      );
    }
    this.#assertNotDisposed();
    this.#assertNotAborted(options.signal, "load");
    if (options.verify) {
      await this.#verify(options.signal);
    }
    // Re-describe, so the returned plan reports what provisioning actually
    // produced instead of the pre-download snapshot the caller passed in.
    return (await this.#describeAssetsQuietly()) ?? plan;
  }

  /**
   * Verifies cached assets against their declared SHA-256 digests.
   *
   * Throws `ASSET_INTEGRITY_FAILED` when a digest mismatches or a file cannot be
   * read, and `NOT_SUPPORTED` when the engine cannot verify — an unverifiable
   * cache must not be reported as a verified one.
   */
  async verifyAssets(options: VerifyAssetsOptions = {}): Promise<IntegrityReport> {
    this.#assertNotDisposed();
    this.#assertNotAborted(options.signal, "load");
    const verify = this.#engine.verifyAssets;
    if (!verify) {
      throw new TtsError(
        TTS_ERROR_CODES.NOT_SUPPORTED,
        `Engine ${this.#engine.id} cannot verify its cached assets`,
      );
    }
    let report: IntegrityReport;
    try {
      report = await verify.call(this.#engine, this.#request);
    } catch (error) {
      throw normalizeTtsError(
        error,
        TTS_ERROR_CODES.ASSET_INTEGRITY_FAILED,
        "Failed to verify the cached assets",
      );
    }
    this.#assertNotDisposed();
    this.#assertNotAborted(options.signal, "load");
    return assertIntegrity(report);
  }

  /**
   * Synthesizes the whole text and returns mono PCM plus an explicit rate.
   *
   * Loads the model lazily. In `offline: "strict"` mode a missing asset rejects
   * with `OFFLINE_ASSETS_MISSING` **before** the engine is invoked.
   *
   * Whitespace-only input resolves to an empty result without touching the
   * model, so a UI cannot trigger a download by rendering an empty field.
   */
  async synthesize(text: string, options: SynthesizeOptions = {}): Promise<AudioResult> {
    this.#assertNotDisposed();
    this.#assertNotAborted(options.signal, "synthesize");
    const prepared = prepareText(text);
    if (prepared.length === 0) {
      return this.#emptyResult();
    }

    await this.#ensureReady(options, "synthesize");
    const start = now();
    this.#debug({
      type: "synthesize-start",
      timestamp: start,
      engine: this.#engine.id,
      textLength: prepared.length,
      ...(this.#request.voice !== undefined && { voice: this.#request.voice }),
    });
    let result: AudioResult;
    try {
      result = await this.#engine.synthesize(prepared, this.#request, passSignal(options));
    } catch (error) {
      throw normalizeTtsError(error, TTS_ERROR_CODES.SYNTHESIS_FAILED, "Synthesis failed");
    }
    this.#assertNotDisposed();
    this.#assertNotAborted(options.signal, "synthesize");
    const audio = validateResult(result, this.#engine.id, this.#request);
    this.#debug({
      type: "synthesize-done",
      timestamp: now(),
      engine: this.#engine.id,
      durationMs: now() - start,
      textLength: prepared.length,
      audioDurationMs: audio.durationMs,
      samples: audio.pcm.length,
      ...(this.#request.voice !== undefined && { voice: this.#request.voice }),
    });
    return audio;
  }

  /**
   * Streams audio for `text` as a bounded `AsyncIterable<AudioChunk>`.
   *
   * The model is loaded before the engine is asked for audio. In `offline:
   * "strict"` mode a cache miss fails the **iteration** rather than the call, so
   * `for await` observes it:
   *
   * ```ts
   * try {
   *   for await (const chunk of tts.stream("Hello")) sink.push(chunk);
   * } catch (error) {
   *   if (isTtsError(error)) console.error(error.code);
   * }
   * ```
   *
   * Aborting `options.signal` terminates the iterator with `TtsError(ABORTED)`;
   * chunks already delivered remain valid. Abandoning the iterator (`break`,
   * `return()`, an early `throw`) aborts the pending engine call too, so a consumer
   * that stops listening never leaves the engine generating into a stream nobody
   * reads — and never waits for it to finish.
   *
   * Empty input yields exactly one empty `final` chunk with `sampleRate: 0`
   * and never loads the model, so the chunk contract holds for every stream.
   */
  stream(text: string, options: SynthesizeOptions = {}): AsyncIterable<AudioChunk> {
    return {
      [Symbol.asyncIterator]: (): AsyncIterator<AudioChunk> => {
        // `#streamFrom` awaits the engine between chunks, and a generator cannot observe a
        // `return()` while it is suspended inside such an `await`. Aborting this controller
        // first is what makes abandonment interruptible: the pending engine call settles,
        // the generator unwinds, and `return()` reports "done" instead of deadlocking against
        // inference that has not finished yet.
        const abandoned = new AbortController();
        const forward = (): void => abandoned.abort();
        options.signal?.addEventListener("abort", forward, { once: true });
        const generator = this.#streamFrom(text, options, abandoned.signal);
        const release = (): void => {
          abandoned.abort();
          options.signal?.removeEventListener("abort", forward);
        };
        return {
          next: (): Promise<IteratorResult<AudioChunk>> => generator.next(),
          return: async (): Promise<IteratorResult<AudioChunk>> => {
            release();
            // Not awaited: the generator is suspended on an engine call that only settles once
            // that call aborts, which is exactly what `release()` just asked for.
            generator.return(undefined).catch(() => undefined);
            return { done: true, value: undefined };
          },
          throw: async (error?: unknown): Promise<IteratorResult<AudioChunk>> => {
            release();
            generator.return(undefined).catch(() => undefined);
            throw error;
          },
        };
      },
    };
  }

  /**
   * Removes the cached assets of this instance's request.
   *
   * Throws `NOT_SUPPORTED` when the engine cannot evict caches. The loaded state
   * is discarded first, so the next `preload()` loads again.
   */
  async removeModel(): Promise<void> {
    this.#assertNotDisposed();
    const remove = this.#engine.removeModel;
    if (!remove) {
      throw new TtsError(
        TTS_ERROR_CODES.NOT_SUPPORTED,
        `Engine ${this.#engine.id} does not support removeModel()`,
      );
    }
    this.#ready = false;
    this.#loadPromise = undefined;
    try {
      await remove.call(this.#engine, this.#request);
    } catch (error) {
      throw normalizeTtsError(error, TTS_ERROR_CODES.DISPOSE_FAILED, "Failed to remove the model");
    }
  }

  /**
   * Releases the engine and aborts every active stream. Idempotent: concurrent
   * and repeated calls share one disposal promise.
   */
  dispose(): Promise<void> {
    if (this.#disposePromise) {
      return this.#disposePromise;
    }
    this.#disposed = true;
    this.#ready = false;
    this.#loadPromise = undefined;
    for (const controller of this.#activeStreams) {
      controller.abort();
    }
    this.#activeStreams.clear();
    this.#disposePromise = (async () => {
      try {
        await this.#engine.dispose();
      } catch (error) {
        throw normalizeTtsError(
          error,
          TTS_ERROR_CODES.DISPOSE_FAILED,
          "Failed to dispose the engine",
        );
      }
    })();
    return this.#disposePromise;
  }

  async *#streamFrom(
    text: string,
    options: SynthesizeOptions,
    abandoned?: AbortSignal,
  ): AsyncGenerator<AudioChunk> {
    this.#assertNotDisposed();
    this.#assertNotAborted(options.signal, "stream");
    const prepared = prepareText(text);
    if (prepared.length === 0) {
      yield { pcm: new Float32Array(0), sampleRate: 0, sequence: 0, final: true };
      return;
    }

    // An internal controller lets `dispose()` and an abandoned consumer stop a stream that is
    // blocked inside the engine; the caller's signal is forwarded into it unchanged.
    const controller = new AbortController();
    const forwardAbandon = (): void => controller.abort();
    abandoned?.addEventListener("abort", forwardAbandon, { once: true });
    const forwardAbort = (): void => controller.abort();
    options.signal?.addEventListener("abort", forwardAbort, { once: true });
    this.#activeStreams.add(controller);
    const start = now();
    this.#debug({
      type: "stream-start",
      timestamp: start,
      engine: this.#engine.id,
      textLength: prepared.length,
      ...(this.#request.voice !== undefined && { voice: this.#request.voice }),
    });
    let chunks = 0;
    let audioDurationMs = 0;
    try {
      await this.#ensureReady(options, "stream");
      this.#assertNotDisposed();
      if (controller.signal.aborted) {
        throw this.#abortError("stream");
      }
      const source = this.#engine.stream(prepared, this.#request, {
        signal: controller.signal,
      });
      const stream = createAudioStream(source, {
        onChunk: (chunk) => {
          if (chunks === 0) {
            this.#debug({
              type: "time-to-first-chunk",
              timestamp: now(),
              engine: this.#engine.id,
              durationMs: now() - start,
            });
          }
          chunks += 1;
          if (chunk.sampleRate > 0) {
            audioDurationMs += (chunk.pcm.length / chunk.sampleRate) * 1000;
          }
        },
      });
      for await (const chunk of stream) {
        if (controller.signal.aborted) {
          throw this.#abortError("stream");
        }
        yield chunk;
      }
      this.#debug({
        type: "stream-done",
        timestamp: now(),
        engine: this.#engine.id,
        durationMs: now() - start,
        chunks,
        audioDurationMs,
      });
    } catch (error) {
      throw normalizeTtsError(error, TTS_ERROR_CODES.STREAM_FAILED, "Audio stream failed");
    } finally {
      options.signal?.removeEventListener("abort", forwardAbort);
      abandoned?.removeEventListener("abort", forwardAbandon);
      this.#activeStreams.delete(controller);
    }
  }

  async #ensureReady(options: SynthesizeOptions, phase: Phase): Promise<void> {
    if (this.#ready) {
      this.#assertNotDisposed();
      this.#assertNotAborted(options.signal, phase);
      return;
    }
    await this.#assertOfflineUsable(options.signal, phase);
    await this.preload();
    this.#assertNotDisposed();
    this.#assertNotAborted(options.signal, phase);
  }

  /**
   * The single strict-offline gate every path into `engine.load()` passes through.
   *
   * `synthesize()`, `stream()` and `preload()` all end up loading the model, and
   * loading is the one operation that can download. Enforcing the policy here
   * rather than at each call site is what makes "nothing downloads without
   * consent" a property of the class instead of a rule three methods remember.
   */
  async #assertOfflineUsable(signal: AbortSignal | undefined, phase: Phase): Promise<void> {
    if (this.#offline !== "strict") {
      return;
    }
    const plan = await this.#describeAssetsQuietly();
    this.#assertNotDisposed();
    this.#assertNotAborted(signal, phase);
    if (plan) {
      if (!plan.complete || plan.sizeMismatches.length > 0) {
        throw this.#offlineError(plan);
      }
    } else {
      // The engine cannot enumerate assets, so the only evidence left is its
      // own cache check. It is a required part of the contract.
      const cached = await this.#isCachedFromEngine(signal, phase);
      if (!cached) {
        throw this.#offlineError(undefined);
      }
    }
  }

  async #isCachedFromEngine(signal: AbortSignal | undefined, phase: Phase): Promise<boolean> {
    const isCached = this.#engine.isCached;
    if (!isCached) {
      throw new TtsError(
        TTS_ERROR_CODES.NOT_SUPPORTED,
        `Engine ${this.#engine.id} cannot report its cache state, so offline "strict" cannot be verified`,
      );
    }
    try {
      const cached = await isCached.call(this.#engine, this.#request);
      this.#assertNotDisposed();
      this.#assertNotAborted(signal, phase);
      return cached;
    } catch (error) {
      throw normalizeTtsError(
        error,
        TTS_ERROR_CODES.OFFLINE_ASSETS_MISSING,
        "Failed to determine the local cache state",
      );
    }
  }

  async #describeAssets(): Promise<AssetPlan | undefined> {
    const describe = this.#engine.describeAssets;
    if (!describe) {
      return undefined;
    }
    try {
      return await describe.call(this.#engine, this.#request);
    } catch (error) {
      throw normalizeTtsError(
        error,
        TTS_ERROR_CODES.OFFLINE_ASSETS_MISSING,
        "Failed to describe the model assets",
      );
    }
  }

  /** Like {@link Tts.#describeAssets}, but treats "cannot describe" as unknown. */
  async #describeAssetsQuietly(): Promise<AssetPlan | undefined> {
    try {
      return await this.#describeAssets();
    } catch {
      return undefined;
    }
  }

  /** Runs the integrity check through the public path, so both share one error contract. */
  async #verify(signal: AbortSignal | undefined): Promise<void> {
    await this.verifyAssets(signal === undefined ? {} : { signal });
  }

  #offlineError(plan: AssetPlan | undefined): TtsError {
    const missing = plan?.missing ?? [];
    const mismatched = plan?.sizeMismatches ?? [];
    const names =
      missing.length > 0
        ? missing.join(", ")
        : mismatched.length > 0
          ? `${mismatched.join(", ")} (size mismatch)`
          : (plan?.modelId ?? "unknown assets");
    return new TtsError(
      TTS_ERROR_CODES.OFFLINE_ASSETS_MISSING,
      `Offline mode is "strict" but the required assets are not usable: ${names}`,
      plan === undefined
        ? {}
        : {
            details: {
              modelId: plan.modelId,
              missing: plan.missing,
              missingBytes: plan.missingBytes,
              sizeMismatches: plan.sizeMismatches,
            },
          },
    );
  }

  #emptyResult(): AudioResult {
    return {
      pcm: new Float32Array(0),
      sampleRate: 0,
      channels: 1,
      durationMs: 0,
      engine: this.#engine.id,
      ...(this.#request.voice !== undefined && { voice: this.#request.voice }),
      ...(this.#request.language !== undefined && { language: this.#request.language }),
    };
  }

  #abortError(phase: Phase): TtsError {
    this.#debug({ type: "abort", timestamp: now(), engine: this.#engine.id, phase });
    return new TtsError(TTS_ERROR_CODES.ABORTED, "Operation aborted");
  }

  #assertNotDisposed(): void {
    if (this.#disposed) {
      throw new TtsError(TTS_ERROR_CODES.DISPOSED, "Tts has been disposed");
    }
  }

  #assertNotAborted(signal: AbortSignal | undefined, phase: Phase): void {
    if (signal?.aborted) {
      throw this.#abortError(phase);
    }
  }

  #debug(event: DebugEvent): void {
    this.#onDebug?.(event);
  }
}

/** Builds the exact-optional-safe options object an engine receives. */
function passSignal(options: SynthesizeOptions): SynthesizeOptions {
  return options.signal === undefined ? {} : { signal: options.signal };
}

/** Checks that an engine result is usable audio and fills in defaulted fields. */
function validateResult(result: AudioResult, engineId: string, request: TtsRequest): AudioResult {
  if (result === null || typeof result !== "object" || !(result.pcm instanceof Float32Array)) {
    throw new TtsError(
      TTS_ERROR_CODES.SYNTHESIS_FAILED,
      `Engine ${engineId} returned a result without Float32Array PCM`,
    );
  }
  const sampleRate = result.sampleRate;
  if (typeof sampleRate !== "number" || !Number.isFinite(sampleRate) || sampleRate < 0) {
    throw new TtsError(
      TTS_ERROR_CODES.SYNTHESIS_FAILED,
      `Engine ${engineId} returned an invalid sampleRate: ${String(sampleRate)}`,
      { details: { sampleRate } },
    );
  }
  const durationMs =
    sampleRate > 0 ? (result.pcm.length / sampleRate) * 1000 : (result.durationMs ?? 0);
  const voice = result.voice ?? request.voice;
  const language = result.language ?? request.language;
  return {
    pcm: result.pcm,
    sampleRate,
    channels: 1,
    durationMs,
    engine: engineId,
    ...(voice !== undefined && { voice }),
    ...(language !== undefined && { language }),
  };
}

/**
 * Creates a TTS instance. Synchronous and side-effect free: no model is
 * downloaded and no engine session is created.
 *
 * Throws `ENGINE_NOT_SUPPORTED` when no engine is available or the given engine
 * does not implement the contract, `LANGUAGE_NOT_SUPPORTED` /
 * `VOICE_NOT_SUPPORTED` when the engine rejects the selection, and
 * `INVALID_ARGUMENT` for an unknown `offline` value.
 */
export function createTts(options: TtsOptions): Tts {
  const engine = resolveEngine(options);
  const offline = options.offline ?? "strict";
  if (offline !== "strict" && offline !== "allow") {
    throw new TtsError(
      TTS_ERROR_CODES.INVALID_ARGUMENT,
      `offline must be "strict" or "allow", got ${String(offline)}`,
      { details: { offline } },
    );
  }
  const voices = engine.voices();
  const request = resolveRequest(voices, options);
  if (request.language.length === 0) {
    throw new TtsError(
      TTS_ERROR_CODES.LANGUAGE_NOT_SUPPORTED,
      `Engine ${engine.id} reports no language; pass TtsOptions.language`,
    );
  }
  if (!engine.supports(request)) {
    throw new TtsError(
      TTS_ERROR_CODES.LANGUAGE_NOT_SUPPORTED,
      `Engine ${engine.id} does not support language ${request.language}`,
      { details: { language: request.language, voice: request.voice } },
    );
  }
  if (
    request.voice !== undefined &&
    voices.length > 0 &&
    !voices.some((voice) => voice.id === request.voice)
  ) {
    throw new TtsError(
      TTS_ERROR_CODES.VOICE_NOT_SUPPORTED,
      `Engine ${engine.id} does not provide voice ${request.voice}`,
      { details: { voice: request.voice, language: request.language } },
    );
  }
  return new Tts({
    engine,
    request,
    offline,
    ...(options.onProgress !== undefined && { onProgress: options.onProgress }),
    ...(options.onDebug !== undefined && { onDebug: options.onDebug }),
  });
}

function resolveEngine(options: TtsOptions): TtsEngine {
  if (options.engine !== undefined) {
    assertTtsEngine(options.engine);
    return options.engine;
  }
  const candidate = options.engines?.[0];
  if (candidate !== undefined) {
    assertTtsEngine(candidate);
    return candidate;
  }
  const registered = getDefaultEngines()[0];
  if (registered) {
    return registered;
  }
  throw new TtsError(
    TTS_ERROR_CODES.ENGINE_NOT_SUPPORTED,
    "No TTS engine: pass TtsOptions.engine or call registerDefaultEngine()",
  );
}

/**
 * Resolves language and voice from the engine's voice list.
 *
 * An explicit `voice` wins. Otherwise the first voice speaks for the request — but when a
 * `language` is asked for, the first voice *of that language* is preferred, so a registry with
 * several languages does not hand a German request an English voice just because English was
 * registered first. Matching accepts a primary-subtag match (`"de"` for `"de-AT"`), because a
 * regional variant of a language is still that language for voice selection.
 *
 * An explicit `language` also wins over the voice's own language, so an engine may accept a
 * language/voice pairing it does not enumerate.
 */
function resolveRequest(voices: readonly VoiceInfo[], options: TtsOptions): TtsRequest {
  const primary = (language: string): string => language.split("-")[0] ?? language;
  const forLanguage = (language: string | undefined): VoiceInfo | undefined =>
    language === undefined
      ? undefined
      : (voices.find((voice) => voice.language === language) ??
        voices.find((voice) => primary(voice.language) === primary(language)));

  const selectedVoice = options.voice ?? forLanguage(options.language)?.id ?? voices[0]?.id;
  const voiceInfo = voices.find((candidate) => candidate.id === selectedVoice);
  const language = options.language ?? voiceInfo?.language ?? voices[0]?.language;
  return {
    language: language ?? "",
    ...(selectedVoice !== undefined && { voice: selectedVoice }),
  };
}
