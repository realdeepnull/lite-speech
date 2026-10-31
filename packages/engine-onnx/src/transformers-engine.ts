/**
 * The `TtsEngine` implementation.
 *
 * Owns the worker — or, where `Worker` does not exist (Node.js), an in-thread host behind
 * the same dispatch boundary — correlates request ids, maps failures onto the core's
 * stable error codes, and turns the pull-based frame protocol into the two shapes the core
 * asks for: one `AudioResult` and an `AsyncIterable<AudioChunk>`.
 *
 * Two behaviours are easy to get subtly wrong, so they are stated once here:
 *
 * - **`final` is the engine's job.** The worker cannot know a frame is last until its
 *   generator ends, so the engine emits the single `final: true` chunk itself when the
 *   session reports `stream-end`. That keeps time-to-first-audio free of look-ahead.
 * - **Abort means "stop waiting", not "stop computing".** ONNX cannot cancel mid-inference,
 *   so aborting rejects the caller, drops the pending entry (late frames then find no entry
 *   and are discarded) and fires a cooperative `cancel` so the worker stops between frames.
 *   The worker stays reusable afterwards, which is the behaviour the milestone gate checks.
 */
import {
  createAssetPlan,
  isAbortError,
  isTtsError,
  normalizeTtsError,
  TTS_ERROR_CODES,
  TtsError,
  type AssetPlan,
  type AssetPlanFile,
  type AudioChunk,
  type AudioResult,
  type DebugCallback,
  type DebugEvent,
  type ModelManifest,
  type ProgressCallback,
  type StreamingClass,
  type SynthesizeOptions,
  type TtsCapabilities,
  type TtsEngine,
  type TtsErrorCode,
  type TtsRequest,
  type VoiceInfo,
  type VoiceManifest,
} from "@lite-speech/core";

import {
  DTYPE_SUFFIX,
  ENGINE_ID,
  PHONON_VOICE_GENDERS,
  PHONON_VOICE_NAMES,
  assetUrl,
  voiceAccent,
  type EngineRegistry,
} from "./models.js";
import { createModelHost, type ModelHost } from "./model-host.js";
import type { LoadedPhonon } from "./phonon.js";
import type {
  ErrorResponse,
  ProgressResponse,
  WorkerRequest,
  WorkerResponse,
} from "./worker-protocol.js";
import type { ResolvedDtype } from "./webgpu.js";

/// How long dispose() waits for the worker to acknowledge a release before terminating it.
const DISPOSE_ACK_TIMEOUT_MS = 2_000;

/** Returns a high-resolution timestamp in milliseconds. */
function now(): number {
  return typeof performance === "undefined" ? Date.now() : performance.now();
}

/** One pending worker request and the callback its intermediate responses feed. */
interface PendingRequest {
  resolve(response: WorkerResponse): void;
  reject(error: unknown): void;
  onEvent?: ((response: WorkerResponse) => void) | undefined;
}

/** What one successful load resolved, so cache URLs and capabilities agree with reality. */
interface LoadedState {
  readonly model: ModelManifest;
  readonly voice: VoiceManifest | undefined;
  readonly device: string;
  readonly dtype: ResolvedDtype;
  readonly sampleRate: number;
  readonly streaming: StreamingClass;
}

/** Reports which manifest files exist in a runtime without Cache Storage (Node.js). */
export interface FsProbe {
  cached(
    urls: readonly string[],
  ): Promise<readonly { readonly cached: boolean; readonly size?: number }[]>;
  remove(urls: readonly string[]): Promise<void>;
}

/** Everything the engine needs beyond the manifests. */
export interface TransformersEngineOptions {
  readonly registry: EngineRegistry;
  /** Resolved device: `"wasm"`, `"webgpu"`, or `"cpu"` under `onnxruntime-node`. */
  readonly device: string;
  readonly dtype: ResolvedDtype;
  /** Worker factory; `undefined` runs inference in-thread. */
  readonly createWorker?: (() => Worker) | undefined;
  /** In-thread loader, required only when {@link createWorker} is absent. */
  readonly inThreadLoader?:
    | ((
        options: {
          readonly modelId: string;
          readonly revision: string;
          readonly device: string;
          readonly dtype: string;
          readonly onProgress: (event: ProgressResponse["event"]) => void;
        },
        voice: string,
      ) => Promise<LoadedPhonon>)
    | undefined;
  /** Filesystem cache probe, used where Cache Storage does not exist. */
  readonly fsProbe?: FsProbe | undefined;
}

/**
 * Engine backed by the Phonon worker.
 *
 * `supports()` and `voices()` are synchronous and side-effect free, `load()` is single-flight
 * per model + voice, and every operation re-checks disposal and abort after each `await`.
 */
export class TransformersEngine implements TtsEngine {
  readonly id = ENGINE_ID;

  readonly #registry: EngineRegistry;
  readonly #device: string;
  readonly #dtype: ResolvedDtype;
  readonly #createWorker: (() => Worker) | undefined;
  readonly #inThreadLoader: TransformersEngineOptions["inThreadLoader"];
  readonly #fsProbe: FsProbe | undefined;

  #worker: Worker | undefined;
  /** In-thread host, used only where `Worker` is unavailable. */
  #host: ModelHost | undefined;
  #disposed = false;
  #requestId = 0;
  readonly #pending = new Map<number, PendingRequest>();
  #loaded: LoadedState | undefined;
  #loadPromise: Promise<LoadedState> | undefined;
  #loadKey: string | undefined;
  /** The id of the stream currently being consumed, so an abandoned one can be closed. */
  #activeStreamId: number | undefined;
  #onDebug: DebugCallback | undefined;

  constructor(options: TransformersEngineOptions) {
    this.#registry = options.registry;
    this.#device = options.device;
    this.#dtype = options.dtype;
    this.#createWorker = options.createWorker;
    this.#inThreadLoader = options.inThreadLoader;
    this.#fsProbe = options.fsProbe;
  }

  // ------------------------------------------------------------------ identity

  /** Whether the registry knows this language. */
  supports(request: TtsRequest): boolean {
    return this.#registry.model(request.language) !== undefined;
  }

  /**
   * Voices of the configured model.
   *
   * A voice the model does not ship is absent here, so `createTts()` rejects it with
   * `VOICE_NOT_SUPPORTED` up front instead of failing at synthesis time.
   */
  voices(): readonly VoiceInfo[] {
    const model = this.#registry.model("en");
    if (!model) {
      return [];
    }
    return this.#registry.voices(model).map((voice) => {
      const accent = voiceAccent(voice.id);
      const license = voice.license ?? "unknown";
      return {
        id: voice.id,
        language: voice.language,
        ...(voice.sampleRate !== undefined && { sampleRate: voice.sampleRate }),
        ...(PHONON_VOICE_NAMES[voice.id] !== undefined && {
          name: PHONON_VOICE_NAMES[voice.id],
        }),
        ...(PHONON_VOICE_GENDERS[voice.id] !== undefined && {
          gender: PHONON_VOICE_GENDERS[voice.id],
        }),
        ...(accent !== undefined
          ? { description: `Accent ${accent}; license ${license}` }
          : { description: `License: ${license}` }),
      };
    });
  }

  /** Resolved capabilities, or `undefined` before the first successful load. */
  capabilities(): TtsCapabilities | undefined {
    const loaded = this.#loaded;
    if (!loaded) {
      return undefined;
    }
    return {
      engine: this.id,
      device: loaded.device,
      dtype: loaded.dtype,
      modelId: loaded.model.id,
      modelVersion: loaded.model.revision,
      sampleRate: loaded.sampleRate,
      streaming: loaded.streaming,
    };
  }

  // --------------------------------------------------------------------- cache

  /** Whether every asset for `request` is locally available. */
  async isCached(request: TtsRequest): Promise<boolean> {
    this.#assertNotDisposed();
    return (await this.describeAssets(request)).complete;
  }

  /**
   * Per-file cache state, sizes, and licenses.
   *
   * Sizes come from the manifest, so a plan is complete before anything is downloaded — this
   * is what lets the core check a budget *before* touching the network.
   */
  async describeAssets(request: TtsRequest): Promise<AssetPlan> {
    this.#assertNotDisposed();
    const model = this.#requireModel(request);
    const voice = this.#resolveVoice(model, request);
    const files = this.#manifestFiles(model, voice);
    const urls = files.map((file) => this.#url(model, file.path));
    const states = await this.#cacheStates(urls);
    const planFiles: AssetPlanFile[] = files.map((file, index) => ({
      path: file.path,
      role: file.path.startsWith("voices/") ? "voice" : "model",
      cached: states[index]?.cached ?? false,
      ...(file.size !== undefined && { size: file.size }),
    }));
    return createAssetPlan({
      modelId: model.id,
      ...(voice !== undefined && { voice: voice.id }),
      revision: model.revision,
      files: planFiles,
      licenses: [model.license, voice?.license].filter(
        (license): license is string => license !== undefined,
      ),
    });
  }

  /**
   * Explicit, opt-in provisioning.
   *
   * The core has already checked the budget, so this only has to make the assets exist.
   * Loading is the only download path Transformers.js offers, so `prepare()` loads and leaves
   * the model resident: the caller asked for a download, and making them wait for the same
   * bytes twice would be worse than keeping the session.
   */
  async prepare(request: TtsRequest, onProgress?: ProgressCallback): Promise<void> {
    this.#assertNotDisposed();
    await this.load(request, onProgress);
  }

  /** Disposes the loaded model and evicts its cached files from every cache. */
  async removeModel(request: TtsRequest): Promise<void> {
    this.#assertNotDisposed();
    const model = this.#requireModel(request);
    const voice = this.#resolveVoice(model, request);
    const urls = this.#manifestFiles(model, voice).map((file) => this.#url(model, file.path));
    if (this.#loaded?.model.id === model.id) {
      await this.#disposeLoaded();
    }
    await this.#deleteCached(urls);
  }

  // ---------------------------------------------------------------------- load

  /**
   * Loads the model and the requested voice.
   *
   * Single-flight per model + voice: concurrent calls await one download. A rejected load is
   * not memoized, so a transient failure does not poison the instance. A load for a *different*
   * model or voice waits for the in-flight one, then disposes it and loads — the worker hosts
   * exactly one model, so two loads must never interleave.
   */
  load(request: TtsRequest, onProgress?: ProgressCallback, onDebug?: DebugCallback): Promise<void> {
    this.#assertNotDisposed();
    if (onDebug !== undefined) {
      this.#onDebug = onDebug;
    }
    const model = this.#requireModel(request);
    const voice = this.#resolveVoice(model, request);
    const key = this.#loadKeyFor(model, voice);
    if (this.#loaded && this.#loadKey === key) {
      return Promise.resolve();
    }
    const inFlight = this.#loadPromise;
    if (inFlight && this.#loadKey !== key) {
      return inFlight.then(
        () => this.load(request, onProgress, onDebug),
        () => this.load(request, onProgress, onDebug),
      );
    }
    const load = this.#beginLoad(model, voice, onProgress);
    this.#loadPromise = load;
    this.#loadKey = key;
    void load.catch(() => {
      if (this.#loadPromise === load) {
        this.#loadPromise = undefined;
        this.#loadKey = undefined;
      }
    });
    return load.then(() => undefined);
  }

  async #beginLoad(
    model: ModelManifest,
    voice: VoiceManifest | undefined,
    onProgress: ProgressCallback | undefined,
  ): Promise<LoadedState> {
    const started = now();
    this.#debug({ type: "load-start", timestamp: started, engine: this.id });
    this.#debug({
      type: "device-resolved",
      timestamp: started,
      engine: this.id,
      device: this.#device,
      dtype: this.#dtype,
    });
    // The worker holds one model: a different model, device, or dtype needs a dispose first.
    const previous = this.#loaded;
    if (
      previous !== undefined &&
      this.#loadKeyFor(previous.model, previous.voice) !== this.#loadKeyFor(model, voice)
    ) {
      await this.#disposeLoaded();
    }

    let sampleRate = model.sampleRate ?? 24_000;
    let streaming: StreamingClass = "incremental-frames";
    try {
      const response = await this.#request(
        {
          kind: "load",
          id: this.#nextId(),
          modelId: model.id,
          revision: model.revision,
          device: this.#device,
          dtype: this.#dtype,
          ...(voice !== undefined && { voice: voice.id }),
          ...(model.family !== undefined && { profile: model.family }),
        },
        undefined,
        (event) => {
          if (event.kind === "progress") {
            onProgress?.(event.event);
          } else if (event.kind === "capabilities") {
            sampleRate = event.sampleRate;
            streaming = event.streaming;
          }
        },
      );
      if (response.kind === "error") {
        throw new Error(response.message);
      }
      if (response.kind !== "loaded") {
        throw new Error(`Unexpected worker response to load: ${response.kind}`);
      }
    } catch (error) {
      throw this.#mapLoadError(error, model, voice);
    }

    const loaded: LoadedState = {
      model,
      voice,
      device: this.#device,
      dtype: this.#dtype,
      sampleRate,
      streaming,
    };
    this.#loaded = loaded;
    this.#debug({
      type: "load-done",
      timestamp: now(),
      engine: this.id,
      durationMs: now() - started,
    });
    return loaded;
  }

  /**
   * Maps a load failure onto the core's codes.
   *
   * Three outcomes must not be conflated: the assets are not cached while offline
   * (`OFFLINE_ASSETS_MISSING`), the model is not one this profile can run
   * (`MODEL_NOT_AVAILABLE`), or loading failed for another reason (`MODEL_LOAD_FAILED`,
   * which also covers a failed download because the core has no separate code for it).
   *
   * The *message* decides, not the error type: a worker boundary turns every failure into a
   * fresh `Error`, so a recognizable reason is all that survives.
   */
  #mapLoadError(error: unknown, model: ModelManifest, voice: VoiceManifest | undefined): TtsError {
    if (isAbortError(error)) {
      return new TtsError(TTS_ERROR_CODES.ABORTED, "The model load was aborted", { cause: error });
    }
    if (isTtsError(error) && error.code !== TTS_ERROR_CODES.MODEL_LOAD_FAILED) {
      return error;
    }
    const message = error instanceof Error ? error.message : String(error);
    const cause = isTtsError(error) ? (error.cause ?? error) : error;
    if (/does not declare a "phonon"|Unsupported dtype|already hosts/i.test(message)) {
      return new TtsError(TTS_ERROR_CODES.MODEL_NOT_AVAILABLE, message, { cause });
    }
    if (isOfflineError(message)) {
      return new TtsError(
        TTS_ERROR_CODES.OFFLINE_ASSETS_MISSING,
        `Model ${model.id}${voice ? ` and voice ${voice.id}` : ""} are not cached and the environment is offline`,
        { cause, details: { modelId: model.id, ...(voice && { voice: voice.id }) } },
      );
    }
    return new TtsError(
      TTS_ERROR_CODES.MODEL_LOAD_FAILED,
      `Failed to load ${model.id}${voice ? ` with voice ${voice.id}` : ""}: ${message}`,
      { cause },
    );
  }

  // ----------------------------------------------------------------- synthesis

  /** Synthesizes the whole text and returns mono PCM with an explicit sample rate. */
  async synthesize(
    text: string,
    request: TtsRequest,
    options: SynthesizeOptions = {},
  ): Promise<AudioResult> {
    this.#assertNotDisposed();
    this.#assertNotAborted(options.signal);
    const loaded = await this.#ensureLoaded(request, options);
    const started = now();
    this.#debug({
      type: "synthesize-start",
      timestamp: started,
      engine: this.id,
      textLength: text.length,
      ...(loaded.voice && { voice: loaded.voice.id }),
    });

    const response = await this.#request(
      {
        kind: "synthesize",
        id: this.#nextId(),
        text,
        ...(loaded.voice && { voice: loaded.voice.id }),
      },
      options.signal,
    );
    if (response.kind === "cancelled") {
      throw new TtsError(TTS_ERROR_CODES.ABORTED, "Synthesis was cancelled");
    }
    if (response.kind !== "result") {
      throw this.#unexpected("SYNTHESIS_FAILED", response);
    }
    const pcm = new Float32Array(response.pcm);
    const audioDurationMs = response.sampleRate > 0 ? (pcm.length / response.sampleRate) * 1000 : 0;
    this.#debug({
      type: "synthesize-done",
      timestamp: now(),
      engine: this.id,
      durationMs: now() - started,
      textLength: text.length,
      audioDurationMs,
      samples: pcm.length,
      ...(loaded.voice && { voice: loaded.voice.id }),
    });
    return {
      pcm,
      sampleRate: response.sampleRate,
      channels: 1,
      durationMs: audioDurationMs,
      engine: this.id,
      ...(loaded.voice && { voice: loaded.voice.id }),
      language: request.language,
    };
  }

  /**
   * Streams audio one frame per consumer turn.
   *
   * Pulling on demand is what keeps `createAudioStream`'s bounded queue from ever having to drop
   * anything — the engine simply does not run ahead of the consumer.
   *
   * Abandoning the iterator (`break`, `return()`, an early `throw`) must stop the model. The
   * generator may be suspended inside an `await` for the next frame, where it cannot observe a
   * `return()`; the iterator returned here therefore aborts its own signal **and** posts
   * `stream-close` synchronously, then reports `done` instead of waiting for a frame nobody wants.
   */
  stream(
    text: string,
    request: TtsRequest,
    options: SynthesizeOptions = {},
  ): AsyncIterable<AudioChunk> {
    return {
      [Symbol.asyncIterator]: (): AsyncIterator<AudioChunk> => {
        const abandoned = new AbortController();
        const forwardAbort = (): void => abandoned.abort();
        options.signal?.addEventListener("abort", forwardAbort, { once: true });
        const generator = this.#streamFrom(text, request, { signal: abandoned.signal });
        let closed = false;
        const close = (): void => {
          if (closed) {
            return;
          }
          closed = true;
          abandoned.abort();
          options.signal?.removeEventListener("abort", forwardAbort);
          // Aborting the pull is not enough on its own: the worker must be told to stop, and the
          // generator may never resume to do it from its own `finally`.
          const active = this.#activeStreamId;
          if (active !== undefined) {
            this.#post({ kind: "stream-close", id: active });
          }
        };
        return {
          next: (): Promise<IteratorResult<AudioChunk>> => generator.next(),
          return: async (): Promise<IteratorResult<AudioChunk>> => {
            close();
            // Not awaited: the generator is suspended on a frame call that settles only once the
            // abort above takes effect, which is exactly what the consumer stopped waiting for.
            generator.return(undefined).catch(() => undefined);
            return { done: true, value: undefined };
          },
          throw: async (error?: unknown): Promise<IteratorResult<AudioChunk>> => {
            close();
            generator.return(undefined).catch(() => undefined);
            throw error;
          },
        };
      },
    };
  }

  async *#streamFrom(
    text: string,
    request: TtsRequest,
    options: SynthesizeOptions,
  ): AsyncGenerator<AudioChunk> {
    this.#assertNotDisposed();
    const loaded = await this.#ensureLoaded(request, options);
    this.#assertNotAborted(options.signal);

    const id = this.#nextId();
    // Registered before the first `await` so an abandoned iterator can post `stream-close` even
    // while this generator is suspended inside the engine call.
    this.#activeStreamId = id;
    const started = now();
    this.#debug({
      type: "stream-start",
      timestamp: started,
      engine: this.id,
      textLength: text.length,
      ...(loaded.voice && { voice: loaded.voice.id }),
    });

    const signal = options.signal;
    const onAbort = (): void => {
      this.#settle(id, new TtsError(TTS_ERROR_CODES.ABORTED, "The stream was aborted"));
      this.#post({ kind: "cancel", id });
    };
    signal?.addEventListener("abort", onAbort, { once: true });

    let sequence = 0;
    let chunks = 0;
    let audioDurationMs = 0;
    let opened = false;
    try {
      const openedResponse = await this.#request(
        {
          kind: "stream-open",
          id,
          text,
          ...(loaded.voice && { voice: loaded.voice.id }),
        },
        signal,
      );
      if (openedResponse.kind === "cancelled") {
        throw new TtsError(TTS_ERROR_CODES.ABORTED, "The stream was aborted");
      }
      if (openedResponse.kind !== "stream-opened") {
        throw this.#unexpected("STREAM_FAILED", openedResponse);
      }
      opened = true;

      for (;;) {
        const next = await this.#request({ kind: "stream-next", id }, signal);
        if (next.kind === "cancelled") {
          throw new TtsError(TTS_ERROR_CODES.ABORTED, "The stream was aborted");
        }
        if (next.kind === "stream-end") {
          // Only here is it known that the model has nothing left, and the contract needs
          // exactly one `final` chunk. A stream that produced no frame still gets one, so a
          // consumer never sees "ended without final".
          yield {
            pcm: new Float32Array(0),
            sampleRate: loaded.sampleRate,
            sequence,
            final: true,
          };
          this.#debug({
            type: "stream-done",
            timestamp: now(),
            engine: this.id,
            durationMs: now() - started,
            chunks,
            audioDurationMs,
          });
          return;
        }
        if (next.kind !== "chunk") {
          throw this.#unexpected("STREAM_FAILED", next);
        }
        const pcm = new Float32Array(next.pcm);
        if (chunks === 0) {
          this.#debug({
            type: "time-to-first-chunk",
            timestamp: now(),
            engine: this.id,
            durationMs: now() - started,
          });
        }
        chunks += 1;
        if (next.sampleRate > 0) {
          audioDurationMs += (pcm.length / next.sampleRate) * 1000;
        }
        yield { pcm, sampleRate: next.sampleRate, sequence, final: false };
        sequence += 1;
      }
    } finally {
      signal?.removeEventListener("abort", onAbort);
      if (this.#activeStreamId === id) {
        this.#activeStreamId = undefined;
      }
      if (opened) {
        // Closing stops the model. Fire-and-forget: the consumer has already left, and awaiting a
        // worker roundtrip would delay `return()` for no benefit. `close()` in the iterator may
        // already have posted it; a second close for a dead id is ignored by the worker.
        this.#post({ kind: "stream-close", id });
      }
    }
  }

  // --------------------------------------------------------------------- life

  /**
   * Releases the model and terminates the worker. Idempotent and never hangs.
   *
   * The worker is asked to release its model first so GPU buffers are freed cooperatively, but that
   * wait is **bounded**: a worker that is wedged, crashed, or already gone must not keep `dispose()`
   * pending forever. The acknowledgement buys an orderly release, not correctness of the call —
   * `terminate()` is what actually guarantees the resources are gone.
   */
  async dispose(): Promise<void> {
    if (this.#disposed) {
      return;
    }
    this.#disposed = true;
    try {
      if (this.#createWorker) {
        await this.#withDeadline(
          this.#request({ kind: "dispose", id: this.#nextId() }),
          DISPOSE_ACK_TIMEOUT_MS,
        );
      } else if (this.#host) {
        // In-thread there is nothing to time out: the host releases synchronously and always answers.
        await this.#host.dispatch({ kind: "dispose", id: this.#nextId() });
      }
    } catch {
      // A worker that failed while releasing is terminated below anyway.
    } finally {
      this.#worker?.terminate();
      this.#worker = undefined;
      this.#host = undefined;
      this.#loaded = undefined;
      this.#loadPromise = undefined;
      this.#loadKey = undefined;
      this.#rejectAll(new TtsError(TTS_ERROR_CODES.DISPOSED, "The engine was disposed"));
    }
  }

  /** Settles with `promise`, or resolves `undefined` after `timeoutMs`. Never rejects on timeout. */
  async #withDeadline<T>(promise: Promise<T>, timeoutMs: number): Promise<T | undefined> {
    // The losing branch of the race must not surface as an unhandled rejection.
    promise.catch(() => undefined);
    let cancel: (() => void) | undefined;
    try {
      return await Promise.race([
        promise,
        new Promise<undefined>((resolve) => {
          // `setTimeout` is typed as `number` in a DOM lib and as `Timeout` in `@types/node`; the
          // structural cast keeps this module free of either dependency.
          const timer = setTimeout(resolve, timeoutMs) as unknown as {
            unref?: () => void;
          };
          // In Node.js an unref'd timer lets the process exit while a worker shuts down.
          timer.unref?.();
          cancel = () => clearTimeout(timer as never);
        }),
      ]);
    } finally {
      cancel?.();
    }
  }

  // ----------------------------------------------------------------- internals

  /** Ensures the requested model + voice is loaded, honouring disposal and abort. */
  async #ensureLoaded(request: TtsRequest, options: SynthesizeOptions): Promise<LoadedState> {
    const current = this.#loaded;
    const model = this.#requireModel(request);
    const voice = this.#resolveVoice(model, request);
    if (
      current !== undefined &&
      this.#loadKeyFor(current.model, current.voice) === this.#loadKeyFor(model, voice)
    ) {
      return current;
    }
    await this.load(request, undefined, this.#onDebug);
    this.#assertNotDisposed();
    this.#assertNotAborted(options.signal);
    if (!this.#loaded) {
      throw new TtsError(TTS_ERROR_CODES.MODEL_LOAD_FAILED, "The model is not loaded after load()");
    }
    return this.#loaded;
  }

  /** Disposes the loaded model inside the worker without tearing the worker down. */
  async #disposeLoaded(): Promise<void> {
    if (!this.#loaded) {
      return;
    }
    this.#loaded = undefined;
    try {
      if (this.#createWorker) {
        await this.#request({ kind: "dispose", id: this.#nextId() });
      } else if (this.#host) {
        await this.#host.dispatch({ kind: "dispose", id: this.#nextId() });
      }
    } catch (error) {
      throw normalizeTtsError(error, TTS_ERROR_CODES.DISPOSE_FAILED, "Failed to release the model");
    }
  }

  /** The worker, created on first use. */
  #ensureWorker(): Worker {
    if (!this.#worker) {
      if (!this.#createWorker) {
        throw new TtsError(
          TTS_ERROR_CODES.NOT_SUPPORTED,
          "This engine has no worker factory; use the in-thread path",
        );
      }
      const worker = this.#createWorker();
      worker.onmessage = (event: MessageEvent<WorkerResponse>): void => {
        this.#handleMessage(event.data);
      };
      worker.onerror = (event: ErrorEvent): void => {
        const message = event.message || "Worker error";
        this.#debug({ type: "worker-error", timestamp: now(), engine: this.id, message });
        this.#rejectAll(new TtsError(TTS_ERROR_CODES.SYNTHESIS_FAILED, message));
      };
      this.#worker = worker;
      this.#debug({ type: "worker-spawn", timestamp: now(), engine: this.id });
    }
    return this.#worker;
  }

  /** The in-thread host, used where `Worker` does not exist. */
  #ensureHost(): ModelHost {
    this.#host ??= createModelHost({
      load: (options, voice) => {
        if (!this.#inThreadLoader) {
          throw new TtsError(
            TTS_ERROR_CODES.NOT_SUPPORTED,
            "In-thread inference was selected but no loader was provided",
          );
        }
        return this.#inThreadLoader(options, voice);
      },
      // In-thread there is no separate thread to answer `cache-check`; the filesystem probe
      // answers instead, so the worker-side cache inspection is unreachable here.
      post: (response) => {
        this.#handleMessage(response);
      },
    });
    return this.#host;
  }

  /**
   * Posts a request whose response nobody awaits (`cancel`, `stream-close`).
   *
   * Reaches the worker or the in-thread host, whichever this engine uses. The host is only created
   * when a worker factory is absent — never on the worker path, where it would duplicate the model.
   */
  #post(request: WorkerRequest): void {
    if (this.#createWorker) {
      this.#ensureWorker().postMessage(request);
      return;
    }
    void this.#ensureHost().dispatch(request as never);
  }

  /**
   * Posts a request and resolves with its settling response.
   *
   * `onEvent` sees intermediate responses (progress, capabilities, timings). Each
   * `stream-next` is its own request, so the pending entry is dropped as soon as one frame
   * arrives and re-registered for the next pull.
   */
  #request(
    request: WorkerRequest,
    signal?: AbortSignal,
    onEvent?: (response: WorkerResponse) => void,
  ): Promise<WorkerResponse> {
    if (signal?.aborted) {
      this.#post({ kind: "cancel", id: request.id });
      return Promise.reject(new TtsError(TTS_ERROR_CODES.ABORTED, "Operation was aborted"));
    }
    const promise = new Promise<WorkerResponse>((resolve, reject) => {
      this.#pending.set(request.id, {
        resolve,
        reject,
        ...(onEvent !== undefined && { onEvent }),
      });
      if (this.#createWorker) {
        this.#ensureWorker().postMessage(request);
        return;
      }
      void this.#ensureHost()
        .dispatch(request as never)
        .catch(reject);
    });
    if (signal === undefined) {
      return promise.finally(() => {
        this.#pending.delete(request.id);
      });
    }
    const onAbort = (): void => {
      this.#settle(request.id, new TtsError(TTS_ERROR_CODES.ABORTED, "Operation was aborted"));
    };
    signal.addEventListener("abort", onAbort, { once: true });
    return promise.finally(() => {
      signal.removeEventListener("abort", onAbort);
      this.#pending.delete(request.id);
    });
  }

  /** Rejects and forgets one pending request. */
  #settle(id: number, error: unknown): void {
    const pending = this.#pending.get(id);
    if (pending) {
      this.#pending.delete(id);
      pending.reject(error);
    }
  }

  /** Correlates one worker response with its pending request. */
  #handleMessage(response: WorkerResponse): void {
    const pending = this.#pending.get(response.id);
    if (!pending) {
      // A late frame of a cancelled or completed request: discard it.
      return;
    }
    switch (response.kind) {
      case "progress":
      case "capabilities":
      case "inference-start":
      case "inference-done":
        pending.onEvent?.(response);
        return;
      case "error":
        this.#pending.delete(response.id);
        pending.reject(this.#errorFor(response));
        return;
      default:
        this.#pending.delete(response.id);
        pending.resolve(response);
    }
  }

  /** Rejects every pending request, used when the worker dies. */
  #rejectAll(error: TtsError): void {
    const pending = [...this.#pending.values()];
    this.#pending.clear();
    for (const request of pending) {
      request.reject(error);
    }
  }

  /**
   * Turns a worker `error` response into the rejection the caller sees.
   *
   * When the worker names a stable code, the error carries it. Otherwise this rejects with a
   * plain `Error`: the operation that was waiting decides the code (a failed `load` becomes
   * `MODEL_LOAD_FAILED`, a failed stream becomes `STREAM_FAILED`), which is more accurate than
   * guessing here and keeps `cause` pointing at the actual failure.
   */
  #errorFor(response: ErrorResponse): unknown {
    const known = response.code !== undefined ? KNOWN_WORKER_CODES.get(response.code) : undefined;
    return known !== undefined
      ? new TtsError(known, response.message)
      : new Error(response.message);
  }

  /** Builds the error for "this response is not the one this code path expects". */
  #unexpected(code: "SYNTHESIS_FAILED" | "STREAM_FAILED", response: WorkerResponse): TtsError {
    const detail =
      response.kind === "error"
        ? response.message
        : `unexpected worker response "${response.kind}"`;
    return new TtsError(
      TTS_ERROR_CODES[code],
      detail,
      response.kind === "error" && response.code !== undefined
        ? { details: { workerCode: response.code } }
        : {},
    );
  }

  #nextId(): number {
    this.#requestId += 1;
    return this.#requestId;
  }

  #assertNotDisposed(): void {
    if (this.#disposed) {
      throw new TtsError(TTS_ERROR_CODES.DISPOSED, "The engine was disposed");
    }
  }

  #assertNotAborted(signal: AbortSignal | undefined): void {
    if (signal?.aborted) {
      throw new TtsError(TTS_ERROR_CODES.ABORTED, "Operation was aborted");
    }
  }

  #debug(event: DebugEvent): void {
    this.#onDebug?.(event);
  }

  /** The load identity of a model + voice on the resolved device and dtype. */
  #loadKeyFor(model: ModelManifest, voice: VoiceManifest | undefined): string {
    return `${model.id}@${model.revision}:${voice?.id ?? ""}:${this.#device}:${this.#dtype}`;
  }

  /** The model a request resolves to, or an explicit error. */
  #requireModel(request: TtsRequest): ModelManifest {
    const model = this.#registry.model(request.language);
    if (!model) {
      throw new TtsError(
        TTS_ERROR_CODES.LANGUAGE_NOT_SUPPORTED,
        `No model is registered for language ${request.language}`,
        { details: { language: request.language } },
      );
    }
    return model;
  }

  /** The voice a request resolves to; an unknown voice is rejected, never ignored. */
  #resolveVoice(model: ModelManifest, request: TtsRequest): VoiceManifest | undefined {
    const voice = this.#registry.voice(model, request.voice);
    if (voice === undefined && request.voice !== undefined) {
      throw new TtsError(
        TTS_ERROR_CODES.VOICE_NOT_SUPPORTED,
        `Model ${model.id} does not ship voice ${request.voice}`,
        { details: { modelId: model.id, voice: request.voice } },
      );
    }
    return voice;
  }

  /** The absolute URL of one manifest file at the model's pinned revision. */
  #url(model: ModelManifest, path: string): string {
    return assetUrl(model.id, model.revision, path);
  }

  /**
   * Every manifest file of a model + voice pair that the resolved dtype actually uses.
   *
   * A manifest lists every variant so the core can price any of them; the engine reports only
   * the files its resolved dtype loads, otherwise `isCached()` would demand weights that are
   * never read and `removeModel()` would delete weights belonging to another configuration.
   */
  #manifestFiles(
    model: ModelManifest,
    voice: VoiceManifest | undefined,
  ): readonly { readonly path: string; readonly size?: number }[] {
    const suffix = DTYPE_SUFFIX[this.#dtype];
    return [...model.files, ...(voice?.files ?? [])].filter((file) => {
      if (!file.path.startsWith("onnx/")) {
        return true;
      }
      return onnxVariant(file.path.slice("onnx/".length)) === suffix;
    });
  }

  /** Per-file cache state, from Cache Storage when available and the filesystem otherwise. */
  async #cacheStates(
    urls: readonly string[],
  ): Promise<readonly { readonly cached: boolean; readonly size?: number }[]> {
    // Cache Storage is reachable from the main thread, so describing assets never needs a
    // loaded model — which is the whole point of a consent dialog that runs before a download.
    if (typeof caches !== "undefined") {
      try {
        return await readCacheStorage(urls);
      } catch {
        // Fall through: a cache failure must not become a wrong "not cached" claim when the
        // filesystem can still answer.
      }
    }
    if (this.#fsProbe) {
      try {
        return await this.#fsProbe.cached(urls);
      } catch {
        // Fall through to the honest "cannot prove" answer below.
      }
    }
    return urls.map(() => ({ cached: false }));
  }

  /** Deletes cached URLs from Cache Storage and the filesystem. */
  async #deleteCached(urls: readonly string[]): Promise<void> {
    if (typeof caches !== "undefined") {
      const names = await caches.keys();
      await Promise.all(
        names.map(async (name) => {
          const cache = await caches.open(name);
          await Promise.all(urls.map((url) => cache.delete(url)));
        }),
      );
    }
    await this.#fsProbe?.remove(urls);
  }
}

/** ONNX variant suffixes a manifest may use, longest first so `_q4f16` wins over `_q4`. */
const ONNX_VARIANTS = ["_q4f16", "_q4", "_fp16", ""] as const;

/**
 * Worker error codes the engine translates itself.
 *
 * A worker `error` response is otherwise rejected as a plain `Error`, so the operation that was
 * waiting decides the code (a failed `load` → `MODEL_LOAD_FAILED`, a failed stream →
 * `STREAM_FAILED`). Only codes that name *why* the model cannot be used are mapped here.
 */
const KNOWN_WORKER_CODES: ReadonlyMap<string, TtsErrorCode> = new Map([
  [TTS_ERROR_CODES.MODEL_NOT_AVAILABLE, TTS_ERROR_CODES.MODEL_NOT_AVAILABLE],
  [TTS_ERROR_CODES.MODEL_LOAD_FAILED, TTS_ERROR_CODES.MODEL_LOAD_FAILED],
  [TTS_ERROR_CODES.OFFLINE_ASSETS_MISSING, TTS_ERROR_CODES.OFFLINE_ASSETS_MISSING],
  [TTS_ERROR_CODES.ABORTED, TTS_ERROR_CODES.ABORTED],
  [TTS_ERROR_CODES.ASSET_INTEGRITY_FAILED, TTS_ERROR_CODES.ASSET_INTEGRITY_FAILED],
]);

/**
 * The dtype variant an ONNX manifest path belongs to, e.g. `"_q4"` for `flow_lm_q4.onnx_data`.
 *
 * Longest suffix first, so `_q4f16` is never mistaken for `_q4`, and an external-weight `_data`
 * chunk is attributed to the graph it accompanies rather than treated as unsuffixed.
 */
function onnxVariant(stem: string): string {
  const base = stem.endsWith(".onnx_data")
    ? stem.slice(0, -".onnx_data".length)
    : stem.endsWith(".onnx")
      ? stem.slice(0, -".onnx".length)
      : stem;
  return ONNX_VARIANTS.find((candidate) => candidate !== "" && base.endsWith(candidate)) ?? "";
}

/**
 * Reads per-URL state out of every Cache Storage cache.
 *
 * Every cache, not just the library's default name: Transformers.js names its cache by default, but
 * an application may rename it or add its own, and `removeModel()` already sweeps all of them. There
 * are typically very few caches, and the lookups run in parallel.
 */
async function readCacheStorage(
  urls: readonly string[],
): Promise<readonly { readonly cached: boolean; readonly size?: number }[]> {
  const names = await caches.keys();
  const opened = await Promise.all(names.map((name) => caches.open(name)));
  return Promise.all(
    urls.map(async (url) => {
      for (const cache of opened) {
        const match = await cache.match(url);
        if (match !== undefined) {
          const length = match.headers.get("content-length");
          return {
            cached: true,
            ...(length !== null && { size: Number(length) }),
          };
        }
      }
      return { cached: false };
    }),
  );
}

/** Whether a message looks like "there is no usable network right now". */
function isOfflineError(message: string): boolean {
  if (typeof navigator !== "undefined" && navigator.onLine === false) {
    return true;
  }
  return /failed to fetch|networkerror|network request failed|load failed|enotfound|econnrefused|fetch failed/i.test(
    message,
  );
}
