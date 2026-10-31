/**
 * The model host: everything the worker does *except* talk to `postMessage`.
 *
 * Separating them keeps the worker entry to a dozen lines of protocol plumbing and makes
 * the interesting invariants testable in Node — one model per host, single-flight loads,
 * voice switching, one stream session at a time, and cancellation that is observable
 * rather than merely "we stopped listening". A test supplies a fake {@link PhononLoader},
 * so none of this needs ONNX, a browser, or a download.
 */
import type { ProgressEvent } from "@lite-speech/core";
import type { LoadedPhonon } from "./phonon.js";
import type {
  CacheCheckRequest,
  CancelRequest,
  DisposeRequest,
  LoadRequest,
  StreamCloseRequest,
  StreamNextRequest,
  StreamOpenRequest,
  SynthesizeRequest,
  WorkerRequest,
  WorkerResponse,
} from "./worker-protocol.js";

/** Loads a model; the real one is `loadPhonon`, tests pass a fake. */
export type PhononLoader = (
  options: {
    readonly modelId: string;
    readonly revision: string;
    readonly device: string;
    readonly dtype: string;
    readonly voiceSha256?: string | undefined;
    readonly onProgress: (event: ProgressEvent) => void;
  },
  voice: string,
) => Promise<LoadedPhonon>;

/** Opens a cache so per-file state can be inspected; `undefined` disables the check. */
export type CacheOpener = () => Promise<{
  match(url: string): Promise<{ headers: { get(name: string): string | null } } | undefined>;
}>;

/** Dependencies of a {@link ModelHost}, all injectable for tests. */
export interface ModelHostOptions {
  readonly load: PhononLoader;
  /** Writes one response; the worker adds transfer lists, tests collect them. */
  readonly post: (response: WorkerResponse) => void;
  /** Opens the browser cache used for `cache-check`; absent means "cannot prove". */
  readonly openCache?: CacheOpener | undefined;
  /** Clock for `inference-done`; defaults to `performance.now()`. */
  readonly now?: () => number;
  /** Base URL for cache lookups; defaults to the public Hub. */
  readonly hubBase?: string;
}

/** The host surface the worker drives. */
export interface ModelHost {
  /** Runs one request, reporting through `post`. Never throws for a request-level error. */
  dispatch(request: Exclude<WorkerRequest, CancelRequest>): Promise<void>;
  /** Applies a cancellation immediately, without waiting for the queue. */
  cancel(request: CancelRequest): void;
}
/** The one open stream session. */
interface StreamSession {
  readonly id: number;
  readonly iterator: AsyncGenerator<Float32Array>;
  readonly sampleRate: number;
}

/**
 * Creates a host for exactly one model.
 *
 * The single-model rule is not an implementation shortcut: it bounds memory, makes
 * "which weights are resident" a question with one answer, and keeps the protocol free of
 * session switching. The engine disposes before changing model, device, or dtype, so a
 * conflicting load is misuse — and an explicit error is better than two models competing
 * for the same GPU.
 */
export function createModelHost(options: ModelHostOptions): ModelHost {
  const post = options.post;
  const now = options.now ?? ((): number => performance.now());
  const hubBase = options.hubBase ?? "https://huggingface.co";

  let model: LoadedPhonon | undefined;
  let loadPromise: Promise<LoadedPhonon> | undefined;
  let loadedModel: { modelId: string; revision: string; device: string; dtype: string } | undefined;
  let loadedVoice: string | undefined;
  let session: StreamSession | undefined;
  /** Requests the caller cancelled; checked between frames and before each pull. */
  const cancelled = new Set<number>();

  /** Forgets the loaded model. Used before a reload and on dispose. */
  function forget(): void {
    model = undefined;
    loadPromise = undefined;
    loadedModel = undefined;
    loadedVoice = undefined;
  }

  async function closeSession(): Promise<void> {
    const active = session;
    session = undefined;
    await active?.iterator.return(undefined);
  }

  function requireModel(): LoadedPhonon {
    if (!model) {
      throw new Error("The model is not loaded; send a load request first");
    }
    return model;
  }

  /** Reports the resolved properties of the loaded model. */
  function postCapabilities(id: number): void {
    if (!model || !loadedModel) {
      return;
    }
    post({
      kind: "capabilities",
      id,
      device: loadedModel.device,
      dtype: loadedModel.dtype,
      modelId: loadedModel.modelId,
      modelVersion: loadedModel.revision,
      sampleRate: model.config.sampleRate,
      streaming: model.profile.streaming,
    });
  }

  /**
   * Loads, or confirms an already loaded model.
   *
   * A failed load is never memoized: the worker would otherwise answer every later attempt
   * with the same stale error instead of retrying a download that may now work.
   */
  async function load(request: LoadRequest): Promise<LoadedPhonon> {
    if (model && loadedModel) {
      if (loadedModel.modelId !== request.modelId || loadedModel.revision !== request.revision) {
        throw new Error(
          `This worker already hosts ${loadedModel.modelId}@${loadedModel.revision}; ` +
            "dispose before loading a different model.",
        );
      }
      if (loadedModel.device !== request.device || loadedModel.dtype !== request.dtype) {
        throw new Error(
          `This worker already hosts ${loadedModel.modelId} on ${loadedModel.device}/${loadedModel.dtype}; ` +
            "dispose before changing device or dtype.",
        );
      }
      return model;
    }
    loadPromise ??= (async (): Promise<LoadedPhonon> => {
      const loaded = await options.load(
        {
          modelId: request.modelId,
          revision: request.revision,
          device: request.device,
          dtype: request.dtype,
          ...(request.voiceSha256 !== undefined && { voiceSha256: request.voiceSha256 }),
          onProgress: (event) => {
            post({ kind: "progress", id: request.id, event });
          },
        },
        request.voice ?? "default",
      );
      model = loaded;
      loadedModel = {
        modelId: request.modelId,
        revision: request.revision,
        device: request.device,
        dtype: request.dtype,
      };
      loadedVoice = request.voice ?? "default";
      return loaded;
    })();
    loadPromise = loadPromise.catch((error: unknown) => {
      loadPromise = undefined;
      throw error;
    });
    return loadPromise;
  }

  /**
   * Switches the loaded voice.
   *
   * A voice is a model input, not a session option, so switching means rebuilding the
   * runner. Doing that eagerly — during `load`, not inside the first synthesis — keeps the
   * cost where the caller can see and report it.
   */
  async function ensureVoice(request: LoadRequest): Promise<LoadedPhonon> {
    const wanted = request.voice ?? "default";
    if (loadedVoice === wanted) {
      return requireModel();
    }
    await closeSession();
    const previous = model;
    forget();
    await previous?.dispose();
    return load(request);
  }

  async function handleLoad(request: LoadRequest): Promise<void> {
    if (model && request.voice !== undefined && loadedVoice !== request.voice) {
      await ensureVoice(request);
    } else {
      await load(request);
    }
    postCapabilities(request.id);
    post({ kind: "loaded", id: request.id });
  }

  async function handleSynthesize(request: SynthesizeRequest): Promise<void> {
    const active = requireModel();
    const started = now();
    post({ kind: "inference-start", id: request.id, textLength: request.text.length });

    const frames: Float32Array[] = [];
    let samples = 0;
    for await (const frame of active.synthesize(request.text, {
      isCancelled: () => cancelled.has(request.id),
      temperature: request.temperature,
    })) {
      frames.push(frame);
      samples += frame.length;
    }
    if (cancelled.delete(request.id)) {
      post({ kind: "cancelled", id: request.id });
      return;
    }

    const sampleRate = active.config.sampleRate;
    const pcm = new Float32Array(samples);
    let offset = 0;
    for (const frame of frames) {
      pcm.set(frame, offset);
      offset += frame.length;
    }
    post({
      kind: "inference-done",
      id: request.id,
      durationMs: now() - started,
      samples,
      sampleRate,
    });
    post({
      kind: "result",
      id: request.id,
      pcm: pcm.buffer as ArrayBuffer,
      sampleRate,
      frames: frames.length,
      durationMs: now() - started,
    });
  }

  async function handleStreamOpen(request: StreamOpenRequest): Promise<void> {
    const active = requireModel();
    await closeSession();
    if (cancelled.delete(request.id)) {
      post({ kind: "cancelled", id: request.id });
      return;
    }
    post({ kind: "inference-start", id: request.id, textLength: request.text.length });
    session = {
      id: request.id,
      sampleRate: active.config.sampleRate,
      iterator: active.synthesize(request.text, {
        isCancelled: () => cancelled.has(request.id),
        temperature: request.temperature,
      }),
    };
    post({ kind: "stream-opened", id: request.id });
  }

  /**
   * Pulls exactly one frame.
   *
   * One frame per request is what makes backpressure structural: the worker cannot get
   * ahead of the consumer because it is not asked to, so memory stays at one frame plus
   * model state with no credit bookkeeping. It is also the cancellation checkpoint — a
   * cancelled stream ends here rather than after the next 30 seconds of audio.
   */
  async function handleStreamNext(request: StreamNextRequest): Promise<void> {
    const active = session;
    if (!active || active.id !== request.id) {
      if (cancelled.delete(request.id)) {
        post({ kind: "cancelled", id: request.id });
        return;
      }
      throw new Error("No stream session is open for this request");
    }
    if (cancelled.has(request.id)) {
      cancelled.delete(request.id);
      await closeSession();
      post({ kind: "cancelled", id: request.id });
      return;
    }
    const next = await active.iterator.next();
    if (next.done) {
      session = undefined;
      if (cancelled.delete(request.id)) {
        post({ kind: "cancelled", id: request.id });
        return;
      }
      post({ kind: "stream-end", id: request.id });
      return;
    }
    post({
      kind: "chunk",
      id: request.id,
      pcm: next.value.buffer as ArrayBuffer,
      sampleRate: active.sampleRate,
    });
  }

  /**
   * Reports per-file cache state.
   *
   * Without a cache (Node.js uses the filesystem, which a worker cannot reach) this
   * reports every file as absent *and* every size as unknown, so the engine's own cache
   * probe decides. Claiming "cached" would be a lie; claiming "missing" would be a wrong
   * offline diagnosis.
   */
  async function handleCacheCheck(request: CacheCheckRequest): Promise<void> {
    const opener = options.openCache;
    if (!opener) {
      post({
        kind: "cache-state",
        id: request.id,
        cached: request.files.map(() => false),
        sizes: request.files.map(() => undefined),
      });
      return;
    }
    const cache = await opener();
    const urls = request.files.map(
      (file) => `${hubBase}/${request.modelId}/resolve/${request.revision}/${file}`,
    );
    const matches = await Promise.all(urls.map((url) => cache.match(url)));
    post({
      kind: "cache-state",
      id: request.id,
      cached: matches.map((match) => match !== undefined),
      sizes: matches.map((match) => {
        const length = match?.headers.get("content-length");
        return length === null || length === undefined ? undefined : Number(length);
      }),
    });
  }

  /**
   * Releases everything and always acknowledges.
   *
   * The acknowledgement is in `finally` so the engine can never wait forever, and state is
   * reset even when releasing the session throws — otherwise the next `load` would see a
   * half-torn-down model.
   */
  async function handleDispose(request: DisposeRequest): Promise<void> {
    try {
      await closeSession();
      await model?.dispose();
    } finally {
      forget();
      cancelled.clear();
      post({ kind: "disposed", id: request.id });
    }
  }

  async function handleQueued(
    request: Exclude<WorkerRequest, CancelRequest | StreamNextRequest | StreamCloseRequest>,
  ): Promise<void> {
    switch (request.kind) {
      case "load":
        return handleLoad(request);
      case "synthesize":
        return handleSynthesize(request);
      case "stream-open":
        return handleStreamOpen(request);
      case "cache-check":
        return handleCacheCheck(request);
      case "dispose":
        return handleDispose(request);
    }
  }

  return {
    async dispatch(request: Exclude<WorkerRequest, CancelRequest>): Promise<void> {
      try {
        switch (request.kind) {
          case "stream-close":
            await closeSession();
            return;
          case "stream-next":
            await handleStreamNext(request);
            return;
          default:
            await handleQueued(request);
        }
      } catch (error) {
        post({
          kind: "error",
          id: request.id,
          message: error instanceof Error ? error.message : String(error),
        });
      }
    },
    cancel(request: CancelRequest): void {
      cancelled.add(request.id);
      const active = session;
      if (active && active.id === request.id) {
        // Dropping the iterator stops the pull loop at the next frame instead of waiting
        // for the caller's in-flight `stream-next` to complete a whole extra frame.
        void active.iterator.return(undefined);
        session = undefined;
      }
      post({ kind: "cancelled", id: request.id });
    },
  };
}
