import type {
  AssetPlan,
  AudioChunk,
  AudioResult,
  DebugCallback,
  IntegrityReport,
  ProgressCallback,
  SynthesizeOptions,
  TtsCapabilities,
  TtsEngine,
  TtsRequest,
} from "../../src/index";
import { createAssetPlan, createIntegrityReport } from "../../src/index";

/** A scripted engine used by every core test; no runtime, no DOM, no network. */
export interface FakeEngineOptions {
  readonly id?: string;
  readonly voices?: readonly { id: string; language: string }[];
  readonly languages?: readonly string[];
  readonly sampleRate?: number;
  readonly chunks?: number;
  readonly samplesPerChunk?: number;
  readonly cached?: boolean;
  readonly assetFiles?: readonly {
    path: string;
    size?: number;
    sha256?: string;
    cached: boolean;
    cachedSize?: number;
  }[];
  readonly withDescribeAssets?: boolean;
  readonly withPrepare?: boolean;
  readonly withRemoveModel?: boolean;
  readonly withCapabilities?: boolean;
  /** Adds `verifyAssets()`; `integrity` decides what it reports. */
  readonly withVerifyAssets?: boolean;
  /** Partial integrity report `verifyAssets()` returns; defaults to "everything fine". */
  readonly integrity?: Partial<Omit<IntegrityReport, "ok">>;
  readonly loadError?: unknown;
  readonly synthesizeError?: unknown;
  readonly streamError?: unknown;
  readonly disposeError?: unknown;
  /** Delay in ms per simulated chunk; `0` keeps tests fast. */
  readonly chunkDelayMs?: number;
  /** Delay in ms applied inside `load()`. */
  readonly loadDelayMs?: number;
  /** Called after the engine's `stream()` produced its iterator. */
  readonly onStream?: (request: TtsRequest) => void;
  /** Violates the contract on purpose: duplicates sequence numbers. */
  readonly breakSequence?: boolean;
  /** Violates the contract on purpose: ends without a final chunk. */
  readonly omitFinal?: boolean;
  /** Violates the contract on purpose: changes sampleRate mid-stream. */
  readonly varySampleRate?: boolean;
}

/** Mutable call counters, assertions read them instead of inspecting state. */
interface FakeEngineStats {
  load: number;
  synthesize: number;
  stream: number;
  dispose: number;
  prepare: number;
  removeModel: number;
  describeAssets: number;
  verifyAssets: number;
  isCached: number;
  lastSynthesizedText: string | undefined;
  lastProgress: ProgressCallback | undefined;
  lastDebug: DebugCallback | undefined;
}

export interface FakeEngine extends TtsEngine {
  readonly stats: FakeEngineStats;
  readonly requests: TtsRequest[];
}

const sleep = (ms: number): Promise<void> =>
  ms <= 0 ? Promise.resolve() : new Promise((resolve) => setTimeout(resolve, ms));

/** Creates an engine that answers deterministically and records what it saw. */
export function createFakeEngine(options: FakeEngineOptions = {}): FakeEngine {
  const stats: FakeEngineStats = {
    load: 0,
    synthesize: 0,
    stream: 0,
    dispose: 0,
    prepare: 0,
    removeModel: 0,
    describeAssets: 0,
    verifyAssets: 0,
    isCached: 0,
    lastSynthesizedText: undefined,
    lastProgress: undefined,
    lastDebug: undefined,
  };
  const requests: TtsRequest[] = [];
  const sampleRate = options.sampleRate ?? 24_000;
  const voices = options.voices ?? [{ id: "voice-1", language: "en" }];
  const languages = options.languages ?? voices.map((voice) => voice.language);

  const engine: FakeEngine = {
    id: options.id ?? "fake",
    stats,
    requests,
    supports: (request) => languages.includes(request.language),
    voices: () => voices.map((voice) => ({ id: voice.id, language: voice.language })),
    async isCached() {
      stats.isCached += 1;
      return options.cached ?? true;
    },
    async load(request, onProgress, onDebug) {
      stats.load += 1;
      stats.lastProgress = onProgress;
      stats.lastDebug = onDebug;
      requests.push(request);
      if (options.loadDelayMs) {
        await sleep(options.loadDelayMs);
      }
      if (options.loadError) {
        throw options.loadError;
      }
      onProgress?.({ phase: "model-download", loaded: 1, total: 1, progress: 1 });
    },
    async synthesize(text, request, callOptions) {
      stats.synthesize += 1;
      stats.lastSynthesizedText = text;
      requests.push(request);
      void callOptions;
      if (options.synthesizeError) {
        throw options.synthesizeError;
      }
      const samples = (options.samplesPerChunk ?? 8) * (options.chunks ?? 1);
      return {
        pcm: new Float32Array(samples).fill(0.25),
        sampleRate,
        channels: 1,
        durationMs: (samples / sampleRate) * 1000,
        engine: engine.id,
      } satisfies AudioResult;
    },
    stream(text, request, callOptions) {
      stats.stream += 1;
      requests.push(request);
      options.onStream?.(request);
      return streamChunks(text, callOptions, options, sampleRate, engine.id);
    },
    async dispose() {
      stats.dispose += 1;
      if (options.disposeError) {
        throw options.disposeError;
      }
    },
  };

  if (options.withDescribeAssets ?? true) {
    engine.describeAssets = async (request) => {
      stats.describeAssets += 1;
      requests.push(request);
      return buildPlan(options, request);
    };
  }
  if (options.withPrepare) {
    engine.prepare = async () => {
      stats.prepare += 1;
    };
  }
  if (options.withRemoveModel) {
    engine.removeModel = async () => {
      stats.removeModel += 1;
    };
  }
  if (options.withVerifyAssets) {
    engine.verifyAssets = async () => {
      stats.verifyAssets += 1;
      return createIntegrityReport(options.integrity);
    };
  }
  if (options.withCapabilities) {
    engine.capabilities = (): TtsCapabilities => ({
      engine: engine.id,
      device: "wasm",
      dtype: "fp32",
      streaming: "incremental-frames",
    });
  }

  return engine;
}

function buildPlan(options: FakeEngineOptions, request: TtsRequest): AssetPlan {
  const files = options.assetFiles ?? [
    { path: "onnx/model_q4.onnx", size: 1000, cached: options.cached ?? true },
    { path: "vocab.json", size: 500, cached: options.cached ?? true },
  ];
  return createAssetPlan({
    modelId: "fake/model",
    revision: "0000000000000000000000000000000000000000",
    ...(request.voice !== undefined && { voice: request.voice }),
    licenses: ["Apache-2.0"],
    files: files.map((file) => ({
      path: file.path,
      role: "model" as const,
      cached: file.cached,
      ...(file.size !== undefined && { size: file.size }),
      ...(file.sha256 !== undefined && { sha256: file.sha256 }),
      ...(file.cachedSize !== undefined && { cachedSize: file.cachedSize }),
    })),
  });
}

async function* streamChunks(
  text: string,
  callOptions: SynthesizeOptions | undefined,
  options: FakeEngineOptions,
  sampleRate: number,
  engineId: string,
): AsyncGenerator<AudioChunk> {
  if (options.streamError) {
    throw options.streamError;
  }
  const total = Math.max(1, options.chunks ?? 3);
  const samples = options.samplesPerChunk ?? 4;
  void text;
  for (let index = 0; index < total; index += 1) {
    if (callOptions?.signal?.aborted) {
      throw new DOMException("Aborted", "AbortError");
    }
    if (options.chunkDelayMs) {
      await sleep(options.chunkDelayMs);
    }
    const isLast = index === total - 1;
    yield {
      pcm: new Float32Array(samples).fill(0.1 * (index + 1)),
      sampleRate: options.varySampleRate && index === 1 ? sampleRate * 2 : sampleRate,
      sequence: options.breakSequence && index === 1 ? 5 : index,
      final: isLast && !options.omitFinal,
    };
  }
  void engineId;
}

/**
 * An engine whose `stream()` never yields unless aborted, so a test can prove
 * that `dispose()` interrupts a blocked stream instead of hanging.
 */
export function createHangingEngine(): FakeEngine {
  const engine = createFakeEngine({ chunks: 1 });
  engine.stream = (_text, request, callOptions) => {
    engine.stats.stream += 1;
    engine.requests.push(request);
    return {
      [Symbol.asyncIterator]: () => ({
        next: () =>
          new Promise<IteratorResult<AudioChunk>>((_resolve, reject) => {
            callOptions?.signal?.addEventListener("abort", () => {
              reject(new DOMException("Aborted", "AbortError"));
            });
          }),
        return: () => Promise.resolve({ done: true, value: undefined }),
      }),
    };
  };
  return engine;
}
