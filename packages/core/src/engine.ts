import { TTS_ERROR_CODES, TtsError } from "./errors.js";
import type { AssetPlan } from "./assets.js";
import type {
  AudioChunk,
  AudioResult,
  DebugCallback,
  ProgressCallback,
  SynthesizeOptions,
  TtsCapabilities,
  TtsRequest,
  VoiceInfo,
} from "./types.js";

/**
 * Engine-independent interface for a text-to-speech backend.
 *
 * The core knows nothing about ONNX, Transformers.js, WASM, WebGPU, workers, or
 * HTTP: an engine only exchanges plain PCM, plain data, and stable error codes.
 * Everything an engine needs to report about its backend is a string in
 * {@link TtsCapabilities}.
 *
 * Required members must be implemented by every engine. Optional members are
 * capabilities the core degrades around explicitly — it never guesses.
 */
export interface TtsEngine {
  /** Stable engine identifier, e.g. `"onnx"`. Echoed in `AudioResult.engine`. */
  readonly id: string;

  /**
   * Synchronous capability check. Must be cheap and must not load or download
   * anything: `createTts()` calls it before returning.
   */
  supports(request: TtsRequest): boolean;

  /**
   * Synchronous voice list for the configured model.
   *
   * Returns an empty array when the engine cannot describe its voices; the
   * core then accepts the configured voice without validating it, because
   * rejecting a voice the engine cannot name would be worse than accepting it.
   */
  voices(): readonly VoiceInfo[];

  /** Whether every asset required by `request` is available locally. */
  isCached(request: TtsRequest): Promise<boolean>;

  /**
   * Loads the model and runtime. Idempotent: repeated calls must not trigger a
   * second download. Honours `onProgress`/`onDebug` when the engine emits them.
   */
  load(request: TtsRequest, onProgress?: ProgressCallback, onDebug?: DebugCallback): Promise<void>;

  /**
   * Synthesizes the whole text at once. Lazily loads the model; honours
   * `options.signal`; returns mono PCM with an explicit sample rate.
   */
  synthesize(text: string, request: TtsRequest, options?: SynthesizeOptions): Promise<AudioResult>;

  /**
   * Produces audio incrementally. Lazily loads the model; honours
   * `options.signal`. Chunks must be ordered, gap-free, and end with exactly
   * one `final: true` chunk — the core enforces this contract.
   */
  stream(text: string, request: TtsRequest, options?: SynthesizeOptions): AsyncIterable<AudioChunk>;

  /** Releases inference sessions, GPU resources, and worker state. */
  dispose(): Promise<void>;

  /** Resolved capabilities of the loaded model, when the engine has any. */
  capabilities?(): TtsCapabilities | undefined;

  /**
   * Explicit, opt-in provisioning: downloads and verifies the assets for
   * `request`. The core calls this only after the budget check passed.
   */
  prepare?(request: TtsRequest, onProgress?: ProgressCallback): Promise<void>;

  /**
   * Per-file cache state and sizes. Required for a budget check and for
   * strict-offline preconditions; without it the core cannot prove either.
   */
  describeAssets?(request: TtsRequest): Promise<AssetPlan>;

  /** Evicts the cached assets for `request` from every cache the engine uses. */
  removeModel?(request: TtsRequest): Promise<void>;
}

const defaultEngines: TtsEngine[] = [];

/**
 * Registers an engine globally so `createTts({ voice })` works without passing
 * `engine` explicitly. A convenience for meta packages; registering the same
 * `id` twice is a no-op.
 */
export function registerDefaultEngine(engine: TtsEngine): void {
  assertTtsEngine(engine);
  if (!defaultEngines.some((candidate) => candidate.id === engine.id)) {
    defaultEngines.push(engine);
  }
}

/**
 * Returns a copy of the registered default engines, so callers cannot mutate
 * the registry while later `createTts()` calls still see newly registered
 * engines.
 */
export function getDefaultEngines(): TtsEngine[] {
  return [...defaultEngines];
}

/** Removes every registered engine. Intended for tests and teardown. */
export function clearDefaultEngines(): void {
  defaultEngines.length = 0;
}

const REQUIRED_MEMBERS = [
  "id",
  "supports",
  "voices",
  "isCached",
  "load",
  "synthesize",
  "stream",
  "dispose",
] as const;

/**
 * Structural check for a value passed as an engine.
 *
 * `TtsOptions.engine` is typed, but options often cross a JavaScript boundary
 * (plain objects, framework wrappers, dynamic imports). Failing here — before
 * any state is created — produces one actionable error instead of a
 * `TypeError` deep inside a synthesis call.
 */
export function assertTtsEngine(value: unknown): asserts value is TtsEngine {
  if (value === null || typeof value !== "object") {
    throw new TtsError(TTS_ERROR_CODES.ENGINE_NOT_SUPPORTED, "No TTS engine was provided");
  }
  const candidate = value as Partial<Record<(typeof REQUIRED_MEMBERS)[number], unknown>>;
  const missing = REQUIRED_MEMBERS.filter((member) => {
    if (member === "id") {
      return typeof candidate.id !== "string" || candidate.id.length === 0;
    }
    return typeof candidate[member] !== "function";
  });
  if (missing.length > 0) {
    throw new TtsError(
      TTS_ERROR_CODES.ENGINE_NOT_SUPPORTED,
      `Invalid TTS engine: missing or invalid ${missing.join(", ")}`,
      { details: { missing } },
    );
  }
}

/** Type guard variant of {@link assertTtsEngine}. */
export function isTtsEngine(value: unknown): value is TtsEngine {
  try {
    assertTtsEngine(value);
    return true;
  } catch {
    return false;
  }
}
