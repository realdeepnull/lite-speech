/**
 * Pure data contracts of `@lite-speech/core`.
 *
 * Nothing in this module has behaviour or imports an engine implementation, so
 * it stays tree-shakeable and free of runtime dependencies. The only type
 * import points at the engine *interface*, which is a type-only edge.
 */
import type { TtsEngine } from "./engine.js";

/** BCP-47-like language code, e.g. `"en"` or `"de-DE"`. */
export type LanguageCode = string;

/**
 * Network policy of a `Tts` instance.
 *
 * - `"strict"` (default) forbids network access for synthesis: the engine's
 *   assets must already be cached, otherwise the call rejects with
 *   `OFFLINE_ASSETS_MISSING` before the engine is asked to synthesize.
 * - `"allow"` permits the engine to download on demand.
 */
export type OfflinePolicy = "strict" | "allow";

/**
 * How a model profile produces audio, reported by the engine through
 * {@link TtsCapabilities.streaming}.
 *
 * This is a truthful classification, not a promise: `stream()` returns an
 * `AsyncIterable` for every profile, but only `"incremental-frames"` yields
 * short frames while the model is still generating.
 */
export type StreamingClass = "incremental-frames" | "sentence-chunks" | "utterance";

/** What the application asks for: one language, optionally one voice. */
export interface TtsRequest {
  /** Language of the text to synthesize. */
  readonly language: LanguageCode;
  /** Voice identifier as reported by `TtsEngine.voices()`. */
  readonly voice?: string;
}

/** Description of a voice provided by an engine. */
export interface VoiceInfo {
  /** Stable identifier passed to `createTts({ voice })`. */
  readonly id: string;
  /** Language this voice speaks. */
  readonly language: LanguageCode;
  /** Human-readable name for UI. */
  readonly name?: string;
  /** Optional gender hint for UI ("female", "male", …). */
  readonly gender?: string;
  /** Native sample rate of this voice, when known. */
  readonly sampleRate?: number;
  /** Free-form note, e.g. licensing or quality caveats. */
  readonly description?: string;
}

/**
 * Result of a full-utterance synthesis.
 *
 * PCM is mono, in `[-1, 1]`, and the sample rate is always explicit — the
 * library never resamples implicitly and never touches an `AudioContext`.
 */
export interface AudioResult {
  /** Mono PCM samples in `[-1, 1]`. */
  readonly pcm: Float32Array;
  /** Sample rate in Hz of `pcm`. `0` means "no audio, rate unknown". */
  readonly sampleRate: number;
  /** Channel count. Always `1`: stereo is an encoder concern. */
  readonly channels: 1;
  /** Duration of `pcm` in milliseconds. */
  readonly durationMs: number;
  /** Engine that produced the audio, e.g. `"onnx"`. */
  readonly engine: string;
  /** Voice that produced the audio, when one was selected. */
  readonly voice?: string;
  /** Language of the synthesized text, when known. */
  readonly language?: LanguageCode;
}

/**
 * One piece of a streamed utterance.
 *
 * `sequence` is monotonic and gap-free per stream, starting at `0`. Exactly one
 * chunk per stream carries `final: true`; the iterator completes after it.
 */
export interface AudioChunk {
  /** Mono PCM samples in `[-1, 1]`. */
  readonly pcm: Float32Array;
  /** Sample rate in Hz; constant for the whole stream. */
  readonly sampleRate: number;
  /** Monotonic, gap-free sequence number starting at `0`. */
  readonly sequence: number;
  /** `true` on the last chunk of the stream. */
  readonly final: boolean;
}

/** Download and load progress of model or voice assets. */
export interface ProgressEvent {
  /** Phase, e.g. `"model-download"`, `"model-load"`, `"voice-download"`. */
  readonly phase: string;
  /** Bytes transferred (or units loaded) so far. */
  readonly loaded: number;
  /** Expected total, or `0` when unknown. */
  readonly total: number;
  /** `loaded / total` in `0..1`, or `NaN` when `total` is unknown. */
  readonly progress: number;
  /** File currently being fetched, when the engine can name it. */
  readonly file?: string;
}

/** Callback for download/load progress. */
export type ProgressCallback = (event: ProgressEvent) => void;

/**
 * Resolved capabilities of the loaded engine.
 *
 * All backend-specific values are plain strings so the core contract never
 * depends on ONNX, WebGPU, or Transformers.js types.
 */
export interface TtsCapabilities {
  /** Stable engine identifier, e.g. `"onnx"`. */
  readonly engine: string;
  /** Resolved compute device, e.g. `"wasm"` or `"webgpu"`. */
  readonly device?: string;
  /** Resolved data type, e.g. `"fp32"` or `"q4"`. */
  readonly dtype?: string;
  /** Model identifier, e.g. `"onnx-community/phonon-31m-en-ONNX"`. */
  readonly modelId?: string;
  /** Pinned model revision (commit SHA). */
  readonly modelVersion?: string;
  /** Native sample rate of the model output, when known. */
  readonly sampleRate?: number;
  /** Truthful streaming classification of this model profile. */
  readonly streaming?: StreamingClass;
}

/**
 * Structured debug event emitted through the optional `onDebug` callback.
 *
 * Every event carries a `timestamp` from `performance.now()`. Events are a
 * discriminated union on `type`; the callback is opt-in and costs nothing when
 * absent (all call sites use `?.()`).
 */
export type DebugEvent =
  | { readonly type: "load-start"; readonly timestamp: number; readonly engine: string }
  | {
      readonly type: "load-done";
      readonly timestamp: number;
      readonly engine: string;
      readonly durationMs: number;
    }
  | {
      readonly type: "device-resolved";
      readonly timestamp: number;
      readonly engine: string;
      readonly device: string;
      readonly dtype: string;
    }
  | {
      readonly type: "device-fallback";
      readonly timestamp: number;
      readonly engine: string;
      readonly from: string;
      readonly to: string;
    }
  | { readonly type: "worker-spawn"; readonly timestamp: number; readonly engine: string }
  | {
      readonly type: "worker-error";
      readonly timestamp: number;
      readonly engine: string;
      readonly message: string;
    }
  | {
      readonly type: "synthesize-start";
      readonly timestamp: number;
      readonly engine: string;
      readonly textLength: number;
      readonly voice?: string;
    }
  | {
      readonly type: "synthesize-done";
      readonly timestamp: number;
      readonly engine: string;
      readonly durationMs: number;
      readonly textLength: number;
      readonly audioDurationMs: number;
      readonly samples: number;
      readonly voice?: string;
    }
  | {
      readonly type: "stream-start";
      readonly timestamp: number;
      readonly engine: string;
      readonly textLength: number;
      readonly voice?: string;
    }
  | {
      readonly type: "time-to-first-chunk";
      readonly timestamp: number;
      readonly engine: string;
      readonly durationMs: number;
    }
  | {
      readonly type: "stream-done";
      readonly timestamp: number;
      readonly engine: string;
      readonly durationMs: number;
      readonly chunks: number;
      readonly audioDurationMs: number;
    }
  | {
      readonly type: "inference-start";
      readonly timestamp: number;
      readonly engine: string;
      readonly requestId: number;
      readonly textLength: number;
    }
  | {
      readonly type: "inference-done";
      readonly timestamp: number;
      readonly engine: string;
      readonly requestId: number;
      readonly durationMs: number;
      readonly samples: number;
      readonly sampleRate: number;
    }
  | {
      readonly type: "abort";
      readonly timestamp: number;
      readonly engine: string;
      readonly phase: "load" | "synthesize" | "stream";
    };

/** Callback for structured debug events. */
export type DebugCallback = (event: DebugEvent) => void;

/** Options for `createTts()`. */
export interface TtsOptions {
  /**
   * Engine to use. Takes precedence over {@link TtsOptions.engines} and the
   * global registry. Validated at runtime, so a value from JavaScript (or a
   * framework wrapper) fails with `ENGINE_NOT_SUPPORTED` instead of a
   * `TypeError` later.
   */
  readonly engine?: TtsEngine;
  /** Candidate engines; the first supported one is used. */
  readonly engines?: readonly TtsEngine[];
  /**
   * Language to synthesize. Defaults to the language of the selected voice, or
   * to the language of the engine's first voice.
   */
  readonly language?: LanguageCode;
  /** Voice to synthesize with. Defaults to the engine's first voice. */
  readonly voice?: string;
  /** Network policy. Defaults to `"strict"`. */
  readonly offline?: OfflinePolicy;
  /** Progress callback for asset download and model load. */
  readonly onProgress?: ProgressCallback;
  /** Opt-in structured debug callback. */
  readonly onDebug?: DebugCallback;
}

/** Options for a single `synthesize()` or `stream()` call. */
export interface SynthesizeOptions {
  /**
   * Cancels the operation. An abort during generation rejects with
   * `TtsError(ABORTED)`; audio chunks already delivered stay valid.
   */
  readonly signal?: AbortSignal;
}
