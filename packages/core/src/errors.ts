/**
 * Stable error codes of `@lite-speech/core`.
 *
 * The string values are part of the public contract: applications branch on
 * them, so they are never renamed without a major version bump. New codes may
 * be added in a minor release.
 */
export const TTS_ERROR_CODES = {
  /** No engine was injected and none is registered globally. */
  ENGINE_NOT_SUPPORTED: "ENGINE_NOT_SUPPORTED",
  /** The engine does not support the requested language. */
  LANGUAGE_NOT_SUPPORTED: "LANGUAGE_NOT_SUPPORTED",
  /** The engine does not provide the requested voice. */
  VOICE_NOT_SUPPORTED: "VOICE_NOT_SUPPORTED",
  /** `offline: "strict"` is set but the required assets are not cached. */
  OFFLINE_ASSETS_MISSING: "OFFLINE_ASSETS_MISSING",
  /** The download budget is smaller than the required asset size. */
  DOWNLOAD_BUDGET_EXCEEDED: "DOWNLOAD_BUDGET_EXCEEDED",
  /** A downloaded (or cached) asset did not match its declared hash. */
  ASSET_INTEGRITY_FAILED: "ASSET_INTEGRITY_FAILED",
  /** The requested model is unknown to the engine or the registry. */
  MODEL_NOT_AVAILABLE: "MODEL_NOT_AVAILABLE",
  /** Loading the model or runtime failed. */
  MODEL_LOAD_FAILED: "MODEL_LOAD_FAILED",
  /** Synthesis failed (engine error or contract violation). */
  SYNTHESIS_FAILED: "SYNTHESIS_FAILED",
  /** The audio stream failed or violated the chunk contract. */
  STREAM_FAILED: "STREAM_FAILED",
  /** The operation was cancelled through its `AbortSignal`. */
  ABORTED: "ABORTED",
  /** The `Tts` instance was disposed. */
  DISPOSED: "DISPOSED",
  /** Releasing engine resources failed. */
  DISPOSE_FAILED: "DISPOSE_FAILED",
  /** An argument or option is invalid. */
  INVALID_ARGUMENT: "INVALID_ARGUMENT",
  /** The engine does not implement an optional capability. */
  NOT_SUPPORTED: "NOT_SUPPORTED",
} as const;

/** Union of all stable error codes. */
export type TtsErrorCode = (typeof TTS_ERROR_CODES)[keyof typeof TTS_ERROR_CODES];

/** Options for {@link TtsError}. */
export interface TtsErrorOptions {
  /** The underlying error, kept for diagnostics. */
  cause?: unknown;
  /** Structured, machine-readable context (sizes, paths, phases, …). */
  details?: Readonly<Record<string, unknown>>;
}

/**
 * The single error type raised by this library.
 *
 * Every error that crosses a public boundary is normalized into a `TtsError`
 * exactly once, so consumers only ever have to check {@link isTtsError}. The
 * original error is preserved in `cause`.
 */
export class TtsError extends Error {
  /** Stable, machine-readable error code. */
  readonly code: TtsErrorCode;
  /** Structured context, e.g. required bytes and budget for a rejection. */
  readonly details: Readonly<Record<string, unknown>> | undefined;

  constructor(code: TtsErrorCode, message: string, options: TtsErrorOptions = {}) {
    super(message, options.cause === undefined ? undefined : { cause: options.cause });
    this.name = "TtsError";
    this.code = code;
    this.details = options.details;
  }
}

/** Type guard for {@link TtsError}. */
export function isTtsError(error: unknown): error is TtsError {
  return error instanceof TtsError;
}

/**
 * Returns `true` for DOM/Node abort errors — both the `AbortError` raised by
 * `AbortSignal` plumbing and the numeric/string `code` variants reported by
 * Node.js and some runtimes.
 */
export function isAbortError(error: unknown): boolean {
  if (error === null || typeof error !== "object") {
    return false;
  }
  const { name, code } = error as { name?: unknown; code?: unknown };
  return name === "AbortError" || code === "ABORT_ERR" || code === 20;
}

/**
 * Normalizes any thrown value into a {@link TtsError}.
 *
 * - A `TtsError` (and its `details`) passes through unchanged, so a nested
 *   `cause` chain is never flattened twice.
 * - An abort error becomes `ABORTED`, regardless of the `code` the caller
 *   would have used.
 * - Everything else is wrapped with the given `code` and kept in `cause`.
 */
export function normalizeTtsError(error: unknown, code: TtsErrorCode, message: string): TtsError {
  if (isTtsError(error)) {
    return error;
  }
  if (isAbortError(error)) {
    return new TtsError(TTS_ERROR_CODES.ABORTED, "Operation aborted", { cause: error });
  }
  return new TtsError(code, message, { cause: error });
}

/**
 * Formats any error into a single human-readable line, e.g. for logs.
 *
 * `TtsError` instances render as `<CODE>: <message>`; anything else falls back
 * to its `message` (or `String(error)`).
 */
export function formatTtsError(error: unknown): string {
  if (isTtsError(error)) {
    return `${error.code}: ${error.message}`;
  }
  const message = (error as { message?: unknown } | null | undefined)?.message;
  return typeof message === "string" ? message : String(error);
}
