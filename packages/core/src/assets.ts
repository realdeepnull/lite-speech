import { TTS_ERROR_CODES, TtsError } from "./errors.js";

/**
 * Declarative description of the files a model needs.
 *
 * Sizes and hashes are optional because an engine may not know them (for
 * example a hand-written manifest); everything that consumes a manifest treats
 * "unknown" as "cannot prove", never as "zero bytes".
 */
export interface AssetFile {
  /** Path inside the model repository, e.g. `"onnx/model_q4.onnx"`. */
  readonly path: string;
  /** Size in bytes, when known. */
  readonly size?: number;
  /** Expected lowercase SHA-256 hex digest, when declared. */
  readonly sha256?: string;
  /** Resolved absolute URL, when the engine already resolved one. */
  readonly url?: string;
}

/** A pinned model revision and the files it consists of. */
export interface ModelManifest {
  /** Model identifier, e.g. `"onnx-community/phonon-31m-en-ONNX"`. */
  readonly id: string;
  /** Immutable revision (commit SHA). Never `main`. */
  readonly revision: string;
  /** License of the weights, e.g. `"Apache-2.0"`. */
  readonly license?: string;
  /** Model family, e.g. `"phonon"`. */
  readonly family?: string;
  /** Native output sample rate in Hz. */
  readonly sampleRate?: number;
  readonly files: readonly AssetFile[];
}

/** A voice: its own files (speaker embedding, config) plus metadata. */
export interface VoiceManifest {
  /** Voice identifier passed to `createTts({ voice })`. */
  readonly id: string;
  /** Model this voice belongs to. */
  readonly modelId: string;
  readonly language: string;
  /** Immutable revision of the voice repository (commit SHA). */
  readonly revision?: string;
  /** License of the voice assets. */
  readonly license?: string;
  /** Native sample rate in Hz, when different from the model. */
  readonly sampleRate?: number;
  readonly files: readonly AssetFile[];
}

/** Origin of one file in an {@link AssetPlan}. */
export type AssetRole = "model" | "voice";

/** A manifest file plus the cache state the engine could determine. */
export interface AssetPlanFile extends AssetFile {
  /** Whether the file is a model or a voice asset. */
  readonly role: AssetRole;
  /** Whether the file is available locally. */
  readonly cached: boolean;
}

/**
 * The complete, resolved asset set for one model + voice combination.
 *
 * Produced by `TtsEngine.describeAssets()` and consumed by
 * {@link estimateDownload}, {@link assertBudget}, and the offline policy in
 * `Tts`.
 */
export interface AssetPlan {
  readonly modelId: string;
  readonly voice?: string;
  readonly revision: string;
  readonly files: readonly AssetPlanFile[];
  /** Sum of every declared file size (unknown sizes count as `0`). */
  readonly totalBytes: number;
  /** Bytes already present locally. */
  readonly cachedBytes: number;
  /** Bytes that still have to be downloaded. */
  readonly missingBytes: number;
  /** `true` when every file is cached, i.e. strict-offline synthesis works. */
  readonly complete: boolean;
  /** Whether every file declares a size (a budget can be proven). */
  readonly sizesComplete: boolean;
  /** Licenses involved, model first, then voice. */
  readonly licenses: readonly string[];
  /** Paths of the files that are not cached. */
  readonly missing: readonly string[];
  /** Paths of the files whose size is unknown. */
  readonly unknownSizes: readonly string[];
}

/** Input of {@link createAssetPlan}. */
export interface AssetPlanInput {
  readonly modelId: string;
  readonly voice?: string;
  readonly revision: string;
  readonly files: readonly AssetPlanFile[];
  readonly licenses?: readonly string[];
}

/** Result of {@link estimateDownload}. */
export interface DownloadEstimate {
  /** Bytes that still have to be downloaded. */
  readonly bytes: number;
  /** Bytes of every asset in the plan, cached or not. */
  readonly totalBytes: number;
  /** `bytes` in decimal megabytes (1 MB = 1 000 000 bytes). */
  readonly megabytes: number;
  /** The files that still have to be downloaded. */
  readonly files: readonly AssetFile[];
}

/** Cache state of a single asset path. */
export type CacheState = "cached" | "missing" | "unknown";

/**
 * Builds an {@link AssetPlan} and derives every aggregate from `files`.
 *
 * Throws `INVALID_ARGUMENT` for duplicate paths or negative sizes: a plan is
 * the data a consent dialog shows, so it must not contain nonsense.
 */
export function createAssetPlan(input: AssetPlanInput): AssetPlan {
  const seen = new Set<string>();
  let totalBytes = 0;
  let cachedBytes = 0;
  let sizesComplete = true;
  const missing: string[] = [];
  const unknownSizes: string[] = [];

  for (const file of input.files) {
    if (seen.has(file.path)) {
      throw new TtsError(TTS_ERROR_CODES.INVALID_ARGUMENT, `Duplicate asset path: ${file.path}`, {
        details: { path: file.path, modelId: input.modelId },
      });
    }
    seen.add(file.path);
    if (file.size !== undefined) {
      if (!Number.isFinite(file.size) || file.size < 0) {
        throw new TtsError(
          TTS_ERROR_CODES.INVALID_ARGUMENT,
          `Invalid size for asset ${file.path}: ${String(file.size)}`,
          { details: { path: file.path, size: file.size } },
        );
      }
      totalBytes += file.size;
      if (file.cached) {
        cachedBytes += file.size;
      }
    } else {
      sizesComplete = false;
      unknownSizes.push(file.path);
    }
    if (!file.cached) {
      missing.push(file.path);
    }
  }

  const licenses: string[] = [];
  for (const license of input.licenses ?? []) {
    if (!licenses.includes(license)) {
      licenses.push(license);
    }
  }

  return {
    modelId: input.modelId,
    ...(input.voice !== undefined && { voice: input.voice }),
    revision: input.revision,
    files: [...input.files],
    totalBytes,
    cachedBytes,
    missingBytes: totalBytes - cachedBytes,
    complete: missing.length === 0,
    sizesComplete,
    licenses,
    missing,
    unknownSizes,
  };
}

/** Returns the files of `plan` that still have to be downloaded. */
export function missingAssets(plan: AssetPlan): readonly AssetPlanFile[] {
  return plan.files.filter((file) => !file.cached);
}

/** Cache state of one path inside a plan. */
export function cacheState(plan: AssetPlan, path: string): CacheState {
  const file = plan.files.find((candidate) => candidate.path === path);
  if (!file) {
    return "unknown";
  }
  return file.cached ? "cached" : "missing";
}

/**
 * The number the UI must show before a user consents to a download.
 *
 * `bytes` counts only the files that are not cached yet, `totalBytes` counts
 * the whole profile. Unknown sizes are reported through `files[*].size ===
 * undefined` and contribute `0`, so callers must not render the estimate as an
 * upper bound unless `plan.sizesComplete` is `true`.
 */
export function estimateDownload(plan: AssetPlan): DownloadEstimate {
  const files = missingAssets(plan);
  return {
    bytes: plan.missingBytes,
    totalBytes: plan.totalBytes,
    megabytes: plan.missingBytes / 1_000_000,
    files,
  };
}

/**
 * Rejects a download that would exceed `maxDownloadMB` **before** anything is
 * fetched.
 *
 * The budget is expressed in decimal megabytes (1 MB = 1 000 000 bytes); the
 * conversion happens here so callers never mix up MiB and MB. `undefined`
 * means "no budget" and returns the estimate unchanged.
 *
 * A plan with unknown sizes is rejected as well: the budget cannot be proven,
 * and the library never promises a limit it cannot keep.
 */
export function assertBudget(plan: AssetPlan, maxDownloadMB: number | undefined): DownloadEstimate {
  const estimate = estimateDownload(plan);
  if (maxDownloadMB === undefined) {
    return estimate;
  }
  if (typeof maxDownloadMB !== "number" || !Number.isFinite(maxDownloadMB) || maxDownloadMB < 0) {
    throw new TtsError(
      TTS_ERROR_CODES.INVALID_ARGUMENT,
      `maxDownloadMB must be a non-negative finite number, got ${String(maxDownloadMB)}`,
      { details: { maxDownloadMB } },
    );
  }
  if (plan.complete) {
    // Nothing to download, so no budget can be exceeded — even when some
    // cached file has no declared size.
    return estimate;
  }
  const budgetBytes = maxDownloadMB * 1_000_000;
  if (!plan.sizesComplete) {
    throw new TtsError(
      TTS_ERROR_CODES.DOWNLOAD_BUDGET_EXCEEDED,
      `Cannot verify the ${maxDownloadMB} MB download budget: ${plan.unknownSizes.length} asset(s) have no declared size`,
      {
        details: {
          budgetBytes,
          unknownSizes: plan.unknownSizes,
          modelId: plan.modelId,
        },
      },
    );
  }
  if (estimate.bytes > budgetBytes) {
    throw new TtsError(
      TTS_ERROR_CODES.DOWNLOAD_BUDGET_EXCEEDED,
      `Model ${plan.modelId} needs ${estimate.bytes} bytes, which exceeds the ${maxDownloadMB} MB budget`,
      {
        details: {
          requiredBytes: estimate.bytes,
          budgetBytes,
          files: estimate.files.map((file) => ({ path: file.path, size: file.size })),
          modelId: plan.modelId,
        },
      },
    );
  }
  return estimate;
}
