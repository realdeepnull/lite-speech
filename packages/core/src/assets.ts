import { TTS_ERROR_CODES, TtsError } from "./errors.js";
import type { LanguageCode } from "./types.js";

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
  /**
   * Name of the build variant this file belongs to, e.g. `"q4"` or `"fp32"`.
   *
   * A file **without** a variant is required by every configuration (configs,
   * tokenizers, voice embeddings). A manifest may therefore list every variant
   * of its weights, and an engine selects the files that match the dtype it
   * resolved — which is what lets `estimateDownload()` price any of them
   * without knowing a single file-name convention.
   */
  readonly variant?: string;
}

/**
 * A pinned model revision and the files it consists of.
 *
 * A manifest is data, not code: it says what to download and which model family
 * runs it, never how to run it. That is why several families can share one
 * registry and one engine.
 */
export interface ModelManifest {
  /** Model identifier, e.g. `"onnx-community/phonon-31m-en-ONNX"`. */
  readonly id: string;
  /** Immutable revision (commit SHA). Never `main`. */
  readonly revision: string;
  /** License of the weights, e.g. `"Apache-2.0"`. */
  readonly license?: string;
  /** Model family, e.g. `"phonon"`. Selects the profile and the runner. */
  readonly family?: string;
  /**
   * Languages this model is registered for, e.g. `["en"]` or `["en", "de"]`.
   *
   * An engine may derive the list from the model's voices when it is omitted,
   * so a hand-written manifest with a voice list still resolves. Declaring it is
   * how a multilingual model is registered under more than one language without
   * pretending its voices switch language.
   */
  readonly languages?: readonly LanguageCode[];
  /** Native output sample rate in Hz. */
  readonly sampleRate?: number;
  readonly files: readonly AssetFile[];
}

/** A voice: its own files (speaker embedding, config) plus metadata. */
export interface VoiceManifest {
  /** Voice identifier passed to `createTts({ voice })`. */
  readonly id: string;
  /**
   * Repository this voice belongs to.
   *
   * Usually the model repository, but a family whose voices live elsewhere (a
   * shared voice pack, for instance) can point at its own.
   */
  readonly modelId: string;
  readonly language: LanguageCode;
  /** Immutable revision of the voice repository (commit SHA). */
  readonly revision?: string;
  /** License of the voice assets. */
  readonly license?: string;
  /** Native sample rate in Hz, when different from the model. */
  readonly sampleRate?: number;
  /** Human-readable name for UI. Presentation data belongs here, not in engine code. */
  readonly name?: string;
  /** Optional gender hint for UI (`"female"`, `"male"`, …). */
  readonly gender?: string;
  /** Free-form note, e.g. the accent of a recording or a quality caveat. */
  readonly description?: string;
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
  /**
   * Size of the on-disk/cache entry, when the engine could measure it.
   *
   * Reported separately from the declared {@link AssetFile.size}: a mismatch
   * means a truncated or foreign file is cached, which is a cache problem the
   * caller can see rather than a silent one.
   */
  readonly cachedSize?: number;
}

/**
 * How much of a plan can be proven byte-for-byte.
 *
 * - `"declared"` — every file declares a SHA-256 digest.
 * - `"partial"` — some files do.
 * - `"none"` — no file does, so integrity cannot be verified, only presence.
 */
export type AssetIntegrityState = "declared" | "partial" | "none";

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
  /** Whether every file declares a SHA-256 digest (integrity can be proven). */
  readonly hashesComplete: boolean;
  /** How much of the plan is covered by declared digests. */
  readonly integrity: AssetIntegrityState;
  /** Licenses involved, model first, then voice. */
  readonly licenses: readonly string[];
  /** Paths of the files that are not cached. */
  readonly missing: readonly string[];
  /** Paths of the files whose size is unknown. */
  readonly unknownSizes: readonly string[];
  /** Paths of the files that declare no digest. */
  readonly unverified: readonly string[];
  /**
   * Paths of cached files whose measured size disagrees with the declared size.
   *
   * A truncated download is the failure this catches: the file exists, so the
   * cache says "cached", but the bytes are not the pinned asset. Reported, not
   * silently repaired — repairing means downloading, which needs consent.
   */
  readonly sizeMismatches: readonly string[];
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
  let hashesComplete = true;
  const missing: string[] = [];
  const unknownSizes: string[] = [];
  const unverified: string[] = [];
  const sizeMismatches: string[] = [];

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
      if (file.cached && file.cachedSize !== undefined && file.cachedSize !== file.size) {
        sizeMismatches.push(file.path);
      }
    } else {
      sizesComplete = false;
      unknownSizes.push(file.path);
    }
    if (file.sha256 === undefined) {
      hashesComplete = false;
      unverified.push(file.path);
    } else if (!/^[0-9a-f]{64}$/u.test(file.sha256)) {
      throw new TtsError(
        TTS_ERROR_CODES.INVALID_ARGUMENT,
        `Invalid sha256 for asset ${file.path}: expected 64 lowercase hex characters`,
        { details: { path: file.path, sha256: file.sha256 } },
      );
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

  const integrity: AssetIntegrityState = hashesComplete
    ? "declared"
    : unverified.length === input.files.length
      ? "none"
      : "partial";

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
    hashesComplete,
    integrity,
    licenses,
    missing,
    unknownSizes,
    unverified,
    sizeMismatches,
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

/** The outcome of verifying declared digests against cached files. */
export interface IntegrityReport {
  /** Paths whose measured digest matched the declared one. */
  readonly verified: readonly string[];
  /** Paths whose measured digest did not match the declared one. */
  readonly mismatched: readonly string[];
  /** Paths that declare no digest, so nothing could be verified. */
  readonly unverified: readonly string[];
  /** Paths the engine could not read at all. */
  readonly unreadable: readonly string[];
  /** `true` when every verified digest matched and nothing was unreadable. */
  readonly ok: boolean;
}

/**
 * Normalizes what an engine reports about integrity checks into one answer.
 *
 * A file that could not be read is never counted as verified: "I could not look"
 * and "I looked and it was fine" are different outcomes, and conflating them is
 * exactly how a corrupted cache turns into a wrong `ok: true`.
 */
export function createIntegrityReport(
  parts: Partial<Omit<IntegrityReport, "ok">> = {},
): IntegrityReport {
  const verified = [...(parts.verified ?? [])];
  const mismatched = [...(parts.mismatched ?? [])];
  const unverified = [...(parts.unverified ?? [])];
  const unreadable = [...(parts.unreadable ?? [])];
  return {
    verified,
    mismatched,
    unverified,
    unreadable,
    ok: mismatched.length === 0 && unreadable.length === 0,
  };
}

/**
 * Rejects an integrity report that found a corrupted or unreadable asset.
 *
 * `ASSET_INTEGRITY_FAILED` is the only honest answer here: the caller asked for
 * a verified cache, and the alternative — quietly deleting and re-downloading —
 * would download without consent.
 */
export function assertIntegrity(report: IntegrityReport): IntegrityReport {
  if (report.ok) {
    return report;
  }
  const failed = [...report.mismatched, ...report.unreadable];
  throw new TtsError(
    TTS_ERROR_CODES.ASSET_INTEGRITY_FAILED,
    `${failed.length} cached asset(s) failed verification: ${failed.join(", ")}`,
    { details: { mismatched: report.mismatched, unreadable: report.unreadable } },
  );
}
