/**
 * Summary statistics for the benchmark harness.
 *
 * The M1 spike published medians and warned that per-run spread reached ±20 % on one laptop, so
 * a single sample cannot be a claim. Every latency in a report is therefore stored as the raw
 * per-run array *and* summarized here as p50/p95, so a reader can see both the typical case and
 * the tail without trusting a hand-picked number.
 *
 * Deliberately dependency-free: the report must be reproducible from a clean checkout, and a
 * statistics package for four functions would be a dependency the library itself avoids.
 */

/** Summary of one measured quantity across runs. All values are in the input unit. */
export interface Summary {
  /** Number of observations. */
  readonly n: number;
  readonly min: number;
  readonly p50: number;
  readonly p95: number;
  readonly max: number;
  readonly mean: number;
}

/**
 * Nearest-rank percentile of an already-sorted array.
 *
 * Nearest-rank (rather than linear interpolation) is used because every quantity here is a
 * duration that actually occurred: reporting a p95 that no run produced would be a fabricated
 * number. With the small sample counts of a report (`n` of 3–10) it is also the only definition
 * whose p95 is distinguishable from the maximum.
 */
export function sortedPercentile(sorted: readonly number[], fraction: number): number {
  if (sorted.length === 0) {
    return Number.NaN;
  }
  const clamped = Math.min(Math.max(fraction, 0), 1);
  const rank = Math.ceil(clamped * sorted.length);
  const index = Math.min(Math.max(rank - 1, 0), sorted.length - 1);
  return sorted[index] ?? Number.NaN;
}

/**
 * Summarizes one quantity.
 *
 * Non-finite values are dropped instead of being coerced: a `NaN` means "this run did not
 * produce the measurement", and averaging it into a latency would silently understate the tail.
 */
export function summarize(values: readonly number[]): Summary {
  const finite = values.filter((value) => Number.isFinite(value));
  if (finite.length === 0) {
    return {
      n: 0,
      min: Number.NaN,
      p50: Number.NaN,
      p95: Number.NaN,
      max: Number.NaN,
      mean: Number.NaN,
    };
  }
  const sorted = [...finite].sort((a, b) => a - b);
  const total = sorted.reduce((sum, value) => sum + value, 0);
  return {
    n: sorted.length,
    min: sorted[0] ?? Number.NaN,
    p50: sortedPercentile(sorted, 0.5),
    p95: sortedPercentile(sorted, 0.95),
    max: sorted.at(-1) ?? Number.NaN,
    mean: total / sorted.length,
  };
}

/** Rounds to a fixed number of decimals, keeping reports readable and diffs small. */
export function round(value: number, decimals = 2): number {
  if (!Number.isFinite(value)) {
    return value;
  }
  const factor = 10 ** decimals;
  return Math.round(value * factor) / factor;
}

/** Summarizes and rounds, which is what every report field actually stores. */
export function summarizeRounded(values: readonly number[], decimals = 2): Summary {
  const summary = summarize(values);
  return {
    n: summary.n,
    min: round(summary.min, decimals),
    p50: round(summary.p50, decimals),
    p95: round(summary.p95, decimals),
    max: round(summary.max, decimals),
    mean: round(summary.mean, decimals),
  };
}

/** Median of a list, used where only one number is wanted (e.g. audio seconds). */
export function median(values: readonly number[]): number {
  return sortedPercentile(
    values.filter((value) => Number.isFinite(value)).sort((a, b) => a - b),
    0.5,
  );
}

/** Decimal megabytes, matching how the docs and `estimateDownload()` phrase sizes. */
export function megabytes(bytes: number, decimals = 1): number {
  return round(bytes / 1e6, decimals);
}
