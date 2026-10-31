/**
 * The statistics a published latency depends on.
 *
 * These are tested because a wrong percentile is invisible: a report with a plausible p50 and p95
 * cannot be told apart from one computed with the wrong definition, and every table in
 * `docs/performance.md` comes from here.
 */
import { describe, expect, it } from "vitest";

import {
  median,
  megabytes,
  round,
  sortedPercentile,
  summarize,
  summarizeRounded,
} from "../src/stats.js";

describe("sortedPercentile", () => {
  it("uses nearest-rank, so every reported percentile is a value that occurred", () => {
    // 1..10: nearest-rank p50 is the 5th value because ceil(0.5 * 10) = 5.
    expect(sortedPercentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.5)).toBe(5);
    expect(sortedPercentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.95)).toBe(10);
    expect(sortedPercentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0)).toBe(1);
    expect(sortedPercentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 1)).toBe(10);
  });

  it("clamps out-of-range fractions instead of reading past the array", () => {
    expect(sortedPercentile([4, 8], 1.5)).toBe(8);
    expect(sortedPercentile([4, 8], -1)).toBe(4);
  });

  it("returns NaN for no observations rather than 0", () => {
    expect(sortedPercentile([], 0.5)).toBeNaN();
  });
});

describe("summarize", () => {
  it("reports n, extremes, and the two percentiles a report quotes", () => {
    expect(summarize([30, 10, 20])).toEqual({
      n: 3,
      min: 10,
      p50: 20,
      p95: 30,
      max: 30,
      mean: 20,
    });
  });

  it("drops non-finite values instead of averaging them in", () => {
    // A `NaN` means "this run produced no measurement"; including it would understate the mean and
    // make the tail look better than it was.
    const summary = summarize([10, Number.NaN, 20]);
    expect(summary.n).toBe(2);
    expect(summary.mean).toBe(15);
  });

  it("reports an empty summary as all-NaN, never as zero", () => {
    expect(summarize([])).toEqual({
      n: 0,
      min: Number.NaN,
      p50: Number.NaN,
      p95: Number.NaN,
      max: Number.NaN,
      mean: Number.NaN,
    });
  });
});

describe("rounding helpers", () => {
  it("rounds to the requested decimals and leaves non-finite values alone", () => {
    expect(round(1.23456, 2)).toBe(1.23);
    expect(round(Number.NaN)).toBeNaN();
    expect(round(Number.POSITIVE_INFINITY)).toBe(Number.POSITIVE_INFINITY);
  });

  it("summarizes with rounding applied to every field", () => {
    expect(summarizeRounded([1.111, 2.222, 3.333], 1)).toEqual({
      n: 3,
      min: 1.1,
      p50: 2.2,
      p95: 3.3,
      max: 3.3,
      mean: 2.2,
    });
  });

  it("converts bytes to decimal megabytes, the unit the docs use", () => {
    // 68 000 000 B is 68.0 MB in decimal, not 64.8 MiB: the manifest's sizes are decimal.
    expect(megabytes(68_000_000)).toBe(68);
    expect(megabytes(161_921_677)).toBe(161.9);
  });

  it("takes the median, not the mean, for a single reported duration", () => {
    expect(median([5, 1, 3])).toBe(3);
    expect(median([4, 1, 3, 2])).toBe(2);
  });
});
