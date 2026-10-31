/**
 * The signal measurements the quality verdict rests on.
 *
 * Two properties matter most and are easy to get subtly wrong: a correlation near zero must not be
 * reported as "noise", and a stream comparison must not be fooled by concatenating buffers of
 * different lengths.
 */
import { describe, expect, it } from "vitest";

import { compareSignals, concat, maxStep, signalStats, stepRatio } from "../src/signal.js";

/** A deterministic pseudo-signal: a sine at `hz`, which is enough to test the maths. */
function sine(length: number, hz = 2): Float32Array {
  const pcm = new Float32Array(length);
  for (let index = 0; index < length; index += 1) {
    pcm[index] = Math.sin((2 * Math.PI * hz * index) / length) * 0.5;
  }
  return pcm;
}

describe("signalStats", () => {
  it("measures level, peak, and duration from the samples themselves", () => {
    const stats = signalStats(new Float32Array([0.5, -0.5, 0.5, -0.5]), 1000);
    expect(stats.samples).toBe(4);
    expect(stats.seconds).toBe(0.004);
    expect(stats.peak).toBe(0.5);
    expect(stats.rms).toBe(0.5);
    expect(stats.clipped).toBe(0);
  });

  it("counts clipped samples instead of clamping them silently", () => {
    const stats = signalStats(new Float32Array([0, 1.4, -1.2, 0]), 1000);
    expect(stats.clipped).toBe(2);
  });

  it("reports the silent fraction, which exposes a dropped frame", () => {
    expect(signalStats(new Float32Array([0, 0, 0.5, 0.5]), 1000).silenceRatio).toBe(0.5);
  });

  it("reports an unknown rate as zero seconds rather than throwing", () => {
    expect(signalStats(new Float32Array([0.1]), 0).seconds).toBe(0);
  });
});

describe("compareSignals", () => {
  it("calls two identical buffers identical with a perfect correlation", () => {
    const signal = sine(64);
    const difference = compareSignals(signal, signal.slice());
    expect(difference.snrDb).toBe(Number.POSITIVE_INFINITY);
    expect(difference.correlation).toBe(1);
    expect(difference.maxDiff).toBe(0);
    expect(difference.lengthDelta).toBe(0);
  });

  it("separates 'added noise' from 'a different utterance' by correlation", () => {
    const reference = sine(256);
    // Added noise: the waveform still follows the reference.
    const noisy = reference.map((sample) => sample + Math.sin(sample * 40) * 0.02);
    const noiseDifference = compareSignals(reference, noisy);
    expect(noiseDifference.correlation).toBeGreaterThan(0.9);

    // A different performance: uncorrelated with the reference, so its SNR is negative because two
    // unrelated waveforms are being subtracted — not because the model added noise.
    const other = sine(256, 7);
    const otherDifference = compareSignals(reference, other);
    expect(Math.abs(otherDifference.correlation ?? 0)).toBeLessThan(0.3);
    expect(otherDifference.snrDb).toBeLessThan(0);
  });

  it("compares only the overlapping part and reports the length difference separately", () => {
    const difference = compareSignals(new Float32Array([1, 1]), new Float32Array([1, 1, 1, 1]));
    expect(difference.samples).toBe(2);
    expect(difference.lengthDelta).toBe(2);
  });

  it("reports an undefined correlation for a silent candidate instead of dividing by zero", () => {
    const difference = compareSignals(sine(32), new Float32Array(32));
    expect(difference.correlation).toBeNaN();
  });
});

describe("concat", () => {
  it("preserves order and total length", () => {
    const joined = concat([new Float32Array([1, 2]), new Float32Array([3]), new Float32Array([])]);
    expect([...joined]).toEqual([1, 2, 3]);
  });
});

describe("step statistics", () => {
  it("finds the largest adjacent jump, which is the signature of a splice", () => {
    expect(maxStep(new Float32Array([0, 0, 0.9, 0]))).toBe(0.9);
  });

  it("normalizes the largest jump against the median, so it does not depend on output level", () => {
    // A smooth ramp has every step equal, so the ratio is 1 regardless of amplitude.
    expect(stepRatio(new Float32Array([0, 0.1, 0.2, 0.3, 0.4]))).toBe(1);
  });

  it("returns NaN for a buffer with no adjacent samples rather than 0", () => {
    expect(stepRatio(new Float32Array([0.5]))).toBeNaN();
    expect(maxStep(new Float32Array([0.5]))).toBe(0);
  });
});
