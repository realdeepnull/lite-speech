/**
 * Signal measurements used by the quality harness.
 *
 * Every function is pure and operates on `Float32Array`, so the same code can compare two dtypes,
 * compare a streamed utterance with a full one, or measure a single buffer. No WAV round-trip: the
 * M1 spike went through 16-bit files because it only needed a listening check, and quantizing twice
 * would hide exactly the differences this harness exists to find.
 */

/** Level and shape of one buffer. */
export interface SignalStats {
  readonly samples: number;
  readonly seconds: number;
  readonly peak: number;
  readonly rms: number;
  /** Samples above 1.0 in absolute value; the model should never produce them, and a clip is audible. */
  readonly clipped: number;
  /** Fraction of samples at exactly `0`, which would indicate a gap or a failed frame. */
  readonly silenceRatio: number;
}

/** Measures one buffer. */
export function signalStats(pcm: Float32Array, sampleRate: number): SignalStats {
  let peak = 0;
  let sumSquares = 0;
  let clipped = 0;
  let silent = 0;
  for (const sample of pcm) {
    const magnitude = Math.abs(sample);
    if (magnitude > peak) {
      peak = magnitude;
    }
    if (magnitude > 1) {
      clipped += 1;
    }
    if (sample === 0) {
      silent += 1;
    }
    sumSquares += sample * sample;
  }
  return {
    samples: pcm.length,
    seconds: sampleRate > 0 ? pcm.length / sampleRate : 0,
    peak: quantize(peak, 4),
    rms: quantize(pcm.length > 0 ? Math.sqrt(sumSquares / pcm.length) : 0, 5),
    clipped,
    silenceRatio: quantize(pcm.length > 0 ? silent / pcm.length : 0, 4),
  };
}

/**
 * Difference between a candidate signal and a reference, after aligning their lengths.
 */
export interface SignalDifference {
  /**
   * Samples compared. Never more than the shorter of the two, so a length difference is reported
   * separately instead of showing up as a fabricated mismatch at the end.
   */
  readonly samples: number;
  /** Signal-to-noise ratio in dB; `Infinity` when the two buffers are identical. */
  readonly snrDb: number;
  /**
   * Pearson correlation of the two buffers.
   *
   * The number that decides how to read {@link SignalDifference.snrDb}. A quantized graph that adds
   * noise to the same waveform stays near `1`; a graph that *diverges* — different prosody, different
   * stop point — lands near `0`, and its SNR is negative simply because two different utterances are
   * subtracted from each other. Reporting both keeps the report from implying "quantization noise"
   * where the truth is "a different performance of the same sentence".
   */
  readonly correlation: number;
  /** Largest absolute sample difference. */
  readonly maxDiff: number;
  /** Length difference in samples, candidate minus reference. */
  readonly lengthDelta: number;
}

/**
 * Compares two signals sample by sample.
 *
 * Only meaningful for deterministic output (`temperature: 0`); the flow model samples by default, so
 * at the default temperature two runs of the *same* dtype also differ. The harness therefore asserts
 * nothing from this function unless the caller pinned the temperature, and says so in the report.
 */
export function compareSignals(reference: Float32Array, candidate: Float32Array): SignalDifference {
  const samples = Math.min(reference.length, candidate.length);
  let signal = 0;
  let noise = 0;
  let maxDiff = 0;
  let dot = 0;
  let referenceEnergy = 0;
  let candidateEnergy = 0;
  for (let index = 0; index < samples; index += 1) {
    const expected = reference[index] ?? 0;
    const actual = candidate[index] ?? 0;
    signal += expected * expected;
    const diff = expected - actual;
    noise += diff * diff;
    if (Math.abs(diff) > maxDiff) {
      maxDiff = Math.abs(diff);
    }
    dot += expected * actual;
    referenceEnergy += expected * expected;
    candidateEnergy += actual * actual;
  }
  const denominator = Math.sqrt(referenceEnergy * candidateEnergy);
  return {
    samples,
    snrDb: noise === 0 ? Number.POSITIVE_INFINITY : quantize(10 * Math.log10(signal / noise), 2),
    correlation: denominator === 0 ? Number.NaN : quantize(dot / denominator, 3),
    maxDiff: quantize(maxDiff, 5),
    lengthDelta: candidate.length - reference.length,
  };
}

/** Concatenates the chunks of a stream, which is how a streamed utterance is compared with a full one. */
export function concat(chunks: readonly Float32Array[]): Float32Array {
  const total = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const pcm = new Float32Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    pcm.set(chunk, offset);
    offset += chunk.length;
  }
  return pcm;
}

/**
 * Largest absolute jump between adjacent samples.
 *
 * A seam between two independently generated frames shows up as a discontinuity: the waveform's
 * slope is bounded by its content, so a step far outside the buffer's own distribution is the
 * signature of a splice. Compared against the same buffer's median step, not against a magic
 * constant, so the check does not depend on output level.
 */
export function maxStep(pcm: Float32Array): number {
  let max = 0;
  for (let index = 1; index < pcm.length; index += 1) {
    const step = Math.abs((pcm[index] ?? 0) - (pcm[index - 1] ?? 0));
    if (step > max) {
      max = step;
    }
  }
  return quantize(max, 5);
}

/**
 * Ratio of the largest sample step to the median sample step.
 *
 * A ratio close to 1 means "no sample stands out"; a large ratio means one discontinuity dominates.
 * Reported instead of a pass/fail threshold because the number is only interpretable next to the
 * `fp32` reference from the same corpus case.
 */
export function stepRatio(pcm: Float32Array): number {
  const steps: number[] = [];
  for (let index = 1; index < pcm.length; index += 1) {
    steps.push(Math.abs((pcm[index] ?? 0) - (pcm[index - 1] ?? 0)));
  }
  if (steps.length === 0) {
    return Number.NaN;
  }
  const sorted = steps.sort((a, b) => a - b);
  const median = sorted[Math.floor(sorted.length / 2)] ?? 0;
  const max = sorted.at(-1) ?? 0;
  return median === 0 ? Number.NaN : quantize(max / median, 2);
}

/** Rounds to a fixed number of decimals, keeping reports readable and diffs small. */
function quantize(value: number, decimals: number): number {
  if (!Number.isFinite(value)) {
    return value;
  }
  const factor = 10 ** decimals;
  return Math.round(value * factor) / factor;
}
