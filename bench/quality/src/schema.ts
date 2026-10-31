/**
 * Shape of a committed quality report.
 *
 * Separate from the benchmark schema because it answers a different question: not "how fast" but
 * "does it still sound right, and does streaming produce the same audio as one-shot synthesis".
 * Both are committed JSON for the same reason — a claim in the docs has to trace to a file.
 */

/** Version of {@link QualityReport}; bump when a field changes meaning. */
export const QUALITY_SCHEMA_VERSION = 1;

/** Measurements of one corpus case in one configuration. */
export interface CaseReport {
  readonly id: string;
  readonly probes: string;
  readonly expectation: "supported" | "known-limitation";
  readonly characters: number;
  /** Audio produced by a full `synthesize()` call. */
  readonly full: {
    readonly seconds: number;
    readonly peak: number;
    readonly rms: number;
    readonly clipped: number;
    readonly silenceRatio: number;
  };
  /**
   * Streamed output compared with the full utterance from the same configuration.
   *
   * `samplesRatio` near 1 and `stepRatio` close to the full buffer's mean the stream is the same
   * audio in pieces; a large length difference means frames are lost at a segment seam.
   */
  readonly stream: {
    readonly chunks: number;
    readonly frameSamples: readonly number[];
    readonly seconds: number;
    readonly samplesRatio: number;
    readonly stepRatio: number;
    readonly fullStepRatio: number;
  };
  /**
   * Difference against the `fp32` reference of the same case, for a quantized dtype.
   *
   * Absent for `fp32` itself (it *is* the reference) and meaningless when the run did not pin
   * `temperature: 0`, which the report states once in {@link QualityReport.temperature}.
   */
  readonly versusReference?: {
    readonly snrDb: number | "identical" | "not-comparable";
    /** Pearson correlation with the reference; near `0` means a different performance, not noise. */
    readonly correlation: number;
    readonly maxDiff: number;
    readonly lengthDelta: number;
  };
}

/** What the whole quality sweep produced. */
export interface QualityReport {
  readonly schemaVersion: number;
  readonly date: string;
  readonly runtime: string;
  readonly device: string;
  readonly dtype: string;
  /** The sampling temperature; `0` is the only setting where a sample-by-sample comparison is valid. */
  readonly temperature: number;
  readonly model: {
    readonly id: string;
    readonly revision: string;
  };
  readonly cases: readonly CaseReport[];
  /**
   * Whether two syntheses of the same text produced identical audio.
   *
   * Present only when the sweep was asked to check it. It decides how to read `versusReference`: a
   * non-deterministic model makes a sample-by-sample comparison against another run meaningless.
   */
  readonly determinism?: {
    readonly caseId: string;
    readonly identical: boolean;
    readonly maxDiff: number;
  };
  /**
   * Listening score, as the documented MOS-style protocol asks for.
   *
   * `"pending"` until a human has actually listened: inventing a number here would be the one
   * dishonest thing this harness could do, since every other field is produced by a measurement.
   */
  readonly listening: {
    readonly score: "pending" | number;
    readonly protocol: string;
    readonly raters: number;
  };
  readonly notes: readonly string[];
}
