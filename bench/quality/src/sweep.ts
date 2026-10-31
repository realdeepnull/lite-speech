/**
 * The quality sweep: for every corpus case, synthesize once and stream once, then compare.
 *
 * Two questions are answered separately, because they fail for different reasons:
 *
 * 1. **Chunk consistency.** Does `stream()` produce the same audio as `synthesize()`, in pieces? A
 *    stream that loses frames at a segment seam, or that changes sample rate halfway, sounds like
 *    speech with holes in it. Comparing the concatenated stream against the full utterance catches
 *    that without a human.
 * 2. **Quantization quality.** How far is `q4`/`q4f16` from `fp32`? Only meaningful at
 *    `temperature: 0`, because the flow model samples by default — the report states the
 *    temperature it ran at, and refuses to compute an SNR when the comparison would be noise.
 *
 * Listening is deliberately *not* automated. The harness produces the audio and a blind protocol;
 * the score stays `"pending"` until a human fills it in, because a fabricated MOS would be the one
 * number in this repository that is not a measurement.
 */
import { encodeWav, type AudioChunk, type Tts } from "@lite-speech/core";

import { CORPUS } from "./corpus.js";
import { QUALITY_SCHEMA_VERSION, type CaseReport, type QualityReport } from "./schema.js";
import { compareSignals, concat, signalStats, stepRatio } from "./signal.js";

/** Audio produced by one full synthesis, kept in memory for the reference comparison. */
export interface ReferenceAudio {
  readonly caseId: string;
  readonly pcm: Float32Array;
  readonly sampleRate: number;
}

/** Options for {@link sweep}. */
export interface SweepOptions {
  /** The instance under test; the sweep disposes it when it finishes. */
  readonly tts: Tts;
  readonly runtime: string;
  readonly device: string;
  readonly dtype: string;
  /**
   * Sampling temperature the engine was configured with. Only `0` makes a sample-by-sample
   * comparison meaningful, so a non-zero value suppresses `versusReference`.
   */
  readonly temperature: number;
  /** Reference audio from the `fp32` run of the same corpus, for a quantized sweep. */
  readonly reference?: readonly ReferenceAudio[];
  /** Where to write one WAV per case, for the listening test. Omit to skip writing audio. */
  readonly writeWav?: (file: string, bytes: Uint8Array) => Promise<void>;
  /**
   * Re-synthesizes the first case and reports whether the output is byte-identical.
   *
   * This is what makes `versusReference` interpretable. If two `fp32` runs of the same case differ,
   * every "difference against fp32" is dominated by the sampler rather than by quantization — and a
   * report that showed an SNR without that check would invite exactly the wrong conclusion. Costs one
   * extra synthesis, so it is opt-in rather than always on.
   */
  readonly checkDeterminism?: boolean;
  /** Model identity for the report header. */
  readonly modelId: string;
  readonly modelRevision: string;
  readonly notes?: readonly string[];
}

/**
 * Runs the sweep.
 *
 * `synthesize()` and `stream()` are both called for every case and both results are kept: the full
 * utterance is the reference a stream must reproduce, and it is also what a listener would otherwise
 * hear, so a seam that only exists in the streamed path is a defect in the stream protocol rather
 * than in the model.
 */
export async function sweep(options: SweepOptions): Promise<{
  report: QualityReport;
  reference: ReferenceAudio[];
}> {
  const tts = options.tts;
  const cases: CaseReport[] = [];
  const reference: ReferenceAudio[] = [];
  const comparable = options.temperature === 0;
  const byId = new Map((options.reference ?? []).map((entry) => [entry.caseId, entry]));
  let determinism: QualityReport["determinism"];

  try {
    await tts.preload();

    for (const corpusCase of CORPUS) {
      const full = await tts.synthesize(corpusCase.text);
      if (full.pcm.length === 0) {
        // An empty result is a finding, not a measurement: recording it as "0 seconds" would look
        // like a very short utterance and hide that the model produced nothing.
        throw new Error(`Case "${corpusCase.id}" produced no audio`);
      }
      const chunks: AudioChunk[] = [];
      for await (const chunk of tts.stream(corpusCase.text)) {
        if (chunk.pcm.length > 0) {
          chunks.push(chunk);
        }
      }
      const streamed = concat(chunks.map((chunk) => chunk.pcm));
      const fullStats = signalStats(full.pcm, full.sampleRate);
      const streamStats = signalStats(streamed, full.sampleRate);
      const expected = byId.get(corpusCase.id);

      cases.push({
        id: corpusCase.id,
        probes: corpusCase.probes,
        expectation: corpusCase.expectation,
        characters: corpusCase.text.length,
        full: {
          seconds: fullStats.seconds,
          peak: fullStats.peak,
          rms: fullStats.rms,
          clipped: fullStats.clipped,
          silenceRatio: fullStats.silenceRatio,
        },
        stream: {
          chunks: chunks.length,
          frameSamples: [...new Set(chunks.map((chunk) => chunk.pcm.length))].sort((a, b) => a - b),
          seconds: streamStats.seconds,
          samplesRatio:
            full.pcm.length === 0 ? Number.NaN : quantize(streamed.length / full.pcm.length, 3),
          stepRatio: stepRatio(streamed),
          fullStepRatio: stepRatio(full.pcm),
        },
        ...(expected !== undefined &&
          comparable && {
            versusReference: describeDifference(expected.pcm, full.pcm),
          }),
      });

      reference.push({ caseId: corpusCase.id, pcm: full.pcm, sampleRate: full.sampleRate });
      if (options.writeWav !== undefined) {
        const wav = encodeWav({ pcm: full.pcm, sampleRate: full.sampleRate }, { bitDepth: 16 });
        await options.writeWav(`${options.device}-${options.dtype}-${corpusCase.id}.wav`, wav);
      }
    }
    // Inside the `try`, because `Tts.dispose()` owns the engine: a repeat after the `finally` would
    // be a disposed instance, which is exactly the bug this check is supposed to catch elsewhere.
    determinism =
      options.checkDeterminism === true ? await checkDeterminism(tts, reference) : undefined;
  } finally {
    await tts.dispose();
  }

  const report: QualityReport = {
    schemaVersion: QUALITY_SCHEMA_VERSION,
    date: new Date().toISOString(),
    runtime: options.runtime,
    device: options.device,
    dtype: options.dtype,
    temperature: options.temperature,
    model: { id: options.modelId, revision: options.modelRevision },
    cases,
    ...(determinism !== undefined && { determinism }),
    listening: {
      score: "pending",
      protocol:
        "Blind A/B/C over the three dtypes per corpus case, 1–5 naturalness, at least three " +
        "raters, median reported. The WAV files this sweep writes are the stimulus set.",
      raters: 0,
    },
    notes: [
      comparable
        ? "Temperature 0: `versusReference` compares sample by sample against the fp32 run."
        : `Temperature ${String(options.temperature)}: the flow model samples, so a sample-by-sample comparison against another run would measure randomness rather than quantization. ` +
          "`versusReference` is omitted; only level, length, and step statistics are comparable.",
      "`stream.samplesRatio` is the concatenated stream length over the full-utterance length from the same configuration, so it isolates the stream path from model variance.",
      ...(options.notes ?? []),
    ],
  };
  return { report, reference };
}

/**
 * Re-synthesizes the first case and compares it with the buffer the sweep already produced.
 *
 * Reports both the exact answer (identical or not) and the shape of the difference, because "not
 * identical" is not automatically a defect: at a non-zero temperature the model is *supposed* to
 * differ. What the field decides is whether `versusReference` measures quantization or randomness.
 */
async function checkDeterminism(
  tts: Tts,
  reference: readonly ReferenceAudio[],
): Promise<{ readonly caseId: string; readonly identical: boolean; readonly maxDiff: number }> {
  const first = reference[0];
  if (first === undefined) {
    return { caseId: "", identical: false, maxDiff: Number.NaN };
  }
  const again = await tts.synthesize(first.caseId === "" ? "" : textOf(first.caseId));
  const difference = compareSignals(first.pcm, again.pcm);
  return {
    caseId: first.caseId,
    identical: difference.snrDb === Number.POSITIVE_INFINITY && difference.lengthDelta === 0,
    maxDiff: difference.maxDiff,
  };
}

/** The corpus text of a case id, so the determinism check re-runs the same input. */
function textOf(caseId: string): string {
  const entry = CORPUS.find((candidate) => candidate.id === caseId);
  if (entry === undefined) {
    throw new Error(`No corpus case named "${caseId}"`);
  }
  return entry.text;
}

/** Builds the reference comparison for one case. */
function describeDifference(
  reference: Float32Array,
  candidate: Float32Array,
): NonNullable<CaseReport["versusReference"]> {
  const difference = compareSignals(reference, candidate);
  return {
    snrDb: Number.isFinite(difference.snrDb) ? difference.snrDb : "identical",
    correlation: difference.correlation,
    maxDiff: difference.maxDiff,
    lengthDelta: difference.lengthDelta,
  };
}

/** Builds the reference comparison for one case. */
function quantize(value: number, decimals: number): number {
  if (!Number.isFinite(value)) {
    return value;
  }
  const factor = 10 ** decimals;
  return Math.round(value * factor) / factor;
}
