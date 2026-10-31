/**
 * One measurement of one configuration: start the model, stream a known text, observe it.
 *
 * The runner is shared by the Node.js and the Chromium entry points on purpose. What makes a
 * benchmark comparable across runtimes is not that the numbers came from the same process, but
 * that they *mean* the same thing — so the phase split, the time-to-first-audio definition, and the
 * real-time factor are written once, here, and the runtimes only supply the engine, memory
 * sampling, and their own identity.
 */
import {
  createTts,
  isTtsError,
  TTS_ERROR_CODES,
  type DebugEvent,
  type Tts,
  type TtsEngine,
} from "@lite-speech/core";

import { REPORT_SCHEMA_VERSION, type BenchmarkReport, type ReportRun } from "./schema.js";
import { median, megabytes, summarizeRounded } from "./stats.js";

/** The text every committed report is measured with, so reports stay comparable. */
const BENCH_TEXT = {
  id: "bench-standard-v1",
  text:
    "Hello from lite-speech. This sentence is the fixed benchmark input, and it is long " +
    "enough to produce several seconds of audio.",
} as const;

/** Runtime identity, contributed by the entry point that knows its runtime. */
type EnvironmentProvider = () => Partial<BenchmarkReport["environment"]>;

/** Options for {@link measure}. */
export interface MeasureOptions {
  readonly engine: TtsEngine;
  readonly runtime: BenchmarkReport["runtime"];
  /** Device the engine resolved, for the report header. */
  readonly device: string;
  /** Dtype the engine resolved, for the report header. */
  readonly dtype: string;
  /** Voice to synthesize with; defaults to the engine's first voice. */
  readonly voice?: string;
  /** How many *additional* streams to measure after the first, cold one. Default 4. */
  readonly warmRuns?: number;
  /** Reads the process's resident set, where the runtime can (Node.js). */
  readonly peakMemory?: () => number | Promise<number>;
  /** Whether the cache was empty when this process started. */
  readonly phase: BenchmarkReport["phase"];
  /** Runtime identity: OS, CPU, user agent, GPU. */
  readonly environment: EnvironmentProvider;
  /** Statements a reader needs, e.g. "the cache was pre-populated by an earlier run". */
  readonly notes?: readonly string[];
  /** Attempts at measuring cancellation latency. Default 3. */
  readonly cancellationAttempts?: number;
}

/** A mutable route for debug events, so one `Tts` can time several distinct runs. */
interface DebugSink {
  route?: ((event: DebugEvent) => void) | undefined;
}

/**
 * Measures one configuration.
 *
 * The first stream is the cold one: it includes whatever the runtime defers until first use
 * (shader compilation, kernel selection, a lazily grown WASM heap), and reporting it separately is
 * the difference between "cold start" and "steady state". Warm runs follow on the same instance,
 * which is what a page that synthesizes twice actually experiences — and what a second `Tts` cannot
 * measure, because `Tts.dispose()` owns the engine.
 *
 * Time-to-first-audio comes from the engine's own `time-to-first-chunk` event rather than from a
 * timestamp around the iterator: the event fires once the first frame exists, so the number is the
 * engine's latency and not this harness's bookkeeping.
 */
export async function measure(options: MeasureOptions): Promise<BenchmarkReport> {
  const warmRuns = options.warmRuns ?? 4;
  const sink: DebugSink = {};
  const tts = createTts({
    engine: options.engine,
    offline: "allow",
    ...(options.voice !== undefined && { voice: options.voice }),
    onDebug: (event) => sink.route?.(event),
  });

  const runs: ReportRun[] = [];
  // Read *before* the first stream: afterwards the assets are cached, so `missingBytes` would
  // describe the state after provisioning rather than what the caller had to download.
  const download = await describeDownload(tts);
  for (let index = 0; index <= warmRuns; index += 1) {
    runs.push(await measureStream(tts, options, sink, index === 0));
  }

  const capabilities = tts.capabilities();
  const cancellation = await measureCancellation(tts, options);
  await tts.dispose();

  const peakReadings = runs
    .map((run) => run.peakRssBytes)
    .filter((value): value is number => value !== undefined);
  const cold = runs[0];

  return {
    schemaVersion: REPORT_SCHEMA_VERSION,
    date: new Date().toISOString(),
    runtime: options.runtime,
    phase: options.phase,
    status: "ok",
    environment: {
      ...options.environment(),
      device: capabilities?.device ?? options.device,
      dtype: capabilities?.dtype ?? options.dtype,
    },
    model: {
      id: capabilities?.modelId ?? "unknown",
      revision: capabilities?.modelVersion ?? "unknown",
      ...(capabilities?.streaming !== undefined && { streaming: capabilities.streaming }),
      ...(capabilities?.sampleRate !== undefined && { sampleRate: capabilities.sampleRate }),
      ...(options.voice !== undefined && { voice: options.voice }),
    },
    text: {
      id: BENCH_TEXT.id,
      characters: BENCH_TEXT.text.length,
      audioSeconds: round(median(runs.map((run) => run.audioSeconds)), 2),
    },
    download,
    // The first `preload()` includes the load; later runs reuse it, so `loadMs` is only meaningful
    // for the cold run and is summarized from that run alone (`n: 1` in the summary).
    ...(cold !== undefined && { coldStartMs: cold.loadMs }),
    runs,
    summary: {
      loadMs: summarizeRounded(cold === undefined ? [] : [cold.loadMs]),
      ttfaMs: summarizeRounded(runs.map((run) => run.ttfaMs)),
      rtf: summarizeRounded(
        runs.map((run) => run.rtf),
        3,
      ),
      streamMs: summarizeRounded(runs.map((run) => run.streamMs)),
      ...(peakReadings.length > 0 && { peakRssBytes: summarizeRounded(peakReadings, 0) }),
    },
    ...(cancellation.length > 0 && { cancellationMs: summarizeRounded(cancellation) }),
    notes: [...(options.notes ?? [])],
  };
}

/**
 * One streamed run.
 *
 * Peak memory is sampled *during* the stream, not read afterwards: a resident set released before
 * the loop returns would otherwise report a comfortable number for a run that allocated hundreds of
 * megabytes. The 25 ms interval is far below the duration of even the shortest measured stream, so
 * the peak is not missed between samples.
 */
async function measureStream(
  tts: Tts,
  options: MeasureOptions,
  sink: DebugSink,
  first: boolean,
): Promise<ReportRun> {
  let ttfcMs: number | undefined;
  sink.route = (event: DebugEvent): void => {
    if (event.type === "time-to-first-chunk" && ttfcMs === undefined) {
      ttfcMs = event.durationMs;
    }
  };

  let peak: number | undefined;
  const sample = async (): Promise<void> => {
    if (options.peakMemory === undefined) {
      return;
    }
    const value = await options.peakMemory();
    peak = peak === undefined ? value : Math.max(peak, value);
  };
  const sampler =
    options.peakMemory === undefined ? undefined : setInterval(() => void sample(), 25);
  await sample();

  const loadStarted = now();
  await tts.preload();
  const loadMs = now() - loadStarted;

  const started = now();
  let chunks = 0;
  let samples = 0;
  let sampleRate = 0;
  try {
    for await (const chunk of tts.stream(BENCH_TEXT.text)) {
      chunks += 1;
      if (chunk.sampleRate > 0) {
        sampleRate = chunk.sampleRate;
      }
      samples += chunk.pcm.length;
    }
  } finally {
    if (sampler !== undefined) {
      clearInterval(sampler);
    }
    await sample();
  }
  const streamMs = now() - started;
  const audioSeconds = sampleRate > 0 ? samples / sampleRate : 0;

  return {
    loadMs: round(loadMs),
    ttfaMs: round(ttfcMs ?? Number.NaN),
    streamMs: round(streamMs),
    audioSeconds: round(audioSeconds, 2),
    rtf: audioSeconds > 0 ? round(streamMs / 1000 / audioSeconds, 3) : Number.NaN,
    chunks,
    ...(peak !== undefined && { peakRssBytes: peak }),
    first,
  };
}

/**
 * Milliseconds from `abort()` to the iteration rejecting with `ABORTED`.
 *
 * The abort is issued after the first chunk exists, so the measured interval contains real
 * generation work; aborting before any work started would only time a promise rejection.
 */
async function measureCancellation(tts: Tts, options: MeasureOptions): Promise<number[]> {
  const timings: number[] = [];
  const attempts = options.cancellationAttempts ?? 3;
  for (let attempt = 0; attempt < attempts; attempt += 1) {
    const controller = new AbortController();
    const stream = tts.stream(BENCH_TEXT.text, { signal: controller.signal });
    const iterator = stream[Symbol.asyncIterator]();
    const first = await iterator.next();
    if (first.done === true) {
      break;
    }
    const abortedAt = now();
    controller.abort();
    try {
      for (;;) {
        const next = await iterator.next();
        if (next.done === true) {
          break;
        }
      }
      // The stream finished before the abort took effect: not a cancellation measurement.
      break;
    } catch (error) {
      if (!isTtsError(error) || error.code !== TTS_ERROR_CODES.ABORTED) {
        // A different failure is a real defect; recording a "cancellation latency" for it would
        // hide the bug behind a plausible number.
        break;
      }
      timings.push(round(now() - abortedAt));
    }
  }
  return timings;
}

/** Reads the asset plan without touching the network. */
async function describeDownload(tts: Tts): Promise<BenchmarkReport["download"]> {
  try {
    const plan = await tts.describeAssets();
    return {
      bytes: plan.missingBytes,
      files: plan.missing.length,
      megabytes: megabytes(plan.missingBytes),
      licenses: plan.licenses,
    };
  } catch {
    return { bytes: 0, files: 0, megabytes: 0, licenses: [] };
  }
}

/** Returns a high-resolution timestamp in milliseconds. */
function now(): number {
  return typeof performance === "undefined" ? Date.now() : performance.now();
}

/** Rounds to a fixed number of decimals, keeping reports readable and diffs small. */
function round(value: number, decimals = 2): number {
  if (!Number.isFinite(value)) {
    return value;
  }
  const factor = 10 ** decimals;
  return Math.round(value * factor) / factor;
}
