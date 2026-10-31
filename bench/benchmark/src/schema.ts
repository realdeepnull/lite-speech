/**
 * The shape of a committed benchmark report.
 *
 * A report is evidence, so it records everything needed to decide whether a number applies to the
 * reader's machine: runtime, device, dtype, model revision, the text that was synthesized, and the
 * raw per-run samples behind every summary. `schemaVersion` exists because reports are committed
 * and read for a long time; a change that would make an old report mean something else has to bump
 * it.
 *
 * Nothing here imports `@lite-speech/core`: the report is a plain JSON contract that the quality
 * harness and `docs/performance.md` also read.
 */
import type { Summary } from "./stats.js";

/** Version of {@link BenchmarkReport}; bump when a field changes meaning. */
export const REPORT_SCHEMA_VERSION = 1;

/** Which runtime produced the numbers. */
type ReportRuntime = "node" | "chromium";

/** Whether the cache was empty or warm when the run started. */
type ReportPhase = "cold" | "warm";

/** Hardware and software identity of the measuring machine. */
interface ReportEnvironment {
  /** Node.js version, when the runtime is Node. */
  readonly node?: string;
  /** `navigator.userAgent`, when the runtime is a browser. */
  readonly userAgent?: string;
  /** Operating system, as reported by the runtime. */
  readonly os?: string;
  /** CPU model, when it can be named. */
  readonly cpu?: string;
  /** CPU architecture, e.g. `"x64"`. */
  readonly arch?: string;
  /** Logical core count. */
  readonly cores?: number;
  /** WebGPU adapter description, when the run used one. */
  readonly gpu?: string;
  /** Resolved ONNX device, e.g. `"cpu"`, `"wasm"`, `"webgpu"`. */
  readonly device: string;
  /** Resolved dtype, e.g. `"fp32"`, `"q4"`. */
  readonly dtype: string;
}

/** The model profile the numbers belong to. */
interface ReportModel {
  readonly id: string;
  /** Pinned commit SHA, never a branch name. */
  readonly revision: string;
  /** Truthful streaming classification reported by the engine. */
  readonly streaming?: string;
  /** Sample rate of the produced audio. */
  readonly sampleRate?: number;
  /** Voice used for the measurement. */
  readonly voice?: string;
}

/** The input every run synthesized. */
interface ReportText {
  /** Stable identifier so two reports can be compared. */
  readonly id: string;
  readonly characters: number;
  /** Duration of the produced audio, median over the runs. */
  readonly audioSeconds: number;
}

/** Asset sizes a consent dialog would show; taken from the manifest, never from the network. */
interface ReportDownload {
  /** Bytes that still had to be fetched when the run started. */
  readonly bytes: number;
  /** Number of files in that set. */
  readonly files: number;
  /** `bytes` in decimal megabytes, for the prose in the docs. */
  readonly megabytes: number;
  /** Licenses of the model (and voice) assets. */
  readonly licenses: readonly string[];
}

/** One measured stream. */
export interface ReportRun {
  /** Wall time from starting `preload()` to the load being complete. */
  readonly loadMs: number;
  /** Wall time from starting the stream to the first non-empty chunk. */
  readonly ttfaMs: number;
  /** Wall time from starting the stream to the last chunk. */
  readonly streamMs: number;
  /** Audio seconds produced. */
  readonly audioSeconds: number;
  /** `streamMs / audioSeconds / 1000`: real-time factor, lower is faster. */
  readonly rtf: number;
  /** Chunks the stream delivered, including the terminal `final` marker. */
  readonly chunks: number;
  /** Peak resident set after the run, when the runtime can report it (Node.js). */
  readonly peakRssBytes?: number;
  /** Whether this run was the first one in the process. */
  readonly first: boolean;
}

/** What the whole measurement produced. */
export interface BenchmarkReport {
  readonly schemaVersion: number;
  /** ISO-8601 timestamp of the run. */
  readonly date: string;
  readonly runtime: ReportRuntime;
  readonly phase: ReportPhase;
  /**
   * `"ok"` when numbers were produced, `"unavailable"` when the runtime could not run this
   * configuration (for example WebGPU without an adapter). An unavailable entry is a result: it
   * keeps the gap visible instead of letting a missing file look like "not attempted".
   */
  readonly status: "ok" | "unavailable";
  /** Why the configuration could not be measured, for `status: "unavailable"`. */
  readonly unavailableReason?: string;
  readonly environment: ReportEnvironment;
  readonly model: ReportModel;
  readonly text: ReportText;
  readonly download: ReportDownload;
  /** Milliseconds from starting `preload()` to the first audio chunk, cold phase only. */
  readonly coldStartMs?: number;
  /** Milliseconds the load itself took during the cold phase. */
  readonly provisionMs?: number;
  /** Raw per-run samples, in run order. */
  readonly runs: readonly ReportRun[];
  /** Latency summaries derived from {@link BenchmarkReport.runs}. */
  readonly summary: {
    readonly loadMs: Summary;
    readonly ttfaMs: Summary;
    readonly rtf: Summary;
    readonly streamMs: Summary;
    readonly peakRssBytes?: Summary;
  };
  /** Milliseconds from `abort()` to the rejected iteration, when measured. */
  readonly cancellationMs?: Summary;
  /** Anything a reader must know to interpret the numbers honestly. */
  readonly notes: readonly string[];
}
