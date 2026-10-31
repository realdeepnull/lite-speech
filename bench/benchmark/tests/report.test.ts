/**
 * The report contract is what makes a published number traceable.
 *
 * These tests cover the two properties a reader depends on: an unavailable configuration keeps its
 * row (a gap must be visible), and every row names the runtime, device, dtype, and phase, because a
 * latency without them is not a measurement of anything in particular.
 */
import { describe, expect, it } from "vitest";

import { renderMarkdown, reportFileName } from "../src/report.js";
import { REPORT_SCHEMA_VERSION, type BenchmarkReport } from "../src/schema.js";
import type { Summary } from "../src/stats.js";

const summary = (p50: number, p95 = p50): Summary => ({
  n: 5,
  min: p50,
  p50,
  p95,
  max: p95,
  mean: (p50 + p95) / 2,
});

/** A minimal measured report; fields a test does not care about are still present. */
function report(overrides: Partial<BenchmarkReport> = {}): BenchmarkReport {
  return {
    schemaVersion: REPORT_SCHEMA_VERSION,
    date: "2026-11-01T00:00:00.000Z",
    runtime: "node",
    phase: "cold",
    status: "ok",
    environment: { device: "cpu", dtype: "q4", node: "v24.18.0", os: "win32 x64" },
    model: { id: "onnx-community/phonon-31m-en-ONNX", revision: "4f0cda7c" },
    text: { id: "bench-standard-v1", characters: 126, audioSeconds: 7.4 },
    download: { bytes: 67_989_146, files: 7, megabytes: 68, licenses: ["Apache-2.0"] },
    runs: [
      {
        loadMs: 5405,
        ttfaMs: 110,
        streamMs: 1600,
        audioSeconds: 7.4,
        rtf: 0.216,
        chunks: 94,
        peakRssBytes: 304_300_000,
        first: true,
      },
    ],
    summary: {
      loadMs: summary(5405),
      ttfaMs: summary(38, 110),
      rtf: summary(0.216, 0.244),
      streamMs: summary(1600),
      peakRssBytes: summary(304_300_000),
    },
    cancellationMs: summary(0.5),
    notes: [],
    ...overrides,
  };
}

describe("reportFileName", () => {
  it("names the configuration, so two reports cannot overwrite each other", () => {
    expect(reportFileName(report())).toBe("node-cpu-q4-cold.json");
  });

  it("distinguishes the phase, which is the difference between a cold and a warm claim", () => {
    expect(reportFileName(report({ phase: "warm" }))).toBe("node-cpu-q4-warm.json");
  });

  it("strips characters a file name should not carry", () => {
    expect(
      reportFileName(report({ environment: { device: "webgpu (amd)", dtype: "q4/f16" } })),
    ).toBe("node-webgpuamd-q4f16-cold.json");
  });
});

describe("renderMarkdown", () => {
  it("renders one row per report with every identifying column", () => {
    const markdown = renderMarkdown([report()]);
    const row = markdown.split("\n").find((line) => line.startsWith("| node"));
    expect(row).toBeDefined();
    expect(row).toContain("| node ");
    expect(row).toContain("| cpu ");
    expect(row).toContain("| q4 ");
    expect(row).toContain("| cold ");
    expect(row).toContain("| 68.0 ");
    expect(row).toContain("| 0.216 ");
  });

  it("labels the load column as a single first measurement, not a percentile", () => {
    // `loadMs` is summarized from the first `preload()` only: later streams reuse it. Calling that a
    // p50 would imply a distribution that does not exist.
    const markdown = renderMarkdown([report()]);
    expect(markdown).toContain("load first ms");
    expect(markdown).not.toContain("load p50");
    expect(markdown).toContain("n = 1");
  });

  it("keeps an unavailable configuration visible instead of dropping the row", () => {
    // A missing file would read as "not attempted"; the point of the row is to record that the
    // configuration *was* attempted and could not run here.
    const markdown = renderMarkdown([
      report({
        status: "unavailable",
        unavailableReason: "no adapter is available",
        runs: [],
      }),
    ]);
    const row = markdown.split("\n").find((line) => line.startsWith("| node"));
    expect(row).toContain("unavailable (no adapter is available)");
    expect(row).not.toContain("0.000");
  });

  it("prints an em dash for a quantity that was not measured", () => {
    const partial: BenchmarkReport = {
      schemaVersion: REPORT_SCHEMA_VERSION,
      date: "2026-11-01T00:00:00.000Z",
      runtime: "node",
      phase: "warm",
      status: "ok",
      environment: { device: "cpu", dtype: "q4" },
      model: { id: "onnx-community/phonon-31m-en-ONNX", revision: "4f0cda7c" },
      text: { id: "bench-standard-v1", characters: 126, audioSeconds: 7.4 },
      download: { bytes: 0, files: 0, megabytes: 0, licenses: [] },
      runs: [],
      summary: {
        loadMs: summary(1),
        ttfaMs: summary(1),
        rtf: summary(1),
        streamMs: summary(1),
      },
      notes: [],
    };
    const markdown = renderMarkdown([partial]);
    const row = markdown.split("\n").find((line) => line.startsWith("| node"));
    expect(row).toContain("—");
  });

  it("states the two definitions a reader would otherwise have to guess", () => {
    const markdown = renderMarkdown([report()]);
    expect(markdown).toContain("TTFA = time to first audio chunk");
    expect(markdown).toContain("RTF = stream wall time / audio duration");
    expect(markdown).toContain("not a browser tab budget");
  });
});
