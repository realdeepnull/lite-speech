/**
 * Writes a report and renders the Markdown table the docs quote from.
 *
 * A report that only exists as a console line cannot be cited, so every run writes
 * `<runtime>-<device>-<dtype>-<phase>.json` plus a Markdown row. The JSON is the evidence; the
 * Markdown is a convenience for reviewing a diff, and both come from the same object so they
 * cannot disagree.
 */
import { mkdir, readdir, readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";

import type { BenchmarkReport } from "./schema.js";

/** File name a report is stored under, derived from the configuration it measured. */
export function reportFileName(report: BenchmarkReport): string {
  const device = report.environment.device.replaceAll(/[^a-z0-9-]/giu, "");
  const dtype = report.environment.dtype.replaceAll(/[^a-z0-9-]/giu, "");
  return `${report.runtime}-${device}-${dtype}-${report.phase}.json`;
}

/** Writes `report` into `directory` and refreshes the Markdown summary beside it. */
export async function writeReport(report: BenchmarkReport, directory: string): Promise<string> {
  await mkdir(directory, { recursive: true });
  const path = join(directory, reportFileName(report));
  await writeFile(path, `${JSON.stringify(report, null, 2)}\n`, "utf8");
  await writeFile(join(directory, "summary.md"), await renderSummary(directory), "utf8");
  return path;
}

/**
 * Renders every report in `directory` as one table.
 *
 * Reports are read back from disk rather than kept in memory so the table also picks up runs from
 * a different invocation — a cold and a warm phase are separate processes by design.
 */
export async function renderSummary(directory: string): Promise<string> {
  const files = (await readdir(directory))
    // Only benchmark reports: the quality harness writes its own JSON beside them, and it has no
    // `runs` array. Filtering by the name the benchmark writer produces keeps the table honest
    // instead of guessing from the contents.
    .filter((file) => /^(node|chromium)-.*\.json$/u.test(file))
    .sort();
  const reports: BenchmarkReport[] = [];
  for (const file of files) {
    reports.push(JSON.parse(await readFile(join(directory, file), "utf8")) as BenchmarkReport);
  }
  return renderMarkdown(reports);
}

/**
 * Renders the Markdown table.
 *
 * Two properties matter more than formatting: an unavailable configuration keeps its row (a gap
 * must be visible, not absent), and every row states the runtime, device, dtype, and phase, because
 * a latency without them is not a measurement of anything in particular.
 */
export function renderMarkdown(reports: readonly BenchmarkReport[]): string {
  const header = [
    "runtime",
    "device",
    "dtype",
    "phase",
    "download MB",
    "load first ms",
    "TTFA cold ms",
    "TTFA p50 ms",
    "TTFA p95 ms",
    "RTF p50",
    "RTF p95",
    "peak RSS MB",
    "cancel p50 ms",
    "runs",
  ];
  const rows = reports.map((report) => {
    if (report.status === "unavailable") {
      return [
        report.runtime,
        report.environment.device,
        report.environment.dtype,
        report.phase,
        "—",
        "—",
        "—",
        "—",
        "—",
        "—",
        "—",
        "—",
        "—",
        `unavailable (${report.unavailableReason ?? "no reason recorded"})`,
      ];
    }
    const first = report.runs[0];
    return [
      report.runtime,
      report.environment.device,
      report.environment.dtype,
      report.phase,
      format(report.download.megabytes, 1),
      format(report.summary.loadMs.p50, 0),
      format(first?.ttfaMs ?? Number.NaN, 0),
      format(report.summary.ttfaMs.p50, 0),
      format(report.summary.ttfaMs.p95, 0),
      format(report.summary.rtf.p50, 3),
      format(report.summary.rtf.p95, 3),
      format((report.summary.peakRssBytes?.max ?? Number.NaN) / 1e6, 1),
      format(report.cancellationMs?.p50 ?? Number.NaN, 1),
      String(report.runs.length),
    ];
  });

  const widths = header.map((cell, index) =>
    Math.max(cell.length, ...rows.map((row) => row[index]?.length ?? 0)),
  );
  const formatRow = (row: readonly string[]): string =>
    `| ${row.map((cell, index) => cell.padEnd(widths[index] ?? 0)).join(" | ")} |`;

  const lines = [
    formatRow(header),
    `| ${widths.map((width) => "-".repeat(width)).join(" | ")} |`,
    ...rows.map(formatRow),
    "",
    "TTFA = time to first audio chunk. RTF = stream wall time / audio duration (lower is faster).",
    '"load first ms" is the first `preload()` in the process (n = 1), because later streams reuse it.',
    "Times are milliseconds from the engine's own `time-to-first-chunk` event, not from a timer",
    "around the iterator. Peak RSS is the whole process, not a browser tab budget.",
  ];
  return `${lines.join("\n")}\n`;
}

/** Formats a number, or an em dash when the value was not measured. */
function format(value: number, decimals: number): string {
  return Number.isFinite(value) ? value.toFixed(decimals) : "—";
}
