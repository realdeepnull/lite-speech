/**
 * Turns `spike/out/*.json` into the Markdown table used in
 * `docs/spike/transformers-js-phonon.md`. Run it after the dtype sweep.
 */
import { readdirSync, readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const outDir = resolve(here, "out");

const mb = (bytes) => (bytes / 1e6).toFixed(1);
const num = (value, digits = 2) =>
  value === null || value === undefined ? "n/a" : Number(value).toFixed(digits);

/** @type {any[]} */
const reports = readdirSync(outDir)
  .filter((file) => file.endsWith(".json"))
  // `-t0` runs exist for the determinism check, not for the latency table.
  .filter((file) => !/-t0\w*\.json$/.test(file))
  .map((file) => JSON.parse(readFileSync(resolve(outDir, file), "utf8")))
  .filter((report) => report.runtime === "node" && report.frames)
  .sort((a, b) => a.dtype.localeCompare(b.dtype) || a.device.localeCompare(b.device));

if (reports.length === 0) {
  console.error("No reports in spike/out. Run `node node.mjs --dtype ...` first.");
  process.exit(1);
}

/** @param {number[]} values */
function median(values) {
  const sorted = values.filter((value) => Number.isFinite(value)).sort((a, b) => a - b);
  if (sorted.length === 0) return null;
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0 ? (sorted[middle - 1] + sorted[middle]) / 2 : sorted[middle];
}

const header = [
  "dtype",
  "device",
  "encoding",
  "download MB",
  "load ms",
  "first frame ms",
  "warm first frame ms",
  "RTF",
  "warm RTF",
  "audio s",
  "frames",
  "RSS MB",
  "range",
];
const rows = reports.map((report) => {
  const runs = report.runs ?? [];
  const warm = runs.length > 1 ? runs.slice(1) : runs;
  return [
    report.dtype,
    report.device,
    report.tokenizerMode,
    mb(report.downloadBytes),
    num(report.loadMs, 0),
    num(median(runs.map((/** @type {any} */ run) => run.firstFrameMs)) ?? report.firstFrameMs, 0),
    num(median(warm.map((/** @type {any} */ run) => run.firstFrameMs)), 0),
    num(median(runs.map((/** @type {any} */ run) => run.rtf)) ?? report.rtf, 3),
    num(median(warm.map((/** @type {any} */ run) => run.rtf)), 3),
    num(report.audioSeconds, 2),
    String(report.frames),
    mb(report.rssBytes),
    `${num(report.range[0], 3)} … ${num(report.range[1], 3)}`,
  ];
});

const widths = header.map((cell, index) =>
  Math.max(cell.length, ...rows.map((row) => row[index].length)),
);
const format = (row) => `| ${row.map((cell, index) => cell.padEnd(widths[index])).join(" | ")} |`;

console.log(format(header));
console.log(`| ${widths.map((width) => "-".repeat(width)).join(" | ")} |`);
for (const row of rows) console.log(format(row));

const environment = reports[0];
console.log(
  `\nMeasured ${environment.date} with Node ${environment.node}, model ${environment.modelId}@${environment.revision}.`,
);
console.log(
  "RTF = synthesis wall time / audio duration (lower is faster). First frame = time to the first 80 ms frame.",
);
