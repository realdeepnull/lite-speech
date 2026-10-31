/**
 * Node.js benchmark entry point.
 *
 * Runs the same `measure()` as the browser on `onnxruntime-node` (device `"cpu"`), which is the
 * only Node backend where inference actually happens. Memory is sampled from the process's own
 * resident set, because Node gives no per-model accounting and a browser's `performance.memory`
 * is not comparable anyway.
 *
 * Usage:
 *   node --experimental-strip-types src/node.ts --dtype q4 [--runs 4] [--phase warm]
 */
import { cpus } from "node:os";
import { rm } from "node:fs/promises";
import { join } from "node:path";

import { createNodeOnnxEngine } from "@lite-speech/engine-onnx/node";

import { measure, type MeasureOptions } from "./measure.js";
import { writeReport } from "./report.js";

/** Parsed `--flag value` arguments; unknown flags are ignored so one-off runs stay easy. */
function parseArgs(argv: readonly string[]): Record<string, string | true> {
  const args: Record<string, string | true> = {};
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === undefined || !arg.startsWith("--")) {
      continue;
    }
    const key = arg.slice(2);
    const next = argv[index + 1];
    if (next === undefined || next.startsWith("--")) {
      args[key] = true;
    } else {
      args[key] = next;
      index += 1;
    }
  }
  return args;
}

const args = parseArgs(process.argv.slice(2));
const dtype = String(args["dtype"] ?? "q4");
const runs = Number(args["runs"] ?? 4);
const phase = args["phase"] === "cold" ? "cold" : "warm";

/** Cache directory for this dtype, shared by the cold and warm phase on purpose. */
const cacheRoot = join(import.meta.dirname, "..", ".cache");
const cacheDir = join(cacheRoot, dtype);

if (phase === "cold") {
  // Empty cache means "cold" is a fact rather than a label: the first stream really fetches.
  // The warm phase reuses this directory, so its `download` figure is what was still missing.
  await rm(cacheDir, { recursive: true, force: true });
}

/** Current resident set in bytes, or `undefined` where the runtime cannot report one. */
function residentBytes(): number | undefined {
  return process.memoryUsage.rss();
}

const engine = await createNodeOnnxEngine({
  dtype: dtype as "fp32" | "q4" | "q4f16",
  cacheDir,
});

const options: MeasureOptions = {
  engine,
  runtime: "node",
  device: "cpu",
  dtype,
  warmRuns: runs,
  phase: phase === "cold" ? "cold" : "warm",
  peakMemory: () => residentBytes() ?? 0,
  environment: () => {
    const cpu = cpus()[0];
    return {
      node: process.version,
      os: `${process.platform} ${process.arch}`,
      arch: process.arch,
      cores: cpus().length,
      ...(cpu?.model !== undefined && { cpu: cpu.model.trim() }),
    };
  },
  notes: [
    phase === "cold"
      ? "Cold phase: the cache directory was deleted before the run, so the first stream downloaded the profile."
      : "Warm phase: the cache directory was reused, so `download` is what still had to be fetched. The model is still instantiated by this process, so `coldStartMs` is a cache-warm, process-cold start.",
    "Peak memory is the Node.js process resident set, not a browser tab budget or a per-model figure.",
    "Voice embeddings are read over the network on every Node.js process start: the engine caches them in Cache Storage, which Node does not have, so `download` cannot include them.",
  ],
};

const report = await measure(options);
await writeReport(report, join(import.meta.dirname, "..", "..", "report"));
console.log(JSON.stringify(report.summary, null, 2));
