/**
 * Node.js quality entry point.
 *
 * Runs the corpus once per dtype **in one process**, so the `fp32` reference and the quantized
 * candidate are compared as in-memory buffers. Writing the reference to disk instead would mean
 * committing tens of megabytes of float samples, which is neither reviewable nor necessary: the
 * comparison is only valid against a run from the same process at the same `temperature`.
 *
 * `--temperature 0` is the default, because that is the only setting where a sample-by-sample
 * comparison measures quantization rather than the model's sampler.
 *
 * Usage:
 *   node dist/quality.mjs                       # fp32, q4, q4f16 at temperature 0
 *   node dist/quality.mjs --dtypes fp32,q4     # a subset
 */
import { mkdir, writeFile } from "node:fs/promises";
import { join } from "node:path";

import { createTts } from "@lite-speech/core";
import { createNodeOnnxEngine } from "@lite-speech/engine-onnx/node";

import { renderListeningProtocol, renderQualityMarkdown, renderQualityRow } from "./report.js";
import { sweep, type ReferenceAudio } from "./sweep.js";

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
const dtypes = String(args["dtypes"] ?? "fp32,q4,q4f16")
  .split(",")
  .map((entry) => entry.trim())
  .filter((entry) => entry.length > 0) as ("fp32" | "q4" | "q4f16")[];
const temperature = Number(args["temperature"] ?? 0);
const reportDir = join(import.meta.dirname, "..", "..", "report");
const audioDir = join(reportDir, "audio");

await mkdir(audioDir, { recursive: true });

/** The fp32 run, kept for the comparisons the quantized runs need. */
let reference: ReferenceAudio[] | undefined;

for (const dtype of dtypes) {
  const engine = await createNodeOnnxEngine({
    dtype,
    temperature,
    cacheDir: join(import.meta.dirname, "..", ".cache", dtype),
  });
  try {
    const { report, reference: produced } = await sweep({
      tts: createTts({ engine, offline: "allow" }),
      runtime: "node",
      device: "cpu",
      dtype,
      temperature,
      modelId: "onnx-community/phonon-31m-en-ONNX",
      modelRevision: "4f0cda7c60dddb4862d0e68ff9ff60c32ffdc9f4",
      checkDeterminism: true,
      ...(reference !== undefined && { reference }),
      writeWav: (file, bytes) => writeFile(join(audioDir, file), bytes),
      notes: [
        "Audio files are written under bench/report/audio/ and are git-ignored; the JSON report is the evidence.",
        'The listening score stays "pending" until a human runs the protocol in listening.md.',
        dtype === "fp32"
          ? "This run is the reference every other dtype is compared against."
          : `\`versusReference\` compares each case with the fp32 run of the same process at temperature ${String(temperature)}.`,
      ],
    });
    await writeFile(
      join(reportDir, `quality-node-cpu-${dtype}.json`),
      `${JSON.stringify(report, null, 2)}\n`,
      "utf8",
    );
    if (dtype === "fp32") {
      reference = produced;
    }
    console.log(renderQualityRow(report));
  } finally {
    await engine.dispose();
  }
}

await writeFile(join(reportDir, "listening.md"), renderListeningProtocol(), "utf8");
await writeFile(join(reportDir, "quality.md"), await renderQualityMarkdown(reportDir), "utf8");
