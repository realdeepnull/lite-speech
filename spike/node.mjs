import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { loadPhonon } from "./lib/phonon.mjs";
import { measureSynthesis } from "./lib/run.mjs";
import { encodeWav } from "./lib/wav.mjs";

const here = dirname(fileURLToPath(import.meta.url));

/**
 * Tiny `--key value` / `--flag` parser. Unknown flags are ignored, so the CLI stays
 * forgiving for one-off experiments.
 * @param {string[]} argv
 */
function parseArgs(argv) {
  /** @type {Record<string, string | boolean>} */
  const args = {};
  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (!arg.startsWith("--")) continue;
    const next = argv[i + 1];
    if (next === undefined || next.startsWith("--")) {
      args[arg.slice(2)] = true;
    } else {
      args[arg.slice(2)] = next;
      i++;
    }
  }
  return args;
}

const args = parseArgs(process.argv.slice(2));
const dtype = /** @type {"fp32" | "q4" | "q4f16"} */ (args.dtype ?? "fp32");
const device = /** @type {"cpu" | "wasm" | "webgpu" | "auto"} */ (args.device ?? "cpu");
const text =
  typeof args.text === "string"
    ? args.text
    : "Hello from lite-speech. This is the second sentence of the inference spike.";
const byteFallback = args["byte-fallback"] === true;
const cold = args.cache === "cold";
const label = typeof args.label === "string" ? `-${args.label}` : "";

const cacheDir = resolve(here, ".cache");
if (cold) {
  const { rmSync } = await import("node:fs");
  rmSync(cacheDir, { recursive: true, force: true });
}

/** @type {Map<string, { loaded: number, total: number }>} */
const downloads = new Map();

const loadStarted = performance.now();
const model = await loadPhonon({
  dtype,
  device,
  cacheDir,
  byteFallback,
  onProgress: (info) => {
    if (info.status === "progress" && info.total) {
      downloads.set(info.file, { loaded: info.loaded, total: info.total });
    }
  },
});
const loadMs = performance.now() - loadStarted;

const result = await measureSynthesis(model, text, {
  ...(args.temperature === undefined ? {} : { temperature: Number(args.temperature) }),
});

// A second run separates cold shader/JIT cost from steady-state throughput.
const repeat = Number(args.repeat ?? "2");
/** @type {any[]} */
const runs = [result];
for (let i = 1; i < repeat; i++) {
  runs.push(
    await measureSynthesis(model, text, {
      ...(args.temperature === undefined ? {} : { temperature: Number(args.temperature) }),
    }),
  );
}
const warm = runs[runs.length - 1];

const downloadBytes = [...downloads.values()].reduce((n, entry) => n + entry.total, 0);
const rss = process.memoryUsage().rss;

const report = {
  date: new Date().toISOString(),
  runtime: "node",
  node: process.version,
  dtype,
  device,
  modelId: model.config._name_or_path ?? "onnx-community/phonon-31m-en-ONNX",
  revision: model.revision,
  tokenizerMode: model.tokenizerMode,
  cold,
  loadMs,
  downloadBytes,
  downloadFiles: Object.fromEntries(downloads),
  firstFrameMs: result.firstFrameMs,
  warmFirstFrameMs: warm.firstFrameMs,
  frameIntervalMs: warm.frameIntervalMs,
  synthMs: result.synthMs,
  warmSynthMs: warm.synthMs,
  audioSeconds: result.audioSeconds,
  rtf: result.rtf,
  warmRtf: warm.rtf,
  frames: result.frames,
  frameSizes: result.frameSizes,
  runs: runs.map((run) => ({
    firstFrameMs: run.firstFrameMs,
    frameIntervalMs: run.frameIntervalMs,
    synthMs: run.synthMs,
    rtf: run.rtf,
  })),
  sampleRate: result.sampleRate,
  channels: result.channels,
  range: [result.min, result.max],
  rms: result.rms,
  rssBytes: rss,
};

mkdirSync(resolve(here, "out"), { recursive: true });
const stem = `${dtype}-${device}${byteFallback ? "-bytefallback" : ""}${label}`;
writeFileSync(resolve(here, "out", `${stem}.json`), JSON.stringify(report, null, 2) + "\n");
writeFileSync(resolve(here, "out", `${stem}.wav`), encodeWav(result.pcm, result.sampleRate));

console.log(JSON.stringify(report, null, 2));
await model.dispose();
