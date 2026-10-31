/**
 * Numerical check the M1 gate asks for: does a quantized graph sound like fp32?
 *
 * Compares WAV files produced at `--temperature 0` (the only setting where the model is
 * deterministic) and reports SNR + max absolute difference. Requires `--temperature 0`
 * runs, otherwise the samples differ by design and only duration/level are comparable.
 */
import { readdirSync, readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const outDir = resolve(here, "out");

/** @param {string} file */
function readWav(file) {
  const buffer = readFileSync(file);
  const channels = buffer.readUInt16LE(22);
  const sampleRate = buffer.readUInt32LE(24);
  const bits = buffer.readUInt16LE(34);
  const dataBytes = buffer.readUInt32LE(40);
  const samples = new Float32Array(dataBytes / 2);
  for (let i = 0; i < samples.length; i++) {
    samples[i] = buffer.readInt16LE(44 + i * 2) / 0x7fff;
  }
  return { samples, sampleRate, channels, bits };
}

const files = readdirSync(outDir).filter((file) => file.endsWith("-t0.wav"));
const reference = files.find((file) => file.startsWith("fp32-"));

if (!reference) {
  console.error("No fp32 temperature-0 WAV found. Run the sweep with --temperature 0 first.");
  process.exit(1);
}

const base = readWav(resolve(outDir, reference));
const rows = files
  .filter((file) => file !== reference)
  .sort()
  .map((file) => {
    const candidate = readWav(resolve(outDir, file));
    const length = Math.min(base.samples.length, candidate.samples.length);
    let signal = 0;
    let noise = 0;
    let maxDiff = 0;
    for (let i = 0; i < length; i++) {
      const a = base.samples[i];
      const b = candidate.samples[i];
      signal += a * a;
      const diff = a - b;
      noise += diff * diff;
      if (Math.abs(diff) > maxDiff) maxDiff = Math.abs(diff);
    }
    const snr = noise === 0 ? Infinity : 10 * Math.log10(signal / noise);
    return {
      file,
      sampleRate: candidate.sampleRate,
      channels: candidate.channels,
      bits: candidate.bits,
      samples: candidate.samples.length,
      lengthDelta: candidate.samples.length - base.samples.length,
      snrDb: snr,
      maxDiff,
    };
  });

console.log(
  JSON.stringify(
    {
      reference: { file: reference, sampleRate: base.sampleRate, samples: base.samples.length },
      comparisons: rows.map((row) => ({
        ...row,
        snrDb: Number.isFinite(row.snrDb) ? Number(row.snrDb.toFixed(2)) : "identical",
        maxDiff: Number(row.maxDiff.toExponential(3)),
      })),
    },
    null,
    2,
  ),
);
