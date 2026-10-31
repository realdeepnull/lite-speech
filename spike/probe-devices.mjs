/**
 * Device/execution-provider probe.
 *
 * The M3 rule "explicit `webgpu` never falls back" must be grounded in what the runtime
 * reports, not in an assumption. This prints the backends the installed ONNX Runtime
 * exposes in the current runtime, and records what an explicit `webgpu` request does.
 *
 * Usage: node probe-devices.mjs
 */
import { env } from "@huggingface/transformers";

import { loadPhonon } from "./lib/phonon.mjs";
import { measureSynthesis } from "./lib/run.mjs";

const onnx = env.backends?.onnx;
const supported =
  typeof onnx?.getSupportedDevices === "function" ? onnx.getSupportedDevices() : null;

/** @type {Record<string, unknown>} */
const report = {
  runtime: "node",
  node: process.version,
  onnxVersions: onnx?.versions ?? null,
  getSupportedDevices: supported,
  hasNavigatorGpu: typeof navigator !== "undefined" && "gpu" in navigator,
};

// A successful explicit-webgpu load in Node does not prove a GPU was used: onnxruntime-node
// may satisfy the request with another provider. Record what happened instead of claiming
// which provider ran.
try {
  const model = await loadPhonon({ dtype: "q4f16", device: "webgpu" });
  const result = await measureSynthesis(model, "Hello from a device probe.");
  report.explicitWebgpu = {
    loadSucceeded: true,
    frames: result.frames,
    firstFrameMs: Number((result.firstFrameMs ?? 0).toFixed(0)),
    rtf: Number(result.rtf.toFixed(3)),
  };
  await model.dispose();
} catch (error) {
  report.explicitWebgpu = {
    loadSucceeded: false,
    name: /** @type {any} */ (error)?.name ?? null,
    message: String(/** @type {any} */ (error)?.message ?? error).slice(0, 300),
  };
}

console.log(JSON.stringify(report, null, 2));
