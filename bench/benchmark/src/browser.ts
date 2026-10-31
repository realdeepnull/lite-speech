/**
 * Browser side of the benchmark: the same `measure()` as the Node.js runner, in Chromium.
 *
 * This module is built to `dist/browser.js` and loaded by `page.html` through an import map, so it
 * consumes the *built* ESM output of both packages — the same artifact a user installs. That is the
 * point of benchmarking in a browser at all: WASM and WebGPU differ from Node in ways no fake can
 * reproduce (thread pool, shader compilation, Cache Storage), and those differences are what the
 * report is supposed to capture.
 */
import { measure } from "./measure.js";
import type { BenchmarkReport } from "./schema.js";

/** What the Playwright driver asks the page to do. */
export interface BrowserRunRequest {
  /** `"wasm"` or `"webgpu"`; the engine resolves and validates the pair. */
  readonly device: "wasm" | "webgpu";
  /** `"auto"` keeps the engine's measured default for that device. */
  readonly dtype: string;
  /** Additional warm streams after the cold one. */
  readonly warmRuns: number;
  /** Whether the driver cleared Cache Storage before asking. */
  readonly phase: "cold" | "warm";
  /** Voice to synthesize with. */
  readonly voice?: string;
}

/** What the driver receives. */
export interface BrowserRunResult {
  readonly report: BenchmarkReport;
}

/** Describes the WebGPU adapter, which is the only honest way to say which GPU produced a number. */
async function describeGpu(): Promise<string | undefined> {
  const gpu = (navigator as unknown as { gpu?: { requestAdapter(): Promise<unknown | null> } }).gpu;
  if (gpu === undefined) {
    return undefined;
  }
  try {
    const adapter = (await gpu.requestAdapter()) as {
      info?: { description?: string; vendor?: string; architecture?: string };
    } | null;
    if (adapter === null) {
      return undefined;
    }
    const info = adapter.info;
    const parts = [info?.vendor, info?.architecture, info?.description].filter(
      (part): part is string => typeof part === "string" && part.length > 0,
    );
    return parts.length > 0 ? parts.join(" ") : "adapter without description";
  } catch {
    return undefined;
  }
}

/** Runs one configuration and returns its report. */
export async function run(request: BrowserRunRequest): Promise<BrowserRunResult> {
  // Resolved before `measure()`, because the environment provider runs synchronously and a
  // renderer's adapter description is part of the evidence: without it, "webgpu" would name a
  // backend but not the GPU that produced the numbers.
  const gpu = await describeGpu();
  const { createOnnxEngine } = await import("@lite-speech/engine-onnx");
  const engine = await createOnnxEngine({
    device: request.device,
    ...(request.dtype !== "auto" && { dtype: request.dtype as "fp32" | "q4" | "q4f16" }),
  });

  const report = await measure({
    engine,
    runtime: "chromium",
    device: request.device,
    dtype: engine.resolved.dtype,
    ...(request.voice !== undefined && { voice: request.voice }),
    warmRuns: request.warmRuns,
    phase: request.phase,
    environment: () => ({
      userAgent: navigator.userAgent,
      os: navigator.platform,
      cores: navigator.hardwareConcurrency,
      ...(gpu !== undefined && { gpu }),
    }),
    notes: [
      request.phase === "cold"
        ? "Cold phase: Cache Storage was cleared before the run, so the first stream downloaded the profile."
        : "Warm phase: Cache Storage was reused; sizes are what still had to be fetched.",
      "Memory is not reported: a Chromium renderer's heap has no comparable per-model figure, and `performance.memory` is not standardized.",
      request.device === "webgpu"
        ? "The device is the one the page resolved; a successful WebGPU load is not by itself proof that the operators ran on the GPU (M1 §5.1)."
        : "WASM is the compatibility baseline; the same profile may be faster on WebGPU on this host.",
    ],
  });

  return { report };
}
