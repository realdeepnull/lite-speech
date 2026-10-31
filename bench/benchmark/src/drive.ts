/**
 * Playwright driver for the browser half of the benchmark.
 *
 * The browser is the only place WASM and WebGPU can be measured for real, but a browser is also
 * where a benchmark most easily lies: a renderer that is still compiling shaders when you sample
 * it, or a cache left over from a previous configuration, yields a plausible number that means
 * nothing. This driver therefore serves the repository over HTTP (a `file://` page cannot load a
 * module worker or a bare-specifier import map), clears the profile for a cold phase, waits for the
 * page's own promise instead of polling, and records *why* a configuration could not run instead of
 * failing the whole sweep.
 *
 * Usage:
 *   node --experimental-strip-types src/drive.ts [--device wasm|webgpu|both] [--dtype q4]
 *        [--runs 4] [--phase cold|warm] [--headed]
 */
import { createReadStream, existsSync, statSync } from "node:fs";
import { createServer } from "node:http";
import { rm } from "node:fs/promises";
import { extname, join, normalize, resolve } from "node:path";

import { chromium } from "playwright";

import type { BrowserRunRequest, BrowserRunResult } from "./browser.js";
import { writeReport } from "./report.js";
import { REPORT_SCHEMA_VERSION, type BenchmarkReport } from "./schema.js";

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
const runs = Number(args["runs"] ?? 4);
const dtype = String(args["dtype"] ?? "auto");
const phase = args["phase"] === "cold" ? "cold" : "warm";
const requested = String(args["device"] ?? "both");
const devices: readonly ("wasm" | "webgpu")[] =
  requested === "both" ? ["wasm", "webgpu"] : [requested === "webgpu" ? "webgpu" : "wasm"];

const here = import.meta.dirname;
const root = resolve(here, "..", "..", "..");
const reportDir = join(root, "bench", "report");
// One persistent profile per device, shared by both phases on purpose: `--phase cold` deletes it,
// `--phase warm` reuses it. A per-phase directory would make every warm run a second cold start and
// report a full download for a cache the previous run had already filled.
const profileRoot = join(root, "bench", ".cache", "chromium-profiles");

if (phase === "cold") {
  // A fresh profile means a fresh Cache Storage, so "cold" is a fact rather than a label.
  await rm(profileRoot, { recursive: true, force: true });
}

/** Extensions the page and the worker need; anything else is served as bytes. */
const CONTENT_TYPES = new Map([
  [".html", "text/html; charset=utf-8"],
  [".js", "text/javascript; charset=utf-8"],
  [".mjs", "text/javascript; charset=utf-8"],
  [".json", "application/json; charset=utf-8"],
  [".map", "application/json; charset=utf-8"],
  [".wasm", "application/wasm"],
  [".bin", "application/octet-stream"],
]);

/**
 * Serves the repository root on a *fixed* port.
 *
 * Fixed, not ephemeral: Cache Storage is keyed by origin, so a different port would be a different
 * origin and every "warm" run would find an empty cache. The number is arbitrary but has to be
 * stable across invocations, and `--port` exists for a host where it is taken.
 */
const preferredPort = Number(args["port"] ?? 8787);

/**
 * A 404 — never an `index.html` fallback — for an unknown path: a dev-server SPA fallback would
 * answer a missing worker script with HTML and status 200, and the worker would then fail with a
 * syntax error instead of the real reason.
 */
const server = createServer((request, response) => {
  const url = new URL(request.url ?? "/", `http://${request.headers.host ?? "localhost"}`);
  let path = normalize(join(root, decodeURIComponent(url.pathname)));
  if (!path.startsWith(root)) {
    response.writeHead(403).end("Forbidden");
    return;
  }
  if (existsSync(path) && statSync(path).isDirectory()) {
    path = join(path, "index.html");
  }
  if (!existsSync(path)) {
    response.writeHead(404).end(`Not found: ${url.pathname}`);
    return;
  }
  response.writeHead(200, {
    "content-type": CONTENT_TYPES.get(extname(path)) ?? "application/octet-stream",
    "cache-control": "no-store",
    // Cross-origin isolation is what lets ONNX Runtime Web use a WASM thread pool. Without it the
    // runtime silently runs single-threaded, which is a large, invisible difference in throughput —
    // so the harness opts in and the docs state the requirement.
    "cross-origin-opener-policy": "same-origin",
    "cross-origin-embedder-policy": "require-corp",
    "cross-origin-resource-policy": "same-origin",
  });
  createReadStream(path).pipe(response);
});

await new Promise<void>((resolveListening, rejectListening) => {
  server.once("error", rejectListening);
  server.listen(preferredPort, "127.0.0.1", () => resolveListening());
});
const pageUrl = `http://127.0.0.1:${String(preferredPort)}/bench/benchmark/page.html`;

const browser = await chromium.launch({
  headless: args["headed"] !== true,
  // Headless Chromium on a Windows host exposes `navigator.gpu` but no adapter at all — measured:
  // `requestAdapter()` returns `null` for every flag combination tried (Vulkan, ANGLE, SwiftShader,
  // `--ignore-gpu-blocklist`), while the same build launched headed reports `amd rdna-3` with
  // `shader-f16`. This instance is only used for the page-level `userAgent` fallback; the measured
  // runs use a persistent context below, for the same reason.
  args: ["--enable-unsafe-webgpu"],
});

/** A report for a configuration this host could not run, so the gap stays visible. */
function unavailable(device: string, reason: string, userAgent: string): BenchmarkReport {
  const empty = {
    n: 0,
    min: Number.NaN,
    p50: Number.NaN,
    p95: Number.NaN,
    max: Number.NaN,
    mean: Number.NaN,
  };
  return {
    schemaVersion: REPORT_SCHEMA_VERSION,
    date: new Date().toISOString(),
    runtime: "chromium",
    phase,
    status: "unavailable",
    unavailableReason: reason,
    environment: { device, dtype, userAgent },
    model: {
      id: "onnx-community/phonon-31m-en-ONNX",
      revision: "4f0cda7c60dddb4862d0e68ff9ff60c32ffdc9f4",
    },
    text: { id: "bench-standard-v1", characters: 0, audioSeconds: 0 },
    download: { bytes: 0, files: 0, megabytes: 0, licenses: [] },
    runs: [],
    summary: { loadMs: empty, ttfaMs: empty, rtf: empty, streamMs: empty },
    notes: ["No numbers were produced for this configuration; see unavailableReason."],
  };
}

try {
  for (const device of devices) {
    // A *persistent* context, not `newContext()`: Cache Storage belongs to the profile, so an
    // ephemeral context would make every "warm" run a cold one and report a full download for a
    // cache the previous run had already filled. The library's own test suite warms a profile for
    // the same reason.
    const context = await chromium.launchPersistentContext(
      join(profileRoot, `${device}-${dtype}`),
      {
        headless: args["headed"] !== true,
        args: ["--enable-unsafe-webgpu"],
      },
    );
    const page = await context.newPage();
    const request: BrowserRunRequest = { device, dtype, warmRuns: runs, phase };

    try {
      await page.goto(pageUrl, { waitUntil: "load" });
      const result = (await page.evaluate(
        (payload) =>
          (
            window as unknown as {
              benchRun: (value: BrowserRunRequest) => Promise<BrowserRunResult>;
            }
          ).benchRun(payload),
        request,
      )) as BrowserRunResult;
      await writeReport(result.report, reportDir);
      console.log(`${device}: ${JSON.stringify(result.report.summary)}`);
    } catch (error) {
      const userAgent = await page
        .evaluate(() => navigator.userAgent)
        .catch((): string => "unknown");
      const report = unavailable(
        device,
        error instanceof Error ? error.message : String(error),
        userAgent,
      );
      await writeReport(report, reportDir);
      console.error(`${device}: unavailable — ${report.unavailableReason ?? "unknown"}`);
    } finally {
      await context.close();
    }
  }
} finally {
  await browser.close();
  server.close();
}
