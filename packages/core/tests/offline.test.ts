import { describe, expect, it } from "vitest";

import { TTS_ERROR_CODES, createTts } from "../src/index";
import { createFakeEngine } from "./helpers/fake-engine";

describe("offline policy", () => {
  it("strict: rejects with OFFLINE_ASSETS_MISSING before the engine synthesizes", async () => {
    const engine = createFakeEngine({
      cached: false,
      assetFiles: [
        { path: "onnx/model_q4.onnx", size: 200, cached: false },
        { path: "vocab.json", size: 100, cached: true },
      ],
    });
    const tts = createTts({ engine, offline: "strict" });

    const error = await tts.synthesize("Hello").catch((value: unknown) => value);

    expect(error).toMatchObject({ code: TTS_ERROR_CODES.OFFLINE_ASSETS_MISSING });
    expect(error).toMatchObject({
      details: { missing: ["onnx/model_q4.onnx"], missingBytes: 200, modelId: "fake/model" },
    });
    expect(engine.stats.synthesize).toBe(0);
    expect(engine.stats.load).toBe(0);
  });

  it("strict: synthesizes when every asset is cached", async () => {
    const engine = createFakeEngine({ cached: true });
    const tts = createTts({ engine, offline: "strict" });

    await expect(tts.synthesize("Hello")).resolves.toMatchObject({ engine: "fake" });
    expect(engine.stats.load).toBe(1);
    expect(engine.stats.synthesize).toBe(1);
  });

  it("strict: fails the stream iteration when assets are missing", async () => {
    const engine = createFakeEngine({ cached: false });
    const tts = createTts({ engine, offline: "strict" });
    const chunks: unknown[] = [];

    await expect(async () => {
      for await (const chunk of tts.stream("Hello")) {
        chunks.push(chunk);
      }
    }).rejects.toMatchObject({ code: TTS_ERROR_CODES.OFFLINE_ASSETS_MISSING });

    expect(chunks).toEqual([]);
    expect(engine.stats.stream).toBe(0);
  });

  it("allow: synthesizes even when nothing is cached", async () => {
    const engine = createFakeEngine({ cached: false });
    const tts = createTts({ engine, offline: "allow" });

    await expect(tts.synthesize("Hello")).resolves.toBeDefined();
    expect(engine.stats.load).toBe(1);
  });

  it("strict: uses engine.isCached() when the engine cannot enumerate assets", async () => {
    const engine = createFakeEngine({ withDescribeAssets: false, cached: false });
    const tts = createTts({ engine, offline: "strict" });

    await expect(tts.synthesize("Hello")).rejects.toMatchObject({
      code: TTS_ERROR_CODES.OFFLINE_ASSETS_MISSING,
    });
    expect(engine.stats.isCached).toBeGreaterThan(0);
    expect(engine.stats.synthesize).toBe(0);
  });

  it("strict: preload() cannot be used to bypass the policy", async () => {
    const engine = createFakeEngine({ cached: false });
    const tts = createTts({ engine, offline: "strict" });

    await expect(tts.preload()).rejects.toMatchObject({
      code: TTS_ERROR_CODES.OFFLINE_ASSETS_MISSING,
    });
    // The engine was never asked to load, so nothing could have been downloaded.
    expect(engine.stats.load).toBe(0);
    expect(tts.isReady()).toBe(false);
  });

  it("strict: preload() succeeds when everything is cached", async () => {
    const engine = createFakeEngine({ cached: true });
    const tts = createTts({ engine, offline: "strict" });

    await tts.preload();
    expect(tts.isReady()).toBe(true);
    expect(engine.stats.load).toBe(1);
  });

  it("allow: preload() loads without a cache check", async () => {
    const engine = createFakeEngine({ cached: false });
    const tts = createTts({ engine, offline: "allow" });

    await tts.preload();
    expect(engine.stats.load).toBe(1);
    expect(engine.stats.describeAssets).toBe(0);
  });
});

describe("Tts.prepare()", () => {
  it("rejects a budget overrun before any download", async () => {
    const engine = createFakeEngine({
      cached: false,
      assetFiles: [{ path: "onnx/model_q4.onnx", size: 2_000_000, cached: false }],
      withPrepare: true,
    });
    const tts = createTts({ engine });

    const error = await tts
      .prepare({ allowNetwork: true, maxDownloadMB: 1 })
      .catch((value: unknown) => value);

    expect(error).toMatchObject({ code: TTS_ERROR_CODES.DOWNLOAD_BUDGET_EXCEEDED });
    expect(error).toMatchObject({
      details: { requiredBytes: 2_000_000, budgetBytes: 1_000_000 },
    });
    expect(engine.stats.prepare).toBe(0);
  });

  it("rejects a budget check it cannot prove (unknown sizes)", async () => {
    const engine = createFakeEngine({
      cached: false,
      assetFiles: [{ path: "onnx/model_q4.onnx", cached: false }],
      withPrepare: true,
    });
    const tts = createTts({ engine });

    await expect(tts.prepare({ allowNetwork: true, maxDownloadMB: 10 })).rejects.toMatchObject({
      code: TTS_ERROR_CODES.DOWNLOAD_BUDGET_EXCEEDED,
      details: { unknownSizes: ["onnx/model_q4.onnx"] },
    });
    expect(engine.stats.prepare).toBe(0);
  });

  it("does nothing without allowNetwork when every asset is cached", async () => {
    const engine = createFakeEngine({ cached: true, withPrepare: true });
    const tts = createTts({ engine });
    const plan = await tts.prepare();

    expect(plan.complete).toBe(true);
    expect(engine.stats.prepare).toBe(0);
  });

  it("rejects OFFLINE_ASSETS_MISSING without allowNetwork when assets are missing", async () => {
    const engine = createFakeEngine({ cached: false, withPrepare: true });
    const tts = createTts({ engine });

    await expect(tts.prepare()).rejects.toMatchObject({
      code: TTS_ERROR_CODES.OFFLINE_ASSETS_MISSING,
    });
    expect(engine.stats.prepare).toBe(0);
  });

  it("delegates to engine.prepare() with allowNetwork", async () => {
    const engine = createFakeEngine({ cached: false, withPrepare: true });
    const tts = createTts({ engine });

    await expect(tts.prepare({ allowNetwork: true })).resolves.toMatchObject({
      complete: false,
    });
    expect(engine.stats.prepare).toBe(1);
  });

  it("honours an abort signal", async () => {
    const engine = createFakeEngine({ cached: false, withPrepare: true });
    const tts = createTts({ engine });
    const controller = new AbortController();
    controller.abort();

    await expect(
      tts.prepare({ allowNetwork: true, signal: controller.signal }),
    ).rejects.toMatchObject({ code: TTS_ERROR_CODES.ABORTED });
    expect(engine.stats.prepare).toBe(0);
  });

  it("returns the plan when the engine has no prepare()", async () => {
    const engine = createFakeEngine({ cached: false, withPrepare: false });
    const tts = createTts({ engine });

    await expect(tts.prepare({ allowNetwork: true })).resolves.toMatchObject({
      complete: false,
    });
  });
});

describe("asset inspection", () => {
  it("reports the download estimate through the public API", async () => {
    const engine = createFakeEngine({
      cached: false,
      assetFiles: [
        { path: "a.onnx", size: 1_500_000, cached: false },
        { path: "b.json", size: 500_000, cached: true },
      ],
    });
    const tts = createTts({ engine });
    const estimate = await tts.estimateDownload();

    expect(estimate.bytes).toBe(1_500_000);
    expect(estimate.totalBytes).toBe(2_000_000);
    expect(estimate.megabytes).toBeCloseTo(1.5, 5);
    expect(estimate.files.map((file) => file.path)).toEqual(["a.onnx"]);
  });

  it("throws NOT_SUPPORTED when the engine cannot describe assets", async () => {
    const engine = createFakeEngine({ withDescribeAssets: false });
    const tts = createTts({ engine });

    await expect(tts.describeAssets()).rejects.toMatchObject({
      code: TTS_ERROR_CODES.NOT_SUPPORTED,
    });
    await expect(tts.estimateDownload()).rejects.toMatchObject({
      code: TTS_ERROR_CODES.NOT_SUPPORTED,
    });
  });

  it("throws NOT_SUPPORTED when the engine cannot remove a model", async () => {
    const engine = createFakeEngine({ withRemoveModel: false });
    const tts = createTts({ engine, offline: "allow" });
    await tts.preload();

    await expect(tts.removeModel()).rejects.toMatchObject({ code: TTS_ERROR_CODES.NOT_SUPPORTED });
  });

  it("resets the loaded state when removeModel() succeeds", async () => {
    const engine = createFakeEngine({ withRemoveModel: true, loadDelayMs: 0 });
    const tts = createTts({ engine, offline: "allow" });
    await tts.preload();
    expect(tts.isReady()).toBe(true);

    await tts.removeModel();

    expect(ts(engine).removeModel).toBe(1);
    expect(tts.isReady()).toBe(false);
    await tts.preload();
    expect(engine.stats.load).toBe(2);
  });
});

function ts(engine: ReturnType<typeof createFakeEngine>) {
  return engine.stats;
}
