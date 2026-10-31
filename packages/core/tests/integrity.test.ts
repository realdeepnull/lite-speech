import { describe, expect, it } from "vitest";

import {
  TTS_ERROR_CODES,
  assertIntegrity,
  createAssetPlan,
  createIntegrityReport,
  createTts,
} from "../src/index";
import { createFakeEngine } from "./helpers/fake-engine";

/** A 64-character lowercase digest, so a manifest fixture cannot accidentally be invalid. */
const DIGEST = "a".repeat(64);

describe("AssetPlan integrity state", () => {
  it("reports 'declared' when every file declares a digest", () => {
    const plan = createAssetPlan({
      modelId: "fake/model",
      revision: "0".repeat(40),
      files: [
        { path: "a.onnx", size: 10, sha256: DIGEST, role: "model", cached: true },
        { path: "b.json", size: 5, sha256: "b".repeat(64), role: "model", cached: false },
      ],
    });

    expect(plan.integrity).toBe("declared");
    expect(plan.hashesComplete).toBe(true);
    expect(plan.unverified).toEqual([]);
  });

  it("reports 'partial' and names the files it cannot prove", () => {
    const plan = createAssetPlan({
      modelId: "fake/model",
      revision: "0".repeat(40),
      files: [
        { path: "a.onnx", size: 10, sha256: DIGEST, role: "model", cached: true },
        { path: "b.json", size: 5, role: "model", cached: true },
      ],
    });

    expect(plan.integrity).toBe("partial");
    expect(plan.hashesComplete).toBe(false);
    expect(plan.unverified).toEqual(["b.json"]);
  });

  it("reports 'none' when no file declares a digest", () => {
    const plan = createAssetPlan({
      modelId: "fake/model",
      revision: "0".repeat(40),
      files: [{ path: "a.onnx", size: 10, role: "model", cached: true }],
    });
    expect(plan.integrity).toBe("none");
    expect(plan.unverified).toEqual(["a.onnx"]);
  });

  it("rejects a malformed digest instead of accepting an unverifiable claim", () => {
    expect(() =>
      createAssetPlan({
        modelId: "fake/model",
        revision: "0".repeat(40),
        files: [{ path: "a.onnx", size: 10, sha256: "ABC", role: "model", cached: true }],
      }),
    ).toThrow(/Invalid sha256/);
  });

  it("reports a size mismatch for a truncated cached file", () => {
    const plan = createAssetPlan({
      modelId: "fake/model",
      revision: "0".repeat(40),
      files: [
        { path: "a.onnx", size: 1000, role: "model", cached: true, cachedSize: 400 },
        { path: "b.json", size: 10, role: "model", cached: true, cachedSize: 10 },
      ],
    });

    expect(plan.sizeMismatches).toEqual(["a.onnx"]);
    // The declared size is what the budget must use, never the truncated measurement.
    expect(plan.totalBytes).toBe(1010);
    expect(plan.complete).toBe(true);
  });
});

describe("createIntegrityReport()", () => {
  it("is ok only when nothing mismatched and nothing was unreadable", () => {
    expect(createIntegrityReport({ verified: ["a"] }).ok).toBe(true);
    expect(createIntegrityReport({ mismatched: ["a"] }).ok).toBe(false);
    expect(createIntegrityReport({ unreadable: ["a"] }).ok).toBe(false);
    // Unverified is not a failure: presence without a declared digest is the documented state.
    expect(createIntegrityReport({ unverified: ["a"] }).ok).toBe(true);
  });

  it("assertIntegrity() raises ASSET_INTEGRITY_FAILED with both failure lists", () => {
    const report = createIntegrityReport({ mismatched: ["a.onnx"], unreadable: ["b.bin"] });
    expect(() => assertIntegrity(report)).toThrow(
      expect.objectContaining({
        code: TTS_ERROR_CODES.ASSET_INTEGRITY_FAILED,
        details: { mismatched: ["a.onnx"], unreadable: ["b.bin"] },
      }),
    );
  });

  it("assertIntegrity() returns an ok report unchanged", () => {
    const report = createIntegrityReport({ verified: ["a.onnx"] });
    expect(assertIntegrity(report)).toBe(report);
  });
});

describe("Tts.verifyAssets()", () => {
  it("returns the engine's report when every digest matched", async () => {
    const engine = createFakeEngine({
      withVerifyAssets: true,
      integrity: { verified: ["a.onnx"] },
    });
    const tts = createTts({ engine });

    await expect(tts.verifyAssets()).resolves.toMatchObject({
      ok: true,
      verified: ["a.onnx"],
    });
    expect(engine.stats.verifyAssets).toBe(1);
  });

  it("rejects with ASSET_INTEGRITY_FAILED when a digest does not match", async () => {
    const engine = createFakeEngine({
      withVerifyAssets: true,
      integrity: { mismatched: ["onnx/model_q4.onnx"] },
    });
    const tts = createTts({ engine });

    await expect(tts.verifyAssets()).rejects.toMatchObject({
      code: TTS_ERROR_CODES.ASSET_INTEGRITY_FAILED,
      details: { mismatched: ["onnx/model_q4.onnx"] },
    });
  });

  it("rejects with NOT_SUPPORTED when the engine cannot verify", async () => {
    const engine = createFakeEngine({ withVerifyAssets: false });
    const tts = createTts({ engine });

    await expect(tts.verifyAssets()).rejects.toMatchObject({
      code: TTS_ERROR_CODES.NOT_SUPPORTED,
    });
  });

  it("normalizes a thrown verification error into ASSET_INTEGRITY_FAILED", async () => {
    const engine = createFakeEngine({ withVerifyAssets: true });
    engine.verifyAssets = async () => {
      throw new Error("cache is unreadable");
    };
    const tts = createTts({ engine });

    await expect(tts.verifyAssets()).rejects.toMatchObject({
      code: TTS_ERROR_CODES.ASSET_INTEGRITY_FAILED,
    });
  });
});

describe("Tts.prepare() verification", () => {
  it("verifies after a network provisioning when asked to", async () => {
    const engine = createFakeEngine({
      cached: false,
      withPrepare: true,
      withVerifyAssets: true,
      integrity: { verified: ["onnx/model_q4.onnx"] },
    });
    const tts = createTts({ engine });

    const plan = await tts.prepare({ allowNetwork: true, verify: true });

    expect(engine.stats.prepare).toBe(1);
    expect(engine.stats.verifyAssets).toBe(1);
    // The returned plan is re-described after provisioning, not the pre-download snapshot.
    expect(plan.modelId).toBe("fake/model");
    expect(engine.stats.describeAssets).toBeGreaterThanOrEqual(2);
  });

  it("does not verify unless asked", async () => {
    const engine = createFakeEngine({ cached: true, withPrepare: true, withVerifyAssets: true });
    const tts = createTts({ engine });

    await tts.prepare();
    expect(engine.stats.verifyAssets).toBe(0);
  });

  it("fails the provisioning when verification fails", async () => {
    const engine = createFakeEngine({
      cached: false,
      withPrepare: true,
      withVerifyAssets: true,
      integrity: { mismatched: ["onnx/model_q4.onnx"] },
    });
    const tts = createTts({ engine });

    await expect(tts.prepare({ allowNetwork: true, verify: true })).rejects.toMatchObject({
      code: TTS_ERROR_CODES.ASSET_INTEGRITY_FAILED,
    });
  });

  it("rejects verify: true when the engine cannot verify at all", async () => {
    const engine = createFakeEngine({ cached: true, withPrepare: true, withVerifyAssets: false });
    const tts = createTts({ engine });

    await expect(tts.prepare({ verify: true })).rejects.toMatchObject({
      code: TTS_ERROR_CODES.NOT_SUPPORTED,
    });
  });
});

describe("strict offline with a damaged cache", () => {
  it("rejects when a cached file has the wrong size", async () => {
    const engine = createFakeEngine({
      cached: true,
      assetFiles: [{ path: "onnx/model_q4.onnx", size: 2_000, cached: true, cachedSize: 12 }],
    });
    const tts = createTts({ engine, offline: "strict" });

    const error = await tts.synthesize("Hello").catch((value: unknown) => value);

    expect(error).toMatchObject({ code: TTS_ERROR_CODES.OFFLINE_ASSETS_MISSING });
    expect(error).toMatchObject({ details: { sizeMismatches: ["onnx/model_q4.onnx"] } });
    expect(engine.stats.synthesize).toBe(0);
  });

  it("names the mismatch in the message, not an empty missing list", async () => {
    const engine = createFakeEngine({
      cached: true,
      assetFiles: [{ path: "onnx/model_q4.onnx", size: 2_000, cached: true, cachedSize: 12 }],
    });
    const tts = createTts({ engine, offline: "strict" });

    await expect(tts.synthesize("Hello")).rejects.toThrow(/size mismatch/);
  });
});
