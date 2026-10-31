import { describe, expect, it } from "vitest";

import {
  TTS_ERROR_CODES,
  TtsError,
  assertBudget,
  cacheState,
  createAssetPlan,
  estimateDownload,
  formatTtsError,
  isAbortError,
  isTtsError,
  missingAssets,
  normalizeTtsError,
} from "../src/index";

const file = (
  path: string,
  size: number | undefined,
  cached: boolean,
): { path: string; size?: number; cached: boolean; role: "model" } => ({
  path,
  role: "model",
  cached,
  ...(size !== undefined && { size }),
});

describe("createAssetPlan()", () => {
  it("derives totals, missing files, and completeness", () => {
    const plan = createAssetPlan({
      modelId: "org/model",
      revision: "abc",
      voice: "female-1",
      licenses: ["Apache-2.0", "Apache-2.0", "CC-BY-NC-4.0"],
      files: [
        file("onnx/model.onnx", 800, true),
        file("vocab.json", 200, false),
        file("voice.bin", 1000, false),
      ],
    });

    expect(plan.totalBytes).toBe(2000);
    expect(plan.cachedBytes).toBe(800);
    expect(plan.missingBytes).toBe(1200);
    expect(plan.complete).toBe(false);
    expect(plan.sizesComplete).toBe(true);
    expect(plan.missing).toEqual(["vocab.json", "voice.bin"]);
    expect(plan.licenses).toEqual(["Apache-2.0", "CC-BY-NC-4.0"]);
  });

  it("tracks unknown sizes without pretending they are zero bytes", () => {
    const plan = createAssetPlan({
      modelId: "org/model",
      revision: "abc",
      files: [file("a.onnx", 100, true), file("b.onnx", undefined, false)],
    });

    expect(plan.sizesComplete).toBe(false);
    expect(plan.unknownSizes).toEqual(["b.onnx"]);
    expect(plan.totalBytes).toBe(100);
  });

  it("rejects duplicate paths and negative sizes", () => {
    expect(() =>
      createAssetPlan({
        modelId: "m",
        revision: "r",
        files: [file("a", 1, true), file("a", 1, true)],
      }),
    ).toThrowError(expect.objectContaining({ code: TTS_ERROR_CODES.INVALID_ARGUMENT }));

    expect(() =>
      createAssetPlan({ modelId: "m", revision: "r", files: [file("a", -1, true)] }),
    ).toThrowError(expect.objectContaining({ code: TTS_ERROR_CODES.INVALID_ARGUMENT }));
  });
});

describe("asset helpers", () => {
  const plan = createAssetPlan({
    modelId: "org/model",
    revision: "abc",
    files: [file("cached.onnx", 300, true), file("missing.onnx", 700, false)],
  });

  it("lists only the missing assets", () => {
    expect(missingAssets(plan).map((entry) => entry.path)).toEqual(["missing.onnx"]);
  });

  it("reports the cache state of a path", () => {
    expect(cacheState(plan, "cached.onnx")).toBe("cached");
    expect(cacheState(plan, "missing.onnx")).toBe("missing");
    expect(cacheState(plan, "unknown.onnx")).toBe("unknown");
  });

  it("estimates the download in bytes and decimal megabytes", () => {
    const estimate = estimateDownload(plan);
    expect(estimate.bytes).toBe(700);
    expect(estimate.totalBytes).toBe(1000);
    expect(estimate.megabytes).toBeCloseTo(0.0007, 8);
  });
});

describe("assertBudget()", () => {
  const incomplete = createAssetPlan({
    modelId: "org/model",
    revision: "abc",
    files: [file("a.onnx", 2_000_000, false)],
  });

  it("passes when the budget covers the download", () => {
    expect(assertBudget(incomplete, 3).bytes).toBe(2_000_000);
  });

  it("rejects before any download when the budget is too small", () => {
    const error = (() => {
      try {
        assertBudget(incomplete, 1);
        return undefined;
      } catch (value) {
        return value;
      }
    })();

    expect(error).toBeInstanceOf(TtsError);
    expect(error).toMatchObject({
      code: TTS_ERROR_CODES.DOWNLOAD_BUDGET_EXCEEDED,
      details: {
        requiredBytes: 2_000_000,
        budgetBytes: 1_000_000,
        files: [{ path: "a.onnx", size: 2_000_000 }],
      },
    });
  });

  it("accepts an unlimited budget", () => {
    expect(assertBudget(incomplete, undefined).bytes).toBe(2_000_000);
  });

  it("accepts a complete plan even when a cached file has no size", () => {
    const complete = createAssetPlan({
      modelId: "m",
      revision: "r",
      files: [file("a.onnx", undefined, true)],
    });
    expect(assertBudget(complete, 0)).toMatchObject({ bytes: 0 });
  });

  it("rejects a budget it cannot prove", () => {
    const unknown = createAssetPlan({
      modelId: "m",
      revision: "r",
      files: [file("a.onnx", undefined, false)],
    });
    expect(() => assertBudget(unknown, 10)).toThrowError(
      expect.objectContaining({
        code: TTS_ERROR_CODES.DOWNLOAD_BUDGET_EXCEEDED,
        details: expect.objectContaining({ unknownSizes: ["a.onnx"] }),
      }),
    );
  });

  it("rejects an invalid budget value", () => {
    for (const value of [-1, Number.NaN, Number.POSITIVE_INFINITY]) {
      expect(() => assertBudget(incomplete, value)).toThrowError(
        expect.objectContaining({ code: TTS_ERROR_CODES.INVALID_ARGUMENT }),
      );
    }
  });
});

describe("errors", () => {
  it("keeps the cause chain and stable code", () => {
    const cause = new Error("socket closed");
    const error = new TtsError(TTS_ERROR_CODES.SYNTHESIS_FAILED, "Synthesis failed", { cause });

    expect(error.code).toBe("SYNTHESIS_FAILED");
    expect(error.cause).toBe(cause);
    expect(isTtsError(error)).toBe(true);
    expect(isTtsError(new Error("plain"))).toBe(false);
    expect(formatTtsError(error)).toBe("SYNTHESIS_FAILED: Synthesis failed");
    expect(formatTtsError("plain")).toBe("plain");
  });

  it("normalizes only once", () => {
    const inner = new TtsError(TTS_ERROR_CODES.ABORTED, "Aborted");
    expect(normalizeTtsError(inner, TTS_ERROR_CODES.SYNTHESIS_FAILED, "wrapped")).toBe(inner);
  });

  it("maps abort errors to ABORTED regardless of the caller's code", () => {
    const domAbort = normalizeTtsError(
      new DOMException("Aborted", "AbortError"),
      TTS_ERROR_CODES.SYNTHESIS_FAILED,
      "failed",
    );
    expect(domAbort.code).toBe(TTS_ERROR_CODES.ABORTED);

    const nodeAbort = Object.assign(new Error("The operation was aborted"), { code: "ABORT_ERR" });
    expect(isAbortError(nodeAbort)).toBe(true);
    expect(normalizeTtsError(nodeAbort, TTS_ERROR_CODES.MODEL_LOAD_FAILED, "failed").code).toBe(
      TTS_ERROR_CODES.ABORTED,
    );
  });

  it("wraps an unknown value with the requested code and cause", () => {
    const cause = new Error("plain");
    const error = normalizeTtsError(cause, TTS_ERROR_CODES.STREAM_FAILED, "Stream failed");

    expect(error.code).toBe(TTS_ERROR_CODES.STREAM_FAILED);
    expect(error.cause).toBe(cause);
    expect(error.message).toBe("Stream failed");
  });
});
