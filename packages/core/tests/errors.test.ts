import { describe, expect, it } from "vitest";

import {
  TTS_ERROR_CODES,
  TtsError,
  formatTtsError,
  isAbortError,
  isTtsError,
  normalizeTtsError,
} from "../src/index";

/**
 * A `TtsError` raised by a *second* copy of this package.
 *
 * Bundling `@lite-speech/core` into an engine — or an application ending up with two copies of a
 * dependency — creates a distinct class object, so an `instanceof` guard would reject a real
 * `TtsError`. That is not cosmetic: an `ABORTED` from an engine would be re-wrapped as
 * `STREAM_FAILED` at the core boundary, breaking the documented cancellation contract.
 */
function foreignError(
  code: string,
  message: string,
): { name: string; code: string; message: string } & Error {
  const error = new Error(message) as { name: string; code: string; message: string } & Error;
  error.name = "TtsError";
  error.code = code;
  return error;
}

describe("isTtsError()", () => {
  it("recognizes a local TtsError", () => {
    expect(isTtsError(new TtsError(TTS_ERROR_CODES.ABORTED, "aborted"))).toBe(true);
  });

  it("recognizes a TtsError from another copy of the package", () => {
    expect(isTtsError(foreignError(TTS_ERROR_CODES.ABORTED, "aborted"))).toBe(true);
  });

  it("rejects a plain Error", () => {
    expect(isTtsError(new Error("nope"))).toBe(false);
  });

  it("rejects an object that only claims the name", () => {
    expect(isTtsError({ name: "TtsError" })).toBe(false);
    expect(isTtsError({ name: "TtsError", code: "ABORTED" })).toBe(false);
    expect(isTtsError({ name: "TtsError", code: 1, message: "x" })).toBe(false);
  });

  it("rejects non-objects", () => {
    expect(isTtsError(null)).toBe(false);
    expect(isTtsError(undefined)).toBe(false);
    expect(isTtsError("ABORTED")).toBe(false);
    expect(isTtsError(42)).toBe(false);
  });
});

describe("normalizeTtsError()", () => {
  it("passes a TtsError through unchanged, so cause chains are never flattened twice", () => {
    const original = new TtsError(TTS_ERROR_CODES.MODEL_NOT_AVAILABLE, "no model");
    expect(normalizeTtsError(original, TTS_ERROR_CODES.SYNTHESIS_FAILED, "wrapped")).toBe(original);
  });

  it("passes a foreign TtsError through unchanged", () => {
    const original = foreignError(TTS_ERROR_CODES.ABORTED, "aborted");
    expect(normalizeTtsError(original, TTS_ERROR_CODES.STREAM_FAILED, "wrapped")).toBe(original);
  });

  it("turns an abort error into ABORTED regardless of the requested code", () => {
    const abort = new Error("aborted");
    abort.name = "AbortError";
    const normalized = normalizeTtsError(
      abort,
      TTS_ERROR_CODES.SYNTHESIS_FAILED,
      "synthesis failed",
    );
    expect(normalized.code).toBe(TTS_ERROR_CODES.ABORTED);
    expect(normalized.cause).toBe(abort);
  });

  it("wraps anything else with the requested code and keeps the cause", () => {
    const cause = new Error("boom");
    const normalized = normalizeTtsError(cause, TTS_ERROR_CODES.MODEL_LOAD_FAILED, "load failed");
    expect(normalized.code).toBe(TTS_ERROR_CODES.MODEL_LOAD_FAILED);
    expect(normalized.message).toBe("load failed");
    expect(normalized.cause).toBe(cause);
  });
});

describe("isAbortError()", () => {
  it("recognizes DOM and Node.js abort shapes", () => {
    const dom = new Error("aborted");
    dom.name = "AbortError";
    expect(isAbortError(dom)).toBe(true);
    expect(isAbortError({ name: "Error", code: "ABORT_ERR" })).toBe(true);
    expect(isAbortError({ name: "Error", code: 20 })).toBe(true);
  });

  it("rejects everything else", () => {
    expect(isAbortError(new Error("aborted"))).toBe(false);
    expect(isAbortError(null)).toBe(false);
    expect(isAbortError("AbortError")).toBe(false);
  });
});

describe("formatTtsError()", () => {
  it("renders a TtsError as CODE: message", () => {
    expect(formatTtsError(new TtsError(TTS_ERROR_CODES.DISPOSED, "gone"))).toBe("DISPOSED: gone");
  });

  it("renders a foreign TtsError the same way", () => {
    expect(formatTtsError(foreignError(TTS_ERROR_CODES.ABORTED, "stopped"))).toBe(
      "ABORTED: stopped",
    );
  });

  it("falls back to the message or the string form", () => {
    expect(formatTtsError(new Error("plain"))).toBe("plain");
    expect(formatTtsError("text")).toBe("text");
    expect(formatTtsError(null)).toBe("null");
  });
});
