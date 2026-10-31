/**
 * Worker protocol helpers.
 *
 * The transfer list is the reason multi-megabyte PCM payloads do not get cloned, so it is worth
 * asserting directly rather than inferring it from a round trip.
 */
import { describe, expect, it } from "vitest";

import {
  isIntermediateResponse,
  transferListFor,
  type WorkerResponse,
} from "../src/worker-protocol.js";

describe("worker protocol", () => {
  it("transfers result and chunk buffers instead of cloning them", () => {
    const result: WorkerResponse = {
      kind: "result",
      id: 1,
      pcm: zeroed(8),
      sampleRate: 24_000,
      frames: 2,
      durationMs: 10,
    };
    const chunk: WorkerResponse = { kind: "chunk", id: 1, pcm: zeroed(4), sampleRate: 24_000 };
    expect(transferListFor(result)).toEqual([result.pcm]);
    expect(transferListFor(chunk)).toEqual([chunk.pcm]);
  });

  it("transfers nothing for control messages", () => {
    expect(transferListFor({ kind: "loaded", id: 1 })).toEqual([]);
    expect(transferListFor({ kind: "disposed", id: 1 })).toEqual([]);
    expect(
      transferListFor({
        kind: "progress",
        id: 1,
        event: { phase: "model-download", loaded: 1, total: 2, progress: 0.5 },
      }),
    ).toEqual([]);
  });

  it("treats telemetry as intermediate, because it settles nothing", () => {
    expect(
      isIntermediateResponse({
        kind: "progress",
        id: 1,
        event: { phase: "model-load", loaded: 0, total: 0, progress: Number.NaN },
      }),
    ).toBe(true);
    expect(
      isIntermediateResponse({
        kind: "capabilities",
        id: 1,
        device: "wasm",
        dtype: "fp32",
        modelId: "m",
        modelVersion: "r",
        sampleRate: 24_000,
        streaming: "incremental-frames",
      }),
    ).toBe(true);
    expect(isIntermediateResponse({ kind: "inference-start", id: 1, textLength: 5 })).toBe(true);
    expect(
      isIntermediateResponse({
        kind: "inference-done",
        id: 1,
        durationMs: 1,
        samples: 1,
        sampleRate: 24_000,
      }),
    ).toBe(true);
  });

  it("treats settling responses as terminal", () => {
    expect(isIntermediateResponse({ kind: "loaded", id: 1 })).toBe(false);
    expect(isIntermediateResponse({ kind: "stream-opened", id: 1 })).toBe(false);
    expect(isIntermediateResponse({ kind: "stream-end", id: 1 })).toBe(false);
    expect(isIntermediateResponse({ kind: "cancelled", id: 1 })).toBe(false);
    expect(isIntermediateResponse({ kind: "disposed", id: 1 })).toBe(false);
    expect(isIntermediateResponse({ kind: "error", id: 1, message: "x" })).toBe(false);
  });
});

/** A small ArrayBuffer of `samples` float32 values. */
function zeroed(samples: number): ArrayBuffer {
  return new Float32Array(samples).buffer as ArrayBuffer;
}
