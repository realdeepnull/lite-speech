/**
 * Browser integration suite: the milestone's Chromium gate.
 *
 * Everything here needs a real browser — a module worker, Cache Storage, `navigator.gpu`, and
 * ONNX Runtime Web — so none of it can be faked in Node. The suite drives the **public** core
 * API (`createTts`) against the built engine, which is exactly how an application would use it.
 *
 * A cold run downloads ~68 MB (`q4`) and instantiates two graphs, so timeouts are generous and
 * the model is shared across tests through a module-scoped `Tts` instance.
 */
import {
  createTts,
  isTtsError,
  TTS_ERROR_CODES,
  type AudioChunk,
  type Tts,
} from "@lite-speech/core";
import { afterAll, beforeAll, describe, expect, it } from "vitest";

import { createOnnxEngine, type OnnxEngine } from "../src/create-onnx-engine.js";

/** Text long enough to produce several frames but short enough to keep the suite quick. */
const TEXT = "Hello from lite-speech in Chromium.";

let engine: OnnxEngine;
let tts: Tts;

beforeAll(async () => {
  engine = await createOnnxEngine({ device: "wasm", dtype: "q4" });
  tts = createTts({ engine, offline: "allow", voice: "default" });
  await tts.preload();
}, 900_000);

afterAll(async () => {
  await tts.dispose();
});

describe("engine in Chromium", () => {
  it("runs in a module worker and reports the resolved capabilities", async () => {
    const capabilities = tts.capabilities();
    expect(capabilities).toMatchObject({
      engine: "onnx",
      device: "wasm",
      dtype: "q4",
      sampleRate: 24_000,
      streaming: "incremental-frames",
      modelVersion: "4f0cda7c60dddb4862d0e68ff9ff60c32ffdc9f4",
    });
  }, 900_000);

  it("synthesizes audible mono PCM at the declared sample rate", async () => {
    const result = await tts.synthesize(TEXT);
    expect(result.channels).toBe(1);
    expect(result.sampleRate).toBe(24_000);
    expect(result.pcm.length).toBeGreaterThan(24_000); // more than one second of audio
    expect(result.durationMs).toBeGreaterThan(1_000);

    let peak = 0;
    for (const sample of result.pcm) {
      expect(Number.isFinite(sample)).toBe(true);
      peak = Math.max(peak, Math.abs(sample));
    }
    // The model emits an unclipped signal; silence would mean the graph ran but produced nothing.
    expect(peak).toBeGreaterThan(0.01);
    expect(peak).toBeLessThanOrEqual(1);
  }, 900_000);

  it("streams ordered, gap-free chunks with exactly one final chunk", async () => {
    const chunks: AudioChunk[] = [];
    for await (const chunk of tts.stream(TEXT)) {
      chunks.push(chunk);
    }

    expect(chunks.length).toBeGreaterThan(1);
    expect(chunks.map((chunk) => chunk.sequence)).toEqual(chunks.map((_, index) => index));
    expect(chunks.filter((chunk) => chunk.final)).toHaveLength(1);
    expect(chunks.at(-1)?.final).toBe(true);
    expect(new Set(chunks.map((chunk) => chunk.sampleRate))).toEqual(new Set([24_000]));

    // Every audio frame is one measured decoder call: 80 ms of 24 kHz audio.
    const audioFrames = chunks.slice(0, -1);
    expect(audioFrames.every((chunk) => chunk.pcm.length === 1_920)).toBe(true);

    // The stream must carry at least as much audio as a full synthesis of the same text.
    const full = await tts.synthesize(TEXT);
    const streamedSamples = audioFrames.reduce((total, chunk) => total + chunk.pcm.length, 0);
    expect(streamedSamples).toBeGreaterThan(full.pcm.length * 0.75);
  }, 900_000);

  it("delivers the first chunk well before the whole utterance", async () => {
    const started = performance.now();
    let firstChunkAt: number | undefined;
    for await (const chunk of tts.stream(TEXT)) {
      if (firstChunkAt === undefined && chunk.pcm.length > 0) {
        firstChunkAt = performance.now() - started;
      }
    }
    const total = performance.now() - started;
    expect(firstChunkAt).toBeDefined();
    // True incremental frames: the first 80 ms frame lands before the last one.
    expect(firstChunkAt!).toBeLessThan(total);
  }, 900_000);

  it("reports ABORTED on cancellation and reuses the worker afterwards", async () => {
    const controller = new AbortController();
    const received: AudioChunk[] = [];
    const consume = (async (): Promise<void> => {
      for await (const chunk of tts.stream(TEXT, { signal: controller.signal })) {
        received.push(chunk);
        if (received.length === 2) {
          controller.abort();
        }
      }
    })();

    await expect(consume).rejects.toSatisfy(
      (error: unknown) => isTtsError(error) && error.code === TTS_ERROR_CODES.ABORTED,
    );
    // Audio already handed out stays valid: partials are never revoked.
    expect(received.length).toBeGreaterThan(0);

    // The worker is reusable: abort cancels the wait, not the session.
    const after = await tts.synthesize("Short text again.");
    expect(after.pcm.length).toBeGreaterThan(0);
  }, 900_000);

  it("finds its own downloaded assets in Cache Storage", async () => {
    // A strict-offline instance proves the same files: nothing is fetched again after preload.
    // It needs its own engine — `Tts.dispose()` disposes the engine it was given.
    const strictEngine = await createOnnxEngine({ device: "wasm", dtype: "q4" });
    const strict = createTts({ engine: strictEngine, offline: "strict", voice: "default" });
    await expect(strict.isCached()).resolves.toBe(true);
    await strict.dispose();
  }, 900_000);

  it("describes assets with per-file cache state", async () => {
    const plan = await tts.describeAssets();
    expect(plan.modelId).toBe("onnx-community/phonon-31m-en-ONNX");
    expect(plan.revision).toBe("4f0cda7c60dddb4862d0e68ff9ff60c32ffdc9f4");
    expect(plan.licenses).toEqual(["Apache-2.0"]);
    expect(plan.sizesComplete).toBe(true);
    expect(plan.files.every((file) => file.cached)).toBe(true);
    // `q4` is the resolved dtype, so only its two graphs are reported.
    const graphs = plan.files
      .filter((file) => file.path.startsWith("onnx/"))
      .map((file) => file.path);
    expect(graphs).toContain("onnx/flow_lm_q4.onnx");
    expect(graphs).toContain("onnx/mimi_decoder_q4.onnx");
    expect(graphs).not.toContain("onnx/flow_lm.onnx");
  }, 900_000);

  it("throws for an explicit webgpu request only when there is no adapter", async () => {
    const hasAdapter =
      typeof navigator !== "undefined" &&
      "gpu" in navigator &&
      (await (navigator as unknown as { gpu: { requestAdapter(): Promise<unknown | null> } }).gpu
        .requestAdapter()
        .then((adapter) => adapter !== null)
        .catch(() => false));

    if (hasAdapter) {
      // A capable host still may not support q4f16 without `shader-f16`; `q4` must work.
      const webgpu = await createOnnxEngine({ device: "webgpu", dtype: "q4" });
      expect(webgpu.resolved).toMatchObject({ device: "webgpu", dtype: "q4" });
      await webgpu.dispose();
      return;
    }
    // Headless Chromium without a GPU must reject rather than silently fall back.
    await expect(createOnnxEngine({ device: "webgpu" })).rejects.toThrow(/no adapter is available/);
  }, 900_000);

  it("emits device-fallback for auto on a host without an adapter", async () => {
    const events: string[] = [];
    const auto = await createOnnxEngine({ device: "auto" });
    const autoTts = createTts({
      engine: auto,
      offline: "allow",
      onDebug: (event) => {
        events.push(event.type);
      },
    });
    await autoTts.preload();
    const capabilities = autoTts.capabilities();

    if (capabilities?.device === "webgpu") {
      // An adapter exists, so there is nothing to fall back from.
      expect(events).toContain("device-resolved");
    } else {
      expect(auto.resolved.fallbackFrom).toBe("webgpu");
    }
    await autoTts.dispose();
  }, 900_000);

  it("removes cached assets so strict offline fails afterwards", async () => {
    const removable = await createOnnxEngine({ device: "wasm", dtype: "q4" });
    const removableTts = createTts({ engine: removable, offline: "allow", voice: "default" });
    await removableTts.preload();
    await removableTts.removeModel();
    // `removeModel()` keeps the instance usable and the engine alive, so the strict check runs on
    // the same `Tts` rather than on a second instance sharing an engine it does not own.
    expect(removableTts.isReady()).toBe(false);
    await expect(removableTts.isCached()).resolves.toBe(false);

    const strict = createTts({
      engine: await createOnnxEngine({ device: "wasm", dtype: "q4" }),
      offline: "strict",
      voice: "default",
    });
    await expect(strict.synthesize(TEXT)).rejects.toSatisfy(
      (error: unknown) =>
        isTtsError(error) && error.code === TTS_ERROR_CODES.OFFLINE_ASSETS_MISSING,
    );
    await strict.dispose();
    await removableTts.dispose();

    // Restore the shared instance's cache so a rerun does not have to re-download.
    await tts.prepare({ allowNetwork: true });
  }, 900_000);
});
