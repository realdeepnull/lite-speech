/**
 * Engine tests through the *public* core API, in Node.
 *
 * The engine is only useful if `createTts()` can drive it, so these tests never reach into the
 * engine's internals: they build a `Tts` with an in-thread engine, then assert on the contract
 * the core enforces — ordering, `final` semantics, aborts, offline policy, and disposal.
 */
import { createTts, TTS_ERROR_CODES, isTtsError, type AudioChunk } from "@lite-speech/core";
import { describe, expect, it } from "vitest";

import { createOnnxEngine, type OnnxEngineOptions } from "../src/create-onnx-engine.js";
import type { FsProbe } from "../src/transformers-engine.js";
import { createFakeLoader } from "./helpers/fake-loader.js";

/** Builds an in-thread engine around a fake loader, the way Node would. */
async function createEngine(
  loaderOptions: Parameters<typeof createFakeLoader>[0] = {},
  engineOptions: Partial<OnnxEngineOptions> = {},
  fsProbe?: FsProbe,
): Promise<{
  engine: Awaited<ReturnType<typeof createOnnxEngine>>;
  loader: ReturnType<typeof createFakeLoader>;
}> {
  const loader = createFakeLoader(loaderOptions);
  const engine = await createOnnxEngine({
    device: "cpu",
    worker: false,
    inThreadLoader: loader.load,
    ...(fsProbe !== undefined && { fsProbe }),
    ...engineOptions,
  });
  return { engine, loader };
}

describe("engine through createTts()", () => {
  it("exposes exactly one verified language and its voices", async () => {
    const { engine } = await createEngine();
    expect(engine.supports({ language: "en" })).toBe(true);
    expect(engine.supports({ language: "de" })).toBe(false);

    const voices = engine.voices();
    expect(voices.length).toBeGreaterThan(0);
    // `default` first, so `createTts({ voice: undefined })` picks what upstream calls canonical.
    expect(voices[0]?.id).toBe("default");
    expect(voices.every((voice) => voice.language.startsWith("en"))).toBe(true);
  });

  it("synthesizes through the core and reports the resolved capabilities", async () => {
    const { engine } = await createEngine({ frames: 4, samplesPerFrame: 100 });
    const tts = createTts({ engine, offline: "allow" });

    const result = await tts.synthesize("Hello from the engine.");
    expect(result.engine).toBe("onnx");
    expect(result.channels).toBe(1);
    expect(result.sampleRate).toBe(24_000);
    expect(result.pcm).toBeInstanceOf(Float32Array);
    expect(result.pcm.length).toBe(4 * 100);
    expect(result.durationMs).toBeCloseTo((result.pcm.length / result.sampleRate) * 1000, 5);

    expect(tts.capabilities()).toMatchObject({
      engine: "onnx",
      device: "cpu",
      dtype: "fp32",
      streaming: "incremental-frames",
    });
    await tts.dispose();
  });

  it("streams ordered, gap-free chunks with exactly one final chunk", async () => {
    const { engine } = await createEngine({ frames: 5, samplesPerFrame: 32 });
    const tts = createTts({ engine, offline: "allow" });

    const chunks: AudioChunk[] = [];
    for await (const chunk of tts.stream("Streamed text.")) {
      chunks.push(chunk);
    }

    // Five audio frames plus the engine's single terminal `final` chunk.
    expect(chunks).toHaveLength(6);
    expect(chunks.map((chunk) => chunk.sequence)).toEqual([0, 1, 2, 3, 4, 5]);
    expect(chunks.filter((chunk) => chunk.final)).toHaveLength(1);
    expect(chunks.at(-1)?.final).toBe(true);
    expect(chunks.slice(0, -1).every((chunk) => !chunk.final)).toBe(true);
    expect(chunks.every((chunk) => chunk.sampleRate === 24_000)).toBe(true);
    expect(chunks.slice(0, -1).every((chunk) => chunk.pcm.length === 32)).toBe(true);
    // The terminal chunk carries no audio: it is a marker, not a frame.
    expect(chunks.at(-1)?.pcm.length).toBe(0);
    await tts.dispose();
  });

  it("streams a final chunk even for an utterance that produced no audio", async () => {
    const { engine } = await createEngine({ frames: 0 });
    const tts = createTts({ engine, offline: "allow" });

    const chunks: AudioChunk[] = [];
    for await (const chunk of tts.stream("Silence.")) {
      chunks.push(chunk);
    }
    expect(chunks).toHaveLength(1);
    expect(chunks[0]).toMatchObject({ sequence: 0, final: true, sampleRate: 24_000 });
    await tts.dispose();
  });

  it("reports TtsError(ABORTED) and leaves the worker reusable after an abort", async () => {
    const { engine, loader } = await createEngine({ frames: 200, frameDelayMs: 1 });
    const tts = createTts({ engine, offline: "allow" });

    const controller = new AbortController();
    const chunks: AudioChunk[] = [];
    const consume = (async (): Promise<void> => {
      for await (const chunk of tts.stream("Very long text.", { signal: controller.signal })) {
        chunks.push(chunk);
        if (chunks.length === 2) {
          controller.abort();
        }
      }
    })();

    await expect(consume).rejects.toSatisfy(
      (error: unknown) => isTtsError(error) && error.code === TTS_ERROR_CODES.ABORTED,
    );
    // Partial audio already delivered stays valid — it is never "un-delivered".
    expect(chunks.length).toBeGreaterThan(0);

    // The same engine still synthesizes: abort cancels the wait, not the worker.
    const after = await tts.synthesize("Short again.");
    expect(after.pcm.length).toBeGreaterThan(0);
    expect(loader.stats.loads).toBe(1);
    await tts.dispose();
  });

  it("does not start a second load when re-synthesizing", async () => {
    const { engine, loader } = await createEngine();
    const tts = createTts({ engine, offline: "allow" });
    await tts.synthesize("One.");
    await tts.synthesize("Two.");
    await tts.preload();
    expect(loader.stats.loads).toBe(1);
    await tts.dispose();
  });

  it("refuses strict offline before the engine is invoked", async () => {
    const { engine, loader } = await createEngine();
    const tts = createTts({ engine, offline: "strict" });

    await expect(tts.synthesize("Hello.")).rejects.toSatisfy(
      (error: unknown) =>
        isTtsError(error) && error.code === TTS_ERROR_CODES.OFFLINE_ASSETS_MISSING,
    );
    expect(loader.stats.loads).toBe(0);
    await tts.dispose();
  });

  it("describes assets with sizes from the manifest, before any download", async () => {
    let probed: readonly string[] = [];
    const probe: FsProbe = {
      cached: async (urls) => {
        probed = urls;
        return urls.map(() => ({ cached: false }));
      },
      remove: async () => undefined,
    };
    const { engine } = await createEngine({}, {}, probe);
    const tts = createTts({ engine, offline: "strict" });

    const plan = await tts.describeAssets();
    expect(plan.modelId).toBe("onnx-community/phonon-31m-en-ONNX");
    expect(plan.revision).toBe("4f0cda7c60dddb4862d0e68ff9ff60c32ffdc9f4");
    expect(plan.licenses).toEqual(["Apache-2.0"]);
    // `fp32` reports one graph per family, not every quantized variant in the manifest.
    expect(
      plan.files.filter((file) => file.path.startsWith("onnx/")).map((file) => file.path),
    ).toEqual(["onnx/flow_lm.onnx", "onnx/mimi_decoder.onnx"]);
    expect(plan.sizesComplete).toBe(true);
    // 4 952 + 303 009 + 144 + 120 089 751 + 41 267 821 + one 256 000-byte voice.
    expect(plan.totalBytes).toBe(4_952 + 303_009 + 144 + 120_089_751 + 41_267_821 + 256_000);
    expect(plan.complete).toBe(false);
    expect(probed.length).toBe(plan.files.length);
    await tts.dispose();
  });

  it("reports a cached plan as complete once every file exists", async () => {
    const probe: FsProbe = {
      cached: async (urls) => urls.map(() => ({ cached: true, size: 1 })),
      remove: async () => undefined,
    };
    const { engine } = await createEngine({}, {}, probe);
    const tts = createTts({ engine, offline: "strict" });
    await expect(tts.isCached()).resolves.toBe(true);
    await tts.dispose();
  });

  it("rejects a download budget before loading anything", async () => {
    const probe: FsProbe = {
      cached: async (urls) => urls.map(() => ({ cached: false })),
      remove: async () => undefined,
    };
    const { engine, loader } = await createEngine({}, {}, probe);
    const tts = createTts({ engine, offline: "strict" });

    await expect(tts.prepare({ allowNetwork: true, maxDownloadMB: 10 })).rejects.toSatisfy(
      (error: unknown) =>
        isTtsError(error) && error.code === TTS_ERROR_CODES.DOWNLOAD_BUDGET_EXCEEDED,
    );
    expect(loader.stats.loads).toBe(0);
    await tts.dispose();
  });

  it("rejects an unknown voice at creation time", async () => {
    const { engine } = await createEngine();
    expect(() => createTts({ engine, voice: "no-such-voice" })).toThrow(/does not provide voice/);
  });

  it("maps a load failure onto MODEL_LOAD_FAILED and keeps the message", async () => {
    const { engine } = await createEngine({ loadError: new Error("HTTP 500 from the Hub") });
    const tts = createTts({ engine, offline: "allow" });

    // The original error object cannot survive a worker boundary, so the contract is the code
    // and a message that carries the reason — not object identity.
    await expect(tts.synthesize("Hello.")).rejects.toSatisfy(
      (error: unknown) =>
        isTtsError(error) &&
        error.code === TTS_ERROR_CODES.MODEL_LOAD_FAILED &&
        error.message.includes("HTTP 500 from the Hub"),
    );
    await tts.dispose();
  });

  it("maps an offline-looking failure onto OFFLINE_ASSETS_MISSING", async () => {
    const { engine } = await createEngine({ loadError: new TypeError("Failed to fetch") });
    const tts = createTts({ engine, offline: "allow" });

    await expect(tts.synthesize("Hello.")).rejects.toSatisfy(
      (error: unknown) =>
        isTtsError(error) && error.code === TTS_ERROR_CODES.OFFLINE_ASSETS_MISSING,
    );
    await tts.dispose();
  });

  it("disposes idempotently and rejects later work", async () => {
    const { engine, loader } = await createEngine();
    const tts = createTts({ engine, offline: "allow" });
    await tts.synthesize("Hello.");

    await Promise.all([tts.dispose(), tts.dispose()]);
    await tts.dispose();
    expect(loader.stats.disposes).toBe(1);

    await expect(tts.synthesize("Again.")).rejects.toSatisfy(
      (error: unknown) => isTtsError(error) && error.code === TTS_ERROR_CODES.DISPOSED,
    );
  });

  it("removes assets through the filesystem probe", async () => {
    const removed: string[][] = [];
    const probe: FsProbe = {
      cached: async (urls) => urls.map(() => ({ cached: true })),
      remove: async (urls) => {
        removed.push([...urls]);
      },
    };
    const { engine } = await createEngine({}, {}, probe);
    const tts = createTts({ engine, offline: "allow" });
    await tts.synthesize("Hello.");
    await tts.removeModel();
    expect(removed).toHaveLength(1);
    expect(removed[0]?.some((url) => url.endsWith("onnx/flow_lm.onnx"))).toBe(true);
    await tts.dispose();
  });
});
