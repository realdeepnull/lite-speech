import { describe, expect, it, vi } from "vitest";

import {
  TtsError,
  TTS_ERROR_CODES,
  createTts,
  getDefaultEngines,
  isTtsError,
  registerDefaultEngine,
  clearDefaultEngines,
  type DebugEvent,
} from "../src/index";
import { createFakeEngine, createHangingEngine } from "./helpers/fake-engine";

describe("createTts()", () => {
  it("returns synchronously without loading or downloading anything", () => {
    const engine = createFakeEngine();
    const tts = createTts({ engine });

    expect(tts).toBeDefined();
    expect(tts.isReady()).toBe(false);
    expect(engine.stats.load).toBe(0);
    expect(engine.stats.describeAssets).toBe(0);
    expect(engine.stats.synthesize).toBe(0);
  });

  it('defaults offline to "strict" and resolves the first voice', () => {
    const engine = createFakeEngine({
      voices: [
        { id: "female-1", language: "en" },
        { id: "male-1", language: "en" },
      ],
    });
    const tts = createTts({ engine });

    expect(tts.offline).toBe("strict");
    expect(tts.request).toEqual({ language: "en", voice: "female-1" });
    expect(tts.engineId).toBe("fake");
  });

  it("prefers an explicit voice and language", () => {
    const engine = createFakeEngine({
      voices: [
        { id: "female-1", language: "en" },
        { id: "female-de", language: "de" },
      ],
      languages: ["en", "de"],
    });
    const tts = createTts({ engine, voice: "female-de" });

    expect(tts.request).toEqual({ language: "de", voice: "female-de" });
  });

  it("throws ENGINE_NOT_SUPPORTED when no engine is available", () => {
    clearDefaultEngines();

    expect(() => createTts({})).toThrowError(TtsError);
    expect(() => createTts({})).toThrowError(
      expect.objectContaining({ code: TTS_ERROR_CODES.ENGINE_NOT_SUPPORTED }),
    );
  });

  it("throws ENGINE_NOT_SUPPORTED for an object that is not an engine", () => {
    const incomplete = { id: "broken", supports: () => true };

    expect(() => createTts({ engine: incomplete as never })).toThrowError(
      expect.objectContaining({ code: TTS_ERROR_CODES.ENGINE_NOT_SUPPORTED }),
    );
  });

  it("throws LANGUAGE_NOT_SUPPORTED when the engine rejects the language", () => {
    const engine = createFakeEngine({ languages: ["de"] });

    expect(() => createTts({ engine, language: "en" })).toThrowError(
      expect.objectContaining({ code: TTS_ERROR_CODES.LANGUAGE_NOT_SUPPORTED }),
    );
  });

  it("throws VOICE_NOT_SUPPORTED when the engine does not list the voice", () => {
    const engine = createFakeEngine({ voices: [{ id: "female-1", language: "en" }] });

    expect(() => createTts({ engine, voice: "nope" })).toThrowError(
      expect.objectContaining({ code: TTS_ERROR_CODES.VOICE_NOT_SUPPORTED }),
    );
  });

  it("accepts any voice when the engine reports no voices", () => {
    const engine = createFakeEngine({ voices: [], languages: ["en"] });
    const tts = createTts({ engine, voice: "custom", language: "en" });

    expect(tts.request).toEqual({ language: "en", voice: "custom" });
  });

  it("throws INVALID_ARGUMENT for an unknown offline policy", () => {
    const engine = createFakeEngine();

    expect(() => createTts({ engine, offline: "sometimes" as never })).toThrowError(
      expect.objectContaining({ code: TTS_ERROR_CODES.INVALID_ARGUMENT }),
    );
  });

  it("uses the globally registered engine when none is passed", () => {
    clearDefaultEngines();
    const engine = createFakeEngine({ id: "registered" });
    registerDefaultEngine(engine);
    const tts = createTts({});

    expect(tts.engineId).toBe("registered");
    expect(getDefaultEngines()).toEqual([engine]);
    clearDefaultEngines();
    expect(getDefaultEngines()).toEqual([]);
  });
});

describe("Tts.preload()", () => {
  it("is single-flight and loads the engine exactly once", async () => {
    const engine = createFakeEngine({ loadDelayMs: 5 });
    const tts = createTts({ engine });

    await Promise.all([tts.preload(), tts.preload(), tts.preload()]);

    expect(engine.stats.load).toBe(1);
    expect(tts.isReady()).toBe(true);
  });

  it("does not memoize a rejected load, so the next call retries", async () => {
    const engine = createFakeEngine();
    let failures = 1;
    engine.load = async () => {
      engine.stats.load += 1;
      if (failures > 0) {
        failures -= 1;
        throw new Error("network blip");
      }
    };
    const tts = createTts({ engine });

    await expect(tts.preload()).rejects.toMatchObject({ code: TTS_ERROR_CODES.MODEL_LOAD_FAILED });
    expect(tts.isReady()).toBe(false);
    await expect(tts.preload()).resolves.toBeUndefined();
    expect(engine.stats.load).toBe(2);
    expect(tts.isReady()).toBe(true);
  });

  it("keeps the cause of an engine load failure", async () => {
    const cause = new Error("boom");
    const engine = createFakeEngine({ loadError: cause });
    const tts = createTts({ engine });

    const error = await tts.preload().catch((value: unknown) => value);

    expect(isTtsError(error)).toBe(true);
    expect((error as TtsError).code).toBe(TTS_ERROR_CODES.MODEL_LOAD_FAILED);
    expect((error as TtsError).cause).toBe(cause);
  });

  it("rejects with DISPOSED after dispose()", async () => {
    const tts = createTts({ engine: createFakeEngine() });
    await tts.dispose();

    await expect(tts.preload()).rejects.toMatchObject({ code: TTS_ERROR_CODES.DISPOSED });
  });
});

describe("Tts.synthesize()", () => {
  it("returns PCM with an explicit sample rate and duration", async () => {
    const engine = createFakeEngine({ chunks: 3, samplesPerChunk: 240 });
    const tts = createTts({ engine, offline: "allow" });

    const audio = await tts.synthesize("  Hello   world  ");

    expect(audio.pcm).toBeInstanceOf(Float32Array);
    expect(audio.sampleRate).toBe(24_000);
    expect(audio.channels).toBe(1);
    expect(audio.engine).toBe("fake");
    expect(audio.voice).toBe("voice-1");
    expect(audio.durationMs).toBeCloseTo((720 / 24_000) * 1000, 5);
    expect(engine.stats.lastSynthesizedText).toBe("Hello world");
  });

  it("resolves an empty result for whitespace-only input without loading", async () => {
    const engine = createFakeEngine();
    const tts = createTts({ engine });
    const audio = await tts.synthesize("   \n\t ");

    expect(audio.pcm.length).toBe(0);
    expect(audio.sampleRate).toBe(0);
    expect(engine.stats.load).toBe(0);
    expect(engine.stats.synthesize).toBe(0);
  });

  it("normalizes an engine failure into SYNTHESIS_FAILED", async () => {
    const cause = new Error("model exploded");
    const engine = createFakeEngine({ synthesizeError: cause });
    const tts = createTts({ engine, offline: "allow" });

    const error = await tts.synthesize("Hello").catch((value: unknown) => value);

    expect(isTtsError(error)).toBe(true);
    expect((error as TtsError).code).toBe(TTS_ERROR_CODES.SYNTHESIS_FAILED);
    expect((error as TtsError).cause).toBe(cause);
  });

  it("rejects with ABORTED when the signal is already aborted", async () => {
    const engine = createFakeEngine();
    const tts = createTts({ engine });
    const controller = new AbortController();
    controller.abort();

    await expect(tts.synthesize("Hello", { signal: controller.signal })).rejects.toMatchObject({
      code: TTS_ERROR_CODES.ABORTED,
    });
    expect(engine.stats.synthesize).toBe(0);
  });

  it("re-checks abort after the model finished loading", async () => {
    const engine = createFakeEngine({ loadDelayMs: 10 });
    const tts = createTts({ engine, offline: "allow" });
    const controller = new AbortController();
    const pending = tts.synthesize("Hello", { signal: controller.signal });
    setTimeout(() => controller.abort(), 1);

    await expect(pending).rejects.toMatchObject({ code: TTS_ERROR_CODES.ABORTED });
    expect(engine.stats.synthesize).toBe(0);
  });

  it("re-checks disposal after the model finished loading", async () => {
    const engine = createFakeEngine({ loadDelayMs: 10 });
    const tts = createTts({ engine, offline: "allow" });
    const pending = tts.synthesize("Hello");
    setTimeout(() => void tts.dispose(), 1);

    await expect(pending).rejects.toMatchObject({ code: TTS_ERROR_CODES.DISPOSED });
    expect(engine.stats.synthesize).toBe(0);
  });

  it("rejects a malformed engine result at the boundary", async () => {
    const engine = createFakeEngine();
    engine.synthesize = async () =>
      ({ pcm: "not audio", sampleRate: 24_000 }) as unknown as Awaited<
        ReturnType<typeof engine.synthesize>
      >;
    const tts = createTts({ engine, offline: "allow" });

    await expect(tts.synthesize("Hello")).rejects.toMatchObject({
      code: TTS_ERROR_CODES.SYNTHESIS_FAILED,
    });
  });

  it("emits debug events only when a callback is registered", async () => {
    const events: DebugEvent[] = [];
    const engine = createFakeEngine();
    const tts = createTts({
      engine,
      offline: "allow",
      onDebug: (event) => events.push(event),
    });

    await tts.synthesize("Hello");

    expect(events.map((event) => event.type)).toContain("load-start");
    expect(events.map((event) => event.type)).toContain("synthesize-start");
    expect(events.map((event) => event.type)).toContain("synthesize-done");
  });

  it("forwards onProgress to the engine", async () => {
    const engine = createFakeEngine();
    const onProgress = vi.fn();
    const tts = createTts({ engine, offline: "allow", onProgress });

    await tts.preload();

    expect(engine.stats.lastProgress).toBe(onProgress);
    expect(onProgress).toHaveBeenCalledWith({
      phase: "model-download",
      loaded: 1,
      total: 1,
      progress: 1,
    });
  });
});

describe("Tts.dispose()", () => {
  it("is idempotent and disposes the engine once", async () => {
    const engine = createFakeEngine();
    const tts = createTts({ engine });

    await Promise.all([tts.dispose(), tts.dispose(), tts.dispose()]);
    await tts.dispose();

    expect(engine.stats.dispose).toBe(1);
    expect(tts.isDisposed()).toBe(true);
  });

  it("interrupts a stream that is blocked inside the engine", async () => {
    const engine = createHangingEngine();
    const tts = createTts({ engine, offline: "allow" });
    const iterator = tts.stream("Hello")[Symbol.asyncIterator]();
    const pending = iterator.next();
    // Let the stream reach the engine before disposing.
    await new Promise((resolve) => setTimeout(resolve, 5));
    await tts.dispose();

    await expect(pending).rejects.toMatchObject({ code: TTS_ERROR_CODES.ABORTED });
  });

  it("lets a consumer abandon a stream that is blocked inside the engine", async () => {
    const engine = createHangingEngine();
    const tts = createTts({ engine, offline: "allow" });
    const iterator = tts.stream("Hello")[Symbol.asyncIterator]();
    const pending = iterator.next();
    await new Promise((resolve) => setTimeout(resolve, 5));

    // Breaking out while the engine is still generating must not wait for that generation: the
    // iterator reports "done" immediately and the abandoned pull settles as ABORTED.
    await expect(iterator.return?.()).resolves.toMatchObject({ done: true });
    await expect(pending).rejects.toMatchObject({ code: TTS_ERROR_CODES.ABORTED });
  });

  it("normalizes an engine dispose failure", async () => {
    const engine = createFakeEngine({ disposeError: new Error("close failed") });
    const tts = createTts({ engine });

    await expect(tts.dispose()).rejects.toMatchObject({ code: TTS_ERROR_CODES.DISPOSE_FAILED });
  });
});
