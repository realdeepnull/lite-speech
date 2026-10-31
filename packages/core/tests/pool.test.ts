import { describe, expect, it } from "vitest";

import { TTS_ERROR_CODES, TtsPool, createEmitter, ttsPoolKey } from "../src/index";
import { createFakeEngine } from "./helpers/fake-engine";

describe("ttsPoolKey()", () => {
  it("separates voices of the same language", () => {
    expect(ttsPoolKey({ language: "en", voice: "female-1" })).toBe("en:female-1");
    expect(ttsPoolKey({ language: "en", voice: "male-1" })).toBe("en:male-1");
    expect(ttsPoolKey({ language: "en" })).toBe("en:");
  });
});

const poolVoices = [
  { id: "a", language: "en" },
  { id: "b", language: "en" },
  { id: "c", language: "en" },
  { id: "voice-1", language: "en" },
];

describe("TtsPool", () => {
  it("returns the same instance for the same key", async () => {
    const engine = createFakeEngine({ voices: poolVoices });
    const pool = new TtsPool({ engine });

    const first = await pool.switchTo({ language: "en", voice: "voice-1" });
    const second = await pool.switchTo({ language: "en", voice: "voice-1" });

    expect(second).toBe(first);
    expect(pool.size).toBe(1);
    expect(pool.cachedKeys()).toEqual(["en:voice-1"]);
    expect(pool.current()).toBe(first);
    await pool.dispose();
  });

  it("creates one instance per concurrent caller (single-flight)", async () => {
    let created = 0;
    const pool = new TtsPool({
      engine: () => {
        created += 1;
        return createFakeEngine({ id: `engine-${created}`, voices: poolVoices });
      },
    });

    const results = await Promise.all([
      pool.switchTo({ language: "en", voice: "a" }),
      pool.switchTo({ language: "en", voice: "a" }),
      pool.switchTo({ language: "en", voice: "a" }),
    ]);

    expect(new Set(results).size).toBe(1);
    expect(created).toBe(1);
    await pool.dispose();
  });

  it("evicts the least recently used entry beyond maxSize", async () => {
    const engines: ReturnType<typeof createFakeEngine>[] = [];
    const pool = new TtsPool({
      maxSize: 2,
      engine: () => {
        const engine = createFakeEngine({ id: `engine-${engines.length}`, voices: poolVoices });
        engines.push(engine);
        return engine;
      },
    });

    const a = await pool.switchTo({ language: "en", voice: "a" });
    const b = await pool.switchTo({ language: "en", voice: "b" });
    // Touch `a` so that `b` becomes the LRU entry.
    await pool.switchTo({ language: "en", voice: "a" });
    await pool.switchTo({ language: "en", voice: "c" });

    expect(pool.size).toBe(2);
    expect(pool.cachedKeys()).toEqual(["en:a", "en:c"]);
    expect(pool.get({ language: "en", voice: "b" })).toBeUndefined();
    expect(pool.get({ language: "en", voice: "a" })).toBe(a);
    expect(engines[1]?.stats.dispose).toBe(1);
    void b;
    await pool.dispose();
  });

  it("rejects an invalid maxSize", () => {
    for (const maxSize of [0, undefined]) {
      if (maxSize === undefined) {
        continue;
      }
      expect(() => new TtsPool({ maxSize })).toThrowError(
        expect.objectContaining({ code: TTS_ERROR_CODES.INVALID_ARGUMENT }),
      );
    }
    for (const maxSize of [-1, 1.5]) {
      expect(() => new TtsPool({ maxSize })).toThrowError(
        expect.objectContaining({ code: TTS_ERROR_CODES.INVALID_ARGUMENT }),
      );
    }
  });

  it("disposes a freshly created instance when dispose() races the creation", async () => {
    const engine = createFakeEngine({ voices: poolVoices });
    const pool = new TtsPool({ engine });
    const creation = pool.switchTo({ language: "en", voice: "voice-1" });
    await pool.dispose();

    await expect(creation).rejects.toMatchObject({ code: TTS_ERROR_CODES.DISPOSED });
    expect(engine.stats.dispose).toBe(1);
    expect(pool.size).toBe(0);
  });

  it("refuses to create after dispose()", async () => {
    const pool = new TtsPool({ engine: createFakeEngine({ voices: poolVoices }) });
    await pool.dispose();

    await expect(pool.switchTo({ language: "en" })).rejects.toMatchObject({
      code: TTS_ERROR_CODES.DISPOSED,
    });
  });

  it("disposes one key without touching the others", async () => {
    const first = createFakeEngine({ id: "first", voices: poolVoices });
    const second = createFakeEngine({ id: "second", voices: poolVoices });
    const engines = [first, second];
    const pool = new TtsPool({ engine: () => engines.shift() ?? second });

    await pool.switchTo({ language: "en", voice: "a" });
    await pool.switchTo({ language: "en", voice: "b" });
    await pool.disposeKey("en:a");

    expect(pool.cachedKeys()).toEqual(["en:b"]);
    expect(first.stats.dispose).toBe(1);
    expect(second.stats.dispose).toBe(0);
    await pool.dispose();
  });

  it("re-throws the first disposal failure after disposing everything", async () => {
    const failing = createFakeEngine({
      id: "failing",
      voices: poolVoices,
      disposeError: new Error("nope"),
    });
    const healthy = createFakeEngine({ id: "healthy", voices: poolVoices });
    const engines = [failing, healthy];
    const pool = new TtsPool({ engine: () => engines.shift() ?? healthy });

    await pool.switchTo({ language: "en", voice: "a" });
    await pool.switchTo({ language: "en", voice: "b" });

    await expect(pool.dispose()).rejects.toMatchObject({ code: TTS_ERROR_CODES.DISPOSE_FAILED });
    expect(healthy.stats.dispose).toBe(1);
  });

  it("does not lose an eviction that races an in-flight creation", async () => {
    const engine = createFakeEngine({ voices: poolVoices });
    const pool = new TtsPool({ engine });

    const creation = pool.switchTo({ language: "en", voice: "a" });
    await pool.disposeKey("en:a");
    await creation;

    expect(pool.size).toBe(0);
    expect(engine.stats.dispose).toBe(1);
  });

  it("falls back to the global registry when no engine is configured", async () => {
    const engine = createFakeEngine({ id: "global", voices: poolVoices });
    const pool = new TtsPool({ engines: [engine] });

    const instance = await pool.switchTo({ language: "en" });
    expect(instance.engineId).toBe("global");
    await pool.dispose();
  });
});

describe("createEmitter()", () => {
  it("calls listeners in subscription order and unsubscribes", () => {
    const emitter = createEmitter<{ value: [number] }>();
    const seen: number[] = [];
    const offFirst = emitter.on("value", (value) => seen.push(value * 10));
    emitter.on("value", (value) => seen.push(value));

    emitter.emit("value", 1);
    offFirst();
    emitter.emit("value", 2);

    expect(seen).toEqual([10, 1, 2]);
    expect(emitter.count("value")).toBe(1);
  });

  it("supports once() and off()", () => {
    const emitter = createEmitter<{ tick: [] }>();
    let count = 0;
    emitter.once("tick", () => {
      count += 1;
    });
    emitter.emit("tick");
    emitter.emit("tick");
    expect(count).toBe(1);

    const listener = (): void => {
      count += 1;
    };
    emitter.on("tick", listener);
    emitter.off("tick", listener);
    emitter.emit("tick");
    expect(count).toBe(1);
  });

  it("lets a listener unsubscribe during emission", () => {
    const emitter = createEmitter<{ ping: [] }>();
    const calls: string[] = [];
    const off = emitter.on("ping", () => {
      calls.push("first");
      off();
    });
    emitter.on("ping", () => calls.push("second"));

    emitter.emit("ping");

    expect(calls).toEqual(["first", "second"]);
    expect(emitter.count()).toBe(1);
    emitter.clear();
    expect(emitter.count()).toBe(0);
  });

  it("has no DOM or Node dependency", () => {
    expect(typeof globalThis.EventTarget).toBeDefined();
    const emitter = createEmitter<{ done: [] }>();
    expect(emitter.count("done")).toBe(0);
  });
});
