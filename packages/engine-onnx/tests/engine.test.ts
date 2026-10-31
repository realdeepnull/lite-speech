/**
 * Engine tests through the *public* core API, in Node.
 *
 * The engine is only useful if `createTts()` can drive it, so these tests never reach into the
 * engine's internals: they build a `Tts` with an in-thread engine, then assert on the contract
 * the core enforces — ordering, `final` semantics, aborts, offline policy, and disposal.
 */
import {
  createTts,
  TTS_ERROR_CODES,
  isTtsError,
  type AudioChunk,
  type DebugEvent,
} from "@lite-speech/core";
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

  it("emits device-fallback before device-resolved when a policy fell back", async () => {
    // `auto` under Node: no `navigator.gpu`, so the policy wants WebGPU and the runtime uses CPU.
    // That is the real fallback the event exists for, not a hand-set field.
    const { engine } = await createEngine({}, { device: "auto" });
    expect(engine.resolved).toMatchObject({ device: "cpu", fallbackFrom: "webgpu" });

    const events: DebugEvent[] = [];
    const tts = createTts({
      engine,
      offline: "allow",
      onDebug: (event) => events.push(event),
    });

    await tts.preload();

    expect(events.find((event) => event.type === "device-fallback")).toEqual(
      expect.objectContaining({
        engine: "onnx",
        from: "webgpu",
        to: "cpu",
        reason: "no WebGPU adapter",
      }),
    );
    // Order matters: "wanted webgpu, used cpu" has to precede the device that actually ran.
    expect(events.findIndex((event) => event.type === "device-fallback")).toBeLessThan(
      events.findIndex((event) => event.type === "device-resolved"),
    );
    await tts.dispose();
  });

  it("emits no device-fallback when the requested device was used", async () => {
    const { engine } = await createEngine();
    const events: DebugEvent[] = [];
    const tts = createTts({
      engine,
      offline: "allow",
      onDebug: (event) => events.push(event),
    });

    await tts.preload();

    expect(events.map((event) => event.type)).toContain("device-resolved");
    expect(events.map((event) => event.type)).not.toContain("device-fallback");
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

  it("reports integrity as declared for the pinned profile", async () => {
    const probe: FsProbe = {
      cached: async (urls) => urls.map(() => ({ cached: false })),
      remove: async () => undefined,
    };
    const { engine } = await createEngine({}, {}, probe);
    const tts = createTts({ engine });

    const plan = await tts.describeAssets();
    expect(plan.integrity).toBe("declared");
    expect(plan.hashesComplete).toBe(true);
    expect(plan.unverified).toEqual([]);
    // Every file resolves to an absolute URL at the pinned revision, which is what makes an
    // eviction or a verification able to address a cache entry at all.
    expect(plan.files.every((file) => file.url?.includes("/resolve/4f0cda7c"))).toBe(true);
    expect(plan.files.every((file) => file.sha256?.length === 64)).toBe(true);
    await tts.dispose();
  });

  it("verifies cached digests through the filesystem probe", async () => {
    const digests = new Map<string, string>();
    const probe: FsProbe = {
      cached: async (urls) => urls.map(() => ({ cached: true, size: 256_000 })),
      remove: async () => undefined,
      // Anything not in the map reports a wrong digest, which is a real mismatch rather than an
      // unreadable file — the two must not be conflated.
      sha256: async (url) => digests.get(url) ?? "0".repeat(64),
    };
    const { engine } = await createEngine({}, {}, probe);
    const tts = createTts({ engine });

    const plan = await tts.describeAssets();
    const voice = plan.files.find((file) => file.role === "voice");
    expect(voice?.sha256).toBeDefined();
    digests.set(voice!.url!, voice!.sha256!);

    await expect(tts.verifyAssets()).rejects.toSatisfy(
      (error: unknown) =>
        isTtsError(error) &&
        error.code === TTS_ERROR_CODES.ASSET_INTEGRITY_FAILED &&
        (error.details?.["mismatched"] as readonly string[]).includes("onnx/flow_lm.onnx"),
    );

    // Give every file its declared digest: verification now passes without touching the network.
    for (const file of plan.files) {
      if (file.url !== undefined && file.sha256 !== undefined) {
        digests.set(file.url, file.sha256);
      }
    }
    await expect(tts.verifyAssets()).resolves.toMatchObject({ ok: true });
    await tts.dispose();
  });

  it("reports an unreadable cached file as a failed verification, not a pass", async () => {
    const probe: FsProbe = {
      cached: async (urls) => urls.map(() => ({ cached: true, size: 256_000 })),
      remove: async () => undefined,
      sha256: async () => undefined,
    };
    const { engine } = await createEngine({}, {}, probe);
    const tts = createTts({ engine });

    await expect(tts.verifyAssets()).rejects.toSatisfy(
      (error: unknown) =>
        isTtsError(error) &&
        error.code === TTS_ERROR_CODES.ASSET_INTEGRITY_FAILED &&
        (error.details?.["unreadable"] as readonly string[]).length > 0,
    );
    await tts.dispose();
  });

  it("rolls back a partial provisioning so the cache returns to its previous state", async () => {
    // The loader fails *after* writing the model files, the way a dropped connection does.
    const written = new Set<string>();
    const probe: FsProbe = {
      cached: async (urls) => urls.map((url) => ({ cached: written.has(url) })),
      remove: async (urls) => {
        for (const url of urls) {
          written.delete(url);
        }
      },
    };
    const loader = createFakeLoader({ loadError: new Error("connection reset") });
    const engine = await createOnnxEngine({
      device: "cpu",
      worker: false,
      fsProbe: probe,
      inThreadLoader: async (options, voice) => {
        // Pretend the download wrote half of the manifest before failing.
        written.add(
          "https://huggingface.co/onnx-community/phonon-31m-en-ONNX/resolve/4f0cda7c60dddb4862d0e68ff9ff60c32ffdc9f4/config.json",
        );
        return loader.load(options, voice);
      },
    });
    const tts = createTts({ engine, offline: "allow" });

    await expect(tts.prepare({ allowNetwork: true })).rejects.toSatisfy(
      (error: unknown) => isTtsError(error) && error.code === TTS_ERROR_CODES.MODEL_LOAD_FAILED,
    );

    // Nothing this attempt wrote survives: one provisioning attempt is one transaction.
    expect([...written]).toEqual([]);
    await tts.dispose();
  });

  it("keeps files that a previous, successful provisioning created", async () => {
    const written = new Set<string>([
      "https://huggingface.co/onnx-community/phonon-31m-en-ONNX/resolve/4f0cda7c60dddb4862d0e68ff9ff60c32ffdc9f4/config.json",
    ]);
    const probe: FsProbe = {
      cached: async (urls) => urls.map((url) => ({ cached: written.has(url) })),
      remove: async (urls) => {
        for (const url of urls) {
          written.delete(url);
        }
      },
    };
    const loader = createFakeLoader({ loadError: new Error("connection reset") });
    const engine = await createOnnxEngine({
      device: "cpu",
      worker: false,
      fsProbe: probe,
      inThreadLoader: loader.load,
    });
    const tts = createTts({ engine, offline: "allow" });

    await expect(tts.prepare({ allowNetwork: true })).rejects.toSatisfy(
      (error: unknown) => isTtsError(error) && error.code === TTS_ERROR_CODES.MODEL_LOAD_FAILED,
    );

    // The file from the earlier run is still there: rollback only removes what it added.
    expect(written.size).toBe(1);
    await tts.dispose();
  });

  it("passes the declared voice digest to the loader, so a corrupt voice is caught at download", async () => {
    const { engine, loader } = await createEngine();
    const tts = createTts({ engine, offline: "allow", voice: "default" });

    await tts.preload();

    // The pinned manifest's digest of `voices/default.bin` at the pinned revision.
    expect(loader.stats.lastVoiceSha256).toBe(
      "d9291d795d2b3b3ca5aaf0d61c44b378918c840824e7c0bc41cfb63e5b0c560b",
    );
    await tts.dispose();
  });

  it("reports a re-described plan after a successful provisioning", async () => {
    const written = new Set<string>();
    const probe: FsProbe = {
      cached: async (urls) => urls.map((url) => ({ cached: written.has(url) })),
      remove: async (urls) => {
        for (const url of urls) {
          written.delete(url);
        }
      },
    };
    const loader = createFakeLoader();
    const engine = await createOnnxEngine({
      device: "cpu",
      worker: false,
      fsProbe: probe,
      inThreadLoader: async (options, voice) => {
        const loaded = await loader.load(options, voice);
        const described = await engine.describeAssets({ language: "en", voice: "default" });
        for (const file of described.files) {
          if (file.url !== undefined) {
            written.add(file.url);
          }
        }
        return loaded;
      },
    });
    const tts = createTts({ engine, offline: "allow", voice: "default" });

    const plan = await tts.prepare({ allowNetwork: true });
    expect(plan.complete).toBe(true);
    expect(plan.cachedBytes).toBe(plan.totalBytes);
    await tts.dispose();
  });

  it("estimates the cold download to the byte, per dtype", async () => {
    // The M4 gate asks for `estimateDownload()` to match the observed byte count. The bytes here
    // are the pinned revision's own tree, so a manifest edit that forgets a file or transposes a
    // digit fails this test instead of surprising a user with a budget overrun.
    const expected: Readonly<Record<string, number>> = {
      // config + tokenizer + tokenizer_config + flow_lm + mimi_decoder + default voice
      fp32: 4_952 + 303_009 + 144 + 120_089_751 + 41_267_821 + 256_000,
      // the same, with the q4 flow graph split into a skeleton and one external data chunk
      q4: 4_952 + 303_009 + 144 + 84_132 + 26_073_088 + 41_267_821 + 256_000,
      // both graphs split, plus the two f16 external data chunks
      q4f16: 4_952 + 303_009 + 144 + 254_451 + 19_828_736 + 54_375 + 20_608_000 + 256_000,
    };

    for (const [dtype, bytes] of Object.entries(expected)) {
      const loader = createFakeLoader();
      const engine = await createOnnxEngine({
        device: "cpu",
        dtype: dtype as "fp32" | "q4" | "q4f16",
        worker: false,
        fsProbe: {
          cached: async (urls) => urls.map(() => ({ cached: false })),
          remove: async () => undefined,
        },
        inThreadLoader: loader.load,
      });
      const tts = createTts({ engine, offline: "allow", voice: "default" });

      const estimate = await tts.estimateDownload();
      expect(`${dtype}:${estimate.bytes}`).toBe(`${dtype}:${bytes}`);
      expect(estimate.bytes).toBe(estimate.totalBytes);
      expect(estimate.megabytes).toBeCloseTo(bytes / 1_000_000, 6);
      await tts.dispose();
    }
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
