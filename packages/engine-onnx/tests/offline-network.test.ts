/**
 * Strict-offline proof for the core gate.
 *
 * These tests assert the property the core actually owns: with `offline: "strict"`, a synthesis
 * whose assets are missing fails **before** the engine is invoked and before any transport is
 * touched. The engine's own loader is deliberately not exercised here — that path only runs once
 * the gate has already passed — so the recorded `fetch` below proves the core never reached a
 * transport, which is exactly the claim being tested.
 *
 * The complementary claim, "the engine's Hub lookups resolve to the pinned revision and find their
 * bytes in the cache", is proven in `tests/revision-pinning.test.ts` against Transformers.js' real
 * resolution code, and in `tests/engine.browser.test.ts` against real Cache Storage.
 */
import { createTts, isTtsError, TTS_ERROR_CODES } from "@lite-speech/core";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import { createOnnxEngine, type OnnxEngine } from "../src/create-onnx-engine.js";
import type { FsProbe } from "../src/transformers-engine.js";
import { createFakeLoader } from "./helpers/fake-loader.js";

/** Requests the runtime attempted while the network was "off". */
let requests: string[] = [];
const realFetch = globalThis.fetch;

/** A cache probe that reports every manifest file as present, i.e. a cold-but-provisioned machine. */
function completeProbe(overrides: Partial<FsProbe> = {}): FsProbe {
  return {
    // No `size`: measured sizes are a separate integrity signal, and this probe only claims
    // presence. A real filesystem probe reports the size it stat'ed.
    cached: async (urls) => urls.map(() => ({ cached: true })),
    remove: async () => undefined,
    ...overrides,
  };
}

/** An engine on the in-thread path, which is how Node.js runs inference. */
async function createOfflineEngine(probe: FsProbe): Promise<OnnxEngine> {
  const loader = createFakeLoader({ frames: 3, samplesPerFrame: 64 });
  return createOnnxEngine({
    device: "cpu",
    worker: false,
    fsProbe: probe,
    inThreadLoader: loader.load,
  });
}

beforeEach(() => {
  requests = [];
  globalThis.fetch = ((input: RequestInfo | URL): Promise<Response> => {
    requests.push(String(input instanceof Request ? input.url : input));
    return Promise.reject(new TypeError("Failed to fetch: the network is disabled"));
  }) as typeof fetch;
});

afterEach(() => {
  globalThis.fetch = realFetch;
});

describe("strict offline with the network blocked", () => {
  it("synthesizes a full utterance without a single request", async () => {
    const tts = createTts({
      engine: await createOfflineEngine(completeProbe()),
      offline: "strict",
      voice: "default",
    });

    const result = await tts.synthesize("Nothing here touches the network.");

    expect(result.pcm.length).toBeGreaterThan(0);
    expect(result.sampleRate).toBe(24_000);
    expect(requests).toEqual([]);
    await tts.dispose();
  });

  it("streams without a single request", async () => {
    const tts = createTts({
      engine: await createOfflineEngine(completeProbe()),
      offline: "strict",
      voice: "default",
    });

    const chunks = [];
    for await (const chunk of tts.stream("Streamed while offline.")) {
      chunks.push(chunk);
    }

    expect(chunks.length).toBeGreaterThan(1);
    expect(chunks.at(-1)?.final).toBe(true);
    expect(requests).toEqual([]);
    await tts.dispose();
  });

  it("preloads without a single request", async () => {
    const tts = createTts({
      engine: await createOfflineEngine(completeProbe()),
      offline: "strict",
      voice: "default",
    });

    await tts.preload();

    expect(tts.isReady()).toBe(true);
    expect(requests).toEqual([]);
    await tts.dispose();
  });

  it("verifies cached digests without a single request", async () => {
    const engine = await createOfflineEngine(
      completeProbe({
        // A local hash source stands in for reading the bytes from disk: no network involved.
        sha256: async () => undefined,
      }),
    );
    const tts = createTts({ engine, offline: "strict", voice: "default" });

    // An unreadable file is a failure, never a silent pass — and it still makes no request.
    await expect(tts.verifyAssets()).rejects.toSatisfy(
      (error: unknown) =>
        isTtsError(error) && error.code === TTS_ERROR_CODES.ASSET_INTEGRITY_FAILED,
    );
    expect(requests).toEqual([]);
    await tts.dispose();
  });

  it("reports OFFLINE_ASSETS_MISSING for a missing asset, again without requesting it", async () => {
    const tts = createTts({
      engine: await createOfflineEngine(
        completeProbe({
          cached: async (urls) =>
            urls.map((url) => ({ cached: !url.endsWith("onnx/flow_lm.onnx") })),
        }),
      ),
      offline: "strict",
    });

    await expect(tts.synthesize("This must not download.")).rejects.toSatisfy(
      (error: unknown) =>
        isTtsError(error) &&
        error.code === TTS_ERROR_CODES.OFFLINE_ASSETS_MISSING &&
        error.message.includes("onnx/flow_lm.onnx"),
    );
    // The point of the gate: a failed strict-offline synthesis never even asks the network.
    expect(requests).toEqual([]);
    await tts.dispose();
  });

  it("names the file in the error so the caller can act on it", async () => {
    const tts = createTts({
      engine: await createOfflineEngine(
        completeProbe({
          cached: async (urls) => urls.map((url) => ({ cached: !url.endsWith("tokenizer.json") })),
        }),
      ),
      offline: "strict",
    });

    const error = await tts.prepare().catch((value: unknown) => value);

    expect(error).toMatchObject({
      code: TTS_ERROR_CODES.OFFLINE_ASSETS_MISSING,
      details: { missing: ["tokenizer.json"] },
    });
    expect(requests).toEqual([]);
    await tts.dispose();
  });
});
