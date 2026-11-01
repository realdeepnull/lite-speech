/**
 * Model-host invariants, in Node, without ONNX or a browser.
 *
 * These are the rules the worker must keep no matter what the engine does: one model per host,
 * single-flight loads, one open stream, cancellation that is acknowledged rather than merely
 * ignored, and a disposal that always answers.
 */
import { describe, expect, it } from "vitest";

import { createModelHost, type ModelHost } from "../src/model-host.js";
import type { WorkerResponse } from "../src/worker-protocol.js";
import { createFakeLoader } from "./helpers/fake-loader.js";

/** Collects everything the host posts. */
function createRecorder(): {
  readonly responses: WorkerResponse[];
  readonly post: (response: WorkerResponse) => void;
  readonly ofKind: <K extends WorkerResponse["kind"]>(
    kind: K,
  ) => Extract<WorkerResponse, { kind: K }>[];
} {
  const responses: WorkerResponse[] = [];
  return {
    responses,
    post: (response) => {
      responses.push(response);
    },
    ofKind: <K extends WorkerResponse["kind"]>(kind: K) =>
      responses.filter(
        (response): response is Extract<WorkerResponse, { kind: K }> => response.kind === kind,
      ),
  };
}

function createHost(options: Parameters<typeof createFakeLoader>[0] = {}): {
  host: ModelHost;
  recorder: ReturnType<typeof createRecorder>;
  loader: ReturnType<typeof createFakeLoader>;
} {
  const recorder = createRecorder();
  const loader = createFakeLoader(options);
  const host = createModelHost({
    families: loader.families,
    post: recorder.post,
    now: () => 0,
  });
  return { host, recorder, loader };
}

const LOAD = {
  kind: "load" as const,
  id: 1,
  modelId: "onnx-community/phonon-31m-en-ONNX",
  revision: "4f0cda7c60dddb4862d0e68ff9ff60c32ffdc9f4",
  family: "phonon",
  device: "cpu",
  dtype: "fp32",
  language: "en",
  voice: "default",
};

describe("model host", () => {
  it("reports capabilities and a loaded acknowledgement", async () => {
    const { host, recorder } = createHost();
    await host.dispatch(LOAD);

    const capabilities = recorder.ofKind("capabilities");
    expect(capabilities).toHaveLength(1);
    expect(capabilities[0]).toMatchObject({
      device: "cpu",
      dtype: "fp32",
      sampleRate: 24_000,
      streaming: "incremental-frames",
      modelVersion: LOAD.revision,
    });
    expect(recorder.ofKind("loaded")).toHaveLength(1);
  });

  it("shares one load between concurrent callers", async () => {
    const { host, loader, recorder } = createHost();
    await Promise.all([host.dispatch(LOAD), host.dispatch({ ...LOAD, id: 2 })]);
    expect(loader.stats.loads).toBe(1);
    expect(recorder.ofKind("loaded")).toHaveLength(2);
  });

  it("rejects a second model in the same host", async () => {
    const { host, recorder } = createHost();
    await host.dispatch(LOAD);
    await host.dispatch({ ...LOAD, id: 2, revision: "different" });

    const errors = recorder.ofKind("error");
    expect(errors).toHaveLength(1);
    expect(errors[0]?.message).toMatch(/already hosts/);
  });

  it("rejects a dtype change without a dispose first", async () => {
    const { host, recorder } = createHost();
    await host.dispatch(LOAD);
    await host.dispatch({ ...LOAD, id: 2, dtype: "q4" });
    expect(recorder.ofKind("error")[0]?.message).toMatch(/dispose before changing device or dtype/);
  });

  it("does not memoize a failed load", async () => {
    const recorder = createRecorder();
    let attempts = 0;
    const host = createModelHost({
      families: {
        phonon: async (options) => {
          attempts += 1;
          if (attempts === 1) {
            throw new Error("network down");
          }
          const { families } = createFakeLoader();
          return families["phonon"]!(options);
        },
      },
      post: recorder.post,
    });
    await host.dispatch(LOAD);
    expect(recorder.ofKind("error")).toHaveLength(1);
    await host.dispatch({ ...LOAD, id: 2 });
    expect(attempts).toBe(2);
    expect(recorder.ofKind("loaded")).toHaveLength(1);
  });

  it("reports an unregistered family as an explicit error", async () => {
    const recorder = createRecorder();
    const host = createModelHost({ families: {}, post: recorder.post });
    await host.dispatch({ ...LOAD, family: "no-such-family" });
    const errors = recorder.ofKind("error");
    expect(errors).toHaveLength(1);
    expect(errors[0]?.message).toMatch(/No runner is registered for model family/);
  });

  it("synthesizes into one transferred result", async () => {
    const { host, recorder } = createHost({ frames: 4, samplesPerFrame: 10 });
    await host.dispatch(LOAD);
    await host.dispatch({ kind: "synthesize", id: 2, text: "Hello world." });

    const results = recorder.ofKind("result");
    expect(results).toHaveLength(1);
    expect(results[0]?.frames).toBe(4);
    expect(results[0]?.sampleRate).toBe(24_000);
    expect(results[0]?.pcm.byteLength).toBe(4 * 10 * 4);
    expect(recorder.ofKind("inference-start")).toHaveLength(1);
    expect(recorder.ofKind("inference-done")[0]?.samples).toBe(40);
  });

  it("streams one frame per pull and ends with stream-end", async () => {
    const { host, recorder } = createHost({ frames: 3, samplesPerFrame: 8 });
    await host.dispatch(LOAD);
    await host.dispatch({ kind: "stream-open", id: 2, text: "Hello." });

    for (let pull = 0; pull < 4; pull += 1) {
      await host.dispatch({ kind: "stream-next", id: 2 });
    }

    expect(recorder.ofKind("stream-opened")).toHaveLength(1);
    const chunks = recorder.ofKind("chunk");
    expect(chunks).toHaveLength(3);
    expect(chunks.every((chunk) => chunk.pcm.byteLength === 8 * 4)).toBe(true);
    expect(recorder.ofKind("stream-end")).toHaveLength(1);
    expect(recorder.ofKind("error")).toHaveLength(0);
  });

  it("stops streaming when the consumer closes the session", async () => {
    const { host, recorder } = createHost({ frames: 100, samplesPerFrame: 8 });
    await host.dispatch(LOAD);
    await host.dispatch({ kind: "stream-open", id: 2, text: "Long text." });
    await host.dispatch({ kind: "stream-next", id: 2 });
    await host.dispatch({ kind: "stream-close", id: 2 });
    await host.dispatch({ kind: "stream-next", id: 2 });

    expect(recorder.ofKind("chunk")).toHaveLength(1);
    // Pulling a closed session is reported, not silently answered.
    expect(recorder.ofKind("error")).toHaveLength(1);
  });

  it("acknowledges a cancellation without waiting for the next frame", async () => {
    const { host, recorder } = createHost({ frames: 100, samplesPerFrame: 8 });
    await host.dispatch(LOAD);
    await host.dispatch({ kind: "stream-open", id: 2, text: "Long text." });
    host.cancel({ kind: "cancel", id: 2 });

    const cancelled = recorder.ofKind("cancelled");
    expect(cancelled).toHaveLength(1);
    expect(cancelled[0]?.id).toBe(2);
  });

  it("cancels a full synthesis cooperatively", async () => {
    const { host, recorder } = createHost({ frames: 50, samplesPerFrame: 8 });
    await host.dispatch(LOAD);
    const synthesize = host.dispatch({ kind: "synthesize", id: 7, text: "Hello." });
    host.cancel({ kind: "cancel", id: 7 });
    await synthesize;

    expect(recorder.ofKind("cancelled").some((response) => response.id === 7)).toBe(true);
    expect(recorder.ofKind("result")).toHaveLength(0);
  });

  it("acknowledges a disposal even when releasing throws", async () => {
    const { host, recorder } = createHost({ disposeError: new Error("gpu lost") });
    await host.dispatch(LOAD);
    await host.dispatch({ kind: "dispose", id: 9 });
    expect(recorder.ofKind("disposed")).toHaveLength(1);
  });

  it("can load again after a dispose", async () => {
    const { host, recorder, loader } = createHost();
    await host.dispatch(LOAD);
    await host.dispatch({ kind: "dispose", id: 2 });
    await host.dispatch({ ...LOAD, id: 3 });
    expect(loader.stats.loads).toBe(2);
    expect(recorder.ofKind("loaded")).toHaveLength(2);
  });

  it("switches the voice by reloading the runner", async () => {
    const { host, recorder, loader } = createHost();
    await host.dispatch(LOAD);
    await host.dispatch({ ...LOAD, id: 2, voice: "LMDIYE0Oa377Qh6Z" });

    expect(loader.stats.voices).toEqual(["default", "LMDIYE0Oa377Qh6Z"]);
    expect(loader.stats.disposes).toBe(1);
    expect(recorder.ofKind("error")).toHaveLength(0);
  });

  it("reports a synthesis failure as an error response", async () => {
    const { host, recorder } = createHost({ synthesizeError: new Error("decoder exploded") });
    await host.dispatch(LOAD);
    await host.dispatch({ kind: "synthesize", id: 2, text: "Hello." });

    const errors = recorder.ofKind("error");
    expect(errors).toHaveLength(1);
    expect(errors[0]?.message).toBe("decoder exploded");
  });

  it("reports cache state from the injected cache", async () => {
    const recorder = createRecorder();
    const stored = new Set(["https://huggingface.co/m/r/resolve/rev/config.json"]);
    const host = createModelHost({
      families: createFakeLoader().families,
      post: recorder.post,
      openCache: async () => ({
        match: async (url: string) =>
          stored.has(url) ? { headers: { get: () => "4952" } } : undefined,
      }),
    });
    await host.dispatch({
      kind: "cache-check",
      id: 4,
      modelId: "m/r",
      revision: "rev",
      files: ["config.json", "tokenizer.json"],
    });

    const state = recorder.ofKind("cache-state")[0];
    expect(state?.cached).toEqual([true, false]);
    expect(state?.sizes).toEqual([4952, undefined]);
  });

  it("reports every file as unprovable without a cache", async () => {
    const { host, recorder } = createHost();
    await host.dispatch({
      kind: "cache-check",
      id: 4,
      modelId: "m/r",
      revision: "rev",
      files: ["config.json"],
    });
    const state = recorder.ofKind("cache-state")[0];
    expect(state?.cached).toEqual([false]);
    expect(state?.sizes).toEqual([undefined]);
  });
});
