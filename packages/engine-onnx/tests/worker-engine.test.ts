/**
 * The worker path, with a fake `Worker`.
 *
 * In Node the engine runs in-thread, so the message-passing half — correlation, abort, late-frame
 * discard, worker death — would otherwise only be covered by the browser suite. A fake worker makes
 * that path deterministic in Node and lets a test observe every posted message.
 */
import { createTts, isTtsError, TTS_ERROR_CODES, type DebugEvent } from "@lite-speech/core";
import { describe, expect, it } from "vitest";

import { createOnnxEngine } from "../src/create-onnx-engine.js";
import type { WorkerResponse } from "../src/worker-protocol.js";

/** A `Worker` stand-in that records requests and lets a test answer them. */
class FakeWorker {
  readonly received: unknown[] = [];
  onmessage: ((event: MessageEvent<WorkerResponse>) => void) | null = null;
  onerror: ((event: ErrorEvent) => void) | null = null;
  terminated = false;

  postMessage(message: unknown): void {
    this.received.push(message);
  }

  terminate(): void {
    this.terminated = true;
  }

  /** Delivers a response as if the worker had posted it. */
  reply(response: WorkerResponse): void {
    this.onmessage?.({ data: response } as MessageEvent<WorkerResponse>);
  }

  /** Every posted request of one kind, in order. */
  all(kind: string): Record<string, unknown>[] {
    return this.received.filter(
      (entry): entry is Record<string, unknown> => (entry as { kind: string }).kind === kind,
    );
  }

  /**
   * Waits until a request of `kind` has been posted and returns the latest one.
   *
   * Polling beats a fixed number of microtask ticks: the engine awaits worker creation and its own
   * load bookkeeping before posting, and that depth is an implementation detail no test should encode.
   */
  async waitFor(kind: string, timeoutMs = 5_000): Promise<Record<string, unknown>> {
    const deadline = Date.now() + timeoutMs;
    for (;;) {
      const found = this.all(kind).at(-1);
      if (found !== undefined) {
        return found;
      }
      if (Date.now() > deadline) {
        throw new Error(
          `Timed out waiting for "${kind}"; posted: ${this.received
            .map((entry) => (entry as { kind: string }).kind)
            .join(", ")}`,
        );
      }
      await new Promise((resolve) => setTimeout(resolve, 1));
    }
  }

  /** Waits until `kind` has been posted at least `count` times. */
  async waitForCount(kind: string, count: number, timeoutMs = 5_000): Promise<void> {
    const deadline = Date.now() + timeoutMs;
    while (this.all(kind).length < count) {
      if (Date.now() > deadline) {
        throw new Error(`Timed out waiting for ${String(count)} "${kind}" requests`);
      }
      await new Promise((resolve) => setTimeout(resolve, 1));
    }
  }
}

/** Builds an engine with a fake worker, plus the worker so a test can answer requests. */
async function createWorkerEngine(): Promise<{
  engine: Awaited<ReturnType<typeof createOnnxEngine>>;
  worker: FakeWorker;
}> {
  const worker = new FakeWorker();
  const engine = await createOnnxEngine({
    device: "wasm",
    dtype: "fp32",
    worker: () => worker as unknown as Worker,
  });
  return { engine, worker };
}

/** Answers one load request with capabilities and a `loaded` acknowledgement. */
function replyLoaded(worker: FakeWorker, id: number): void {
  worker.reply({
    kind: "capabilities",
    id,
    device: "wasm",
    dtype: "fp32",
    modelId: "onnx-community/phonon-31m-en-ONNX",
    modelVersion: "4f0cda7c60dddb4862d0e68ff9ff60c32ffdc9f4",
    sampleRate: 24_000,
    streaming: "incremental-frames",
  });
  worker.reply({ kind: "loaded", id });
}

/** Creates a `Tts` and completes its first load against the fake worker. */
async function preloaded(
  engine: Awaited<ReturnType<typeof createOnnxEngine>>,
  worker: FakeWorker,
): Promise<ReturnType<typeof createTts>> {
  const tts = createTts({ engine, offline: "allow", voice: "default" });
  const preload = tts.preload();
  const load = await worker.waitFor("load");
  replyLoaded(worker, load["id"] as number);
  await preload;
  return tts;
}

describe("engine on a worker", () => {
  it("posts a load request shaped by the manifest", async () => {
    const { engine, worker } = await createWorkerEngine();
    const tts = createTts({ engine, offline: "allow", voice: "default" });
    const preload = tts.preload();
    const load = await worker.waitFor("load");

    expect(load).toMatchObject({
      kind: "load",
      modelId: "onnx-community/phonon-31m-en-ONNX",
      revision: "4f0cda7c60dddb4862d0e68ff9ff60c32ffdc9f4",
      family: "phonon",
      device: "wasm",
      dtype: "fp32",
      language: "en",
      voice: "default",
      voicePath: "voices/default.bin",
    });
    // The voice is addressed by its manifest, not by a path the engine invents.
    expect(load["voiceSha256"]).toBe(
      "d9291d795d2b3b3ca5aaf0d61c44b378918c840824e7c0bc41cfb63e5b0c560b",
    );
    replyLoaded(worker, load["id"] as number);
    await preload;
    expect(tts.isReady()).toBe(true);
    await tts.dispose();
  });

  it("correlates a result and uses the sample rate the worker reports", async () => {
    const { engine, worker } = await createWorkerEngine();
    const tts = await preloaded(engine, worker);

    const synthesize = tts.synthesize("Hello.");
    const request = await worker.waitFor("synthesize");
    worker.reply({
      kind: "result",
      id: request["id"] as number,
      pcm: new Float32Array([0.1, -0.1, 0.2, -0.2]).buffer as ArrayBuffer,
      sampleRate: 16_000,
      frames: 1,
      durationMs: 1,
    });

    const result = await synthesize;
    // The worker is authoritative for the rate; the manifest's 24 kHz is only a fallback.
    expect(result.sampleRate).toBe(16_000);
    expect(result.durationMs).toBeCloseTo((4 / 16_000) * 1000, 6);
    expect([...result.pcm].map((value) => Number(value.toFixed(5)))).toEqual([
      0.1, -0.1, 0.2, -0.2,
    ]);
    await tts.dispose();
  });

  it("pulls one frame per chunk and appends the single final chunk itself", async () => {
    const { engine, worker } = await createWorkerEngine();
    const tts = await preloaded(engine, worker);

    const sizes: number[] = [];
    const finals: boolean[] = [];
    const consume = (async (): Promise<void> => {
      for await (const chunk of tts.stream("Hello.")) {
        sizes.push(chunk.pcm.length);
        finals.push(chunk.final);
      }
    })();

    const open = await worker.waitFor("stream-open");
    worker.reply({ kind: "stream-opened", id: open["id"] as number });

    for (const [index, samples] of [1_920, 1_920].entries()) {
      // A `stream-next` is only posted after the previous frame resolved. That is the backpressure
      // guarantee, so the test waits for exactly one outstanding pull at a time.
      await worker.waitForCount("stream-next", index + 1);
      const next = worker.all("stream-next").at(-1)!;
      worker.reply({
        kind: "chunk",
        id: next["id"] as number,
        pcm: new Float32Array(samples).buffer as ArrayBuffer,
        sampleRate: 24_000,
      });
    }

    // The third pull is the one that discovers the end.
    await worker.waitForCount("stream-next", 3);
    const last = worker.all("stream-next").at(-1)!;
    worker.reply({ kind: "stream-end", id: last["id"] as number });
    await consume;

    // The worker never says "final"; the engine adds the terminal chunk because only it knows the
    // stream ended, and the core contract requires exactly one.
    expect(sizes).toEqual([1_920, 1_920, 0]);
    expect(finals).toEqual([false, false, true]);
    expect(worker.all("stream-next")).toHaveLength(3);
    await tts.dispose();
  });

  it("closes the session when the consumer breaks out of the loop", async () => {
    const { engine, worker } = await createWorkerEngine();
    const tts = await preloaded(engine, worker);

    const first = tts.stream("Hello.")[Symbol.asyncIterator]();
    const consumeOpen = first.next();

    const open = await worker.waitFor("stream-open");
    worker.reply({ kind: "stream-opened", id: open["id"] as number });
    const next = await worker.waitFor("stream-next");
    worker.reply({
      kind: "chunk",
      id: next["id"] as number,
      pcm: new Float32Array(8).buffer as ArrayBuffer,
      sampleRate: 24_000,
    });
    const chunk = await consumeOpen;
    expect(chunk.done).toBe(false);

    // Breaking out of the loop is `return()` on the iterator, which must stop the model rather than
    // merely stop listening to it.
    await first.return?.();
    expect(worker.all("stream-close")).toHaveLength(1);
    await tts.dispose();
  });

  it("rejects with ABORTED, cancels the worker, and stays usable", async () => {
    const { engine, worker } = await createWorkerEngine();
    const tts = await preloaded(engine, worker);

    const controller = new AbortController();
    const consume = (async (): Promise<void> => {
      for await (const chunk of tts.stream("Hello.", { signal: controller.signal })) {
        void chunk;
      }
    })();

    const open = await worker.waitFor("stream-open");
    worker.reply({ kind: "stream-opened", id: open["id"] as number });
    const pendingPull = await worker.waitFor("stream-next");

    controller.abort();

    await expect(consume).rejects.toSatisfy(
      (error: unknown) => isTtsError(error) && error.code === TTS_ERROR_CODES.ABORTED,
    );
    // A cooperative cancel goes out so the model stops between frames.
    const cancel = await worker.waitFor("cancel");
    expect(cancel["id"]).toBe(open["id"]);

    // A frame that arrives after the abort finds no pending entry and is discarded.
    worker.reply({
      kind: "chunk",
      id: pendingPull["id"] as number,
      pcm: new Float32Array(8).buffer as ArrayBuffer,
      sampleRate: 24_000,
    });

    // The engine is reusable: abort cancelled the wait, not the worker.
    const synthesize = tts.synthesize("Again.");
    const request = await worker.waitFor("synthesize");
    worker.reply({
      kind: "result",
      id: request["id"] as number,
      pcm: new Float32Array([0.5]).buffer as ArrayBuffer,
      sampleRate: 24_000,
      frames: 1,
      durationMs: 1,
    });
    expect((await synthesize).pcm.length).toBe(1);
    await tts.dispose();
  });

  it("rejects every pending request when the worker dies", async () => {
    const { engine, worker } = await createWorkerEngine();
    const tts = createTts({ engine, offline: "allow" });
    const preload = tts.preload();
    await worker.waitFor("load");

    worker.onerror?.({ message: "worker crashed" } as ErrorEvent);

    await expect(preload).rejects.toSatisfy((error: unknown) => isTtsError(error));
    await tts.dispose();
  });

  it("reports a worker error response as a load failure", async () => {
    const { engine, worker } = await createWorkerEngine();
    const tts = createTts({ engine, offline: "allow" });
    const preload = tts.preload();
    const load = await worker.waitFor("load");

    worker.reply({ kind: "error", id: load["id"] as number, message: "no phonon config" });

    await expect(preload).rejects.toSatisfy(
      (error: unknown) => isTtsError(error) && error.code === TTS_ERROR_CODES.MODEL_LOAD_FAILED,
    );
    await tts.dispose();
  });

  it("reports a worker error response mid-stream as a stream failure", async () => {
    const { engine, worker } = await createWorkerEngine();
    const tts = await preloaded(engine, worker);

    const consume = (async (): Promise<void> => {
      for await (const chunk of tts.stream("Hello.")) {
        void chunk;
      }
    })();
    const open = await worker.waitFor("stream-open");
    worker.reply({ kind: "stream-opened", id: open["id"] as number });
    const next = await worker.waitFor("stream-next");
    worker.reply({ kind: "error", id: next["id"] as number, message: "decoder exploded" });

    await expect(consume).rejects.toSatisfy(
      (error: unknown) => isTtsError(error) && error.code === TTS_ERROR_CODES.STREAM_FAILED,
    );
    await tts.dispose();
  });

  it("survives the engine and core being separate copies of the package", async () => {
    // The published engine bundles `@lite-speech/core`, so at runtime the engine's `TtsError` and
    // the application's are two distinct classes. An `instanceof`-based guard would turn the
    // engine's ABORTED into STREAM_FAILED at the core boundary, so this asserts the structural
    // guard holds for a stream aborted mid-flight.
    const { engine, worker } = await createWorkerEngine();
    const tts = await preloaded(engine, worker);

    const controller = new AbortController();
    const consume = (async (): Promise<void> => {
      for await (const chunk of tts.stream("Hello.", { signal: controller.signal })) {
        void chunk;
      }
    })();
    const open = await worker.waitFor("stream-open");
    worker.reply({ kind: "stream-opened", id: open["id"] as number });
    await worker.waitFor("stream-next");
    controller.abort();

    await expect(consume).rejects.toSatisfy(
      (error: unknown) => isTtsError(error) && error.code === TTS_ERROR_CODES.ABORTED,
    );
    await tts.dispose();
  });

  it("asks the worker to dispose and then terminates it", async () => {
    const { engine, worker } = await createWorkerEngine();
    const tts = await preloaded(engine, worker);

    const disposing = tts.dispose();
    const dispose = await worker.waitFor("dispose");
    worker.reply({ kind: "disposed", id: dispose["id"] as number });
    await disposing;

    expect(worker.terminated).toBe(true);
  });

  it("emits the lifecycle debug events once", async () => {
    const events: string[] = [];
    const worker = new FakeWorker();
    const engine = await createOnnxEngine({
      device: "wasm",
      dtype: "fp32",
      worker: () => worker as unknown as Worker,
    });
    const tts = createTts({
      engine,
      offline: "allow",
      onDebug: (event) => {
        events.push(event.type);
      },
    });
    const preload = tts.preload();
    const load = await worker.waitFor("load");
    replyLoaded(worker, load["id"] as number);
    await preload;

    expect(events.filter((type) => type === "worker-spawn")).toHaveLength(1);
    expect(events).toContain("load-start");
    expect(events).toContain("load-done");
    expect(events).toContain("device-resolved");
    await tts.dispose();
  });

  it("forwards the worker's model-call timing as inference events", async () => {
    const worker = new FakeWorker();
    const engine = await createOnnxEngine({
      device: "wasm",
      dtype: "fp32",
      worker: () => worker as unknown as Worker,
    });
    const events: DebugEvent[] = [];
    const tts = createTts({
      engine,
      offline: "allow",
      onDebug: (event) => events.push(event),
    });
    const preload = tts.preload();
    const load = await worker.waitFor("load");
    replyLoaded(worker, load["id"] as number);
    await preload;

    // A stream's frames are separate `stream-next` requests, so `inference-done` arrives after the
    // last frame request settled. It must still reach `onDebug`.
    const iterator = tts.stream("Hello there.")[Symbol.asyncIterator]();
    // The generator body runs on the first `next()`, so the pull has to start before the wait.
    const firstPull = iterator.next();
    const open = await worker.waitFor("stream-open");
    const openId = open["id"] as number;
    worker.reply({ kind: "inference-start", id: openId, textLength: 12 });
    worker.reply({ kind: "stream-opened", id: openId });
    const next = await worker.waitFor("stream-next");
    worker.reply({
      kind: "chunk",
      id: next["id"] as number,
      pcm: new Float32Array([0.1, -0.1]).buffer as ArrayBuffer,
      sampleRate: 24_000,
    });
    await firstPull;
    worker.reply({
      kind: "inference-done",
      id: openId,
      durationMs: 12,
      samples: 2,
      sampleRate: 24_000,
    });
    worker.reply({ kind: "stream-end", id: next["id"] as number });

    expect(events.filter((event) => event.type === "inference-start")).toEqual([
      expect.objectContaining({ requestId: openId, textLength: 12, engine: "onnx" }),
    ]);
    expect(events.filter((event) => event.type === "inference-done")).toEqual([
      expect.objectContaining({ requestId: openId, samples: 2, sampleRate: 24_000 }),
    ]);
    await iterator.return?.();
    await tts.dispose();
  });

  it("returns from an abandoned stream that is blocked on a frame", async () => {
    const { engine, worker } = await createWorkerEngine();
    const tts = await preloaded(engine, worker);

    const iterator = tts.stream("Hello.")[Symbol.asyncIterator]();
    const pendingFrame = iterator.next();

    const open = await worker.waitFor("stream-open");
    worker.reply({ kind: "stream-opened", id: open["id"] as number });
    const pull = await worker.waitFor("stream-next");
    // The worker never answers this pull — the consumer gives up while waiting for it.
    void pull;

    // `return()` must not wait for a frame nobody wants: it ends the iteration instead of
    // deadlocking against the unanswered pull.
    await expect(iterator.return?.()).resolves.toMatchObject({ done: true });
    // The abandoned pull settles as ABORTED rather than hanging forever.
    await expect(pendingFrame).rejects.toSatisfy(
      (error: unknown) => isTtsError(error) && error.code === TTS_ERROR_CODES.ABORTED,
    );
    expect(worker.all("stream-close")).toHaveLength(1);
    await tts.dispose();
  });

  it("disposes even when the worker never acknowledges", async () => {
    const { engine, worker } = await createWorkerEngine();
    const tts = await preloaded(engine, worker);

    // No reply to the `dispose` request: the call must still settle, and the worker must die.
    const started = Date.now();
    await tts.dispose();
    expect(Date.now() - started).toBeLessThan(10_000);
    expect(worker.terminated).toBe(true);
  }, 15_000);

  it("never creates a second worker for a second load", async () => {
    let created = 0;
    const worker = new FakeWorker();
    const engine = await createOnnxEngine({
      device: "wasm",
      dtype: "fp32",
      worker: () => {
        created += 1;
        return worker as unknown as Worker;
      },
    });
    const tts = await preloaded(engine, worker);
    expect(created).toBe(1);
    // A second, already-loaded call must not spawn another worker.
    await expect(tts.preload()).resolves.toBeUndefined();
    expect(created).toBe(1);
    await tts.dispose();
  });
});
