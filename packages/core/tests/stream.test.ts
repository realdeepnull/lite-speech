import { describe, expect, it } from "vitest";

import {
  TTS_ERROR_CODES,
  createAudioStream,
  createTts,
  encodeWav,
  type AudioChunk,
} from "../src/index";
import { createFakeEngine } from "./helpers/fake-engine";

function chunk(sequence: number, final: boolean, sampleRate = 24_000): AudioChunk {
  return { pcm: new Float32Array([0.1, -0.1]), sampleRate, sequence, final };
}

async function* fromList(chunks: readonly AudioChunk[]): AsyncGenerator<AudioChunk> {
  for (const item of chunks) {
    yield item;
  }
}

function deferred(): { promise: Promise<void>; resolve: () => void } {
  let resolve: () => void = () => {};
  const promise = new Promise<void>((resolveValue) => {
    resolve = resolveValue;
  });
  return { promise, resolve };
}

describe("createAudioStream()", () => {
  it("passes valid chunks through in order and completes after the final chunk", async () => {
    const source = fromList([chunk(0, false), chunk(1, false), chunk(2, true)]);
    const seen: number[] = [];

    for await (const item of createAudioStream(source)) {
      seen.push(item.sequence);
    }

    expect(seen).toEqual([0, 1, 2]);
  });

  it("rejects a duplicated or skipped sequence number", async () => {
    const source = fromList([chunk(0, false), chunk(2, true)]);

    await expect(async () => {
      for await (const chunk of createAudioStream(source)) {
        void chunk;
      }
    }).rejects.toMatchObject({ code: TTS_ERROR_CODES.STREAM_FAILED });
  });

  it("rejects a sample-rate change mid-stream", async () => {
    const source = fromList([chunk(0, false), chunk(1, true, 48_000)]);

    await expect(async () => {
      for await (const chunk of createAudioStream(source)) {
        void chunk;
      }
    }).rejects.toMatchObject({ code: TTS_ERROR_CODES.STREAM_FAILED });
  });

  it("rejects a stream that ends without a final chunk", async () => {
    const source = fromList([chunk(0, false), chunk(1, false)]);

    await expect(async () => {
      for await (const chunk of createAudioStream(source)) {
        void chunk;
      }
    }).rejects.toMatchObject({ code: TTS_ERROR_CODES.STREAM_FAILED });
  });

  it("rejects a stream that yields nothing", async () => {
    const source = fromList([]);

    await expect(async () => {
      for await (const chunk of createAudioStream(source)) {
        void chunk;
      }
    }).rejects.toMatchObject({ code: TTS_ERROR_CODES.STREAM_FAILED });
  });

  it("rejects non-Float32Array PCM", async () => {
    const source = fromList([
      { pcm: [1, 2] as unknown as Float32Array, sampleRate: 24_000, sequence: 0, final: true },
    ]);

    await expect(async () => {
      for await (const chunk of createAudioStream(source)) {
        void chunk;
      }
    }).rejects.toMatchObject({ code: TTS_ERROR_CODES.STREAM_FAILED });
  });

  it("applies backpressure: never pulls more than highWaterMark ahead", async () => {
    let pulled = 0;
    const gate = deferred();
    async function* source(): AsyncGenerator<AudioChunk> {
      for (let index = 0; index < 10; index += 1) {
        pulled += 1;
        if (index === 1) {
          await gate.promise;
        }
        yield chunk(index, index === 9);
      }
    }

    const stream = createAudioStream(source(), { highWaterMark: 2 });
    const iterator = stream[Symbol.asyncIterator]();
    const first = await iterator.next();
    expect(first.value?.sequence).toBe(0);
    // Give the prefetcher a chance to over-run; it must stop at the watermark.
    await new Promise((resolve) => setTimeout(resolve, 10));
    expect(pulled).toBeLessThanOrEqual(3);

    gate.resolve();
    let count = 1;
    for (let next = await iterator.next(); !next.done; next = await iterator.next()) {
      count += 1;
    }
    expect(count).toBe(10);
  });

  it("closes the source when the consumer stops early", async () => {
    let returned = false;
    const source: AsyncIterable<AudioChunk> = {
      [Symbol.asyncIterator]: () => {
        let index = 0;
        return {
          next: async () => ({ done: false as const, value: chunk(index++, false) }),
          return: async () => {
            returned = true;
            return { done: true as const, value: undefined };
          },
        };
      },
    };

    for await (const chunk of createAudioStream(source, { highWaterMark: 1 })) {
      expect(chunk.sequence).toBe(0);
      break;
    }

    expect(returned).toBe(true);
  });

  it("propagates a source error normalized to STREAM_FAILED", async () => {
    async function* source(): AsyncGenerator<AudioChunk> {
      yield chunk(0, false);
      throw new Error("worker died");
    }

    const error = await (async () => {
      try {
        for await (const chunk of createAudioStream(source())) {
          void chunk;
        }
        return undefined;
      } catch (value) {
        return value;
      }
    })();

    expect(error).toMatchObject({ code: TTS_ERROR_CODES.STREAM_FAILED });
    expect((error as { cause?: unknown }).cause).toBeInstanceOf(Error);
  });
});

describe("Tts.stream()", () => {
  it("yields ordered chunks with one final chunk and a stable sample rate", async () => {
    const engine = createFakeEngine({ chunks: 4, samplesPerChunk: 240 });
    const tts = createTts({ engine, offline: "allow" });
    const chunks: AudioChunk[] = [];

    for await (const item of tts.stream("Stream me")) {
      chunks.push(item);
    }

    expect(chunks.map((item) => item.sequence)).toEqual([0, 1, 2, 3]);
    expect(chunks.filter((item) => item.final)).toHaveLength(1);
    expect(chunks.at(-1)?.final).toBe(true);
    expect(new Set(chunks.map((item) => item.sampleRate))).toEqual(new Set([24_000]));
  });

  it("yields one empty final chunk for empty input without loading", async () => {
    const engine = createFakeEngine();
    const tts = createTts({ engine });
    const chunks: AudioChunk[] = [];

    for await (const item of tts.stream("   ")) {
      chunks.push(item);
    }

    expect(chunks).toEqual([{ pcm: new Float32Array(0), sampleRate: 0, sequence: 0, final: true }]);
    expect(engine.stats.load).toBe(0);
  });

  it("surfaces a contract violation as STREAM_FAILED", async () => {
    const engine = createFakeEngine({ chunks: 3, breakSequence: true });
    const tts = createTts({ engine, offline: "allow" });

    await expect(async () => {
      for await (const chunk of tts.stream("Hello")) {
        void chunk;
      }
    }).rejects.toMatchObject({ code: TTS_ERROR_CODES.STREAM_FAILED });
  });

  it("terminates with ABORTED when the signal aborts mid-stream", async () => {
    const engine = createFakeEngine({ chunks: 20, chunkDelayMs: 2 });
    const tts = createTts({ engine, offline: "allow" });
    const controller = new AbortController();
    const received: AudioChunk[] = [];

    await expect(async () => {
      for await (const item of tts.stream("Hello", { signal: controller.signal })) {
        received.push(item);
        if (received.length === 1) {
          controller.abort();
        }
      }
    }).rejects.toMatchObject({ code: TTS_ERROR_CODES.ABORTED });

    expect(received).toHaveLength(1);
    // Audio that was already delivered stays valid.
    expect(received[0]?.pcm).toBeInstanceOf(Float32Array);
  });

  it("propagates an engine stream error as STREAM_FAILED", async () => {
    const engine = createFakeEngine({ streamError: new Error("synthesis crashed") });
    const tts = createTts({ engine, offline: "allow" });

    await expect(async () => {
      for await (const chunk of tts.stream("Hello")) {
        void chunk;
      }
    }).rejects.toMatchObject({ code: TTS_ERROR_CODES.STREAM_FAILED });
  });

  it("emits time-to-first-chunk before stream-done", async () => {
    const types: string[] = [];
    const engine = createFakeEngine({ chunks: 2 });
    const tts = createTts({
      engine,
      offline: "allow",
      onDebug: (event) => types.push(event.type),
    });

    for await (const chunk of tts.stream("Hello")) {
      void chunk;
    }

    expect(types).toContain("time-to-first-chunk");
    expect(types.indexOf("time-to-first-chunk")).toBeLessThan(types.indexOf("stream-done"));
  });
});

describe("encodeWav()", () => {
  it("writes a valid 16-bit mono RIFF header and clamps samples", () => {
    const pcm = new Float32Array([0, 0.5, -0.5, 2, -2]);
    const wav = encodeWav({ pcm, sampleRate: 16_000 });
    const view = new DataView(wav.buffer, wav.byteOffset, wav.byteLength);

    expect(String.fromCharCode(...wav.subarray(0, 4))).toBe("RIFF");
    expect(String.fromCharCode(...wav.subarray(8, 12))).toBe("WAVE");
    expect(String.fromCharCode(...wav.subarray(36, 40))).toBe("data");
    expect(wav.byteLength).toBe(44 + pcm.length * 2);
    expect(view.getUint16(22, true)).toBe(1);
    expect(view.getUint32(24, true)).toBe(16_000);
    expect(view.getUint16(34, true)).toBe(16);
    expect(view.getUint32(40, true)).toBe(pcm.length * 2);
    expect(view.getInt16(44 + 6, true)).toBe(0x7fff);
    expect(view.getInt16(44 + 8, true)).toBe(-0x8000);
  });

  it("writes IEEE float samples for bitDepth 32", () => {
    const pcm = new Float32Array([0.25, -0.25]);
    const wav = encodeWav({ pcm, sampleRate: 24_000 }, { bitDepth: 32 });
    const view = new DataView(wav.buffer, wav.byteOffset, wav.byteLength);

    expect(view.getUint16(20, true)).toBe(3);
    expect(view.getUint16(34, true)).toBe(32);
    expect(view.getFloat32(44, true)).toBeCloseTo(0.25, 5);
    expect(view.getFloat32(48, true)).toBeCloseTo(-0.25, 5);
  });

  it("rejects a missing sample rate and an unsupported bit depth", () => {
    expect(() => encodeWav({ pcm: new Float32Array(1), sampleRate: 0 })).toThrowError(
      expect.objectContaining({ code: TTS_ERROR_CODES.INVALID_ARGUMENT }),
    );
    expect(() =>
      encodeWav({ pcm: new Float32Array(1), sampleRate: 24_000 }, { bitDepth: 24 as never }),
    ).toThrowError(expect.objectContaining({ code: TTS_ERROR_CODES.INVALID_ARGUMENT }));
  });
});
