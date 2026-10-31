import { normalizeTtsError, TTS_ERROR_CODES, TtsError } from "./errors.js";
import type { AudioChunk } from "./types.js";

/**
 * Bounded, contract-enforcing wrapper around an engine's audio stream.
 *
 * Core owns the queue so that a fast engine cannot grow memory when the
 * consumer is slow, and so that the chunk contract (`sequence` starts at `0`
 * and increments by one, `sampleRate` never changes, exactly one `final` chunk,
 * nothing after it) is checked in one place instead of in every engine.
 */

/** Options for {@link createAudioStream}. */
export interface AudioStreamOptions {
  /**
   * Maximum number of chunks buffered ahead of the consumer. Higher values
   * decouple engine latency from consumer latency at the cost of memory.
   * Default: 4. Must be a positive integer.
   */
  readonly highWaterMark?: number;
  /** Called with every validated chunk, before it is handed to the consumer. */
  readonly onChunk?: (chunk: AudioChunk) => void;
}

const DEFAULT_HIGH_WATER_MARK = 4;

/**
 * Wraps `source` into an `AsyncIterable<AudioChunk>` with a bounded prefetch
 * queue.
 *
 * Violations of the chunk contract abort the source and fail the iteration with
 * `TtsError(STREAM_FAILED)`. Breaking out of the loop (or calling `return()` on
 * the iterator) closes the source, so a cancelled stream stops pulling from the
 * model.
 */
export function createAudioStream(
  source: AsyncIterable<AudioChunk>,
  options: AudioStreamOptions = {},
): AsyncIterable<AudioChunk> {
  const highWaterMark = options.highWaterMark ?? DEFAULT_HIGH_WATER_MARK;
  if (!Number.isInteger(highWaterMark) || highWaterMark < 1) {
    throw new TtsError(
      TTS_ERROR_CODES.INVALID_ARGUMENT,
      `highWaterMark must be a positive integer, got ${String(highWaterMark)}`,
      { details: { highWaterMark } },
    );
  }
  if (
    source === null ||
    typeof source !== "object" ||
    typeof source[Symbol.asyncIterator] !== "function"
  ) {
    throw new TtsError(
      TTS_ERROR_CODES.INVALID_ARGUMENT,
      "createAudioStream() requires an AsyncIterable<AudioChunk>",
    );
  }
  return {
    [Symbol.asyncIterator](): AsyncIterator<AudioChunk> {
      return new AudioStreamIterator(source, highWaterMark, options.onChunk);
    },
  };
}

class AudioStreamIterator implements AsyncIterator<AudioChunk> {
  readonly #source: AsyncIterable<AudioChunk>;
  readonly #highWaterMark: number;
  readonly #onChunk: ((chunk: AudioChunk) => void) | undefined;
  #iterator: AsyncIterator<AudioChunk> | undefined;
  #buffer: AudioChunk[] = [];
  #waiters: Array<() => void> = [];
  #pulling = false;
  #done = false;
  #failure: TtsError | undefined;
  #expectedSequence = 0;
  #sampleRate: number | undefined;
  #sawFinal = false;

  constructor(
    source: AsyncIterable<AudioChunk>,
    highWaterMark: number,
    onChunk: ((chunk: AudioChunk) => void) | undefined,
  ) {
    this.#source = source;
    this.#highWaterMark = highWaterMark;
    this.#onChunk = onChunk;
  }

  async next(): Promise<IteratorResult<AudioChunk>> {
    for (;;) {
      const buffered = this.#buffer.shift();
      if (buffered !== undefined) {
        // Refill the window immediately so the prefetch pipeline stays full
        // instead of stalling until the consumer's next call.
        this.#startPull();
        this.#wake();
        return { done: false, value: buffered };
      }
      if (this.#failure) {
        throw this.#failure;
      }
      if (this.#done) {
        return { done: true, value: undefined };
      }
      this.#startPull();
      await this.#wait();
    }
  }

  async return(): Promise<IteratorResult<AudioChunk>> {
    await this.#close();
    this.#buffer = [];
    return { done: true, value: undefined };
  }

  async throw(error?: unknown): Promise<IteratorResult<AudioChunk>> {
    await this.#close();
    this.#buffer = [];
    throw error;
  }

  #startPull(): void {
    if (this.#pulling || this.#done || this.#failure) {
      return;
    }
    this.#pulling = true;
    void this.#pull();
  }

  async #pull(): Promise<void> {
    const iterator = (this.#iterator ??= this.#source[Symbol.asyncIterator]());
    try {
      while (this.#buffer.length < this.#highWaterMark && !this.#done) {
        const result = await iterator.next();
        if (result.done) {
          this.#finish();
          return;
        }
        const chunk = this.#accept(result.value);
        this.#buffer.push(chunk);
        this.#wake();
        if (chunk.final) {
          this.#done = true;
          await this.#close();
          return;
        }
      }
    } catch (error) {
      this.#fail(normalizeTtsError(error, TTS_ERROR_CODES.STREAM_FAILED, "Audio stream failed"));
    } finally {
      this.#pulling = false;
      this.#wake();
    }
  }

  /** Validates one chunk against the contract and returns it unchanged. */
  #accept(value: AudioChunk): AudioChunk {
    if (value === null || typeof value !== "object" || !(value.pcm instanceof Float32Array)) {
      throw this.#violation("chunk.pcm must be a Float32Array");
    }
    if (
      typeof value.sampleRate !== "number" ||
      !Number.isFinite(value.sampleRate) ||
      value.sampleRate < 0
    ) {
      throw this.#violation(
        `chunk.sampleRate must be a non-negative number, got ${String(value.sampleRate)}`,
      );
    }
    if (this.#sampleRate === undefined) {
      this.#sampleRate = value.sampleRate;
    } else if (value.sampleRate !== this.#sampleRate) {
      throw this.#violation(
        `chunk.sampleRate changed mid-stream (${this.#sampleRate} → ${value.sampleRate})`,
      );
    }
    if (value.sequence !== this.#expectedSequence) {
      throw this.#violation(
        `chunk.sequence must be ${this.#expectedSequence}, got ${String(value.sequence)}`,
      );
    }
    if (typeof value.final !== "boolean") {
      throw this.#violation("chunk.final must be a boolean");
    }
    this.#expectedSequence += 1;
    this.#sawFinal = value.final;
    this.#onChunk?.(value);
    return value;
  }

  #violation(message: string): TtsError {
    return new TtsError(
      TTS_ERROR_CODES.STREAM_FAILED,
      `Audio stream contract violation: ${message}`,
      {
        details: { sequence: this.#expectedSequence },
      },
    );
  }

  #finish(): void {
    if (!this.#sawFinal && this.#expectedSequence > 0) {
      this.#failure = this.#violation("stream ended without a final chunk");
    } else if (!this.#sawFinal && this.#expectedSequence === 0) {
      // A stream that yields nothing at all is treated as a contract violation:
      // the contract guarantees exactly one final chunk.
      this.#failure = this.#violation("stream ended without producing a chunk");
    }
    this.#done = true;
    this.#wake();
  }

  #fail(error: TtsError): void {
    this.#failure ??= error;
    this.#done = true;
    this.#buffer = [];
    this.#wake();
  }

  async #close(): Promise<void> {
    this.#done = true;
    const iterator = this.#iterator;
    this.#iterator = undefined;
    if (!iterator?.return) {
      return;
    }
    try {
      await iterator.return();
    } catch {
      // Closing a broken source must not mask the original failure.
    }
  }

  #wake(): void {
    const waiters = this.#waiters;
    this.#waiters = [];
    for (const resolve of waiters) {
      resolve();
    }
  }

  #wait(): Promise<void> {
    return new Promise<void>((resolve) => {
      this.#waiters.push(resolve);
    });
  }
}
