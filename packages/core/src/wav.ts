import { TTS_ERROR_CODES, TtsError } from "./errors.js";
import type { AudioResult } from "./types.js";

/**
 * Optional PCM → WAV encoder.
 *
 * This lives above the inference contract on purpose: engines only produce
 * mono `Float32Array` PCM, and encoding is an application concern. The module
 * exports one function so bundlers can drop it entirely when it is unused.
 */

/** Supported sample encodings. */
export type WavBitDepth = 16 | 32;

/** Options for {@link encodeWav}. */
export interface WavEncodeOptions {
  /**
   * Sample rate in Hz. Defaults to `audio.sampleRate`; required when encoding
   * raw PCM that carries no rate.
   */
  readonly sampleRate?: number;
  /**
   * `16` writes 16-bit signed integers (the interoperable default), `32`
   * writes IEEE-754 floats. Default: `16`.
   */
  readonly bitDepth?: WavBitDepth;
}

/** Minimal input accepted by {@link encodeWav}. */
export interface WavSource {
  readonly pcm: Float32Array;
  readonly sampleRate: number;
}

/**
 * Encodes mono PCM into a RIFF/WAVE byte array.
 *
 * Values outside `[-1, 1]` are clamped rather than wrapped around, so an
 * aggressive engine cannot turn a loud peak into noise. The function never
 * resamples and never changes the channel count.
 */
export function encodeWav(
  audio: WavSource | AudioResult,
  options: WavEncodeOptions = {},
): Uint8Array {
  const sampleRate = options.sampleRate ?? audio.sampleRate;
  const bitDepth = options.bitDepth ?? 16;
  if (typeof sampleRate !== "number" || !Number.isFinite(sampleRate) || sampleRate <= 0) {
    throw new TtsError(
      TTS_ERROR_CODES.INVALID_ARGUMENT,
      `encodeWav() needs a positive sample rate, got ${String(sampleRate)}`,
      { details: { sampleRate } },
    );
  }
  if (bitDepth !== 16 && bitDepth !== 32) {
    throw new TtsError(
      TTS_ERROR_CODES.INVALID_ARGUMENT,
      `encodeWav() supports bitDepth 16 or 32, got ${String(bitDepth)}`,
      { details: { bitDepth } },
    );
  }
  const pcm = audio.pcm;
  if (!(pcm instanceof Float32Array)) {
    throw new TtsError(TTS_ERROR_CODES.INVALID_ARGUMENT, "encodeWav() requires Float32Array PCM");
  }

  const channels = 1;
  const bytesPerSample = bitDepth / 8;
  const blockAlign = channels * bytesPerSample;
  const dataBytes = pcm.length * bytesPerSample;
  const buffer = new ArrayBuffer(44 + dataBytes);
  const view = new DataView(buffer);

  writeAscii(view, 0, "RIFF");
  view.setUint32(4, 36 + dataBytes, true);
  writeAscii(view, 8, "WAVE");
  writeAscii(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, bitDepth === 32 ? 3 : 1, true);
  view.setUint16(22, channels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * blockAlign, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, bitDepth, true);
  writeAscii(view, 36, "data");
  view.setUint32(40, dataBytes, true);

  let offset = 44;
  if (bitDepth === 16) {
    for (let index = 0; index < pcm.length; index += 1) {
      const sample = clamp(pcm[index]!);
      view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
      offset += 2;
    }
  } else {
    for (let index = 0; index < pcm.length; index += 1) {
      view.setFloat32(offset, clamp(pcm[index]!), true);
      offset += 4;
    }
  }
  return new Uint8Array(buffer);
}

function clamp(value: number): number {
  if (Number.isNaN(value)) {
    return 0;
  }
  return value < -1 ? -1 : value > 1 ? 1 : value;
}

function writeAscii(view: DataView, offset: number, text: string): void {
  for (let index = 0; index < text.length; index += 1) {
    view.setUint8(offset + index, text.charCodeAt(index));
  }
}
