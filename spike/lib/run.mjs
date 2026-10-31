import { SAMPLE_RATE } from "./phonon.mjs";

/**
 * Runs one synthesis and collects the measurements the M1 gate needs.
 * Shared by the Node.js CLI and the browser page so both report identically.
 *
 * @param {{ speak: (text: string, options?: { temperature?: number }) => AsyncGenerator<Float32Array> }} model
 * @param {string} text
 * @param {{ temperature?: number }} [options]
 */
export async function measureSynthesis(model, text, options = {}) {
  /** @type {Float32Array[]} */
  const frames = [];
  /** @type {number[]} */
  const frameTimes = [];
  /** @type {number | null} */
  let firstFrameMs = null;

  const started = performance.now();
  for await (const pcm of model.speak(text, options)) {
    if (firstFrameMs === null) firstFrameMs = performance.now() - started;
    frames.push(pcm);
    frameTimes.push(performance.now() - started);
  }
  const synthMs = performance.now() - started;

  const length = frames.reduce((n, frame) => n + frame.length, 0);
  const pcm = new Float32Array(length);
  frames.reduce((offset, frame) => (pcm.set(frame, offset), offset + frame.length), 0);

  let min = Infinity;
  let max = -Infinity;
  let sumSq = 0;
  for (const sample of pcm) {
    if (sample < min) min = sample;
    if (sample > max) max = sample;
    sumSq += sample * sample;
  }

  const audioSeconds = length / SAMPLE_RATE;
  const frameSizes = [...new Set(frames.map((frame) => frame.length))];

  return {
    text,
    sampleRate: SAMPLE_RATE,
    channels: 1,
    samples: length,
    audioSeconds,
    frames: frames.length,
    frameSizes,
    firstFrameMs,
    frameIntervalMs:
      frameTimes.length > 1
        ? (frameTimes[frameTimes.length - 1] - frameTimes[0]) / (frameTimes.length - 1)
        : null,
    synthMs,
    rtf: synthMs / 1000 / audioSeconds,
    min,
    max,
    rms: Math.sqrt(sumSq / length),
    pcm,
  };
}
