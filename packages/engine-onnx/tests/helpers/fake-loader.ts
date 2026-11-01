/**
 * A scripted stand-in for a real runner factory.
 *
 * The engine's value is in its protocol handling, not in ONNX, so every engine and host test
 * drives a fake model. Faking it keeps the suite in Node, offline, and fast, and lets a test
 * make a stream take exactly as many frames as an assertion needs.
 */
import type { ProgressEvent, StreamingClass } from "@lite-speech/core";

import type {
  FamilyRegistry,
  ModelProfile,
  ModelRunner,
  RunnerLoadOptions,
} from "../../src/families.js";
import type { PhononConfig } from "../../src/phonon-profile.js";

/** Counters a test asserts on, instead of inspecting private state. */
export interface FakeLoaderStats {
  loads: number;
  voices: string[];
  disposes: number;
  lastVoice: string | undefined;
  lastLanguage: string | undefined;
  lastFamily: string | undefined;
  lastDtype: string | undefined;
  lastDevice: string | undefined;
  lastVoiceSha256: string | undefined;
  lastVoicePath: string | undefined;
  lastCacheDir: string | undefined;
}

/** Options for {@link createFakeLoader}. */
export interface FakeLoaderOptions {
  /** Frames each synthesized segment produces; default 3. */
  readonly frames?: number;
  /** Samples per frame; default 1 920 (80 ms at 24 kHz). */
  readonly samplesPerFrame?: number;
  /** Delay before the first frame, simulating model warm-up. */
  readonly firstFrameDelayMs?: number;
  /** Delay between frames, simulating per-frame compute. */
  readonly frameDelayMs?: number;
  /** Progress events to emit during load. */
  readonly progress?: readonly ProgressEvent[];
  /** Error to reject the load with. */
  readonly loadError?: unknown;
  /** Error to throw while synthesizing. */
  readonly synthesizeError?: unknown;
  /** Error to throw while disposing. */
  readonly disposeError?: unknown;
  /** Runs before the first frame, so a test can cancel deterministically. */
  readonly onFirstFrame?: () => void;
  /** Ignores the cancel flag, to prove the engine does not depend on the worker honouring it. */
  readonly ignoreCancel?: boolean;
  /** Sample rate the fake claims; default 24 kHz. */
  readonly sampleRate?: number;
  /** Streaming class the fake claims; default incremental frames. */
  readonly streaming?: StreamingClass;
}

/** Configuration the fake reports, matching the real Phonon values. */
export const FAKE_CONFIG: PhononConfig = {
  sampleRate: 24_000,
  frameRate: 12.5,
  samplesPerFrame: 1_920,
  latentDim: 32,
  voiceDim: 512,
  eosThreshold: 4,
  temperature: 0.3,
  flow: { numLayers: 2, numHeads: 2, headDim: 4 },
  mimi: { numLayers: 1, numHeads: 2, headDim: 4, convStates: {} },
};

const sleep = (ms: number): Promise<void> =>
  ms <= 0 ? Promise.resolve() : new Promise((resolve) => setTimeout(resolve, ms));

/** The fake runner factory plus the stats a test asserts on. */
export interface FakeLoader {
  /** A family registry containing one `"phonon"` factory backed by this fake. */
  readonly families: FamilyRegistry;
  readonly stats: FakeLoaderStats;
}

/** Creates a loader that answers deterministically and records what it was asked for. */
export function createFakeLoader(options: FakeLoaderOptions = {}): FakeLoader {
  const stats: FakeLoaderStats = {
    loads: 0,
    voices: [],
    disposes: 0,
    lastVoice: undefined,
    lastLanguage: undefined,
    lastFamily: undefined,
    lastDtype: undefined,
    lastDevice: undefined,
    lastVoiceSha256: undefined,
    lastVoicePath: undefined,
    lastCacheDir: undefined,
  };
  const frames = options.frames ?? 3;
  const samplesPerFrame = options.samplesPerFrame ?? 1_920;

  const load = async (loadOptions: RunnerLoadOptions): Promise<ModelRunner> => {
    stats.loads += 1;
    stats.lastVoice = loadOptions.voice.id;
    stats.lastLanguage = loadOptions.voice.language;
    stats.lastFamily = loadOptions.family;
    stats.lastDtype = loadOptions.dtype;
    stats.lastDevice = loadOptions.device;
    stats.lastVoiceSha256 = loadOptions.voice.sha256;
    stats.lastVoicePath = loadOptions.voice.path;
    stats.lastCacheDir = loadOptions.cacheDir;
    stats.voices.push(loadOptions.voice.id);
    for (const event of options.progress ?? []) {
      loadOptions.onProgress?.(event);
    }
    if (options.loadError !== undefined) {
      throw options.loadError;
    }

    const profile: ModelProfile = {
      id: loadOptions.family,
      streaming: options.streaming ?? "incremental-frames",
      frameRate: FAKE_CONFIG.frameRate,
      prepareSegment: (segment) => segment,
      maxFrames: () => frames,
    };

    const loaded: ModelRunner = {
      profile,
      sampleRate: options.sampleRate ?? FAKE_CONFIG.sampleRate,
      async *synthesize(_segment, runOptions) {
        if (options.synthesizeError !== undefined) {
          throw options.synthesizeError;
        }
        await sleep(options.firstFrameDelayMs ?? 0);
        for (let frame = 0; frame < frames; frame += 1) {
          if (frame === 0) {
            options.onFirstFrame?.();
          }
          if (!options.ignoreCancel && runOptions.isCancelled()) {
            return;
          }
          if (frame > 0) {
            await sleep(options.frameDelayMs ?? 0);
          }
          yield Float32Array.from(
            { length: samplesPerFrame },
            (_, index) => frame + index / (samplesPerFrame * 100),
          );
        }
      },
      async dispose() {
        stats.disposes += 1;
        if (options.disposeError !== undefined) {
          throw options.disposeError;
        }
      },
    };
    return loaded;
  };

  return { families: { phonon: load }, stats };
}
