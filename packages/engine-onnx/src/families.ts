/**
 * The family seam: what a model family must provide to be usable.
 *
 * The worker speaks one generic protocol (load / synthesize / stream / cancel / dispose).
 * Everything that is true of a *model family* rather than of the runtime lives behind a
 * {@link ModelProfile} plus the {@link ModelRunner} that implements it. Adding a family —
 * another checkpoint, another architecture, another language — means registering a runner
 * and a manifest, never touching the engine, the worker protocol, or `@lite-speech/core`.
 *
 * A profile is deliberately *data plus pure text rules*: the classification a consumer is
 * allowed to rely on, and the shaping a segment needs before the model sees it. The graph
 * driving stays inside the runner, which only the worker bundle can reach.
 */
import type { LanguageCode, ProgressEvent, StreamingClass } from "@lite-speech/core";

/**
 * Where one voice's bytes live.
 *
 * A voice is addressed as a repository + revision + path rather than as
 * `"voices/<id>.bin"`, so a family may keep its embeddings in the model repository, in a
 * separate voice pack, or under a path that is not derived from the id at all.
 */
export interface VoiceRef {
  /** Voice identifier, as reported by `TtsEngine.voices()`. */
  readonly id: string;
  /** Language this voice speaks. */
  readonly language: LanguageCode;
  /** Repository holding the voice assets. */
  readonly repository: string;
  /** Immutable revision of that repository (commit SHA). */
  readonly revision: string;
  /** Path of the voice asset inside the repository. */
  readonly path: string;
  /** Expected SHA-256 of the asset, when the manifest declares one. */
  readonly sha256?: string | undefined;
}

/**
 * What a family promises about how it produces audio, and how a segment is shaped for it.
 *
 * `prepareSegment` and `maxFrames` receive only the prepared text and a token count, so a
 * profile is testable without a runtime — and no family can smuggle a graph call into it.
 */
export interface ModelProfile {
  /** Family id, e.g. `"phonon"`. Matches `ModelManifest.family`. */
  readonly id: string;
  /** Truthful streaming classification, surfaced through `TtsCapabilities`. */
  readonly streaming: StreamingClass;
  /** Frames per second of audio, when the family is frame-based. */
  readonly frameRate?: number;
  /** Turns one core-prepared segment into the text the model reads. */
  readonly prepareSegment: (segment: string) => string;
  /**
   * Hard frame ceiling for one segment, or `undefined` when the family is not
   * frame-based and its own stop condition is the only bound.
   */
  readonly maxFrames?: ((prepared: string, inputLength: number) => number) | undefined;
}

/** Per-call knobs a runner must honour. */
export interface RunOptions {
  /** Sampled between frames so a cancel lands without waiting for the end of the utterance. */
  readonly isCancelled: () => boolean;
  /** Overrides the model's own sampling temperature, when the family has one. */
  readonly temperature?: number | undefined;
}

/** A loaded model, independent of how it is transported. */
export interface ModelRunner {
  /** The family profile this runner implements. */
  readonly profile: ModelProfile;
  /** Native sample rate of the frames this runner yields. */
  readonly sampleRate: number;
  /**
   * Runs one prepared segment, yielding mono PCM frames in `[-1, 1]`.
   *
   * Frames may be of any length; the worker concatenates them for `synthesize()` and
   * passes them through one at a time for `stream()`. A family that can only produce a
   * whole utterance yields one frame and declares `streaming: "utterance"`.
   */
  synthesize(segment: string, options: RunOptions): AsyncGenerator<Float32Array>;
  /** Releases every session and GPU buffer. Idempotent. */
  dispose(): Promise<void>;
}

/** Everything a runner factory needs, all taken from the engine's manifests. */
export interface RunnerLoadOptions {
  readonly modelId: string;
  readonly revision: string;
  /** The family to run; also the manifest's `family`. */
  readonly family: string;
  /** Resolved device: `"wasm"`, `"webgpu"`, or `"cpu"` under `onnxruntime-node`. */
  readonly device: string;
  /** Resolved dtype, e.g. `"fp32"`, `"q4"`, or `"q4f16"`. */
  readonly dtype: string;
  /** The voice to load, with everything needed to address its bytes. */
  readonly voice: VoiceRef;
  /**
   * Directory for downloaded files. Only meaningful under Node.js, where the library caches to the
   * filesystem; the cache probe must resolve the same directory, so it is set in one place.
   */
  readonly cacheDir?: string | undefined;
  readonly onProgress?: ((event: ProgressEvent) => void) | undefined;
}

/** Loads one model family. The real ones live in `runners.ts`; tests pass a fake. */
export type ModelRunnerFactory = (options: RunnerLoadOptions) => Promise<ModelRunner>;

/** Family id → runner factory. `createOnnxEngine({ families })` extends it. */
export type FamilyRegistry = Readonly<Record<string, ModelRunnerFactory>>;

/**
 * Resolves the runner for a manifest's family.
 *
 * An *unknown* family returns `undefined` and the caller reports it: falling back to a default
 * family would run a different architecture through these input shapes and produce noise
 * instead of an error.
 */
export function resolveRunner(
  family: string | undefined,
  registry: FamilyRegistry,
): ModelRunnerFactory | undefined {
  return family === undefined ? undefined : registry[family];
}

/** Builds the "no runner registered" message, naming what is available. */
export function unknownFamilyError(family: string, registry: FamilyRegistry): Error {
  return new Error(
    `No runner is registered for model family "${family}". ` +
      `Known families: ${Object.keys(registry).join(", ") || "(none)"}`,
  );
}
