/**
 * Model-profile seam.
 *
 * The worker speaks one generic protocol (load / synthesize / stream / cancel /
 * dispose); everything that is true of a *model family* rather than of the runtime
 * lives behind {@link ModelProfile}. Adding a family means registering a profile —
 * not touching the engine, the worker protocol, or `@lite-speech/core`.
 *
 * This module holds types plus the pure text rules, so the unit suite can exercise
 * them in Node. The graph-driving half of the Phonon profile lives in `phonon.ts`,
 * which is only reachable from the worker bundle.
 */
import type { StreamingClass } from "@lite-speech/core";

/** Stable id of the one verified profile of the MVP. */
export const PHONON_PROFILE_ID = "phonon";

/**
 * Shapes and sampling constants the profile needs from the model's `config.json`
 * (`phonon.*`). Read at load time, so a revision with more layers needs no code change.
 */
export interface PhononConfig {
  /** Native sample rate of the decoder output. */
  readonly sampleRate: number;
  /** Latent frames per second of audio (12.5 for Phonon: 80 ms per frame). */
  readonly frameRate: number;
  /** PCM samples produced by one decoder call (1 920 at 24 kHz). */
  readonly samplesPerFrame: number;
  /** Width of one latent frame. */
  readonly latentDim: number;
  /** Width of one voice-embedding column. */
  readonly voiceDim: number;
  /** Flow-LM stop threshold applied to the `eos_logit` output. */
  readonly eosThreshold: number;
  /** Default sampling temperature; `0` makes generation deterministic. */
  readonly temperature: number;
  /** Shapes of the flow graph's key/value cache. */
  readonly flow: {
    readonly numLayers: number;
    readonly numHeads: number;
    readonly headDim: number;
  };
  /** Shapes of the decoder graph's transformer cache and convolution state. */
  readonly mimi: {
    readonly numLayers: number;
    readonly numHeads: number;
    readonly headDim: number;
    /** Per-convolution state tensors, keyed by their `past_*` input name. */
    readonly convStates: Readonly<Record<string, readonly number[]>>;
  };
}

/**
 * A model family: how to phrase a segment, how many frames to allow, and which
 * streaming class it honestly delivers.
 */
export interface ModelProfile {
  /** Profile id, stored as `metadata.profile` in the manifest. */
  readonly id: string;
  /** Truthful streaming classification, surfaced through `TtsCapabilities`. */
  readonly streaming: StreamingClass;
  /** Frames per second of audio, when the profile is frame-based. */
  readonly frameRate?: number;
  /** Turns one core-prepared segment into the text the model reads. */
  readonly prepareSegment: (segment: string) => string;
  /**
   * Hard frame ceiling for one segment.
   *
   * The model's own stop logit ends generation long before this in practice; the
   * ceiling exists so a pathological input cannot allocate without bound.
   */
  readonly maxFrames: (prepared: string, inputLength: number, config: PhononConfig) => number;
}

/** Profile registry: id → profile. `createOnnxEngine({ profiles })` extends it. */
export type ProfileRegistry = Readonly<Record<string, ModelProfile>>;

/** Characters per model token assumed by the frame budget (M1 spike rule). */
const CHARS_PER_TOKEN = 3;
/** Frames reserved so the first word of a very short segment is not clipped. */
const MINIMUM_FRAMES = 2;
/** Longest segment, in whitespace-separated words, that gets the longer tail. */
const SHORT_SEGMENT_WORDS = 4;
/** Trailing frames generated after the stop logit fires for a short segment. */
const SHORT_SEGMENT_TAIL_FRAMES = 3;
/** Trailing frames generated after the stop logit fires for a longer segment. */
const LONG_SEGMENT_TAIL_FRAMES = 1;

/**
 * The Phonon profile.
 *
 * Text handling stops at what the M1 spike measured: collapse whitespace, capitalize,
 * and terminate a trailing letter or digit. There is no number, date, or currency
 * expansion here — M1 §9.1 records that as a known limitation, and adding one inside
 * the engine would change pronunciation invisibly instead of as a documented feature.
 */
export const PHONON_PROFILE: ModelProfile = {
  id: PHONON_PROFILE_ID,
  streaming: "incremental-frames",
  frameRate: 12.5,
  prepareSegment: normalizeSegment,
  maxFrames: (prepared: string, inputLength: number, config: PhononConfig): number => {
    const budget = Math.ceil((inputLength / CHARS_PER_TOKEN + MINIMUM_FRAMES) * config.frameRate);
    return Math.max(budget, MINIMUM_FRAMES) + tailFrames(prepared);
  },
};

/** The built-in registry: exactly one verified profile. */
export const DEFAULT_PROFILES: ProfileRegistry = { [PHONON_PROFILE_ID]: PHONON_PROFILE };

/**
 * Resolves the profile declared by a manifest.
 *
 * A missing id means "the built-in profile", so a manifest written before profiles
 * existed still loads. An *unknown* id returns `undefined` and the caller reports it:
 * falling back to Phonon would run a different family through these input shapes and
 * produce noise instead of an error.
 */
export function resolveProfile(
  profileId: string | undefined,
  registry: ProfileRegistry = DEFAULT_PROFILES,
): ModelProfile | undefined {
  return profileId === undefined ? registry[PHONON_PROFILE_ID] : registry[profileId];
}

/**
 * Turns a prepared segment into the string the model reads.
 *
 * The model runs past the end of its text unless the segment ends in a letter or
 * digit, so a period is appended when it does not. Capitalization and whitespace
 * collapsing are copied from the spike for the same reason: they are measured
 * inputs, not cosmetics.
 */
export function normalizeSegment(segment: string): string {
  const collapsed = segment.trim().replace(/\s+/gu, " ");
  if (collapsed.length === 0) {
    return collapsed;
  }
  const first = String.fromCodePoint(collapsed.codePointAt(0) ?? 0);
  const capitalized = first.toUpperCase() + collapsed.slice(first.length);
  return /[\p{L}\p{N}]$/u.test(capitalized) ? `${capitalized}.` : capitalized;
}

/** Trailing-frame reserve, sized by the word count of the prepared segment. */
function tailFrames(prepared: string): number {
  const trimmed = prepared.trim();
  const words = trimmed.length === 0 ? 0 : trimmed.split(/\s+/u).length;
  return words <= SHORT_SEGMENT_WORDS ? SHORT_SEGMENT_TAIL_FRAMES : LONG_SEGMENT_TAIL_FRAMES;
}
