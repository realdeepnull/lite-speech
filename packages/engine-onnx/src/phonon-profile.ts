/**
 * The Phonon family: its configuration shape, its text rules, and its frame budget.
 *
 * This module is deliberately pure — no ONNX, no Transformers.js — so the unit suite can
 * exercise the rules that decide *what* the model is asked to say without downloading a
 * model. The graph driving lives in `phonon.ts`.
 *
 * Adding a family never edits this file: a family is a {@link ModelProfile} plus a
 * {@link ModelRunner}, both registered in `runners.ts`.
 */
import type { StreamingClass } from "@lite-speech/core";

import type { ModelProfile } from "./families.js";

/** Stable id of the Phonon family. Used as `metadata.profile` / manifest `family`. */
export const PHONON_FAMILY = "phonon";

/**
 * Shapes and sampling constants the runner needs from the model's `config.json`
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

/** The `phonon` section of the pinned model's `config.json`. */
interface PhononConfigJson {
  readonly sample_rate: number;
  readonly frame_rate: number;
  readonly samples_per_frame: number;
  readonly latent_dim: number;
  readonly voice_dim: number;
  readonly eos_threshold: number;
  readonly temperature: number;
  readonly flow_lm: {
    readonly num_layers: number;
    readonly num_heads: number;
    readonly head_dim: number;
  };
  readonly mimi_decoder: {
    readonly num_layers: number;
    readonly num_heads: number;
    readonly head_dim: number;
    readonly conv_states: Readonly<Record<string, readonly number[]>>;
  };
}

/** The slice of `config.json` the Phonon family reads. */
interface PhononModelConfigJson {
  readonly phonon?: PhononConfigJson;
}

/** Truthful streaming classification; a family that buffers says so. */
export const PHONON_STREAMING: StreamingClass = "incremental-frames";

/** Frames per second of audio; `undefined` for a family that is not frame-based. */
export const PHONON_FRAME_RATE = 12.5;

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
 * Turns a prepared segment into the string the model reads.
 *
 * The model runs past the end of its text unless the segment ends in a letter or
 * digit, so a period is appended when it does not. Capitalization and whitespace
 * collapsing are copied from the spike for the same reason: they are measured
 * inputs, not cosmetics.
 *
 * Language-neutral by construction: it touches only whitespace and the first
 * code point, so a German, Spanish, or Turkish segment is not mangled. Anything
 * script- or number-specific belongs to an opt-in normalizer, not here.
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

/**
 * Hard frame ceiling for one segment.
 *
 * The model's own stop logit ends generation long before this in practice; the
 * ceiling exists so a pathological input cannot allocate without bound.
 */
export function phononMaxFrames(
  prepared: string,
  inputLength: number,
  config: Pick<PhononConfig, "frameRate"> = { frameRate: PHONON_FRAME_RATE },
): number {
  const budget = Math.ceil((inputLength / CHARS_PER_TOKEN + MINIMUM_FRAMES) * config.frameRate);
  return Math.max(budget, MINIMUM_FRAMES) + tailFrames(prepared);
}

/** Trailing-frame reserve, sized by the word count of the prepared segment. */
function tailFrames(prepared: string): number {
  const trimmed = prepared.trim();
  const words = trimmed.length === 0 ? 0 : trimmed.split(/\s+/u).length;
  return words <= SHORT_SEGMENT_WORDS ? SHORT_SEGMENT_TAIL_FRAMES : LONG_SEGMENT_TAIL_FRAMES;
}

/**
 * The Phonon family's profile: pure data plus the measured text rules.
 *
 * This is the *static* declaration — what the family promises before a model is loaded. The
 * runner replaces `frameRate` and `maxFrames` with the values from the loaded `config.json`,
 * so a revision with a different frame rate needs no code change.
 */
export const PHONON_PROFILE: ModelProfile = {
  id: PHONON_FAMILY,
  streaming: PHONON_STREAMING,
  frameRate: PHONON_FRAME_RATE,
  prepareSegment: normalizeSegment,
  maxFrames: (prepared: string, inputLength: number): number =>
    phononMaxFrames(prepared, inputLength),
};

/**
 * Reads and validates the `phonon` section of a loaded configuration.
 * A model without it is a different family: reporting that as an error instead of
 * guessing at input shapes is what keeps a mismatch a message rather than noise.
 */
export function readPhononConfig(config: unknown): PhononConfig {
  const phonon = (config as PhononModelConfigJson | null | undefined)?.phonon;
  if (phonon === undefined || phonon === null) {
    throw new Error(
      `The model does not declare a "phonon" configuration, so the "${PHONON_FAMILY}" family ` +
        "cannot run it. Register a runner for this model family or point the engine at a Phonon export.",
    );
  }
  return {
    sampleRate: phonon.sample_rate,
    frameRate: phonon.frame_rate,
    samplesPerFrame: phonon.samples_per_frame,
    latentDim: phonon.latent_dim,
    voiceDim: phonon.voice_dim,
    eosThreshold: phonon.eos_threshold,
    temperature: phonon.temperature,
    flow: {
      numLayers: phonon.flow_lm.num_layers,
      numHeads: phonon.flow_lm.num_heads,
      headDim: phonon.flow_lm.head_dim,
    },
    mimi: {
      numLayers: phonon.mimi_decoder.num_layers,
      numHeads: phonon.mimi_decoder.num_heads,
      headDim: phonon.mimi_decoder.head_dim,
      convStates: phonon.mimi_decoder.conv_states,
    },
  };
}
