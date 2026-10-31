import type { ModelManifest, VoiceManifest } from "@lite-speech/core";

/** Stable engine identifier. Echoed in `AudioResult.engine` and capabilities. */
export const ENGINE_ID = "onnx";

/** Origin of every default asset. Overridable per engine via `createOnnxEngine()`. */
const HF_BASE = "https://huggingface.co";

/**
 * The one verified Phonon profile of the MVP.
 *
 * Everything here is a pin, not a pointer: `revision` is the commit SHA the M1
 * spike measured, never `main`. Sizes come from the Hub tree and are what a
 * consent dialog must show; the hashes are the Hub's own blob digests, so a
 * later integrity check (M4) can verify a download without a second source of
 * truth.
 */
export const PHONON_MODEL: ModelManifest = {
  id: "onnx-community/phonon-31m-en-ONNX",
  revision: "4f0cda7c60dddb4862d0e68ff9ff60c32ffdc9f4",
  license: "Apache-2.0",
  family: "phonon",
  sampleRate: 24000,
  files: [
    { path: "config.json", size: 4952 },
    { path: "tokenizer.json", size: 303009 },
    { path: "tokenizer_config.json", size: 144 },
    // `fp32` ships the two graphs without external data.
    { path: "onnx/flow_lm.onnx", size: 120089751 },
    { path: "onnx/mimi_decoder.onnx", size: 41267821 },
    // `q4` splits the flow graph into a skeleton plus one external data chunk.
    { path: "onnx/flow_lm_q4.onnx", size: 84132 },
    { path: "onnx/flow_lm_q4.onnx_data", size: 26073088 },
    { path: "onnx/mimi_decoder_q4.onnx", size: 41267821 },
    // `q4f16` splits both graphs and needs `shader-f16` on WebGPU.
    { path: "onnx/flow_lm_q4f16.onnx", size: 254451 },
    { path: "onnx/flow_lm_q4f16.onnx_data", size: 19828736 },
    { path: "onnx/mimi_decoder_q4f16.onnx", size: 54375 },
    { path: "onnx/mimi_decoder_q4f16.onnx_data", size: 20608000 },
  ],
};

/** One English voice shipped by the pinned repository. */
interface PhononVoice {
  readonly id: string;
  readonly name: string;
  readonly gender: "female" | "male";
  /** Accent variant of the recording, kept as a label rather than as a language. */
  readonly locale: "en-US" | "en-GB";
}

/**
 * The English voices of `voices/voices.jsonl` at the pinned revision.
 *
 * The repository carries 102 voices across six design languages, but this profile always speaks
 * English (M1 §9.4): the design language of a Spanish or German embedding is metadata, not a
 * capability. Only English-designed voices are registered, and every one of them reports the
 * language `"en"` — advertising `"en-GB"` as a language would make `supports("en-GB")` a promise
 * this profile cannot keep, since it does not switch pronunciation.
 *
 * `default` is the model card's own default voice rather than a speaker from `voices.jsonl`; it
 * is kept first so `createTts()` picks what the upstream runtime also treats as canonical.
 */
const PHONON_VOICES: readonly PhononVoice[] = [
  { id: "default", name: "Default", gender: "female", locale: "en-US" },
  { id: "NbpkqMVS3CJeq2j8", name: "Zoey", gender: "female", locale: "en-US" },
  { id: "NIWIOv5vTi5eRMj8", name: "Juniper", gender: "female", locale: "en-US" },
  { id: "vjFyp2BcCyTlUmYb", name: "Everly", gender: "female", locale: "en-US" },
  { id: "4SZHfMpw-p46Ywgs", name: "Harper", gender: "female", locale: "en-US" },
  { id: "bXwrkHhURL1VFJdG", name: "Hadley", gender: "female", locale: "en-US" },
  { id: "nGzMHMPAWGaJFHOo", name: "Delia", gender: "female", locale: "en-US" },
  { id: "LMDIYE0Oa377Qh6Z", name: "Easton", gender: "male", locale: "en-US" },
  { id: "40QrMVgAxFiUnP9f", name: "Micah", gender: "male", locale: "en-US" },
  { id: "0zLApsZfVLLNaEGl", name: "Trevor", gender: "male", locale: "en-US" },
  { id: "_6Aslh2DxfmnRLmP", name: "Russell", gender: "male", locale: "en-US" },
  { id: "qIxlZ7Ei8KvWlyQH", name: "Preston", gender: "male", locale: "en-US" },
  { id: "d9XCyhjo4JhkHF1d", name: "Miles", gender: "male", locale: "en-US" },
  { id: "6PWnV0Nq4wu7RVBT", name: "Maeve", gender: "female", locale: "en-GB" },
  { id: "GgfEkEJtxZR7gnpy", name: "Freya", gender: "female", locale: "en-GB" },
  { id: "uem82D50GRv2Dwma", name: "Pippa", gender: "female", locale: "en-GB" },
  { id: "y9bQqIDnyxwqK01t", name: "Elodie-Rose", gender: "female", locale: "en-GB" },
  { id: "gDT1nz7Ie36ZhL-C", name: "Imogen", gender: "female", locale: "en-GB" },
  { id: "m9MxpY6IUGXtWp76", name: "Lottie", gender: "female", locale: "en-GB" },
  { id: "X6NceqF285Gr04gb", name: "Alfie", gender: "male", locale: "en-GB" },
  { id: "YyKcxPG7iIgS3jsK", name: "Hugh", gender: "male", locale: "en-GB" },
  { id: "zUCxzrOrugTSis03", name: "Edwin", gender: "male", locale: "en-GB" },
  { id: "JOW0vchyswkQUMDS", name: "Spencer", gender: "male", locale: "en-GB" },
  { id: "dME3IWyZBvmh1n1q", name: "Toby", gender: "male", locale: "en-GB" },
  { id: "Zn7g6NxLqXlJb7Qi", name: "Teddy", gender: "male", locale: "en-GB" },
  { id: "gfP44RlHwr2kEwk7", name: "Bryony", gender: "female", locale: "en-GB" },
];

/** Voice manifests of {@link PHONON_MODEL}, one 256 000-byte embedding each. */
export const PHONON_VOICES_MANIFEST: readonly VoiceManifest[] = PHONON_VOICES.map((voice) => ({
  id: voice.id,
  modelId: PHONON_MODEL.id,
  language: "en",
  revision: PHONON_MODEL.revision,
  license: "Apache-2.0",
  sampleRate: 24000,
  files: [{ path: `voices/${voice.id}.bin`, size: 256_000 }],
}));

/** Human-readable name of every registered voice. */
export const PHONON_VOICE_NAMES: Readonly<Record<string, string>> = Object.fromEntries(
  PHONON_VOICES.map((voice) => [voice.id, voice.name]),
);

/** Gender hint of every registered voice. */
export const PHONON_VOICE_GENDERS: Readonly<Record<string, string>> = Object.fromEntries(
  PHONON_VOICES.map((voice) => [voice.id, voice.gender]),
);

/**
 * The accent label of one voice, e.g. `"en-GB"`, or `undefined` for an unregistered id.
 *
 * The accent is a label, not a language: this profile speaks `"en"` regardless, so reporting
 * `"en-GB"` as a *language* would promise a pronunciation the model cannot switch to.
 */
export function voiceAccent(voiceId: string): string | undefined {
  return PHONON_VOICES.find((voice) => voice.id === voiceId)?.locale;
}

/** Language every manifest is registered under. */
export const PHONON_LANGUAGE = "en";

/**
 * ONNX filename suffix per dtype, matching Transformers.js' own mapping.
 *
 * The suffix is the one place where a resolved dtype becomes a file name, so both the asset
 * manifest resolution and the loader read it from here instead of duplicating the table.
 */
export const DTYPE_SUFFIX: Readonly<Record<string, string>> = {
  fp32: "",
  q4: "_q4",
  q4f16: "_q4f16",
};

/** Graph base names of the Phonon export. */
export const PHONON_GRAPHS = ["flow_lm", "mimi_decoder"] as const;

/** A synchronous, declarative registry: the engine never awaits a model lookup. */
export interface EngineRegistry {
  /** Resolves a request to its model manifest, or `undefined` when unknown. */
  model(language: string): ModelManifest | undefined;
  /** Resolves a request to its voice manifest, or `undefined` when unknown. */
  voice(model: ModelManifest, voice: string | undefined): VoiceManifest | undefined;
  /** Every voice of `model`, in registration order. */
  voices(model: ModelManifest): readonly VoiceManifest[];
}

/** Builds the URL of one repository file at the manifest's pinned revision. */
export function assetUrl(modelId: string, revision: string, path: string): string {
  return `${HF_BASE}/${modelId}/resolve/${revision}/${path}`;
}

/**
 * Creates the built-in registry: one Phonon model for `"en"`, plus every
 * registered English voice.
 *
 * `models` overrides the language → {@link ModelManifest} map and `voices`
 * overrides the voice list, which is how an application points at a mirror, a
 * self-hosted copy, or a different (future) profile without a new package.
 */
export function createDefaultRegistry(
  models: Readonly<Record<string, ModelManifest>> = { [PHONON_LANGUAGE]: PHONON_MODEL },
  voices: readonly VoiceManifest[] = PHONON_VOICES_MANIFEST,
): EngineRegistry {
  return {
    model(language: string): ModelManifest | undefined {
      return models[language];
    },
    voice(model: ModelManifest, voice: string | undefined): VoiceManifest | undefined {
      const candidates = voices.filter((candidate) => candidate.modelId === model.id);
      if (voice === undefined) {
        return candidates[0];
      }
      return candidates.find((candidate) => candidate.id === voice);
    },
    voices(model: ModelManifest): readonly VoiceManifest[] {
      return voices.filter((candidate) => candidate.modelId === model.id);
    },
  };
}
