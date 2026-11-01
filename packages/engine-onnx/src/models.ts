/**
 * The built-in catalogue: the models and voices the library ships manifests for.
 *
 * This module is *data only*. Adding a model, a language, or a family is an entry here — or
 * manifests passed to `createOnnxEngine()` by the application — never an engine change.
 *
 * Every entry is a pin: `revision` is the commit SHA the measurements were taken at, never
 * `main`, and every file declares its size and its Hub blob digest, because that is what a
 * consent dialog and an integrity check are made of.
 */
import type { LanguageCode, ModelManifest, VoiceManifest } from "@lite-speech/core";

import { createRegistry, type EngineRegistry } from "./catalog.js";

/** The language the built-in catalogue registers today. */
export const PRIMARY_LANGUAGE = "en";

/** One voice in the catalogue, with the presentation data a UI needs. */
export interface CatalogueVoice {
  readonly id: string;
  readonly name: string;
  readonly gender: "female" | "male";
  /**
   * Accent of the recording, e.g. `"en-GB"`.
   *
   * A label, not a language: the model does not switch pronunciation, so reporting the accent
   * as a language would make `supports("en-GB")` a promise the profile cannot keep.
   */
  readonly accent: string;
}

/**
 * The one verified Phonon profile of the MVP.
 *
 * Sizes come from the Hub tree and are what a consent dialog must show; the hashes are the
 * Hub's own blob digests, so an integrity check needs no second source of truth.
 *
 * `variant` marks the build each weight file belongs to. The engine keeps the files matching
 * the dtype it resolved, so pricing a profile and running it never disagree.
 */
export const PHONON_MODEL: ModelManifest = {
  id: "onnx-community/phonon-31m-en-ONNX",
  revision: "4f0cda7c60dddb4862d0e68ff9ff60c32ffdc9f4",
  license: "Apache-2.0",
  family: "phonon",
  languages: [PRIMARY_LANGUAGE],
  sampleRate: 24000,
  files: [
    {
      path: "config.json",
      size: 4952,
      sha256: "1c942e652a16d0ff56fb492a248d14d1b603e66427525f085ccb0def71187852",
    },
    {
      path: "tokenizer.json",
      size: 303009,
      sha256: "401de2690406fccc33f139921ba2d4fc22784b5c32f52c7b6db33ff0f635bf4a",
    },
    {
      path: "tokenizer_config.json",
      size: 144,
      sha256: "11797d02983710748a33d848ad0af623843969dcb005670093f43fec0717cfd4",
    },
    // `fp32` ships the two graphs without external data.
    {
      path: "onnx/flow_lm.onnx",
      variant: "fp32",
      size: 120089751,
      sha256: "27f933db151b81d30606b27f9a490dc9d73d93a4ee302c4f5ac29401fa858974",
    },
    {
      path: "onnx/mimi_decoder.onnx",
      variant: "fp32",
      size: 41267821,
      sha256: "842ba85dc849f9ed19d2a34dfaed2985fde729de75c7647f6033707454ec1d3d",
    },
    // `q4` splits the flow graph into a skeleton plus one external data chunk.
    {
      path: "onnx/flow_lm_q4.onnx",
      variant: "q4",
      size: 84132,
      sha256: "84b113376e3a768f833a0fcd1fc596af3fa5b9f33aed62ee4a40e0f1c21862be",
    },
    {
      path: "onnx/flow_lm_q4.onnx_data",
      variant: "q4",
      size: 26073088,
      sha256: "77420b5f392d2573d61b7cafcde2ec2a1a36ebe3d62be8ca3b3ef3b3ad73a6c5",
    },
    {
      path: "onnx/mimi_decoder_q4.onnx",
      variant: "q4",
      size: 41267821,
      sha256: "842ba85dc849f9ed19d2a34dfaed2985fde729de75c7647f6033707454ec1d3d",
    },
    // `q4f16` splits both graphs and needs `shader-f16` on WebGPU.
    {
      path: "onnx/flow_lm_q4f16.onnx",
      variant: "q4f16",
      size: 254451,
      sha256: "8c286f794689d85f07dc253a797196319bf654384198fa649d80666bad0aa9ec",
    },
    {
      path: "onnx/flow_lm_q4f16.onnx_data",
      variant: "q4f16",
      size: 19828736,
      sha256: "79651df4567d7b8a7fd6178406ae57c3f9dce24ac25beccbabceac9024ab3c22",
    },
    {
      path: "onnx/mimi_decoder_q4f16.onnx",
      variant: "q4f16",
      size: 54375,
      sha256: "010b6abb9a3022c2cb119cb8a540da902fcc44d363625010aa2d7dff54d8d139",
    },
    {
      path: "onnx/mimi_decoder_q4f16.onnx_data",
      variant: "q4f16",
      size: 20608000,
      sha256: "b4021bb79ffc278883c649d8095fc5fca024db12f375f74e163036aefa566e7b",
    },
  ],
};

/**
 * The English voices of `voices/voices.jsonl` at the pinned revision.
 *
 * The repository carries 102 voices across six design languages, but this profile always speaks
 * English (M1 §9.4): the design language of a Spanish or German embedding is metadata, not a
 * capability, so only English-designed voices are registered and every one reports `"en"`.
 *
 * `default` is the model card's own default voice rather than a speaker from `voices.jsonl`; it
 * is kept first so `createTts()` picks what the upstream runtime also treats as canonical.
 */
export const PHONON_VOICES: readonly CatalogueVoice[] = [
  { id: "default", name: "Default", gender: "female", accent: "en-US" },
  { id: "NbpkqMVS3CJeq2j8", name: "Zoey", gender: "female", accent: "en-US" },
  { id: "NIWIOv5vTi5eRMj8", name: "Juniper", gender: "female", accent: "en-US" },
  { id: "vjFyp2BcCyTlUmYb", name: "Everly", gender: "female", accent: "en-US" },
  { id: "4SZHfMpw-p46Ywgs", name: "Harper", gender: "female", accent: "en-US" },
  { id: "bXwrkHhURL1VFJdG", name: "Hadley", gender: "female", accent: "en-US" },
  { id: "nGzMHMPAWGaJFHOo", name: "Delia", gender: "female", accent: "en-US" },
  { id: "LMDIYE0Oa377Qh6Z", name: "Easton", gender: "male", accent: "en-US" },
  { id: "40QrMVgAxFiUnP9f", name: "Micah", gender: "male", accent: "en-US" },
  { id: "0zLApsZfVLLNaEGl", name: "Trevor", gender: "male", accent: "en-US" },
  { id: "_6Aslh2DxfmnRLmP", name: "Russell", gender: "male", accent: "en-US" },
  { id: "qIxlZ7Ei8KvWlyQH", name: "Preston", gender: "male", accent: "en-US" },
  { id: "d9XCyhjo4JhkHF1d", name: "Miles", gender: "male", accent: "en-US" },
  { id: "6PWnV0Nq4wu7RVBT", name: "Maeve", gender: "female", accent: "en-GB" },
  { id: "GgfEkEJtxZR7gnpy", name: "Freya", gender: "female", accent: "en-GB" },
  { id: "uem82D50GRv2Dwma", name: "Pippa", gender: "female", accent: "en-GB" },
  { id: "y9bQqIDnyxwqK01t", name: "Elodie-Rose", gender: "female", accent: "en-GB" },
  { id: "gDT1nz7Ie36ZhL-C", name: "Imogen", gender: "female", accent: "en-GB" },
  { id: "m9MxpY6IUGXtWp76", name: "Lottie", gender: "female", accent: "en-GB" },
  { id: "X6NceqF285Gr04gb", name: "Alfie", gender: "male", accent: "en-GB" },
  { id: "YyKcxPG7iIgS3jsK", name: "Hugh", gender: "male", accent: "en-GB" },
  { id: "zUCxzrOrugTSis03", name: "Edwin", gender: "male", accent: "en-GB" },
  { id: "JOW0vchyswkQUMDS", name: "Spencer", gender: "male", accent: "en-GB" },
  { id: "dME3IWyZBvmh1n1q", name: "Toby", gender: "male", accent: "en-GB" },
  { id: "Zn7g6NxLqXlJb7Qi", name: "Teddy", gender: "male", accent: "en-GB" },
  { id: "gfP44RlHwr2kEwk7", name: "Bryony", gender: "female", accent: "en-GB" },
];

/**
 * SHA-256 of every registered voice embedding at the pinned revision.
 *
 * Taken from the Hub's own blob digests, so the manifest needs no second source of truth: a
 * voice downloaded from a mirror is verified against the same number the public Hub reports. The
 * table is keyed by the file name (`<voice>.bin`) because one voice id is not unique across
 * revisions.
 */
const PHONON_VOICE_SHA256: Readonly<Record<string, string>> = {
  default: "d9291d795d2b3b3ca5aaf0d61c44b378918c840824e7c0bc41cfb63e5b0c560b",
  NbpkqMVS3CJeq2j8: "2023c0fddede488a609b911803345ebc333b406fbfb39d3444fae5c7459000db",
  NIWIOv5vTi5eRMj8: "762de2a472be583d3e358ad1c1e3fca0e16e7d0be7c96c70ff5116ce7437e633",
  vjFyp2BcCyTlUmYb: "a2bbd50a8587c0236c0d84ed81e10d91bce89ff92bb6604f830c4dcc0fc294e2",
  "4SZHfMpw-p46Ywgs": "331b969dbe31746edfdff9937fa8c27029d21c1fb3f41a6aa3dee6def0d17ce2",
  bXwrkHhURL1VFJdG: "6c0812d2a2da1659202b46b8e08c68708a6ba30cf684ffc05737a1e7eda01383",
  nGzMHMPAWGaJFHOo: "a5e6eadb0436a229373833fe0cb790d3317e04ab227d4c8fbd2e3e68411dd329",
  LMDIYE0Oa377Qh6Z: "6467e69cb22525b9a2784f661a543c0b018cba655ccc30be2e153ed2ac0f30ef",
  "40QrMVgAxFiUnP9f": "186b193668d0e53f7570d24aa0f68609b6111208740eede15c15d52861c0457a",
  "0zLApsZfVLLNaEGl": "fc996d45cca76d38aa33f9cbedf5ac9c70bd692029287c8d4c49bc2a06b90206",
  _6Aslh2DxfmnRLmP: "c1d99829cddde28781e82aa81f020fb33a529b79c980406e0e03fb57f3dc1e08",
  qIxlZ7Ei8KvWlyQH: "1b81d1228569972a7abeb81a854d34f18f37c1f6f7243a02b31c184308ebbae7",
  d9XCyhjo4JhkHF1d: "ab542932f553e16be04d530ea2cddf1fbfc655d13074343d416bb76efaee86ae",
  "6PWnV0Nq4wu7RVBT": "f00756d86bcd873c0ab3c240a52a7470ac7f02e8cb12718e86d7563ecd9fa932",
  GgfEkEJtxZR7gnpy: "ad1c36dc0e9a601ee13c75a418314263b7555ab44ee4428490c27c36d9b0ffa6",
  uem82D50GRv2Dwma: "d6f8b84bd21462f3adf3b9a1b867ed01cfb57f7bc07ec000e7745d792f6be6d3",
  y9bQqIDnyxwqK01t: "67398bd736ee16da7c4311f7810b5e4c31f7470780e36538f1a9ad5574f857ea",
  "gDT1nz7Ie36ZhL-C": "3158772a170e412789815f86182f953c8b55abf1d657817b7d0c4fa80498a06c",
  m9MxpY6IUGXtWp76: "4a5e5aa843c7f0435301924a168b02578a532b4bd9e174af8036738214383625",
  X6NceqF285Gr04gb: "58792454eab6b4f10235bec6f37c50794965ad60e24af349acaf784cdae3539b",
  YyKcxPG7iIgS3jsK: "f73e14157996adc0b4269acbe3d5b6d7d069e814452edd811686aaa913090af4",
  zUCxzrOrugTSis03: "4c24cb5301e5a0412d4db3161f76b710f6674b25226c9ea6851c900873c7c0fb",
  JOW0vchyswkQUMDS: "40de871900a04e3ba8ca5c4cb2cbff899d2c1f0e763584758fb775f0a641c99d",
  dME3IWyZBvmh1n1q: "235d93042993b414466b634d85a819525962f4eabd49f6b72bc04dcc452b493b",
  Zn7g6NxLqXlJb7Qi: "b9e3ac95e09b85eea9bf92b79ac498bea93ab8c3c6e1402e14af5968bdcc0f14",
  gfP44RlHwr2kEwk7: "03e842cf581b6ca3071706e8314c5536dc9b0b649aef602477100e6c22406bd2",
};

/**
 * A voice manifest built from one catalogue entry.
 *
 * The digest is skipped rather than guessed when a voice has none, so a manifest can never claim
 * an integrity check it cannot perform.
 */
export function voiceManifest(
  model: ModelManifest,
  voice: CatalogueVoice,
  language: LanguageCode = PRIMARY_LANGUAGE,
): VoiceManifest {
  const path = `voices/${voice.id}.bin`;
  const sha256 = PHONON_VOICE_SHA256[voice.id];
  return {
    id: voice.id,
    modelId: model.id,
    language,
    revision: model.revision,
    license: model.license ?? "unknown",
    sampleRate: model.sampleRate ?? 24000,
    name: voice.name,
    gender: voice.gender,
    description: `Accent ${voice.accent}`,
    files: [{ path, size: 256_000, ...(sha256 !== undefined && { sha256 }) }],
  };
}

/** Voice manifests of {@link PHONON_MODEL}, one 256 000-byte embedding each. */
export const PHONON_VOICES_MANIFEST: readonly VoiceManifest[] = PHONON_VOICES.map((voice) =>
  voiceManifest(PHONON_MODEL, voice),
);

/**
 * The built-in catalogue.
 *
 * One entry per model. A new language or family is a new entry here, or an override passed to
 * `createOnnxEngine()` — this array is the only place that knows which models exist by default.
 */
export const BUILT_IN_MODELS: readonly ModelManifest[] = [PHONON_MODEL];

/** The built-in voices, for every model of {@link BUILT_IN_MODELS}. */
export const BUILT_IN_VOICES: readonly VoiceManifest[] = [...PHONON_VOICES_MANIFEST];

/**
 * The registry the engine uses when the caller does not supply one.
 *
 * Built from {@link BUILT_IN_MODELS} and {@link BUILT_IN_VOICES}, so overriding either is a
 * matter of passing manifests, not of replacing a function.
 */
export function createDefaultRegistry(
  models: readonly ModelManifest[] = BUILT_IN_MODELS,
  voices: readonly VoiceManifest[] = BUILT_IN_VOICES,
): EngineRegistry {
  return createRegistry({ models, voices });
}
