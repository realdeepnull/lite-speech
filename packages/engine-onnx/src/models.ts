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
      size: 120089751,
      sha256: "27f933db151b81d30606b27f9a490dc9d73d93a4ee302c4f5ac29401fa858974",
    },
    {
      path: "onnx/mimi_decoder.onnx",
      size: 41267821,
      sha256: "842ba85dc849f9ed19d2a34dfaed2985fde729de75c7647f6033707454ec1d3d",
    },
    // `q4` splits the flow graph into a skeleton plus one external data chunk.
    {
      path: "onnx/flow_lm_q4.onnx",
      size: 84132,
      sha256: "84b113376e3a768f833a0fcd1fc596af3fa5b9f33aed62ee4a40e0f1c21862be",
    },
    {
      path: "onnx/flow_lm_q4.onnx_data",
      size: 26073088,
      sha256: "77420b5f392d2573d61b7cafcde2ec2a1a36ebe3d62be8ca3b3ef3b3ad73a6c5",
    },
    {
      path: "onnx/mimi_decoder_q4.onnx",
      size: 41267821,
      sha256: "842ba85dc849f9ed19d2a34dfaed2985fde729de75c7647f6033707454ec1d3d",
    },
    // `q4f16` splits both graphs and needs `shader-f16` on WebGPU.
    {
      path: "onnx/flow_lm_q4f16.onnx",
      size: 254451,
      sha256: "8c286f794689d85f07dc253a797196319bf654384198fa649d80666bad0aa9ec",
    },
    {
      path: "onnx/flow_lm_q4f16.onnx_data",
      size: 19828736,
      sha256: "79651df4567d7b8a7fd6178406ae57c3f9dce24ac25beccbabceac9024ab3c22",
    },
    {
      path: "onnx/mimi_decoder_q4f16.onnx",
      size: 54375,
      sha256: "010b6abb9a3022c2cb119cb8a540da902fcc44d363625010aa2d7dff54d8d139",
    },
    {
      path: "onnx/mimi_decoder_q4f16.onnx_data",
      size: 20608000,
      sha256: "b4021bb79ffc278883c649d8095fc5fca024db12f375f74e163036aefa566e7b",
    },
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

/**
 * SHA-256 of every registered voice embedding at the pinned revision.
 *
 * Taken from the Hub's own blob digests, so the manifest needs no second source of truth: a voice
 * downloaded from a mirror is verified against the same number the public Hub reports. The table
 * is keyed by the file name (`<voice>.bin`) because one voice id is not unique across revisions.
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

/** Voice manifests of {@link PHONON_MODEL}, one 256 000-byte embedding each. */
export const PHONON_VOICES_MANIFEST: readonly VoiceManifest[] = PHONON_VOICES.map((voice) => ({
  id: voice.id,
  modelId: PHONON_MODEL.id,
  language: "en",
  revision: PHONON_MODEL.revision,
  license: "Apache-2.0",
  sampleRate: 24000,
  files: [
    {
      path: `voices/${voice.id}.bin`,
      size: 256_000,
      ...(PHONON_VOICE_SHA256[voice.id] !== undefined && {
        sha256: PHONON_VOICE_SHA256[voice.id],
      }),
    },
  ],
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
 * The `remotePathTemplate` that pins every Hub lookup to `revision`.
 *
 * Transformers.js resolves model files *and* the tokenizer-file probe
 * (`get_tokenizer_files()` → `get_file_metadata()`, which drops the caller's
 * `revision` option) through `env.remotePathTemplate`. Left at its default
 * (`{model}/resolve/{revision}/` with `revision: "main"`), that probe asks for
 * `resolve/main/...`, which has two consequences a pinned model cannot accept:
 * the request targets a moving branch, and its Cache Storage key differs from the
 * one a provisioned cache holds — so a fully provisioned profile still misses the
 * probe, and `offline: "strict"` makes a request whose only outcome is failure.
 *
 * Pinning the template fixes both at once, because the cache key is derived from the
 * same URL the request would use. `main` is left alone: a caller who asks for a
 * moving branch gets one.
 */
export function hubRevisionTemplate(revision: string): string {
  return `{model}/resolve/${encodeURIComponent(revision)}/`;
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
