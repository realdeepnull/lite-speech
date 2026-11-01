/**
 * Public entry point of `@lite-speech/engine-onnx`.
 *
 * The engine is a `TtsEngine`, so an application needs this module plus
 * `@lite-speech/core` and nothing else:
 *
 * ```ts
 * import { createTts } from "@lite-speech/core";
 * import { createOnnxEngine } from "@lite-speech/engine-onnx";
 *
 * const engine = await createOnnxEngine({ device: "auto" });
 * const tts = createTts({ engine, voice: "LMDIYE0Oa377Qh6Z" });
 * ```
 *
 * Nothing here imports Transformers.js: the model runtime is reachable only through the
 * `./worker` export, which is a separate build entry. That keeps this module inside the size
 * budget and keeps the runtime out of an application's main bundle.
 *
 * Every export is deliberate. The catalogue and family seams are exported because
 * `createOnnxEngine({ models, voices, families })` is the documented way to add a model, a
 * language, or a runner; the registry helpers are exported because a mirror or a self-hosted
 * copy needs to build its own manifests without reaching into internals.
 */
export { createOnnxEngine, type OnnxEngine, type OnnxEngineOptions } from "./create-onnx-engine.js";

export {
  TransformersEngine,
  type FsProbe,
  type TransformersEngineOptions,
} from "./transformers-engine.js";

export {
  ENGINE_ID,
  assetUrl,
  createRegistry,
  hubRevisionTemplate,
  languagesOf,
  selectVariantFiles,
  variantOf,
  type EngineRegistry,
  type RegistryInput,
} from "./catalog.js";

export {
  BUILT_IN_MODELS,
  BUILT_IN_VOICES,
  PHONON_MODEL,
  PHONON_VOICES,
  PHONON_VOICES_MANIFEST,
  PRIMARY_LANGUAGE,
  createDefaultRegistry,
  voiceManifest,
  type CatalogueVoice,
} from "./models.js";

export {
  PHONON_FAMILY,
  PHONON_FRAME_RATE,
  PHONON_PROFILE,
  PHONON_STREAMING,
  normalizeSegment,
  phononMaxFrames,
  readPhononConfig,
  type PhononConfig,
} from "./phonon-profile.js";

export {
  resolveRunner,
  unknownFamilyError,
  type FamilyRegistry,
  type ModelProfile,
  type ModelRunner,
  type ModelRunnerFactory,
  type RunOptions,
  type RunnerLoadOptions,
  type VoiceRef,
} from "./families.js";

export {
  detectWebGpu,
  isFp16Supported,
  resolveDeviceDtype,
  type DeviceResolution,
  type OnnxDevice,
  type OnnxDtype,
  type ResolvedCapabilities,
  type ResolvedDevice,
  type ResolvedDtype,
} from "./webgpu.js";
