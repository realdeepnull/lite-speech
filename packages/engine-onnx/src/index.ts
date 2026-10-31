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
 * Every export is deliberate. The profile and WebGPU seams are exported because
 * `createOnnxEngine({ profiles })` and a custom `device` policy are documented extension
 * points; the registry helpers are exported because a mirror or a self-hosted copy needs to
 * build its own manifests without reaching into internals.
 */
export {
  createOnnxEngine,
  type InThreadLoader,
  type OnnxEngine,
  type OnnxEngineOptions,
} from "./create-onnx-engine.js";

export {
  TransformersEngine,
  type FsProbe,
  type TransformersEngineOptions,
} from "./transformers-engine.js";

export {
  DTYPE_SUFFIX,
  ENGINE_ID,
  PHONON_GRAPHS,
  PHONON_LANGUAGE,
  PHONON_MODEL,
  PHONON_VOICE_GENDERS,
  PHONON_VOICE_NAMES,
  PHONON_VOICES_MANIFEST,
  assetUrl,
  createDefaultRegistry,
  hubRevisionTemplate,
  voiceAccent,
  type EngineRegistry,
} from "./models.js";

export {
  DEFAULT_PROFILES,
  PHONON_PROFILE,
  PHONON_PROFILE_ID,
  normalizeSegment,
  resolveProfile,
  type ModelProfile,
  type PhononConfig,
  type ProfileRegistry,
} from "./profiles.js";

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
