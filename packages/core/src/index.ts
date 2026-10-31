/**
 * Public entry point of `@lite-speech/core`.
 *
 * This is the only module applications import. Everything exported here is
 * deliberate: the contract (`createTts`, `Tts`, `TtsEngine`), the pure data
 * contracts, the offline/asset policy helpers, the stream and error primitives,
 * and the two optional layers above inference (`encodeWav`, typed emitter).
 *
 * Nothing here knows about ONNX, Transformers.js, Web Workers, WebGPU, or audio
 * playback. The core has zero runtime dependencies and stays usable in Node.js
 * and in a worker.
 */

export {
  TTS_ERROR_CODES,
  TtsError,
  formatTtsError,
  isAbortError,
  isTtsError,
  normalizeTtsError,
  type TtsErrorCode,
  type TtsErrorOptions,
} from "./errors.js";

export type {
  AudioChunk,
  AudioResult,
  DebugCallback,
  DebugEvent,
  LanguageCode,
  OfflinePolicy,
  ProgressCallback,
  ProgressEvent,
  StreamingClass,
  SynthesizeOptions,
  TtsCapabilities,
  TtsOptions,
  TtsRequest,
  VoiceInfo,
} from "./types.js";

export {
  assertTtsEngine,
  clearDefaultEngines,
  getDefaultEngines,
  isTtsEngine,
  registerDefaultEngine,
  type TtsEngine,
} from "./engine.js";

export { Tts, createTts, type PrepareAssetsOptions, type VerifyAssetsOptions } from "./tts.js";

export {
  DEFAULT_MAX_CHARS_PER_SEGMENT,
  prepareText,
  segmentText,
  type PrepareTextOptions,
  type SegmentTextOptions,
} from "./prepare.js";

export {
  assertBudget,
  assertIntegrity,
  cacheState,
  createAssetPlan,
  createIntegrityReport,
  estimateDownload,
  missingAssets,
  type AssetFile,
  type AssetIntegrityState,
  type AssetPlan,
  type AssetPlanFile,
  type AssetPlanInput,
  type AssetRole,
  type CacheState,
  type DownloadEstimate,
  type IntegrityReport,
  type ModelManifest,
  type VoiceManifest,
} from "./assets.js";

export { createAudioStream, type AudioStreamOptions } from "./stream.js";

export { TtsPool, ttsPoolKey, type TtsPoolOptions, type TtsPoolRequest } from "./pool.js";

export { createEmitter, type Emitter, type ListenerMap } from "./emitter.js";

export { encodeWav, type WavBitDepth, type WavEncodeOptions, type WavSource } from "./wav.js";
