/**
 * `createOnnxEngine()`: the one factory an application calls.
 *
 * It resolves device/dtype *asynchronously* once, then hands a fully configured
 * {@link TransformersEngine} to the caller. `createTts()` is synchronous, so it cannot await
 * a WebGPU probe — which is why the factory is async while the engine it builds is not.
 */
import type { ModelManifest, VoiceManifest } from "@lite-speech/core";

import {
  PHONON_LANGUAGE,
  PHONON_MODEL,
  PHONON_VOICES_MANIFEST,
  createDefaultRegistry,
  type EngineRegistry,
} from "./models.js";
import { DEFAULT_PROFILES, resolveProfile, type ProfileRegistry } from "./profiles.js";
import { TransformersEngine, type FsProbe } from "./transformers-engine.js";
import {
  resolveDeviceDtype,
  type DeviceResolution,
  type OnnxDevice,
  type OnnxDtype,
} from "./webgpu.js";

/** A loader for the in-thread fallback, normally `loadPhonon` imported lazily. */
export type InThreadLoader = NonNullable<
  ConstructorParameters<typeof TransformersEngine>[0]["inThreadLoader"]
>;

/** Options for {@link createOnnxEngine}. */
export interface OnnxEngineOptions {
  /**
   * Device selection. Defaults to `"auto"`: WebGPU when a real adapter answers the probe,
   * otherwise WASM (or `cpu` under `onnxruntime-node`).
   *
   * `"webgpu"` never falls back — it fails at creation time when no adapter exists, because a
   * silent retry would hide the caller's wrong assumption behind working audio. `"wasm"` and
   * `"cpu"` skip probing entirely.
   */
  readonly device?: OnnxDevice;
  /**
   * Dtype override. Defaults to the measured safe choice per device: `fp32` on WASM/CPU
   * (quantized graphs were 20–26 % slower warm there) and `q4` on WebGPU (smallest variant that
   * does not need `shader-f16`).
   */
  readonly dtype?: OnnxDtype;
  /** Overrides the language → model map. */
  readonly models?: Readonly<Record<string, ModelManifest>>;
  /** Overrides the voice list. */
  readonly voices?: readonly VoiceManifest[];
  /** Replaces the built-in registry entirely; takes precedence over `models`/`voices`. */
  readonly registry?: EngineRegistry;
  /** Additional or replacement profiles, merged over the built-in one. */
  readonly profiles?: ProfileRegistry;
  /**
   * Worker factory. Defaults to a module worker created from this package's `./worker`
   * export. Pass `false` to run inference in-thread (Node.js), together with
   * {@link OnnxEngineOptions.inThreadLoader}.
   */
  readonly worker?: (() => Worker) | false;
  /** Loader for the in-thread path; required when `worker` is `false` or `Worker` is missing. */
  readonly inThreadLoader?: InThreadLoader;
  /** Filesystem cache probe for Node.js, where Cache Storage does not exist. */
  readonly fsProbe?: FsProbe;
}

/** The engine plus the device/dtype decision that produced it. */
export interface OnnxEngine extends TransformersEngine {
  /** Device and dtype this engine was created with, plus any fallback reason. */
  readonly resolved: DeviceResolution;
  /** Registry the engine reads; useful for introspection and tests. */
  readonly registry: EngineRegistry;
}

/** Environment default when the caller does not ask for a specific device. */
function supportsWorkers(): boolean {
  return typeof Worker === "function";
}

/**
 * Creates the default module worker.
 *
 * `new URL("./worker.js", import.meta.url)` is the form every bundler understands, and the
 * `./worker` package export exists so an application that copies its own worker file has a
 * documented entry to point at. A relative URL only resolves in a browser; in Node.js the
 * caller passes `worker: false` and an `inThreadLoader`.
 */
function defaultWorkerFactory(): () => Worker {
  return () => new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
}

/**
 * Creates the ONNX engine.
 *
 * Async only because device resolution is: a WebGPU probe cannot be awaited from the
 * synchronous `createTts()` boundary. Everything after that is the engine's own business.
 *
 * @example
 * ```ts
 * const engine = await createOnnxEngine({ device: "auto" });
 * const tts = createTts({ engine, voice: "LMDIYE0Oa377Qh6Z", offline: "strict" });
 * ```
 */
export async function createOnnxEngine(options: OnnxEngineOptions = {}): Promise<OnnxEngine> {
  const registry =
    options.registry ??
    createDefaultRegistry(
      options.models ?? { [PHONON_LANGUAGE]: PHONON_MODEL },
      options.voices ?? PHONON_VOICES_MANIFEST,
    );

  // Validating the profile registry here, not at synthesis time, turns "I registered a
  // profile under the wrong id" into an immediate error instead of silent Phonon audio.
  const profiles = { ...DEFAULT_PROFILES, ...options.profiles };
  for (const model of Object.values(options.models ?? { [PHONON_LANGUAGE]: PHONON_MODEL })) {
    if (model.family !== undefined && resolveProfile(model.family, profiles) === undefined) {
      throw new Error(
        `No profile is registered for model family "${model.family}". ` +
          `Known profiles: ${Object.keys(profiles).join(", ")}`,
      );
    }
  }

  const resolved = await resolveDeviceDtype(options.device ?? defaultDeviceOption(), options.dtype);

  const createWorker =
    options.worker === false
      ? undefined
      : (options.worker ?? (supportsWorkers() ? defaultWorkerFactory() : undefined));

  const engine = new TransformersEngine({
    registry,
    device: resolved.device,
    dtype: resolved.dtype,
    ...(createWorker !== undefined && { createWorker }),
    ...(options.inThreadLoader !== undefined && { inThreadLoader: options.inThreadLoader }),
    ...(options.fsProbe !== undefined && { fsProbe: options.fsProbe }),
  });

  return Object.assign(engine, { resolved, registry });
}

/** Device option default: `"auto"` in a worker-capable runtime, `"cpu"` under Node.js. */
function defaultDeviceOption(): OnnxDevice {
  return supportsWorkers() ? "auto" : "cpu";
}
