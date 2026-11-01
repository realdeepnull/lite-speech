/**
 * `createOnnxEngine()`: the one factory an application calls.
 *
 * It resolves device/dtype *asynchronously* once, then hands a fully configured
 * {@link TransformersEngine} to the caller. `createTts()` is synchronous, so it cannot await a
 * WebGPU probe — which is why the factory is async while the engine it builds is not.
 *
 * The factory is catalogue-agnostic: models, voices, and families are inputs. The built-in
 * catalogue is only the default, so supporting another model, another language, or another
 * runner never requires a code change here.
 */
import type { ModelManifest, VoiceManifest } from "@lite-speech/core";

import { createRegistry, type EngineRegistry } from "./catalog.js";
import { resolveRunner, unknownFamilyError, type FamilyRegistry } from "./families.js";
import { PHONON_MODEL, PHONON_VOICES_MANIFEST, createDefaultRegistry } from "./models.js";
import { TransformersEngine, type FsProbe } from "./transformers-engine.js";
import {
  resolveDeviceDtype,
  type DeviceResolution,
  type OnnxDevice,
  type OnnxDtype,
} from "./webgpu.js";

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
  /**
   * The models to register. Each manifest declares its `family` and the `languages` it serves,
   * so this is how an application adds a model, a language, or a mirror without a new package.
   */
  readonly models?: readonly ModelManifest[];
  /** The voices to register, across every model. */
  readonly voices?: readonly VoiceManifest[];
  /** Replaces the built-in catalogue entirely; takes precedence over `models`/`voices`. */
  readonly registry?: EngineRegistry;
  /**
   * Additional or replacement runner factories, merged over any the caller already built.
   *
   * The built-in families live in `runners.ts` and import the model runtime, so they are not
   * imported here: the Node entry and the worker both pass their own registry, which keeps this
   * module — the engine's main-thread surface — free of Transformers.js.
   */
  readonly families?: FamilyRegistry;
  /**
   * Worker factory. Defaults to a module worker created from this package's `./worker` export.
   * Pass `false` to run inference in-thread (Node.js).
   */
  readonly worker?: (() => Worker) | false;
  /**
   * Sampling temperature. Defaults to the model's own value; `0` makes generation deterministic,
   * which is what a quality comparison between dtypes needs.
   */
  readonly temperature?: number;
  /** Filesystem cache probe for Node.js, where Cache Storage does not exist. */
  readonly fsProbe?: FsProbe;
}

/** The engine plus the device/dtype decision that produced it. */
export interface OnnxEngine extends TransformersEngine {
  /** Device and dtype this engine was created with, plus any fallback reason. */
  readonly resolved: DeviceResolution;
  /** Registry the engine reads; useful for introspection and tests. */
  readonly registry: EngineRegistry;
  /** Runner factories available to the in-thread path. */
  readonly families: FamilyRegistry;
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
 * caller passes `worker: false`.
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
  const models = options.models ?? [PHONON_MODEL];
  const voices = options.voices ?? PHONON_VOICES_MANIFEST;
  const registry =
    options.registry ??
    (options.models === undefined && options.voices === undefined
      ? createDefaultRegistry()
      : createRegistry({ models, voices }));
  const families: FamilyRegistry = { ...options.families };

  const resolved = await resolveDeviceDtype(options.device ?? defaultDeviceOption(), options.dtype);

  const createWorker =
    options.worker === false
      ? undefined
      : (options.worker ?? (supportsWorkers() ? defaultWorkerFactory() : undefined));

  // Validating the family registry here, not at synthesis time, turns "I registered a runner
  // under the wrong id" into an immediate error instead of a worker that fails on first load.
  // Only the in-thread path is checked: a worker resolves its own built-in families, so the
  // manifests it hosts cannot be cross-checked from here.
  if (createWorker === undefined) {
    for (const model of registry.models()) {
      const family = model.family;
      if (family !== undefined && resolveRunner(family, families) === undefined) {
        throw unknownFamilyError(family, families);
      }
    }
  }

  const engine = new TransformersEngine({
    registry,
    families,
    device: resolved.device,
    dtype: resolved.dtype,
    ...(resolved.fallbackFrom !== undefined && { fallbackFrom: resolved.fallbackFrom }),
    ...(resolved.reason !== undefined && { fallbackReason: resolved.reason }),
    ...(options.temperature !== undefined && { temperature: options.temperature }),
    ...(createWorker !== undefined && { createWorker }),
    ...(options.fsProbe !== undefined && { fsProbe: options.fsProbe }),
  });

  return Object.assign(engine, { resolved, registry, families });
}

/** Device option default: `"auto"` in a worker-capable runtime, `"cpu"` under Node.js. */
function defaultDeviceOption(): OnnxDevice {
  return supportsWorkers() ? "auto" : "cpu";
}
