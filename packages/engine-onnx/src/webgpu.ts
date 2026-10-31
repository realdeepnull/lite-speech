/**
 * Device and dtype policy.
 *
 * The default is deliberately boring: **WASM, `fp32`**. WebGPU is opt-in; `"auto"`
 * probes an adapter, `"webgpu"` demands one and throws without it.
 *
 * Two rules from the M1 spike are enforced here, not in documentation:
 *
 * 1. **An accepted `webgpu` request is not evidence of GPU execution.** In Node.js
 *    `getSupportedDevices()` reports `webgpu` and an explicit WebGPU load *succeeds*
 *    even though `navigator.gpu` does not exist (M1 §5.1). So the probe gates on
 *    `navigator.gpu` and adapter features, never on the runtime accepting a request.
 * 2. **`q4f16` requires `shader-f16`.** Asking for it on an adapter without the
 *    feature fails at load time by design, so the request is rejected up front with an
 *    actionable message instead of producing a half-loaded worker.
 *
 * The adapter probe is memoized **including rejections**: `requestAdapter()` can cost
 * tens of milliseconds and must not run twice per load.
 */

/** Device selection modes accepted by the engine. */
export type OnnxDevice = "auto" | "webgpu" | "wasm" | "cpu";

/** Concrete device after resolution (never `"auto"`). */
export type ResolvedDevice = "webgpu" | "wasm" | "cpu";

/** Dtype overrides accepted by the engine. */
export type OnnxDtype = "auto" | "fp32" | "q4" | "q4f16";

/** Concrete dtype after resolution (never `"auto"`). */
export type ResolvedDtype = "fp32" | "q4" | "q4f16";

/** A fully resolved device/dtype pair. */
export interface ResolvedCapabilities {
  readonly device: ResolvedDevice;
  readonly dtype: ResolvedDtype;
}

/** Why a device was chosen or rejected, so the engine can emit an honest event. */
export interface DeviceResolution extends ResolvedCapabilities {
  /** Set when `"auto"` wanted WebGPU but had to use WASM. */
  readonly fallbackFrom?: "webgpu";
  /** Human-readable reason for a fallback or a rejection. */
  readonly reason?: string;
}

/** Minimal `navigator.gpu` shape, to avoid a `@webgpu/types` dependency. */
interface NavigatorGpu {
  requestAdapter(options?: {
    powerPreference?: "low-power" | "high-performance";
  }): Promise<AdapterLike | null>;
}

interface AdapterLike {
  readonly features: { has(feature: string): boolean };
}

interface ProbeResult {
  readonly available: boolean;
  readonly shaderF16: boolean;
  /** Adapter description, when the browser exposes it. */
  readonly description?: string;
}

function getNavigatorGpu(): NavigatorGpu | undefined {
  if (typeof navigator === "undefined") {
    return undefined;
  }
  const gpu = (navigator as unknown as { gpu?: NavigatorGpu }).gpu;
  return typeof gpu?.requestAdapter === "function" ? gpu : undefined;
}

let adapterProbe: Promise<ProbeResult> | undefined;

/** Memoized adapter probe, including rejections. */
function probeAdapter(): Promise<ProbeResult> {
  adapterProbe ??= (async (): Promise<ProbeResult> => {
    const gpu = getNavigatorGpu();
    if (!gpu) {
      return { available: false, shaderF16: false };
    }
    try {
      const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
      if (!adapter) {
        return { available: false, shaderF16: false };
      }
      const info = (adapter as unknown as { info?: { description?: string; vendor?: string } })
        .info;
      const description = info?.description ?? info?.vendor;
      return {
        available: true,
        shaderF16: adapter.features.has("shader-f16"),
        ...(description !== undefined && { description }),
      };
    } catch {
      return { available: false, shaderF16: false };
    }
  })();
  return adapterProbe;
}

/** Whether a real WebGPU adapter answered the probe. */
export async function detectWebGpu(): Promise<boolean> {
  return (await probeAdapter()).available;
}

/** Whether the probed adapter can run `fp16` arithmetic (`shader-f16`). */
export async function isFp16Supported(): Promise<boolean> {
  return (await probeAdapter()).shaderF16;
}

/** Clears the memoized probe. Intended for tests; production never needs it. */
export function resetWebGpuProbe(): void {
  adapterProbe = undefined;
}

/**
 * Default dtype per device, taken from the M1 measurements rather than from file
 * size alone.
 *
 * - `wasm`/`cpu`: `fp32`. Every quantized variant was 20–26 % slower warm on CPU,
 *   so quantization buys download size and memory, not speed; the safe default stays
 *   full precision, and `q4` is a documented, cheaper opt-in.
 * - `webgpu`: `q4`. It is the smallest variant that does **not** require
 *   `shader-f16`, so it cannot fail on a capable-but-not-`f16` adapter.
 */
const DEFAULT_DTYPE_BY_DEVICE: Readonly<Record<ResolvedDevice, ResolvedDtype>> = {
  wasm: "fp32",
  cpu: "fp32",
  webgpu: "q4",
};

/** Environment defaults: Node.js uses `onnxruntime-node`, whose device name is `cpu`. */
function defaultDevice(): ResolvedDevice {
  return typeof Worker === "undefined" ? "cpu" : "wasm";
}

/**
 * Resolves `device`/`dtype` to concrete values.
 *
 * - `"wasm"` / `"cpu"` never probe anything.
 * - `"webgpu"` throws when there is no adapter, and throws for `q4f16` when the
 *   adapter lacks `shader-f16`. There is no silent retry: the caller asked for a
 *   specific backend, and a fallback would hide a wrong assumption behind working
 *   audio.
 * - `"auto"` prefers WebGPU, reports a fallback (`fallbackFrom`) when it has to use
 *   WASM, and keeps an explicit dtype only when the resulting device can honor it.
 */
export async function resolveDeviceDtype(
  device: OnnxDevice = "auto",
  dtype: OnnxDtype = "auto",
): Promise<DeviceResolution> {
  const wanted = dtype === "auto" ? undefined : dtype;

  if (device === "wasm" || device === "cpu") {
    return { device, dtype: wanted ?? DEFAULT_DTYPE_BY_DEVICE[device] };
  }

  const probe = await probeAdapter();

  if (device === "webgpu") {
    if (!probe.available) {
      throw new Error(
        'WebGPU was requested but no adapter is available. Use device: "auto" for a documented ' +
          'WASM fallback, or device: "wasm" to skip probing entirely.',
      );
    }
    const resolved = wanted ?? DEFAULT_DTYPE_BY_DEVICE.webgpu;
    if (resolved === "q4f16" && !probe.shaderF16) {
      throw new Error(
        'dtype "q4f16" requires the WebGPU "shader-f16" feature, which this adapter lacks. ' +
          'Use dtype: "q4" or dtype: "fp32", or device: "wasm".',
      );
    }
    return { device: "webgpu", dtype: resolved };
  }

  const fallbackDevice = defaultDevice();
  if (!probe.available) {
    return {
      device: fallbackDevice,
      dtype: wanted ?? DEFAULT_DTYPE_BY_DEVICE[fallbackDevice],
      fallbackFrom: "webgpu",
      reason: "no WebGPU adapter",
    };
  }
  // `q4f16` is only safe with `shader-f16`; on an adapter without it, `auto` keeps
  // running on the GPU with the smallest dtype that does not need the feature instead
  // of failing a request the caller marked as "whatever works".
  if (wanted === "q4f16" && !probe.shaderF16) {
    return {
      device: "webgpu",
      dtype: DEFAULT_DTYPE_BY_DEVICE.webgpu,
      reason: 'q4f16 needs "shader-f16"; using q4 instead',
    };
  }
  return { device: "webgpu", dtype: wanted ?? DEFAULT_DTYPE_BY_DEVICE.webgpu };
}
