/**
 * Device and dtype policy.
 *
 * These rules are the ones the milestone gate names explicitly: an explicit `"webgpu"` must
 * throw instead of falling back, `"auto"` must report a fallback, and `q4f16` must be refused
 * without `shader-f16`. The probe is memoized including rejections, so a fake `navigator.gpu` is
 * installed per test together with a reset.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  detectWebGpu,
  isFp16Supported,
  resetWebGpuProbe,
  resolveDeviceDtype,
} from "../src/webgpu.js";

/** Installs a `navigator.gpu` stub; `null` adapter means "API present, no GPU". */
function installGpu(
  mode: "absent" | "no-adapter" | "adapter" | "adapter-without-f16" | "throws",
  options: { vendor?: string } = {},
): { requestAdapter: ReturnType<typeof vi.fn> } {
  const requestAdapter = vi.fn(async () => {
    switch (mode) {
      case "absent":
        return null;
      case "no-adapter":
        return null;
      case "throws":
        throw new Error("adapter creation failed");
      case "adapter":
      case "adapter-without-f16":
        return {
          features: { has: (feature: string) => feature === "shader-f16" && mode === "adapter" },
          info: { vendor: options.vendor ?? "test" },
        };
    }
  });
  const gpu = mode === "absent" ? undefined : { requestAdapter };
  vi.stubGlobal("navigator", gpu === undefined ? {} : { gpu });
  resetWebGpuProbe();
  return { requestAdapter };
}

describe("webgpu policy", () => {
  beforeEach(() => {
    resetWebGpuProbe();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    resetWebGpuProbe();
  });

  it("never probes for an explicit wasm or cpu device", async () => {
    const { requestAdapter } = installGpu("adapter");
    await expect(resolveDeviceDtype("wasm")).resolves.toEqual({ device: "wasm", dtype: "fp32" });
    await expect(resolveDeviceDtype("cpu")).resolves.toEqual({ device: "cpu", dtype: "fp32" });
    expect(requestAdapter).not.toHaveBeenCalled();
  });

  it("throws for an explicit webgpu request without an adapter", async () => {
    installGpu("no-adapter");
    await expect(resolveDeviceDtype("webgpu")).rejects.toThrow(/no adapter is available/);
  });

  it("throws for an explicit webgpu request when the probe itself fails", async () => {
    installGpu("throws");
    await expect(resolveDeviceDtype("webgpu")).rejects.toThrow(/no adapter is available/);
  });

  it("uses webgpu when an adapter answers", async () => {
    installGpu("adapter");
    await expect(resolveDeviceDtype("webgpu")).resolves.toEqual({ device: "webgpu", dtype: "q4" });
  });

  it("falls back to the environment default and says so for auto without an adapter", async () => {
    installGpu("no-adapter");
    // Node has no `Worker`, so the environment default is `cpu`; a browser would get `wasm`.
    const resolved = await resolveDeviceDtype("auto");
    expect(resolved).toMatchObject({
      device: "cpu",
      dtype: "fp32",
      fallbackFrom: "webgpu",
      reason: "no WebGPU adapter",
    });
  });

  it("does not report a fallback when auto finds an adapter", async () => {
    installGpu("adapter");
    const resolved = await resolveDeviceDtype("auto");
    expect(resolved.device).toBe("webgpu");
    expect(resolved.fallbackFrom).toBeUndefined();
  });

  it("refuses q4f16 without shader-f16 on an explicit webgpu request", async () => {
    installGpu("adapter-without-f16");
    await expect(resolveDeviceDtype("webgpu", "q4f16")).rejects.toThrow(/shader-f16/);
  });

  it("downgrades q4f16 to q4 for auto without shader-f16", async () => {
    installGpu("adapter-without-f16");
    const resolved = await resolveDeviceDtype("auto", "q4f16");
    expect(resolved).toMatchObject({ device: "webgpu", dtype: "q4" });
  });

  it("keeps q4f16 for auto when the adapter supports it", async () => {
    installGpu("adapter");
    await expect(resolveDeviceDtype("auto", "q4f16")).resolves.toMatchObject({
      device: "webgpu",
      dtype: "q4f16",
    });
  });

  it("honours an explicit dtype on wasm without a probe", async () => {
    installGpu("adapter");
    await expect(resolveDeviceDtype("wasm", "q4")).resolves.toEqual({
      device: "wasm",
      dtype: "q4",
    });
  });

  it("memoizes the adapter probe including its cost", async () => {
    const { requestAdapter } = installGpu("adapter");
    expect(await detectWebGpu()).toBe(true);
    expect(await detectWebGpu()).toBe(true);
    expect(await isFp16Supported()).toBe(true);
    expect(requestAdapter).toHaveBeenCalledTimes(1);
  });

  it("memoizes a failed probe instead of retrying per call", async () => {
    const { requestAdapter } = installGpu("throws");
    expect(await detectWebGpu()).toBe(false);
    expect(await detectWebGpu()).toBe(false);
    expect(requestAdapter).toHaveBeenCalledTimes(1);
  });

  it("reports no WebGPU when the API is missing", async () => {
    installGpu("absent");
    expect(await detectWebGpu()).toBe(false);
    await expect(resolveDeviceDtype("auto")).resolves.toMatchObject({
      device: "cpu",
      fallbackFrom: "webgpu",
    });
  });
});
