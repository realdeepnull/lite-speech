import { defineConfig } from "tsdown";

// A Node-only build: the corpus sweep drives the real engine on `onnxruntime-node`, and the
// packages under test stay external so the harness measures what ships.
export default defineConfig({
  entry: { quality: "src/quality.ts" },
  platform: "node",
  format: ["esm"],
  target: "node22",
  dts: false,
  deps: {
    neverBundle: ["@lite-speech/core", "@lite-speech/engine-onnx", "@lite-speech/engine-onnx/node"],
  },
});
