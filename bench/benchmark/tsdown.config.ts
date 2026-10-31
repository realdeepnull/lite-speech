import { defineConfig } from "tsdown";

// Two builds for two runtimes, the same split the library itself uses:
//
// - `browser`: the module `page.html` loads. The two library packages stay external and are
//   resolved by the page's import map, so the benchmark consumes the built ESM output exactly like
//   an application would.
// - `node`: the runners and the driver. They import `node:` builtins and `playwright`, so a Node
//   build (rather than `--experimental-strip-types`) is what makes them runnable: Node's type
//   stripping requires `.ts` specifiers, which the repository's `.js`-style imports do not use.
export default defineConfig([
  {
    entry: { browser: "src/browser.ts" },
    platform: "browser",
    format: ["esm"],
    target: "es2022",
    dts: false,
    deps: {
      neverBundle: ["@lite-speech/core", "@lite-speech/engine-onnx"],
    },
  },
  {
    entry: { node: "src/node.ts", drive: "src/drive.ts", render: "src/render.ts" },
    platform: "node",
    format: ["esm"],
    target: "node22",
    dts: false,
    deps: {
      // The packages under measurement are the workspace's own published artifacts, and the model
      // runtime is a dependency of the engine, not of the harness: bundling either would benchmark
      // a copy instead of the thing that ships.
      neverBundle: [
        "playwright",
        "@lite-speech/core",
        "@lite-speech/engine-onnx",
        "@lite-speech/engine-onnx/node",
        "@huggingface/transformers",
      ],
    },
  },
]);
