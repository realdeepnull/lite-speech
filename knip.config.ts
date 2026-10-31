import type { KnipConfig } from "knip";

const config: KnipConfig = {
  workspaces: {
    ".": {
      project: ["*.{js,ts}"],
    },
    "packages/core": {
      project: ["src/**/*.ts", "tests/**/*.ts"],
    },
    "packages/engine-onnx": {
      project: ["src/**/*.ts", "tests/**/*.ts"],
    },
    "examples/demo": {
      // The demo is bundler-free: `index.html` loads `main.js` via a `<script type="module">`,
      // and `serve.mjs` is only invoked by the package script, so neither is an import edge.
      entry: ["index.html", "main.js"],
    },
    "bench/benchmark": {
      entry: [],
    },
    "bench/quality": {
      entry: [],
    },
  },
  // The protocol module is a type-level contract shared by two build entries; a response type is
  // "unused" when nobody names it, even though the union containing it is what the worker and the
  // engine agree on.
  ignoreIssues: {
    "packages/engine-onnx/src/worker-protocol.ts": ["exports", "types"],
    "packages/engine-onnx/src/phonon.ts": ["types"],
    "packages/engine-onnx/tests/helpers/fake-loader.ts": ["types"],
  },
};

export default config;
