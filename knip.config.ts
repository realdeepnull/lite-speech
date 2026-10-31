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
      entry: ["index.html"],
    },
    "bench/benchmark": {
      entry: [],
    },
    "bench/quality": {
      entry: [],
    },
  },
};

export default config;
