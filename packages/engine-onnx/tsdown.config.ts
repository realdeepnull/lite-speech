import { defineConfig } from "tsdown";

// Two builds, because the two runtimes disagree about the platform:
//
// - `platform: "browser"` for the main-thread entry and the worker. `@huggingface/transformers` is
//   bundled into `worker.js` because a browser cannot resolve a bare specifier inside a worker,
//   and `@lite-speech/core` is bundled so the ESM output runs without a bundler.
// - `platform: "node"` for the Node entry. Node resolves bare specifiers, so the model runtime
//   stays an ordinary dependency there instead of being inlined into a 1.5 MB file; only the
//   `node:` builtins must remain external, which they are by default.
export default defineConfig([
  {
    entry: { index: "src/index.ts", worker: "src/worker.ts" },
    platform: "browser",
    format: ["esm"],
    // Browser baseline, independent of the Node floor in `engines`.
    target: "es2022",
    dts: true,
    deps: {
      alwaysBundle: ["@lite-speech/core", "@huggingface/transformers"],
    },
  },
  {
    entry: { node: "src/node.ts" },
    platform: "node",
    format: ["esm"],
    target: "node22",
    dts: true,
    // The browser build already emitted `.js`; pin the Node build to `.mjs` so `exports` stays
    // unambiguous without a second `package.json` "type" override.
    outExtensions: () => ({ js: ".mjs", dts: ".d.mts" }),
    deps: {
      alwaysBundle: ["@lite-speech/core"],
    },
  },
]);
