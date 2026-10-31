import { defineConfig } from "tsdown";

export default defineConfig({
  entry: ["src/index.ts"],
  platform: "browser",
  format: ["esm"],
  // Browser baseline, independent of the Node floor in `engines`: the core must
  // not emit Node-specific target assumptions for consumers bundling for web.
  target: "es2022",
  dts: true,
});
