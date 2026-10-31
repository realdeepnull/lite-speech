import { defineConfig } from "tsdown";

export default defineConfig({
  entry: ["src/index.ts"],
  platform: "browser",
  format: ["esm"],
  // Browser baseline, independent of the Node floor in `engines`.
  target: "es2022",
  dts: true,
});
