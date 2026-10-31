import js from "@eslint/js";
import { defineConfig, globalIgnores } from "eslint/config";
import tseslint from "typescript-eslint";

/**
 * The spike is a plain-JavaScript, throwaway experiment: Node.js scripts plus one
 * bundler-free browser page. It is not part of any package's type-checked build, so the
 * environment globals are declared explicitly instead of adding a `globals` dependency.
 */
const nodeGlobals = {
  console: "readonly",
  process: "readonly",
  URL: "readonly",
  fetch: "readonly",
  performance: "readonly",
  TextEncoder: "readonly",
  // `probe-devices.mjs` reads `navigator` defensively to report that Node.js has no GPU.
  navigator: "readonly",
};

const browserGlobals = {
  console: "readonly",
  document: "readonly",
  navigator: "readonly",
  performance: "readonly",
  fetch: "readonly",
  TextEncoder: "readonly",
  AudioContext: "readonly",
};

export default defineConfig(
  js.configs.recommended,
  tseslint.configs.recommended,
  globalIgnores([
    "**/dist/**",
    "**/node_modules/**",
    "**/coverage/**",
    "spike/out/**",
    "spike/.cache/**",
  ]),
  {
    rules: {
      "@typescript-eslint/no-explicit-any": "warn",
      "@typescript-eslint/consistent-type-imports": [
        "error",
        { prefer: "type-imports", fixStyle: "inline-type-imports" },
      ],
    },
  },
  {
    files: ["spike/**/*.mjs", "spike/**/*.js"],
    languageOptions: {
      ecmaVersion: 2023,
      sourceType: "module",
      globals: nodeGlobals,
    },
  },
  {
    files: ["spike/browser/**/*.js"],
    languageOptions: {
      globals: browserGlobals,
    },
  },
);
