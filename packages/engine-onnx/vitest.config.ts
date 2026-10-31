import { defineConfig } from "vitest/config";

// M0 keeps the placeholder suite in Node so the workspace has no browser
// tooling yet. M3 switches this to Vitest browser mode once the worker,
// Cache Storage, and WebGPU code exists.
export default defineConfig({
  test: {
    include: ["tests/**/*.test.ts"],
    environment: "node",
  },
});
