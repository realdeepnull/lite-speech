import { defineConfig } from "vitest/config";

// The harness's pure logic is tested here; the sweep itself needs the real model and runs through
// the CLI, so it is not part of a fast unit suite.
export default defineConfig({
  test: {
    include: ["tests/**/*.test.ts"],
    environment: "node",
  },
});
