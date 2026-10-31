import { defineConfig } from "vitest/config";

// The harness's pure logic — statistics, signal comparison, corpus invariants — is worth a unit
// suite: a percentile definition or an off-by-one in a correlation is exactly the kind of bug that
// would make every published number subtly wrong while every run still "passes".
export default defineConfig({
  test: {
    include: ["tests/**/*.test.ts"],
    environment: "node",
  },
});
