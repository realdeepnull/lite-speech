import { playwright } from "@vitest/browser-playwright";
import { defineConfig } from "vitest/config";

// Browser integration suite: workers, Cache Storage, `navigator.gpu`, and real ONNX inference.
// These have no Node equivalent, so this is the only place the milestone's Chromium gate can be
// proven. Headless Chromium with a long timeout, because a cold run downloads ~68 MB and
// instantiates two graphs.
export default defineConfig({
  optimizeDeps: {
    include: ["@huggingface/transformers"],
  },
  test: {
    include: ["tests/**/*.browser.test.ts"],
    testTimeout: 900_000,
    hookTimeout: 900_000,
    browser: {
      enabled: true,
      provider: playwright(),
      instances: [{ browser: "chromium" }],
      headless: true,
    },
  },
});
