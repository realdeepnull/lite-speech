import { defineConfig } from "vitest/config";

// The unit suite runs in Node. The browser suite (`tests/**/*.browser.test.ts`) has its own
// config because it downloads model weights and takes minutes; the fast default must not pay
// for it. `npm test` runs this one, `npm run test:browser` runs the other.
export default defineConfig({
  test: {
    include: ["tests/**/*.test.ts"],
    exclude: ["tests/**/*.browser.test.ts", "**/node_modules/**"],
    environment: "node",
  },
});
