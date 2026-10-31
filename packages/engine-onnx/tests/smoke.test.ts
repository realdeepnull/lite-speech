import { describe, expect, it } from "vitest";

import { PACKAGE_NAME } from "../src/index";

describe("@lite-speech/engine-onnx", () => {
  it("exposes its entry point", () => {
    expect(PACKAGE_NAME).toBe("@lite-speech/engine-onnx");
  });
});
