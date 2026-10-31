/**
 * Profile seam and registry.
 *
 * The profile is where model-family knowledge lives, so its text rules and frame budget are
 * tested directly rather than only through a synthesis. The registry is tested for the one
 * property that matters to `createTts()`: every registered voice resolves for its language, and
 * an unknown voice resolves to nothing instead of to a default.
 */
import { describe, expect, it } from "vitest";

import {
  PHONON_VOICES_MANIFEST,
  PHONON_VOICE_GENDERS,
  PHONON_VOICE_NAMES,
  assetUrl,
  createDefaultRegistry,
} from "../src/models.js";
import {
  DEFAULT_PROFILES,
  PHONON_PROFILE,
  PHONON_PROFILE_ID,
  normalizeSegment,
  resolveProfile,
  type PhononConfig,
} from "../src/profiles.js";
import { FAKE_CONFIG } from "./helpers/fake-loader.js";

const config: PhononConfig = FAKE_CONFIG;

describe("phonon profile", () => {
  it("declares true incremental frames, not a buffered approximation", () => {
    expect(PHONON_PROFILE.streaming).toBe("incremental-frames");
    expect(PHONON_PROFILE.frameRate).toBe(12.5);
    expect(DEFAULT_PROFILES[PHONON_PROFILE_ID]).toBe(PHONON_PROFILE);
  });

  it("resolves a missing profile id to the built-in profile", () => {
    expect(resolveProfile(undefined)).toBe(PHONON_PROFILE);
  });

  it("returns undefined for an unregistered profile instead of guessing", () => {
    expect(resolveProfile("some-other-family")).toBeUndefined();
  });

  it("finds an explicitly registered profile", () => {
    const custom = { ...PHONON_PROFILE, id: "custom" };
    expect(resolveProfile("custom", { custom })).toBe(custom);
  });

  it("capitalizes and terminates a segment", () => {
    expect(normalizeSegment("hello world")).toBe("Hello world.");
    expect(normalizeSegment("  hello   world  ")).toBe("Hello world.");
  });

  it("does not add a period to a segment that already ends in punctuation", () => {
    expect(normalizeSegment("Hello world?")).toBe("Hello world?");
    expect(normalizeSegment("Wait!")).toBe("Wait!");
    expect(normalizeSegment("A list:")).toBe("A list:");
  });

  it("keeps a trailing digit, which the model reads as unfinished", () => {
    expect(normalizeSegment("chapter 3")).toBe("Chapter 3.");
  });

  it("returns an empty string for an empty segment", () => {
    expect(normalizeSegment("   ")).toBe("");
  });

  it("does not mangle a leading surrogate pair", () => {
    expect(normalizeSegment("😀 smile")).toBe("😀 smile.");
  });

  it("bounds frames for a long segment by its token estimate", () => {
    const long = "a".repeat(300);
    const budget = PHONON_PROFILE.maxFrames(long, 100, config);
    // (100 / 3 + 2) * 12.5 ≈ 442 frames, plus the one-frame long-segment tail.
    expect(budget).toBeGreaterThan(400);
    expect(budget).toBeLessThan(500);
  });

  it("reserves extra tail frames for a very short segment", () => {
    const short = PHONON_PROFILE.maxFrames("Hi", 4, config);
    const long = PHONON_PROFILE.maxFrames("one two three four five six", 30, config);
    expect(short - Math.ceil((4 / 3 + 2) * 12.5)).toBe(3);
    expect(long - Math.ceil((30 / 3 + 2) * 12.5)).toBe(1);
  });

  it("never returns a budget below one frame", () => {
    expect(PHONON_PROFILE.maxFrames("", 0, config)).toBeGreaterThanOrEqual(1);
  });
});

describe("default registry", () => {
  const registry = createDefaultRegistry();

  it("resolves the pinned Phonon model for English only", () => {
    expect(registry.model("en")?.id).toBe("onnx-community/phonon-31m-en-ONNX");
    expect(registry.model("de")).toBeUndefined();
  });

  it("pins an immutable revision, never a branch name", () => {
    const model = registry.model("en");
    expect(model?.revision).toMatch(/^[0-9a-f]{40}$/);
    expect(model?.revision).not.toBe("main");
  });

  it("declares the license for the model and every voice", () => {
    const model = registry.model("en");
    expect(model?.license).toBe("Apache-2.0");
    for (const voice of PHONON_VOICES_MANIFEST) {
      expect(voice.license).toBe("Apache-2.0");
    }
  });

  it("resolves the first voice when none is named", () => {
    const model = registry.model("en")!;
    expect(registry.voice(model, undefined)?.id).toBe("default");
  });

  it("resolves a named voice", () => {
    const model = registry.model("en")!;
    expect(registry.voice(model, "LMDIYE0Oa377Qh6Z")?.id).toBe("LMDIYE0Oa377Qh6Z");
  });

  it("resolves an unknown voice to nothing, never to a default", () => {
    const model = registry.model("en")!;
    expect(registry.voice(model, "nope")).toBeUndefined();
  });

  it("labels every registered voice with a name and a gender", () => {
    for (const voice of PHONON_VOICES_MANIFEST) {
      expect(PHONON_VOICE_NAMES[voice.id]).toBeTypeOf("string");
      expect(PHONON_VOICE_GENDERS[voice.id]).toBeTypeOf("string");
    }
  });

  it("declares a sample rate and one embedding file per voice", () => {
    for (const voice of PHONON_VOICES_MANIFEST) {
      expect(voice.sampleRate).toBe(24_000);
      expect(voice.files).toHaveLength(1);
      expect(voice.files[0]?.size).toBe(256_000);
    }
  });

  it("builds URLs at the pinned revision", () => {
    expect(assetUrl("m/r", "abc", "onnx/flow_lm.onnx")).toBe(
      "https://huggingface.co/m/r/resolve/abc/onnx/flow_lm.onnx",
    );
  });

  it("accepts a custom model map", () => {
    const custom = createDefaultRegistry({
      en: { id: "mirror/model", revision: "0".repeat(40), files: [] },
    });
    expect(custom.model("en")?.id).toBe("mirror/model");
    expect(custom.voices(custom.model("en")!)).toEqual([]);
  });
});
