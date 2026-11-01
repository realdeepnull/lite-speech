/**
 * The family seam and the catalogue.
 *
 * Two properties carry the extensibility claim, so they are tested directly rather than only
 * through a synthesis: a *new* family runs without touching the engine, and a *new language*
 * resolves without touching the registry implementation. The Phonon profile's text rules and
 * frame budget are tested here too, because they are the only family-specific behaviour that is
 * pure enough to assert on without a runtime.
 */
import { describe, expect, it } from "vitest";

import {
  assetUrl,
  createRegistry,
  languagesOf,
  selectVariantFiles,
  variantOf,
} from "../src/catalog.js";
import {
  resolveRunner,
  unknownFamilyError,
  type FamilyRegistry,
  type RunnerLoadOptions,
} from "../src/families.js";
import {
  BUILT_IN_MODELS,
  BUILT_IN_VOICES,
  PHONON_MODEL,
  PHONON_VOICES_MANIFEST,
  PHONON_VOICES,
  createDefaultRegistry,
  voiceManifest,
} from "../src/models.js";
import {
  PHONON_FAMILY,
  PHONON_PROFILE,
  normalizeSegment,
  phononMaxFrames,
  type PhononConfig,
} from "../src/phonon-profile.js";
import { FAKE_CONFIG, createFakeLoader } from "./helpers/fake-loader.js";

const config: PhononConfig = FAKE_CONFIG;

describe("phonon family", () => {
  it("declares true incremental frames, not a buffered approximation", () => {
    expect(PHONON_PROFILE.streaming).toBe("incremental-frames");
    expect(PHONON_PROFILE.frameRate).toBe(12.5);
    expect(PHONON_PROFILE.id).toBe(PHONON_FAMILY);
  });

  it("is the family every built-in model resolves to", () => {
    for (const model of BUILT_IN_MODELS) {
      expect(model.family).toBe(PHONON_FAMILY);
    }
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

  it("is language-neutral, so a non-English segment is not mangled", () => {
    // The rule touches only whitespace and the first code point: it does not transliterate, strip
    // diacritics, or insert language-specific punctuation.
    expect(normalizeSegment("grüße aus münchen")).toBe("Grüße aus münchen.");
    expect(normalizeSegment("¿qué hora es")).toBe("¿qué hora es.");
    expect(normalizeSegment("日本語のテキスト")).toBe("日本語のテキスト.");
  });

  it("bounds frames for a long segment by its token estimate", () => {
    const long = "a".repeat(300);
    const budget = phononMaxFrames(long, 100, config);
    // (100 / 3 + 2) * 12.5 ≈ 442 frames, plus the one-frame long-segment tail.
    expect(budget).toBeGreaterThan(400);
    expect(budget).toBeLessThan(500);
  });

  it("reserves extra tail frames for a very short segment", () => {
    const short = phononMaxFrames("Hi", 4, config);
    const long = phononMaxFrames("one two three four five six", 30, config);
    expect(short - Math.ceil((4 / 3 + 2) * 12.5)).toBe(3);
    expect(long - Math.ceil((30 / 3 + 2) * 12.5)).toBe(1);
  });

  it("never returns a budget below one frame", () => {
    expect(phononMaxFrames("", 0, config)).toBeGreaterThanOrEqual(1);
  });
});

describe("catalogue", () => {
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

  it("carries name and gender in the manifest, so the engine holds no voice table", () => {
    for (const voice of PHONON_VOICES_MANIFEST) {
      expect(voice.name).toBeTypeOf("string");
      expect(voice.gender).toBeTypeOf("string");
      expect(voice.description).toMatch(/^Accent /u);
    }
    expect(PHONON_VOICES_MANIFEST.length).toBe(PHONON_VOICES.length);
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
});

describe("catalogue extensibility", () => {
  it("registers a second language without touching the registry implementation", () => {
    // A German model plus a German voice. Nothing here is Phonon-specific: the registry only reads
    // manifests, which is what makes "more languages" a data change.
    const german = {
      ...PHONON_MODEL,
      id: "mirror/phonon-de",
      revision: "a".repeat(40),
      languages: ["de"],
    };
    const germanVoice = voiceManifest(german, PHONON_VOICES[0]!, "de");
    const registry = createRegistry({ models: [PHONON_MODEL, german], voices: [germanVoice] });

    expect(registry.languages()).toEqual(["en", "de"]);
    expect(registry.model("de")?.id).toBe("mirror/phonon-de");
    expect(registry.model("de-AT")?.id).toBe("mirror/phonon-de");
    // The English model still resolves to English; adding a language did not shadow it.
    expect(
      createRegistry({ models: [PHONON_MODEL, german], voices: [germanVoice] }).model("en"),
    ).toBe(PHONON_MODEL);
  });

  it("prefers an exact language match over a primary-subtag fallback", () => {
    const generic = { ...PHONON_MODEL, id: "m/generic", languages: ["de"] };
    const austrian = { ...PHONON_MODEL, id: "m/at", languages: ["de-AT"] };
    const registry = createRegistry({ models: [generic, austrian], voices: [] });

    expect(registry.model("de-AT")?.id).toBe("m/at");
    expect(registry.model("de")?.id).toBe("m/generic");
    // `de-CH` has no exact match, so it falls back to the primary subtag rather than failing.
    expect(registry.model("de-CH")?.id).toBe("m/generic");
  });

  it("derives languages from voices when a manifest does not declare any", () => {
    const bare: typeof PHONON_MODEL = { ...PHONON_MODEL };
    delete (bare as { languages?: readonly string[] }).languages;
    const voice = voiceManifest(PHONON_MODEL, PHONON_VOICES[1]!, "es");
    expect(languagesOf(bare, [voice])).toEqual(["es"]);
  });

  it("accepts a custom model list and keeps the voices that belong to it", () => {
    const custom = createRegistry({
      models: [{ id: "mirror/model", revision: "0".repeat(40), files: [], languages: ["en"] }],
      voices: [],
    });
    expect(custom.model("en")?.id).toBe("mirror/model");
    expect(custom.voices(custom.model("en")!)).toEqual([]);
  });

  it("keeps every built-in voice attached to the built-in model", () => {
    const registry = createDefaultRegistry();
    const model = registry.model("en")!;
    expect(registry.voices(model)).toHaveLength(BUILT_IN_VOICES.length);
    for (const voice of BUILT_IN_VOICES) {
      expect(voice.modelId).toBe(PHONON_MODEL.id);
    }
  });
});

describe("family registry", () => {
  it("resolves a registered family", () => {
    const { families } = createFakeLoader();
    expect(resolveRunner(PHONON_FAMILY, families)).toBeTypeOf("function");
  });

  it("returns undefined for an unregistered family instead of guessing", () => {
    expect(resolveRunner("some-other-family", {})).toBeUndefined();
    expect(resolveRunner(undefined, {})).toBeUndefined();
  });

  it("names the available families in the error", () => {
    const { families } = createFakeLoader();
    expect(unknownFamilyError("nope", families).message).toMatch(/Known families: phonon/u);
  });

  it("runs a second family through the same registry without an engine change", async () => {
    // This is the extensibility claim made executable: a factory under another id receives the
    // same load options and produces a runner, so a new architecture needs no change to the host,
    // the protocol, or core.
    const { families } = createFakeLoader({ frames: 2 });
    const seen: RunnerLoadOptions[] = [];
    const extended: FamilyRegistry = {
      ...families,
      "other-family": (options) => {
        seen.push(options);
        return families[PHONON_FAMILY]!(options);
      },
    };

    const factory = resolveRunner("other-family", extended);
    expect(factory).toBeTypeOf("function");

    const runner = await factory!({
      modelId: "some/other-model",
      revision: "b".repeat(40),
      family: "other-family",
      device: "wasm",
      dtype: "q4",
      voice: {
        id: "speaker",
        language: "es",
        repository: "some/other-model",
        revision: "b".repeat(40),
        path: "voices/speaker.bin",
      },
      onProgress: () => undefined,
    });

    // The runner it produced is indistinguishable from a built-in one to the host.
    expect(seen[0]?.family).toBe("other-family");
    expect(runner.profile.id).toBe("other-family");
    const frames: Float32Array[] = [];
    for await (const frame of runner.synthesize("hola", { isCancelled: () => false })) {
      frames.push(frame);
    }
    expect(frames).toHaveLength(2);
    await runner.dispose();
  });
});

describe("asset variants", () => {
  it("reads the declared variant, so a file may be named anything", () => {
    expect(variantOf({ path: "weights/whatever.bin", variant: "q4" })).toBe("q4");
  });

  it("falls back to the ONNX naming convention for a hand-written manifest", () => {
    expect(variantOf({ path: "onnx/flow_lm_q4.onnx" })).toBe("q4");
    expect(variantOf({ path: "onnx/flow_lm_q4f16.onnx_data" })).toBe("q4f16");
    expect(variantOf({ path: "onnx/flow_lm.onnx" })).toBeUndefined();
    expect(variantOf({ path: "config.json" })).toBeUndefined();
  });

  it("keeps the files of the resolved variant plus everything variant-less", () => {
    const files = PHONON_MODEL.files.map((file) => ({ ...file }));
    for (const dtype of ["fp32", "q4", "q4f16"] as const) {
      const selected = selectVariantFiles(files, dtype).map((file) => file.path);
      expect(selected.filter((path) => path.startsWith("onnx/")).length).toBeGreaterThan(0);
      for (const path of selected.filter((candidate) => candidate.startsWith("onnx/"))) {
        expect(variantOf(files.find((file) => file.path === path)!)).toBe(dtype);
      }
      // The non-weight files are required by every configuration.
      expect(selected).toContain("config.json");
      expect(selected).toContain("tokenizer.json");
    }
  });

  it("marks every weight file in the built-in manifest with a variant", () => {
    const unlabelled = PHONON_MODEL.files.filter(
      (file) => file.path.startsWith("onnx/") && file.variant === undefined,
    );
    expect(unlabelled).toEqual([]);
  });
});
