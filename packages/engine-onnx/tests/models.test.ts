/**
 * Manifest and registry unit tests.
 *
 * Everything here is pure data plus pure functions, so it runs in Node without a runtime, a
 * browser, or a network. The point is to pin the *facts* a consent dialog relies on: sizes,
 * digests, and the exact URLs every lookup resolves to.
 */
import { describe, expect, it } from "vitest";

import { PHONON_MODEL, PHONON_VOICES_MANIFEST } from "../src/models.js";
import { assetUrl, hubRevisionTemplate } from "../src/catalog.js";

/** The pinned commit this repository's measurements were taken at. */
const REVISION = "4f0cda7c60dddb4862d0e68ff9ff60c32ffdc9f4";

describe("PHONON_MODEL", () => {
  it("pins an immutable revision, never `main`", () => {
    expect(PHONON_MODEL.revision).toBe(REVISION);
    expect(PHONON_MODEL.revision).toMatch(/^[0-9a-f]{40}$/u);
  });

  it("declares a size and a digest for every file", () => {
    const problems = PHONON_MODEL.files.flatMap((file) => [
      ...(file.size === undefined ? [`${file.path} has no size`] : []),
      ...(/^[0-9a-f]{64}$/u.test(file.sha256 ?? "") ? [] : [`${file.path} has no sha256`]),
    ]);
    expect(problems).toEqual([]);
  });

  it("declares one distinct digest per distinct file", () => {
    // `mimi_decoder_q4.onnx` and `mimi_decoder.onnx` are the same blob at this revision, so equal
    // digests are expected — but a digest must never be shared by files with different sizes.
    const byDigest = new Map<string, string[]>();
    for (const file of PHONON_MODEL.files) {
      const key = file.sha256 ?? "";
      byDigest.set(key, [...(byDigest.get(key) ?? []), file.path]);
    }
    const inconsistent = [...byDigest.values()].filter((paths) => {
      const sizes = new Set(
        paths.map((path) => PHONON_MODEL.files.find((file) => file.path === path)?.size),
      );
      return sizes.size > 1;
    });
    expect(inconsistent).toEqual([]);
  });

  it("ships every dtype variant the engine can resolve to", () => {
    const paths = new Set(PHONON_MODEL.files.map((file) => file.path));
    for (const path of [
      "onnx/flow_lm.onnx",
      "onnx/flow_lm_q4.onnx",
      "onnx/flow_lm_q4.onnx_data",
      "onnx/flow_lm_q4f16.onnx",
      "onnx/flow_lm_q4f16.onnx_data",
      "onnx/mimi_decoder.onnx",
      "onnx/mimi_decoder_q4.onnx",
      "onnx/mimi_decoder_q4f16.onnx",
      "onnx/mimi_decoder_q4f16.onnx_data",
    ]) {
      expect(paths.has(path)).toBe(true);
    }
  });

  it("carries the non-graph files every graph load needs", () => {
    const paths = new Set(PHONON_MODEL.files.map((file) => file.path));
    for (const path of ["config.json", "tokenizer.json", "tokenizer_config.json"]) {
      expect(paths.has(path)).toBe(true);
    }
  });
});

describe("PHONON_VOICES_MANIFEST", () => {
  it("declares a size and a digest for every voice", () => {
    expect(PHONON_VOICES_MANIFEST.length).toBeGreaterThan(0);
    const problems = PHONON_VOICES_MANIFEST.flatMap((voice) => {
      const file = voice.files[0];
      return [
        ...(file?.size === 256_000 ? [] : [`${voice.id} has no 256000-byte size`]),
        ...(/^[0-9a-f]{64}$/u.test(file?.sha256 ?? "") ? [] : [`${voice.id} has no sha256`]),
        ...(file?.path === `voices/${voice.id}.bin` ? [] : [`${voice.id} has a wrong path`]),
      ];
    });
    expect(problems).toEqual([]);
  });

  it("pins the same revision and license as the model", () => {
    for (const voice of PHONON_VOICES_MANIFEST) {
      expect(voice.revision).toBe(PHONON_MODEL.revision);
      expect(voice.license).toBe(PHONON_MODEL.license);
      expect(voice.language).toBe("en");
    }
  });

  it("uses one digest per voice, so a swapped embedding is detectable", () => {
    const digests = new Set(PHONON_VOICES_MANIFEST.map((voice) => voice.files[0]?.sha256));
    expect(digests.size).toBe(PHONON_VOICES_MANIFEST.length);
  });
});

describe("assetUrl()", () => {
  it("resolves a path at the pinned revision", () => {
    expect(assetUrl(PHONON_MODEL.id, PHONON_MODEL.revision, "config.json")).toBe(
      `https://huggingface.co/${PHONON_MODEL.id}/resolve/${REVISION}/config.json`,
    );
  });
});

describe("hubRevisionTemplate()", () => {
  it("pins the revision segment, so a probe without a revision resolves to the same commit", () => {
    expect(hubRevisionTemplate(REVISION)).toBe(`{model}/resolve/${REVISION}/`);
  });

  it("agrees with assetUrl() for every manifest file", () => {
    // This is the property that makes strict offline work in a browser: the template and the
    // manifest URLs must produce identical strings, because the Cache Storage key is the URL.
    const template = hubRevisionTemplate(PHONON_MODEL.revision);
    for (const file of PHONON_MODEL.files) {
      const fromTemplate = `https://huggingface.co/${template
        .replace("{model}", PHONON_MODEL.id)
        .replaceAll("{revision}", PHONON_MODEL.revision)}${file.path}`;
      expect(fromTemplate).toBe(assetUrl(PHONON_MODEL.id, PHONON_MODEL.revision, file.path));
    }
  });

  it("percent-encodes a revision that needs it", () => {
    expect(hubRevisionTemplate("refs/pr/1")).toBe("{model}/resolve/refs%2Fpr%2F1/");
  });
});
