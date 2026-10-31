import { describe, expect, it } from "vitest";

import {
  DEFAULT_MAX_CHARS_PER_SEGMENT,
  TTS_ERROR_CODES,
  prepareText,
  segmentText,
} from "../src/index";

describe("prepareText()", () => {
  it("collapses whitespace and trims", () => {
    expect(prepareText("  Hello \n\t world  ")).toBe("Hello world");
  });

  it("preserves content — never truncates", () => {
    const text = "a".repeat(5_000);
    expect(prepareText(text)).toHaveLength(5_000);
  });

  it("can preserve line breaks", () => {
    expect(prepareText("first\nsecond", { collapseWhitespace: false })).toBe("first\nsecond");
  });

  it("returns an empty string for empty or whitespace-only input", () => {
    expect(prepareText("")).toBe("");
    expect(prepareText("   \n ")).toBe("");
  });

  it("rejects a non-string input", () => {
    expect(() => prepareText(42 as unknown as string)).toThrowError(
      expect.objectContaining({ code: TTS_ERROR_CODES.INVALID_ARGUMENT }),
    );
  });
});

describe("segmentText()", () => {
  it("returns no segments for empty input", () => {
    expect(segmentText("")).toEqual([]);
    expect(segmentText("   ")).toEqual([]);
  });

  it("splits at sentence terminators and keeps the punctuation", () => {
    expect(segmentText("Hello there. How are you? Fine!")).toEqual([
      "Hello there.",
      "How are you?",
      "Fine!",
    ]);
  });

  it("does not split at a dot that is not followed by whitespace", () => {
    expect(segmentText("Pi is 3.14 and e is 2.71.")).toEqual(["Pi is 3.14 and e is 2.71."]);
  });

  it("keeps closing quotes with the sentence", () => {
    expect(segmentText('He said "stop." Then he left.')).toEqual([
      'He said "stop."',
      "Then he left.",
    ]);
  });

  it("splits at semicolons and newlines", () => {
    expect(segmentText("one; two\nthree")).toEqual(["one;", "two", "three"]);
  });

  it("preserves every non-whitespace character", () => {
    const text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. ".repeat(20);
    const joined = segmentText(text, { maxCharsPerSegment: 40 }).join(" ").replace(/\s+/g, " ");
    expect(joined).toBe(text.trim().replace(/\s+/g, " "));
  });

  it("splits over-long segments to at most maxCharsPerSegment", () => {
    const text = "Alpha beta gamma delta epsilon zeta eta theta iota kappa.";
    const segments = segmentText(text, { maxCharsPerSegment: 20 });

    expect(segments.length).toBeGreaterThan(1);
    for (const segment of segments) {
      expect(segment.length).toBeLessThanOrEqual(20);
    }
    expect(segments.join(" ").replace(/\s+/g, " ")).toBe(text);
  });

  it("hard-splits a single word that exceeds the limit", () => {
    const word = "x".repeat(50);
    expect(segmentText(word, { maxCharsPerSegment: 10 })).toHaveLength(5);
    expect(segmentText(word, { maxCharsPerSegment: 10 }).join("")).toBe(word);
  });

  it("does not split surrogate pairs when hard-cutting", () => {
    const text = "😀".repeat(10);
    const segments = segmentText(text, { maxCharsPerSegment: 3 });
    expect(segments.join("")).toBe(text);
    for (const segment of segments) {
      expect([...segment]).toHaveLength(segment.length / 2);
    }
  });

  it("uses a documented default limit", () => {
    expect(DEFAULT_MAX_CHARS_PER_SEGMENT).toBe(200);
    expect(segmentText("short text")).toEqual(["short text"]);
  });

  it("rejects an invalid maxCharsPerSegment", () => {
    for (const value of [0, -1, 1.5, Number.NaN]) {
      expect(() => segmentText("text", { maxCharsPerSegment: value })).toThrowError(
        expect.objectContaining({ code: TTS_ERROR_CODES.INVALID_ARGUMENT }),
      );
    }
  });
});
