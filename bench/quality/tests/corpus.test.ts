/**
 * The corpus rules that make a quality report comparable with the previous one.
 *
 * A corpus is only useful if it is stable and if every case states what it is probing. Both are
 * checked here, because adding a case silently would invalidate every earlier report without
 * anything failing.
 */
import { describe, expect, it } from "vitest";

import { CORPUS, corpusCase } from "../src/corpus.js";

describe("corpus", () => {
  it("has unique ids, so a report case can be looked up unambiguously", () => {
    const ids = CORPUS.map((entry) => entry.id);
    expect(new Set(ids).size).toBe(ids.length);
  });

  it("states what every case probes", () => {
    for (const entry of CORPUS) {
      expect(entry.probes.length, `case ${entry.id} must say what it probes`).toBeGreaterThan(20);
      expect(entry.text.trim().length).toBeGreaterThan(0);
      expect(["supported", "known-limitation"]).toContain(entry.expectation);
    }
  });

  it("covers both a supported baseline and the documented limitations", () => {
    // A corpus of only well-behaved sentences would hide the byte-fallback limitation the M1 spike
    // recorded; a corpus of only broken ones would not show what does work.
    expect(CORPUS.some((entry) => entry.expectation === "supported")).toBe(true);
    expect(CORPUS.some((entry) => entry.expectation === "known-limitation")).toBe(true);
  });

  it("includes the constructs the profile is documented to mishandle", () => {
    const ids = new Set(CORPUS.map((entry) => entry.id));
    for (const required of ["digits", "currency", "punctuation", "long-unpunctuated"]) {
      expect(ids.has(required), `missing corpus case "${required}"`).toBe(true);
    }
  });

  it("looks a case up by id, and returns undefined for an unknown one", () => {
    expect(corpusCase("digits")?.expectation).toBe("known-limitation");
    expect(corpusCase("no-such-case")).toBeUndefined();
  });
});
