/**
 * The fixed corpus the quality harness measures.
 *
 * Fixed, and committed, because a quality claim is only comparable to itself: if every run picked
 * its own sentences, "no regression" would mean nothing. The cases are chosen for what the profile
 * is known to handle badly rather than for what it does well — the M1 spike documented that the
 * pinned tokenizer applies no Unigram byte fallback, so a character outside the vocabulary degrades
 * to `<unk>`. A corpus of well-behaved English would hide exactly that.
 *
 * Each case states what it is probing, so a reader of the report can tell a real defect from an
 * expected limitation.
 */

/** One measured sentence. */ export interface CorpusCase {
  /** Stable identifier, used as the file name of the per-case audio. */
  readonly id: string;
  /** The text as the caller would pass it to `synthesize()` or `stream()`. */
  readonly text: string;
  /** What this case is meant to reveal. */
  readonly probes: string;
  /**
   * Whether a difference against the `fp32` reference is expected to be audible.
   *
   * `false` marks a known limitation rather than a defect: the harness reports it instead of
   * failing on it, so the limitation stays visible without pretending it can be fixed by dtype.
   */
  readonly expectation: "supported" | "known-limitation";
}

/**
 * The corpus.
 *
 * Ordered by length so a reader can see the latency/quality trade-off per length class. It is
 * deliberately short — nine cases keep a full sweep reproducible in minutes on a laptop, which is
 * what makes it likely to be re-run.
 */
export const CORPUS: readonly CorpusCase[] = [
  {
    id: "short-declarative",
    text: "The kettle is on.",
    probes: "Shortest supported input: one clause, one sentence terminator.",
    expectation: "supported",
  },
  {
    id: "short-question",
    text: "Is the window open?",
    probes: "Question intonation and the `?` terminator.",
    expectation: "supported",
  },
  {
    id: "medium-two-sentences",
    text: "The library closes at six. We should leave before then.",
    probes:
      "Two sentences in one call: the seam between them is where a per-segment model can clip.",
    expectation: "supported",
  },
  {
    id: "long-paragraph",
    text:
      "Speech synthesis is only useful if it sounds like speech. A model that mangles " +
      "every name it has not seen is not a smaller model; it is a broken one, and no amount " +
      "of careful documentation makes a broken voice acceptable to a user who just wants " +
      "their page read out loud.",
    probes:
      "The sentence the benchmark input is based on: long enough to cross any segment budget and to expose drift over several seconds.",
    expectation: "supported",
  },
  {
    id: "digits",
    text: "The train leaves from platform 4 at 10:45.",
    probes:
      "Numbers and a clock time. The profile has no number expansion (M1 §9.1), and a colon is outside the vocabulary — so this is expected to be read as bytes, not as `four` and `ten forty-five`.",
    expectation: "known-limitation",
  },
  {
    id: "currency",
    text: "It costs 20 euro.",
    probes:
      "Currency. `€` is outside the 4 000-token vocabulary, so it degrades to `<unk>` without byte fallback (M1 §8).",
    expectation: "known-limitation",
  },
  {
    id: "proper-nouns",
    text: "Dr. Alvarez met with Saoirse in Reykjavik.",
    probes:
      "Unseen proper nouns, an abbreviation with a period, and non-ASCII letters (`í`). Pronounces what the vocabulary can encode.",
    expectation: "supported",
  },
  {
    id: "punctuation",
    text: "Wait — really?! That cannot be right…",
    probes:
      "Em dashes, an exclamation-question pair, and an ellipsis: whether punctuation changes prosody or is dropped.",
    expectation: "known-limitation",
  },
  {
    id: "long-unpunctuated",
    text:
      "This sentence deliberately runs on and on without any punctuation at all so that the " +
      "engine has to decide where the pauses belong because the text itself provides no clue " +
      "about phrasing structure or when one thought ends and the next begins",
    probes:
      "No terminators at all: the segmenter cannot split it, so the model receives one long segment and has to invent the phrasing.",
    expectation: "known-limitation",
  },
];

/** Looks a case up by id. */
export function corpusCase(id: string): CorpusCase | undefined {
  return CORPUS.find((entry) => entry.id === id);
}
