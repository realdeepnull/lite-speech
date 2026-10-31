/**
 * Encoder evidence for the M1 exit gate.
 *
 * 1. The pinned release ships no Unigram byte fallback, so a piece outside the vocabulary
 *    would collapse into the unknown token.
 * 2. `createPublicEncoder` reproduces the model card's private-field workaround exactly,
 *    using public members only.
 * 3. Which characters actually need it, per group.
 *
 * Usage: node probe-encoder.mjs
 */
import { AutoTokenizer } from "@huggingface/transformers";

import { createPublicEncoder, MODEL_ID, MODEL_REVISION, patchByteFallback } from "./lib/phonon.mjs";

const corpus = [
  "Hello from lite-speech. This is the second sentence of the inference spike.",
  "Digits 1234, times 3:45 pm, 12.5%, phone +49 30 1234567.",
  "It costs \u20ac5.50 and ships from the U.S.",
  "Typographic \u201cquotes\u201d and a dash \u2014 like this.",
  "Caf\u00e9 na\u00efve r\u00e9sum\u00e9 jalape\u00f1o.",
  "Emoji \ud83d\ude42 and CJK \u4f60\u597d.",
  "e-mail: hello@example.com",
  "A colon: and a plus+ and an at@ sign.",
  "Symbols ~^|\\/_=*&%$#<>[]{}()",
];

const groups = {
  letters: "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",
  digits: "0123456789",
  punctuation: ".,!?;:'\"-()[]{}<>/\\|_=*&^%$@#~`+",
  typographic: "\u2018\u2019\u201c\u201d\u2013\u2014\u2026",
  accents: "\u00e9\u00ef\u00f1\u00fc\u00df",
};

// Two independent loads: one receives the private patch, the other stays untouched so
// the plain-render path can be measured on the same code path a consumer would use.
const reference = await AutoTokenizer.from_pretrained(MODEL_ID, {
  revision: MODEL_REVISION,
});
const patchMode = patchByteFallback(reference);
const untouched = await AutoTokenizer.from_pretrained(MODEL_ID, {
  revision: MODEL_REVISION,
});

const encodePublic = createPublicEncoder(untouched);
const vocab = untouched.get_vocab();

const comparisons = corpus.map((text) => {
  const expected = reference.encode(text, { add_special_tokens: false });
  const publicIds = encodePublic(text);
  return {
    text,
    matches: JSON.stringify(expected) === JSON.stringify(publicIds),
    expected,
    publicIds,
  };
});

const unknownTokenId = untouched.unk_token_id ?? 0;
const coverage = Object.fromEntries(
  Object.entries(groups).map(([group, chars]) => {
    const missing = [];
    for (const char of chars) {
      const inWord = untouched.encode(`a${char}b`, { add_special_tokens: false });
      if (inWord.includes(unknownTokenId)) missing.push(char);
    }
    return [group, { size: chars.length, missing }];
  }),
);

const failed = comparisons.filter((row) => !row.matches);

console.log(
  JSON.stringify(
    {
      modelId: MODEL_ID,
      revision: MODEL_REVISION,
      privatePatchMode: patchMode,
      vocabularySize: vocab.size,
      byteTokenCoverage: `<0x00> … <0xFF>: ${
        Array.from({ length: 256 }, (_, i) =>
          vocab.has(`<0x${i.toString(16).toUpperCase().padStart(2, "0")}>`),
        ).filter(Boolean).length
      }/256 present`,
      encoder: {
        corpus: comparisons.length,
        matchesPrivateReference: comparisons.length - failed.length,
        failed: failed.map((row) => ({
          text: row.text,
          expected: row.expected,
          publicIds: row.publicIds,
        })),
      },
      charsetCoverage: coverage,
      asciiEnglishUnchanged:
        coverage.letters.missing.length === 0 && coverage.digits.missing.length === 0,
      note: "`missing` lists characters that collapse into the unknown token without byte fallback.",
    },
    null,
    2,
  ),
);
