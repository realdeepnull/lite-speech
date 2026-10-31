import { TTS_ERROR_CODES, TtsError } from "./errors.js";

/**
 * Deterministic, model-independent text preparation.
 *
 * Both functions are pure and dependency-free so they can be tested in
 * isolation and reused by any engine. They never remove content: a segment that
 * is too long is split, never truncated.
 */

/** Options for {@link prepareText}. */
export interface PrepareTextOptions {
  /**
   * Collapse every whitespace run (including newlines) into a single space.
   * Default `true`. With `false`, line breaks are preserved and only runs of
   * horizontal whitespace are collapsed.
   */
  readonly collapseWhitespace?: boolean;
}

/** Options for {@link segmentText}. */
export interface SegmentTextOptions extends PrepareTextOptions {
  /**
   * Maximum length of a segment in characters. Longer sentences are split at
   * clause boundaries and, only as a last resort, hard at the limit.
   * Default: 200. Must be a positive integer.
   */
  readonly maxCharsPerSegment?: number;
}

/** Default {@link SegmentTextOptions.maxCharsPerSegment}. */
export const DEFAULT_MAX_CHARS_PER_SEGMENT = 200;

const SENTENCE_TERMINATORS = new Set([".", "!", "?", ";", "…"]);
const CLOSING_MARKS = new Set(['"', "'", "”", "’", "»", ")", "]", "}"]);

/**
 * Normalizes text into the form the engine receives.
 *
 * This does **not** expand numbers, dates, or currency — that is a later,
 * opt-in stage that must be tested in isolation before it ships.
 */
export function prepareText(text: string, options: PrepareTextOptions = {}): string {
  if (typeof text !== "string") {
    throw new TtsError(
      TTS_ERROR_CODES.INVALID_ARGUMENT,
      `Expected a string to synthesize, got ${typeof text}`,
    );
  }
  const collapse = options.collapseWhitespace ?? true;
  const normalized = collapse
    ? text.replace(/\s+/gu, " ")
    : text.replace(/\r\n?/gu, "\n").replace(/[^\S\n]+/gu, " ");
  return normalized.trim();
}

/**
 * Splits text into segments in the original order.
 *
 * Boundaries are `.`, `!`, `?`, `;`, `…`, and newlines; a terminator only ends a
 * segment when the next character is whitespace or the end of the input, so
 * `3.14` and `e.g.`-like forms are not torn apart at the dot. Over-long
 * sentences are split at clause boundaries (`,`, `:`, `—`) and, when a single
 * word exceeds the limit, hard at the limit.
 *
 * Returns an empty array for empty or whitespace-only input — never `[""]`.
 */
export function segmentText(text: string, options: SegmentTextOptions = {}): string[] {
  const maxChars = options.maxCharsPerSegment ?? DEFAULT_MAX_CHARS_PER_SEGMENT;
  if (!Number.isInteger(maxChars) || maxChars < 1) {
    throw new TtsError(
      TTS_ERROR_CODES.INVALID_ARGUMENT,
      `maxCharsPerSegment must be a positive integer, got ${String(maxChars)}`,
      { details: { maxCharsPerSegment: maxChars } },
    );
  }
  // Sentence splitting happens at newlines, so they must survive normalization.
  // Only horizontal whitespace is collapsed unless the caller says otherwise.
  const prepared = prepareText(text, {
    collapseWhitespace: options.collapseWhitespace ?? false,
  });
  if (prepared.length === 0) {
    return [];
  }
  const segments: string[] = [];
  for (const sentence of splitSentences(prepared)) {
    for (const piece of splitLongSegment(sentence, maxChars)) {
      if (piece.length > 0) {
        segments.push(piece);
      }
    }
  }
  return segments;
}

/** Splits at sentence terminators and newlines; preserves all content. */
function splitSentences(text: string): string[] {
  const characters = Array.from(text);
  const sentences: string[] = [];
  let current = "";

  const flush = (): void => {
    const trimmed = current.trim();
    if (trimmed.length > 0) {
      sentences.push(trimmed);
    }
    current = "";
  };

  for (let index = 0; index < characters.length; index += 1) {
    const character = characters[index]!;
    if (character === "\n") {
      flush();
      continue;
    }
    current += character;
    if (!SENTENCE_TERMINATORS.has(character)) {
      continue;
    }
    let next = index + 1;
    while (next < characters.length && CLOSING_MARKS.has(characters[next]!)) {
      current += characters[next]!;
      next += 1;
    }
    const following = characters[next];
    if (following === undefined || /\s/u.test(following)) {
      flush();
      index = next - 1;
    }
  }
  flush();
  return sentences;
}

/** Splits a single sentence into pieces of at most `maxChars`, losslessly. */
function splitLongSegment(segment: string, maxChars: number): string[] {
  if (segment.length <= maxChars) {
    return [segment];
  }
  const pieces: string[] = [];
  let rest = segment;
  while (rest.length > maxChars) {
    const cut = findCut(rest, maxChars);
    pieces.push(rest.slice(0, cut).trim());
    rest = rest.slice(cut).trim();
  }
  if (rest.length > 0) {
    pieces.push(rest);
  }
  return pieces.filter((piece) => piece.length > 0);
}

/**
 * Returns the index to cut `text` at, at most `maxChars`.
 *
 * Prefers a clause boundary in the second half of the window, then any
 * whitespace, then a hard cut. Never splits a surrogate pair.
 */
function findCut(text: string, maxChars: number): number {
  const window = text.slice(0, maxChars);
  const clauseCut = lastBoundary(window, /[,:;–—][^\S\n]|[^\S\n]\s*/gu, Math.floor(maxChars / 2));
  if (clauseCut > 0) {
    return clauseCut;
  }
  const whitespaceCut = lastBoundary(window, /[^\S\n]+/gu, 1);
  if (whitespaceCut > 0) {
    return whitespaceCut;
  }
  return safeCut(text, maxChars);
}

/** Index just after the last match of `pattern` whose start is at least `minStart`. */
function lastBoundary(window: string, pattern: RegExp, minStart: number): number {
  let cut = -1;
  for (const match of window.matchAll(pattern)) {
    if (match.index !== undefined && match.index >= minStart) {
      cut = match.index + match[0].length;
    }
  }
  return cut;
}

/** Hard cut at `maxChars`, backed off by one when it would split a surrogate pair. */
function safeCut(text: string, maxChars: number): number {
  const code = text.charCodeAt(maxChars - 1);
  const isHighSurrogate = code >= 0xd800 && code <= 0xdbff;
  return isHighSurrogate && maxChars > 1 ? maxChars - 1 : maxChars;
}
