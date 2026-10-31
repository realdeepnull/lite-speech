/**
 * The engine ↔ worker protocol.
 *
 * One worker hosts one loaded model and processes messages strictly in arrival
 * order. Every request carries a numeric `id` that its responses echo; a response
 * whose `id` is no longer pending is dropped, which is exactly how the late frames
 * of a cancelled request disappear.
 *
 * **Streaming is pull-based.** The engine opens a session, then asks for one frame
 * at a time. Backpressure therefore needs no credit accounting — the worker cannot
 * outrun the consumer because it is not asked to — memory stays bounded to a single
 * frame plus the model state, and cancellation has a checkpoint between every frame
 * instead of only between requests. A push protocol would need an explicit
 * high-water mark in the worker; this one cannot buffer unboundedly by construction.
 *
 * Types and pure helpers only: the main-thread engine and the worker both import
 * this module, and neither may drag the other's runtime in.
 */
import type { ProgressEvent, StreamingClass } from "@lite-speech/core";

/** `load`: prepare the tokenizer, both graphs, and one voice. */
export interface LoadRequest {
  readonly kind: "load";
  readonly id: number;
  readonly modelId: string;
  readonly revision: string;
  /** Resolved device: `"wasm"`, `"webgpu"`, or a Node.js backend such as `"cpu"`. */
  readonly device: string;
  /** Resolved dtype: `"fp32"`, `"q4"`, or `"q4f16"`. */
  readonly dtype: string;
  /** `metadata.profile` of the manifest; omitted means the built-in profile. */
  readonly profile?: string;
  /** Voice to preload; per-request `voice` can switch it later. */
  readonly voice?: string;
  /** Expected SHA-256 of the voice file, when the manifest declares one. */
  readonly voiceSha256?: string;
}

/** `synthesize`: one full utterance in a single response. */
export interface SynthesizeRequest {
  readonly kind: "synthesize";
  readonly id: number;
  readonly text: string;
  readonly voice?: string;
  readonly temperature?: number;
}

/** `stream-open`: starts a pull-based session for `text`. */
export interface StreamOpenRequest {
  readonly kind: "stream-open";
  readonly id: number;
  readonly text: string;
  readonly voice?: string;
  readonly temperature?: number;
}

/** `stream-next`: asks the open session for exactly one frame. */
export interface StreamNextRequest {
  readonly kind: "stream-next";
  readonly id: number;
}

/** `stream-close`: abandons the open session without waiting for a frame. */
export interface StreamCloseRequest {
  readonly kind: "stream-close";
  readonly id: number;
}

/** `cancel`: cooperative cancellation; the worker answers `cancelled`. */
export interface CancelRequest {
  readonly kind: "cancel";
  readonly id: number;
}

/** `cache-check`: which of `files` the runtime already holds, and how big they are. */
export interface CacheCheckRequest {
  readonly kind: "cache-check";
  readonly id: number;
  readonly modelId: string;
  readonly revision: string;
  readonly files: readonly string[];
}

/** `dispose`: releases everything and always acknowledges. */
export interface DisposeRequest {
  readonly kind: "dispose";
  readonly id: number;
}

/** Every message the engine may post. */
export type WorkerRequest =
  | LoadRequest
  | SynthesizeRequest
  | StreamOpenRequest
  | StreamNextRequest
  | StreamCloseRequest
  | CancelRequest
  | CacheCheckRequest
  | DisposeRequest;

/** Download/load progress, forwarded from Transformers.js with a normalized shape. */
export interface ProgressResponse {
  readonly kind: "progress";
  readonly id: number;
  readonly event: ProgressEvent;
}

/** What the worker resolved, reported once per successful `load`. */
export interface CapabilitiesResponse {
  readonly kind: "capabilities";
  readonly id: number;
  readonly device: string;
  readonly dtype: string;
  readonly modelId: string;
  readonly modelVersion: string;
  readonly sampleRate: number;
  readonly streaming: StreamingClass;
}

/** The model is ready. Settles `load`. */
export interface LoadedResponse {
  readonly kind: "loaded";
  readonly id: number;
}

/** Generation started; brackets the model call for timing. */
export interface InferenceStartResponse {
  readonly kind: "inference-start";
  readonly id: number;
  readonly textLength: number;
}

/** Generation finished; `samples` counts audio samples, not frames. */
export interface InferenceDoneResponse {
  readonly kind: "inference-done";
  readonly id: number;
  readonly durationMs: number;
  readonly samples: number;
  readonly sampleRate: number;
}

/** A complete utterance. `pcm` is transferred, never copied. */
export interface ResultResponse {
  readonly kind: "result";
  readonly id: number;
  readonly pcm: ArrayBuffer;
  readonly sampleRate: number;
  readonly frames: number;
  readonly durationMs: number;
}

/**
 * One frame of a stream.
 *
 * There is deliberately no `final` flag: the worker does not know a frame is last until
 * the generator ends, and peeking one frame ahead to find out would double
 * time-to-first-audio. The engine therefore appends the single `final: true` chunk the
 * core contract requires when it sees {@link StreamEndResponse}.
 */
export interface ChunkResponse {
  readonly kind: "chunk";
  readonly id: number;
  readonly pcm: ArrayBuffer;
  readonly sampleRate: number;
}

/** The session opened and is ready for `stream-next`. */
export interface StreamOpenedResponse {
  readonly kind: "stream-opened";
  readonly id: number;
}

/** The session ended without producing a final chunk. A contract violation. */
export interface StreamEndResponse {
  readonly kind: "stream-end";
  readonly id: number;
}

/** The request was cancelled cooperatively. */
export interface CancelledResponse {
  readonly kind: "cancelled";
  readonly id: number;
}

/** Per-file cache state. `sizes` entries are `undefined` when a size is unknown. */
export interface CacheStateResponse {
  readonly kind: "cache-state";
  readonly id: number;
  readonly cached: readonly boolean[];
  readonly sizes: readonly (number | undefined)[];
}

/** Disposal finished. Always sent, even when releasing resources failed. */
export interface DisposedResponse {
  readonly kind: "disposed";
  readonly id: number;
}

/** A request failed. `code` carries a stable core error code when the worker knows one. */
export interface ErrorResponse {
  readonly kind: "error";
  readonly id: number;
  readonly message: string;
  readonly code?: string;
}

/** Every message the worker may post. */
export type WorkerResponse =
  | ProgressResponse
  | CapabilitiesResponse
  | LoadedResponse
  | InferenceStartResponse
  | InferenceDoneResponse
  | ResultResponse
  | ChunkResponse
  | StreamOpenedResponse
  | StreamEndResponse
  | CancelledResponse
  | CacheStateResponse
  | DisposedResponse
  | ErrorResponse;

/**
 * Responses that keep a pending request alive rather than settling it.
 *
 * `inference-start`/`inference-done` are timing telemetry only: they arrive around
 * a `result` or a sequence of `chunk`s and never resolve anything themselves.
 */
export function isIntermediateResponse(response: WorkerResponse): boolean {
  switch (response.kind) {
    case "progress":
    case "capabilities":
    case "inference-start":
    case "inference-done":
      return true;
    default:
      return false;
  }
}

/** Returns the transfer list for a response, so PCM buffers move instead of cloning. */
export function transferListFor(response: WorkerResponse): Transferable[] {
  switch (response.kind) {
    case "result":
    case "chunk":
      return [response.pcm];
    default:
      return [];
  }
}
