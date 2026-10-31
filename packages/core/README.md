# `@lite-speech/core`

Engine-independent, privacy-friendly, offline-capable text-to-speech core API.

The core owns the public contract: `createTts`, the `TtsEngine` interface,
lifecycle and streaming, text preparation, asset/offline policy, and the error
model. It ships with **zero runtime dependencies** and never imports an engine,
an ONNX runtime, or Transformers.js.

> **Status:** M2 implemented; M4 added the provisioning and offline-proof layer. The
> public API is covered by a fake-engine unit suite that runs on Node.js without DOM
> shims. A runnable end-to-end quick start needs
> [`@lite-speech/engine-onnx`](https://www.npmjs.com/package/@lite-speech/engine-onnx),
> which landed with M3.

## Install

```sh
npm install @lite-speech/core
```

An inference engine such as
[`@lite-speech/engine-onnx`](https://www.npmjs.com/package/@lite-speech/engine-onnx)
is required to synthesize audio.

## Quick start

```ts
import { createTts, encodeWav } from "@lite-speech/core";
import { createOnnxEngine } from "@lite-speech/engine-onnx";

// `createOnnxEngine()` is async because resolving WebGPU requires an adapter
// probe; `createTts()` itself stays synchronous, like every engine creation.
const engine = await createOnnxEngine({ device: "auto" });

const tts = createTts({
  engine,
  voice: "LMDIYE0Oa377Qh6Z",
  offline: "strict", // assets must already be cached
});

// Explicit provisioning: budget check first, then the download, then verification.
const plan = await tts.prepare({ allowNetwork: true, maxDownloadMB: 120, verify: true });
// plan.integrity === "declared" | "partial" | "none"

const audio = await tts.synthesize("Hello from lite-speech.");
// audio: { pcm: Float32Array, sampleRate: number, channels: 1, durationMs }

const wav = encodeWav(audio); // optional, tree-shakeable helper

const controller = new AbortController();
for await (const chunk of tts.stream("Generated entirely offline.", {
  signal: controller.signal,
})) {
  // chunk.pcm / chunk.sampleRate feed the application's audio sink.
}

await tts.dispose();
```

## Public surface

| Export                                                | Purpose                                                           |
| ----------------------------------------------------- | ----------------------------------------------------------------- |
| `createTts`, `Tts`                                    | Synchronous factory, lifecycle, `preload`, `synthesize`, `stream` |
| `TtsEngine`                                           | The engine SPI; the only coupling point with a backend            |
| `registerDefaultEngine`                               | Optional global registry for convenience meta packages            |
| `prepareText`, `segmentText`                          | Deterministic, model-independent text preparation                 |
| `createAudioStream`                                   | Bounded `AsyncIterable` with the chunk contract enforced          |
| `createAssetPlan`, `estimateDownload`, `assertBudget` | Offline/budget policy helpers                                     |
| `createIntegrityReport`, `assertIntegrity`            | Digest-verification result helpers                                |
| `TtsPool`                                             | Single-flight, LRU-cached multi-voice instances                   |
| `TTS_ERROR_CODES`, `TtsError`                         | Stable error taxonomy with `cause` chaining                       |
| `createEmitter`                                       | Typed emitter without a DOM dependency                            |
| `encodeWav`                                           | Optional PCM → WAV encoder above the inference contract           |

## Semantics worth knowing

- **`offline: "strict"` is enforced in core**, not trusted to the engine: a
  synthesis call whose assets are not cached rejects with
  `OFFLINE_ASSETS_MISSING` before the engine is asked to synthesize. The same
  gate covers `preload()`, so it cannot be used to warm up around the policy.
- **A budget is checked before the first byte**: `prepare({ maxDownloadMB })`
  rejects with `DOWNLOAD_BUDGET_EXCEEDED` and lists the files that did not fit.
  A plan without declared sizes is rejected too — an unpriced budget is not a
  budget.
- **Integrity is a state, not a promise**: every plan reports
  `integrity: "declared" | "partial" | "none"`. `prepare({ verify: true })` and
  `verifyAssets()` read the cached bytes back and reject with
  `ASSET_INTEGRITY_FAILED` on a mismatch or an unreadable file. Verification is
  opt-in because hashing a 68 MB profile is not free.
- **A truncated file is not cached**: a `cachedSize` that disagrees with the
  declared size makes strict offline reject rather than synthesize from bad
  weights.
- **`preload()` is single-flight, and failures are not memoized**, so a
  transient network error does not poison the instance.
- **`stream()` chunks are contract-checked**: `sequence` starts at `0` and is
  gap-free, `sampleRate` never changes mid-stream, and exactly one chunk carries
  `final: true`. A violation fails with `STREAM_FAILED`.
- **Backpressure is mandatory**: the queue is bounded, so a fast engine cannot
  grow memory when the consumer is slow.
- **Aborts keep partials**: audio delivered before an abort stays valid; the
  iterator then fails with `ABORTED`.
- **Streaming is classified, not promised**: `TtsCapabilities.streaming` reports
  `"incremental-frames" | "sentence-chunks" | "utterance"` per model profile.
- **Debug events are opt-in and free when unused**: `onDebug` receives a discriminated
  union covering load, device resolution and fallback, stream start, time to first
  chunk, the model call itself, and aborts. Measured values per runtime are in
  [docs/performance.md](https://github.com/realdeepnull/lite-speech/blob/main/docs/performance.md).
- **Zero runtime dependencies**; `size-limit` guards an 8 kB min+gzip budget.
  _Measured after M4: 6.95 kB. The budget was 7 kB after M2 (6.35 kB measured); M4's
  provisioning and integrity layer consumed the headroom, so the limit moved to 8 kB
  rather than shipping a budget with none left._

## Offline and provisioning

`describeAssets()` answers "what would this cost" without fetching anything;
`prepare()` is the only call that may download, and only with
`allowNetwork: true`. Build-time recipes (Node.js filesystem cache and a
Playwright-warmed browser cache) are in
[provisioning.md](https://github.com/realdeepnull/lite-speech/blob/main/docs/provisioning.md).

```ts
const plan = await tts.describeAssets();
plan.complete; // everything is cached
plan.missing; // what would have to be downloaded
plan.integrity; // "declared" | "partial" | "none"
plan.sizeMismatches; // cached files whose size is wrong
```

## Building an engine

Implement `TtsEngine` (`supports`, `voices`, `isCached`, `load`, `synthesize`,
`stream`, `dispose`) and optionally `capabilities`, `prepare`, `describeAssets`,
`verifyAssets`, and `removeModel`. The core never imports an engine module, so an
engine can live in its own package with its own dependencies.

`verifyAssets()` is optional but load-bearing for the offline guarantee: without it,
`prepare({ verify: true })` rejects with `NOT_SUPPORTED` rather than reporting an
unverified cache as a verified one.

## Documentation

- [Architecture](https://github.com/realdeepnull/lite-speech/blob/main/docs/ARCHITECTURE.md)
- [Provisioning](https://github.com/realdeepnull/lite-speech/blob/main/docs/provisioning.md)
- [Roadmap](https://github.com/realdeepnull/lite-speech/blob/main/docs/ROADMAP.md)
- [Product concept](https://github.com/realdeepnull/lite-speech/blob/main/docs/lite-speech-concept.md)

## License

MIT — see [LICENSE](LICENSE).

Model weights are **not** part of this package and keep their own licenses.
Gated or non-commercial (`CC BY-NC 4.0`) weights are never bundled or
redistributed by `lite-speech`; commercial use of such a model family must be
evaluated independently of this library's MIT license. Full details:
[NOTICE](https://github.com/realdeepnull/lite-speech/blob/main/NOTICE.md).
