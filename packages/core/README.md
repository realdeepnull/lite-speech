# `@lite-speech/core`

Engine-independent, privacy-friendly, offline-capable text-to-speech core API.

The core owns the public contract: `createTts`, the `TtsEngine` interface,
lifecycle and streaming, text preparation, asset/offline policy, and the error
model. It ships with **zero runtime dependencies** and never imports an engine,
an ONNX runtime, or Transformers.js.

> **Status:** M2. The public API is implemented and covered by a fake-engine
> unit suite that runs on Node.js without DOM shims. The first runnable
> end-to-end quick start lands with the M3 engine and the M6 example.

## Install

```sh
npm install @lite-speech/core
```

An inference engine such as
[`@lite-speech/engine-onnx`](https://www.npmjs.com/package/@lite-speech/engine-onnx)
is required to synthesize audio.

## Quick start

The composition below is the target API. The core half (M2) is implemented and
tested with a fake engine; `createOnnxEngine` lands with M3, so a runnable
end-to-end sample arrives with the M6 example.

```ts
import { createTts, encodeWav } from "@lite-speech/core";
import { createOnnxEngine } from "@lite-speech/engine-onnx";

const tts = createTts({
  engine: createOnnxEngine({ model: "onnx-community/phonon-31m-en-ONNX" }),
  voice: "female-1",
  offline: "strict", // assets must already be cached
});

// Explicit provisioning: budget check first, then the download.
await tts.prepare({ allowNetwork: true, maxDownloadMB: 120 });

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
| `TtsPool`                                             | Single-flight, LRU-cached multi-voice instances                   |
| `TTS_ERROR_CODES`, `TtsError`                         | Stable error taxonomy with `cause` chaining                       |
| `createEmitter`                                       | Typed emitter without a DOM dependency                            |
| `encodeWav`                                           | Optional PCM → WAV encoder above the inference contract           |

## Semantics worth knowing

- **`offline: "strict"` is enforced in core**, not trusted to the engine: a
  synthesis call whose assets are not cached rejects with
  `OFFLINE_ASSETS_MISSING` before the engine is asked to synthesize.
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
- **Zero runtime dependencies**; `size-limit` guards a 7 kB min+gzip budget.

## Building an engine

Implement `TtsEngine` (`supports`, `voices`, `isCached`, `load`, `synthesize`,
`stream`, `dispose`) and optionally `capabilities`, `prepare`, `describeAssets`,
and `removeModel`. The core never imports an engine module, so an engine can live
in its own package with its own dependencies.

## Documentation

- [Architecture](https://github.com/realdeepnull/lite-speech/blob/main/docs/ARCHITECTURE.md)
- [Roadmap](https://github.com/realdeepnull/lite-speech/blob/main/docs/ROADMAP.md)
- [Product concept](https://github.com/realdeepnull/lite-speech/blob/main/docs/lite-speech-concept.md)

## License

MIT — see [LICENSE](LICENSE).

Model weights are **not** part of this package and keep their own licenses.
Gated or non-commercial (`CC BY-NC 4.0`) weights are never bundled or
redistributed by `lite-speech`; commercial use of such a model family must be
evaluated independently of this library's MIT license. Full details:
[NOTICE](https://github.com/realdeepnull/lite-speech/blob/main/NOTICE.md).
