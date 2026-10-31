# `@lite-speech/engine-onnx`

Local text-to-speech engine for `lite-speech`, powered by Transformers.js and
ONNX Runtime (WASM by default, WebGPU opt-in).

The engine implements the `TtsEngine` contract from `@lite-speech/core`: it runs
inference in a Web Worker, resolves model profiles, keeps a pinned model and
voice registry, and manages cache state. It is the only package allowed to
depend on an ML runtime, and it never publishes model weights.

> **Status:** milestones M3 and M4. One verified profile (Phonon 31M EN,
> `fp32`/`q4`/`q4f16`) synthesizes through the public core API in Chromium and Node.js.
> Streaming delivers true incremental 80 ms frames. Provisioning is budgeted,
> verified against pinned SHA-256 digests, and rolled back on failure.

## Install

```sh
npm install @lite-speech/core @lite-speech/engine-onnx
```

## Quick start (browser)

```ts
import { createTts } from "@lite-speech/core";
import { createOnnxEngine } from "@lite-speech/engine-onnx";

// Async because resolving WebGPU needs an adapter probe; the engine it returns is
// synchronous, so `createTts()` stays synchronous.
const engine = await createOnnxEngine({ device: "auto" });
const tts = createTts({ engine, voice: "LMDIYE0Oa377Qh6Z", offline: "allow" });

const { pcm, sampleRate } = await tts.synthesize("Hello from your own device.");
```

Streaming is a first-class path, and it is **pull-based**: the engine asks the worker
for one frame per consumer turn, so a slow consumer cannot make memory grow.

```ts
for await (const chunk of tts.stream("A longer passage.", { signal })) {
  sink.enqueue(chunk.pcm, chunk.sampleRate); // exactly one chunk has `final: true`
}
```

## Quick start (Node.js)

Node has no `Worker` and no Cache Storage, so it has its own subpath. Inference runs
in the current thread and cache state is read from the filesystem.

```ts
import { createTts } from "@lite-speech/core";
import { createNodeOnnxEngine } from "@lite-speech/engine-onnx/node";

const engine = await createNodeOnnxEngine();
const tts = createTts({ engine, offline: "allow" });
const { pcm, sampleRate } = await tts.synthesize("Hello from Node.js.");
```

## Device and dtype

The default is boring on purpose.

| `device`           | Behaviour                                                                                 | Default dtype |
| ------------------ | ----------------------------------------------------------------------------------------- | ------------- |
| `"wasm"` / `"cpu"` | No probing at all                                                                         | `fp32`        |
| `"webgpu"`         | Requires a real adapter, and `shader-f16` for `q4f16`; **throws** instead of falling back | `q4`          |
| `"auto"`           | Prefers WebGPU, otherwise the environment default plus `fallbackFrom: "webgpu"`           | `q4`          |

Why `fp32` on CPU: every quantized variant measured 20–26 % **slower** warm in the
[M1 spike](https://github.com/realdeepnull/lite-speech/blob/main/docs/spike/transformers-js-phonon.md),
so quantization there buys download size and memory, not speed. `q4` is the
documented cheaper opt-in.

An accepted `webgpu` request is **not** evidence that a GPU ran: under
`onnxruntime-node` the runtime lists `webgpu` and an explicit WebGPU load succeeds
without `navigator.gpu`. The probe therefore gates on `navigator.gpu` and on adapter
features.

## Offline

`@lite-speech/core` owns the policy; this engine supplies the facts.

```ts
const tts = createTts({ engine, offline: "strict" }); // the default
await tts.isCached(); // model *and* voice files, from Cache Storage
await tts.estimateDownload(); // what a consent dialog shows, before any request

// Explicit provisioning, after which strict offline works with the network disabled.
const plan = await tts.prepare({ allowNetwork: true, maxDownloadMB: 80, verify: true });
plan.integrity; // "declared": every file has a pinned SHA-256
```

What the engine contributes to that policy:

- **Sizes and digests from the pinned manifest.** Every file in the Phonon registry
  declares its byte size and its Hub blob digest, at the commit SHA the M1 spike
  measured. `estimateDownload()` is therefore exact — 67 989 146 bytes for `q4`
  including the default voice — not an approximation.
- **Per-file cache state, measured.** `describeAssets()` reports `cached`, the
  manifest `size`, and the `cachedSize` it actually found. A disagreement is a
  truncated file, which strict offline refuses to synthesize from.
- **Integrity verification.** `tts.verifyAssets()` reads the cached bytes back and
  hashes them: `crypto.subtle` over Cache Storage in a browser, `node:crypto` over the
  filesystem cache in Node.js. The voice embedding is checked _before_ it is cached, so
  a corrupt download never becomes a corrupt cache.
- **All-or-nothing provisioning.** A failed `prepare()` removes only the files that
  attempt added, so a retry starts from the state it began with.
- **Eviction across every cache.** `removeModel()` disposes the session first, then
  deletes each file by URL from every Cache Storage cache and the filesystem layout.

Voice embeddings are written to the same Cache Storage cache the model files use, so
`isCached()` is honest about them and `removeModel()` removes them.

Build-time recipes (Node.js filesystem cache, Playwright-warmed browser profile) are in
[provisioning.md](https://github.com/realdeepnull/lite-speech/blob/main/docs/provisioning.md).

## Packaging

- `@lite-speech/engine-onnx` — browser entry; bundles core and Transformers.js so the
  built ESM runs without a bundler.
- `@lite-speech/engine-onnx/worker` — the worker entry, a documented export so a
  bundler can see it explicitly.
- `@lite-speech/engine-onnx/node` — Node entry; keeps `node:` builtins external and
  Transformers.js an ordinary dependency.

Pass `{ worker: false, inThreadLoader }` to run inference in-thread where `Worker` is
unavailable.

## Limitations

These come from the model profile, not from the engine:

- English only. Voices designed in other languages are not registered, because this
  profile does not switch pronunciation.
- No text normalization: numbers, dates, currency, and some symbols are read as bytes
  rather than spelled out.
- No voice cloning. Voices are precomputed embeddings.
- Determinism requires `temperature: 0`; the default samples.
- Quantized output is intelligible but not numerically close to `fp32`, so quality has
  to be assessed by listening.

## Documentation

- [Architecture](https://github.com/realdeepnull/lite-speech/blob/main/docs/ARCHITECTURE.md)
- [Provisioning](https://github.com/realdeepnull/lite-speech/blob/main/docs/provisioning.md)
- [Roadmap](https://github.com/realdeepnull/lite-speech/blob/main/docs/ROADMAP.md)
- [M1 inference spike](https://github.com/realdeepnull/lite-speech/blob/main/docs/spike/transformers-js-phonon.md)
- [Product concept](https://github.com/realdeepnull/lite-speech/blob/main/docs/lite-speech-concept.md)

## License

MIT — see [LICENSE](LICENSE).

Model weights are **not** part of this package and keep their own licenses. The
shipped Phonon profile is `Apache-2.0`. Gated or non-commercial (`CC BY-NC 4.0`)
weights are never bundled or redistributed by `lite-speech`; commercial use of such a
model family must be evaluated independently of this library's MIT license. Full
details: [NOTICE](https://github.com/realdeepnull/lite-speech/blob/main/NOTICE.md).
