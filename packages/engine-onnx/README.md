# `@lite-speech/engine-onnx`

Local text-to-speech engine for `lite-speech`, powered by Transformers.js and
ONNX Runtime (WASM by default, WebGPU opt-in).

The engine implements the `TtsEngine` contract from `@lite-speech/core`: it runs
inference in a Web Worker, resolves a model's family to a runner, keeps a declarative
catalogue of models and voices, and manages cache state. It is the only package allowed
to depend on an ML runtime, and it never publishes model weights.

> **Status:** milestones M3–M6. One verified model (Phonon 31M EN,
> `fp32`/`q4`/`q4f16`) synthesizes through the public core API in Chromium and Node.js.
> Streaming delivers true incremental 80 ms frames. Provisioning is budgeted,
> verified against pinned SHA-256 digests, and rolled back on failure. Models,
> languages, and model families are declared in a catalogue and registered through
> the factory, so adding one needs no engine change.

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
documented cheaper opt-in. The M5 sweep re-measured this with percentiles rather than a single
sample; see [docs/performance.md](https://github.com/realdeepnull/lite-speech/blob/main/docs/performance.md)
for the per-device numbers and whether your host is expected to behave the same way.

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

Pass `{ worker: false, families }` to run inference in-thread where `Worker` is
unavailable; `createNodeOnnxEngine` does exactly that with the built-in families.

## Models, languages, and families

Three independent declarations, each read from exactly one place:

| Decision          | Where it lives                       | Engine code needed |
| ----------------- | ------------------------------------ | ------------------ |
| Which model runs  | `ModelManifest` (`id`, `revision`)   | no                 |
| Which language    | `ModelManifest.languages`            | no                 |
| Which weights     | `AssetFile.variant`                  | no                 |
| How it runs       | `ModelManifest.family` → a runner    | only a new family  |
| Which voice bytes | `VoiceManifest` (`modelId`, `files`) | no                 |

```ts
import { createRegistry, createOnnxEngine } from "@lite-speech/engine-onnx";

const engine = await createOnnxEngine({
  registry: createRegistry({ models: [myModel], voices: myVoices }),
});
const tts = createTts({ engine, language: "de" }); // a voice of that language is selected
```

- **Another language** is another model manifest (or another entry in `languages`).
  A region code falls back to its primary subtag (`de-AT` → `de`); a language no model
  declares is rejected with `LANGUAGE_NOT_SUPPORTED` and the list of supported ones.
- **Another checkpoint of the same architecture** is a manifest with the same `family`
  and its own pin. Every variant is priced by core through `variant`, so a `q4` device
  does not download `fp32`.
- **Another architecture** is a `ModelRunner` factory added to `families`. It receives
  the resolved model, device, dtype, and voice, and yields mono PCM frames; nothing in
  the worker protocol or in core changes.

The full recipe, including what a runner must guarantee and which tests keep it honest, is
[`docs/adding-a-model.md`](https://github.com/realdeepnull/lite-speech/blob/main/docs/adding-a-model.md).

## Limitations

These come from the shipped _model and family_, not from the engine's structure:

- The built-in catalogue still registers English only. Phonon 31M is an English
  checkpoint, so registering a German voice in it would advertise a pronunciation it
  cannot produce. A German model is a new manifest, not a new voice.
- No text normalization: numbers, dates, currency, and some symbols are read as bytes
  rather than spelled out. Normalization is per language by nature, so it belongs in an
  opt-in layer, not in the family's measured text rules.
- No voice cloning. Voices are precomputed embeddings.
- Determinism requires `temperature: 0`; the default samples. Pass
  `createOnnxEngine({ temperature: 0 })` — without it the option was unreachable, and the
  claim in this list would have been untestable.
- Quantized output is intelligible but not numerically close to `fp32`: at temperature 0 the two
  are individually reproducible yet uncorrelated with each other, so the difference is a different
  performance of the same sentence rather than added noise. Quality therefore has to be assessed by
  listening, not by waveform distance — see
  [docs/performance.md](https://github.com/realdeepnull/lite-speech/blob/main/docs/performance.md).

## Documentation

- [Adding a model, a language, or a family](https://github.com/realdeepnull/lite-speech/blob/main/docs/adding-a-model.md)
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
