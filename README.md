# lite-speech

Privacy-friendly, local, offline-capable text-to-speech for JavaScript and
TypeScript. Text never leaves the user's device.

`lite-speech` runs small speech models locally through ONNX Runtime — in the
browser, in Electron, and in Node.js — behind one predictable API. No hosted
inference service, no hidden downloads, no telemetry.

- **Core**: [`@lite-speech/core`](packages/core) — engine-independent API, the
  `TtsEngine` contract, lifecycle, streaming, text preparation, and the
  asset/offline policy. Zero runtime dependencies.
- **Engine**: [`@lite-speech/engine-onnx`](packages/engine-onnx) — local speech
  synthesis with Transformers.js/ONNX in a Web Worker, with a declarative model,
  voice, and model-family catalogue.

## Install

```sh
npm install @lite-speech/core @lite-speech/engine-onnx
```

## Quick start

```ts
import { createTts } from "@lite-speech/core";
import { createOnnxEngine } from "@lite-speech/engine-onnx";

// Async because resolving WebGPU requires an adapter probe.
const engine = await createOnnxEngine({ device: "auto" });
const tts = createTts({ engine, voice: "LMDIYE0Oa377Qh6Z", offline: "allow" });

const { pcm, sampleRate } = await tts.synthesize("Hello from your own device.");

// Streaming asks the worker for one frame per consumer turn: backpressure is structural.
for await (const chunk of tts.stream("A longer passage.")) {
  sink.enqueue(chunk.pcm, chunk.sampleRate); // exactly one chunk has `final: true`
}
```

In Node.js, use `createNodeOnnxEngine` from `@lite-speech/engine-onnx/node`; it runs
inference in-thread and reads cache state from the filesystem. A bundler-free browser
demo against the built ESM output lives in [examples/demo](examples/demo).

## Other models and languages

Models, languages, and model families are declarations, not code. Register your own
catalogue and pass it to the engine:

```ts
const engine = await createOnnxEngine({
  models: [
    {
      id: "your-org/phonon-de-ONNX",
      revision: "<40-char commit sha>", // a pin, never a branch
      family: "phonon", // which runner runs it
      languages: ["de"], // which requests it serves
      license: "Apache-2.0",
      sampleRate: 24000,
      files: [
        { path: "config.json", size: 4_952, sha256: "…" },
        { path: "onnx/flow_lm_q4.onnx", variant: "q4", size: 84_132, sha256: "…" },
      ],
    },
  ],
  voices: [], // `VoiceManifest`s as returned by your own registry
});

const tts = createTts({ engine, language: "de" });
```

A new _architecture_ is a new entry in the engine's family registry, not a new package
and not an engine edit. Sizes and digests are optional but recommended: without them the
download cannot be priced and the cache cannot be proven, which is exactly what
`offline: "strict"` refuses to do.

The full recipe is [docs/adding-a-model.md](docs/adding-a-model.md).

## Status

Pre-release. Milestones are tracked in the [roadmap](docs/ROADMAP.md):

| Milestone | Theme                      | State |
| --------- | -------------------------- | ----- |
| M0        | Foundation                 | done  |
| M1        | Inference spike            | done  |
| M2        | Core MVP                   | done  |
| M3        | ONNX engine MVP            | done  |
| M4        | Assets & provisioning      | done  |
| M5        | Measurement                | done  |
| M6        | Modular models & languages | done  |
| M7        | Beta `0.1.0`               | next  |

The M1 evidence — pins, measured sizes and latencies, streaming classification, and
the go/no-go decision — is in
[docs/spike/transformers-js-phonon.md](docs/spike/transformers-js-phonon.md). Its
runnable scripts live in [spike/](spike) and are never published.

The M5 evidence — per-runtime latency, memory, download size, and the quality sweep — is in
[docs/performance.md](docs/performance.md), with the JSON reports it is derived from under
[bench/report/](bench/report). The scripts that produce them live in [bench/](bench).

## Packages

| Package                                            | Runtime dependencies        | Purpose                     |
| -------------------------------------------------- | --------------------------- | --------------------------- |
| [`@lite-speech/core`](packages/core)               | none                        | public API and engine SPI   |
| [`@lite-speech/engine-onnx`](packages/engine-onnx) | `@huggingface/transformers` | Transformers.js/ONNX engine |

Model weights are never published inside the npm packages; they are pinned
remote assets resolved through a manifest.

## Development

```sh
npm install
npm run build        # tsdown per package
npm run typecheck    # tsc --noEmit per package
npm run lint         # ESLint (repository-wide)
npm run format:check # Prettier (repository-wide)
npm test             # Vitest per package
npm run knip         # unused files, exports, dependencies
npm run size         # size-limit per package
npm run pack:check   # build + artifact validation per package
```

Benchmark and quality measurements are separate from this sequence — they need the network for a
cold run and take minutes, so they are documented in [bench/README.md](bench/README.md) and are
committed as reports rather than run on every change.

Per-package scripts fan out with `--workspaces --if-present`, so a package opts
into `build`, `typecheck`, `test`, `size`, or `pack:check` only when it defines
the script; `lint`, `format`, `format:check`, and `knip` run over the repository.
Tooling is added just in time; the adoption plan lives in the
[roadmap](docs/ROADMAP.md#5-tooling-adoption-plan).

## Documentation

- [Product concept](docs/lite-speech-concept.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Provisioning](docs/provisioning.md)
- [Performance and quality](docs/performance.md)
- [Roadmap](docs/ROADMAP.md)

## License

MIT. Model weights keep their own licenses (`Apache-2.0` vs. `CC BY-NC 4.0`);
see [NOTICE.md](NOTICE.md).
