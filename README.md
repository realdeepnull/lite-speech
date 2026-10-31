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
  synthesis with Transformers.js/ONNX in a Web Worker, with pinned model
  profiles and a model + voice registry.

## Install

```sh
npm install @lite-speech/core @lite-speech/engine-onnx
```

## Quick start

The public API (`createTts`, `synthesize`, `stream`, `dispose`) is implemented in
milestone M2; the first runnable quick start lands with the M6 beta.

## Status

Pre-release. Milestones are tracked in the [roadmap](docs/ROADMAP.md):

| Milestone | Theme                 | State   |
| --------- | --------------------- | ------- |
| M0        | Foundation            | done    |
| M1        | Inference spike       | done    |
| M2        | Core MVP              | next    |
| M3        | ONNX engine MVP       | planned |
| M4        | Assets & provisioning | planned |
| M5        | Measurement           | planned |
| M6        | Beta `0.1.0`          | planned |

The M1 evidence — pins, measured sizes and latencies, streaming classification, and
the go/no-go decision — is in
[docs/spike/transformers-js-phonon.md](docs/spike/transformers-js-phonon.md). Its
runnable scripts live in [spike/](spike) and are never published.

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

Per-package scripts fan out with `--workspaces --if-present`, so a package opts
into `build`, `typecheck`, `test`, `size`, or `pack:check` only when it defines
the script; `lint`, `format`, `format:check`, and `knip` run over the repository.
Tooling is added just in time; the adoption plan lives in the
[roadmap](docs/ROADMAP.md#5-tooling-adoption-plan).

## Documentation

- [Product concept](docs/lite-speech-concept.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Roadmap](docs/ROADMAP.md)

## License

MIT. Model weights keep their own licenses (`Apache-2.0` vs. `CC BY-NC 4.0`);
see [NOTICE.md](NOTICE.md).
