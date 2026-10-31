# `@lite-speech/core`

Engine-independent, privacy-friendly, offline-capable text-to-speech core API.

The core owns the public contract: `createTts`, the `TtsEngine` interface,
lifecycle and streaming, text preparation, asset/offline policy, and the error
model. It ships with **zero runtime dependencies** and never imports an engine,
an ONNX runtime, or Transformers.js.

> **Status:** M0 scaffold. The public API is implemented in milestone M2. Until
> then this package only exposes its packaging contract.

## Install

```sh
npm install @lite-speech/core
```

An inference engine such as
[`@lite-speech/engine-onnx`](https://www.npmjs.com/package/@lite-speech/engine-onnx)
is required to synthesize audio.

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
