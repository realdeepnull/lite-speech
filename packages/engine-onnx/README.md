# `@lite-speech/engine-onnx`

Local text-to-speech engine for `lite-speech`, powered by Transformers.js and
ONNX Runtime (WASM by default, WebGPU opt-in).

The engine implements the `TtsEngine` contract from `@lite-speech/core`: it runs
inference in a Web Worker, resolves model profiles, keeps a pinned model and
voice registry, and manages cache state. It is the only package allowed to
depend on an ML runtime, and it never publishes model weights.

> **Status:** M0 scaffold. The inference spike is milestone M1 and the engine
> implementation is milestone M3. Until then this package only exposes its
> packaging contract.

## Install

```sh
npm install @lite-speech/core @lite-speech/engine-onnx
```

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
