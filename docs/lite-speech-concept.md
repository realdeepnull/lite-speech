# lite-speech — Modular Offline Text-to-Speech for JavaScript and TypeScript

> **Status:** High-level product and architecture concept — not a final implementation specification  
> **Date:** October 31, 2026  
> **Focus:** Phonon, small downloads, offline inference, streaming, performance, and a reliable TypeScript API

## 1. Vision

**lite-speech** is a modular, offline-first text-to-speech (TTS) toolkit for JavaScript and TypeScript. It runs compact speech models locally, without requiring a hosted API or transmitting user text to a server.

**Phonon is the initial model family**, but the public API must remain model-agnostic so additional architectures can be supported without breaking applications.

The initial project consists of **two independently versioned and publishable npm packages**:

- **`@lite-speech/core`** — the model-independent API, lifecycle management, streaming, offline policies, and shared TTS capabilities.
- **`@lite-speech/engine-onnx`** — a model-inference adapter built around **`@huggingface/transformers` (Transformers.js)** and ONNX Runtime, initially targeting Phonon.

The `@lite-speech` npm scope is a proposed name and must be confirmed before publishing.

**Product promise:** One predictable API for private, reproducible, local speech synthesis — without requiring application developers to manage ONNX sessions, model assets, caching, audio chunks, or backend details themselves.

### Design principles

1. **Offline-first:** Once all required assets are explicitly provisioned, inference requires no network access.
2. **Small, transparent downloads:** Select quantized variants where appropriate; fetch only required assets and disclose total download size before installation.
3. **Performance by design:** Reuse model sessions and voice resources, limit concurrency, and measure performance on the actual target runtime.
4. **Typed, stable API:** The core package exposes no Hugging Face or ONNX implementation details.
5. **Modularity:** Engines are replaceable and tested independently; browser and Node.js support is verified rather than assumed.
6. **Explicit behavior:** No hidden downloads, silent model substitutions, backend changes, or quality downgrades.
7. **Minimal dependencies:** Keep the core lightweight and platform-independent; backend-specific dependencies belong in their respective engine packages.

## 2. Package Architecture

```text
Application (Node.js / Browser / Electron)
                     |
                     v
             @lite-speech/core
       +--------------------------------+
       | Public API and lifecycle       |
       | Text preprocessing             |
       | Streaming and cancellation     |
       | Offline and asset policies     |
       | Voice and model metadata       |
       | Errors and observability       |
       +---------------+----------------+
                       |
                       | TtsEngine interface
                       v
          @lite-speech/engine-onnx
       +--------------------------------+
       | Transformers.js                |
       | ONNX Runtime (WASM / WebGPU)    |
       | Phonon model adapter           |
       | Model and voice loading        |
       | Inference session lifecycle    |
       +---------------+----------------+
                       |
                       v
         Version-pinned local assets
       (ONNX / config / tokenizer / voices)
```

**Dependency rule:** `engine-onnx` depends only on the **public contracts** exported by `core`. The core must **never import** Transformers.js or a concrete engine. The application composes them using dependency injection.

### Package A: `@lite-speech/core`

The core defines a stable contract between applications and inference engines.

**Responsibilities**

- A small, consistent API: `createTts()`, `synthesize()`, `stream()`, and `dispose()`.
- A standard audio representation: **mono `Float32Array` PCM plus an explicit `sampleRate`**. Optional encoders, such as WAV, sit above the inference layer.
- A typed `TtsEngine` interface and capability discovery (for example, supported languages, voices, and streaming capabilities).
- **Streaming contract:** `AsyncIterable<AudioChunk>`, `AbortSignal`, bounded buffering/backpressure, and clear completion and error semantics.
- **Asset management contract:** model manifests, pinned revisions, required files, sizes, integrity checks, and cache state.
- **Offline policy:** provisioning can explicitly permit downloads; `offline: 'strict'` prohibits network access during inference.
- A small, testable text-processing pipeline for whitespace handling, sentence segmentation, and optional language-specific normalization.
- Structured errors and opt-in diagnostics for load progress, time to first audio, generation duration, and memory usage where available. Never transmit user text by default.

**Out of scope for core:** ONNX sessions, GPU buffers, private Transformers.js APIs, built-in browser playback, and direct filesystem-specific implementations. Platform capabilities should be accessed through narrow adapters.

### Package B: `@lite-speech/engine-onnx`

The ONNX engine implements the core contract using **Transformers.js** and its supported inference runtime.

**Responsibilities**

- A dedicated **Phonon adapter** as the first supported model implementation.
- Loading a **revision-pinned Hugging Face ONNX repository**, with only the required graph, quantization variant, configuration, tokenizer, and voice files.
- Choosing **WASM or WebGPU** based on verified execution-provider and operator compatibility — not merely device availability.
- Reusing initialized model sessions and relevant tokenizer/voice resources.
- Releasing inference sessions, GPU resources, and generator state during cancellation and `dispose()`.
- Validating model architecture, required assets, and runtime compatibility with actionable errors.
- Isolating version-specific workarounds so future native Phonon support can replace them without changing the core API.

**Best practice:** Prefer documented, public Transformers.js APIs. Do not build a permanent dependency on private tokenizer fields, internal tensor operations, or unpublished branches. If a published Transformers.js version cannot execute the required Phonon path reliably, treat it as a **release blocker for that feature**, not something to conceal behind an unstable fallback.

## 3. Phonon as the Initial Model Family

| Candidate | Intended use | Current considerations |
|---|---|---|
| [onnx-community/phonon-31m-en-ONNX](https://huggingface.co/onnx-community/phonon-31m-en-ONNX) | **English MVP** | ONNX export available; Apache-2.0; Transformers.js examples are documented, with version-dependent limitations. |
| [gradium/phonon-40m-de](https://huggingface.co/gradium/phonon-40m-de) | **German, later** | 40M-parameter model; gated; CC BY-NC 4.0; no directly usable Transformers.js ONNX export verified in this concept. |
| [gradium/phonon-40m-multilingual](https://huggingface.co/gradium/phonon-40m-multilingual) | **Optional, later** | Multilingual, gated, non-commercial; requires separate export, compatibility, and quality validation. |

For **Phonon 31M EN ONNX**, the model card reports approximate sizes for its two model graphs combined: **41 MB (`q4f16`)**, **67 MB (`q4`)**, or **161 MB (`fp32`)**. Tokenizer, configuration, voice assets, and runtime files are **additional**. Measure the full install size before declaring a profile suitable for a given download budget.

Phonon produces **24 kHz mono audio** and operates internally on latent frames of roughly **80 ms**. This does **not** automatically mean the high-level JavaScript API delivers low-latency, frame-by-frame audio. Distinguish genuine incremental model generation from sentence-level chunking in both the implementation and documentation.

### Compatibility and integration constraints

- **Python Transformers support is not equivalent to Transformers.js support.**
- An English Phonon 31M ONNX export and low-level JavaScript usage examples exist.
- At the reference date, native Phonon integration for the higher-level Transformers.js API was being developed in [PR #1811](https://github.com/huggingface/transformers.js/pull/1811) and [PR #1825](https://github.com/huggingface/transformers.js/pull/1825). **Do not assume** that `pipeline('text-to-speech', phononRepo)` works reliably in Transformers.js 4.3.2.
- The first engineering spike must demonstrate the required inference flow using a **pinned, published** Transformers.js release. If that fails, limit the MVP to a validated public low-level path or defer the feature.
- A **German Phonon profile** requires an independently verified, legally usable ONNX export. GGUF and Safetensors assets are not interchangeable with ONNX files.

### Licensing

- The **English Phonon 31M** export is listed under **Apache-2.0**.
- The referenced **Phonon 40M** models are **CC BY-NC 4.0** and access-restricted on Hugging Face; commercial usage cannot be assumed.
- Validate model, voice, and applicable data licenses separately. Access gated models only after accepting their terms. Never embed Hugging Face access tokens in shipped browser bundles.

## 4. Proposed Public API

This is an **illustrative target API**, not an existing implementation:

```ts
import { createTts } from '@lite-speech/core';
import { createOnnxEngine } from '@lite-speech/engine-onnx';

const tts = createTts({
  engine: createOnnxEngine({
    model: 'onnx-community/phonon-31m-en-ONNX',
    revision: '<verified-commit-sha>',
    device: 'auto',
    quantization: 'auto',
  }),
  offline: 'strict', // Assets must already be available locally.
});

// Model-independent PCM output.
const audio = await tts.synthesize('Hello from lite-speech.');
// audio: { pcm: Float32Array, sampleRate: number }

const controller = new AbortController();
for await (const chunk of tts.stream('Generated entirely offline.', {
  signal: controller.signal,
})) {
  // Forward chunk.pcm and chunk.sampleRate to an audio sink.
}

await tts.dispose();
```

**Separate provisioning step:** An operation such as `prepare({ allowNetwork: true, maxDownloadMB: 100 })` should download and verify required model and voice assets **before** strict offline inference. Provisioning should eventually be possible during a build or deployment step, without requiring a dedicated CLI package in the MVP.

A model that exceeds the configured budget should fail **before downloading**, with a clear explanation of the required files and total expected size.

## 5. Reliability and Performance Requirements

- **Downloads:** Explicit opt-in, immutable revision pins, verifiable checksums, atomic cache writes, selective asset fetching, and no unannounced network access.
- **Memory:** Reuse model sessions, cap concurrent inference work, keep caching bounded, and make `dispose()` deterministic.
- **Streams:** Support cancellation during generation, preserve chunk ordering, avoid unbounded queues, and document end-of-stream and failure behavior.
- **Hardware:** WASM as the compatibility baseline; WebGPU only after capability and operator checks. Choose quantization by **measured compatibility and performance**, not file size alone.
- **Audio:** Keep channel count and sample rate explicit; do not resample implicitly. Keep encoding and playback outside the engine.
- **Testing:** Unit-test the core with a fake engine; run integration tests with real models; maintain a supported Node.js/browser test matrix and offline tests with network access blocked.
- **Benchmarking:** Report download size, initialization latency, warm time-to-first-audio (p50/p95), real-time factor, peak RAM, cancellation latency, and listening-test outcomes. Always state hardware, runtime, model revision, and quantization.

### Important streaming distinction

A method returning `AsyncIterable<AudioChunk>` is **not proof** of low-latency incremental synthesis. Release documentation should state whether a specific model/backend provides true frame-level streaming, buffered chunk streaming, or only full-utterance output.

## 6. MVP Scope and Roadmap

### MVP — deliberately small

1. **`@lite-speech/core`** with the engine interface, basic text preprocessing, synthesis, `AsyncIterable` streaming, cancellation, and lifecycle management.
2. **`@lite-speech/engine-onnx`** with **one verified Phonon 31M ONNX profile** through Transformers.js.
3. Reproducible asset preparation, one strictly offline end-to-end test, and baseline benchmarks.
4. One Node.js example and one browser example using **the same core API**.

### Later, only when justified by validation

- Lower time-to-first-audio through genuinely incremental output, rather than sentence-based chunking alone.
- German Phonon support after ONNX compatibility, licensing, and quality checks.
- Richer text normalization (dates, numbers, currency) and integration with streamed LLM text.
- An optional provisioning CLI and additional engines/model families **without a breaking change to the core API**.

### Definition of done for the first public release

- A specific Phonon model runs reproducibly with pinned dependency and model versions in every officially supported runtime.
- A fully provisioned bundle runs with network access disabled; missing assets produce a clear error.
- Full download size, approximate memory requirements, and license information are documented for the supported model profile.
- Streaming, cancellation, and disposal are reliable and covered by automated tests.
- The core public API does not leak Transformers.js types and can be tested with a mock engine.
- Benchmarks are reproducible, and performance claims are evidence-based.

## 7. Explicitly Out of Scope for v1

Training or fine-tuning, voice cloning, automatic routing across many model families, a built-in audio player, a mandatory server, bundling model weights directly inside the npm package, and any promise of commercial rights for non-commercial Phonon models.

## 8. Technical References

- [ARCHITECTURE.md](./ARCHITECTURE.md) — proposed architecture (package split, engine contract, worker protocol, audio model, streaming, assets)
- [ONNX Community: Phonon 31M EN — model card, ONNX variants, examples](https://huggingface.co/onnx-community/phonon-31m-en-ONNX)
- [Gradium: Phonon 40M German — license and model metadata](https://huggingface.co/gradium/phonon-40m-de)
- [Gradium: Phonon runtime and browser library](https://github.com/gradium-ai/xn-ptts/blob/main/ptts-wasm/js/README.md)
- [Transformers.js PR #1811 — Phonon integration](https://github.com/huggingface/transformers.js/pull/1811)
- [Transformers.js PR #1825 — optimized Phonon integration](https://github.com/huggingface/transformers.js/pull/1825)
- [Transformers.js npm package](https://www.npmjs.com/package/@huggingface/transformers)

---

**Summary:** Build **two focused packages** around **one stable core contract** and a **Phonon-first ONNX engine**. Validate model compatibility, full offline operation, legal usage, and measurable performance before expanding the model catalog or feature set.
