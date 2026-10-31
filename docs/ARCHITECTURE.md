# Architecture of `lite-speech`

> **Status:** Proposed architecture (design baseline for the first implementation)
> **Date:** October 31, 2026
> **Reference implementation:** [`lite-translator`](https://github.com/realdeepnull/lite-translator) v0.3.0 (`C:\Users\laigh\Projekte\lite-translator`)
> **Product concept:** [lite-speech-concept.md](./lite-speech-concept.md)

---

## Table of Contents

- [1. Purpose and scope](#1-purpose-and-scope)
- [2. Architectural summary](#2-architectural-summary)
- [3. Core layer (`@lite-speech/core`)](#3-core-layer-lite-speechcore)
- [4. Engine layer (`@lite-speech/engine-onnx`)](#4-engine-layer-lite-speechengine-onnx)
- [5. Audio data model](#5-audio-data-model)
- [6. Streaming and cancellation](#6-streaming-and-cancellation)
- [7. Assets, offline policy, and provisioning](#7-assets-offline-policy-and-provisioning)
- [8. End-to-end control flow](#8-end-to-end-control-flow)
- [9. Cross-cutting design rules](#9-cross-cutting-design-rules)
- [10. Mapping from `lite-translator`](#10-mapping-from-lite-translator)
- [11. Build, packaging, and repository layout](#11-build-packaging-and-repository-layout)
- [12. Test and benchmark strategy](#12-test-and-benchmark-strategy)
- [13. Risks and open questions](#13-risks-and-open-questions)

---

## 1. Purpose and scope

`lite-speech` is a small, privacy-friendly, offline-capable text-to-speech library
for JavaScript and TypeScript. Its shape is deliberately copied from the sibling
project **`lite-translator`**, which already implements the same architecture
end-to-end: a model-agnostic core, a backend-specific engine, worker-isolated
inference, declarative asset manifests, cache inspection, lazy single-flight
loading, opt-in observability, and a stable error taxonomy.

This document describes the **intended** architecture of `lite-speech`
(`lite-speech` has no implementation yet). Where a decision mirrors
`lite-translator`, the reference file is named, so the first implementation can
follow a proven pattern instead of re-deriving one. Where speech synthesis needs
something that translation does not — audio buffers, voices, real streaming,
provisioning budgets — the difference is called out explicitly.

**In scope:** package split, module map, engine contract, lifecycle invariants,
audio types, streaming semantics, asset/offline handling, worker protocol,
model-profile extensibility, error model, observability, build layout, and
validation strategy.

**Out of scope:** product positioning and model licensing details (see
[lite-speech-concept.md](./lite-speech-concept.md)), training, voice cloning,
built-in audio playback, and a mandatory server.

---

## 2. Architectural summary

Two independently versioned and publishable npm packages plus one private
workspace root:

| Package                    | Depends on                                                        | Responsibility                                                                                                                                 |
| -------------------------- | ----------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| `@lite-speech/core`        | **nothing at runtime**                                            | Public API, `TtsEngine` contract, lifecycle, text preparation, streaming, asset/offline policy, errors, events                                 |
| `@lite-speech/engine-onnx` | `@huggingface/transformers` (runtime), `@lite-speech/core` (peer) | Transformers.js/ONNX inference in a Web Worker, configurable model profiles, model + voice registry, device/dtype resolution, cache management |
| root workspace             | private                                                           | Build/typecheck/lint/format/test/knip/size/pack orchestration, benchmark harness, demo                                                         |

The `@lite-speech` scope is a proposed name and must be confirmed before
publishing.

```text
Application (browser / Node.js / Electron)
        |
        v
  @lite-speech/core                  <-- no model, no ONNX, no Transformers.js
  +--------------------------------+
  | createTts / Tts                 |
  | TtsEngine contract              |
  | Text preparation + segmentation |
  | Streaming (AsyncIterable)       |
  | Asset & offline policy          |
  | Voice/model metadata            |
  | Emitter, errors, debug events   |
  +---------------+----------------+
                  |  TtsEngine (injected)
                  v
  @lite-speech/engine-onnx
  +--------------------------------+
  | TransformersEngine             |
  | configurable model profiles    |
  | device/dtype resolution        |
  | static model + voice registry  |
  | Cache Storage management       |
  +---------------+----------------+
                  |  postMessage protocol
                  v
  Web Worker (worker.js)
  +--------------------------------+
  | @huggingface/transformers      |
  | TTS pipeline                    |
  | ONNX Runtime: WASM / WebGPU     |
  +---------------+----------------+
                  |
                  v
   Version-pinned assets on the HF Hub
   (ONNX graphs, configs, tokenizer, voice embeddings)
```

**Dependency rule (to be enforced, not only documented):**

- `core` imports nothing from `engine-onnx`, and has **zero runtime
  dependencies** — `packages/core/package.json` will list dev tooling only.
- `engine-onnx` imports only the **public** surface of `@lite-speech/core`
  (`src/index.ts` via TS `paths`, `dist` for consumers), never internal modules.
- The application composes both packages; nothing auto-imports a model.
- The engine contract is the **only** coupling point between the layers.

As in `lite-translator` (`packages/core/src/engine.ts`), the engine is passed
explicitly through `TtsOptions.engines`, with a global registry
(`registerDefaultEngine` / `getDefaultEngines`) as an optional convenience so a
meta package could register an engine on import.

---

## 3. Core layer (`@lite-speech/core`)

### 3.1 Module map

| Module       | Public surface                                                                                                                      | Role                                                                                              |
| ------------ | ----------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| `tts.ts`     | `createTts`, `Tts`, `PrepareAssetsOptions`                                                                                          | Owns one engine, one offline policy, and the lifecycle; orchestration, state, error normalization |
| `engine.ts`  | `TtsEngine`, `registerDefaultEngine`, `getDefaultEngines`, `clearDefaultEngines`, `assertTtsEngine`, `isTtsEngine`                  | Engine SPI and default-engine registry                                                            |
| `types.ts`   | All DTOs (`AudioResult`, `AudioChunk`, `VoiceInfo`, `ProgressEvent`, `DebugEvent`, `TtsCapabilities`, …)                            | Pure data contracts, no behaviour                                                                 |
| `prepare.ts` | `prepareText`, `segmentText`, `DEFAULT_MAX_CHARS_PER_SEGMENT`                                                                       | Whitespace/text normalization and sentence segmentation                                           |
| `stream.ts`  | `createAudioStream`, `AudioStreamOptions`                                                                                           | Bounded-queue `AsyncIterable<AudioChunk>` with backpressure and contract checks                   |
| `assets.ts`  | `AssetPlan`, `ModelManifest`, `VoiceManifest`, `createAssetPlan`, `estimateDownload`, `assertBudget`, `missingAssets`, `cacheState` | Asset/offline policy, budget checks, cache-state helpers                                          |
| `pool.ts`    | `TtsPool`, `ttsPoolKey`                                                                                                             | Multi-voice/multi-language instance cache, single-flight creation, LRU eviction                   |
| `wav.ts`     | `encodeWav`                                                                                                                         | Optional PCM → WAV encoder, layered **above** the inference contract                              |
| `emitter.ts` | `createEmitter`, `Emitter`, `ListenerMap`                                                                                           | Typed event emitter (no `EventTarget`, so Node tests work)                                        |
| `errors.ts`  | `TTS_ERROR_CODES`, `TtsError`, `isTtsError`, `isAbortError`, `normalizeTtsError`, `formatTtsError`                                  | Stable error taxonomy                                                                             |
| `index.ts`   | Re-exports                                                                                                                          | The only public entry point                                                                       |

Every public export is listed in `packages/core/src/index.ts`; `knip` enforces
that no source file or export is orphaned — same policy as
`lite-translator/packages/core/src/index.ts`.

### 3.2 `Tts` lifecycle

`createTts()` returns synchronously, matching the illustrative API in the
concept (`const tts = createTts({ … })` without `await`). Unlike
`createTranslator()`, which returns a promise because engine lookup by language
pair is part of creation, TTS has no routing decision that would justify an
async boundary: the engine is injected and voice support is answered
synchronously.

1. Resolve the engine: `options.engines?.[0] ?? getDefaultEngines()[0]`.
2. Throw `TtsError(ENGINE_NOT_SUPPORTED)` when no engine is available.
3. Validate the requested voice/language against `engine.supports()`; throw
   `VOICE_NOT_SUPPORTED` / `LANGUAGE_NOT_SUPPORTED` immediately.
4. Return a `Tts` bound to that engine. **No model is downloaded here.**

State inside a `Tts` is deliberately small (`#`-prefixed names mean the state
stays truly private, as in `lite-translator`):

```text
#engine        TtsEngine               immutable
#offline       "strict" | "allow"      immutable policy
#onProgress    ProgressCallback?       forwarded to engine.load()
#onDebug       DebugCallback?          opt-in, zero cost when absent
#ready         boolean                 model loaded
#disposed      boolean                 terminal flag
#loadPromise   Promise<void>?          single-flight preload
#activeStreams Set<AbortController>    disposed on dispose()
```

Key invariants carried over from
`lite-translator/packages/core/src/translator.ts`:

- **Lazy loading with single-flight preload.** `preload()` memoizes the load
  promise. A _rejected_ promise is deliberately un-memoized, so a transient
  failure (offline blip, aborted download) does not permanently poison the
  instance; the next call retries.
- **Every operation re-checks `disposed` and `signal.aborted` after every
  `await`**, because disposal or an abort can happen while the model loads.
- **Every operation re-checks `disposed` and `signal.aborted` after every
  `await`**, because disposal or an abort can happen while the model loads.
- **Errors are normalized at the boundary.** Only `TtsError` passes through
  unchanged; anything else becomes a `TtsError` with a stable code and the
  original error as `cause`.
- **`dispose()` is idempotent**, aborts active streams, and disposes the engine.
- **`offline: "strict"` is enforced in core**, not trusted to the engine: a
  synthesis call that requires a download rejects with
  `OFFLINE_ASSETS_MISSING` before the engine is invoked. Core prefers
  `describeAssets()` (it can name the missing files) and falls back to the
  required `isCached()`.
- **Whitespace-only input short-circuits**: `synthesize()` resolves an empty
  result and `stream()` yields one empty `final` chunk, both without loading the
  model.
- **No implicit resampling, no implicit playback.** Core never touches
  `AudioContext`, so the library stays usable in Node and in workers.

### 3.3 The `TtsEngine` contract

The architectural hinge: core knows nothing about ONNX, Transformers.js, WASM,
WebGPU, workers, or HTTP.

Required members:

| Member                                 | Contract                                                                 |
| -------------------------------------- | ------------------------------------------------------------------------ |
| `id`                                   | Stable engine identifier (e.g. `"onnx"`), echoed in `AudioResult.engine` |
| `supports(request)`                    | **Synchronous** capability check (language + voice)                      |
| `voices()`                             | **Synchronous** list of `VoiceInfo` for the configured model             |
| `isCached(request)`                    | Whether **all** model _and_ voice assets are locally available           |
| `load(request, onProgress?, onDebug?)` | Idempotent; may be called repeatedly without a second download           |
| `synthesize(text, request, options?)`  | Lazily loads; honours `AbortSignal`; returns full-utterance audio        |
| `stream(text, request, options?)`      | Lazily loads; returns `AsyncIterable<AudioChunk>`; honours `AbortSignal` |
| `dispose()`                            | Releases inference sessions, GPU resources, and worker state             |

Optional members:

| Member                          | Contract                                                 |
| ------------------------------- | -------------------------------------------------------- |
| `capabilities()`                | Resolved `device`, `dtype`, `modelId`, `modelVersion`    |
| `removeModel(request)`          | Evicts cached assets from every cache it knows about     |
| `describeAssets(request)`       | Per-file cache state and total size (for UI/consent)     |
| `prepare(request, onProgress?)` | Explicit, opt-in provisioning (download + verify)        |
| `assetPlan(request)`            | Declarative list of required files with sizes and hashes |

Why `supports()` and `voices()` are synchronous: composition must be possible
without an async negotiation round trip, exactly as `TranslationEngine.supports(pair)`
is synchronous in `lite-translator/packages/core/src/engine.ts`.

Why `stream()` lives on the engine and not only in core: only the engine knows
whether the model can emit genuine incremental frames. Core must not fake
low-latency streaming by sentence-chunking a full-utterance result and
presenting it as frame-level synthesis.

### 3.4 Text preparation and segmentation

`prepareText()` and `segmentText()` are pure functions with no dependencies:

```text
input text
   |
   v
normalize (whitespace collapse, optional language normalization)
   |
   v
segment at . ! ? ; and newlines  -->  complete sentences
   |                                   trailing remainder -> final segment
   v
apply maxCharsPerSegment (split over-long sentences by clause)
   |
   v
string[] (order preserved, never empty except for empty input)
```

Rules:

- Segmentation is **deterministic and testable in isolation** — it must not
  depend on the model.
- Sentence boundaries are a _latency_ tool, not a streaming guarantee: documents
  must state whether a given model profile yields true incremental audio or
  sentence-level chunking.
- Normalization never removes content (no silent truncation); numbers, dates, and
  currency expansion is a later, opt-in stage.
- Empty or whitespace-only input resolves to an empty result without invoking
  the model.

### 3.5 `TtsPool` — multi-voice management

Replaces the `Map<string, Tts>` + manual `switchTo()` pattern that would
otherwise be re-implemented in every framework integration. Mirrors
`lite-translator/packages/core/src/pool.ts`:

- `switchTo({ language, voice })` returns the cached instance immediately, or
  creates one.
- **Single-flight creation**: concurrent `switchTo()` calls for the same key
  share one promise (no duplicate workers, no duplicate downloads).
- **LRU by `Map` insertion order** with optional `maxSize`; eviction is
  fire-and-forget so `switchTo()` never blocks on tearing down an unused model.
- `current()`, `get()`, `cachedKeys()`, `size`, `disposeKey()`, `dispose()`.
- A `dispose()` racing a creation disposes the freshly created instance instead
  of leaking it; `dispose()` uses `Promise.allSettled` and re-throws the first
  failure.

---

## 4. Engine layer (`@lite-speech/engine-onnx`)

### 4.1 Module map

| Module                   | Role                                                                                                                         |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------------------- |
| `create-onnx-engine.ts`  | Factory: builds the registry from defaults + overrides, resolves device/dtype once, async because a WebGPU probe is          |
| `transformers-engine.ts` | `TransformersEngine implements TtsEngine` — worker lifecycle, request correlation, cancellation, cache checks, error mapping |
| `model-host.ts`          | Everything the worker does except post messages: single model, single-flight loads, one stream session, cooperative cancel   |
| `worker.ts`              | The worker entry: serialized message queue and protocol plumbing                                                             |
| `worker-protocol.ts`     | The request/response message contract shared by the engine and the worker                                                    |
| `phonon.ts`              | The graph runner: `flow_lm` + `mimi_decoder`, public tokenizer encoding, per-frame generation                                |
| `models.ts`              | `ENGINE_ID = "onnx"`, the pinned model and voice manifests, `createDefaultRegistry()`                                        |
| `profiles.ts`            | `ModelProfile` type, the built-in Phonon profile, text and frame-budget rules                                                |
| `webgpu.ts`              | `detectWebGpu`, `isFp16Supported`, `resolveDeviceDtype`, device/dtype types                                                  |
| `node.ts`                | Node.js entry: in-thread inference plus a filesystem cache probe                                                             |
| `index.ts`               | Public engine surface                                                                                                        |
| `./worker` subpath       | `./worker` → `dist/worker.js`, instantiated via `new Worker(new URL("./worker.js", import.meta.url), { type: "module" })`    |

`profiles.ts` exists so that model-family specifics stay behind a single,
replaceable seam. A `ModelProfile` describes everything the generic worker must
not hard-code:

```ts
interface ModelProfile {
  id: string; // e.g. "phonon"
  streaming: StreamingClass; // truthful classification, surfaced in capabilities
  frameRate?: number; // 12.5 for a frame-based model
  prepareSegment(segment: string): string; // measured text shaping
  maxFrames(prepared: string, inputLength: number, config: PhononConfig): number; // runaway guard
}
```

Two things deliberately stay **out** of the profile. The graph driving lives in
`phonon.ts`, which only the worker bundle can reach; and the profile has no
`build(pipeline)` step, because the pinned Transformers.js release has no class
for this architecture, so a profile is configuration over one runner rather than a
class factory. The worker resolves a profile by the manifest's `family` (falling
back to the built-in one), and an _unknown_ id is an explicit error rather than a
silent Phonon fallback — running a different family through these input shapes
would produce noise instead of an error. Built-in profiles carry their own frame
sizes, conditioning inputs, and PCM normalization; version-specific workarounds
stay inside them.

### 4.2 Device and dtype policy

The default is deliberately boring: **`device: "wasm"`** (or `"cpu"` under
`onnxruntime-node`), with the dtype chosen from the measured compatibility matrix
for the selected model profile. WebGPU is opt-in (`"auto"` probes `navigator.gpu`,
`"webgpu"` requires it and throws otherwise). `createOnnxEngine()` is therefore
**async**: a WebGPU probe cannot be awaited from the synchronous `createTts()`
boundary, so resolution happens once, in the factory, and the engine it returns is
synchronous like every other engine.

```text
device "wasm"    -> { wasm, dtype ?? "fp32" }   (no probe)
device "cpu"     -> { cpu,  dtype ?? "fp32" }   (no probe)
device "webgpu"  -> probe adapter; throw if absent; q4 default, q4f16 only with shader-f16
device "auto"    -> webgpu if an adapter answers, else the environment default + device-fallback
```

Rules inherited from `lite-translator` (`src/webgpu.ts`) and extended for TTS:

- The adapter probe is **memoized including rejections**; `requestAdapter()` can
  cost tens of milliseconds and must not run twice per `load()`.
- An explicit `"webgpu"` **never silently retries** on WASM. Only `"auto"` may
  fall back, and it reports `fallbackFrom: "webgpu"` so the caller can surface it.
- **An accepted `webgpu` request is not evidence of GPU execution.** In Node.js the
  runtime reports `webgpu` as a supported device and an explicit WebGPU load
  _succeeds_ without `navigator.gpu` (M1 §5.1). The probe therefore gates on
  `navigator.gpu` **and** adapter features, never on the runtime accepting a request.
- `q4f16` is rejected up front without `shader-f16`, and `"auto"` silently demotes it
  to `q4` on that adapter — an explicit request must not appear to work when it
  cannot.
- Quantization is chosen by **measured quality and performance**, never by file
  size alone. On CPU/WASM every quantized variant measured 20–26 % slower warm, so
  `fp32` is the default there and `q4` is a documented, cheaper opt-in.
- The resolved dtype also drives asset URLs (`_q4.onnx`, `_q4f16.onnx`, …) for cache
  checks and cache eviction, since Transformers.js selects files by dtype suffix.

### 4.3 Worker protocol

The worker is a single-model, single-threaded inference host behind an
`id`-correlated request/response protocol. It follows
`lite-translator/packages/engine-onnx/src/worker.ts` and adds **pull-based**
streaming:

```text
engine -> worker                          worker -> engine
{ kind: "load", id, modelId,       { kind: "progress", id, event }        (during load only)
  revision, device, dtype,        { kind: "capabilities", id, device, dtype,
  profile, voice }                  modelId, modelVersion, sampleRate,
                                    streaming }
                                   { kind: "loaded", id }

{ kind: "synthesize", id, text,    { kind: "inference-start", id, textLength }
  voice, temperature }             { kind: "inference-done", id, durationMs,
                                     samples, sampleRate }
                                   { kind: "result", id, pcm, sampleRate,
                                     frames, durationMs }                  (full utterance)

{ kind: "stream-open", id, text,   { kind: "stream-opened", id }
  voice, temperature }            { kind: "chunk", id, pcm, sampleRate }   (one per pull)
{ kind: "stream-next", id }        { kind: "stream-end", id }              (nothing left)
{ kind: "stream-close", id }       (no response)

{ kind: "cancel", id }             { kind: "cancelled", id }
{ kind: "cache-check", id,         { kind: "cache-state", id, cached, sizes }
  modelId, revision, files }
{ kind: "dispose", id }            { kind: "disposed", id }
                                   { kind: "error", id, message, code? }
```

Why it is shaped this way:

- **Inference off the main thread.** Model download, graph instantiation, and
  decoding never block rendering or input handling.
- **Strict serialization.** `self.onmessage` is async, so every message is
  chained onto a single `queue` promise; otherwise a `load` awaiting a download
  could interleave with a `synthesize` against a half-initialized pipeline.
  `cancel` deliberately bypasses the queue: it must take effect while a long
  generation is still running.
- **`ArrayBuffer` transfer, not copy.** Audio chunks are posted with a transfer
  list so multi-megabyte `Float32Array` payloads are moved, not cloned.
- **Pull-based streaming.** The engine opens a session and then asks for one frame
  per consumer turn. Backpressure is therefore structural — the worker cannot
  outrun the consumer because it is never asked to — so memory stays at one frame
  plus model state with no credit accounting, and every `stream-next` doubles as a
  cancellation checkpoint. A push protocol would need an explicit high-water mark
  in the worker; this one cannot buffer unboundedly by construction.
- **The engine owns `final`.** The worker cannot know a frame is last until its
  generator ends, so `stream-end` is the signal and the engine emits the single
  `final: true` chunk. Peeking one frame ahead would double time-to-first-audio.
- **One model per worker.** Loading a second model throws; the engine therefore
  disposes the previous model before switching, which lets one worker serve
  many voices over time.
- **`env.allowLocalModels = false`.** An SPA dev-server fallback returns
  `index.html` with HTTP 200 for unknown URLs; Transformers.js would parse that
  HTML as an ONNX protobuf. Remote HF Hub models only.
- **Failed loads are not memoized** in the worker either (`loadPromise`
  resets), mirroring the core-level retry semantics.
- **`handleDispose` resets all state in a `finally`** and always acknowledges, so
  the engine never waits forever and the next `load()` never sees a torn-down
  pipeline. The engine's own `dispose()` additionally bounds that wait, because a
  wedged worker must not keep the call pending; `terminate()` is what actually
  guarantees the resources are gone.
- **Cancellation is cooperative.** The worker cannot stop mid-inference; it
  checks a cancel flag between emitted frames, and the engine drops any late
  frames whose request is no longer pending. Because the engine never has more
  than one pull outstanding, "late" means "after `cancel`", not "after a race".

### 4.4 Engine-side request, ordering, abort, and cache logic

- **`#pending: Map<id, PendingRequest>`** correlates responses; `progress` and
  `capabilities` go to `onEvent` and keep the request pending; `audio-chunk`
  feeds the stream queue; `result`/`error`/`disposed` settle the promise.
- **`worker.onerror` rejects all pending requests** and clears the map; a worker
  crash cannot leave callers hanging.
- **Abort** rejects the caller's promise, deletes the pending entry, and lets
  late worker messages be dropped by the `!pending` guard. The listener is
  removed in `finally`.
- **`isCached()`** checks Cache Storage in parallel for **all** descriptor URLs
  (graph, configs, tokenizer, **and voice files**), recomputing ONNX URLs from
  the resolved dtype once known.
- **`removeModel()`** disposes the loaded model first, then deletes the same URL
  set from _every_ Cache Storage cache (Transformers.js may use any cache name;
  applications have few caches).
- **Error mapping**: offline detection (`navigator.onLine === false` or a
  network-error message pattern) plus a cache check yields
  `OFFLINE_ASSETS_MISSING`; otherwise `MODEL_DOWNLOAD_FAILED` or
  `MODEL_LOAD_FAILED`, decided by whether a progress event indicated a download
  had started. A missing/invalid voice file yields `ASSET_INTEGRITY_FAILED`.
- **Node.js support**: when `Worker` is unavailable the engine falls back to
  in-thread inference behind the same private `#dispatch()` boundary, and when
  `Cache Storage` is unavailable `isCached()` reports from the runtime's cache
  adapter instead of returning `false` unconditionally. This is a deliberate
  deviation from `lite-translator`, where Node is only a test environment.

### 4.5 Model and voice registry

`createDefaultRegistry()` maps a language to a core `ModelManifest` and every voice
to a `VoiceManifest`. Both are the core's own types, not an engine-private shape, so
budgets and offline state stay engine-agnostic:

```text
ModelManifest {
  id: "onnx-community/phonon-31m-en-ONNX",
  revision: "<40-char commit sha>",   // immutable pin, never "main"
  license: "Apache-2.0",
  family: "phonon",                   // selects the ModelProfile used to run it
  sampleRate: 24000,                  // declared per model, never assumed globally
  files: [
    { path: "config.json",         size: 4952 },
    { path: "tokenizer.json",      size: 303009 },
    { path: "tokenizer_config.json", size: 144 },
    { path: "onnx/flow_lm.onnx",   size: 120089751 },   // fp32
    { path: "onnx/mimi_decoder.onnx", size: 41267821 },
    ... every quantized variant, so core can price any of them
  ]
}

VoiceManifest {
  id, modelId, language: "en", revision, license: "Apache-2.0",
  sampleRate: 24000,
  files: [{ path: "voices/<voice>.bin", size: 256000 }]
}
```

The manifest lists **every** dtype variant; the engine reports only the files its
resolved dtype loads. Otherwise `isCached()` would demand weights that are never
read and `removeModel()` would delete weights belonging to another configuration.

Every registered voice reports the language `"en"`. The repository's voices have
design accents (`en-US`, `en-GB`, …), but this profile does not switch
pronunciation, so advertising `"en-GB"` as a _language_ would make
`supports("en-GB")` a promise it cannot keep; the accent travels as a label in
`VoiceInfo.description` instead.

Three entry points for customization:
`createOnnxEngine({ models })` for a different model map,
`createOnnxEngine({ registry })` for a fully custom registry, and
`createOnnxEngine({ profiles })` to register additional `ModelProfile`
implementations. Selecting a model therefore never requires a new package, and
adding a model family never changes the core API.

The registry is a **declarative asset manifest** — URLs, optional `size`,
optional `sha256` — which is what lets core reason about download budgets and
offline state without knowing the backend.

---

## 5. Audio data model

The inference contract deals exclusively in **plain PCM**:

```ts
interface AudioResult {
  pcm: Float32Array; // mono, interleaved-free, range -1..1
  sampleRate: number; // explicit, never assumed (declared per model manifest)
  channels: 1; // literal — stereo is an encoder concern
  durationMs: number;
  engine: string;
  voice?: string;
}

interface AudioChunk {
  pcm: Float32Array;
  sampleRate: number;
  sequence: number; // monotonic, gap-free per stream
  final: boolean;
}
```

Rules:

- **Channel count and sample rate are always explicit.** No implicit
  resampling; a consumer that needs 48 kHz resamples itself or uses an optional
  helper.
- **Encoding is layered above the contract.** `encodeWav(result)` lives in
  `core/wav.ts` (and is tree-shakeable); the engine returns PCM only.
- **Playback is never part of the library.** `AudioContext`, `AudioWorklet`, and
  platform sinks stay in the application.
- **Ownership is explicit.** Chunk buffers transferred from the worker are owned
  by the consumer; the engine never mutates a buffer after posting it.

---

## 6. Streaming and cancellation

`stream()` is a first-class contract, not a convenience wrapper:

```text
consumer                 core                       engine / worker
  |  for await (chunk)     |                            |
  |----------------------->|  (lazy load if needed)     |
  |                        |--------------------------->|
  |                        |   <-- audio-chunk(seq=0) --|
  |  <-- chunk 0 ----------|                            |
  |  (await next)          |   <-- audio-chunk(seq=1) --|
  |  <-- chunk 1 ----------|                            |
  |  ...                   |   <-- result(final) ------ |
  |  <-- done              |                            |
```

Semantics:

- `AsyncIterable<AudioChunk>` with a **bounded queue** (`highWaterMark` in
  frames or milliseconds). When the consumer is slow, the engine stops pulling
  from the model instead of buffering without limit — this is why core owns the
  queue and the engine only produces.
- **Ordering is guaranteed** by the monotonic `sequence`; a gap or duplicate is
  a contract violation, not a tolerated edge case, so it is asserted in tests.
- **Completion** is signalled by exactly one `final: true` chunk; the iterator
  then completes. Errors surface through the iterator and also reject a
  concurrent `synthesize()` if one is outstanding for the same request.
- **Cancellation is cooperative and observable.** `AbortSignal` aborts stop
  scheduling further segments, mark in-flight work as dropped, and terminate the
  iterator with `TtsError(ABORTED)`. Audio already delivered stays valid —
  partials are never "un-delivered", and the documentation states that
  explicitly.
- **Backpressure is mandatory**, because TTS generation can be slower than
  realtime on WASM and much faster than realtime on WebGPU; both cases must not
  grow memory.
- **Truthfulness about streaming.** Release notes and the API docs must state,
  per model profile, whether the backend delivers true incremental frames,
  buffered sentence chunks, or only full-utterance output. An `AsyncIterable`
  return type is not evidence of low time-to-first-audio.

---

## 7. Assets, offline policy, and provisioning

Offline capability is a first-class state, not an afterthought:

| Concern               | Design                                                                                                                |
| --------------------- | --------------------------------------------------------------------------------------------------------------------- |
| Network policy        | `offline: "strict"` (default) forbids network access during inference; `offline: "allow"` permits on-demand downloads |
| Explicit provisioning | `await tts.prepare({ allowNetwork: true, maxDownloadMB: 120 })` before strict-offline use                             |
| Budget failures       | A profile that exceeds `maxDownloadMB` fails **before downloading**, listing required files and total size            |
| Revision pinning      | Every asset URL resolves to an immutable commit SHA; `main` is never used in a shipped default                        |
| Integrity             | Optional `sha256` per file, verified at (or immediately after) download; mismatch → `ASSET_INTEGRITY_FAILED`          |
| Atomicity             | Writes are all-or-nothing per model profile; a partial download never reports as cached                               |
| Inspection            | `isCached()` covers model **and** voice files; `describeAssets()` reports per-file cache state and total size         |
| Eviction              | `removeModel()` clears every relevant cache after disposing the loaded session                                        |
| Transparency          | `estimateDownload()` returns the byte total the UI must show before a user consents                                   |

`prepare()` is intentionally a core-level operation that delegates to the
engine's optional `prepare()`/`assetPlan()` members, so the policy (budget,
consent, offline transitions) lives in one place while the URLs remain
engine-specific. A CLI is a later addition and must not be required for the MVP.

---

## 8. End-to-end control flow

### 8.1 First synthesis (cold start)

```text
app: createTts({ engine: createOnnxEngine({...}), offline: "strict" })
  core: validate engine + voice                -> Tts (no download)

app: tts.synthesize("Hello world")
  core: offline strict? -> engine.isCached()  -> false -> OFFLINE_ASSETS_MISSING
        (or, after tts.prepare(), continue)

  core: not ready -> preload()
          engine.load(request, onProgress, onDebug)
                   resolveDeviceDtype()             (debug: device-resolved)
                   ensure worker                    (debug: worker-spawn)
                   post {load, modelId, device, dtype, voice}
          worker: pipeline(...) downloads/loads -> progress events forwarded
                  {capabilities}, {loaded}          (debug: load-done)

  core: engine.synthesize(text, request, {signal})  (debug: synthesize-start)
          engine: post {synthesize, id, text, voice}
          worker: {inference-start}, run profile, {result}, {inference-done}
  core: wrap in AudioResult, normalize errors       (debug: synthesize-done)
```

### 8.2 Streaming with cancellation

```text
app: controller = new AbortController()
     for await (chunk of tts.stream(longText, { signal: controller.signal }))
       sink.push(chunk)                  // consumer-driven pull
app: controller.abort()
  core: mark segments unscheduled, abort in-flight request
  engine: reject pending promise, drop late audio-chunk messages
  worker: finishes or discards current inference, stays alive for reuse
  core: iterator terminates with TtsError(ABORTED)
```

### 8.3 Voice switching

`ttsPool.switchTo({ language: "en", voice: "female-1" })` → cached instance or
created one → `load()` → worker `{dispose}` → worker `{load}` with the new voice
assets. One worker, sequentially reloaded, unless the application creates
several engines or pools.

---

## 9. Cross-cutting design rules

These rules are the reusable core of the `lite-translator` architecture and
apply unchanged to `lite-speech`:

1. **Core has zero runtime dependencies.** Anything backend-specific lives in an
   engine package. `size-limit` budgets the core bundle.
2. **The engine interface is the only coupling point**, and capability checks
   (`supports`, `voices`) are synchronous so composition needs no async
   negotiation.
3. **Lazy, idempotent, failure-forgiving loading.** No import-time downloads;
   rejected load promises are never cached; `dispose`/abort are re-checked after
   every `await`.
4. **Single-flight everywhere.** `preload()`, `TtsPool` creation, and the worker
   `loadPromise` all deduplicate concurrent work.
5. **Concurrency correctness by monotonic sequencing**, because the backend
   cannot cancel mid-inference. "Cancel" means "stop waiting and drop the
   result", plus an `AbortSignal` to shed future work.
6. **Explicit bounds everywhere**: stream queue depth, segments per request,
   characters per segment, download budget, concurrency cap.
7. **Opt-in observability** with typed discriminated-union events and a `?.(…)`
   call site, so production pays nothing for diagnostics.
8. **Stable error codes with `cause` chaining**; errors are normalized once, at
   the public boundary.
9. **Declarative asset manifests** with dtype-dependent URL derivation, plus
   explicit cache inspection and eviction.
10. **Offline is a first-class state** (`isCached`, `OFFLINE_ASSETS_MISSING`,
    cache removal) — not an afterthought.
11. **Worker isolation with a serialized message queue** and a single active
    model, keeping memory bounded and logic simple.
12. **No silent substitution**: never swap model, voice, backend, dtype, or
    quality without surfacing it (an event and a capability report).

---

## 10. Mapping from `lite-translator`

Direct analogues (rename, keep the shape):

| `lite-translator`                                                                                               | `lite-speech`                                                               |
| --------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| `createTranslator({ from, to })` → `Translator`                                                                 | `createTts({ engine, offline })` → `Tts`                                    |
| `TranslationEngine` (`translate`, `translateBatch`)                                                             | `TtsEngine` (`synthesize`, `stream`)                                        |
| `TranslationResult { text, from, to, engine }`                                                                  | `AudioResult { pcm, sampleRate, channels, engine, voice }`                  |
| `LanguagePair` + engine registry                                                                                | language + **voice** selection and voice manifest                           |
| `ModelRegistry` / `ModelDescriptor`                                                                             | `ModelManifest` / `VoiceManifest` (pinned revisions, sizes, hashes)         |
| `ProgressEvent { phase, loaded, total, progress }`                                                              | same, plus voice-asset phases                                               |
| `LiveSession` (segments, debounce, sentence cache → **no direct analogue**: there is no user-incremental input) | segmentation now lives in `core/prepare.ts` and feeds `stream()`            |
| `TranslationStore` / `t()`                                                                                      | not needed — no i18n key/value store in TTS                                 |
| `TranslatorPool`                                                                                                | `TtsPool`; same LRU + single-flight semantics                               |
| `onDebug` `DebugEvent` union                                                                                    | diagnostics: load, time-to-first-audio, generation duration, frames, memory |
| `AbortSignal`                                                                                                   | same, plus explicit partial-audio semantics                                 |
| WASM default, opt-in WebGPU                                                                                     | same policy, with a heavier model and stricter operator checks              |
| `encode` nothing (text out)                                                                                     | `encodeWav` layered above the inference contract                            |

Deliberate differences to plan for:

- **Audio, not text, is the result type** — mono `Float32Array` plus an explicit
  `sampleRate`; encoders sit above the contract.
- **Streaming is a contract with backpressure**, not an incremental convenience.
- **Voices are assets too** — manifests must describe, verify, and cache them.
- **Long-running generation** makes cancellation semantics a contract question:
  what happens to partially synthesized audio must be documented and tested.
- **Node.js is a supported runtime**, not only a test environment, so the engine
  must degrade gracefully without `Worker`, `Cache Storage`, or WebGPU.
- **Legal metadata travels with the manifest** (`license` per model and voice),
  because model families differ (Apache-2.0 vs. CC BY-NC 4.0).

---

## 11. Build, packaging, and repository layout

```text
packages/core            tsdown -> dist/index.js + index.d.ts (browser/neutral, ESM, dts)
packages/engine-onnx     tsdown -> dist/index.js + dist/worker.js (+ d.ts)   [platform: browser]
                                   deps.alwaysBundle: ["@lite-speech/core",
                                                       "@huggingface/transformers"]
                         tsdown -> dist/node.mjs + dist/node.d.mts           [platform: node]
                                   deps.alwaysBundle: ["@lite-speech/core"]
examples/demo/index.html  bundler-free demo against the built ESM output
bench/benchmark           cold start, time-to-first-audio, RTF, memory; JSON reports
bench/quality             listening-test corpus, determinism, chunk-consistency
```

Packaging conventions carried over from `lite-translator`:

- ESM only, `"type": "module"`, `exports` map with `types` + `import`.
- `"sideEffects": false` so bundlers can tree-shake.
- `files: ["dist", "README.md", "LICENSE"]`; `publishConfig.access: "public"`.
- The engine has **two builds**, because the two runtimes disagree about the
  platform. The browser build bundles core and Transformers.js inline — a browser
  cannot resolve a bare specifier inside a worker, and the demo loads native ESM
  without a bundler, which is why `worker.js` is a separate `exports` subpath. The
  Node build targets `node22`, keeps the `node:` builtins external, and leaves
  Transformers.js as an ordinary dependency rather than inlining 1.5 MB into it.
- `core` and `engine-onnx` are versioned independently; `engine-onnx` declares
  `@lite-speech/core` as a **peer** dependency with a compatible range.
- Bundling core into the engine means an application can end up with two copies of
  the core module. That is why `isT.tsError()` is a **structural** guard (`name` +
  `code` + `message`) rather than `instanceof`: with two class objects, an
  `instanceof` check would reject the engine's `ABORTED` and the core boundary would
  re-wrap it as `STREAM_FAILED`. That guard is `isTtsError()`, and a core test asserts
  a foreign-shaped `TtsError` still passes it.
- `tsconfig.base.json` is strict: `strict`, `noUncheckedIndexedAccess`,
  `exactOptionalPropertyTypes`, `verbatimModuleSyntax`, `isolatedModules`.
  `exactOptionalPropertyTypes` shapes the API style — optional properties are
  spread conditionally (`...(x !== undefined && { x })`) rather than set to
  `undefined`.

Repository-level scripts (all workspaces, `--if-present`), matching
[AGENTS.md](../AGENTS.md):

| Script                  | Purpose                                                          |
| ----------------------- | ---------------------------------------------------------------- |
| `build`                 | `tsdown` per package                                             |
| `typecheck`             | `tsc --noEmit`                                                   |
| `lint` / `format:check` | ESLint + Prettier                                                |
| `test`                  | core: Vitest in `node`; engine: Vitest browser mode + Playwright |
| `knip`                  | unused files, exports, dependencies (workspace-aware entry list) |
| `size`                  | `size-limit` per package                                         |
| `pack:check`            | build + `publint` + `attw` on the packed tarball                 |

---

## 12. Test and benchmark strategy

**Unit tests (Node, `environment: "node"`).** Core is testable in Node only
because it never touches DOM-only APIs; the custom emitter and the
`performance`/`Date` fallbacks exist for exactly this reason. A fake engine
covers lifecycle, offline policy, budget failures, segmentation, stream
backpressure, abort semantics, pool single-flight and eviction, error
normalization, and WAV encoding.

**Integration tests (real browser).** Workers, `Cache Storage`, `navigator.gpu`,
and ONNX/WASM/WebGPU have no Node equivalent, so engine tests run in Vitest
browser mode with Playwright/Chromium, headless, with long timeouts, and are kept
separate from the fast unit suite. Coverage focus: load/reload/switch, download
progress, cache detection and eviction, offline (`network blocked`) behaviour,
cancellation during generation, chunk ordering, and device fallback.

**Benchmarks are first-class** (JSON reports under `bench/report/`, with UA,
hardware, model revision, and quantization always recorded):

| Metric                                                         | Why                                      |
| -------------------------------------------------------------- | ---------------------------------------- |
| Download size (model + voice, per dtype)                       | Transparency promise                     |
| Cold-start latency (`createTts` + `preload`)                   | Perceived onboarding cost                |
| Time-to-first-audio, p50/p95 (cold and warm)                   | The metric that matters for UI           |
| Real-time factor (audio seconds per wall second)               | Whether WASM is viable on target devices |
| Peak RAM                                                       | Mobile browser constraints               |
| Cancellation latency                                           | Responsiveness under user interruption   |
| Chunk-consistency (stream vs. full utterance)                  | Detects seam artifacts                   |
| Listening-test outcomes (documented corpus + MOS-style rating) | Quality regressions that numbers miss    |

Thresholds belong in the release definition of done: a profile is only declared
supported when it meets its documented download, latency, and quality envelope on
the stated hardware.

---

## 13. Risks and open questions

| Risk                                                                       | Consequence                                            | Mitigation                                                                                                                                                                                                                                            |
| -------------------------------------------------------------------------- | ------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Published Transformers.js may not support the required model path reliably | MVP blocked for that model profile                     | Pin an exact Transformers.js version, verify in a spike **before** building the API around it; prefer documented public APIs over private tokenizer/tensor internals; treat a broken path as a release blocker for the feature, not a hidden fallback |
| Licensing of non-commercial model families (gated, CC BY-NC 4.0)           | Commercial use cannot be assumed                       | Record `license` in the manifest, never ship gated weights, document restrictions per model profile                                                                                                                                                   |
| "Streaming" overstated                                                     | Users expect frame-level audio and get sentence chunks | Publish a per-profile streaming classification and measure time-to-first-audio                                                                                                                                                                        |
| WebGPU operator gaps for speech ops                                        | Silent failures or wrong audio                         | Resolve device only after operator/capability checks; `auto` may fall back and emits `device-fallback`; explicit `webgpu` never falls back silently                                                                                                   |
| Quantization quality loss                                                  | Robotic or unstable output                             | Choose dtype by measured quality + performance; document the matrix in code                                                                                                                                                                           |
| Node.js runtime differences (`Worker`, `Cache Storage`, GPU)               | Browser-only assumptions leak                          | Adapter boundaries inside the engine; Node-specific tests in CI                                                                                                                                                                                       |
| Scope creep toward model catalog                                           | API instability                                        | One verified profile for the MVP; extend behind the same contract                                                                                                                                                                                     |

Open questions to close before beta:

1. Does the first release ship `prepare()` in core, or defer provisioning to a
   documented application-side download step?
2. Is the core's "one engine per instance" rule sufficient, or is a
   multi-engine fan-out (e.g. WASM + WebGPU simultaneously) needed for voice
   switching?
3. What is the minimum supported streaming granularity we are willing to call
   "streaming" in the public docs?

---

## Related documents

- [lite-speech-concept.md](./lite-speech-concept.md) — product vision, model
  candidates, licensing, and MVP scope.
- [ROADMAP.md](./ROADMAP.md) — milestone plan, sequencing, tooling adoption,
  and release gates.
- [AGENTS.md](../AGENTS.md) — repository layout and the mandatory validation
  sequence.
- `lite-translator` reference sources: `packages/core/src/engine.ts`,
  `packages/core/src/translator.ts`, `packages/core/src/registry.ts`,
  `packages/core/src/pool.ts`, `packages/core/src/emitter.ts`,
  `packages/core/src/errors.ts`, `packages/engine-onnx/src/transformers-engine.ts`,
  `packages/engine-onnx/src/worker.ts`, `packages/engine-onnx/src/webgpu.ts`,
  and its `docs/` deep dives (`api.md`, `live-translation.md`,
  `webgpu-acceleration.md`, `debug-output.md`).
