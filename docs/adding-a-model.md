# Adding a model, a language, or a family

`lite-speech` treats "which model speaks" as configuration. This document is the recipe
for using that: registering another checkpoint, another language, or another
architecture, and knowing which of the three you are actually doing.

The short version:

| You want…                                        | You write…                                     | Engine code |
| ------------------------------------------------ | ---------------------------------------------- | ----------- |
| another checkpoint of a known family             | one `ModelManifest` (+ voices)                 | none        |
| a language no shipped model speaks               | a manifest whose `languages` include it        | none        |
| a second language on one multilingual checkpoint | one manifest with several `languages`          | none        |
| a new architecture                               | a `ModelRunner` factory + one `families` entry | one factory |

Everything below is public API of `@lite-speech/engine-onnx`.

---

## 1. The manifest is the whole contract

A model is a **pin plus a description**, never code:

```ts
import type { ModelManifest } from "@lite-speech/core";

const model: ModelManifest = {
  id: "your-org/phonon-31m-de-ONNX", // repository that holds the files
  revision: "<40-char commit sha>", // a pin, never "main"
  license: "Apache-2.0", // shown in a consent dialog
  family: "phonon", // which runner executes it
  languages: ["de"], // which requests it serves
  sampleRate: 24_000, // decoded audio rate, never assumed globally
  files: [
    { path: "config.json", size: 4_952, sha256: "…" },
    { path: "tokenizer.json", size: 303_009, sha256: "…" },
    { path: "tokenizer_config.json", size: 144, sha256: "…" },
    // One entry per build. `variant` is what the engine selects on.
    { path: "onnx/flow_lm_q4.onnx", variant: "q4", size: 84_132, sha256: "…" },
    { path: "onnx/flow_lm_q4.onnx_data", variant: "q4", size: 26_073_088, sha256: "…" },
    { path: "onnx/flow_lm.onnx", variant: "fp32", size: 120_089_751, sha256: "…" },
  ],
};
```

Four fields carry the extensibility, and each is read from exactly one place:

- **`family`** selects the runner. An id no runner claims is an explicit error listing
  the known ones — never a silent fallback to another architecture, which would run
  different input shapes and produce noise instead of audio.
- **`languages`** selects the model for a request. Omitted, it is derived from the
  languages of the model's own voices, so a minimal manifest still works.
- **`variant`** selects the weights. A file without a variant is required by every
  configuration (configs, tokenizers, embeddings). This is why `estimateDownload()`
  can price a profile the device will not run: every variant is declared, and only the
  selected one is counted.
- **`revision`** is the cache key and the URL. Both are derived from it, so the bytes
  that were measured and the bytes that run cannot diverge.

Sizes and digests are optional. Omitting them is legal and means "cannot prove":
`offline: "strict"` will still refuse to synthesize, and `verifyAssets()` will report
the file as unverified rather than claim success.

---

## 2. Voices are their own manifests

A voice is addressed by its own repository, revision, and path:

```ts
import type { VoiceManifest } from "@lite-speech/core";

const voices: VoiceManifest[] = [
  {
    id: "thorsten",
    modelId: "your-org/phonon-31m-de-ONNX", // repository holding the voice bytes
    revision: "<commit sha>", // may differ from the model's
    language: "de",
    license: "CC0-1.0",
    sampleRate: 24_000,
    name: "Thorsten", // presentation data belongs here, not in engine code
    gender: "male",
    description: "Accent de-DE",
    files: [{ path: "voices/thorsten.bin", size: 256_000, sha256: "…" }],
  },
];
```

Two rules are worth stating because they are easy to get wrong:

1. **The first file is the voice asset.** A family whose speaker needs several files
   needs that contract extended deliberately; appending files and hoping the first one
   is right is not a contract.
2. **`language` is what the voice actually speaks.** A voice designed in Spanish but
   used by an English model still speaks English — its design language is metadata.
   Reporting `"de-AT"` for an English model would make `supports("de-AT")` a promise
   the audio cannot keep.

---

## 3. Register and use it

```ts
import { createTts } from "@lite-speech/core";
import { createOnnxEngine, createRegistry } from "@lite-speech/engine-onnx";

const engine = await createOnnxEngine({
  registry: createRegistry({ models: [model], voices }),
});
const tts = createTts({ engine, language: "de", offline: "strict" });
```

`createTts()` now:

- resolves `"de"` to your model through the registry;
- picks the first voice registered for German, not the first voice overall;
- accepts `"de-AT"` too, by primary-subtag fallback, because the audio is the same.

If you want the built-in English model to keep working alongside yours, build the
registry from both:

```ts
import { BUILT_IN_MODELS, BUILT_IN_VOICES, createRegistry } from "@lite-speech/engine-onnx";

const registry = createRegistry({
  models: [...BUILT_IN_MODELS, model],
  voices: [...BUILT_IN_VOICES, ...voices],
});
```

---

## 4. A new architecture is one factory

A `ModelRunner` is the only interface the worker uses after a load. Implement it and
add one entry to `families`:

```ts
import type { FamilyRegistry, ModelRunner } from "@lite-speech/engine-onnx";

const runner = async ({
  modelId,
  revision,
  dtype,
  device,
  voice,
  onProgress,
}): Promise<ModelRunner> => {
  // Load what you need; report progress through `onProgress`.
  const say = async function* (segment: string, { isCancelled }) {
    for (const frame of await synth(segment)) {
      if (isCancelled()) return; // a cancel must land between frames
      yield frame; // mono PCM in [-1, 1]
    }
  };

  return {
    profile: {
      id: "my-family",
      streaming: "sentence-chunks", // tell the truth; see below
      prepareSegment: (segment) => segment,
    },
    sampleRate: 22_050,
    synthesize: say,
    dispose: async () => {
      /* release sessions and GPU buffers */
    },
  };
};

const families: FamilyRegistry = { "my-family": runner };
const engine = await createOnnxEngine({ models: [model], voices, families });
```

Four obligations come with that freedom:

1. **`streaming` must be truthful.** `"utterance"` means one frame at the end;
   `"sentence-chunks"` means buffered segments; `"incremental-frames"` means audio
   while the model is still generating. `stream()` returns an `AsyncIterable` in all
   three cases, so the classification is the only way a consumer can plan.
2. **`isCancelled()` must be sampled between frames.** The engine cannot interrupt a
   running inference call; a runner that never checks will hold the worker until it
   finishes.
3. **`dispose()` must be idempotent.** The host calls it on reload, on eviction, and on
   worker teardown.
4. **`sampleRate` is per runner.** Nothing resamples; the value is reported, and a
   consumer that cannot play it must handle that itself.

In-thread (Node.js) the same registry is used, so a family is testable without a
browser. In a browser the worker hosts its own built-in registry; pass your families
through `createOnnxEngine({ families })` and run with `worker: false` if you need a
custom family in-thread.

---

## 5. What to test

The tests that keep this honest are cheap, and the repository already runs the shape
of each one:

- **Every file has a size and a digest**, and no digest is shared by two files of
  different sizes. A consent dialog and an integrity check both depend on it.
- **A dtype selects its own variant plus every variant-less file.** `describeAssets()`
  after `createOnnxEngine({ dtype })` is the assertion.
- **A second language resolves, and the first language still does.** Adding a model
  must not shadow an existing one.
- **An unknown family is an error naming the known ones**, not a default.
- **Voice presentation data comes from the manifest**, so the engine has no voice
  table that could disagree with the catalogue.

Run them with `npm test`. Nothing in this recipe requires a browser, a download, or a
GPU: the seams are data, and a fake runner proves the engine's behaviour without ONNX.
