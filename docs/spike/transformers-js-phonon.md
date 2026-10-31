# M1 spike — Transformers.js + Phonon 31M EN

**Status:** M1 complete. Decision: **GO** for `@lite-speech/engine-onnx` on the
low-level `PreTrainedModel` path, **NO-GO** for the high-level `text-to-speech`
pipeline in `@huggingface/transformers@4.3.2`.
**Date:** October 31, 2026
**Related:** [ROADMAP.md](../ROADMAP.md) · [ARCHITECTURE.md](../ARCHITECTURE.md) ·
[lite-speech-concept.md](../lite-speech-concept.md)

---

## 1. Question

> Can a pinned, published Transformers.js release synthesize Phonon 31M EN audio
> end-to-end through public APIs only, in Node.js and Chromium?

Everything below is evidence for that question. The runnable artifacts live in
[`spike/`](../../spike); nothing in that folder is published.

## 2. Pins

| Artifact                    | Pin                                                                              | How it was resolved               |
| --------------------------- | -------------------------------------------------------------------------------- | --------------------------------- |
| `@huggingface/transformers` | `4.3.2`                                                                          | `npm view` dist-tag `latest`      |
| `onnxruntime-node`          | `1.30.0`                                                                         | transitive, `npm ls --depth=2`    |
| `onnxruntime-web`           | `1.31.0-dev.20260914-8d85527a0`                                                  | transitive, `npm ls --depth=2`    |
| Node.js                     | `v24.18.0` (`^24.15.0` line from `engines`)                                      | `node --version` on the test host |
| Model                       | `onnx-community/phonon-31m-en-ONNX` @ `4f0cda7c60dddb4862d0e68ff9ff60c32ffdc9f4` | Hub API `sha`, never `main`       |
| Upstream checkpoint         | `gradium/phonon-31m-en` @ `52433c46975bfa13144c56204c9f4ee1c5b148aa`             | `config.json` → `phonon.source`   |
| Voices                      | `gradium/phonon-voices` @ `0ee17205f254b5abc43b266c8a0f215565c352f7`             | model card, Apache-2.0            |

`onnxruntime-web` is pinned transitively by Transformers.js. Installing it directly
would create a second, drifting copy, so the spike records the resolved version instead
of adding a redundant pin.

## 3. Reproduce

```sh
cd spike
npm install                     # exact versions from package-lock.json
npm ls @huggingface/transformers onnxruntime-web onnxruntime-node --depth=2
node --version

# Node.js (each dtype: one cold run plus four warm runs, the table below uses medians)
node node.mjs --dtype fp32   --device cpu --cache cold --repeat 5
node node.mjs --dtype q4     --device cpu --repeat 5
node node.mjs --dtype q4f16  --device cpu --repeat 5
node node.mjs --dtype q4     --device cpu --temperature 0 --label t0
node compare.mjs              # Markdown table for §5
node compare-audio.mjs        # fp32 vs. quantized SNR at temperature 0

# Chromium (bundler-free)
node serve.mjs                # http://localhost:8080/spike/browser/
```

Each run writes `spike/out/<dtype>-<device>.json` plus a WAV file for the listening
check. `spike/out/` and `spike/.cache/` are git-ignored.

`--repeat N` runs the same text N times so cold and warm behaviour stay separate;
`--label` keeps extra runs from overwriting each other. The browser page performs a cold
and a warm run per configuration.

The browser page loads the bundled ESM build
(`dist/transformers.min.js`) through an import map. The unbundled `transformers.web.js`
is not usable without a bundler in `4.3.2`: it imports `onnxruntime-web/webgpu`, a bare
specifier a browser cannot resolve.

## 4. High-level pipeline: not available

`probe-pipeline.mjs` records what `pipeline("text-to-speech", …)` actually does:

```text
[resolve_model_type] Architecture(s) not found in MODEL_TYPE_MAPPING: [(none)] for model
type 'gradium_phonon_tts'. Falling back to EncoderOnly (single model.onnx file).

Error: Unsupported model type "gradium_phonon_tts" for task "text-to-audio".
None of the candidate model classes support this type.
```

Transformers.js `4.3.2` has no `GradiumPhononTTSForConditionalGeneration`. The two PRs
that add it ([#1811](https://github.com/huggingface/transformers.js/pull/1811),
[#1825](https://github.com/huggingface/transformers.js/pull/1825)) are still open. Using
`4.4.0-next.0` or a branch build would violate the "pinned, published release" rule, so
the pipeline path is out of scope.

**Consequence:** the engine must drive the two ONNX graphs directly. That is exactly what
`spike/lib/phonon.mjs` does, and it uses public members only.

## 5. Measured results

Node.js 24.18.0, Windows 11 x64, AMD RDNA-3 (WebGPU runs in Electron 43 / Chrome 150).
Model revision `4f0cda7c`. Text: two sentences, ≈ 4.6 s of audio. Node numbers are the
median of five runs per variant; "warm" excludes the first run, "first frame" is the median
over all five, so it includes one cold run.

| dtype   | device | download MB | load ms | first frame ms | warm RTF | audio s | frames | RSS MB | range          |
| ------- | ------ | ----------- | ------- | -------------- | -------- | ------- | ------ | ------ | -------------- |
| `fp32`  | cpu    | 161.7       | 1209    | 31             | 0.231    | 4.64    | 58     | 351.7  | -0.390 … 0.376 |
| `q4`    | cpu    | 67.7        | 1075    | 31             | 0.204    | 4.56    | 57     | 236.0  | -0.364 … 0.282 |
| `q4f16` | cpu    | 41.1        | 1327    | 33             | 0.258    | 4.56    | 57     | 270.6  | -0.323 … 0.280 |

Chromium (same host, WebGPU on an RDNA-3 adapter with `shader-f16`):

| dtype   | device | download MB | load ms | first frame ms | warm first frame ms | RTF   | warm RTF |
| ------- | ------ | ----------- | ------- | -------------- | ------------------- | ----- | -------- |
| `q4`    | wasm   | 67.7        | 1147    | 78             | 76                  | 0.507 | 0.450    |
| `q4f16` | webgpu | 41.1        | 2836    | 1479           | 31                  | 0.604 | 0.270    |

Reading of the numbers:

- **Download size** matches the Hub tree exactly: 161 665 677 B / 67 733 146 B /
  41 053 667 B. `estimateDownload()` in M4 can rely on plain file sizes.
- **Load time** is dominated by parsing and initializing the graphs, not by the network
  (warm cache: ~1.1–1.3 s for all three variants). Quantization does not make loading
  faster on this machine.
- **RTF** is 0.20–0.26 warm on CPU for every variant: WASM/CPU is the safe default, and
  quantization buys download size and memory, not steady-state speed. `q4` was the fastest
  variant in this sweep, matching the model card's note that 4-bit matmuls are not
  automatically faster for a model this small.
- **WebGPU** pays a one-off shader/JIT cost (≈ 1.5 s to first frame) and is then the
  fastest path (0.27 warm RTF, 21 ms per 80 ms frame). It needs `shader-f16`; without it
  `q4f16` fails by design and `fp32`/`q4` are the alternatives. A second WebGPU run measured
  0.31 warm RTF with 24 ms per frame, so the browser advantage over WASM is real on this
  host but smaller than the model card's numbers on an M3 Max.
- **`webgpu` is not a reliable capability signal.** `env.backends.onnx.getSupportedDevices()`
  reports `["dml", "webgpu", "cpu"]` in Node.js on this host, and an explicit
  `device: "webgpu"` load _succeeds_ there instead of failing (§5.1). The engine must not
  infer "a GPU exists" from an accepted request.
- **RSS** stays around 236–352 MB, i.e. roughly 1.5–2× the graph size plus the runtime
  heap, measured for the whole Node.js process rather than as a browser memory budget.
- Per-run spread on this laptop was up to ±20 % RTF, which is why the table reports medians
  and M5 must publish p50/p95 instead of a single sample.

### 5.1 WebGPU availability is not a trustworthy signal

`spike/probe-devices.mjs` records what the installed runtime reports in Node.js:

```json
{
  "onnxVersions": { "common": "1.30.0", "node": "1.30.0" },
  "getSupportedDevices": ["dml", "webgpu", "cpu"],
  "hasNavigatorGpu": false,
  "explicitWebgpu": { "loadSucceeded": true, "frames": 26, "firstFrameMs": 557, "rtf": 0.419 }
}
```

There is no `navigator.gpu` in Node.js, yet `webgpu` is listed and an explicit
`device: "webgpu"` load succeeds. So:

- An accepted `webgpu` request does **not** prove that a GPU ran, and it does not prove that
  the operators were available.
- The browser is the only place where `shader-f16` and adapter identity can actually be
  probed, via `navigator.gpu.requestAdapter()` (the browser page does this and reported
  `vendor: "amd"`, `architecture: "rdna-3"`, `shaderF16: true`).
- Consequently M3's `webgpu.ts` must gate on `navigator.gpu` **and** adapter features, and
  must treat a successful explicit-`webgpu` load as no evidence of GPU execution. The
  capability report has to state the device it _requested_ and the device it could verify.

## 6. Output properties

Verified on every run, not assumed:

| Property       | Result                                                          |
| -------------- | --------------------------------------------------------------- |
| Sample rate    | 24 000 Hz                                                       |
| Channels       | 1 (mono)                                                        |
| Sample type    | `Float32Array` per frame, 1 920 samples = 80 ms                 |
| Value range    | never exceeded ±0.44 in any run; no clipping at ±1.0            |
| Frame count    | 57–59 frames for the test sentence, one 1 920-sample frame each |
| Audible speech | yes, listening check on `spike/out/*-cpu.wav`                   |

The value range is reported per run instead of asserted: the model emits an unclipped
float signal, and `encodeWav()` clamps only for the 16-bit WAV used in the listening
check.

## 7. Streaming classification

| Path                                    | Classification               | Evidence                                                                          |
| --------------------------------------- | ---------------------------- | --------------------------------------------------------------------------------- |
| `pipeline("text-to-speech", …)`         | unavailable in 4.3.2         | §4                                                                                |
| Pipeline / `generate_speech` (4.4+)     | buffered **sentence** chunks | model card: flow model runs to the end of a chunk, decoder decodes it in one call |
| Low-level loop (`spike/lib/phonon.mjs`) | **true incremental frames**  | one `flow_lm` step + one `mimi_decoder` call per 80 ms frame, yielded immediately |

The spike implements the low-level loop, so the measured "first frame" is genuine: 31–33 ms
into synthesis on CPU (median), 31 ms warm on WebGPU. Frame interval is 14–20 ms on CPU and
21 ms on WebGPU, all well below the 80 ms of audio each frame carries.

**Design consequence for M2/M3:** the `AudioChunk` contract can carry 80 ms frames, but the
engine must not promise that every profile produces them. The profile seam gets a
`streaming: "incremental-frames" | "sentence-chunks" | "utterance"` field, and the Phonon
profile declares `incremental-frames`.

## 8. Tokenizer: byte fallback without private fields

The model card's low-level snippet reaches into `tokenizer._tokenizer.model` to add Unigram
byte fallback, which the exit gate forbids. Findings (`spike/probe-encoder.mjs`):

- The pinned release does **not** apply byte fallback. A character outside the 4 000-token
  vocabulary becomes `id 0` (`<unk>`), so e.g. `€` and `:` silently degrade.
- The vocabulary contains **all 256** `<0x00>` … `<0xFF>` byte tokens.
- `tokenize()` returns the raw pieces, and `get_vocab()` / `convert_tokens_to_ids()` are
  public. Expanding an out-of-vocabulary piece into its UTF-8 byte tokens with those three
  members reproduces the private patch **exactly on 9/9 test inputs**.
- At `--temperature 0` the WAV output of the public encoder is **byte-identical** to the
  output of the private patch.

Character coverage (characters that collapse to `<unk>` when surrounded by letters):

| Group                     | Covered | Missing    |
| ------------------------- | ------- | ---------- |
| ASCII letters             | 52/52   | —          |
| Digits                    | 10/10   | —          |
| ASCII punctuation         | 19/32   | `: < > / \ | _ = ^ @ # ~ +` |
| Typographic quotes/dashes | 7/7     | —          |
| Latin-1 accents           | 3/5     | `ï`, `ß`   |

**Consequence:** ordinary English text (letters, digits, sentence punctuation) needs no
workaround at all. The public encoder is a small, documented normalization step, not a
private-API dependency, so the exit gate is met. Text containing symbols, other scripts, or
emoji is synthesized as byte-tokenized input: it does not crash, but its pronunciation is a
documented limitation of this profile, not a claimed capability.

## 9. Known limitations

1. **No text normalization.** The Rust/Python runtimes spell out currency, e-mail addresses,
   and other symbols before tokenizing; this repo does not. `€5.50` is read as bytes, `@` is
   dropped. M7 owns richer normalization.
2. **No trailing-silence post-processing.** Only the Python processor applies it.
3. **No voice cloning.** Voices are precomputed 512-dim embeddings (125 frames, 256 000 B);
   the model has no audio encoder.
4. **English only.** The profile speaks English regardless of the voice's design language.
5. **Node.js uses `onnxruntime-node`**, the browser uses `onnxruntime-web`; operator support
   can differ between the two, and only this host was measured.
6. **WebGPU needs `shader-f16` for `q4f16`.** Explicit `webgpu` must throw rather than fall
   back, which is what the engine will enforce in M3. Note that an accepted `webgpu` request
   is not evidence of GPU execution (§5.1), so the check has to happen before the request.
7. **Determinism requires `temperature: 0`.** Default sampling is 0.3, so two runs of the
   same text differ; audio comparisons must use temperature 0.
8. **Quantized variants are not numerically close to fp32** at temperature 0 (measured
   SNR ≈ -1.6 dB against fp32, max difference ≈ 0.6 on the 16-bit scale) while keeping
   identical duration and level. Per the model card this is expected — the quantized graphs
   sample different but intelligible speech — so quality assessment must be a listening
   test, not an SNR gate. Within one dtype, temperature 0 is reproducible byte for byte.

## 10. Decision

**GO**, with these constraints entering M2/M3:

| Constraint                                                                         | Where it lands                                            |
| ---------------------------------------------------------------------------------- | --------------------------------------------------------- |
| Drive `flow_lm` + `mimi_decoder` directly                                          | `engine-onnx/worker.ts`, not `pipeline()`                 |
| Public-only tokenization with byte fallback                                        | `engine-onnx` profile code, unit-tested against fixed ids |
| Profile declares its streaming class                                               | `ModelProfile.streaming` in `engine-onnx/profiles.ts`     |
| Default device/dtype: `wasm`/`fp32`, `q4` for smaller downloads                    | `webgpu.ts` `resolveDeviceDtype()`                        |
| `q4f16` only with `shader-f16`, never silently                                     | `webgpu.ts` explicit-failure rule                         |
| Gate GPU support on `navigator.gpu` + adapter features, not on an accepted request | `webgpu.ts` capability probe                              |
| Cache feedback tensors on the GPU                                                  | `worker.ts` `preferredOutputLocation` setup               |
| Documented: no symbol normalization, English only                                  | `docs/model-profiles.md` (M6)                             |

M2 stays engine-agnostic: nothing from this spike leaks into `@lite-speech/core`. The core
contract only needs "chunks arrive in order, each with `sampleRate`", which both the
incremental and the sentence-chunk path can satisfy.
