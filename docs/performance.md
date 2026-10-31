# Performance and quality of `lite-speech`

> **Status:** measured for milestone M5.
> **Date:** November 1, 2026
> **Measured on:** AMD Ryzen AI MAX PRO 390 (24 cores), Radeon 8050S (RDNA-3, `shader-f16`),
> Windows 11 x64, Node.js 24.18.0, Chromium 156.
> **Profile:** `onnx-community/phonon-31m-en-ONNX` at revision
> `4f0cda7c60dddb4862d0e68ff9ff60c32ffdc9f4`, voice `default`.
> **Evidence:** the JSON reports under [`bench/report/`](../bench/report) and the scripts in
> [`bench/benchmark`](../bench/benchmark) and [`bench/quality`](../bench/quality).

Everything below is a measurement. Where a number could not be produced on this host, the gap is
stated rather than estimated — an absent measurement is more useful than a plausible guess.

---

## 1. How to read this

Two rules decide whether a number here applies to your machine:

- **Every row names runtime, device, dtype, and phase.** A latency without them is not a measurement
  of anything in particular. `phase: cold` means the asset cache was deleted before the run;
  `phase: warm` means it was left in place, so the model still had to be instantiated but not
  downloaded.
- **Time-to-first-audio (TTFA) and real-time factor (RTF) are defined once:**
  TTFA is the engine's own `time-to-first-chunk` duration, measured from the start of the stream
  call, so it excludes text preparation and harness overhead. RTF is stream wall time divided by
  audio duration; **lower is faster**, and anything below 1.0 is faster than real time.

The benchmark text is fixed (`bench-standard-v1`, 126 characters) and produces 7–8 seconds of audio,
so rows are comparable with each other and with a future run.

## 2. Reproducing

```sh
npm run build

# Node.js, onnxruntime-node, device "cpu"
npm run bench:node --workspace @lite-speech/bench-benchmark -- --dtype q4 --runs 4 --phase cold
npm run bench:node --workspace @lite-speech/bench-benchmark -- --dtype q4 --runs 4 --phase warm

# Chromium, WASM and WebGPU. WebGPU needs the headed driver on Windows — see §6.
npm run bench:browser --workspace @lite-speech/bench-benchmark -- \
  --device both --dtype q4 --runs 4 --phase cold --headed

# Quality sweep (fp32 reference, then the quantized variants)
npm run quality:node --workspace @lite-speech/bench-quality

# Re-render the tables from the committed JSON
npm run report --workspace @lite-speech/bench-benchmark
```

Each command writes one JSON report per configuration. `bench/report/summary.md` and
`bench/report/quality.md` are rendered from those files, so they cannot disagree with them.

## 3. Latency

Milliseconds. `load` is the first `preload()` in the process: it measures instantiating the graphs
from a warm cache, and later streams reuse it. `TTFA cold` is the first stream in the process and
includes whatever the runtime defers until first use.

| runtime  | device | dtype | phase | download MB | load first | TTFA cold | TTFA p50 | TTFA p95 | RTF p50 | RTF p95 | peak RSS MB | cancel p50 |
| -------- | ------ | ----- | ----- | ----------- | ---------- | --------- | -------- | -------- | ------- | ------- | ----------- | ---------- |
| node     | cpu    | fp32  | cold  | 161.9       | 8196       | 58        | 44       | 58       | 0.258   | 0.275   | 403.6       | 0.5        |
| node     | cpu    | fp32  | warm  | 0.3         | 1127       | 106       | 46       | 106      | 0.302   | 0.361   | 515.0       | 2.1        |
| node     | cpu    | q4    | cold  | 68.0        | 5405       | 110       | 38       | 110      | 0.216   | 0.244   | 304.3       | 0.5        |
| node     | cpu    | q4    | warm  | 0.3         | 1181       | 91        | 46       | 91       | 0.241   | 0.310   | 356.9       | 0.4        |
| node     | cpu    | q4f16 | cold  | 41.3        | 5073       | 68        | 46       | 68       | 0.296   | 0.304   | 307.7       | 0.6        |
| node     | cpu    | q4f16 | warm  | 0.3         | 1281       | 27        | 47       | 50       | 0.317   | 0.335   | 312.4       | 0.5        |
| chromium | wasm   | q4    | cold  | 68.0        | 6719       | 100       | 74       | 100      | 0.417   | 0.481   | —           | 0.3        |
| chromium | wasm   | q4    | warm  | 0.0         | 1075       | 88        | 75       | 88       | 0.413   | 0.441   | —           | 0.2        |
| chromium | webgpu | q4    | cold  | 68.0        | 6863       | 680       | 29       | 680      | 0.316   | 0.413   | —           | 0.2        |
| chromium | webgpu | q4    | warm  | 0.0         | 1226       | 87        | 27       | 87       | 0.336   | 0.343   | —           | 0.2        |
| chromium | webgpu | q4f16 | cold  | 41.3        | 5856       | 618       | 33       | 618      | 0.317   | 0.404   | —           | 0.2        |
| chromium | webgpu | q4f16 | warm  | 0.0         | 1117       | 97        | 31       | 97       | 0.322   | 0.349   | —           | 0.2        |

What the table says, and what it does not:

- **`load` does not depend on the download.** Warm `load` is 1.0–1.3 s for every dtype and both
  runtimes: it is graph parsing and kernel initialization, not the network. The cold column's larger
  numbers are dominated by the fetch, which a provisioning step moves off the critical path.
- **WASM is the slow runtime, not the slow profile.** WASM RTF is 0.41–0.48 against Node's 0.22–0.32
  on the same CPU. A browser tab without WebGPU is roughly half as fast as the same machine under
  `onnxruntime-node`.
- **WebGPU is faster in steady state and slower to start.** Warm RTF is 0.32–0.34 versus WASM's
  0.41–0.44, but the first stream pays a one-off shader/JIT cost: cold TTFA is 618–680 ms against
  WASM's 100 ms. A user who synthesizes once per page load sees the _slower_ path with WebGPU.
- **`q4` is not slower than `fp32` on Node in this sweep** (0.216 versus 0.258 RTF, cold). The M1
  spike measured the opposite on a different text and a single sample; that is why both numbers are
  reported with a percentile instead of as a fact. `q4f16` is the exception: at 0.296 cold and 0.317
  warm it is the slowest variant on CPU here, which is consistent with the M1 finding that
  quantization does not buy steady-state speed for a model this small.
- **Cancellation lands in under a millisecond** everywhere, except one 2.1 ms outlier on a warm
  `fp32` run. That is a property of the pull-based protocol: the abort is observed between frames,
  not after a batch.
- **Per-run spread is real.** RTF p95 is up to 31 % above p50 (WebGPU `q4`, cold: 0.316 → 0.413) and
  20 % on a warm `fp32` Node run (0.302 → 0.361). Any single run of this benchmark can be reported
  as a number well below or above the typical case, which is exactly why every table quotes both
  percentiles.

## 4. Memory

Peak resident set, sampled every 25 ms during the stream, on the whole Node.js process:

| dtype   | peak RSS (MB) |
| ------- | ------------- |
| `fp32`  | 404–515       |
| `q4`    | 304–357       |
| `q4f16` | 308–312       |

This is **not** a browser tab budget and not a per-model figure: it includes the ONNX Runtime
native library, the WebAssembly heap, and the harness. It is reported because the _relative_
difference is informative — `fp32` costs ~100 MB more than either quantized variant, which matches
the 94 MB difference in graph size.

Chromium memory is **not reported**. A renderer's heap has no comparable per-model figure and
`performance.memory` is not standardized, so there is no honest number to put in the table.

## 5. Quality

Signal measurements, `temperature: 0`, nine-case corpus. See
[`bench/report/quality.md`](../bench/report/quality.md) for the per-case tables and
[`bench/report/listening.md`](../bench/report/listening.md) for the listening protocol.

**Chunk consistency.** For all nine cases and all three dtypes, the concatenated stream is exactly as
long as the same configuration's full-utterance output (`stream/full` = 1.000, every frame exactly
1 920 samples). Streaming loses nothing and invents nothing: the frames are the same audio in
pieces.

**Determinism.** Re-running `temperature: 0` produced **byte-identical** audio for the same dtype.
Reproducibility is therefore not a source of noise in any table here.

**Quantization distance is not a quality score.** Comparing `q4` and `q4f16` against `fp32` sample by
sample gives an SNR of **−1.4 to −2.9 dB** with a correlation of **≈ 0** (`−0.11` to `0.16`). That is
the signature of _a different performance of the same sentence_, not of added noise: the quantized
graph takes a different trajectory through the flow model, so different prosody and a different stop
point. Subtracting two different utterances yields a negative ratio by construction.

The consequence is a decision, not a caveat:

- A waveform-distance metric **cannot** rank the dtypes. Any future "quality regression" test must
  compare listening outcomes, not samples.
- The listening score is therefore `"pending"`. The harness writes one WAV per case per dtype plus a
  blind protocol (three raters, 1–5 naturalness, median reported); it does not invent the number.
  Until that score exists, the dtype recommendation below is about measured latency, size, and
  memory — not about how the voice sounds.

**Known limitations, confirmed by the corpus.** Four of the nine cases are marked
`known-limitation` and behave as documented in M1: digits and a clock time are not expanded,
`€` degrades, and unpunctuated or heavily punctuated text loses prosody. These are model and
tokenizer properties, not engine defects, and they are identical for all three dtypes.

## 6. Supported envelope

Measured, not assumed. "Supported" here means "runs, and the numbers above apply"; it is not a
compatibility promise for every device in the class.

| Property       | Value                                                                 |
| -------------- | --------------------------------------------------------------------- |
| Download       | 41.3 MB (`q4f16`), 68.0 MB (`q4`), 161.9 MB (`fp32`), voice included  |
| Peak memory    | ~300 MB (`q4`/`q4f16`), ~400 MB (`fp32`) as a Node.js process         |
| Latency band   | TTFA 27–106 ms warm; 58–680 ms cold, depending on device              |
| Throughput     | RTF 0.22–0.44 warm; faster than real time everywhere measured         |
| Cancellation   | under 1 ms typical                                                    |
| Runtime floor  | Node.js 22; Chromium 156 (WASM), Chromium with a WebGPU adapter (GPU) |
| Minimum device | any x86-64 host with ~400 MB free and no GPU, at RTF ≤ 0.5            |

Three facts a consumer has to plan around:

1. **Cross-origin isolation.** A page that wants the WASM thread pool must be served with
   `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp`. The
   benchmark server sets both; without them the runtime runs single-threaded and the WASM numbers
   above become worse.
2. **WebGPU needs a real adapter, and headless Chromium has none on this host.** `navigator.gpu`
   exists, but `requestAdapter()` returns `null` for every flag combination tried (Vulkan, ANGLE,
   SwiftShader, `--ignore-gpu-blocklist`); the same Chromium launched headed reports `amd rdna-3`
   with `shader-f16`. WebGPU measurements here are therefore **headed**, and a CI job without a GPU
   cannot reproduce them.
3. **Node.js has no Cache Storage**, so the voice embedding is fetched over the network on every
   process start. The `download` column for a warm Node run (0.3 MB) is the voice, not the model.
   Browser warm runs correctly report 0.0 MB.

## 7. Decisions closed by these measurements

- **D5 — default dtype per device.** Unchanged: `fp32` on WASM/CPU, `q4` on WebGPU. The sweep does
  not support a change: quantized graphs are not measurably faster on CPU here (`q4` 0.216–0.241,
  `q4f16` 0.296–0.317, `fp32` 0.258–0.302), while they clearly save 94–121 MB of download and
  memory. Quality could not be used as an argument either way, because waveform distance turned out
  to be meaningless and the listening score is still pending. **Revisit D5 when the listening score
  exists**: `q4` on WebGPU is currently chosen for size and for not needing `shader-f16`, not because
  it was judged to sound as good.
- **D6 — supported matrix for `0.1.0`.** Node.js 22, 24, 26 on `onnxruntime-node`; Chromium/Edge with
  WASM on any host; WebGPU only where `requestAdapter()` answers, documented as such and never
  promised as a capability (M1 §5.1 still applies: an accepted `webgpu` request is not proof that
  operators ran on the GPU).

## 8. Not measured

Stated so that nobody has to guess what is missing:

- **Listening score.** Pending by design; see §5 and the protocol.
- **Firefox and WebKit.** The engine is Chromium-verified; no numbers exist for the others.
- **Mobile devices.** No phone or tablet was measured, so the "minimum device class" is an
  x86-64 laptop with no GPU, not a phone.
- **Long-text throughput.** The benchmark text is 126 characters. Segment-heavy inputs (a whole
  article) would add per-segment overhead this table does not show.
- **Cold install over a slow network.** Download _size_ is exact; download _time_ is not reported
  because it would describe this host's connection rather than the profile.
- **GPU memory.** Neither runtime exposes a comparable figure, so only the process RSS is quoted.
