# Benchmarks

Private, unpublished workspace packages. They measure the shipped profile and write JSON reports
under `bench/report/`, which **are committed**: every number in
[docs/performance.md](../docs/performance.md) has to trace back to a file there. Only the WAV
stimulus set (`bench/report/audio/`) and the warmed caches (`bench/.cache/`) are artifacts.

- `benchmark/` — cold start, time-to-first-audio (p50/p95), real-time factor, peak RAM,
  cancellation latency, and download size per dtype, on Node.js (`onnxruntime-node`), Chromium
  WASM, and Chromium WebGPU.
- `quality/` — fixed corpus, chunk consistency (streamed vs. full utterance), determinism, and the
  distance between quantized and `fp32` audio; plus the blind listening protocol whose score stays
  `"pending"` until a human fills it in.

No threshold gating happens here. A regression shows up as a diff in a committed report, not as a
failing build — the numbers describe a machine as much as the library, and a CI runner is not the
machine the envelope is documented for.

## Running

```sh
npm run build

# Node.js: one JSON report per dtype and phase.
npm run bench:node --workspace @lite-speech/bench-benchmark -- --dtype q4 --runs 4 --phase cold
npm run bench:node --workspace @lite-speech/bench-benchmark -- --dtype q4 --runs 4 --phase warm

# Chromium. WebGPU needs a real adapter, which headless Chromium does not expose on Windows —
# `--headed` is required for it, and the driver records "unavailable" rather than guessing.
npm run bench:browser --workspace @lite-speech/bench-benchmark -- \
  --device wasm --dtype q4 --runs 4 --phase cold --headed
npm run bench:browser --workspace @lite-speech/bench-benchmark -- \
  --device webgpu --dtype q4 --runs 4 --phase cold --headed

# Quality sweep: fp32 reference plus the quantized variants, in one process.
npm run quality:node --workspace @lite-speech/bench-quality

# Re-render summary.md and quality.md from the committed JSON.
npm run report --workspace @lite-speech/bench-benchmark
```

`--phase cold` deletes the cache for that dtype first; `--phase warm` reuses it, so the `download`
figure is what still had to be fetched. Both phases are separate processes on purpose: a warm number
produced by the same process that populated the cache would not describe a returning user.

## Reading a report

`schemaVersion`, `runtime`, `device`, `dtype`, `phase`, and the model revision are in every file,
because a latency without them is not a measurement of anything in particular. `runs` keeps the raw
per-run samples behind every summary, so a reader can recompute a percentile instead of trusting
one.

Two definitions are fixed by the harness:

- **TTFA** is the engine's own `time-to-first-chunk` duration, measured from the start of the stream
  call. It therefore excludes text preparation and harness overhead.
- **RTF** is stream wall time divided by audio duration. Lower is faster; below 1.0 is faster than
  real time.
