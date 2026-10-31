# M1 inference spike

Throwaway experiment for milestone M1. **Not published, not a workspace package, and not
part of any package build.** The findings live in
[../docs/spike/transformers-js-phonon.md](../docs/spike/transformers-js-phonon.md).

```sh
npm install
npm ls @huggingface/transformers onnxruntime-web onnxruntime-node --depth=2
node --version
```

## Node.js

```sh
node node.mjs --dtype fp32  --device cpu --cache cold --repeat 5
node node.mjs --dtype q4    --device cpu --repeat 5
node node.mjs --dtype q4f16 --device cpu --repeat 5
node node.mjs --dtype q4    --device cpu --temperature 0 --label t0
node compare.mjs        # Markdown table of all runs in out/
node compare-audio.mjs  # fp32 vs quantized, requires --temperature 0 runs
```

`--dtype` accepts `fp32 | q4 | q4f16`, `--device` accepts `cpu | wasm | webgpu | auto`
(Node.js uses `onnxruntime-node`, so `cpu` is the meaningful value there), `--cache cold`
deletes `.cache/` first, `--repeat N` adds warm runs, `--label X` suffixes the output file
name, `--byte-fallback` swaps the public encoder for the model card's private-field
workaround (verification only).

Output: `out/<dtype>-<device>[-label].json` and a matching `.wav` for the listening check.
Both directories are git-ignored.

## Browser (Chromium, no bundler)

```sh
node serve.mjs          # http://localhost:8080/spike/browser/
```

The page mirrors `node.mjs`, reports WebGPU adapter capabilities, and prints the same JSON
fields. It loads `dist/transformers.min.js` through an import map; the unbundled
`transformers.web.js` cannot be used without a bundler because it imports the bare
specifier `onnxruntime-web/webgpu`.

## Evidence scripts

| Script               | Answers                                                                     |
| -------------------- | --------------------------------------------------------------------------- |
| `probe-pipeline.mjs` | Does `pipeline("text-to-speech", …)` work in the pinned release?            |
| `probe-encoder.mjs`  | Does the public encoder match the private byte-fallback patch exactly?      |
| `probe-devices.mjs`  | Which devices/EPs does the runtime report, and does explicit `webgpu` fail? |
| `compare.mjs`        | dtype/device table (median of the runs in `out/`).                          |
| `compare-audio.mjs`  | SNR and max difference of quantized audio against fp32 at temp 0.           |

## Files

- `lib/phonon.mjs` — graph loading, voice prompt, per-frame streaming generator, public
  tokenizer encoder.
- `lib/run.mjs` — shared measurement helper (frames, RTF, range, RMS).
- `lib/wav.mjs` — minimal mono 16-bit RIFF writer.
- `node.mjs` / `browser/` — the two runners.
- `serve.mjs` — zero-dependency static server rooted at the repository.
