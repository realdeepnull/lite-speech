# Examples

Private, unpublished workspace packages. They are never published (`private: true`,
no `files` field) and consume the packages the way a user would.

- `demo/` — bundler-free browser demo against the built ESM output.

## Demo

```sh
npm run build
node examples/demo/serve.mjs        # or: npm run serve --workspace examples/demo
# open http://localhost:8080/examples/demo/
```

The page imports `packages/*/dist` through an import map, so it is a real consumer
test: if the published `exports`, the `./worker` subpath, or the bundling policy were
wrong, it would fail. It needs a static server because a module worker cannot be
loaded from `file://`.

What it demonstrates:

- A **catalogue built from manifests** (`createRegistry({ models, voices })`), so the
  language and voice selectors are the catalogue rather than a hard-coded list — the M6
  seam, visible in the smallest possible consumer.
- `createOnnxEngine({ device: "auto", registry })` resolving WebGPU or WASM, and
  reporting the resolved device/dtype (including a documented fallback) before anything
  is downloaded.
- Language-aware selection: the voice list is filtered by the chosen language, and
  `createTts({ language, voice })` is rebuilt when either changes.
- `describeAssets()` rendering the cache state per selection, including the **dtype
  variant** the resolved device priced, and `estimateDownload()` as the consent figure.
  Switching to an uncached voice shows the missing file _before_ Speak is pressed.
- `prepare({ allowNetwork: true, verify: true })` and `verifyAssets()` as the
  provisioning flow, including a visible integrity state.
- `synthesize()` and `stream()`, with the stream's time-to-first-chunk visible, and the
  sample rate read from the chunks rather than assumed.
- `AbortController` cancellation through the Stop button.
- `removeModel()` eviction, after which strict offline correctly refuses.

The server answers `404` for unknown paths on purpose: an SPA fallback would return
`index.html` with status 200 for a missing worker script, and the worker would then
fail with a syntax error instead of a clear 404.

No example defines `build`/`test`/`size`, so the root `--if-present` scripts skip it.
