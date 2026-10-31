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

- `createOnnxEngine({ device: "auto" })` resolving WebGPU or WASM, and reporting which.
- The voice list read synchronously from the engine, before anything is downloaded.
- `estimateDownload()` and `prepare({ allowNetwork: true })` as the consent flow.
- `synthesize()` and `stream()`, with the stream's time-to-first-chunk visible.
- `AbortController` cancellation through the Stop button.

The server answers `404` for unknown paths on purpose: an SPA fallback would return
`index.html` with status 200 for a missing worker script, and the worker would then
fail with a syntax error instead of a clear 404.

No example defines `build`/`test`/`size`, so the root `--if-present` scripts skip it.
