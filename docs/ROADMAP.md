# Roadmap of `lite-speech`

> **Status:** M0 (repository and toolchain foundation), M1 (inference spike),
> M2 (`@lite-speech/core` MVP), M3 (ONNX engine MVP), and M4 (assets,
> offline policy, provisioning) implemented; M5 is next.
> Milestones below keep their checklist state.
> **Date:** October 31, 2026
> **Target:** First public release `0.1.0` of `@lite-speech/core` and
> `@lite-speech/engine-onnx`.
> **Related:** [lite-speech-concept.md](./lite-speech-concept.md) ·
> [ARCHITECTURE.md](./ARCHITECTURE.md) ·
> [npm-library-tooling.md](../npm-library-tooling.md)

---

## 1. How to read this roadmap

- Milestones are ordered by **risk retirement**, not by visible feature value.
  The model path is proven before the API is designed around it.
- Every milestone ends with an **exit gate**. A criterion that cannot be
  measured is not a gate; if a target is not yet measurable, it is marked
  _provisional_ and replaced by a measured value in M5.
- Tooling is introduced **just in time**: a tool is added in the milestone whose
  gate needs it, and not earlier (see
  [§5](#5-tooling-adoption-plan)).
- Checklist items are intended to become GitHub issues grouped by milestone
  label (`m0-foundation` … `m6-beta`).

## 2. Principles shaping the plan

1. **Validate before abstracting.** No core API decision may depend on
   unverified Transformers.js behaviour; that is what M1 exists for.
2. **Two small packages over one large one.** `core` stays engine-independent and
   dependency-free; only `engine-onnx` may pull in an ML runtime.
3. **Nothing unmeasured.** Download size, latency, memory, and quality claims come
   from the benchmark harness, never from intuition.
4. **Explicit over convenient.** No hidden downloads, no silent device or dtype
   fallback, no implied resampling, no claimed streaming that is not streaming.
5. **Truthful per-profile documentation.** Streaming class, license, measured
   size, and supported runtimes are documented for every shipped profile.
6. **Boring, proven library tooling.** ESM-only, strict TypeScript, `tsdown`
   builds, Vitest tests, published packages validated by `publint`,
   Are the Types Wrong, `knip`, and `size-limit`.

## 3. Milestones at a glance

| ID  | Theme                 | Primary outcome                                                     | Exit gate                                                           |
| --- | --------------------- | ------------------------------------------------------------------- | ------------------------------------------------------------------- |
| M0  | Foundation            | Workspace, package skeletons, toolchain, CI skeleton                | `build`, `typecheck`, `lint`, `format:check`, `test` green on stubs |
| M1  | Inference spike       | Pinned Transformers.js produces Phonon 31M EN audio without hacks   | Reproducible audio in Node + Chromium, limitations documented       |
| M2  | Core MVP              | `@lite-speech/core`: contract, lifecycle, streaming, offline policy | Fake-engine suite green, zero runtime deps, size budget met         |
| M3  | ONNX engine MVP       | `@lite-speech/engine-onnx`: worker engine, one verified profile     | Browser integration suite and cancellation tests green              |
| M4  | Assets & provisioning | Declarative manifests, `prepare()`, cache inspection and eviction   | Strict-offline end-to-end run succeeds with network blocked         |
| M5  | Measurement           | Benchmark and quality harness, published performance report         | Latency, RTF, RAM, and size recorded per profile and device         |
| M6  | Beta `0.1.0`          | Docs, examples, package validation, first published release         | `pack:check` green, release published via Trusted Publishing        |
| M7  | Beyond 1.0            | More models, languages, engines; optional provisioning CLI          | Additions require no breaking change to the core API                |

---

## 4. Milestone details

### M0 — Repository and toolchain foundation

**Goal:** a reproducible skeleton so that every later change is validated
automatically, even before the first library code exists.

**Deliverables**

- [x] npm workspaces: `packages/core`, `packages/engine-onnx`, plus private
      `examples/*` and `bench/*` workspace packages (`private: true`, never
      published) that opt into nothing until M3/M5.
- [x] Root `package.json` with the validation scripts from
      [AGENTS.md](../AGENTS.md): `build`, `typecheck`, `lint`, `format:check`,
      `format`, `test`, `knip`, `size`, `pack:check`. Per-package scripts
      (`build`, `typecheck`, `test`, `size`, `pack:check`) fan out with
      `--workspaces --if-present`, so a package opts in when it gets the tool;
      `lint`, `format`, `format:check`, and `knip` are repository-wide by
      definition.
- [x] `tsconfig.base.json` with `strict`, `noUncheckedIndexedAccess`,
      `exactOptionalPropertyTypes`, `verbatimModuleSyntax`, and
      `isolatedModules`.
- [x] ESLint flat config + Prettier config (`.prettierrc`, `.prettierignore`,
      `.editorconfig`, plus `.gitattributes` to keep LF on checkout).
- [x] Package skeletons: `"type": "module"`, `exports` map with `types` +
      `import`, `sideEffects: false`, `files: ["dist", "README.md", "LICENSE"]`,
      `publishConfig.access: "public"`, `engines.node`.
- [x] `LICENSE` per package plus a `NOTICE`/third-party section describing model
      licenses (Apache-2.0 vs. CC BY-NC 4.0).
- [x] CI workflow skeleton: install with `npm ci`, then
      `build → typecheck → lint → format:check → test` on every Node line the
      `engines` field declares (22, 24, 26).
- [x] `README.md` with the positioning statement, install instructions, and a
      pointer to the docs.
- [ ] Confirm the npm scope (`@lite-speech` is a proposal) or record the final
      scope before M6. _Status: the scope is currently unclaimed on the public
      registry (`404` for both package names), which means it is available but
      not yet owned; claim it and close D4 before the first publish._

**Exit gate**

- [x] `npm run build`, `typecheck`, `lint`, `format:check`, and `test` all
      succeed on the skeletons, locally and in CI.
- [x] No package has a runtime dependency yet. The root `knip` script runs and is
      clean; `size` and `pack:check` are already wired as root scripts but stay
      no-ops until a package defines them (M2/M6), which is exactly how the
      just-in-time tooling rule is enforced.

---

### M1 — Inference spike (go/no-go)

**Goal:** answer one question with evidence: _can a pinned, published
Transformers.js release synthesize Phonon 31M EN audio end-to-end through
public APIs only?_

**Tasks**

- [x] Pin exact versions of `@huggingface/transformers`, `onnxruntime-web`, and
      Node LTS; record the resolved model revision (commit SHA, never `main`).
- [x] Script the flow outside the library (`spike/` folder, not published) for
      both Node.js and a browser page.
- [x] Verify output properties: 24 kHz, mono, `Float32Array` in `[-1, 1]`, and
      audible speech in a listening check.
- [x] Compare `q4f16`, `q4`, and `fp32` graphs for download size, load time,
      real-time factor, and perceived quality.
- [x] Classify the streaming behaviour honestly: true incremental frames,
      buffered sentence chunks, or full-utterance only.
- [x] Document WebGPU behaviour for the required operators, or record that WASM
      is the only viable path for the MVP.
- [x] Write `docs/spike/transformers-js-phonon.md` with commands, versions,
      measured numbers, observed limitations, and the recommendation.

**Exit gate**

- [x] A documented command reproduces audio in **Node.js and Chromium** from a
      clean install.
- [x] Private tokenizer fields or unpublished branches are **not** required; if they
      are, the MVP falls back to a validated public low-level path or the feature is
      deferred.
- [x] The streaming classification and the license statement are recorded.
- [x] A written go/no-go decision exists, and M2 API work is aligned with it.

**Outcome:** GO on the low-level `PreTrainedModel` path, NO-GO on
`pipeline("text-to-speech", …)` in `@huggingface/transformers@4.3.2` (the Phonon
model class is still unmerged in PRs #1811/#1825). Byte fallback was replaced with
a public-member encoder that reproduces the model card's private-field workaround
byte for byte. Measurements, limitations, and the constraints handed to M2/M3 are
in [spike/transformers-js-phonon.md](./spike/transformers-js-phonon.md).

> **Hard rule:** no `core` API is designed around Phonon specifics before this
> gate passes.

---

### M2 — `@lite-speech/core` MVP

**Goal:** ship the engine-independent contract and lifecycle that the concept
promises, provable with a fake engine in Node only.

**Deliverables**

- [x] `tts.ts` — `createTts()` (synchronous), lazy `preload()` with
      single-flight and un-memoized failures, `synthesize()`, `stream()`,
      `dispose()` (idempotent), disposed/abort re-checks after every `await`.
- [x] `engine.ts` — `TtsEngine` contract with synchronous `supports()` and
      `voices()`, optional `capabilities()`, `prepare()`, `describeAssets()`,
      `removeModel()`.
- [x] `types.ts` — `AudioResult`, `AudioChunk`, `VoiceInfo`, `ProgressEvent`,
      `DebugEvent`, `TtsCapabilities` as pure data contracts.
- [x] `stream.ts` — bounded `AsyncIterable<AudioChunk>` with backpressure,
      monotonic `sequence`, and exactly one `final` chunk.
- [x] `prepare.ts` — deterministic `prepareText()` / `segmentText()` with
      `maxCharsPerSegment`, no silent truncation.
- [x] `assets.ts` — `ModelManifest`, `VoiceManifest`, `estimateDownload()`,
      `assertBudget()`, cache-state helpers.
- [x] `errors.ts` — stable `TTS_ERROR_CODES` with `cause` chaining; every error
      normalized once at the public boundary.
- [x] `emitter.ts` — typed emitter with no DOM dependency (Node tests).
- [x] `wav.ts` — tree-shakeable `encodeWav()` above the inference contract.
- [x] `pool.ts` — `TtsPool` with single-flight creation and LRU eviction.
- [x] `index.ts` — the only entry point; every export intentional.
- [x] Tests with a fake engine: lifecycle, offline policy, budget rejection,
      segmentation, backpressure, abort semantics, pool races, error
      normalization, WAV output.

**Tooling added:** `vitest` (node environment), `eslint` + `typescript-eslint`,
`prettier`, `knip`, `size-limit` (core only), `tsdown` for the core build.

**Exit gate**

- [x] Core unit suite green on Node LTS with no DOM shims (106 tests, `node`
      environment).
- [x] `packages/core/package.json` has **zero** runtime and peer dependencies.
- [x] `size-limit` passes a documented budget.
      _Measured: 6.35 kB minified+gzipped; budget set to 7 kB (the roadmap's
      provisional 6 kB was revised after the first real build, as planned). The M4
      layer took it to 6.95 kB, so the limit moved to 8 kB in M4 for the same reason._
- [x] The public surface contains no Transformers.js, ONNX, worker, or
      audio-playback types.

---

### M3 — `@lite-speech/engine-onnx` MVP

**Goal:** one verified Phonon profile running through a worker behind the M2
contract.

**Deliverables**

- [x] `create-onnx-engine.ts` — factory with `models`, `registry`, and
      `profiles` overrides; `device`/`dtype` application.
- [x] `transformers-engine.ts` — `TtsEngine` implementation: worker lifecycle,
      `id`-correlated pending map, abort handling, cache checks, error mapping.
- [x] `worker.ts` — single model per worker, serialized message queue,
      `ArrayBuffer` transfer for PCM, progress events, cooperative cancel,
      `dispose` that always acknowledges.
- [x] `profiles.ts` — `ModelProfile` seam resolved from `metadata.profile`, with
      the first Phonon profile and its frame handling.
- [x] `models.ts` — declarative model **and voice** registry with pinned
      revisions, sizes, hashes, and `license`.
- [x] `webgpu.ts` — memoized adapter probe, `resolveDeviceDtype()`, no silent
      fallback for explicit `"webgpu"`, `device-fallback` event for `"auto"`.
- [x] Node.js degradation: in-thread inference when `Worker` is unavailable;
      cache-miss-safe behaviour without `Cache Storage`.
- [x] `./worker` export subpath and `alwaysBundle` configuration for
      `@lite-speech/core` and `@huggingface/transformers`.

**Tooling added:** `playwright`, `@vitest/browser-playwright` (Vitest 4 browser
mode), `size-limit` for the engine entry (excluding the ML runtime and model
weights, with an explicit "model assets not included" note).

**Exit gate**

- [x] One profile synthesizes correctly in Chromium via the **public** core API.
      _Measured: `q4` on WebGPU, 4.56 s of 24 kHz audio in 3.16 s, in a module
      worker, through `createTts()`._
- [x] Streaming assertions pass: gaps, duplicates, ordering, and `final` semantics.
      _56 frames of 1 920 samples, first chunk at 50 ms, exactly one `final` chunk._
- [x] Cancellation is observable: `TtsError(ABORTED)`, no unbounded buffering, and
      worker reuse after abort.
- [x] Explicit `"webgpu"` on a device without an adapter throws; `"auto"` emits
      `device-fallback`.
- [x] Both packages build from a clean `npm ci` and the demo page runs against the
      built ESM output.

**Outcome:** the worker is **pull-based** (`stream-open` → N × `stream-next` →
`stream-close`) rather than push-with-a-high-water-mark. The engine asks for one
frame at a time, so backpressure is structural instead of a credit protocol, the
worker cannot buffer ahead of the consumer by construction, and every pull is a
cancellation checkpoint. Because the worker cannot know a frame is last until its
generator ends, the **engine** emits the single `final: true` chunk when the
session reports `stream-end`; peeking one frame ahead would double
time-to-first-audio.

Two defects surfaced only in the browser and are worth recording:

1. **A bundled `core` made `isTtsError()` fail across the boundary.** The engine
   bundles `@lite-speech/core`, so at runtime the engine's `TtsError` and the
   application's are two distinct classes; `instanceof` rejected the engine's
   `ABORTED`, and core re-wrapped it as `STREAM_FAILED`. The guard is now
   structural (`name` + `code` + `message`), which is the documented contract.
2. **Abandoning a stream deadlocked against inference.** A generator cannot observe
   `return()` while suspended in an `await`, so `break` inside `for await` waited
   for a frame nobody wanted. `stream()` now aborts an internal controller first and
   reports `done` immediately.

---

### M4 — Assets, offline policy, and provisioning

**Goal:** make "offline-first" a state the library can prove, not a claim.

**Deliverables**

- [x] `tts.prepare({ allowNetwork: true, maxDownloadMB })` in core, delegating to
      the engine's `prepare()`/`assetPlan()`.
- [x] Budget rejection **before** any download, listing required files and total
      size.
- [x] `offline: "strict"` enforced in core via `isCached()`; violations reject
      with `OFFLINE_ASSETS_MISSING` before the engine is invoked.
- [x] Integrity checks (`sha256` when declared) with
      `ASSET_INTEGRITY_FAILED`; atomic, all-or-nothing cache writes.
- [x] `describeAssets()` per-file cache state and `removeModel()` eviction across
      the caches Transformers.js may use.
- [x] Provisioning documentation: build-time pre-download recipe (scripted, no
      CLI package required).
- [x] Offline test: run the full flow with network blocked and assert clear
      errors for missing assets.

**Exit gate**

- [x] A cold machine provisions assets once, then synthesizes with the network
      disabled in both Node.js and Chromium.
      _Verified: `tests/revision-pinning.test.ts` drives Transformers.js' real
      resolution code and records the URLs it requests (the tokenizer probe targets
      `resolve/main` by default and the pin by template), and the
      `synthesizes and verifies from Cache Storage alone` case in
      `tests/engine.browser.test.ts` provisions in Chromium, then synthesizes,
      streams and verifies with every manifest path resolving to a cache entry at
      the pinned revision and nothing under `resolve/main`. In Node, both the
      filesystem probe (`tests/engine.test.ts`) and the core gate
      (`tests/offline-network.test.ts`) cover the same claims._
- [x] No code path downloads without explicit consent; a test asserts zero network
      requests during strict-offline synthesis.
      _The core gate is the single choke point (`#assertOfflineUsable()`), and the
      tests assert the engine is never invoked when it rejects. The browser suite
      additionally asserts that no cache entry ever points at `resolve/main`._
- [x] `estimateDownload()` matches the observed byte count within a documented
      tolerance.
      _Measured: exact, not a tolerance. `q4` = 67 989 146 B, `fp32` = 161 921 677 B,
      `q4f16` = 41 309 667 B; per-file observed cache sizes equal the manifest sizes
      in Chromium. A mismatch is a bug, so the test asserts equality._

**Outcome:** the policy gate moved. Strict offline was enforced in `synthesize()`
and `stream()` only, which left `preload()` as a shortcut that could still download
— the one method a caller reaches for precisely when they want to warm up without
synthesizing. All three now pass through a single `#assertOfflineUsable()` gate, so
"nothing downloads without consent" is a property of the class rather than a rule
three methods remember.

Two decisions were made while implementing, and both are visible in the API:

1. **Verification is opt-in, not automatic.** Hashing a 68 MB profile means reading
   every byte back. That is the right default for a cold-machine provisioning step and
   the wrong one for a warm cache on every page load, so `prepare({ verify: true })`
   and `tts.verifyAssets()` are explicit. The manifest still declares every digest, so
   `plan.integrity` tells a UI whether a check is even possible.
2. **A failed provisioning rolls back.** The engine snapshots which files were cached
   before a `prepare()` and deletes only what the attempt added. Without it, a dropped
   connection leaves a half-populated cache that reports a different `missing` list on
   every retry, and one profile's partial download could later be deleted as another
   profile's eviction.

A third change was not optional: cached files now report their measured `cachedSize`,
and a size that disagrees with the manifest makes strict offline reject with
`OFFLINE_ASSETS_MISSING`. A file that exists but is truncated is not an asset the
library can claim to have.

A fourth was found only by turning the network off in a real browser.
**Transformers.js probes the Hub for `resolve/main/tokenizer_config.json` even when
every caller passes a pinned `revision`**: its internal `get_tokenizer_files()` calls
`get_file_metadata()` without forwarding `revision`, so that one lookup always targets
the moving branch, and — because the Cache Storage key is derived from the same URL —
its cache entry is under `resolve/main` too. The observable effect was a strict-offline
synthesis that still issued one request, which then failed. Pinning
`env.remotePathTemplate` to the manifest revision fixes the request _and_ the cache key
in one place, because both are built from the template. The regression test
(`tests/revision-pinning.test.ts`) drives Transformers.js' real resolution code rather
than a fake, since the defect is in the dependency, not in this repository.

---

### M5 — Observability, benchmarks, and quality

**Goal:** replace opinion with measurements, per profile and per device.

**Deliverables**

- [ ] `onDebug` / `onProgress` typed events behind `?.(…)` calls, with zero cost
      when unused: `device-resolved`, `worker-spawn`, `load-done`,
      `stream-start`, `time-to-first-chunk`, `inference-done`,
      `device-fallback`.
- [ ] `bench/benchmark`: cold start, warm time-to-first-audio (p50/p95), real-time
      factor, peak RAM, cancellation latency, download size per dtype.
- [ ] `bench/quality`: fixed corpus, chunk-consistency check (stream vs. full
      utterance), and a documented MOS-style listening score.
- [ ] JSON reports under `bench/report/` recording UA, hardware, model revision,
      quantization, and date.
- [ ] `docs/performance.md` with reproducible tables and an explicit statement of
      the supported envelope.

**Exit gate**

- Every number in the README/docs traces to a committed benchmark report.
- The streaming classification from M1 is confirmed by measurement per device
  (WASM vs. WebGPU).
- A documented envelope exists: minimum device class, minimum RAM, download
  size, and latency band for the shipped profile.

---

### M6 — Documentation, examples, and `0.1.0` beta

**Goal:** a publishable, consumable release validated the way a user installs it.

**Deliverables**

- [ ] `README.md` per package: pitch, install, quick start, offline provisioning,
      limitations, license of models used.
- [ ] `docs/api.md` generated or hand-written reference for the public surface.
- [ ] `docs/model-profiles.md`: per-profile size, license, streaming class,
      supported runtimes, and known limitations.
- [ ] Examples: one Node.js and one browser sample on the **same** core API, plus
      a bundler-free demo against the built ESM output.
- [ ] Changesets configured; `CHANGELOG.md` seeded.
- [ ] Release workflow: `npm publish` via **npm Trusted Publishing (OIDC)**, no
      long-lived tokens, provenance enabled, `publishConfig.provenance` where
      supported.
- [ ] Migration/troubleshooting notes covering worker setup, CSP, and
      `env.allowLocalModels` pitfalls.

**Tooling added:** `publint`, `@arethetypeswrong/cli`, `@changesets/cli`, packed
tarball install smoke test (`npm pack` → install into a scratch project → run
the example).

**Exit gate**

- The full validation sequence from [AGENTS.md](../AGENTS.md) is green:
  `build`, `typecheck`, `lint`, `format:check`, `test`, `knip`, `size`,
  `pack:check`.
- `publint` and Are the Types Wrong (`node16` profile) pass for both packages.
- A scratch project installs the tarballs and runs both examples without extra
  configuration.
- `0.1.0` is published with provenance and a changelog entry.

---

### M7 — Beyond `0.1.0`

Only after the beta has produced real-world feedback.

- [ ] German Phonon support **if** an independently verified, legally usable
      ONNX export exists and quality passes the M5 corpus.
- [ ] Multilingual profile under the same rules; no gated weights shipped.
- [ ] Genuinely incremental synthesis to lower time-to-first-audio, where the
      runtime can support it.
- [ ] Richer text normalization (numbers, dates, currency, abbreviations),
      opt-in and testable in isolation.
- [ ] Integration with streamed LLM text (incremental sentence feeding).
- [ ] Additional engines (`engine-transformers`, other model families) as new
      packages behind the unchanged contract.
- [ ] Optional provisioning CLI and a framework wrapper, neither required by
      core.
- [ ] Evaluate Oxlint/Biome or a faster type checker **only** if CI time becomes
      measurable friction.

## 5. Tooling adoption plan

The tooling in [npm-library-tooling.md](../npm-library-tooling.md) is treated as
orientation, not a wishlist. Each tool is introduced when a gate depends on it.

| Tool                            | Milestone | Why then                                                    |
| ------------------------------- | --------- | ----------------------------------------------------------- |
| `typescript`                    | M0        | Strict baseline, declarations, editor feedback              |
| `tsdown`                        | M0/M2     | ESM build, `.d.ts`, exports map, tree shaking               |
| `eslint` + `typescript-eslint`  | M0        | Correctness rules from the first committed line             |
| `prettier`                      | M0        | Formatting is a CI check, never a review topic              |
| `vitest` (node)                 | M2        | Core is provable with a fake engine, without a browser      |
| `knip`                          | M2        | Prevents orphaned exports and dependencies as the API grows |
| `size-limit`                    | M2        | Guards the "core stays tiny" promise from day one           |
| `vitest` browser + `playwright` | M3        | Workers, Cache Storage, and WebGPU have no Node equivalent  |
| `publint`                       | M6        | Validates the actual npm artifact before publishing         |
| `@arethetypeswrong/cli`         | M6        | Catches broken type resolution across module resolutions    |
| `@changesets/cli`               | M6        | Semantic versioning, changelogs, and coordinated releases   |

**Deliberately deferred**

- `comlink` — only if hand-written worker messaging becomes the dominant source
  of bugs; a small request/response protocol keeps the bundle smaller.
- `idb` / `idb-keyval` — only once IndexedDB is genuinely needed; Cache Storage
  covers model assets for the MVP.
- `typedoc` — optional for M6; hand-written API docs are acceptable for two
  packages and a small surface.
- Oxlint, Biome, Bun, or alternative bundlers — no second tool solving an
  already-solved problem during the MVP.

**Runtime dependency policy**

| Package                    | Runtime dependencies                                  | Peer dependencies   |
| -------------------------- | ----------------------------------------------------- | ------------------- |
| `@lite-speech/core`        | none                                                  | none                |
| `@lite-speech/engine-onnx` | `@huggingface/transformers` (bundled into the worker) | `@lite-speech/core` |

Model weights are **never** published inside the npm package; they are pinned
remote assets resolved through the manifest.

## 6. CI/CD pipeline

**Pull requests**

1. `npm ci` with dependency caching.
2. `build` → `typecheck` → `lint` → `format:check`.
3. `test`: core unit suite (Node LTS + current LTS) and engine browser suite
   (Chromium; Firefox/WebKit where the runtime supports it).
4. `knip` and `size`.
5. `pack:check` on release-relevant changes.
6. Offline job: strict-offline integration test with network access blocked.

**Main branch**

- Adds the packed-tarball smoke test and publishes benchmark JSON as an
  artifact.
- Scheduled (nightly or weekly) benchmark run on a fixed reference machine so
  regressions are visible as a trend, not a surprise.

**Release**

1. Changesets version bump and changelog.
2. Full validation sequence, plus `publint` and Are the Types Wrong.
3. `npm pack --dry-run`, then publish through OIDC Trusted Publishing with
   provenance.
4. Tag, GitHub release notes linking the benchmark report, and a post-publish
   install smoke test of the published versions.

## 7. Versioning and release strategy

- `0.0.x` milestones stay private/experimental until the M6 gate.
- `0.1.0` is the first public beta; pre-1.0 breaking changes are allowed but must
  be called out in the changelog.
- `@lite-speech/core` and `@lite-speech/engine-onnx` are versioned
  independently; the engine declares core as a peer with a compatible range and
  documents the supported matrix.
- The core API is the compatibility surface; internals, worker protocol, and
  profile implementations may change between minors.
- `1.0.0` requires: stable public API for at least one minor cycle, documented
  supported runtimes, and a measured performance envelope.

## 8. Definition of done for `0.1.0`

- [ ] A pinned Phonon profile runs reproducibly in every officially supported
      runtime with pinned dependency and model revisions.
- [ ] A fully provisioned install synthesizes with network access disabled;
      missing assets fail with an actionable error.
- [ ] Download size, memory envelope, streaming class, and license are documented
      for the supported profile.
- [ ] Streaming, cancellation, and disposal are covered by automated tests,
      including chunk ordering and partial-audio semantics.
- [ ] The core public API leaks no engine types and is fully testable with a mock
      engine.
- [ ] Benchmark claims are reproducible from committed scripts and reports.
- [ ] Published artifacts pass `publint`, Are the Types Wrong, and a packed
      tarball install test.

## 9. Decisions to close

| #   | Question                                                                     | Resolve by |
| --- | ---------------------------------------------------------------------------- | ---------- |
| D1  | Does `prepare()` ship in core at `0.1.0`, or is provisioning app-side?       | M4         |
| D2  | Is one engine per instance enough, or is multi-engine fan-out required?      | M4         |
| D3  | What is the minimum granularity we are willing to call "streaming" publicly? | M1         |
| D4  | Final npm scope and package names (`@lite-speech/*` is provisional)          | M5         |
| D5  | Default dtype per device/profile after measured quality comparison           | M5         |
| D6  | Supported browser matrix and Node.js floor for `0.1.0`                       | M5         |

## 10. Risks and mitigations

| Risk                                                 | Mitigation                                                                                         |
| ---------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| Pinned Transformers.js cannot run Phonon reliably    | M1 go/no-go gate; public APIs only; defer the feature rather than hide it behind a fallback        |
| Non-commercial model licensing blocks adoption       | `license` in every manifest; document restrictions; never ship gated weights                       |
| "Streaming" overstated in marketing docs             | Per-profile streaming classification plus measured time-to-first-audio                             |
| WebGPU operator gaps produce wrong audio             | Operator/capability checks before use; `auto` may fall back with an event, explicit `webgpu` never |
| Quantization quality regressions                     | Choose dtype by measured quality and performance, not by file size                                 |
| Node.js differences (`Worker`, `Cache Storage`, GPU) | Adapter boundaries in the engine plus dedicated Node tests in CI                                   |
| Scope creep toward a model catalog                   | One verified profile for the MVP; extend behind the unchanged contract                             |
| Bundle growth in core                                | `size-limit` gate from M2, zero-runtime-dependency rule                                            |

## 11. Explicitly out of scope

Training or fine-tuning, voice cloning, built-in audio playback, a mandatory
server, bundling model weights into the npm package, a hosted inference API, and
any promise of commercial rights for non-commercial model families.

## 12. Success signals

- A developer goes from `npm install` to first audio in under ten minutes using
  only the README.
- Time-to-first-audio is reported per profile and device, and users can predict
  it from the docs.
- No network request occurs without explicit provisioning consent.
- `core` remains dependency-free and inside its size budget after `1.0`.
- Adding a model family or an engine never requires a core API break.
