# Provisioning model assets

`lite-speech` never downloads a model as a side effect of loading a page. Inference
needs bytes that are not in the npm package, so somebody has to fetch them — once,
explicitly, where the application can see and price it.

This document covers that "once": the consent flow, the budget, integrity checks, and
two build-time recipes (a script and a CI cache step). No CLI and no extra package is
required, by design — [M4](../docs/ROADMAP.md#m4--assets-offline-policy-and-provisioning)
deliberately does not ship one.

---

## 1. The two states

| `offline`            | Meaning                                                                                                   |
| -------------------- | --------------------------------------------------------------------------------------------------------- |
| `"strict"` (default) | Synthesis never touches the network. Missing assets reject in the **core**, before the engine is invoked. |
| `"allow"`            | The engine may download on demand, the way a plain Transformers.js app does.                              |

`"strict"` is the default because the alternative — a library that silently fetches
68 MB the first time a user presses "Speak" — is the behaviour this project exists to
avoid. An application that genuinely wants on-demand downloads opts in once at
`createTts()`.

The policy is enforced in **one place**:

```ts
synthesize() ─┐
stream()     ─┼─> #assertOfflineUsable() ─> engine.load()
preload()    ─┘
```

`preload()` is included deliberately. It is the shortcut a caller reaches for to
"warm up" without synthesizing, and a policy that `preload()` could bypass would not
be a policy. A test asserts that a strict instance asks the engine for nothing when
the assets are missing.

## 2. The consent flow

```ts
const engine = await createOnnxEngine({ device: "auto" });
const tts = createTts({ engine, offline: "strict" });

// What the dialog shows. No bytes are fetched to answer this.
const plan = await tts.describeAssets();
const estimate = await tts.estimateDownload();

console.log(`${estimate.megabytes.toFixed(1)} MB in ${estimate.files.length} files`);
console.log(plan.licenses); // e.g. ["Apache-2.0"]

// Only now, after consent.
await tts.prepare({
  allowNetwork: true,
  maxDownloadMB: 120,
  onProgress: (event) => progressBar.set(event.progress),
  verify: true, // hash every byte that was written
});
```

`describeAssets()` is synchronous in cost terms — it stats the cache and reads the
manifest — so it is safe to call on render. `prepare()` is the only entry point that
may download, and it needs `allowNetwork: true` explicitly.

### The budget is checked before the first byte

`maxDownloadMB` is a decimal-megabyte limit (1 MB = 1 000 000 bytes) applied to
`plan.missingBytes`, and it rejects with `DOWNLOAD_BUDGET_EXCEEDED` **before** the
engine is called:

```
TtsError: Model onnx-community/phonon-31m-en-ONNX needs 67989146 bytes,
          which exceeds the 60 MB budget
  details: { requiredBytes, budgetBytes, files: [{ path, size }], modelId }
```

The `files` array is the actionable part: it names exactly what did not fit, so a UI
can suggest a smaller dtype instead of an unhelpful "too big".

A plan whose files do not all declare a size is rejected too. A budget the library
cannot compute is not a budget it should promise.

### Integrity

Every file in the pinned manifest carries a SHA-256 digest. Two moments can verify it:

| Moment                      | What happens                                                                                                                |
| --------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| Download of a voice file    | Checked **before** the bytes are cached, so a corrupt file never enters the cache.                                          |
| `prepare({ verify: true })` | Every cached file is read back and hashed after provisioning. `ASSET_INTEGRITY_FAILED` on a mismatch or an unreadable file. |
| `tts.verifyAssets()`        | The same check, callable at any time — e.g. from a diagnostics screen.                                                      |

A file that was **not cached** is never read (verifying must not become downloading),
and a file with **no declared digest** is reported as `unverified` rather than assumed
good. "I could not look" and "I looked and it was fine" are different answers, and the
error code only fires for the second one failing.

Reading 68 MB back to hash it is not free, which is why `verify` defaults to `false`
for a warm cache and belongs in a cold-machine provisioning step or a build script.

### All-or-nothing writes

`prepare()` snapshots which files were cached before it starts. If the download fails
— connection dropped, budget exhausted by a race, worker crash — it deletes what the
attempt added and disposes a partially loaded model. One provisioning attempt is one
transaction:

- a retry starts from the same state, so the reported `missing` list is stable;
- a half-written profile can never be mistaken for a provisioned one;
- files from an **earlier successful** provisioning are left alone, because they were
  not added by this attempt.

The snapshot is the important half. A rollback that cannot tell "added now" from
"already there" would delete another profile's assets.

## 3. Build-time pre-download

The cleanest provisioning is the one that happened before the user arrived. Two
recipes, both plain Node scripts.

### 3.1 Cache into a directory you ship

Node.js caches to the filesystem. Point the engine at a directory, run one
synthesis-like preparation, and copy the result wherever your deployment needs it:

```js
// scripts/fetch-model.mjs   (run once, e.g. `node scripts/fetch-model.mjs`)
import { createTts } from "@lite-speech/core";
import { createNodeOnnxEngine } from "@lite-speech/engine-onnx/node";

const cacheDir = new URL("../.model-cache/", import.meta.url).pathname;

const engine = await createNodeOnnxEngine({ cacheDir, dtype: "q4" });
const tts = createTts({ engine, offline: "allow" });

const estimate = await tts.estimateDownload();
console.log(`fetching ${estimate.megabytes.toFixed(1)} MB`);

const plan = await tts.prepare({
  allowNetwork: true,
  verify: true,
  onProgress: ({ phase, loaded, total }) => {
    if (total > 0) process.stdout.write(`\r${phase} ${Math.round((loaded / total) * 100)}%`);
  },
});

console.log(`\n${plan.files.length} files, ${plan.cachedBytes} bytes cached`);
await tts.dispose();
```

The layout under `cacheDir` mirrors the Hub URL:

```text
.model-cache/
└─ onnx-community/
   └─ phonon-31m-en-ONNX/
      └─ 4f0cda7c60dddb4862d0e68ff9ff60c32ffdc9f4/
         ├─ config.json
         ├─ tokenizer.json
         ├─ tokenizer_config.json
         ├─ onnx/flow_lm_q4.onnx
         ├─ onnx/flow_lm_q4.onnx_data
         ├─ onnx/mimi_decoder_q4.onnx
         └─ voices/default.bin
```

Ship that tree (or bake it into a container image) and point the runtime at the same
directory. The second run reports `complete: true` and never asks the network.

### 3.2 Warm a browser cache in Playwright

For a browser deployment the cache that matters is **Cache Storage**, which only a
browser can write. A Playwright step provisions it once and the profile directory is
reused by later runs:

```js
// scripts/warm-browser-cache.mjs
import { chromium } from "playwright";

const browser = await chromium.launchPersistentContext(".browser-profile", { headless: true });
const page = await browser.newPage();
await page.goto("http://localhost:8080/examples/demo/");

const result = await page.evaluate(async () => {
  const { createTts } = await import("@lite-speech/core");
  const { createOnnxEngine } = await import("@lite-speech/engine-onnx");
  const engine = await createOnnxEngine({ device: "wasm", dtype: "q4" });
  const tts = createTts({ engine, offline: "allow" });
  const plan = await tts.prepare({ allowNetwork: true, verify: true });
  await tts.dispose();
  return { complete: plan.complete, bytes: plan.cachedBytes };
});

console.log(result);
await browser.close();
```

The persistent context is the point: `Cache Storage` lives in the profile, so the
warmed profile is what a later run (or an Electron shell, or a Chromium kiosk) loads.

## 4. Eviction

```ts
await tts.removeModel(); // disposes the session, then deletes every cached file
```

`removeModel()` disposes the loaded model first, then sweeps **every** Cache Storage
cache by URL — not just the library's default name, because an application may have
renamed it — and every filesystem path the Node probe knows about. It leaves the
instance usable, so the next `preload()` loads again.

## 5. Checking the state from a UI

```ts
const plan = await tts.describeAssets();
plan.complete; // every file is cached
plan.sizeMismatches; // cached files whose size is not the declared size
plan.integrity; // "declared" | "partial" | "none"
plan.missing; // paths that would have to be downloaded
plan.totalBytes; // the whole profile
plan.cachedBytes; // what is already there
plan.licenses; // license strings, model first, then voice
```

`sizeMismatches` is the cheap corruption signal: it catches a truncated file without
reading it. `verifyAssets()` is the expensive, conclusive one.

## 6. What the tests assert

The M4 gate is only real if it is tested end to end:

| Claim                                        | Test                                                               |
| -------------------------------------------- | ------------------------------------------------------------------ |
| Strict offline rejects before the engine     | `tests/offline-network.test.ts` (a recording `fetch` that rejects) |
| Hub lookups stay on the pinned revision      | `tests/revision-pinning.test.ts` (drives Transformers.js itself)   |
| A provisioned cache answers every path       | `tests/engine.browser.test.ts` (real Cache Storage)                |
| `preload()` cannot bypass the policy         | `packages/core/tests/offline.test.ts`                              |
| A budget rejects before the engine is called | `packages/core/tests/offline.test.ts`, `tests/engine.test.ts`      |
| Integrity failures are reported, not hidden  | `packages/core/tests/integrity.test.ts`, `tests/engine.test.ts`    |
| A failed provisioning rolls back             | `tests/engine.test.ts`                                             |
| Real digests match in a real browser cache   | `tests/engine.browser.test.ts`                                     |

`estimateDownload()` matching the observed byte count is the one gate that needs a
live Hub; the benchmark report records the observed figure against
`plan.totalBytes` rather than a stored constant, so a manifest edit that forgets a file
shows up as a mismatch.

## 7. Revision pinning is part of the offline guarantee

Transformers.js builds every Hub URL from `env.remotePathTemplate`. Its tokenizer-file
probe (`get_tokenizer_files()`) calls `get_file_metadata()` **without** forwarding the
`revision` option, so a load pinned to a commit still asks the Hub about
`resolve/main/tokenizer_config.json`. In a browser the Cache Storage key comes from the
same URL, so the probe also _misses_ a complete cache.

The engine therefore pins the template to the manifest revision when it loads. Both
consequences disappear at once, because the request URL and the cache key are the same
string. The rule is one line and exported for testing:

```ts
hubRevisionTemplate("4f0cda7…") === "{model}/resolve/4f0cda7…/";
```

`tests/revision-pinning.test.ts` asserts that default behaviour still targets `main`
(so the test would notice if upstream changed) and that the pinned template targets the
commit.
