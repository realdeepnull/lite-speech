/**
 * Node.js entry point.
 *
 * Node has no `Worker` in its global scope and no Cache Storage, so the browser entry cannot
 * simply be reused: inference has to run in-thread and cache inspection has to read the
 * filesystem. This module wires both — and it is the only entry that imports the model
 * runtime, which is why it is a subpath rather than part of the browser bundle.
 *
 * @example
 * ```ts
 * import { createTts } from "@lite-speech/core";
 * import { createNodeOnnxEngine } from "@lite-speech/engine-onnx/node";
 *
 * const engine = await createNodeOnnxEngine();
 * const tts = createTts({ engine, offline: "allow" });
 * const { pcm, sampleRate } = await tts.synthesize("Hello from Node.js.");
 * ```
 */
import { join } from "node:path";
import { stat, unlink } from "node:fs/promises";

import { createOnnxEngine, type OnnxEngine, type OnnxEngineOptions } from "./create-onnx-engine.js";
import { loadPhonon } from "./phonon.js";
import type { FsProbe } from "./transformers-engine.js";

/** Options for {@link createNodeOnnxEngine}. */
export interface NodeOnnxEngineOptions extends Omit<
  OnnxEngineOptions,
  "worker" | "inThreadLoader" | "fsProbe" | "device"
> {
  /**
   * Device under `onnxruntime-node`. Only `"cpu"` is meaningful; it is the default and the only
   * accepted value, because an explicit `"webgpu"` "succeeds" there without a GPU (M1 §5.1).
   */
  readonly device?: "cpu";
  /**
   * Directory Transformers.js uses for downloaded files. Defaults to the layout the library
   * picks itself (`.cache/` beside its own `dist/`), so no environment variable is needed for
   * the common case.
   */
  readonly cacheDir?: string;
}

/**
 * Resolves the cache directory Transformers.js uses by default.
 *
 * The library computes it from its own module location, not from `process.cwd()`, and a Node
 * program is usually started somewhere else entirely — so the probe mirrors that resolution
 * instead of guessing.
 */
function defaultCacheDir(): string {
  const { createRequire } = globalThis.process.getBuiltinModule("node:module");
  const require = createRequire(import.meta.url);
  const entry = require.resolve("@huggingface/transformers");
  // `…/@huggingface/transformers/dist/transformers.node.mjs`; the cache sits at the package root.
  return join(entry, "..", "..", ".cache");
}
/**
 * Filesystem cache probe.
 *
 * Transformers.js stores Node downloads under `<cacheDir>/<owner>/<model>/<revision>/<path>`.
 * Reproducing that layout is the only way `isCached()` can be honest in Node without Cache
 * Storage, so it is written down once, here, instead of being inferred at each call site.
 */
function createFileSystemProbe(cacheDir: string): FsProbe {
  /** Maps a Hub URL onto the file Transformers.js would have written. */
  const localPath = (url: string): string | undefined => {
    const marker = "huggingface.co/";
    const origin = url.indexOf(marker);
    if (origin < 0) {
      return undefined;
    }
    const parts = url.slice(origin + marker.length).split("/");
    const [owner, name, resolve, revision] = parts;
    if (
      owner === undefined ||
      name === undefined ||
      resolve !== "resolve" ||
      revision === undefined
    ) {
      return undefined;
    }
    const filePath = parts.slice(4).join("/");
    return filePath === "" ? undefined : join(cacheDir, owner, name, revision, filePath);
  };

  return {
    async cached(urls) {
      return Promise.all(
        urls.map(async (url) => {
          const path = localPath(url);
          if (path === undefined) {
            return { cached: false };
          }
          try {
            const stats = await stat(path);
            return stats.isFile() ? { cached: true, size: stats.size } : { cached: false };
          } catch {
            return { cached: false };
          }
        }),
      );
    },
    async remove(urls) {
      await Promise.all(
        urls.map(async (url) => {
          const path = localPath(url);
          if (path === undefined) {
            return;
          }
          try {
            await unlink(path);
          } catch {
            // Already absent is the state the caller asked for.
          }
        }),
      );
    },
  };
}

/**
 * Creates an engine that runs inference in the current thread.
 *
 * There is no worker in Node, so backpressure comes from the same pull-based protocol with the
 * host called directly: each `stream-next` runs one graph step and returns. The public API is
 * identical to the browser engine, which is what lets one application share code across both.
 */
export async function createNodeOnnxEngine(
  options: NodeOnnxEngineOptions = {},
): Promise<OnnxEngine> {
  const { cacheDir, ...rest } = options;
  const resolvedCacheDir = cacheDir ?? defaultCacheDir();
  return createOnnxEngine({
    ...rest,
    device: "cpu",
    worker: false,
    inThreadLoader: (loadOptions, voice) =>
      loadPhonon({ ...loadOptions, cacheDir: resolvedCacheDir }, voice),
    fsProbe: createFileSystemProbe(resolvedCacheDir),
  });
}
