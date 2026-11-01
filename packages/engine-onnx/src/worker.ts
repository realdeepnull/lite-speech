/**
 * The worker entry: protocol plumbing only.
 *
 * `self.onmessage` is not serialized by the platform, so two messages could otherwise
 * interleave — a `load` awaiting a multi-megabyte download while a `synthesize` runs
 * against a half-initialized model. Every request is chained onto one promise, which also
 * makes the observable order identical to the posted order. `cancel` bypasses the queue
 * deliberately: it must take effect while a long generation is still running.
 *
 * Everything interesting lives in `model-host.ts` (invariants), `families.ts` (the seam), and
 * one module per family (`phonon.ts`). This file exists so the model runtime is reachable as
 * its own build entry and package export: a bundler can see it explicitly, and it never lands
 * in the main-thread bundle.
 */
import { createModelHost, type CacheOpener } from "./model-host.js";
import { DEFAULT_FAMILIES } from "./runners.js";
import type { WorkerRequest, WorkerResponse } from "./worker-protocol.js";
import { transferListFor } from "./worker-protocol.js";

/** The scope the worker runs in, described without a `lib.dom` dependency. */
interface WorkerScope {
  onmessage: ((event: MessageEvent<WorkerRequest>) => void) | null;
  postMessage(message: WorkerResponse, transfer?: Transferable[]): void;
}

const scope = self as unknown as WorkerScope;

/** Posts a response, moving PCM buffers instead of cloning them. */
function post(response: WorkerResponse): void {
  const transfer = transferListFor(response);
  if (transfer.length > 0) {
    scope.postMessage(response, transfer);
  } else {
    scope.postMessage(response);
  }
}

/**
 * Opens the browser cache Transformers.js writes to.
 *
 * Absent outside the browser: Node.js caches to the filesystem, which a worker cannot
 * reach, so the engine falls back to its own probe rather than to a guess.
 */
const openCache: CacheOpener | undefined =
  typeof caches === "undefined"
    ? undefined
    : () => caches.open("transformers-cache") as unknown as ReturnType<CacheOpener>;

const host = createModelHost({
  families: DEFAULT_FAMILIES,
  post,
  ...(openCache !== undefined && { openCache }),
});

/**
 * Requests are serialized, except cancellation, which is applied immediately.
 *
 * The host reports request errors through `post`, so a rejection here should not happen;
 * the empty `catch` keeps the chain alive anyway, so one bad message can never silence
 * every later request.
 */
let queue: Promise<void> = Promise.resolve();

scope.onmessage = (event: MessageEvent<WorkerRequest>): void => {
  const request = event.data;
  if (request.kind === "cancel") {
    host.cancel(request);
    return;
  }
  queue = queue.then(() => host.dispatch(request)).catch(() => undefined);
};
