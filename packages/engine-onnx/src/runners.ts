/**
 * The built-in family registry: family id → runner factory.
 *
 * This module is the one place that names a concrete runner, so it is imported only by the
 * worker entry, the Node entry, and the factory default — never by a module the engine's
 * main-thread surface needs at runtime beyond that one default. What must stay out of an
 * application's main bundle is the runner *implementation*, which lives in `phonon.ts` and is
 * reachable only through `./worker`.
 *
 * A new family is a new entry here. It does not change the worker protocol, the engine, or the
 * core contract, which is the whole point of the seam.
 */
import type { FamilyRegistry, RunnerLoadOptions, ModelRunner } from "./families.js";
import { loadPhonon } from "./phonon.js";

/** The Phonon family: `flow_lm` + `mimi_decoder`, driven frame by frame. */
const PHONON_RUNNER: FamilyRegistry[string] = (options) => loadPhonon(options);

/** Every family this package can run. */
export const DEFAULT_FAMILIES: FamilyRegistry = {
  phonon: PHONON_RUNNER,
};

/**
 * Returns the same families with a fixed cache directory.
 *
 * Only meaningful where the runtime caches to the filesystem (Node.js): Transformers.js resolves
 * that directory from its own module location, so a Node program started elsewhere needs it set
 * explicitly. Wrapping the factories keeps the runner unaware of *why* the directory matters.
 */
export function withCacheDir(families: FamilyRegistry, cacheDir: string): FamilyRegistry {
  return Object.fromEntries(
    Object.entries(families).map(([family, factory]) => [
      family,
      (options: RunnerLoadOptions): Promise<ModelRunner> => factory({ ...options, cacheDir }),
    ]),
  );
}
