/**
 * The catalogue seam: which model speaks which language, and where its files live.
 *
 * Everything here is data and lookup, never inference. That is what allows one engine to
 * serve several model families and several languages: a manifest names a `family` (which the
 * worker resolves to a runner) and the languages it is registered for, and a voice manifest
 * names the language it actually speaks. Nothing in this module knows about Phonon, ONNX,
 * WASM, or WebGPU.
 *
 * A manifest is a *pin*: `revision` is a commit SHA, never a branch. Both the download URL and
 * the cache key are derived from it, so "the bytes we measured" and "the bytes we run" cannot
 * drift apart.
 */
import type { LanguageCode, ModelManifest, VoiceManifest } from "@lite-speech/core";

/** Stable engine identifier. Echoed in `AudioResult.engine` and capabilities. */
export const ENGINE_ID = "onnx";

/** Origin of every default asset. Overridable per engine via `createOnnxEngine()`. */
const HF_BASE = "https://huggingface.co";

/** Builds the URL of one repository file at a pinned revision. */
export function assetUrl(repoId: string, revision: string, path: string): string {
  return `${HF_BASE}/${repoId}/resolve/${revision}/${path}`;
}

/**
 * The `remotePathTemplate` that pins every Hub lookup to `revision`.
 *
 * Transformers.js resolves model files *and* the tokenizer-file probe
 * (`get_tokenizer_files()` → `get_file_metadata()`, which drops the caller's `revision`
 * option) through `env.remotePathTemplate`. Left at its default
 * (`{model}/resolve/{revision}/` with `revision: "main"`), that probe asks for
 * `resolve/main/...`, which has two consequences a pinned model cannot accept: the request
 * targets a moving branch, and its Cache Storage key differs from the one a provisioned cache
 * holds — so a fully provisioned profile still misses the probe, and `offline: "strict"` makes
 * a request whose only outcome is failure.
 *
 * Pinning the template fixes both at once, because the cache key is derived from the same URL
 * the request would use. `main` is left alone: a caller who asks for a moving branch gets one.
 */
export function hubRevisionTemplate(revision: string): string {
  return `{model}/resolve/${encodeURIComponent(revision)}/`;
}

/**
 * The build variant of one asset path, e.g. `"q4"`.
 *
 * Read from the manifest's `variant` when present. The file-name heuristic is the fallback for
 * a hand-written manifest: it recognizes the ONNX convention (`flow_lm_q4.onnx`, `.onnx_data`)
 * but never *decides* which variant to run — it only labels a file that forgot to say.
 */
export function variantOf(file: {
  readonly path: string;
  readonly variant?: string;
}): string | undefined {
  if (file.variant !== undefined) {
    return file.variant;
  }
  const name = file.path.slice(file.path.lastIndexOf("/") + 1);
  const base = name.endsWith(".onnx_data")
    ? name.slice(0, -".onnx_data".length)
    : name.endsWith(".onnx")
      ? name.slice(0, -".onnx".length)
      : undefined;
  if (base === undefined) {
    return undefined;
  }
  // Longest suffix first, so `_q4f16` is never mistaken for `_q4`.
  const suffix = ONNX_VARIANT_SUFFIXES.find((candidate) => base.endsWith(`_${candidate}`));
  return suffix;
}

/** ONNX dtype suffixes a hand-written manifest may omit in its `variant` field. */
const ONNX_VARIANT_SUFFIXES = ["q4f16", "q8", "q4", "fp16", "fp32"] as const;

/**
 * The files of a manifest that one dtype actually needs.
 *
 * A manifest lists every variant so the core can price any of them; an engine must report only
 * the files its resolved dtype loads, otherwise `isCached()` would demand weights that are never
 * read and `removeModel()` would delete weights belonging to another configuration.
 *
 * A file without a variant is required by all of them (configs, tokenizers, embeddings).
 */
export function selectVariantFiles<T extends { readonly path: string; readonly variant?: string }>(
  files: readonly T[],
  variant: string,
): readonly T[] {
  return files.filter((file) => {
    const fileVariant = variantOf(file);
    return fileVariant === undefined || fileVariant === variant;
  });
}

/** A synchronous, declarative registry: the engine never awaits a model lookup. */
export interface EngineRegistry {
  /** Every language any registered model claims, in registration order. */
  languages(): readonly LanguageCode[];
  /** Every registered model, in registration order. */
  models(): readonly ModelManifest[];
  /**
   * Resolves a request language to its model.
   *
   * An exact match wins. Otherwise a region-specific code falls back to the model registered
   * for its primary subtag (`"de-AT"` → `"de"`), because refusing a dialect of a language the
   * engine can speak would help nobody: the audio comes from the same model either way, and no
   * pronunciation is promised that the model cannot deliver.
   */
  model(language: LanguageCode): ModelManifest | undefined;
  /** The repository a model's assets come from. Usually the model id itself. */
  repository(model: ModelManifest): string;
  /** Resolves a request to its voice manifest, or `undefined` when unknown. */
  voice(model: ModelManifest, voice: string | undefined): VoiceManifest | undefined;
  /** Every voice of `model`, in registration order. */
  voices(model: ModelManifest): readonly VoiceManifest[];
}

/** The data a registry is built from. */
export interface RegistryInput {
  readonly models: readonly ModelManifest[];
  readonly voices?: readonly VoiceManifest[];
}

/** One model plus everything derived from the manifests once, at creation time. */
interface CatalogEntry {
  readonly model: ModelManifest;
  readonly voices: readonly VoiceManifest[];
  readonly languages: readonly LanguageCode[];
}

/** The primary subtag of a language tag: `"de-AT"` → `"de"`. */
function primarySubtag(language: LanguageCode): LanguageCode {
  return language.split("-")[0] ?? language;
}

/**
 * Languages a model is registered for.
 *
 * An explicit `languages` list wins, so a multilingual checkpoint can be registered under
 * several codes. Without one, the languages of its own voices are used — which keeps a
 * hand-written manifest with nothing but a voice list valid.
 */
export function languagesOf(
  model: ModelManifest,
  voices: readonly VoiceManifest[],
): readonly LanguageCode[] {
  if (model.languages !== undefined) {
    return model.languages;
  }
  return [
    ...new Set(voices.filter((voice) => voice.modelId === model.id).map((voice) => voice.language)),
  ];
}

/**
 * Creates the registry.
 *
 * No defaults: which models exist is a catalogue decision, and a registry that silently invented
 * one would make "the library downloaded a model I never named" possible. The built-in catalogue
 * passes its own manifests instead.
 */
export function createRegistry(input: RegistryInput): EngineRegistry {
  const allVoices = input.voices ?? [];
  const entries: readonly CatalogEntry[] = input.models.map((model) => ({
    model,
    voices: allVoices.filter((voice) => voice.modelId === model.id),
    languages: languagesOf(model, allVoices),
  }));

  const entryFor = (model: ModelManifest): CatalogEntry | undefined =>
    entries.find((entry) => entry.model === model || entry.model.id === model.id);

  return {
    languages(): readonly LanguageCode[] {
      return [...new Set(entries.flatMap((entry) => entry.languages))];
    },
    models(): readonly ModelManifest[] {
      return entries.map((entry) => entry.model);
    },
    model(language: LanguageCode): ModelManifest | undefined {
      const exact = entries.find((entry) => entry.languages.includes(language));
      if (exact !== undefined) {
        return exact.model;
      }
      const primary = primarySubtag(language);
      return entries.find((entry) =>
        entry.languages.some((registered) => primarySubtag(registered) === primary),
      )?.model;
    },
    repository(model: ModelManifest): string {
      return model.id;
    },
    voice(model: ModelManifest, voice: string | undefined): VoiceManifest | undefined {
      const candidates = entryFor(model)?.voices ?? [];
      if (voice === undefined) {
        return candidates[0];
      }
      return candidates.find((candidate) => candidate.id === voice);
    },
    voices(model: ModelManifest): readonly VoiceManifest[] {
      return entryFor(model)?.voices ?? [];
    },
  };
}
