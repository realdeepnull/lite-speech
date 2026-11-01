/**
 * The Phonon graph runner.
 *
 * Drives the two ONNX graphs directly (`flow_lm`, then `mimi_decoder`) because the pinned
 * Transformers.js release has no model class for this architecture (M1 §4). Every member used
 * here is public: `AutoConfig`, `AutoTokenizer`, `PreTrainedModel`, `Tensor`, and the
 * tokenizer's `tokenize` / `get_vocab` / `convert_tokens_to_ids`.
 *
 * Only the worker entry imports this module, which is what keeps the model runtime out of the
 * engine's main-thread bundle.
 */
import {
  AutoConfig,
  AutoTokenizer,
  PreTrainedModel,
  Tensor,
  env,
  type PreTrainedTokenizer,
} from "@huggingface/transformers";

import type { ProgressEvent } from "@lite-speech/core";
import { assetUrl, hubRevisionTemplate } from "./catalog.js";
import type { ModelProfile, ModelRunner, RunnerLoadOptions } from "./families.js";
import {
  PHONON_PROFILE,
  normalizeSegment,
  phononMaxFrames,
  readPhononConfig,
  type PhononConfig,
} from "./phonon-profile.js";
/** What one session call returns: named output tensors. */
type GraphOutputs = Record<string, Tensor>;

/** A callable ONNX session as exposed by `PreTrainedModel`. */
type GraphSession = ((inputs: Record<string, unknown>) => Promise<GraphOutputs>) & {
  dispose(): Promise<void>;
};

/** A tensor state map: the KV cache, the decoder convolution state, and friends. */
type TensorState = Record<string, Tensor>;

/** Maximum accepted voice-file size, so a wrong URL cannot be read into memory. */
const MAX_VOICE_BYTES = 8 * 1024 * 1024;

/**
 * Base names of the two graphs.
 *
 * These are the unsuffixed stems: Transformers.js appends the dtype suffix itself
 * (`flow_lm` → `flow_lm_q4.onnx`), so passing a suffixed name would request
 * `flow_lm_q4_q4.onnx`.
 */
const FLOW_GRAPH = "flow_lm";
const MIMI_GRAPH = "mimi_decoder";

/** Builds a `float32` tensor from `dims` and optional data. */
function f32(dims: readonly number[], data?: Float32Array): Tensor {
  return new Tensor(
    "float32",
    data ?? new Float32Array(dims.reduce((total, dim) => total * dim, 1)),
    [...dims],
  );
}

/** Builds an `int64` tensor. */
function i64(values: readonly number[], dims: readonly number[]): Tensor {
  return new Tensor(
    "int64",
    BigInt64Array.from(values, (value) => BigInt(value)),
    [...dims],
  );
}

/** `present.*` cache names of one graph, in layer order. */
function cacheNames(layers: number): string[] {
  return Array.from({ length: layers }, (_, index) => [
    `present.${index}.key`,
    `present.${index}.value`,
  ]).flat();
}

/** An empty transformer cache: one `[1, heads, 0, headDim]` tensor per layer and kind. */
function emptyCache(layers: number, heads: number, headDim: number): TensorState {
  const state: TensorState = {};
  for (let layer = 0; layer < layers; layer += 1) {
    for (const kind of ["key", "value"]) {
      state[`past_key_values.${layer}.${kind}`] = f32([1, heads, 0, headDim]);
    }
  }
  return state;
}

/**
 * Maps every `present*` output of a call to the `past*` input of the next one.
 *
 * One mapping covers both graphs: the flow graph emits `present.<n>.key`, the decoder adds
 * `present_offset` and `present_conv.*`, and each input is named by the same two substitutions.
 */
function nextState(outputs: GraphOutputs): TensorState {
  const state: TensorState = {};
  for (const [name, value] of Object.entries(outputs)) {
    if (!name.startsWith("present")) {
      continue;
    }
    state[name.replace("present.", "past_key_values.").replace("present_", "past_")] = value;
  }
  return state;
}

/**
 * Frees state tensors that were replaced.
 *
 * `dispose()` is a no-op for CPU tensors and mandatory for `gpu-buffer` ones: the runtime does not
 * garbage-collect them, and leaking one cache set per frame would exhaust the GPU heap long
 * before a sentence ends.
 */
function release(...states: TensorState[]): void {
  for (const state of states) {
    for (const tensor of Object.values(state)) {
      if (tensor.location !== "cpu") {
        tensor.dispose();
      }
    }
  }
}

/** Converts a Transformers.js progress record into the core's `ProgressEvent`. */
function toProgressEvent(info: Record<string, unknown>): ProgressEvent | undefined {
  const status = info["status"];
  if (status !== "progress" && status !== "download") {
    return undefined;
  }
  const file = typeof info["file"] === "string" ? info["file"] : undefined;
  const loaded = typeof info["loaded"] === "number" ? info["loaded"] : 0;
  const total = typeof info["total"] === "number" ? info["total"] : 0;
  return {
    phase: status === "download" ? "model-download" : "model-load",
    loaded,
    total,
    progress: total > 0 ? loaded / total : Number.NaN,
    ...(file !== undefined && { file }),
  };
}

/**
 * Encodes text with public tokenizer members only.
 *
 * The pinned release does not apply Unigram byte fallback (M1 §8), so a piece outside the vocabulary
 * would collapse into `<unk>`. Every piece from `tokenize()` is therefore kept, or expanded into
 * its UTF-8 `<0xXX>` byte tokens, which this vocabulary contains for all 256 byte values. The spike
 * verified this reproduces the model card's private-field patch byte for byte at temperature `0`.
 *
 * Byte fallback is also what makes a non-Latin segment representable at all, so the encoder stays
 * language-agnostic: the profile decides what the text says, the encoder only makes its bytes
 * addressable.
 */
function createPublicEncoder(tokenizer: PreTrainedTokenizer): (text: string) => readonly number[] {
  const vocab = tokenizer.get_vocab();
  const encoder = new TextEncoder();
  const byteToken = (byte: number): string =>
    `<0x${byte.toString(16).toUpperCase().padStart(2, "0")}>`;
  return (text: string): readonly number[] => {
    const ids: number[] = [];
    for (const piece of tokenizer.tokenize(text)) {
      if (vocab.has(piece)) {
        ids.push(tokenizer.convert_tokens_to_ids(piece));
        continue;
      }
      const bytes = Array.from(encoder.encode(piece), byteToken);
      if (bytes.every((token) => vocab.has(token))) {
        for (const token of bytes) {
          ids.push(tokenizer.convert_tokens_to_ids(token));
        }
      } else {
        ids.push(tokenizer.convert_tokens_to_ids(piece));
      }
    }
    return ids;
  };
}

/** Cache name Transformers.js uses, so voice files are found by the same cache lookups. */
const MODEL_CACHE_NAME = "transformers-cache";

/** Returns the SHA-256 of `bytes` as lowercase hex, or `undefined` without `crypto.subtle`. */
async function sha256Hex(bytes: Uint8Array): Promise<string | undefined> {
  const subtle = (globalThis as { crypto?: { subtle?: SubtleCrypto } }).crypto?.subtle;
  if (subtle === undefined) {
    return undefined;
  }
  const digest = await subtle.digest("SHA-256", bytes as unknown as BufferSource);
  return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, "0")).join("");
}

/**
 * Points Transformers.js' remote path template at an immutable revision.
 *
 * See {@link hubRevisionTemplate} for why the template, not the URL, is the thing to fix. A
 * worker hosts exactly one model and the engine disposes before changing model, device, or
 * dtype, so one pinned template per load is the honest global.
 */
function pinRevisionTemplate(revision: string): void {
  if (revision === "main") {
    return;
  }
  env.remotePathTemplate = hubRevisionTemplate(revision);
}

/** Opens the model cache, tolerating an origin where Cache Storage is unavailable. */
async function openModelCache(): Promise<Cache | undefined> {
  try {
    return await caches.open(MODEL_CACHE_NAME);
  } catch {
    return undefined;
  }
}

/** Box–Muller normal sample, matching the spike's sampler. */
function randomNormal(): number {
  return Math.sqrt(-2 * Math.log(1 - Math.random())) * Math.cos(2 * Math.PI * Math.random());
}

/**
 * Downloads and validates one voice embedding.
 *
 * The voice is fetched here rather than through Transformers.js, so it is cached here too — with
 * the same Cache Storage API and the same cache name the library uses. Without that, `isCached()`
 * would report the voice as missing forever: `offline: "strict"` could never work in a browser, and
 * `removeModel()` would leave the embedding behind.
 *
 * In Node.js there is no Cache Storage, so this reads over the network and the filesystem probe
 * reports the file absent — accurate, because nothing writes it there.
 */
async function loadVoice(options: RunnerLoadOptions, config: PhononConfig): Promise<Tensor> {
  const { voice } = options;
  const url = assetUrl(voice.repository, voice.revision, voice.path);
  const cache = typeof caches === "undefined" ? undefined : await openModelCache();
  const cached = await cache?.match(url);
  const response = cached ?? (await fetch(url));
  if (!response.ok) {
    throw new Error(`Voice download failed with HTTP ${String(response.status)}: ${url}`);
  }
  const bytes = new Uint8Array(await response.arrayBuffer());
  if (voice.sha256 !== undefined) {
    const actual = await sha256Hex(bytes);
    if (actual !== voice.sha256) {
      throw new Error(
        `Voice ${voice.path} failed its integrity check: expected sha256 ${voice.sha256}, ` +
          `got ${actual ?? "no digest (this runtime has no crypto.subtle)"}`,
      );
    }
  }
  if (cache !== undefined && cached === undefined) {
    // Store from the bytes already read: a `Response` body can only be consumed once. The
    // `content-length` header is set explicitly so `describeAssets()` can report the real size of
    // a cached voice and compare it with the manifest, like it does for every other file.
    await cache.put(
      url,
      new Response(bytes, {
        headers: {
          "content-type": "application/octet-stream",
          "content-length": String(bytes.byteLength),
        },
      }),
    );
  }
  const columnBytes = config.voiceDim * Float32Array.BYTES_PER_ELEMENT;
  if (
    bytes.byteLength === 0 ||
    bytes.byteLength % columnBytes !== 0 ||
    bytes.byteLength > MAX_VOICE_BYTES
  ) {
    throw new Error(
      `Voice file ${voice.path} is not a whole number of ${String(config.voiceDim)}-float columns ` +
        `(${String(bytes.byteLength)} bytes)`,
    );
  }
  const data = new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
  return f32([1, config.voiceDim, data.length / config.voiceDim], data);
}

/**
 * Loads the configuration, tokenizer, both graphs, and one voice; returns a runner.
 *
 * Order is deliberate: configuration first (it carries the cache shapes), then the tokenizer (a
 * small download that makes progress visible before the graphs start), then the graphs, then the
 * voice.
 */
export async function loadPhonon(options: RunnerLoadOptions): Promise<ModelRunner> {
  // A dev-server SPA fallback answers unknown URLs with `index.html` and status 200, and
  // Transformers.js would parse that HTML as ONNX. Remote Hub assets only.
  env.allowLocalModels = false;
  env.allowRemoteModels = true;
  // Cache to Cache Storage only where it exists; under Node.js the library must keep using its
  // filesystem cache, and forcing the browser cache there makes every download fail.
  env.useBrowserCache = typeof caches !== "undefined";
  if (options.cacheDir !== undefined) {
    env.cacheDir = options.cacheDir;
  }
  // A tokenizer probe inside Transformers.js drops the `revision` option, so it would otherwise
  // request `resolve/main/...` *and* cache under that key. See `pinRevisionTemplate`.
  pinRevisionTemplate(options.revision);

  const progress_callback = options.onProgress
    ? (info: unknown): void => {
        const event = toProgressEvent((info ?? {}) as Record<string, unknown>);
        if (event) {
          options.onProgress?.(event);
        }
      }
    : undefined;

  const config = await AutoConfig.from_pretrained(options.modelId, {
    revision: options.revision,
    ...(progress_callback && { progress_callback }),
  });
  const phonon = readPhononConfig(config);
  const tokenizer = await AutoTokenizer.from_pretrained(options.modelId, {
    revision: options.revision,
    ...(progress_callback && { progress_callback }),
  });

  // Recurrent state must stay on the GPU when WebGPU is the device; otherwise every frame
  // round-trips host memory and the GPU loses its only advantage (M1 §10).
  const gpuOutputs = (names: readonly string[]): Record<string, unknown> =>
    options.device === "webgpu"
      ? {
          session_options: {
            preferredOutputLocation: Object.fromEntries(names.map((name) => [name, "gpu-buffer"])),
          },
        }
      : {};

  const flow = (await PreTrainedModel.from_pretrained(options.modelId, {
    revision: options.revision,
    config,
    model_file_name: FLOW_GRAPH,
    dtype: options.dtype as never,
    device: options.device as never,
    ...(progress_callback && { progress_callback }),
    ...gpuOutputs(["latent_out", ...cacheNames(phonon.flow.numLayers)]),
  })) as unknown as GraphSession;
  const mimi = (await PreTrainedModel.from_pretrained(options.modelId, {
    revision: options.revision,
    config,
    model_file_name: MIMI_GRAPH,
    dtype: options.dtype as never,
    device: options.device as never,
    ...(progress_callback && { progress_callback }),
    ...gpuOutputs([
      ...cacheNames(phonon.mimi.numLayers),
      "present_offset",
      ...Object.keys(phonon.mimi.convStates).map((name) => name.replace("past_", "present_")),
    ]),
  })) as unknown as GraphSession;

  const encode = createPublicEncoder(tokenizer);
  const voiceTensor = await loadVoice(options, phonon);

  /**
   * One flow-graph call.
   *
   * The first call also carries the voice embedding, which is when the speaker prompt is computed;
   * later calls pass an empty `voice_latents` and rely on the cached state.
   */
  const runFlow = (
    past: TensorState,
    text: readonly number[],
    inputs: { voiceLatents?: Tensor; latent?: Tensor; noise?: Tensor } = {},
  ): Promise<GraphOutputs> =>
    flow({
      text_ids: i64(text, [1, text.length]),
      voice_latents: inputs.voiceLatents ?? f32([1, phonon.voiceDim, 0]),
      latent: inputs.latent ?? f32([1, 0, phonon.latentDim]),
      noise: inputs.noise ?? f32([1, phonon.latentDim]),
      num_speakers: i64([1], [1]),
      duration_delta: f32([1]),
      ...past,
    });

  // The voice prompt is computed once and reused, so every segment starts from the same speaker
  // state instead of re-encoding the 256 kB embedding per sentence.
  const voiceState = nextState(
    await runFlow(
      emptyCache(phonon.flow.numLayers, phonon.flow.numHeads, phonon.flow.headDim),
      [],
      { voiceLatents: voiceTensor },
    ),
  );

  const profile: ModelProfile = {
    ...PHONON_PROFILE,
    frameRate: phonon.frameRate,
    maxFrames: (prepared: string, inputLength: number): number =>
      phononMaxFrames(prepared, inputLength, phonon),
  };

  const synthesize = async function* (
    segment: string,
    runOptions: {
      readonly isCancelled: () => boolean;
      readonly temperature?: number | undefined;
    },
  ): AsyncGenerator<Float32Array> {
    const text = normalizeSegment(segment);
    if (text.length === 0) {
      return;
    }
    const ids = encode(text);
    if (ids.length === 0) {
      return;
    }
    let state = nextState(await runFlow(voiceState, ids));
    let codec: TensorState = {
      past_offset: i64([0], [1]),
      ...emptyCache(phonon.mimi.numLayers, phonon.mimi.numHeads, phonon.mimi.headDim),
    };
    for (const [name, dims] of Object.entries(phonon.mimi.convStates)) {
      codec[name] = f32(dims);
    }

    // `NaN` marks the first frame, which starts from an empty decoder state.
    let latent = f32([1, 1, phonon.latentDim], new Float32Array(phonon.latentDim).fill(Number.NaN));
    const temperature = runOptions.temperature ?? phonon.temperature;
    const maxFrames = profile.maxFrames?.(text, ids.length) ?? Number.POSITIVE_INFINITY;
    let countdown: number | null = null;

    try {
      for (let frame = 0; frame < maxFrames; frame += 1) {
        if (runOptions.isCancelled()) {
          return;
        }
        const noise = f32(
          [1, phonon.latentDim],
          Float32Array.from(
            { length: phonon.latentDim },
            () => randomNormal() * Math.sqrt(temperature),
          ),
        );
        const step = await runFlow(state, [], { latent, noise });
        release(state, { latent });
        state = nextState(step);
        latent = step["latent_out"]!;

        const decoded = await mimi({ latent, ...codec });
        release(codec);
        codec = nextState(decoded);
        const audio = decoded["audio"];
        if (audio === undefined) {
          throw new Error('The mimi decoder produced no "audio" output');
        }
        yield audio.data as Float32Array;

        const eos = step["eos_logit"];
        if (countdown === null && eos !== undefined && Number(eos.data[0]) > phonon.eosThreshold) {
          // A short utterance gets a longer tail so the last word is not clipped.
          countdown = text.split(" ").length <= 4 ? 3 : 1;
        }
        if (countdown !== null && countdown-- === 0) {
          return;
        }
      }
    } finally {
      // Both the normal exit and the cancel path release state; the `return` above relies on this.
      release(state, codec, { latent });
    }
  };

  let disposed = false;
  return {
    profile,
    sampleRate: phonon.sampleRate,
    synthesize,
    dispose: async (): Promise<void> => {
      if (disposed) {
        return;
      }
      disposed = true;
      release(voiceState);
      voiceTensor.dispose();
      await Promise.all([flow.dispose(), mimi.dispose()]);
    },
  };
}
