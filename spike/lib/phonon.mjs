import { AutoConfig, AutoTokenizer, PreTrainedModel, Tensor, env } from "@huggingface/transformers";

/** Pinned, published model repository and revision (M1 task: never `main`). */
export const MODEL_ID = "onnx-community/phonon-31m-en-ONNX";
export const MODEL_REVISION = "4f0cda7c60dddb4862d0e68ff9ff60c32ffdc9f4";
export const SAMPLE_RATE = 24000;

/** @param {number[]} dims @param {Float32Array} [data] */
const f32 = (dims, data) =>
  new Tensor("float32", data ?? new Float32Array(dims.reduce((a, b) => a * b, 1)), dims);

/** @param {number[]} values @param {number[]} dims */
const i64 = (values, dims) => new Tensor("int64", BigInt64Array.from(values, BigInt), dims);

/**
 * `present.*` output names that only feed back into a session. On WebGPU these must
 * stay on the GPU (`preferredOutputLocation`), otherwise every frame round-trips.
 * @param {{ num_layers: number }} cfg
 */
const cacheNames = (cfg) =>
  Array.from({ length: cfg.num_layers }, (_, i) => [
    `present.${i}.key`,
    `present.${i}.value`,
  ]).flat();

/** @param {{ num_layers: number, num_heads: number, head_dim: number }} cfg */
const emptyCache = (cfg) =>
  Object.fromEntries(
    Array.from({ length: cfg.num_layers }, (_, i) =>
      ["key", "value"].map((k) => [
        `past_key_values.${i}.${k}`,
        f32([1, cfg.num_heads, 0, cfg.head_dim]),
      ]),
    ).flat(),
  );

/**
 * Maps every `present*` output to the `past*` input name of the next call.
 * @param {Record<string, any>} outputs
 */
function nextState(outputs) {
  return Object.fromEntries(
    Object.entries(outputs)
      .filter(([name]) => name.startsWith("present"))
      .map(([name, value]) => [
        name.replace("present.", "past_key_values.").replace("present_", "past_"),
        value,
      ]),
  );
}

/** GPU tensors are not garbage collected; free them once they are replaced. */
const release = (...states) =>
  states.forEach((state) =>
    Object.values(state).forEach((t) => {
      if (t?.location === "gpu-buffer") t.dispose();
    }),
  );

/**
 * Reference workaround from the model card: patches the private
 * `tokenizer._tokenizer.model` Unigram instance. Kept only to prove that the public
 * implementation below produces the same ids, so the M1 exit gate can be met without
 * private fields. Never use this outside the spike.
 *
 * @param {any} tokenizer
 * @returns {"native" | "patched" | "unsupported"}
 */
export function patchByteFallback(tokenizer) {
  const unigram = tokenizer._tokenizer?.model;
  if (!unigram?.tokens_to_ids?.has || typeof unigram.encode !== "function") {
    return "unsupported";
  }
  if (unigram.byte_fallback) return "native";

  const encodePieces = unigram.encode.bind(unigram);
  unigram.encode = (pieces) =>
    encodePieces(pieces).flatMap((piece) => {
      if (unigram.tokens_to_ids.has(piece)) return [piece];
      const bytes = Array.from(
        new TextEncoder().encode(piece),
        (b) => `<0x${b.toString(16).toUpperCase().padStart(2, "0")}>`,
      );
      return bytes.every((b) => unigram.tokens_to_ids.has(b)) ? bytes : [piece];
    });
  return "patched";
}

const byteTokens = (piece) =>
  Array.from(
    new TextEncoder().encode(piece),
    (b) => `<0x${b.toString(16).toUpperCase().padStart(2, "0")}>`,
  );

/**
 * Builds a byte-fallback encoder from public tokenizer members only
 * (`tokenize`, `get_vocab`, `convert_tokens_to_ids`).
 *
 * The pinned release does not apply Unigram byte fallback, so a piece outside the
 * vocabulary would otherwise collapse into `<unk>`. Every piece produced by `tokenize()`
 * is therefore either kept or expanded into its UTF-8 `<0xXX>` tokens, which the
 * vocabulary contains for all 256 byte values.
 *
 * @param {any} tokenizer
 * @returns {(text: string) => number[]}
 */
export function createPublicEncoder(tokenizer) {
  const vocab = tokenizer.get_vocab();
  const toId = (/** @type {string} */ token) =>
    /** @type {number} */ (tokenizer.convert_tokens_to_ids(token));

  return (text) => {
    /** @type {number[]} */
    const ids = [];
    for (const piece of tokenizer.tokenize(text)) {
      if (vocab.has(piece)) {
        ids.push(toId(piece));
        continue;
      }
      const tokens = byteTokens(piece);
      ids.push(...(tokens.every((t) => vocab.has(t)) ? tokens.map(toId) : [toId(piece)]));
    }
    return ids;
  };
}

/**
 * Loads the two Phonon graphs and the tokenizer through public APIs only.
 *
 * @param {{
 *   dtype?: "fp32" | "q4" | "q4f16",
 *   device?: "cpu" | "wasm" | "webgpu" | "auto",
 *   revision?: string,
 *   cacheDir?: string | null,
 *   byteFallback?: boolean,
 *   onProgress?: (info: any) => void,
 * }} [options]
 */
export async function loadPhonon({
  dtype = "fp32",
  device = "cpu",
  revision = MODEL_REVISION,
  cacheDir = null,
  byteFallback = false,
  onProgress,
} = {}) {
  if (cacheDir) env.cacheDir = cacheDir;
  const progress_callback = onProgress ? (info) => onProgress(info) : undefined;

  const config = await AutoConfig.from_pretrained(MODEL_ID, {
    revision,
    progress_callback,
  });
  const tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID, {
    revision,
    progress_callback,
  });
  const cfg = config.phonon;

  const onGpu = (/** @type {string[]} */ names) =>
    device === "webgpu"
      ? {
          session_options: {
            preferredOutputLocation: Object.fromEntries(names.map((n) => [n, "gpu-buffer"])),
          },
        }
      : {};

  const flow = await PreTrainedModel.from_pretrained(MODEL_ID, {
    revision,
    config,
    model_file_name: "flow_lm",
    dtype,
    device,
    progress_callback,
    ...onGpu(["latent_out", ...cacheNames(cfg.flow_lm)]),
  });

  const mimi = await PreTrainedModel.from_pretrained(MODEL_ID, {
    revision,
    config,
    model_file_name: "mimi_decoder",
    dtype,
    device,
    progress_callback,
    ...onGpu([
      ...cacheNames(cfg.mimi_decoder),
      ...Object.keys(cfg.mimi_decoder.conv_states).map((n) => n.replace("past_", "present_")),
    ]),
  });

  const tokenizerMode = byteFallback ? patchByteFallback(tokenizer) : "public";
  const encode = byteFallback
    ? (/** @type {string} */ text) => tokenizer.encode(text, { add_special_tokens: false })
    : createPublicEncoder(tokenizer);

  const voiceUrl = `https://huggingface.co/${MODEL_ID}/resolve/${revision}/voices/default.bin`;
  const voiceBytes = await fetch(voiceUrl).then((r) => {
    if (!r.ok) throw new Error(`Voice download failed: ${r.status} ${voiceUrl}`);
    return r.arrayBuffer();
  });
  const voiceData = new Float32Array(voiceBytes);
  const voice = f32([1, cfg.voice_dim, voiceData.length / cfg.voice_dim], voiceData);

  const runFlow = (past, inputs = {}) => {
    const {
      text = [],
      voiceLatents = f32([1, cfg.voice_dim, 0]),
      latent = f32([1, 0, cfg.latent_dim]),
      noise = f32([1, cfg.latent_dim]),
    } = inputs;
    return flow({
      text_ids: i64(text, [1, text.length]),
      voice_latents: voiceLatents,
      latent,
      noise,
      num_speakers: i64([1], [1]),
      duration_delta: f32([1]),
      ...past,
    });
  };

  /** Voice prompt is computed once and reused; every sentence starts from this state. */
  const voiceState = nextState(await runFlow(emptyCache(cfg.flow_lm), { voiceLatents: voice }));

  /**
   * Yields one `Float32Array` of 1920 samples (80 ms at 24 kHz) per latent frame.
   * One call handles about one sentence (at most ~50 tokens).
   *
   * @param {string} sentence
   * @param {{ temperature?: number }} [options]
   */
  async function* speak(sentence, { temperature = cfg.temperature } = {}) {
    let text = sentence.trim().replace(/\s+/g, " ");
    text = text[0].toUpperCase() + text.slice(1) + (/[\p{L}\p{N}]$/u.test(text) ? "." : "");

    const ids = encode(text);
    let state = nextState(await runFlow(voiceState, { text: ids }));

    let codec = { past_offset: i64([0], [1]), ...emptyCache(cfg.mimi_decoder) };
    for (const [name, dims] of Object.entries(cfg.mimi_decoder.conv_states)) {
      codec[name] = f32(/** @type {number[]} */ (dims));
    }

    // NaN marks the first frame, which starts from an empty decoder state.
    let latent = f32([1, 1, cfg.latent_dim], new Float32Array(cfg.latent_dim).fill(NaN));
    const randn = () =>
      Math.sqrt(-2 * Math.log(1 - Math.random())) * Math.cos(2 * Math.PI * Math.random());

    /** @type {number | null} */
    let countdown = null;
    const maxFrames = Math.ceil((ids.length / 3 + 2) * cfg.frame_rate);

    try {
      for (let frame = 0; frame < maxFrames; frame++) {
        const noise = f32(
          [1, cfg.latent_dim],
          Float32Array.from({ length: cfg.latent_dim }, () => randn() * Math.sqrt(temperature)),
        );
        const step = await runFlow(state, { latent, noise });
        release(state, { latent });
        state = nextState(step);
        latent = step.latent_out;

        const decoded = await mimi({ latent, ...codec });
        release(codec);
        codec = nextState(decoded);
        yield decoded.audio.data;

        if (countdown === null && step.eos_logit.data[0] > cfg.eos_threshold) {
          countdown = text.split(" ").length <= 4 ? 3 : 1;
        }
        if (countdown !== null && countdown-- === 0) break;
      }
    } finally {
      release(state, codec, { latent });
    }
  }

  return {
    config,
    cfg,
    tokenizer,
    encode,
    tokenizerMode,
    revision,
    speak,
    dispose: () => Promise.all([flow.dispose(), mimi.dispose()]),
  };
}
