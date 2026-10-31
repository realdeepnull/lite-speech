import { loadPhonon, MODEL_ID, MODEL_REVISION } from "../lib/phonon.mjs";
import { measureSynthesis } from "../lib/run.mjs";

const TEXT = "Hello from lite-speech. This is the second sentence of the inference spike.";

const $ = (selector) => {
  const element = document.querySelector(selector);
  if (!element) throw new Error(`Missing element: ${selector}`);
  return element;
};

const status = $("[data-role=status]");
const log = $("[data-role=log]");
const play = $("[data-role=play]");
const run = $("[data-role=run]");
const picker = $("[data-role=config]");
const gpuInfo = $("[data-role=gpu]");
const textArea = $("[data-role=text]");

/** @type {any} */
let model = null;
/** @type {Float32Array | null} */
let pcm = null;
let sampleRate = 24000;

const setStatus = (message) => {
  status.textContent = message;
};

const append = (message) => {
  log.textContent += `${message}\n`;
  log.scrollTop = log.scrollHeight;
};

async function probeWebGPU() {
  if (!("gpu" in navigator)) {
    return { available: false, reason: "navigator.gpu missing" };
  }
  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) return { available: false, reason: "no adapter" };
    const info = adapter.info ?? {};
    return {
      available: true,
      vendor: info.vendor ?? null,
      architecture: info.architecture ?? null,
      description: info.description ?? null,
      shaderF16: adapter.features.has("shader-f16"),
    };
  } catch (error) {
    return { available: false, reason: String(error) };
  }
}

async function synthesize() {
  const [dtype, device] = picker.value.split("/");
  const text = textArea.value.trim() || TEXT;
  run.disabled = true;
  setStatus(`loading ${dtype} on ${device} …`);
  append(`--- ${dtype} / ${device} ---`);

  try {
    /** @type {Map<string, number>} */
    const downloads = new Map();
    const started = performance.now();
    model = await loadPhonon({
      dtype,
      device,
      onProgress: (info) => {
        if (info.status === "progress" && info.total && !downloads.has(info.file)) {
          downloads.set(info.file, info.total);
        }
      },
    });
    const loadMs = performance.now() - started;

    const result = await measureSynthesis(model, text);
    const warm = await measureSynthesis(model, text);
    pcm = warm.pcm;
    sampleRate = warm.sampleRate;
    play.disabled = false;

    append(
      JSON.stringify({
        runtime: "chromium",
        userAgent: navigator.userAgent,
        date: new Date().toISOString(),
        dtype,
        device,
        modelId: MODEL_ID,
        revision: MODEL_REVISION,
        tokenizerMode: model.tokenizerMode,
        loadMs,
        downloadBytes: [...downloads.values()].reduce((n, size) => n + size, 0),
        firstFrameMs: result.firstFrameMs,
        warmFirstFrameMs: warm.firstFrameMs,
        frameIntervalMs: warm.frameIntervalMs,
        synthMs: result.synthMs,
        warmSynthMs: warm.synthMs,
        audioSeconds: warm.audioSeconds,
        rtf: result.rtf,
        warmRtf: warm.rtf,
        frames: warm.frames,
        frameSizes: warm.frameSizes,
        sampleRate: warm.sampleRate,
        channels: warm.channels,
        range: [warm.min, warm.max],
        rms: warm.rms,
      }),
    );
    setStatus(`${dtype}/${device} ok`);
  } catch (error) {
    // An explicit device never falls back silently: the failure is the measurement.
    append(`FAILED ${dtype}/${device}: ${error?.name ?? "Error"}: ${error?.message ?? error}`);
    setStatus(`${dtype}/${device} failed`);
    console.error(error);
  } finally {
    run.disabled = false;
  }
}

play.addEventListener("click", async () => {
  if (!pcm) return;
  const context = new AudioContext({ sampleRate });
  const buffer = context.createBuffer(1, pcm.length, sampleRate);
  buffer.copyToChannel(pcm, 0);
  const source = context.createBufferSource();
  source.buffer = buffer;
  source.connect(context.destination);
  source.addEventListener("ended", () => context.close());
  source.start();
});

run.addEventListener("click", synthesize);

probeWebGPU().then((gpu) => {
  gpuInfo.textContent = JSON.stringify(gpu, null, 2);
  setStatus("ready");
  if (gpu.available && !gpu.shaderF16) {
    append("WebGPU lacks shader-f16: q4f16 will fail by design (no silent fallback).");
  }
  if (!gpu.available) {
    append("WebGPU unavailable: WASM is the only viable browser path here.");
  }
});
