/**
 * lite-speech demo, bundler-free.
 *
 * Imports the *built* ESM output of both packages through an import map, so this page is a real
 * consumer test rather than a source-level demo: if the published `exports`, the worker subpath,
 * or the bundling policy were wrong, this page would fail.
 */
import { createTts, isTtsError } from "@lite-speech/core";
import { createOnnxEngine, detectWebGpu } from "@lite-speech/engine-onnx";

const $ = (selector) => {
  const element = document.querySelector(selector);
  if (!element) {
    throw new Error(`Missing element: ${selector}`);
  }
  return element;
};

const gpuInfo = $("[data-role=gpu]");
const voiceCount = $("[data-role=voice-count]");
const modelInfo = $("[data-role=model]");
const voiceSelect = $("[data-role=voice]");
const textArea = $("[data-role=text]");
const status = $("[data-role=status]");
const progress = $("[data-role=progress]");
const log = $("[data-role=log]");
const offlineToggle = $("[data-role=offline]");
const speak = $("[data-role=speak]");
const stream = $("[data-role=stream]");
const stopButton = $("[data-role=stop]");
const prepareButton = $("[data-role=prepare]");
const evictButton = $("[data-role=evict]");

/** Engine is created once; `Tts` instances are created per (voice, offline) combination. */
let engine;
/** The active instance, so switching the policy does not re-download anything. */
let tts;
let controller;
let audioContext;
let source;

const setStatus = (message) => {
  status.textContent = message;
};

const append = (message) => {
  log.textContent += `${message}\n`;
  log.scrollTop = log.scrollHeight;
};

const busy = (value) => {
  for (const button of [speak, stream, prepareButton, evictButton]) {
    button.disabled = value;
  }
  stopButton.disabled = !value || controller === undefined;
};

/** Plays mono PCM through an `AudioContext`, which the library deliberately does not do. */
const playPcm = async (pcm, sampleRate) => {
  audioContext ??= new AudioContext({ sampleRate });
  source?.stop();
  const buffer = audioContext.createBuffer(1, pcm.length, sampleRate);
  buffer.copyToChannel(pcm, 0);
  const next = audioContext.createBufferSource();
  next.buffer = buffer;
  next.connect(audioContext.destination);
  next.start();
  source = next;
};

/** Concatenates streamed chunks; a player would enqueue each chunk instead. */
const concatenate = (chunks) => {
  const total = chunks.reduce((sum, chunk) => sum + chunk.pcm.length, 0);
  const pcm = new Float32Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    pcm.set(chunk.pcm, offset);
    offset += chunk.pcm.length;
  }
  return pcm;
};

/** Creates (or recreates) the `Tts` for the selected voice and network policy. */
const currentTts = async () => {
  const voice = voiceSelect.value || undefined;
  const offline = offlineToggle.checked ? "strict" : "allow";
  if (tts && tts.request.voice === voice && tts.offline === offline) {
    return tts;
  }
  // A `Tts` owns the engine it was given, so a policy or voice change needs a fresh engine:
  // reusing one would hand a disposed engine to the next instance.
  if (tts) {
    await tts.dispose();
    tts = undefined;
  }
  engine = await createOnnxEngine({ device: "auto" });
  tts = createTts({ engine, offline, ...(voice !== undefined && { voice }) });
  return tts;
};

const onProgress = (event) => {
  progress.hidden = false;
  progress.value = Number.isFinite(event.progress) ? event.progress : 0;
  if (event.loaded > 0 && Number.isFinite(event.progress)) {
    setStatus(`${event.phase}: ${Math.round(event.progress * 100)}%`);
  }
};

async function runSynthesis(streamed) {
  controller = new AbortController();
  busy(true);
  try {
    const active = await currentTts();
    const text = textArea.value;
    const started = performance.now();

    if (streamed) {
      const chunks = [];
      let firstChunkAt;
      for await (const chunk of active.stream(text, { signal: controller.signal })) {
        if (chunk.pcm.length > 0 && firstChunkAt === undefined) {
          firstChunkAt = performance.now() - started;
        }
        chunks.push(chunk);
        setStatus(`${chunks.length} chunks, ${(chunk.sequence * 80) / 1000}s of audio`);
      }
      const pcm = concatenate(chunks);
      const elapsed = performance.now() - started;
      append(
        JSON.stringify({
          mode: "stream",
          chunks: chunks.length,
          firstChunkMs: Math.round(firstChunkAt ?? 0),
          totalMs: Math.round(elapsed),
          audioSeconds: Number((pcm.length / 24_000).toFixed(2)),
          finalChunks: chunks.filter((chunk) => chunk.final).length,
          streamedSamples: pcm.length,
        }),
      );
      if (pcm.length > 0) {
        await playPcm(pcm, 24_000);
      }
    } else {
      const result = await active.synthesize(text, { signal: controller.signal });
      const elapsed = performance.now() - started;
      append(
        JSON.stringify({
          mode: "full",
          totalMs: Math.round(elapsed),
          audioSeconds: Number((result.durationMs / 1000).toFixed(2)),
          samples: result.pcm.length,
          sampleRate: result.sampleRate,
        }),
      );
      await playPcm(result.pcm, result.sampleRate);
    }
    setStatus("done");
  } catch (error) {
    if (isTtsError(error)) {
      append(`error ${error.code}: ${error.message}`);
      setStatus(error.code);
    } else {
      append(`error: ${error?.message ?? String(error)}`);
      setStatus("error");
      console.error(error);
    }
  } finally {
    controller = undefined;
    busy(false);
    progress.hidden = true;
  }
}

speak.addEventListener("click", () => {
  void runSynthesis(false);
});

stream.addEventListener("click", () => {
  void runSynthesis(true);
});

stopButton.addEventListener("click", () => {
  controller?.abort();
  source?.stop();
  setStatus("stopped");
});

prepareButton.addEventListener("click", async () => {
  controller = new AbortController();
  busy(true);
  try {
    const active = await currentTts();
    const estimate = await active.estimateDownload();
    append(`missing ${(estimate.bytes / 1e6).toFixed(1)} MB in ${estimate.files.length} files`);
    await active.prepare({
      allowNetwork: true,
      onProgress,
      signal: controller.signal,
    });
    setStatus("assets ready");
    append(`cached: ${JSON.stringify(await active.isCached())}`);
  } catch (error) {
    append(`error: ${error?.message ?? String(error)}`);
    setStatus("prepare failed");
  } finally {
    controller = undefined;
    busy(false);
    progress.hidden = true;
  }
});

evictButton.addEventListener("click", async () => {
  busy(true);
  try {
    const active = await currentTts();
    await active.removeModel();
    append(`cached: ${JSON.stringify(await active.isCached())}`);
    setStatus("cache cleared");
  } catch (error) {
    append(`error: ${error?.message ?? String(error)}`);
    setStatus("evict failed");
  } finally {
    busy(false);
  }
});

voiceSelect.addEventListener("change", () => {
  // Force a new instance on the next run so the new voice is actually used.
  if (tts) {
    void tts.dispose();
    tts = undefined;
  }
});

void (async () => {
  try {
    const available = await detectWebGpu();
    gpuInfo.textContent = available ? "adapter available" : "no adapter (WASM path)";
    append(`webgpu: ${available}`);

    // A probe-only engine, so the voice list can be shown before anything is downloaded.
    // It is disposed immediately: `createTts()` owns the engine it receives.
    const probe = await createOnnxEngine({ device: "auto" });
    const voices = probe.voices();
    voiceCount.textContent = String(voices.length);
    voiceSelect.replaceChildren(
      ...voices.map((voice) => {
        const option = document.createElement("option");
        option.value = voice.id;
        option.textContent = voice.name ? `${voice.name} (${voice.id})` : voice.id;
        return option;
      }),
    );
    append(`resolved: ${JSON.stringify(probe.resolved)}`);
    await probe.dispose();

    const active = await currentTts();
    const cached = await active.isCached();
    append(`cached: ${JSON.stringify(cached)}`);
    modelInfo.textContent = cached ? "cached" : "not cached";
    setStatus("ready");
    busy(false);
  } catch (error) {
    append(`init failed: ${error?.message ?? String(error)}`);
    setStatus("init failed");
    console.error(error);
  }
})();
