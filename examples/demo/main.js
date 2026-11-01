/**
 * lite-speech demo, bundler-free.
 *
 * Imports the *built* ESM output of both packages through an import map, so this page is a real
 * consumer test rather than a source-level demo: if the published `exports`, the worker subpath,
 * or the bundling policy were wrong, this page would fail.
 *
 * The page is also the smallest honest example of the M6 seams: the catalogue is built from
 * manifests (models, languages, voices), the language selector comes from that catalogue, and
 * nothing here assumes a 24 kHz English Phonon checkpoint.
 */
import { createTts, isTtsError } from "@lite-speech/core";
import {
  BUILT_IN_MODELS,
  BUILT_IN_VOICES,
  createOnnxEngine,
  createRegistry,
  detectWebGpu,
  languagesOf,
} from "@lite-speech/engine-onnx";

const $ = (selector) => {
  const element = document.querySelector(selector);
  if (!element) {
    throw new Error(`Missing element: ${selector}`);
  }
  return element;
};

const gpuInfo = $("[data-role=gpu]");
const deviceInfo = $("[data-role=device]");
const languagesInfo = $("[data-role=languages]");
const modelsInfo = $("[data-role=models]");
const voiceCount = $("[data-role=voice-count]");
const modelInfo = $("[data-role=model]");
const catalogueInfo = $("[data-role=catalogue]");
const languageSelect = $("[data-role=language]");
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
const verifyButton = $("[data-role=verify]");
const evictButton = $("[data-role=evict]");
const assetsInfo = $("[data-role=assets]");
const integrityInfo = $("[data-role=integrity]");

/**
 * The catalogue this page runs, built from manifests only.
 *
 * This is the whole extensibility story: registering another checkpoint, another language, or a
 * mirror means appending to `models`/`voices` here. Nothing in the engine, the worker protocol,
 * or `@lite-speech/core` changes. `BUILT_IN_*` is the shipped catalogue; a real application would
 * pass its own manifests instead (or alongside).
 */
const registry = createRegistry({ models: BUILT_IN_MODELS, voices: BUILT_IN_VOICES });

/** Every voice of the catalogue, each tagged with the model that ships it. */
const catalogueVoices = registry
  .models()
  .flatMap((model) => registry.voices(model).map((voice) => ({ ...voice, modelId: model.id })));

/** Engine is created per (language, voice, offline) combination; `Tts` owns it. */
let engine;
/** The active instance, so switching the policy does not re-download anything. */
let tts;
/** Device/dtype the last engine resolved, used to label the assets actually selected. */
let resolved;
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
  for (const button of [speak, stream, prepareButton, verifyButton, evictButton]) {
    button.disabled = value;
  }
  stopButton.disabled = !value || controller === undefined;
  languageSelect.disabled = value;
  voiceSelect.disabled = value;
};

/** Renders the device/dtype decision, including a documented fallback. */
const showResolved = (resolution) => {
  deviceInfo.textContent = resolution.fallbackFrom
    ? `${resolution.device} · ${resolution.dtype} (fell back from ${resolution.fallbackFrom})`
    : `${resolution.device} · ${resolution.dtype}`;
};

/** Renders the cache state, so "offline-capable" is a visible fact rather than a claim. */
const showPlan = (plan) => {
  const megabytes = (plan.cachedBytes / 1e6).toFixed(1);
  const total = (plan.totalBytes / 1e6).toFixed(1);
  // The dtype the engine resolved is the variant this plan priced; a manifest may list several.
  const variant = resolved ? ` · ${resolved.dtype}` : "";
  assetsInfo.textContent = plan.complete
    ? `complete: ${megabytes} MB in ${plan.files.length} files${variant}`
    : `missing: ${plan.missing.length} of ${plan.files.length} files (${total} MB total)${variant}`;
  integrityInfo.textContent =
    plan.integrity === "declared"
      ? `sha256 declared for all ${plan.files.length} files`
      : `sha256 declared for ${plan.files.length - plan.unverified.length} of ${plan.files.length} files`;
  const family = registry.models().find((model) => model.id === plan.modelId)?.family;
  modelInfo.textContent = `${plan.modelId}${family ? ` (${family})` : ""} — ${
    plan.complete ? `cached (${megabytes} MB)` : "not cached"
  }`;
};

/** Plays mono PCM through an `AudioContext`, which the library deliberately does not do. */
const playPcm = async (pcm, sampleRate) => {
  if (pcm.length === 0 || sampleRate <= 0) {
    return;
  }
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

/** Replaces the voice list with the voices that speak `language`. */
const populateVoices = (language) => {
  const previous = voiceSelect.value;
  const voices = catalogueVoices.filter((voice) => voice.language === language);
  voiceSelect.replaceChildren(
    ...voices.map((voice) => {
      const option = document.createElement("option");
      option.value = voice.id;
      option.textContent = voice.name ? `${voice.name} (${voice.id})` : voice.id;
      return option;
    }),
  );
  // Keep the selection when it still speaks the chosen language; otherwise the first voice wins,
  // which is what `createTts()` would choose anyway.
  if (voices.some((voice) => voice.id === previous)) {
    voiceSelect.value = previous;
  }
  voiceCount.textContent = String(voices.length);
};

/** Drops the active instances so a new language/voice/policy is actually used. */
const resetTts = async () => {
  if (!tts) {
    return;
  }
  await tts.dispose();
  tts = undefined;
  engine = undefined;
};

/** Creates (or recreates) the `Tts` for the selected language, voice, and network policy. */
const currentTts = async () => {
  const language = languageSelect.value || undefined;
  // A voice only speaks one language, so a stale selection would be rejected by `createTts()`.
  const selected = catalogueVoices.find((voice) => voice.id === voiceSelect.value);
  const voice = selected?.language === language ? selected.id : undefined;
  const offline = offlineToggle.checked ? "strict" : "allow";
  if (
    tts &&
    tts.request.language === language &&
    tts.request.voice === voice &&
    tts.offline === offline
  ) {
    return tts;
  }
  // A `Tts` owns the engine it was given, so any change needs a fresh engine: reusing one would
  // hand a disposed engine to the next instance.
  await resetTts();
  engine = await createOnnxEngine({ device: "auto", registry });
  resolved = engine.resolved;
  showResolved(resolved);
  tts = createTts({
    engine,
    offline,
    ...(language !== undefined && { language }),
    ...(voice !== undefined && { voice }),
  });
  const selection = `${tts.request.language}${tts.request.voice ? ` / ${tts.request.voice}` : ""}`;
  append(`using ${selection} on ${resolved.device}/${resolved.dtype}`);
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
      let sampleRate = 0;
      let samples = 0;
      for await (const chunk of active.stream(text, { signal: controller.signal })) {
        if (chunk.sampleRate > 0) {
          // Never assume a rate: another model in the catalogue may decode at 22.05 kHz.
          sampleRate = chunk.sampleRate;
        }
        if (chunk.pcm.length > 0 && firstChunkAt === undefined) {
          firstChunkAt = performance.now() - started;
        }
        chunks.push(chunk);
        samples += chunk.pcm.length;
        setStatus(
          sampleRate > 0
            ? `${chunks.length} chunks, ${(samples / sampleRate).toFixed(2)}s of audio`
            : `${chunks.length} chunks`,
        );
      }
      const pcm = concatenate(chunks);
      const elapsed = performance.now() - started;
      append(
        JSON.stringify({
          mode: "stream",
          chunks: chunks.length,
          firstChunkMs: Math.round(firstChunkAt ?? 0),
          totalMs: Math.round(elapsed),
          audioSeconds: Number((sampleRate > 0 ? pcm.length / sampleRate : 0).toFixed(2)),
          finalChunks: chunks.filter((chunk) => chunk.final).length,
          streamedSamples: pcm.length,
          sampleRate,
        }),
      );
      await playPcm(pcm, sampleRate);
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
          language: result.language,
          voice: result.voice,
        }),
      );
      await playPcm(result.pcm, result.sampleRate);
    }
    setStatus("done");
    if (resolved) {
      showPlan(await active.describeAssets());
    }
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
    const plan = await active.prepare({
      allowNetwork: true,
      verify: true,
      onProgress,
      signal: controller.signal,
    });
    showPlan(plan);
    setStatus("assets ready");
    append(`verified: ${JSON.stringify(await active.isCached())}`);
  } catch (error) {
    append(`error: ${error?.message ?? String(error)}`);
    setStatus("prepare failed");
  } finally {
    controller = undefined;
    busy(false);
    progress.hidden = true;
  }
});

verifyButton.addEventListener("click", async () => {
  busy(true);
  setStatus("verifying…");
  try {
    const active = await currentTts();
    const report = await active.verifyAssets();
    integrityInfo.textContent = `verified ${report.verified.length} files, ${
      report.unverified.length
    } without a declared digest`;
    append(`verify ${JSON.stringify(report)}`);
    setStatus(report.ok ? "cache verified" : "cache failed verification");
  } catch (error) {
    integrityInfo.textContent = "verification failed";
    append(
      isTtsError(error)
        ? `verify error ${error.code}: ${error.message}`
        : `verify error: ${error?.message ?? String(error)}`,
    );
    setStatus("verify failed");
  } finally {
    busy(false);
  }
});

evictButton.addEventListener("click", async () => {
  busy(true);
  try {
    const active = await currentTts();
    await active.removeModel();
    showPlan(await active.describeAssets());
    integrityInfo.textContent = "not checked";
    append(`cached: ${JSON.stringify(await active.isCached())}`);
    setStatus("cache cleared");
  } catch (error) {
    append(`error: ${error?.message ?? String(error)}`);
    setStatus("evict failed");
  } finally {
    busy(false);
  }
});

/** Renders the cache state of the current selection, creating the engine if needed. */
const refreshPlan = async () => {
  try {
    const active = await currentTts();
    showPlan(await active.describeAssets());
  } catch (error) {
    append(`plan failed: ${error?.message ?? String(error)}`);
  }
};

languageSelect.addEventListener("change", () => {
  populateVoices(languageSelect.value);
  // Force a new instance on the next run so the new model is actually used, and refresh the asset
  // panel so "the new voice is not cached yet" is visible before Speak is pressed.
  void resetTts().then(refreshPlan);
});

voiceSelect.addEventListener("change", () => {
  void resetTts().then(refreshPlan);
});

offlineToggle.addEventListener("change", () => {
  void resetTts().then(refreshPlan);
});

/** Fills the environment panel and the language selector from the catalogue. */
const showCatalogue = () => {
  catalogueInfo.textContent = registry
    .models()
    .map((model) => {
      const voices = registry.voices(model);
      const languages = languagesOf(model, voices);
      return [
        `${model.id}`,
        `  family: ${model.family ?? "-"} · languages: ${languages.join(", ") || "-"}`,
        `  ${voices.length} voices · license: ${model.license ?? "unknown"} · revision ${model.revision.slice(0, 7)}`,
      ].join("\n");
    })
    .join("\n\n");

  const languages = registry.languages();
  languagesInfo.textContent = languages.join(", ");
  modelsInfo.textContent = String(registry.models().length);
  languageSelect.replaceChildren(
    ...languages.map((language) => {
      const option = document.createElement("option");
      option.value = language;
      option.textContent = language;
      return option;
    }),
  );
  populateVoices(languages[0] ?? "");
};

void (async () => {
  try {
    showCatalogue();
    append(
      `catalogue: ${registry.models().length} models, ${catalogueVoices.length} voices, languages ${registry.languages().join(", ")}`,
    );

    const available = await detectWebGpu();
    gpuInfo.textContent = available ? "adapter available" : "no adapter (WASM path)";
    append(`webgpu: ${available}`);

    const active = await currentTts();
    const plan = await active.describeAssets();
    showPlan(plan);
    append(`cached: ${JSON.stringify(plan.complete)}`);
    setStatus("ready");
    busy(false);
  } catch (error) {
    append(`init failed: ${error?.message ?? String(error)}`);
    setStatus("init failed");
    console.error(error);
  }
})();
