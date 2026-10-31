/**
 * Records what the high-level `text-to-speech` pipeline does with the Phonon repository.
 * The M1 exit gate needs the observed failure, not an assumption about it.
 */
import { pipeline } from "@huggingface/transformers";

import { MODEL_ID, MODEL_REVISION } from "./lib/phonon.mjs";

const started = performance.now();
try {
  const tts = await pipeline("text-to-speech", MODEL_ID, {
    revision: MODEL_REVISION,
    dtype: "q4",
    device: "cpu",
  });
  const output = await tts("Hello from the pipeline path.");
  const pcm = output?.audio?.data ?? output?.audio;
  console.log(
    JSON.stringify(
      {
        ok: true,
        ms: performance.now() - started,
        sampleRate: output?.sampling_rate ?? null,
        samples: pcm?.length ?? null,
        shape: pcm?.constructor?.name ?? null,
      },
      null,
      2,
    ),
  );
} catch (error) {
  console.log(
    JSON.stringify(
      {
        ok: false,
        ms: performance.now() - started,
        name: error?.name ?? null,
        message: error?.message ?? String(error),
        stack: (error?.stack ?? "").split("\n").slice(0, 6),
      },
      null,
      2,
    ),
  );
  process.exitCode = 1;
}
