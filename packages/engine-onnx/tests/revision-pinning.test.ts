/**
 * Revision-pinning proof, against Transformers.js' real resolution code.
 *
 * The defect being guarded against is not in this repository: Transformers.js'
 * `get_tokenizer_files()` calls `get_file_metadata()` **without** the caller's `revision`, so a
 * load pinned to a commit SHA still probes the Hub for `resolve/main/tokenizer_config.json`. In a
 * browser that probe is also uncacheable, which is why a fully provisioned profile could still
 * make a request under `offline: "strict"`.
 *
 * A fake would prove nothing here, so these tests drive the installed library and record what it
 * actually asks for. `env.fetch` is the single transport seam the library uses for Hub lookups.
 */
import { AutoTokenizer, env } from "@huggingface/transformers";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import { hubRevisionTemplate } from "../src/catalog.js";

/** The pinned commit the manifest uses. */
const REVISION = "4f0cda7c60dddb4862d0e68ff9ff60c32ffdc9f4";

const realFetch = env.fetch;
let requested: string[] = [];

/**
 * Records what the library asks for and fails every request, so the test stays offline and fast.
 *
 * Distinct model ids are used per case on purpose: `get_file_metadata()` memoizes by model id, so
 * reusing one would return the first case's answer without touching the transport at all.
 */
beforeEach(() => {
  requested = [];
  env.fetch = ((input: string | URL) => {
    requested.push(typeof input === "string" ? input : input.toString());
    return Promise.resolve(new Response(null, { status: 404 }));
  }) as typeof env.fetch;
});

afterEach(() => {
  env.fetch = realFetch;
  env.remotePathTemplate = "{model}/resolve/{revision}/";
});

/** Loads a tokenizer that cannot exist, and returns the URL the library probed for it. */
async function firstRequestFor(modelId: string, revision: string): Promise<string> {
  env.allowLocalModels = false;
  env.allowRemoteModels = true;
  // The tokenizer cannot load from a 404, which is fine: the request already happened.
  await expect(AutoTokenizer.from_pretrained(modelId, { revision })).rejects.toThrow();
  return requested[0] ?? "";
}

describe("the tokenizer probe", () => {
  it("targets `main` by default, which is the upstream behaviour the engine compensates for", async () => {
    const url = await firstRequestFor("lite-speech/probe-default", REVISION);

    expect(url).toContain("/lite-speech/probe-default/resolve/main/tokenizer_config.json");
    expect(url).not.toContain(REVISION);
  });

  it("targets the pinned revision once the template is pinned", async () => {
    env.remotePathTemplate = hubRevisionTemplate(REVISION);

    const url = await firstRequestFor("lite-speech/probe-pinned", REVISION);

    expect(url).toBe(
      `https://huggingface.co/lite-speech/probe-pinned/resolve/${REVISION}/tokenizer_config.json`,
    );
  });

  it("leaves every other lookup at the pinned revision too", async () => {
    env.remotePathTemplate = hubRevisionTemplate(REVISION);

    const url = await firstRequestFor("lite-speech/probe-consistent", REVISION);

    // The probe and the files it is asking about must agree, or the cache key and the manifest URL
    // refer to different entries and a complete cache still looks incomplete.
    expect(url).toContain(`/resolve/${REVISION}/`);
  });
});
