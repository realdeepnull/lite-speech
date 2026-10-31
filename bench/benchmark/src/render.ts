/**
 * Re-renders `bench/report/summary.md` from the committed JSON reports.
 *
 * Kept as its own entry point so the table can be refreshed without re-running a measurement — and
 * so a reviewer can regenerate it and see that it matches what is committed.
 *
 * Usage: node --experimental-strip-types src/render.ts
 */
import { join } from "node:path";
import { writeFile } from "node:fs/promises";

import { renderSummary } from "./report.js";

const reportDir = join(import.meta.dirname, "..", "..", "report");
const markdown = await renderSummary(reportDir);
await writeFile(join(reportDir, "summary.md"), markdown, "utf8");
console.log(markdown);
