/**
 * Zero-dependency static server for the demo.
 *
 * Serves the repository root so `examples/demo/index.html` can import the built `dist` output of
 * both packages without a bundler — which is the point of the demo: it consumes what would be
 * published, not the sources.
 *
 * Usage: `node examples/demo/serve.mjs [port]` from the repository root.
 */
import { createReadStream, existsSync, statSync } from "node:fs";
import { createServer } from "node:http";
import { dirname, extname, join, normalize, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, "..", "..");
const port = Number(process.argv[2] ?? 8080);

const types = new Map([
  [".html", "text/html; charset=utf-8"],
  [".js", "text/javascript; charset=utf-8"],
  [".mjs", "text/javascript; charset=utf-8"],
  [".json", "application/json; charset=utf-8"],
  [".map", "application/json; charset=utf-8"],
  [".wasm", "application/wasm"],
]);

createServer((request, response) => {
  const url = new URL(request.url ?? "/", `http://${request.headers.host}`);
  let filePath = normalize(join(root, decodeURIComponent(url.pathname)));

  if (!filePath.startsWith(root)) {
    response.writeHead(403).end("Forbidden");
    return;
  }
  if (existsSync(filePath) && statSync(filePath).isDirectory()) {
    filePath = join(filePath, "index.html");
  }
  if (!existsSync(filePath)) {
    // Deliberately a 404, never `index.html`: an SPA fallback would answer a missing worker
    // script with HTML and status 200, which the worker loader would reject with a syntax error.
    response.writeHead(404).end(`Not found: ${url.pathname}`);
    return;
  }

  response.writeHead(200, {
    "content-type": types.get(extname(filePath)) ?? "application/octet-stream",
    "cache-control": "no-cache",
  });
  createReadStream(filePath).pipe(response);
}).listen(port, () => {
  console.log(`demo:      http://localhost:${port}/examples/demo/`);
  console.log(`repo root: ${root}`);
});
