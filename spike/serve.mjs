/**
 * Zero-dependency static server for the browser half of the spike.
 * Serves the repository root so the page can import the installed ESM build from
 * `spike/node_modules` without a bundler.
 *
 * Usage: node serve.mjs [port]
 */
import { createReadStream, existsSync, statSync } from "node:fs";
import { createServer } from "node:http";
import { dirname, extname, join, normalize, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, "..");
const port = Number(process.argv[2] ?? 8080);

const types = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".wasm": "application/wasm",
  ".bin": "application/octet-stream",
  ".wav": "audio/wav",
};

createServer((request, response) => {
  const url = new URL(request.url ?? "/", `http://${request.headers.host}`);
  const requested = decodeURIComponent(url.pathname);
  let filePath = normalize(join(root, requested));

  if (!filePath.startsWith(root)) {
    response.writeHead(403).end("Forbidden");
    return;
  }
  if (existsSync(filePath) && statSync(filePath).isDirectory()) {
    filePath = join(filePath, "index.html");
  }
  if (!existsSync(filePath)) {
    response.writeHead(404).end(`Not found: ${requested}`);
    return;
  }

  response.writeHead(200, {
    "content-type": types[extname(filePath)] ?? "application/octet-stream",
    // The spike loads ONNX Runtime WASM builds; keep the same origin simple.
    "cross-origin-opener-policy": "same-origin",
    "cross-origin-embedder-policy": "require-corp",
  });
  createReadStream(filePath).pipe(response);
}).listen(port, () => {
  console.log(`spike server: http://localhost:${port}/spike/browser/`);
  console.log(`repo root:    ${root}`);
});
