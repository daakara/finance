import { build } from "esbuild";
import fs from "node:fs";
import path from "node:path";

// Uses the candidate's production CSS. Output lives only in ignored local build artifacts.
const folder = path.resolve("out/__ux-fixture");
fs.mkdirSync(folder, { recursive: true });
await build({ entryPoints: ["scripts/post-audit-ux-fixture.tsx"], outfile: path.join(folder, "fixture.js"),
  bundle: true, platform: "browser", format: "iife", jsx: "automatic", define: {
    "process.env.NODE_ENV": '"production"', "process.env": '{"__NEXT_ROUTER_BASEPATH":""}', "process.browser": "true",
  } });
const index = fs.readFileSync("out/index.html", "utf8");
const styles = [...index.matchAll(/<link[^>]+rel="stylesheet"[^>]*>/g)].map(match => match[0]).join("");
fs.writeFileSync(path.join(folder, "index.html"), `<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">${styles}</head><body><div id="fixture-root"></div><script src="fixture.js"></script></body></html>`);
console.log("Local fixture generated: /__ux-fixture/ (synthetic data, no application route)");
