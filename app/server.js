#!/usr/bin/env node
// Marvijo Web Search 1.1 test app: type a request in the browser, the real agent runs it.
//
//   node app/server.js                 then open http://127.0.0.1:8770
//   node app/server.js --port 9000 --host 127.0.0.1
//
// The model and the tools are the same code the CLI uses (runtime/agent4.js,
// runtime/tools.js); this file only adds an HTTP surface and streams each step.
const http = require("http");
const fs = require("fs");
const path = require("path");
const { loadModel, runEpisode, RUNS } = require("../runtime/agent4.js");
const { Browser } = require("../runtime/tools.js");

const HERE = __dirname;
const MAX_REQUEST_CHARS = 300;
const DEFAULT_STEPS = 6;
// Model naming follows the Hugging Face style used for the published checkpoints:
// marvijo-web-search-1.1-<recipe>-<size>, repo id marvijo/<that name>.
const MODEL_FAMILY = "marvijo-web-search-1.1";
const MODEL_SIZE = "2.7m";
function modelName(lane) {
  const recipe = lane.replace(/^v4-/, "");
  const tag = recipe === "small" ? "" : `-${recipe}`;
  return `${MODEL_FAMILY}${tag}-${MODEL_SIZE}`;
}
// "Unlimited" means: run until the model says done/fail or its 768 char context fills.
// The context guard ends every episode, this ceiling only stops a pathological loop.
const SAFETY_MAX_STEPS = 200;

function models() {
  if (!fs.existsSync(RUNS)) return [];
  return fs.readdirSync(RUNS)
    .filter((d) => d.startsWith("v4-") && fs.existsSync(path.join(RUNS, d, "weights", "model.bin")))
    .sort()
    .map((d) => {
      const meta = JSON.parse(fs.readFileSync(path.join(RUNS, d, "weights", "model.json"), "utf8"));
      const c = meta.config || {};
      // tied embeddings + per-layer attention, norms and SwiGLU feed-forward
      const params = c.d_model
        ? c.vocab_size * c.d_model + c.n_layer * (4 * c.d_model * c.d_model + 2 * c.d_model + 3 * c.d_ff * c.d_model) + c.d_model
        : null;
      const status = fs.existsSync(path.join(RUNS, d, "STATUS"))
        ? fs.readFileSync(path.join(RUNS, d, "STATUS"), "utf8").trim().split("\n")[0] : "";
      return { lane: d, model: modelName(d), dir: path.join(RUNS, d, "weights"), meta, params, status };
    });
}

function send(res, code, body, type = "application/json") {
  res.writeHead(code, { "content-type": type, "cache-control": "no-store" });
  res.end(body);
}

function serveStatic(res, file) {
  const full = path.join(HERE, file);
  if (!full.startsWith(HERE) || !fs.existsSync(full)) return send(res, 404, "not found", "text/plain");
  const type = full.endsWith(".html") ? "text/html; charset=utf-8"
    : full.endsWith(".js") ? "text/javascript; charset=utf-8" : "text/plain; charset=utf-8";
  send(res, 200, fs.readFileSync(full), type);
}

async function main() {
  const argv = process.argv.slice(2);
  let host = "127.0.0.1", port = 8770;
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] === "--port") port = Number(argv[++i]);
    else if (argv[i] === "--host") host = argv[++i];
  }

  const list = models();
  if (!list.length) {
    console.error("no runs/v4-*/weights found. Train or pull a checkpoint first.");
    process.exit(1);
  }
  const cache = new Map();
  const model = (name) => {
    const m = list.find((x) => x.lane === name) || list.find((x) => x.lane === "v4-mix2") || list[0];
    if (!cache.has(m.lane)) {
      const t0 = Date.now();
      cache.set(m.lane, loadModel(m.dir));
      console.log(`loaded ${m.model} in ${Date.now() - t0} ms`);
    }
    return { entry: m, model: cache.get(m.lane) };
  };
  console.log("preloading " + (list.find((x) => x.lane === "v4-mix2") || list[0]).model);
  model("v4-mix2");

  const browser = await new Browser().open();
  const server = http.createServer(async (req, res) => {
    const url = new URL(req.url, `http://${req.headers.host}`);
    if (url.pathname === "/") return serveStatic(res, "index.html");
    if (url.pathname === "/api/models") return send(res, 200, JSON.stringify(list.map((m) => ({
      lane: m.lane, name: m.model, params: m.params, status: m.status,
      vocab: m.meta && m.meta.config ? m.meta.config.vocab_size : null,
      max_len: m.meta && m.meta.config ? m.meta.config.max_len : null,
    }))));
    if (url.pathname !== "/api/run") return send(res, 404, JSON.stringify({ error: "not found" }));

    const request = (url.searchParams.get("request") || "").trim();
    const rawSteps = (url.searchParams.get("steps") || "").trim();
    const unlimited = rawSteps === "unlimited" || rawSteps === "0";
    let steps = DEFAULT_STEPS;
    if (unlimited) {
      steps = SAFETY_MAX_STEPS;
    } else if (rawSteps) {
      steps = Number(rawSteps);
      if (!Number.isInteger(steps) || steps < 1 || steps > SAFETY_MAX_STEPS) {
        return send(res, 400, JSON.stringify({ error: `steps must be a whole number from 1 to ${SAFETY_MAX_STEPS}, or unlimited` }));
      }
    }
    if (!request) return send(res, 400, JSON.stringify({ error: "empty request" }));
    if (request.length > MAX_REQUEST_CHARS) {
      return send(res, 400, JSON.stringify({ error: `request is longer than ${MAX_REQUEST_CHARS} characters` }));
    }
    const picked = model(url.searchParams.get("model") || "");

    res.writeHead(200, {
      "content-type": "text/event-stream; charset=utf-8",
      "cache-control": "no-store",
      connection: "keep-alive",
    });
    let closed = false;
    req.on("close", () => { closed = true; });
    const emit = (e) => { if (!closed) res.write(`data: ${JSON.stringify(e)}\n\n`); };

    emit({ type: "start", model: picked.entry.model, lane: picked.entry.lane, request, steps, unlimited });
    try {
      const r = await runEpisode(picked.model, request, browser, steps, emit);
      emit({ type: "done", ...r });
    } catch (e) {
      emit({ type: "error", message: String((e && e.message) || e) });
    } finally {
      if (!closed) res.end();
    }
  });

  server.listen(port, host, () => {
    console.log(`Marvijo Web Search 1.1 app on http://${host}:${port}  (${list.length} checkpoints: ${list.map((m) => m.model).join(", ")})`);
  });
  const bye = async () => { try { await browser.close(); } catch (e) { /* ignore */ } process.exit(0); };
  process.on("SIGINT", bye);
  process.on("SIGTERM", bye);
}

main().catch((e) => { console.error(e.stack || String(e)); process.exit(1); });
