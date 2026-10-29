#!/usr/bin/env node
// MarvijoWebSearch v4 agent: the model writes one tool call per turn, Puppeteer runs it (tools.js),
// and the one-line observation goes back into the model's context.
//
//   node agent4.js "what are the uefa fixtures today"
//   node agent4.js --weights ../runs/v4-mix2/weights --out result.json "get the images on wikipedia for cape town"
//   node agent4.js --all "..."          run the same request with every runs/v4-*/weights model
//
// Episode format (as in data/gen_agent.py): <bos>request<sep>call<eos>obs<sep>call<eos>...
const fs = require("fs");
const path = require("path");
const MarvijoWebSearch = require("./marvijo-web-search.js");
const { Browser } = require("./tools.js");

const { BOS, SEP, EOS } = MarvijoWebSearch.TOKENS;
const RUNS = path.join(__dirname, "..", "runs");

function parseArgs(argv) {
  const a = { weights: path.join(RUNS, "v4-mix2", "weights"), out: null, all: false, maxSteps: 8, request: [] };
  for (let i = 0; i < argv.length; i++) {
    const k = argv[i];
    if (k === "--weights") a.weights = argv[++i];
    else if (k === "--out") a.out = argv[++i];
    else if (k === "--all") a.all = true;
    else if (k === "--max-steps") a.maxSteps = Number(argv[++i]);
    else a.request.push(k);
  }
  a.request = a.request.join(" ");
  return a;
}

function loadModel(dir) {
  const meta = JSON.parse(fs.readFileSync(path.join(dir, "model.json"), "utf8"));
  const buf = fs.readFileSync(path.join(dir, "model.bin"));
  return new MarvijoWebSearch(meta, buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
}

// Feeds ids through the model, keeping one KV cache for the whole episode.
class Episode {
  constructor(model) {
    this.m = model;
    this.cache = Array.from({ length: model.cfg.n_layer }, () => ({ k: [], v: [] }));
    this.pos = 0;
    this.logits = null;
  }
  full() { return this.pos >= this.m.cfg.max_len; }
  feed(ids) { for (const t of ids) { if (this.full()) return; this.logits = this.m.step(t, this.pos++, this.cache); } }
  // Greedy decode one call. The closing <eos> is fed too, so the next observation follows it.
  call(maxNew = 160) {
    let out = "";
    for (let n = 0; n < maxNew && !this.full(); n++) {
      let best = 0;
      for (let i = 1; i < this.logits.length; i++) if (this.logits[i] > this.logits[best]) best = i;
      this.feed([best]);
      if (best === EOS) break;
      out += this.m.vocab[best];
    }
    return out;
  }
}

async function runTool(browser, call) {
  switch (call.tool) {
    case "search": return browser.search(call.q || "");
    case "goto": return browser.goto(call.url);
    case "extract": return browser.extract(call.fields, call.find);
    default: return { obs: "error bad_tool" };
  }
}

async function runEpisode(model, request, browser, maxSteps, onEvent) {
  const emit = onEvent || ((e) => {
    if (e.type === "call") console.error(`  ${e.call}`);
    else console.error(`    -> ${e.obs}`);
  });
  const ep = new Episode(model);
  ep.feed([BOS, ...model.encode(request), SEP]);
  const steps = [];
  let data = null, ended = "max_steps";
  const t0 = performance.now();
  for (let k = 0; k < maxSteps; k++) {
    if (ep.full()) { ended = "context_full"; break; }
    const t1 = performance.now();
    const raw = ep.call();
    const step = { call: raw, model_ms: Math.round(performance.now() - t1) };
    steps.push(step);
    emit({ type: "call", index: k, call: raw, model_ms: step.model_ms });
    let call;
    try { call = JSON.parse(raw); } catch (e) { ended = "invalid_json"; break; }
    if (call.tool === "done") { ended = "done"; break; }
    if (call.tool === "fail") { ended = "fail " + (call.reason || ""); break; }
    const r = await runTool(browser, call);
    step.obs = r.obs;
    if (r.final_url) step.url = r.final_url;
    if (r.data) data = r.data;
    emit({ type: "obs", index: k, obs: r.obs, url: r.final_url, data: r.data });
    ep.feed([...model.encode(r.obs), SEP]);
  }
  return { request, ended, steps, total_ms: Math.round(performance.now() - t0), data };
}

async function main() {
  const a = parseArgs(process.argv.slice(2));
  if (!a.request) {
    console.error('usage: node agent4.js [--weights dir | --all] [--out file] [--max-steps n] "request"');
    process.exit(2);
  }
  const dirs = a.all
    ? fs.readdirSync(RUNS).filter((d) => d.startsWith("v4-") && fs.existsSync(path.join(RUNS, d, "weights", "model.bin")))
      .sort().map((d) => path.join(RUNS, d, "weights"))
    : [a.weights];
  const browser = await new Browser().open();
  const results = [];
  try {
    for (const dir of dirs) {
      const model = loadModel(dir);
      const name = path.basename(path.dirname(dir));
      console.error(`[${name}] ${(model.nParams / 1e6).toFixed(2)}M params`);
      const r = await runEpisode(model, a.request, browser, a.maxSteps);
      console.error(`[${name}] ${r.ended} after ${r.steps.length} calls, ${r.total_ms} ms\n`);
      results.push({ model: name, ...r });
    }
  } finally {
    await browser.close();
  }
  const json = JSON.stringify(a.all ? results : results[0], null, 2);
  if (a.out) fs.writeFileSync(a.out, json);
  console.log(json);
}

if (require.main === module) {
  main().catch((e) => { console.error(e.stack || String(e)); process.exit(1); });
}

module.exports = { Episode, loadModel, runEpisode, parseArgs, RUNS };
