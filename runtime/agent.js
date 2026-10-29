#!/usr/bin/env node
// MarvijoWebSearch agent: the model plans, Puppeteer does the browsing.
//
//   node agent.js "go to example.com and give me the data"
//   node agent.js --weights ../runs/small/weights --screenshot shot.png --out result.json "..."
//
// 1. MarvijoWebSearch (runtime/marvijo-web-search.js, plain JS on the CPU) turns the request into a JSON plan.
// 2. Puppeteer opens the url headless and extracts exactly the fields in the plan.
const fs = require("fs");
const path = require("path");
const MarvijoWebSearch = require("./marvijo-web-search.js");

const ALL = ["title", "meta", "headings", "text", "links", "images", "tables"];

function parseArgs(argv) {
  const a = { weights: path.join(__dirname, "weights"), out: null, screenshot: null, planOnly: false, request: [] };
  for (let i = 0; i < argv.length; i++) {
    const k = argv[i];
    if (k === "--weights") a.weights = argv[++i];
    else if (k === "--out") a.out = argv[++i];
    else if (k === "--screenshot") a.screenshot = argv[++i];
    else if (k === "--plan-only") a.planOnly = true;
    else if (k === "--manual-plan") a.manualPlan = argv[++i]; // test Puppeteer without the model
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

function findChrome() {
  const local = process.env.LOCALAPPDATA || "";
  const pw = path.join(local, "ms-playwright");
  const cands = [process.env.CHROME_PATH];
  if (fs.existsSync(pw)) {
    for (const d of fs.readdirSync(pw).filter((d) => d.startsWith("chromium_headless_shell")).sort().reverse()) {
      cands.push(path.join(pw, d, "chrome-headless-shell-win64", "chrome-headless-shell.exe"));
      cands.push(path.join(pw, d, "chrome-linux", "headless_shell"));
    }
  }
  cands.push(path.join(local, "Google", "Chrome", "Application", "chrome.exe"),
    "/usr/bin/chromium", "/usr/bin/chromium-browser", "/usr/bin/google-chrome");
  return cands.find((p) => p && fs.existsSync(p));
}

// Runs inside the page. Returns only the requested fields.
function extractInPage(fields) {
  const clean = (s) => (s || "").replace(/\s+/g, " ").trim();
  const want = new Set(fields);
  const out = {};
  if (want.has("title")) out.title = document.title;
  if (want.has("meta")) {
    out.meta = { lang: document.documentElement.lang || null };
    for (const m of document.querySelectorAll("meta[name], meta[property]")) {
      const k = m.getAttribute("name") || m.getAttribute("property");
      if (k && m.content) out.meta[k] = m.content;
    }
    const canon = document.querySelector("link[rel=canonical]");
    if (canon) out.meta.canonical = canon.href;
  }
  if (want.has("headings")) {
    out.headings = [...document.querySelectorAll("h1,h2,h3,h4,h5,h6")]
      .map((h) => ({ level: Number(h.tagName[1]), text: clean(h.innerText) })).filter((h) => h.text);
  }
  if (want.has("text")) {
    const root = document.querySelector("main, article") || document.body;
    out.text = [...root.querySelectorAll("p, li, blockquote, pre, td")].map((e) => clean(e.innerText)).filter(Boolean);
    if (!out.text.length) out.text = clean(root.innerText).split(/(?<=[.!?])\s+/).filter(Boolean);
  }
  if (want.has("links")) {
    const seen = new Set();
    out.links = [...document.querySelectorAll("a[href]")]
      .map((a) => ({ text: clean(a.innerText), href: a.href }))
      .filter((l) => l.href.startsWith("http") && !seen.has(l.href) && seen.add(l.href));
  }
  if (want.has("images")) {
    out.images = [...document.querySelectorAll("img")].map((i) => ({ src: i.currentSrc || i.src, alt: i.alt || "", width: i.naturalWidth, height: i.naturalHeight })).filter((i) => i.src);
  }
  if (want.has("tables")) {
    out.tables = [...document.querySelectorAll("table")].map((t) => ({
      caption: clean(t.caption && t.caption.innerText),
      rows: [...t.rows].map((r) => [...r.cells].map((c) => clean(c.innerText))),
    }));
  }
  return out;
}

async function main() {
  const a = parseArgs(process.argv.slice(2));
  if (!a.request) {
    console.error('usage: node agent.js [--weights dir] [--out file] [--screenshot file] "request"');
    process.exit(2);
  }
  let raw, planMs = 0;
  if (a.manualPlan) {
    raw = a.manualPlan;
    console.error("[marvijo-web-search] model skipped, manual plan: " + raw);
  } else {
    const model = loadModel(a.weights);
    const t0 = performance.now();
    raw = model.generate(a.request);
    planMs = performance.now() - t0;
    console.error(`[marvijo-web-search] ${(model.nParams / 1e6).toFixed(2)}M params, plan in ${planMs.toFixed(0)} ms on CPU`);
    console.error(`[marvijo-web-search] ${raw}`);
  }

  let plan;
  try { plan = JSON.parse(raw); } catch (e) {
    console.log(JSON.stringify({ request: a.request, plan_raw: raw, error: "model output is not valid JSON" }, null, 2));
    process.exit(1);
  }
  const result = { request: a.request, plan, plan_ms: Math.round(planMs) };
  if (plan.error || a.planOnly) {
    if (plan.error) result.message = plan.error === "no_url"
      ? "I scrape web pages, but your request has no website in it. Add a url or domain."
      : "I only do one thing: open a website and extract its contents.";
    console.log(JSON.stringify(result, null, 2));
    return;
  }

  const puppeteer = require("puppeteer-core");
  const executablePath = findChrome();
  if (!executablePath) throw new Error("No Chrome found. Set CHROME_PATH.");
  const fields = plan.extract.includes("all") ? ALL : plan.extract.filter((f) => ALL.includes(f));
  const browser = await puppeteer.launch({ executablePath, headless: true, args: ["--no-sandbox"] });
  try {
    const page = await browser.newPage();
    await page.setViewport({ width: 1280, height: 800 });
    const t1 = performance.now();
    const resp = await page.goto(plan.url, { waitUntil: "networkidle2", timeout: 45000 });
    result.status = resp ? resp.status() : null;
    result.final_url = page.url();
    result.data = await page.evaluate(extractInPage, fields);
    result.browse_ms = Math.round(performance.now() - t1);
    if (a.screenshot) await page.screenshot({ path: a.screenshot });
  } finally {
    await browser.close();
  }
  const json = JSON.stringify(result, null, 2);
  if (a.out) fs.writeFileSync(a.out, json);
  console.log(json);
}

main().catch((e) => { console.error(e.stack || String(e)); process.exit(1); });
