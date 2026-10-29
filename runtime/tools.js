// The Puppeteer tools MarvijoWebSearch v4 can call, and the one-line observations it reads back.
//
// Tool calls (the model writes these, one per turn):
//   {"tool":"search","q":"uefa fixtures today"}
//   {"tool":"goto","url":"https://en.wikipedia.org/wiki/Cape_Town"}
//   {"tool":"extract","fields":["images"]}
//   {"tool":"done"}
//   {"tool":"fail","reason":"not_found"}
//
// Observations (the runtime writes these back into the model's context):
//   results 1 https://a.com/x 2 https://b.org/y 3 https://c.net/z      | results none
//   ok 200 text:812 links:1534 images:143 tables:12                     | error 404 | error timeout | error dns | error blocked | error unreachable
//   got images:143
//
// data/gen_agent.py builds training episodes with exactly these strings: keep the two in step.
const fs = require("fs");
const path = require("path");

const ALL = ["title", "meta", "headings", "text", "links", "images", "tables"];
const COUNTED = ["text", "links", "images", "tables"];
const MAX_URL = 90; // longer search results are skipped: the model copies urls char by char

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

// Runs inside the page. opts = { fields, find, countsOnly, dates }.
// Text comes from a walk over every text node, shadow DOM included (sites such as uefa.com
// draw their match lists inside web components, which innerText does not see).
function inPage(opts) {
  const clean = (s) => (s || "").replace(/\s+/g, " ").trim();
  const SKIP = /cookie|consent|onetrust|didomi|gdpr|cmp-/i;
  const lines = [];
  const walk = (n) => {
    if (lines.length > 4000) return;
    if (n.nodeType === 3) {
      const t = clean(n.textContent);
      if (t && t !== lines[lines.length - 1]) lines.push(t);
      return;
    }
    if (n.nodeType !== 1 && n.nodeType !== 11) return;
    if (n.nodeType === 1) {
      if (/^(SCRIPT|STYLE|NOSCRIPT|TEMPLATE|SVG|IFRAME)$/i.test(n.tagName)) return;
      if (SKIP.test((n.id || "") + " " + (typeof n.className === "string" ? n.className : ""))) return;
      if (n.shadowRoot) walk(n.shadowRoot);
    }
    for (const c of n.childNodes) walk(c);
  };
  walk(document.body || document.documentElement);

  const hrefs = [...new Set([...document.querySelectorAll("a[href]")].map((a) => a.href).filter((h) => h.startsWith("http")))];
  const imgs = [...new Set([...document.querySelectorAll("img")].map((i) => i.currentSrc || i.src).filter((s) => s && s.startsWith("http")))];
  const tables = [...document.querySelectorAll("table")].filter((t) => t.rows.length);
  if (opts.countsOnly) {
    return {
      text: lines.length, links: hrefs.length, images: imgs.length, tables: tables.length,
      blocked: /captcha|are you a robot|access denied|unusual traffic|verify you are human/i.test(document.title + " " + lines.slice(0, 40).join(" ")),
    };
  }

  const want = new Set(opts.fields);
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
    out.text = lines;
    if (opts.find) {
      // A date word: keep the section that starts at that date, up to the next date heading.
      const day = opts.dates && opts.dates[opts.find];
      if (day) {
        const hit = (l) => day.some((d) => l.toLowerCase().includes(d.toLowerCase()));
        const heading = /\b(mon|tue|wed|thu|fri|sat|sun)[a-z]*\b.{0,4}\d{1,2}\b|\b\d{1,2} (jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]* \d{4}\b/i;
        const i = lines.findIndex(hit);
        let j = i + 1;
        while (i >= 0 && j < lines.length && j < i + 200 && !(heading.test(lines[j]) && !hit(lines[j]))) j++;
        out.text = i >= 0 ? lines.slice(i, j) : [];
      } else {
        const w = opts.find.toLowerCase();
        out.text = lines.filter((l) => l.toLowerCase().includes(w));
      }
    }
  }
  if (want.has("links")) {
    const seen = new Set();
    out.links = [...document.querySelectorAll("a[href]")]
      .map((a) => ({ text: clean(a.innerText), href: a.href }))
      .filter((l) => l.href.startsWith("http") && !seen.has(l.href) && seen.add(l.href));
  }
  if (want.has("images")) {
    const seen = new Set();
    out.images = [...document.querySelectorAll("img")]
      .map((i) => ({ src: i.currentSrc || i.src, alt: i.alt || "", width: i.naturalWidth, height: i.naturalHeight }))
      .filter((i) => i.src && i.src.startsWith("http") && !seen.has(i.src) && seen.add(i.src));
  }
  if (want.has("tables")) {
    out.tables = tables.map((t) => ({
      caption: clean(t.caption && t.caption.innerText),
      rows: [...t.rows].map((r) => [...r.cells].map((c) => clean(c.innerText))),
    }));
  }
  return out;
}

// The ways a page may write a date: "Thursday 22 October 2026", "October 22", "2026-10-22", "22/10/2026".
function dateForms(offsetDays) {
  const d = new Date(Date.now() + offsetDays * 86400000);
  const M = d.toLocaleString("en-US", { month: "long" }), m3 = M.slice(0, 3), D = d.getDate(), Y = d.getFullYear();
  const W = d.toLocaleString("en-US", { weekday: "long" });
  const mm = String(d.getMonth() + 1).padStart(2, "0"), dd = String(D).padStart(2, "0");
  return [`${D} ${M}`, `${M} ${D}`, `${D} ${m3} `, `${m3} ${D},`, `${Y}-${mm}-${dd}`, `${dd}/${mm}/${Y}`, `${W} ${D}`, `${W}, ${M} ${D}`];
}

function classifyError(e) {
  const m = String(e && e.message || e);
  if (/timeout/i.test(m)) return "timeout";
  if (/ERR_NAME_NOT_RESOLVED|ERR_NAME_RESOLUTION|ENOTFOUND/i.test(m)) return "dns";
  return "unreachable";
}

class Browser {
  async open() {
    const puppeteer = require("puppeteer-core");
    const executablePath = findChrome();
    if (!executablePath) throw new Error("No Chrome found. Set CHROME_PATH.");
    this.browser = await puppeteer.launch({ executablePath, headless: true, args: ["--no-sandbox", "--lang=en-US"] });
    this.page = await this.browser.newPage();
    await this.page.setViewport({ width: 1280, height: 800 });
    // The headless shell announces itself as "HeadlessChrome"; many sites block that string.
    const ua = (await this.browser.userAgent()).replace("HeadlessChrome", "Chrome");
    await this.page.setUserAgent(ua);
    await this.page.setExtraHTTPHeaders({ "Accept-Language": "en-US,en;q=0.9" });
    return this;
  }

  async close() { if (this.browser) await this.browser.close(); }

  async load(url, timeout = 30000) {
    const resp = await this.page.goto(url, { waitUntil: "networkidle2", timeout });
    return resp ? resp.status() : 0;
  }

  // ---- tools. Each returns { obs, ...details } and never throws for a page problem.
  async goto(url) {
    if (!/^https?:\/\/[^\s/]+\.[^\s]+/.test(url || "")) return { obs: "error bad_url" };
    let status;
    try { status = await this.load(url); } catch (e) {
      // a slow page may still have rendered enough to read
      if (classifyError(e) !== "timeout") return { obs: `error ${classifyError(e)}`, error: String(e.message || e) };
      status = -1;
    }
    let c;
    try { c = await this.page.evaluate(inPage, { countsOnly: true }); } catch (e) { return { obs: "error timeout" }; }
    if (status === -1 && c.text + c.links === 0) return { obs: "error timeout" };
    if (c.blocked || status === 403 || status === 429) return { obs: "error blocked", status, final_url: this.page.url() };
    if (status >= 400) return { obs: `error ${status}`, status, final_url: this.page.url() };
    const counts = COUNTED.map((k) => `${k}:${c[k]}`).join(" ");
    return { obs: `ok ${status === -1 ? 200 : status} ${counts}`, status, title: await this.page.title(), final_url: this.page.url(), counts: c };
  }

  async search(q) {
    const results = [];
    const engines = [
      ["duckduckgo", `https://html.duckduckgo.com/html/?q=${encodeURIComponent(q)}`, () => [...document.querySelectorAll("a.result__a")].map((a) => a.href)],
      ["bing", `https://www.bing.com/search?q=${encodeURIComponent(q)}&setlang=en`, () => [...document.querySelectorAll("li.b_algo h2 a")].map((a) => a.href)],
    ];
    let engine = null;
    for (const [name, url, pick] of engines) {
      try { await this.load(url, 20000); } catch (e) { continue; }
      const hrefs = await this.page.evaluate(pick).catch(() => []);
      for (let h of hrefs) {
        try {
          const u = new URL(h);
          if (u.hostname.endsWith("duckduckgo.com") && u.searchParams.get("uddg")) h = u.searchParams.get("uddg");
          else if (u.hostname.endsWith("duckduckgo.com")) continue; // ads go through y.js
          if (u.hostname.endsWith("bing.com") && u.searchParams.get("u")) {
            const b = u.searchParams.get("u").replace(/^a1/, "");
            h = Buffer.from(b.replace(/-/g, "+").replace(/_/g, "/"), "base64").toString("utf8");
          }
        } catch (e) { continue; }
        if (!/^https?:\/\//.test(h) || h.length > MAX_URL || /\.pdf($|\?)/i.test(h) || /[\s"\\]/.test(h)) continue;
        if (!results.includes(h)) results.push(h);
        if (results.length === 3) break;
      }
      if (results.length) { engine = name; break; }
    }
    const obs = results.length ? "results " + results.map((u, i) => `${i + 1} ${u}`).join(" ") : "results none";
    return { obs, results, engine };
  }

  async extract(fields, find) {
    const want = (fields || []).includes("all") ? ALL : (fields || []).filter((f) => ALL.includes(f));
    if (!want.length) return { obs: "error bad_fields" };
    const dates = { today: dateForms(0), tomorrow: dateForms(1), yesterday: dateForms(-1) };
    let data;
    try { data = await this.page.evaluate(inPage, { fields: want, find: find || null, dates }); } catch (e) { return { obs: "error no_page" }; }
    const got = want.filter((k) => Array.isArray(data[k])).map((k) => `${k}:${data[k].length}`);
    return { obs: "got " + (got.length ? got.join(" ") : want.join(" ")), data, final_url: this.page.url() };
  }
}

module.exports = { Browser, ALL, inPage, dateForms, findChrome };
