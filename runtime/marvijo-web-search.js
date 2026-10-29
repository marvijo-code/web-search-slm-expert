// MarvijoWebSearch inference in plain JavaScript. No libraries, no GPU.
// Runs in Node (agent.js) and in the browser (browser).
// Mirrors model.py exactly: char tokens, RMSNorm, RoPE, causal attention with a KV cache,
// SwiGLU feed-forward, tied output head, greedy decoding.
(function (root) {
  "use strict";
  const PAD = 0, BOS = 1, SEP = 2, EOS = 3;

  function f16ToF32(u16) {
    const out = new Float32Array(u16.length);
    for (let i = 0; i < u16.length; i++) {
      const h = u16[i];
      const s = h & 0x8000 ? -1 : 1;
      const e = (h >> 10) & 0x1f;
      const m = h & 0x3ff;
      if (e === 0) out[i] = s * Math.pow(2, -14) * (m / 1024);
      else if (e === 31) out[i] = m ? NaN : s * Infinity;
      else out[i] = s * Math.pow(2, e - 15) * (1 + m / 1024);
    }
    return out;
  }

  // y[o] = sum_i W[o, i] * x[i]   (W stored [out, in], row-major, like torch Linear)
  function matvec(W, x, out, nOut, nIn) {
    for (let o = 0; o < nOut; o++) {
      let s = 0;
      const base = o * nIn;
      for (let i = 0; i < nIn; i++) s += W[base + i] * x[i];
      out[o] = s;
    }
    return out;
  }

  function rmsnorm(x, w, eps, out) {
    let ss = 0;
    for (let i = 0; i < x.length; i++) ss += x[i] * x[i];
    const r = 1 / Math.sqrt(ss / x.length + eps);
    for (let i = 0; i < x.length; i++) out[i] = x[i] * r * w[i];
    return out;
  }

  class MarvijoWebSearch {
    constructor(meta, buffer) {
      this.cfg = meta.config;
      this.vocab = meta.vocab;
      this.stoi = new Map();
      this.vocab.forEach((c, i) => { if (i >= 4) this.stoi.set(c, i); });
      const all = f16ToF32(new Uint16Array(buffer));
      this.w = {};
      for (const t of meta.tensors) this.w[t.name] = all.subarray(t.offset, t.offset + t.size);
      const c = this.cfg;
      this.dh = c.d_model / c.n_head;
      const half = this.dh / 2;
      this.cos = new Float32Array(c.max_len * half);
      this.sin = new Float32Array(c.max_len * half);
      for (let p = 0; p < c.max_len; p++) {
        for (let j = 0; j < half; j++) {
          const inv = 1 / Math.pow(c.rope_theta, (2 * j) / this.dh);
          this.cos[p * half + j] = Math.cos(p * inv);
          this.sin[p * half + j] = Math.sin(p * inv);
        }
      }
      this.nParams = meta.tensors.reduce((a, t) => a + t.size, 0);
    }

    encode(s) {
      const q = this.stoi.get("?");
      return Array.from(s, (ch) => (this.stoi.has(ch) ? this.stoi.get(ch) : q));
    }

    rope(v, pos) { // in place, every head
      const half = this.dh / 2;
      for (let h = 0; h < this.cfg.n_head; h++) {
        const b = h * this.dh;
        for (let j = 0; j < half; j++) {
          const c = this.cos[pos * half + j], s = this.sin[pos * half + j];
          const x1 = v[b + j], x2 = v[b + half + j];
          v[b + j] = x1 * c - x2 * s;
          v[b + half + j] = x1 * s + x2 * c;
        }
      }
    }

    // One token through the network. Returns logits. Appends k/v to the cache.
    step(token, pos, cache, trace) {
      const c = this.cfg, D = c.d_model, F = c.d_ff, H = c.n_head, dh = this.dh;
      const x = new Float32Array(D);
      x.set(this.w["emb.weight"].subarray(token * D, token * D + D));
      const xn = new Float32Array(D), q = new Float32Array(D), att = new Float32Array(D), o = new Float32Array(D);
      const a = new Float32Array(F), b = new Float32Array(F);
      const scale = 1 / Math.sqrt(dh);
      for (let l = 0; l < c.n_layer; l++) {
        const p = `blocks.${l}.`;
        const L = cache[l];
        rmsnorm(x, this.w[p + "n1.weight"], c.norm_eps, xn);
        matvec(this.w[p + "attn.wq.weight"], xn, q, D, D);
        const k = matvec(this.w[p + "attn.wk.weight"], xn, new Float32Array(D), D, D);
        const v = matvec(this.w[p + "attn.wv.weight"], xn, new Float32Array(D), D, D);
        this.rope(q, pos);
        this.rope(k, pos);
        L.k.push(k);
        L.v.push(v);
        const T = L.k.length;
        const avg = trace && l === c.n_layer - 1 ? new Float32Array(T) : null;
        for (let h = 0; h < H; h++) {
          const hb = h * dh;
          const sc = new Float32Array(T);
          let mx = -Infinity;
          for (let t = 0; t < T; t++) {
            let s = 0;
            const kt = L.k[t];
            for (let d = 0; d < dh; d++) s += q[hb + d] * kt[hb + d];
            sc[t] = s * scale;
            if (sc[t] > mx) mx = sc[t];
          }
          let sum = 0;
          for (let t = 0; t < T; t++) { sc[t] = Math.exp(sc[t] - mx); sum += sc[t]; }
          for (let d = 0; d < dh; d++) att[hb + d] = 0;
          for (let t = 0; t < T; t++) {
            const wgt = sc[t] / sum;
            if (avg) avg[t] += wgt / H;
            const vt = L.v[t];
            for (let d = 0; d < dh; d++) att[hb + d] += wgt * vt[hb + d];
          }
        }
        if (avg) trace.attention = avg;
        matvec(this.w[p + "attn.wo.weight"], att, o, D, D);
        for (let i = 0; i < D; i++) x[i] += o[i];
        rmsnorm(x, this.w[p + "n2.weight"], c.norm_eps, xn);
        matvec(this.w[p + "ff.w1.weight"], xn, a, F, D);
        matvec(this.w[p + "ff.w3.weight"], xn, b, F, D);
        for (let i = 0; i < F; i++) a[i] = (a[i] / (1 + Math.exp(-a[i]))) * b[i]; // SiLU(a) * b
        matvec(this.w[p + "ff.w2.weight"], a, o, D, F);
        for (let i = 0; i < D; i++) x[i] += o[i];
      }
      rmsnorm(x, this.w["norm.weight"], c.norm_eps, xn);
      return matvec(this.w["emb.weight"], xn, new Float32Array(c.vocab_size), c.vocab_size, D);
    }

    // Greedy decode. opts.onStep({char, top, attention, prompt}) fires for every generated char.
    generate(request, opts = {}) {
      const maxNew = opts.maxNew || 200;
      const ids = [BOS, ...this.encode(request), SEP];
      const cache = Array.from({ length: this.cfg.n_layer }, () => ({ k: [], v: [] }));
      const tracing = !!(opts.onStep || opts.trace);
      let logits, trace = tracing ? {} : null;
      for (let i = 0; i < ids.length; i++) logits = this.step(ids[i], i, cache, i === ids.length - 1 ? trace : null);
      let out = "", pos = ids.length;
      const steps = [];
      for (let n = 0; n < maxNew && pos < this.cfg.max_len; n++) {
        let best = 0;
        for (let i = 1; i < logits.length; i++) if (logits[i] > logits[best]) best = i;
        let top = null;
        // attention[t] = how much the last layer looked at token t (BOS, request chars, SEP, output so far)
        const attention = trace && trace.attention;
        if (tracing) {
          let mx = -Infinity; for (const z of logits) if (z > mx) mx = z;
          let sum = 0; const pr = Array.from(logits, (z) => { const e = Math.exp(z - mx); sum += e; return e; });
          top = pr.map((p, i) => ({ id: i, p: p / sum }))
            .sort((u, v) => v.p - u.p).slice(0, 5)
            .map((t) => ({ char: t.id === EOS ? "<end>" : this.vocab[t.id], p: t.p }));
        }
        const info = { char: best === EOS ? "<end>" : this.vocab[best], top, attention, done: best === EOS };
        if (opts.trace) steps.push(info);
        if (opts.onStep) opts.onStep(info);
        if (best === EOS) break;
        out += this.vocab[best];
        trace = tracing ? {} : null;
        logits = this.step(best, pos, cache, trace);
        pos++;
      }
      return opts.trace ? { text: out, steps } : out;
    }
  }

  MarvijoWebSearch.TOKENS = { PAD, BOS, SEP, EOS };
  if (typeof module !== "undefined" && module.exports) module.exports = MarvijoWebSearch;
  else root.MarvijoWebSearch = MarvijoWebSearch;
})(typeof globalThis !== "undefined" ? globalThis : this);
