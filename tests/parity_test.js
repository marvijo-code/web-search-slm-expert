// Compare runtime/marvijo-web-search.js against PyTorch logits dumped by tests/parity_dump.py
const fs = require("fs");
const path = require("path");
const MarvijoWebSearch = require("../runtime/marvijo-web-search.js");
const [dir, expectedFile] = process.argv.slice(2);
const meta = JSON.parse(fs.readFileSync(path.join(dir, "model.json"), "utf8"));
const buf = fs.readFileSync(path.join(dir, "model.bin"));
const m = new MarvijoWebSearch(meta, buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
let worst = 0, ok = true;
for (const e of JSON.parse(fs.readFileSync(expectedFile, "utf8"))) {
  const ids = [1, ...m.encode(e.prompt), 2];
  const cache = Array.from({ length: m.cfg.n_layer }, () => ({ k: [], v: [] }));
  let logits;
  ids.forEach((t, i) => { logits = m.step(t, i, cache, null); });
  const diff = Math.max(...e.logits.map((z, i) => Math.abs(z - logits[i])));
  worst = Math.max(worst, diff);
  const gen = m.generate(e.prompt);
  const same = gen === e.generated;
  if (!same) ok = false;
  console.log(`max |logit diff| ${diff.toExponential(2)}  same text: ${same}  js="${gen.slice(0, 70)}"`);
}
console.log(worst < 1e-3 && ok ? "PARITY OK" : "PARITY FAIL");
process.exit(worst < 1e-3 && ok ? 0 : 1);
