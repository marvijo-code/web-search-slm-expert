// Score MarvijoWebSearch with the shipped JavaScript runtime (CPU, no PyTorch).
//   node eval.js --weights ../runs/v2-small/weights --set ../data/final_blind_test.jsonl [--out report.json]
// The regex baseline is scored in Python (evaluate.py); this file scores the model only.
const fs = require("fs");
const path = require("path");
const MarvijoWebSearch = require("./marvijo-web-search.js");

function arg(name, def) {
  const i = process.argv.indexOf(name);
  return i > 0 ? process.argv[i + 1] : def;
}
const weights = arg("--weights", path.join(__dirname, "weights"));
const sets = process.argv.flatMap((v, i) => (process.argv[i - 1] === "--set" ? [v] : []));
const outFile = arg("--out", null);

const meta = JSON.parse(fs.readFileSync(path.join(weights, "model.json"), "utf8"));
const buf = fs.readFileSync(path.join(weights, "model.bin"));
const model = new MarvijoWebSearch(meta, buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));

const report = { weights, params: model.nParams, sets: {} };
for (const file of sets) {
  const rows = fs.readFileSync(file, "utf8").split("\n").filter((l) => l.trim()).map((l) => JSON.parse(l));
  let em = 0, valid = 0, urlOk = 0, fieldsOk = 0, urlRows = 0;
  const outputs = [];
  const t0 = performance.now();
  for (const r of rows) {
    const got = model.generate(r.input);
    const gold = JSON.parse(r.output);
    let parsed = null;
    try { parsed = JSON.parse(got); valid++; } catch (e) { /* invalid JSON */ }
    const ok = parsed !== null && JSON.stringify(parsed) === JSON.stringify(gold);
    if (ok) em++;
    if (gold.url) {
      urlRows++;
      if (parsed && parsed.url === gold.url) urlOk++;
      if (parsed && JSON.stringify(parsed.extract) === JSON.stringify(gold.extract)) fieldsOk++;
    }
    outputs.push({ input: r.input, expected: r.output, got, ok });
  }
  const ms = (performance.now() - t0) / rows.length;
  const s = {
    n: rows.length, exact_match: +(em / rows.length).toFixed(4), json_valid: +(valid / rows.length).toFixed(4),
    url_match: +(urlOk / Math.max(1, urlRows)).toFixed(4), fields_match: +(fieldsOk / Math.max(1, urlRows)).toFixed(4),
    ms_per_request: +ms.toFixed(1), outputs,
  };
  report.sets[path.basename(file)] = s;
  console.log(`${path.basename(file).padEnd(32)} EM ${s.exact_match}  json ${s.json_valid}  url ${s.url_match}  fields ${s.fields_match}  ${s.ms_per_request} ms/req`);
}
if (outFile) fs.writeFileSync(outFile, JSON.stringify(report, null, 2));
