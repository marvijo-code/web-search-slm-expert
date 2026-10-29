"""python results_table.py - write RESULTS.md from the real run artifacts (runs/<lane>/js_eval.json, train_log.jsonl) plus the regex baseline."""
import json, os, sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import load_jsonl, regex_baseline, score
SETS = [("handwritten_eval.jsonl", "Hand-written (50)"), ("independent_eval.jsonl", "Independent dev (120)"),
        ("final_blind_test.jsonl", "Final blind (150)")]
lanes = [l for l in ["small", "v2-small", "v2-wide", "v3-small", "v3-long"] if os.path.exists(f"runs/{l}/js_eval.json")]
rows = ["| Model | Data | Params | Train min | " + " | ".join(s[1] for s in SETS) + " | Blind ms/request |",
        "|---|---|---|---|" + "---|" * len(SETS) + "---|"]
base = [score([regex_baseline(r["input"]) for r in load_jsonl(f"data/{k}")], load_jsonl(f"data/{k}"))["exact_match"] for k, _ in SETS]
rows.append("| Regex baseline | - | - | - | " + " | ".join(f"{100*b:.1f}%" for b in base) + " | - |")
for l in lanes:
    js = json.load(open(f"runs/{l}/js_eval.json"))["sets"]
    st = json.loads(open(f"runs/{l}/train_log.jsonl").readline())
    data = l.split("-")[0] if l.startswith(("v2-", "v3-")) else "v1"
    rows.append(f"| {l} | {data} | {st['params']/1e6:.2f}M | {st['args']['minutes']:.0f} | "
                + " | ".join(f"{100*js[k]['exact_match']:.1f}%" for k, _ in SETS)
                + f" | {js['final_blind_test.jsonl']['ms_per_request']:.0f} |")
md = ["# MarvijoWebSearch results", "",
      "Exact-match plans (URL and field list both right), scored by the JavaScript runtime on the laptop CPU.",
      "The shipped model is chosen on the independent dev set only. The blind test was committed before any v2 result and is reported, never tuned on.", "",
      *rows, ""]
open("RESULTS.md", "w", encoding="utf-8", newline="\n").write("\n".join(md))
print("\n".join(rows))
