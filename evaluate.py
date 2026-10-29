"""Score MarvijoWebSearch (and a hand-written regex baseline) on every test set.

python evaluate.py --ckpt runs/v1/ckpt.pt --data data/build --out runs/v1/eval.json

Metrics per test set:
- exact_match : the whole JSON plan is right (url AND fields AND error type)
- json_valid  : the output parses as JSON
- url_match   : the url is right (rows that have a url)
- fields_match: the extract list is right (rows that have a url)
"""
import argparse
import json
import re
import time


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def score(preds, rows):
    n = len(rows)
    em = valid = url_ok = fields_ok = url_rows = 0
    fails = []
    for p, r in zip(preds, rows):
        gold = json.loads(r["output"])
        try:
            got = json.loads(p)
            valid += 1
        except Exception:
            got = None
        if got == gold:
            em += 1
        elif len(fails) < 25:
            fails.append({"input": r["input"], "expected": r["output"], "got": p})
        if "url" in gold:
            url_rows += 1
            if isinstance(got, dict) and got.get("url") == gold["url"]:
                url_ok += 1
            if isinstance(got, dict) and got.get("extract") == gold["extract"]:
                fields_ok += 1
    return {"n": n, "exact_match": round(em / n, 4), "json_valid": round(valid / n, 4),
            "url_match": round(url_ok / max(1, url_rows), 4),
            "fields_match": round(fields_ok / max(1, url_rows), 4), "failures": fails}


def agent_score(model, rows, max_fails=25):
    """v4: replay each gold episode. At every step the model sees the request and the GOLD
    calls and tool results so far (teacher forcing) and must write the next call.
    - step_acc     : calls that match exactly (as parsed JSON)
    - episode_exact: episodes where every call matches
    - first_call   : the first call matches (the decision v3 made in one shot)
    - json_valid   : calls that parse as JSON
    """
    from model import BOS, SEP, EOS, encode
    was = model.training
    model.eval()
    steps = ok = ep_ok = first_ok = valid = 0
    fails = []
    for r in rows:
        prefix = [BOS] + encode(r["input"]) + [SEP]
        all_ok = True
        for k, s in enumerate(r["steps"]):
            got = model.generate_ids(prefix, 160)
            try:
                good = json.loads(got) == json.loads(s["call"])
                valid += 1
            except Exception:
                good = False
            steps += 1
            ok += good
            if k == 0:
                first_ok += good
            if not good:
                all_ok = False
                if len(fails) < max_fails:
                    fails.append({"input": r["input"], "step": k + 1, "expected": s["call"], "got": got,
                                  "after": r["steps"][k - 1]["obs"] if k else None})
            prefix += encode(s["call"]) + [EOS]
            if s.get("obs") is not None:
                prefix += encode(s["obs"]) + [SEP]
        ep_ok += all_ok
    model.train(was)
    n = len(rows)
    return {"n": n, "steps": steps, "step_acc": round(ok / steps, 4), "episode_exact": round(ep_ok / n, 4),
            "first_call": round(first_ok / n, 4), "json_valid": round(valid / steps, 4), "failures": fails}


def exact_match(model, rows):
    if rows and "steps" in rows[0]:  # v4 agent episodes: train.py logs episode_exact as em_<set>
        s = agent_score(model, rows, 0)
        return {"exact_match": s["episode_exact"], "step_acc": s["step_acc"]}
    was = model.training
    model.eval()
    preds = [model.generate(r["input"]) for r in rows]
    model.train(was)
    return score(preds, rows)


# ---------------------------------------------------------------- regex baseline
FIELD_KEYS = {
    "title": ["title", "page name", "name of the page"],
    "meta": ["meta", "seo", "description"],
    "headings": ["heading", "header", "h1", "section title", "subheading"],
    "text": ["text", "paragraph", "copy", "wording", "written content"],
    "links": ["link", "url", "href", "anchor"],
    "images": ["image", "picture", "photo", "img", "pics"],
    "tables": ["table", "tabular"],
}
URL_RE = re.compile(r"(https?://)?((?:[a-z0-9-]+\.)+[a-z]{2,})(/[^\s\"'<>)]*)?", re.I)


def regex_baseline(text):
    """What a careful developer writes in 15 minutes without any model."""
    low = text.lower()
    m = None
    for cand in URL_RE.finditer(text):
        if "@" in text[max(0, cand.start() - 1):cand.start()]:
            continue  # part of an email address
        m = cand
        break
    if not m:
        if any(w in low for w in ["scrape", "extract", "site", "page", "url", "website", "crawl", "fetch"]):
            return json.dumps({"error": "no_url"}, separators=(",", ":"))
        return json.dumps({"error": "out_of_scope"}, separators=(",", ":"))
    scheme = (m.group(1) or "https://").lower()
    path = (m.group(3) or "").rstrip(".,!?;:")
    url = scheme + m.group(2).lower() + path
    rest = low.replace(m.group(0).lower(), " ")
    fields = [f for f, keys in FIELD_KEYS.items() if any(k in rest for k in keys)]
    if "url" in rest and "links" in fields and not re.search(r"\burls\b", rest):
        fields.remove("links")
    return json.dumps({"url": url, "extract": fields or ["all"]}, separators=(",", ":"))


def main():
    import torch
    from model import Config, MarvijoWebSearch

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="data/build")
    ap.add_argument("--handwritten", default="data/handwritten_eval.jsonl")
    ap.add_argument("--independent", default="data/independent_eval.jsonl")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    ck = torch.load(a.ckpt, map_location="cpu")
    model = MarvijoWebSearch(Config(**ck["config"]))
    model.load_state_dict(ck["model"])
    model.eval()

    val = load_jsonl(f"{a.data}/val.jsonl")
    if val and "steps" in val[0]:  # v4 agent episodes: step-by-step replay, no regex baseline
        report = {"ckpt_step": ck.get("step"), "params": model.n_params(), "sets": {}}
        for name, rows in {"val": val[:500],
                           "unseen topics": load_jsonl(f"{a.data}/test_unseen_topics.jsonl"),
                           "unseen phrasings": load_jsonl(f"{a.data}/test_unseen_phrasings.jsonl")}.items():
            t = time.time()
            s = agent_score(model, rows)
            s["ms_per_step"] = round(1000 * (time.time() - t) / s["steps"], 2)
            report["sets"][name] = s
            print(f"{name:20s} episode EM {s['episode_exact']:.4f}  step acc {s['step_acc']:.4f}  "
                  f"first call {s['first_call']:.4f}  json {s['json_valid']:.4f}  {s['ms_per_step']} ms/step", flush=True)
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        return

    sets = {
        "val (seen phrasings, new sentences)": load_jsonl(f"{a.data}/val.jsonl")[:500],
        "unseen domains": load_jsonl(f"{a.data}/test_unseen_domains.jsonl"),
        "unseen phrasings": load_jsonl(f"{a.data}/test_unseen_phrasings.jsonl"),
        "handwritten": load_jsonl(a.handwritten),
        "independent dev (written by another agent)": load_jsonl(a.independent),
    }
    report = {"ckpt_step": ck.get("step"), "params": model.n_params(), "sets": {}}
    for name, rows in sets.items():
        t = time.time()
        preds = [model.generate(r["input"]) for r in rows]
        dt = time.time() - t
        s = score(preds, rows)
        s["ms_per_request"] = round(1000 * dt / len(rows), 2)
        b = score([regex_baseline(r["input"]) for r in rows], rows)
        report["sets"][name] = {"marvijo_web_search": s, "regex_baseline": {k: v for k, v in b.items() if k != "failures"},
                                "baseline_failures": b["failures"][:10]}
        print(f"{name:40s} marvijo-web-search EM {s['exact_match']:.4f}  baseline EM {b['exact_match']:.4f}  {s['ms_per_request']} ms/req", flush=True)
    for key in ["handwritten", "independent dev (written by another agent)"]:
        report[key + " outputs"] = [{"input": r["input"], "expected": r["output"], "got": model.generate(r["input"])}
                                    for r in sets[key]]
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
