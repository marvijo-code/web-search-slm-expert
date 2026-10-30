# Marvijo Web Search 1.1 - a tiny LLM built from scratch

`marvijo-web-search-1.1` is the model family. Four checkpoints were trained; the shipped one
is `marvijo-web-search-1.1-mix2-2.7m` (Hugging Face repo id
`marvijo/marvijo-web-search-1.1-mix2-2.7m`): 2.68M parameters, float16, runs in plain
JavaScript on a laptop CPU.

The model does one thing: it reads a request like `go to example.com and give me the data`
and writes a JSON plan (`{"url":"https://example.com","extract":["all"]}`), or in v4 one
tool call at a time, which a Puppeteer script runs to pull the page contents.

| File | What it does |
|---|---|
| `data/gen_data.py` | builds the synthetic training data and the held-out test sets |
| `data/handwritten_src.py` | hand-written test requests |
| `model.py` | the transformer (char tokens, RMSNorm, RoPE, SwiGLU, tied embeddings) |
| `train.py` | training loop with a wall-clock budget, logs to `train_log.jsonl` |
| `evaluate.py` | exact-match scoring plus a regex baseline |
| `export.py` | checkpoint to float16 `model.bin` + `model.json` |
| `run_lane.sh` | one full training lane (data, speed probe, train, eval, export) |
| `runtime/marvijo-web-search.js` | the same model in plain JavaScript (Node and browser) |
| `tests/` | JavaScript vs PyTorch parity check |

Background notes: `docs/research.md`. Results: `RESULTS.md`.

## Test it in a browser

```
cd web-search-slm-expert
node app/server.js                  # then open http://127.0.0.1:8770
```

Type a request, pick one of the four checkpoints, press Run. Every card is one real model
call plus the result the headless browser returned. Max calls is free text (1 to 200) or
tick Unlimited, which is on by default: the episode then runs until the model says done or
fail, or its 768 character context fills. "Collapse tool calls and responses" is also on by
default, so each call shows as one line (tool, its argument, ok/error, latency) and the
extracted output stays open; click a line to expand it. Both settings are remembered in the
browser. Under the episode JSON the same run is also shown as a rendered markdown report
with clickable links, which you can switch to its source or copy, and every url the run
touched is listed at the bottom in pages of ten, one link per row.

## Try the agent on your own machine

The v4 models write one tool call per turn (search, goto, extract, done, fail) and a real
browser runs each call. This needs internet access and a Chrome or Chromium install.

```
cd runtime
npm install
# Windows PowerShell, only if Chrome is not found automatically:
#   $env:CHROME_PATH = "C:\Program Files\Google\Chrome\Application\chrome.exe"
node agent4.js "what are the uefa fixtures today"                        # shipped checkpoint
node agent4.js --weights ../runs/v4-small/weights "images of cape town on wikipedia"
node agent4.js --all "get the headings from the polars docs"            # every checkpoint in turn
```

Each step prints the model's call and the page's one-line answer. The full result,
including the extracted data, is printed as JSON (or written with `--out file.json`).
