"""Dump PyTorch logits for a few prompts so runtime/parity_test.js can compare."""
import json, sys
sys.path.insert(0, ".")
import torch
from model import Config, MarvijoWebSearch, build_sequence
ck = torch.load(sys.argv[1], map_location="cpu")
m = MarvijoWebSearch(Config(**ck["config"])); m.load_state_dict(ck["model"]); m.eval()
# match the fp16 export: round weights to half precision first
with torch.no_grad():
    for p in m.parameters(): p.copy_(p.half().float())
prompts = ["go to example.com and give me the data", "scrape github.com for links", "tell me a joke"]
out = []
for p in prompts:
    ids = build_sequence(p)
    logits = m(torch.tensor([ids]))[0, -1].tolist()
    out.append({"prompt": p, "logits": logits, "generated": m.generate(p)})
json.dump(out, open(sys.argv[2], "w"))
