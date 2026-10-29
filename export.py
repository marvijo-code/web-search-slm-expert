"""Export a checkpoint to the format runtime/marvijo-web-search.js reads.

python export.py --ckpt runs/v1/ckpt.pt --out runtime/weights

Writes:
- model.json : config, vocab, and the list of tensors (name, shape, offset)
- model.bin  : all tensors as float16, little endian, in that order
"""
import argparse
import json
import os

import numpy as np
import torch

from model import VOCAB


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ck = torch.load(a.ckpt, map_location="cpu")
    tensors, offset, blobs = [], 0, []
    for name, t in ck["model"].items():
        arr = t.float().numpy().astype("<f2")
        tensors.append({"name": name, "shape": list(arr.shape), "offset": offset, "size": int(arr.size)})
        offset += arr.size
        blobs.append(arr.ravel())
    with open(f"{a.out}/model.bin", "wb") as f:
        f.write(np.concatenate(blobs).tobytes())
    meta = {"config": ck["config"], "vocab": VOCAB, "dtype": "float16", "step": ck.get("step"), "tensors": tensors}
    with open(f"{a.out}/model.json", "w") as f:
        json.dump(meta, f)
    print(f"{offset:,} weights -> {os.path.getsize(f'{a.out}/model.bin') / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
