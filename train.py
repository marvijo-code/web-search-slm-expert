"""Train MarvijoWebSearch from random weights.

python train.py --data data/build --out runs/v1 --minutes 60

The run has a wall-clock budget: it measures its own speed over the first steps, then
fixes the total step count so the cosine schedule ends right at the budget.
Every log line goes to <out>/train_log.jsonl (the docs page draws its charts from it).

v4 rows hold whole tool-calling episodes ({"input", "steps"}); loss is only on the calls.
--resume continues from <out>/last.pt (saved every few minutes), so a killed cloud worker
loses minutes, not the run.
"""
import argparse
import json
import math
import os
import random
import time

import torch
import torch.nn.functional as F

from model import Config, MarvijoWebSearch, build_sequence, build_episode, PAD, SEP
from evaluate import exact_match, load_jsonl


def make_batches(rows, batch, rng):
    """Shuffle, then group rows of similar length so little compute goes to padding."""
    rows = rows[:]
    rng.shuffle(rows)
    chunk = batch * 50
    out = []
    for i in range(0, len(rows), chunk):
        part = sorted(rows[i:i + chunk], key=lambda r: len(r[0]))
        out += [part[j:j + batch] for j in range(0, len(part), batch)]
    rng.shuffle(out)
    return out


def to_item(r):
    """(token ids, target flags). v3: the answer after <sep>. v4: every call in the episode."""
    if "steps" in r:
        return build_episode(r["input"], r["steps"])
    ids = build_sequence(r["input"], r["output"])
    sep = ids.index(SEP)
    return ids, [i > sep for i in range(len(ids))]


def to_tensors(items):
    T = max(len(ids) for ids, _ in items)
    x = torch.full((len(items), T - 1), PAD, dtype=torch.long)
    y = torch.full((len(items), T - 1), -100, dtype=torch.long)  # -100 = no loss
    for i, (ids, tgt) in enumerate(items):
        t = torch.tensor(ids)
        x[i, :len(ids) - 1] = t[:-1]
        m = torch.tensor(tgt[1:])
        y[i, :len(ids) - 1] = torch.where(m, t[1:], torch.full_like(t[1:], -100))
    return x, y


@torch.no_grad()
def val_loss(model, seqs, batch=128):
    model.eval()
    tot, n, correct = 0.0, 0, 0
    for i in range(0, len(seqs), batch):
        x, y = to_tensors(seqs[i:i + batch])
        logits = model(x)
        m = y != -100
        tot += F.cross_entropy(logits[m], y[m], reduction="sum").item()
        correct += (logits[m].argmax(-1) == y[m]).sum().item()
        n += m.sum().item()
    model.train()
    return tot / n, correct / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/build")
    ap.add_argument("--out", default="runs/v1")
    ap.add_argument("--minutes", type=float, default=60)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--min-lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--n-layer", type=int, default=6)
    ap.add_argument("--d-model", type=int, default=192)
    ap.add_argument("--n-head", type=int, default=6)
    ap.add_argument("--d-ff", type=int, default=512)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--bf16", action="store_true", help="bfloat16 autocast (fast on CPUs with AMX)")
    ap.add_argument("--probe", type=int, default=0, help="run N steps, print seconds per step, exit")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--resume", action="store_true", help="continue from <out>/last.pt if it exists")
    ap.add_argument("--save-every", type=float, default=4.0, help="minutes between last.pt saves")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    torch.set_num_threads(a.threads or os.cpu_count() or 4)

    train = [to_item(r) for r in load_jsonl(f"{a.data}/train.jsonl")]
    val_rows = load_jsonl(f"{a.data}/val.jsonl")
    val = [to_item(r) for r in val_rows][:1000]
    agent = "steps" in val_rows[0]
    names = ["unseen_topics", "unseen_phrasings"] if agent else ["unseen_domains", "unseen_phrasings"]
    tests = {n: load_jsonl(f"{a.data}/test_{n}.jsonl") for n in names}

    cfg = Config(n_layer=a.n_layer, d_model=a.d_model, n_head=a.n_head, d_ff=a.d_ff, max_len=a.max_len)
    assert max(len(ids) for ids, _ in train) <= cfg.max_len
    model = MarvijoWebSearch(cfg)
    n_params = model.n_params()
    print(f"params {n_params:,}  train rows {len(train):,}  threads {torch.get_num_threads()}", flush=True)

    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=a.lr, betas=(0.9, 0.95))

    # The schedule runs on TRAINING time (eval time excluded): cosine decay ends when
    # the training clock reaches the budget.
    budget = a.minutes * 60
    step, epoch, tokens_seen = 0, 0, 0
    t0 = time.time()
    train_time = 0.0
    best = float("inf")
    batches = iter(())
    last = f"{a.out}/last.pt"
    resumed = a.resume and os.path.exists(last)
    if resumed:
        ck = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step, epoch, tokens_seen, train_time, best = ck["step"], ck["epoch"], ck["tokens"], ck["train_time"], ck["best"]
        rng.setstate(ck["rng"])
        torch.set_rng_state(ck["torch_rng"])
        t0 -= ck["elapsed"]
        # ckpt.pt may hold a better eval than last.pt knew about (saved after the last save)
        if os.path.exists(f"{a.out}/ckpt.pt"):
            best = min(best, torch.load(f"{a.out}/ckpt.pt", map_location="cpu").get("val_loss", best))
        print(f"resumed at step {step}, train_t {train_time / 60:.1f}m", flush=True)
    last_save = time.time()

    log = open(f"{a.out}/train_log.jsonl", "a" if resumed else "w", buffering=1)
    meta = {"event": "resume" if resumed else "start", "params": n_params, "config": cfg.to_dict(),
            "args": vars(a), "train_rows": len(train), "threads": torch.get_num_threads(), "step": step}
    log.write(json.dumps(meta) + "\n")

    def lr_at(s):
        if s < a.warmup:
            return a.lr * (s + 1) / a.warmup
        p = min(1.0, train_time / budget)
        return a.min_lr + 0.5 * (a.lr - a.min_lr) * (1 + math.cos(math.pi * p))

    autocast = torch.autocast("cpu", dtype=torch.bfloat16, enabled=a.bf16)

    while True:
        try:
            batch = next(batches)
        except StopIteration:
            batches = iter(make_batches(train, a.batch, rng))
            epoch += 1
            batch = next(batches)
        x, y = to_tensors(batch)
        ts = time.time()
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        with autocast:
            logits = model(x)
        loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.view(-1), ignore_index=-100)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        opt.step()
        step += 1
        train_time += time.time() - ts
        tokens_seen += int((x != PAD).sum())
        elapsed = time.time() - t0

        if a.probe and step == a.probe:
            print(json.dumps({"probe": True, "bf16": a.bf16, "sec_per_step": round(train_time / step, 4),
                              "params": n_params}), flush=True)
            return

        if step % 25 == 0:
            log.write(json.dumps({"event": "train", "step": step, "epoch": epoch, "t": round(elapsed, 1),
                                  "train_t": round(train_time, 1), "lr": lr_at(step), "loss": round(loss.item(), 5),
                                  "gnorm": round(gnorm, 4), "tokens": tokens_seen}) + "\n")
        if step % 250 == 0:
            print(f"step {step} loss {loss.item():.4f} train_t {train_time / 60:.1f}m", flush=True)

        done = train_time >= budget
        if step % a.eval_every == 0 or done:
            vl, vacc = val_loss(model, val)
            rec = {"event": "eval", "step": step, "t": round(elapsed, 1), "train_t": round(train_time, 1),
                   "val_loss": round(vl, 5), "val_char_acc": round(vacc, 5)}
            if step % (a.eval_every * 4) == 0 or done:
                for name, rows in tests.items():
                    rec[f"em_{name}"] = exact_match(model, rows[:100])["exact_match"]
            log.write(json.dumps(rec) + "\n")
            print(json.dumps(rec), flush=True)
            if vl < best:
                best = vl
                torch.save({"config": cfg.to_dict(), "model": model.state_dict(), "step": step, "val_loss": vl},
                           f"{a.out}/ckpt.pt")
        if done:
            break
        if time.time() - last_save > a.save_every * 60:
            # on resume the interrupted epoch restarts with a fresh batch order from the saved rng
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "epoch": epoch,
                        "tokens": tokens_seen, "train_time": train_time, "best": best, "rng": rng.getstate(),
                        "torch_rng": torch.get_rng_state(), "elapsed": time.time() - t0}, last + ".tmp")
            os.replace(last + ".tmp", last)
            last_save = time.time()

    log.write(json.dumps({"event": "end", "step": step, "t": round(time.time() - t0, 1),
                          "tokens": tokens_seen, "best_val_loss": best}) + "\n")
    print("done", step, "steps", flush=True)


if __name__ == "__main__":
    main()
