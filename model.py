"""MarvijoWebSearch: a tiny decoder-only transformer written from scratch.

Same family as GPT-2 / Llama, shrunk to about 2.7 million parameters:
- character tokenizer (95 printable ASCII chars + 4 special tokens)
- pre-norm blocks with RMSNorm (Zhang and Sennrich 2019)
- rotary position embeddings, RoPE (Su et al. 2021)
- SwiGLU feed-forward (Shazeer 2020)
- no bias terms, input and output embeddings shared (Press and Wolf 2017)

runtime/marvijo-web-search.js re-implements exactly this forward pass in plain JavaScript.
"""
import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------- tokenizer
PAD, BOS, SEP, EOS = 0, 1, 2, 3
SPECIALS = ["<pad>", "<bos>", "<sep>", "<eos>"]
CHARS = [chr(c) for c in range(32, 127)]  # printable ASCII
VOCAB = SPECIALS + CHARS
STOI = {c: i + len(SPECIALS) for i, c in enumerate(CHARS)}


def encode(s):
    return [STOI.get(c, STOI["?"]) for c in s]


def decode(ids):
    return "".join(VOCAB[i] for i in ids if i >= len(SPECIALS))


def build_sequence(inp, out=None):
    """<bos> request <sep> json <eos>"""
    ids = [BOS] + encode(inp) + [SEP]
    if out is not None:
        ids += encode(out) + [EOS]
    return ids


def build_episode(inp, steps):
    """v4 agent episode: <bos> request <sep> call <eos> result <sep> call <eos> ... call <eos>

    steps = [{"call": str, "obs": str or None}, ...]. Returns (ids, target) where target[i]
    is True when token i is something the model must predict (a call char or its <eos>).
    The request and every tool result are context only: no loss on them.
    """
    ids = [BOS] + encode(inp) + [SEP]
    target = [False] * len(ids)
    for s in steps:
        c = encode(s["call"]) + [EOS]
        ids += c
        target += [True] * len(c)
        if s.get("obs") is not None:
            o = encode(s["obs"]) + [SEP]
            ids += o
            target += [False] * len(o)
    return ids, target


# ---------------------------------------------------------------- model
@dataclass
class Config:
    vocab_size: int = len(VOCAB)
    n_layer: int = 6
    n_head: int = 6
    d_model: int = 192
    d_ff: int = 512
    max_len: int = 512
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5

    def to_dict(self):
        return asdict(self)


class RMSNorm(nn.Module):
    def __init__(self, d, eps):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def rope_tables(head_dim, max_len, theta):
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_len).float()
    freqs = torch.outer(t, inv)  # (T, head_dim/2)
    return freqs.cos(), freqs.sin()


def apply_rope(x, cos, sin):
    # x: (B, H, T, Dh). Rotate pairs (first half, second half).
    h = x.shape[-1] // 2
    x1, x2 = x[..., :h], x[..., h:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.h = c.n_head
        self.dh = c.d_model // c.n_head
        self.wq = nn.Linear(c.d_model, c.d_model, bias=False)
        self.wk = nn.Linear(c.d_model, c.d_model, bias=False)
        self.wv = nn.Linear(c.d_model, c.d_model, bias=False)
        self.wo = nn.Linear(c.d_model, c.d_model, bias=False)

    def forward(self, x, cos, sin, cache=None):
        B, T, D = x.shape
        q = self.wq(x).view(B, T, self.h, self.dh).transpose(1, 2)
        k = self.wk(x).view(B, T, self.h, self.dh).transpose(1, 2)
        v = self.wv(x).view(B, T, self.h, self.dh).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if cache is not None:  # incremental decoding
            if "k" in cache:
                k = torch.cat([cache["k"], k], dim=2)
                v = torch.cat([cache["v"], v], dim=2)
            cache["k"], cache["v"] = k, v
            causal = T > 1
        else:
            causal = True
        y = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        return self.wo(y.transpose(1, 2).reshape(B, T, D))


class SwiGLU(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.w1 = nn.Linear(c.d_model, c.d_ff, bias=False)
        self.w3 = nn.Linear(c.d_model, c.d_ff, bias=False)
        self.w2 = nn.Linear(c.d_ff, c.d_model, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n1 = RMSNorm(c.d_model, c.norm_eps)
        self.attn = Attention(c)
        self.n2 = RMSNorm(c.d_model, c.norm_eps)
        self.ff = SwiGLU(c)

    def forward(self, x, cos, sin, cache=None):
        x = x + self.attn(self.n1(x), cos, sin, cache)
        return x + self.ff(self.n2(x))


class MarvijoWebSearch(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.emb = nn.Embedding(c.vocab_size, c.d_model)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layer))
        self.norm = RMSNorm(c.d_model, c.norm_eps)
        cos, sin = rope_tables(c.d_model // c.n_head, c.max_len, c.rope_theta)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.apply(self._init)
        for b in self.blocks:  # GPT-2 style: scale residual projections by depth
            nn.init.normal_(b.attn.wo.weight, std=0.02 / math.sqrt(2 * c.n_layer))
            nn.init.normal_(b.ff.w2.weight, std=0.02 / math.sqrt(2 * c.n_layer))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, start=0, caches=None):
        T = idx.shape[1]
        cos, sin = self.cos[start:start + T], self.sin[start:start + T]
        x = self.emb(idx)
        for i, b in enumerate(self.blocks):
            x = b(x, cos, sin, None if caches is None else caches[i])
        return self.norm(x) @ self.emb.weight.T  # tied output head

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def generate(self, request, max_new=200):
        """Greedy decoding with a KV cache. Returns the JSON string."""
        return self.generate_ids(build_sequence(request), max_new)

    @torch.no_grad()
    def generate_ids(self, ids, max_new=200):
        """Greedy decoding from any prefix (v4: an episode so far). Stops at <eos>."""
        caches = [{} for _ in self.blocks]
        logits = self(torch.tensor([ids]), 0, caches)
        out, pos = [], len(ids)
        for _ in range(max_new):
            nxt = int(logits[0, -1].argmax())
            if nxt == EOS or pos >= self.c.max_len:
                break
            out.append(nxt)
            logits = self(torch.tensor([[nxt]]), pos, caches)
            pos += 1
        return decode(out)
