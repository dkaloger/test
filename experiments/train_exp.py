"""Experimental trainer: baseline GPT + switchable training improvements.

Every idea is a flag, so runs differ only in what is being tested.
Final score = mean cross-entropy over the WHOLE validation split (deterministic).

    python experiments/train_exp.py --name baseline
    python experiments/train_exp.py --name kd --ngram-kd 0.3
"""
import argparse
import copy
import json
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

p = argparse.ArgumentParser()
p.add_argument("--name", required=True)
p.add_argument("--data", default="data/input.txt")
p.add_argument("--iters", type=int, default=3000)
p.add_argument("--batch", type=int, default=32)
p.add_argument("--block", type=int, default=128)
p.add_argument("--layers", type=int, default=4)
p.add_argument("--heads", type=int, default=4)
p.add_argument("--embd", type=int, default=256)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--wd", type=float, default=0.1)
p.add_argument("--dropout", type=float, default=0.1)
p.add_argument("--seed", type=int, default=1337)
p.add_argument("--threads", type=int, default=0)
# --- ideas ---
p.add_argument("--modern", action="store_true", help="RMSNorm + RoPE + SwiGLU")
p.add_argument("--muon", action="store_true", help="Muon optimizer for hidden matrices")
p.add_argument("--muon-lr", type=float, default=0.02)
p.add_argument("--ema", type=float, default=0.0, help="EMA decay of weights, e.g. 0.999 (0=off)")
p.add_argument("--input-noise", type=float, default=0.0, help="prob. of replacing an input char")
p.add_argument("--ngram-kd", type=float, default=0.0, help="weight of trigram-teacher soft targets")
p.add_argument("--kd-anneal", action="store_true", help="linearly fade KD weight to 0")
p.add_argument("--mtp", type=float, default=0.0, help="weight of next-next-char prediction loss")
p.add_argument("--save", default="")
p.add_argument("--eval-every", type=int, default=1000)
args = p.parse_args()

if args.threads:
    torch.set_num_threads(args.threads)
torch.manual_seed(args.seed)

text = open(args.data, encoding="utf-8").read()
chars = sorted(set(text))
V = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]

# ----------------------------------------------------------------- trigram teacher
if args.ngram_kd > 0:
    # P(next | prev2, prev1), add-k smoothed, backed by train split only.
    tri = torch.zeros(V * V, V)
    a, b, c = train_data[:-2], train_data[1:-1], train_data[2:]
    tri.index_put_((a * V + b, c), torch.ones(len(c)), accumulate=True)
    tri = (tri + 0.1) / (tri + 0.1).sum(-1, keepdim=True)


# ----------------------------------------------------------------- model
class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return self.w * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)


def rope(x, cos, sin):  # x: B,H,T,D
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), -1).flatten(-2)


class Attn(nn.Module):
    def __init__(s):
        super().__init__()
        s.qkv = nn.Linear(args.embd, 3 * args.embd, bias=False)
        s.proj = nn.Linear(args.embd, args.embd, bias=False)
        s.drop = nn.Dropout(args.dropout)

    def forward(s, x, cs):
        B, T, C = x.shape
        H = args.heads
        q, k, v = (t.view(B, T, H, C // H).transpose(1, 2) for t in s.qkv(x).split(C, 2))
        if cs is not None:
            q, k = rope(q, *cs), rope(k, *cs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                           dropout_p=args.dropout if s.training else 0.0)
        return s.drop(s.proj(y.transpose(1, 2).reshape(B, T, C)))


class MLP(nn.Module):
    def __init__(s):
        super().__init__()
        d = args.embd
        if args.modern:  # SwiGLU, hidden sized to keep parameter count equal
            h = int(8 * d / 3)
            s.up = nn.Linear(d, 2 * h, bias=False)
            s.down = nn.Linear(h, d, bias=False)
        else:
            s.up = nn.Linear(d, 4 * d, bias=False)
            s.down = nn.Linear(4 * d, d, bias=False)
        s.drop = nn.Dropout(args.dropout)

    def forward(s, x):
        if args.modern:
            a, g = s.up(x).chunk(2, -1)
            return s.drop(s.down(F.silu(g) * a))
        return s.drop(s.down(F.gelu(s.up(x))))


Norm = RMSNorm if args.modern else nn.LayerNorm


class Block(nn.Module):
    def __init__(s):
        super().__init__()
        s.n1, s.a, s.n2, s.m = Norm(args.embd), Attn(), Norm(args.embd), MLP()

    def forward(s, x, cs):
        x = x + s.a(s.n1(x), cs)
        return x + s.m(s.n2(x))


class GPT(nn.Module):
    def __init__(s):
        super().__init__()
        d = args.embd
        s.tok = nn.Embedding(V, d)
        if not args.modern:
            s.pos = nn.Embedding(args.block, d)
        s.drop = nn.Dropout(args.dropout)
        s.blocks = nn.ModuleList(Block() for _ in range(args.layers))
        s.nf = Norm(d)
        s.head = nn.Linear(d, V, bias=False)
        s.head.weight = s.tok.weight
        if args.mtp > 0:
            s.mtp = nn.Sequential(nn.Linear(d, d, bias=False), nn.GELU(), Norm(d))
        for m in s.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, 0, 0.02)
        if args.modern:
            hd = d // args.heads
            inv = 1.0 / (10000 ** (torch.arange(0, hd, 2).float() / hd))
            f = torch.outer(torch.arange(args.block).float(), inv)
            s.register_buffer("cos", f.cos(), persistent=False)
            s.register_buffer("sin", f.sin(), persistent=False)

    def forward(s, idx):
        T = idx.shape[1]
        x = s.tok(idx)
        cs = None
        if args.modern:
            cs = (s.cos[:T], s.sin[:T])
        else:
            x = x + s.pos(torch.arange(T))
        x = s.drop(x)
        for b in s.blocks:
            x = b(x, cs)
        h = s.nf(x)
        return s.head(h), h


model = GPT()
print(f"[{args.name}] params {sum(p.numel() for p in model.parameters())/1e6:.3f}M", flush=True)


# ----------------------------------------------------------------- optimizers
@torch.no_grad()
def newton_schulz(G, steps=5):
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G / (G.norm() + 1e-7)
    tr = X.size(0) > X.size(1)
    if tr:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    return X.T if tr else X


class Muon(torch.optim.Optimizer):
    def __init__(s, params, lr, momentum=0.95, wd=0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum, wd=wd))

    @torch.no_grad()
    def step(s):
        for g in s.param_groups:
            for p in g["params"]:
                if p.grad is None:
                    continue
                st = s.state[p]
                buf = st.setdefault("buf", torch.zeros_like(p))
                buf.mul_(g["momentum"]).add_(p.grad)
                u = newton_schulz(p.grad.add(buf, alpha=g["momentum"]))  # nesterov
                scale = max(1, p.size(0) / p.size(1)) ** 0.5
                p.mul_(1 - g["lr"] * g["wd"])
                p.add_(u, alpha=-g["lr"] * scale)


hidden = [p for nm, p in model.named_parameters() if p.dim() == 2 and "blocks" in nm]
hidden_ids = {id(p) for p in hidden}
rest = [p for p in model.parameters() if id(p) not in hidden_ids]
if args.muon:
    opts = [Muon(hidden, lr=args.muon_lr, wd=args.wd),
            torch.optim.AdamW(rest, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95))]
    base_lrs = [args.muon_lr, args.lr]
else:
    opts = [torch.optim.AdamW([{"params": hidden, "weight_decay": args.wd},
                               {"params": rest, "weight_decay": 0.0}],
                              lr=args.lr, betas=(0.9, 0.95))]
    base_lrs = [args.lr]


def lr_mult(it):
    warm = 100
    if it < warm:
        return it / warm
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (it - warm) / (args.iters - warm)))


ema = copy.deepcopy(model).eval() if args.ema > 0 else None


# ----------------------------------------------------------------- data
def get_batch():
    ix = torch.randint(2, len(train_data) - args.block - 2, (args.batch,))
    off = torch.arange(args.block)
    pos = ix[:, None] + off  # positions of inputs
    return pos


@torch.no_grad()
def full_val(m):
    """Mean CE over every val char (non-overlapping windows; first window token has no context)."""
    m.eval()
    T = args.block
    nwin = (len(val_data) - 1) // T
    X = val_data[: nwin * T].view(nwin, T)
    Y = val_data[1: nwin * T + 1].view(nwin, T)
    tot = 0.0
    for i in range(0, nwin, 64):
        lg, _ = m(X[i:i + 64])
        tot += F.cross_entropy(lg.reshape(-1, V), Y[i:i + 64].reshape(-1), reduction="sum").item()
    m.train()
    return tot / (nwin * T)


# ----------------------------------------------------------------- train
t0 = time.time()
curve = []
for it in range(1, args.iters + 1):
    mult = lr_mult(it)
    for o, blr in zip(opts, base_lrs):
        for g in o.param_groups:
            g["lr"] = blr * mult

    pos = get_batch()
    x, y = train_data[pos], train_data[pos + 1]
    if args.input_noise > 0:
        mask = torch.rand(x.shape) < args.input_noise
        x = torch.where(mask, torch.randint(V, x.shape), x)

    logits, h = model(x)
    logp = F.log_softmax(logits, -1)
    ce = F.nll_loss(logp.reshape(-1, V), y.reshape(-1))
    loss = ce
    if args.ngram_kd > 0:
        teacher = tri[train_data[pos - 1] * V + train_data[pos]]  # from clean context
        kd = -(teacher * logp).sum(-1).mean()
        w = args.ngram_kd * (1 - it / args.iters) if args.kd_anneal else args.ngram_kd
        loss = (1 - w) * ce + w * kd
    if args.mtp > 0:
        y2 = train_data[pos + 2]
        lg2 = model.head(model.mtp(h))
        loss = loss + args.mtp * F.cross_entropy(lg2.reshape(-1, V), y2.reshape(-1))

    for o in opts:
        o.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    for o in opts:
        o.step()

    if ema is not None:
        with torch.no_grad():
            d = args.ema if it > 200 else 0.0  # start averaging after warmup
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1 - d)

    if it % args.eval_every == 0 or it == args.iters:
        v = full_val(ema if ema is not None else model)
        curve.append((it, round(v, 4)))
        print(f"[{args.name}] it {it} ce {ce.item():.3f} val {v:.4f} {time.time()-t0:.0f}s", flush=True)

res = {"name": args.name, "val": curve[-1][1], "best": min(v for _, v in curve),
       "curve": curve, "secs": round(time.time() - t0), "args": vars(args)}
os.makedirs("experiments/results", exist_ok=True)
json.dump(res, open(f"experiments/results/{args.name}.json", "w"), indent=1)
print("RESULT", json.dumps({k: res[k] for k in ("name", "val", "best", "secs")}), flush=True)
if args.save:
    torch.save({"state_dict": (ema if ema is not None else model).state_dict(), "chars": chars, "args": vars(args)}, args.save)
