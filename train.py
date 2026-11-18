"""Train the GPT on a text file. Runs on CPU or GPU.

    python train.py --data data/input.txt --iters 3000
"""
import argparse
import math
import time

import torch

from model import GPT, GPTConfig, CharTokenizer

p = argparse.ArgumentParser()
p.add_argument("--data", default="data/input.txt")
p.add_argument("--out", default="checkpoints/model.pt")
p.add_argument("--iters", type=int, default=3000)
p.add_argument("--batch", type=int, default=32)
p.add_argument("--block", type=int, default=128)
p.add_argument("--layers", type=int, default=4)
p.add_argument("--heads", type=int, default=4)
p.add_argument("--embd", type=int, default=256)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--eval-every", type=int, default=250)
args = p.parse_args()

torch.manual_seed(1337)
device = "cuda" if torch.cuda.is_available() else "cpu"

text = open(args.data, encoding="utf-8").read()
tok = CharTokenizer(sorted(set(text)))
data = torch.tensor(tok.encode(text), dtype=torch.long)
split = int(0.9 * len(data))
train_data, val_data = data[:split], data[split:]

cfg = GPTConfig(vocab_size=len(tok.chars), block_size=args.block, n_layer=args.layers,
                n_head=args.heads, n_embd=args.embd, dropout=0.1)
model = GPT(cfg).to(device)
print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M  device: {device}")

opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1, betas=(0.9, 0.95))


def lr_at(it):  # linear warmup then cosine decay
    warm = 100
    if it < warm:
        return args.lr * it / warm
    return 0.1 * args.lr + 0.9 * args.lr * 0.5 * (1 + math.cos(math.pi * (it - warm) / (args.iters - warm)))


def batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - args.block - 1, (args.batch,))
    x = torch.stack([d[i:i + args.block] for i in ix])
    y = torch.stack([d[i + 1:i + args.block + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def evaluate():
    model.eval()
    out = {}
    for s in ("train", "val"):
        out[s] = sum(model(*batch(s))[1].item() for _ in range(20)) / 20
    model.train()
    return out


t0 = time.time()
for it in range(args.iters + 1):
    for g in opt.param_groups:
        g["lr"] = lr_at(it)
    if it % args.eval_every == 0:
        l = evaluate()
        print(f"iter {it:5d} | train {l['train']:.3f} | val {l['val']:.3f} | {time.time()-t0:.0f}s", flush=True)
    x, y = batch("train")
    _, loss = model(x, y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()

import os
os.makedirs(os.path.dirname(args.out), exist_ok=True)
torch.save({"config": cfg.__dict__, "chars": tok.chars, "state_dict": model.state_dict()}, args.out)
print("saved", args.out)
