"""ProcessTransformer (Bukhsh et al. 2021; official Keras code, commit 3f041e4) re-implemented in PyTorch, on our split files.

  python -m baselines.run_processtransformer --log BPIC15_1 --seed 0 [--epochs 100] [--out runs_real]
As transformer.py: two separate models, next activity (cross entropy) and next Δt (LogCosh on the standardized
target); 1 transformer block, d 36, 4 heads with key_dim 36, ff 64, dropout 0.1, no attention mask, mean pooling;
Adam 1e-3 (Keras eps 1e-7), batch 12, reshuffled each epoch; Keras initializers. Inputs as processor.py/loader.py:
prefix tokens left-padded to the longest case (as run_seplstm.py; official: longest train prefix); 3 time features of the last prefix event
(recent_time = gap to the event 2 back, latest_time = gap to the previous event, time_passed = time since case
start) and the target Δt, both standardized with train statistics (StandardScaler).
Deviations (README): exact time in days instead of whole days; our vocabulary (PAD 0, OOV as UNK); each model
selected by lowest val loss, <= 100 epochs, stop after 10 without improvement (official: 10 epochs, lowest train
loss). Test: p_act = softmax of the activity model; Δt̂ clamped >= 0, in days. Writes <out>/<log>/PT_s<seed>/.
"""
import argparse
import copy
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from baselines.run_sutran import REPO, set_seed
from train.all0 import target_rows
from train.dataset import OUT_DIR, load_log

HP = {"d_model": 36, "heads": 4, "key_dim": 36, "ff": 64, "dropout": 0.1, "lr": 1e-3, "adam_eps": 1e-7,
      "batch_size": 12, "patience": 10}


def instances(d: dict, W: int) -> tuple:
    """Prefix tokens [N, W] (left-padded/truncated), time features [N, 3] (days), labels, (case, pos) keys."""
    keys, tok, feat, act, dt = [], [], [], [], []
    for i, c in enumerate(d["case"]):
        types, t = d["types"][i], d["event_time"][i]
        for j in d["target_mask"][i].nonzero().flatten().tolist():
            p = types[max(0, j - W):j]
            row = torch.zeros(W, dtype=torch.long)
            row[W - len(p):] = p
            last = j - 1
            feat.append([float(t[last] - t[last - 2]) if last >= 2 else 0.0,     # recent_time
                         float(t[last] - t[last - 1]) if last >= 1 else 0.0,     # latest_time
                         float(t[last])])                                         # time_passed
            keys.append((c, j))
            tok.append(row)
            act.append(int(types[j]) - 1)
            dt.append(float(d["dt"][i, j]))
    return torch.stack(tok), torch.tensor(feat), torch.tensor(act), torch.tensor(dt).unsqueeze(1), keys


class MHA(nn.Module):
    """Keras MultiHeadAttention(num_heads, key_dim): per-head projections of size key_dim, output back to d."""

    def __init__(self, d: int, heads: int, key: int):
        super().__init__()
        self.h, self.k = heads, key
        self.q, self.kk, self.v = nn.Linear(d, heads * key), nn.Linear(d, heads * key), nn.Linear(d, heads * key)
        self.o = nn.Linear(heads * key, d)

    def forward(self, x):
        B, L, _ = x.shape
        q, k, v = (f(x).view(B, L, self.h, self.k).transpose(1, 2) for f in (self.q, self.kk, self.v))
        a = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(self.k), -1)
        return self.o((a @ v).transpose(1, 2).reshape(B, L, -1))


class Block(nn.Module):
    """Official TransformerBlock: post-norm attention and feed-forward, LayerNorm eps 1e-6."""

    def __init__(self, d: int, heads: int, key: int, ff: int, rate: float):
        super().__init__()
        self.att = MHA(d, heads, key)
        self.ffn = nn.Sequential(nn.Linear(d, ff), nn.ReLU(), nn.Linear(ff, d))
        self.n1, self.n2 = nn.LayerNorm(d, eps=1e-6), nn.LayerNorm(d, eps=1e-6)
        self.drop1, self.drop2 = nn.Dropout(rate), nn.Dropout(rate)

    def forward(self, x):
        x = self.n1(x + self.drop1(self.att(x)))
        return self.n2(x + self.drop2(self.ffn(x)))


class PT(nn.Module):
    """get_next_activity_model (time=False) or get_next_time_model (time=True) of the official transformer.py."""

    def __init__(self, W: int, vocab_size: int, out: int, time: bool):
        super().__init__()
        d, rate = HP["d_model"], HP["dropout"]
        self.tok, self.pos = nn.Embedding(vocab_size, d), nn.Embedding(W, d)
        self.block = Block(d, HP["heads"], HP["key_dim"], HP["ff"], rate)
        self.time = nn.Sequential(nn.Linear(3, 32), nn.ReLU()) if time else None
        width, hidden = (d + 32, 128) if time else (d, 64)
        self.head = nn.Sequential(nn.Dropout(rate), nn.Linear(width, hidden), nn.ReLU(), nn.Dropout(rate),
                                  nn.Linear(hidden, out))
        keras_init(self)

    def forward(self, x, f=None):
        h = self.block(self.tok(x) + self.pos.weight[:x.size(1)]).mean(1)
        if self.time is not None:
            h = torch.cat([h, self.time(f)], -1)
        return self.head(h)


def keras_init(model: PT) -> None:
    """Keras defaults: glorot_uniform kernels (attention kernels with EinsumDense fans), zero biases, U(±0.05) embeddings."""
    glorot = lambda w, fan_in, fan_out: nn.init.uniform_(w, -math.sqrt(6 / (fan_in + fan_out)), math.sqrt(6 / (fan_in + fan_out)))
    for m in model.modules():
        if isinstance(m, nn.Embedding):
            nn.init.uniform_(m.weight, -0.05, 0.05)
        elif isinstance(m, nn.Linear):
            glorot(m.weight, m.in_features, m.out_features)
            nn.init.zeros_(m.bias)
    d, h, k = HP["d_model"], HP["heads"], HP["key_dim"]
    for m in model.modules():                         # kernels [d, h, k] and [h, k, d]: Keras fans for rank-3 shapes
        if isinstance(m, MHA):
            for w in (m.q.weight, m.kk.weight, m.v.weight):
                glorot(w, h * d, k * d)
            glorot(m.o.weight, k * h, d * h)


def logcosh(p, y):
    """Keras LogCosh: mean of log(cosh(p - y)), computed stably."""
    x = p - y
    return (x + F.softplus(-2 * x) - math.log(2)).mean()


@torch.no_grad()
def val_loss(model, data: tuple, loss_fn, batch: int = 4096) -> float:
    """Mean loss over all validation instances."""
    model.eval()
    n = len(data[-1])
    return sum(float(loss_fn(model(*[t[b:b + batch] for t in data[:-1]]), data[-1][b:b + batch])) * len(data[-1][b:b + batch])
               for b in range(0, n, batch)) / n


def fit(model, train: tuple, val: tuple, loss_fn, epochs: int, name: str) -> int:
    """Train with Adam, batch 12, reshuffled each epoch; restore the epoch with the lowest val loss; return it."""
    opt = torch.optim.Adam(model.parameters(), lr=HP["lr"], eps=HP["adam_eps"])
    best, best_epoch, state, wait = math.inf, -1, None, 0
    for epoch in range(epochs):
        model.train()
        for idx in torch.randperm(len(train[-1]), device=train[-1].device).split(HP["batch_size"]):
            loss = loss_fn(model(*[t[idx] for t in train[:-1]]), train[-1][idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        v = val_loss(model, val, loss_fn)
        print(f"{name} epoch {epoch:3d}  last batch {float(loss):.4f}  val {v:.4f}", flush=True)
        if v < best:
            best, best_epoch, state, wait = v, epoch, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= HP["patience"]:
                break
    model.load_state_dict(state)
    return best_epoch


def main():
    """Build inputs, train both models, predict on test, write the run folder."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--out", type=Path, default=REPO / "runs_real")
    args = ap.parse_args()
    folder = args.out / args.log / f"PT_s{args.seed}"
    assert not (folder / "test_pred.pt").exists(), f"{folder}: run exists"
    folder.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train, val, test, vocab = load_log(args.log)
    T = len(vocab)
    W = max(d["types"].size(1) for d in (train, val, test))              # longest case, as run_seplstm.py
    tr, va, te = (instances(d, W) for d in (train, val, test))
    fm, fs = tr[1].mean(0), tr[1].std(0, unbiased=False).clamp(min=1e-12)
    ym, ys = float(tr[3].mean()), float(tr[3].std(unbiased=False))
    z = lambda s: [t.to(dev) for t in ((s[1] - fm) / fs, (s[3] - ym) / ys)]
    print(f"{args.log}: W={W} instances train={len(tr[4])} val={len(va[4])} test={len(te[4])} classes={T}")

    set_seed(args.seed)
    act = PT(W, T + 1, T, time=False).to(dev)
    e_act = fit(act, (tr[0].to(dev), tr[2].to(dev)), (va[0].to(dev), va[2].to(dev)), F.cross_entropy, args.epochs, "act")
    tim = PT(W, T + 1, 1, time=True).to(dev)
    e_time = fit(tim, (tr[0].to(dev), *z(tr)), (va[0].to(dev), *z(va)), logcosh, args.epochs, "time")

    act.eval()
    tim.eval()
    with torch.no_grad():
        x, f = te[0].to(dev), z(te)[0]
        p_act = torch.cat([torch.softmax(act(x[b:b + 4096]), -1) for b in range(0, len(x), 4096)]).cpu()
        dt_hat = torch.cat([tim(x[b:b + 4096], f[b:b + 4096]) for b in range(0, len(x), 4096)]).cpu() * ys + ym

    rows = target_rows(test)
    assert list(zip(rows["case"], rows["pos"])) == te[4], "instance order differs from test targets"
    rows["p_act"] = p_act
    rows["time"] = dt_hat.clamp(min=0)
    torch.save(rows, folder / "test_pred.pt")
    cfg = {"variant": "PT", "log": args.log, "seed": str(args.seed), "n_mix": 0, "vocab": vocab,
           "split_hashes": json.loads((OUT_DIR / "hashes.json").read_text())[args.log],
           "pt_commit": "3f041e4eae9a2efcea0da02166e98efaa953801c", "hp": HP, "epochs": args.epochs, "window": W,
           "best_epoch_act": e_act, "best_epoch_time": e_time,
           "train_stats": {"features_mean": fm.tolist(), "features_std": fs.tolist(), "dt": [ym, ys]}}
    (folder / "config.json").write_text(json.dumps(cfg, indent=1))
    print(f"best epochs act {e_act}, time {e_time} -> {folder}")


if __name__ == "__main__":
    main()
