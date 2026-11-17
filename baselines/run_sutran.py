"""SuTraN (NDA, official code at baselines/ext/SuffixTransformerNetwork, commit f120e6f) on our split files.

  python -m baselines.run_sutran --log BPIC15_1 --seed 0 [--epochs 200] [--out runs_real]
One SuTraN instance per target of train.dataset.load_log (same hash check, vocab with <OOV>, targets): prefix =
events 0..j-1, suffix tokens = events j-1..n-1, labels = activities j..n-1 + END, Δt j..n-1 + 0, remaining time.
Times in seconds, standardized with train statistics (as SuTraN's preprocessing, log_transformed=False); padding
values as SuTraN (prefix 0, suffix -1, labels -100). Window W = longest case, no case dropped.
Training and best-epoch rule (summed val rank of activity-suffix DL and remaining-time MAE) as TRAIN_EVAL_SUTRAN_NDA.py.
Test: decoder step 0 = next event; read from the teacher-forced pass with dropout off (the look-ahead mask makes
step 0 see only the SOS token), checked equal to the greedy decode. p_act = softmax over our activities (PAD and END
dropped, END mass in config.json); Δt̂ = step-0 time in days. Writes <out>/<log>/SUTRAN_s<seed>/ in train.py's format.
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset

REPO = Path(__file__).resolve().parent.parent
SUTRAN_DIR = REPO / "baselines" / "ext" / "SuffixTransformerNetwork"
sys.path.insert(0, str(SUTRAN_DIR))
from SuTraN.SuTraN import SuTraN_no_context          # noqa: E402
from SuTraN.train_procedure import train_model        # noqa: E402

from train.all0 import target_rows                    # noqa: E402
from train.dataset import OUT_DIR, load_log           # noqa: E402

DAY = 86400.0
HP = {"d_model": 32, "num_prefix_encoder_layers": 4, "num_decoder_layers": 4, "num_heads": 8, "d_ff": 128,
      "dropout": 0.2, "lr": 2e-4, "weight_decay": 1e-4, "decay": 0.96, "batch_size": 128, "patience": 24}


def instances(d: dict, W: int, eos: int) -> tuple:
    """Raw (unstandardized, seconds) SuTraN NDA tuple of one split, one instance per target, plus (case, pos) keys."""
    keys = [(c, j) for i, c in enumerate(d["case"]) for j in d["target_mask"][i].nonzero().flatten().tolist()]
    N = len(keys)
    pa, pn, pm = torch.zeros(N, W, dtype=torch.long), torch.zeros(N, W, 2), torch.ones(N, W, dtype=torch.bool)
    sa, sn = torch.zeros(N, W, dtype=torch.long), torch.full((N, W, 2), float("nan"))
    ttne, rrt, al = torch.full((N, W, 1), float("nan")), torch.full((N, W, 1), float("nan")), torch.zeros(N, W, dtype=torch.long)
    row = {c: i for i, c in enumerate(d["case"])}
    for k, (c, j) in enumerate(keys):
        i = row[c]
        n = int((d["types"][i] > 0).sum())
        types, tss, tsp = d["types"][i, :n], d["event_time"][i, :n] * DAY, d["dt"][i, :n] * DAY
        L = n - j + 1                                                     # suffix tokens: events j-1..n-1
        pa[k, :j], pn[k, :j, 0], pn[k, :j, 1], pm[k, :j] = types[:j], tss[:j], tsp[:j], False
        sa[k, :L], sn[k, :L, 0], sn[k, :L, 1] = types[j - 1:], tss[j - 1:], tsp[j - 1:]
        al[k, :L - 1], al[k, L - 1] = types[j:], eos
        ttne[k, :L - 1, 0], ttne[k, L - 1, 0] = tsp[j:], 0.0
        rrt[k, :L - 1, 0], rrt[k, L - 1, 0] = tss[-1] - tss[j - 1:-1], 0.0
    return [pa, pn, pm, sa, sn, ttne, rrt, al], keys


def train_stats(t: list) -> dict:
    """[mean, std] (population, as StandardScaler) of each time feature over the train rows SuTraN fits on."""
    valid_p, valid_s = ~t[2], t[3] > 0
    ms = lambda x: [float(x.double().mean()), float(x.double().std(unbiased=False))]
    return {"tss_pref": ms(t[1][..., 0][valid_p]), "tsp_pref": ms(t[1][..., 1][valid_p]),
            "tss_suf": ms(t[4][..., 0][valid_s]), "tsp_suf": ms(t[4][..., 1][valid_s]),
            "ttne": ms(t[5][..., 0][valid_s]), "rrt": ms(t[6][:, 0, 0])}


def standardize(t: list, s: dict) -> list:
    """Standardize with train statistics, then SuTraN's padding values (prefix 0, suffix -1, labels -100)."""
    pa, pn, pm, sa, sn, ttne, rrt, al = [x.clone() for x in t]
    z = lambda x, m: (x - m[0]) / m[1]
    valid_p, valid_s = ~pm, sa > 0
    pn[..., 0], pn[..., 1] = z(pn[..., 0], s["tss_pref"]), z(pn[..., 1], s["tsp_pref"])
    pn[~valid_p] = 0.0
    sn[..., 0], sn[..., 1] = z(sn[..., 0], s["tss_suf"]), z(sn[..., 1], s["tsp_suf"])
    sn[~valid_s] = -1.0
    ttne, rrt = z(ttne, s["ttne"]), z(rrt, s["rrt"])
    ttne[~valid_s], rrt[~valid_s] = -100.0, -100.0
    return [pa, pn, pm, sa, sn, ttne, rrt, al]


def set_seed(seed: int) -> None:
    """Seeds and deterministic cuDNN as in TRAIN_EVAL_SUTRAN_NDA.py (which fixes seed 24)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False


def new_model(num_activities: int) -> SuTraN_no_context:
    """SuTraN NDA with the paper's hyperparameters."""
    return SuTraN_no_context(num_activities=num_activities, d_model=HP["d_model"],
                             num_prefix_encoder_layers=HP["num_prefix_encoder_layers"],
                             num_decoder_layers=HP["num_decoder_layers"], num_heads=HP["num_heads"],
                             d_ff=HP["d_ff"], dropout=HP["dropout"], remaining_runtime_head=True,
                             layernorm_embeds=True, outcome_bool=False)


def best_epoch(folder: Path) -> int:
    """Official selection: lowest summed rank of val 1-DL (higher better) and val remaining-time MAE."""
    df = pd.read_csv(folder / "backup_results.csv")
    rank = (df["RRT - mintues MAE validation"].rank(method="min")
            + df["Activity suffix: 1-DL (validation)"].rank(method="min", ascending=False))
    return int(df.loc[rank.idxmin(), "epoch"])


@torch.no_grad()
def step0(model, t: list, s: dict, W: int, device, batch: int = 1024) -> tuple:
    """Step-0 logits [N, C] and Δt̂ in days from the teacher-forced pass with dropout off; first batch checked against the greedy decode."""
    model.eval()
    logits, ttne = [], []
    for b in range(0, len(t[0]), batch):
        x = [v[b:b + batch].to(device) for v in t[:5]]
        model.training = True                                         # top-level flag only: teacher-forced branch
        lg, tt, _ = model(x)
        model.training = False
        if b == 0:                                                    # same step 0 as SuTraN's greedy decoding
            acts, tts, _ = model([v.clone() for v in x], window_size=W, mean_std_ttne=s["ttne"],
                                 mean_std_tsp=s["tsp_suf"], mean_std_tss=s["tss_suf"])
            masked = lg[:, 0].clone()
            masked[:, 0] = -1e9
            assert torch.equal(masked.argmax(-1), acts[:, 0]), "step-0 activity differs from greedy decode"
            assert torch.allclose(tt[:, 0, 0], tts[:, 0], atol=1e-5), "step-0 time differs from greedy decode"
        logits.append(lg[:, 0].cpu())
        ttne.append(tt[:, 0, 0].cpu())
    sec = (torch.cat(ttne) * s["ttne"][1] + s["ttne"][0]).clamp(min=0)   # as SuTraN: seconds, no negatives
    return torch.cat(logits), sec / DAY


def main():
    """Build tensors, train SuTraN, predict step 0 on test, write the run folder."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--out", type=Path, default=REPO / "runs_real")
    args = ap.parse_args()
    folder = args.out / args.log / f"SUTRAN_s{args.seed}"
    assert not (folder / "test_pred.pt").exists(), f"{folder}: run exists"
    folder.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train, val, test, vocab = load_log(args.log)
    T = len(vocab)
    C, eos = T + 2, T + 1                                              # PAD 0, activities 1..T, END T+1
    W = max(d["types"].size(1) for d in (train, val, test))
    raw = {name: instances(d, W, eos) for name, d in (("train", train), ("val", val), ("test", test))}
    s = train_stats(raw["train"][0])
    tr, va, te = (standardize(raw[n][0], s) for n in ("train", "val", "test"))
    print(f"{args.log}: W={W} instances train={len(tr[0])} val={len(va[0])} test={len(te[0])} classes={C}")

    set_seed(args.seed)
    model = new_model(C).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=HP["lr"], weight_decay=HP["weight_decay"])
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=HP["decay"])
    train_model(model, opt, TensorDataset(*tr), tuple(va), 0, args.epochs, True, False, C, 800, str(folder), 1,
                s["ttne"], s["tsp_suf"], s["tss_suf"], s["rrt"], HP["batch_size"], patience=HP["patience"],
                lr_scheduler_present=True, lr_scheduler=sched)

    epoch = best_epoch(folder)
    model = new_model(C).to(device)
    model.load_state_dict(torch.load(folder / f"model_epoch_{epoch}.pt", map_location=device)["model_state_dict"])
    logits, dt_hat = step0(model, te, s, W, device)

    rows = target_rows(test)
    assert list(zip(rows["case"], rows["pos"])) == raw["test"][1], "instance order differs from test targets"
    full = torch.softmax(logits.masked_fill(torch.arange(C) == 0, -1e9), -1)
    rows["p_act"] = torch.softmax(logits[:, 1:T + 1], -1)              # our 0-based activities
    rows["time"] = dt_hat.unsqueeze(1)                                  # time[:, 0] = Δt̂ (days)
    torch.save(rows, folder / "test_pred.pt")
    cfg = {"variant": "SUTRAN", "log": args.log, "seed": str(args.seed), "n_mix": 0, "vocab": vocab,
           "split_hashes": json.loads((OUT_DIR / "hashes.json").read_text())[args.log],
           "sutran_commit": "f120e6f7067302b0b5ef26f043322ee2619e07a4", "hp": HP, "epochs": args.epochs,
           "window": W, "best_epoch": epoch, "train_stats": s, "mean_end_mass": float(full[:, eos].mean())}
    (folder / "config.json").write_text(json.dumps(cfg, indent=1))
    print(f"best epoch {epoch}; mean END mass at step 0 {cfg['mean_end_mass']:.4f} -> {folder}")


if __name__ == "__main__":
    main()
