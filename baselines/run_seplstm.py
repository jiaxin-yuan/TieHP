"""SEP-LSTM (SuTraN repo's PyTorch re-implementation of Tax et al. 2017, commit f120e6f) on our split files.

  python -m baselines.run_seplstm --log BPIC15_1 --seed 0 [--epochs 500] [--out runs_real]
Same instances, train statistics and standardization as baselines/run_sutran.py, converted by the official
convert_to_SEP_data (left padding; labels = step 0 only). Training and best epoch (lowest composite val loss) as
TRAIN_EVAL_SEP_LSTM.py: d_model 100, 1 shared + 1 specialized LSTM layer, dropout 0.2, NAdam 2e-3,
ReduceLROnPlateau(0.5, patience 10), <= 500 epochs, stop after 42 epochs without improvement, batch 128.
Test: one forward pass = next activity logits and Δt; p_act = softmax over our activities (PAD, END dropped);
Δt̂ clamped >= 0 (as the official inference), in days. Writes <out>/<log>/SEPLSTM_s<seed>/ in train.py's format.
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import TensorDataset

from baselines.run_sutran import DAY, REPO, instances, set_seed, standardize, train_stats
from OneStepAheadBenchmarks.convert_tensordata import convert_to_SEP_data   # path set by run_sutran
from OneStepAheadBenchmarks.models import SEP_Benchmarks_time
from OneStepAheadBenchmarks.train_procedure import train_model
from train.all0 import target_rows
from train.dataset import OUT_DIR, load_log

HP = {"d_model": 100, "dropout": 0.2, "num_shared_layers": 1, "num_specialized_layers": 1, "lr": 2e-3,
      "plateau_factor": 0.5, "plateau_patience": 10, "batch_size": 128}


def sep(t: list) -> tuple:
    """Official conversion of our NDA tuple (activity tensor at 0, time features at 1) to the SEP-LSTM tuple."""
    return convert_to_SEP_data(data=tuple(t), outcome_bool=False, num_categoricals_pref=1, act_input_index=0, tss_index=0)


def new_model(num_activities: int) -> SEP_Benchmarks_time:
    """SEP-LSTM with the official hyperparameters."""
    return SEP_Benchmarks_time(d_model=HP["d_model"], num_activities=num_activities, dropout=HP["dropout"],
                               num_shared_layers=HP["num_shared_layers"],
                               num_specialized_layers=HP["num_specialized_layers"])


@torch.no_grad()
def predict(model, t: tuple, ttne: list, device, batch: int = 2048) -> tuple:
    """Next-activity logits [N, C] and Δt̂ in days (clamped >= 0) for every instance."""
    model.eval()
    logits, out = [], []
    for b in range(0, len(t[0]), batch):
        lg, tt = model([v[b:b + batch].to(device) for v in t[:2]])
        logits.append(lg.cpu())
        out.append(tt[:, 0].cpu())
    sec = (torch.cat(out) * ttne[1] + ttne[0]).clamp(min=0)
    return torch.cat(logits), sec / DAY


def main():
    """Build tensors, train SEP-LSTM, predict on test, write the run folder."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--out", type=Path, default=REPO / "runs_real")
    args = ap.parse_args()
    folder = args.out / args.log / f"SEPLSTM_s{args.seed}"
    assert not (folder / "test_pred.pt").exists(), f"{folder}: run exists"
    folder.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train, val, test, vocab = load_log(args.log)
    T = len(vocab)
    C, eos = T + 2, T + 1
    W = max(d["types"].size(1) for d in (train, val, test))
    raw = {name: instances(d, W, eos) for name, d in (("train", train), ("val", val), ("test", test))}
    s = train_stats(raw["train"][0])
    tr, va, te = (sep(standardize(raw[n][0], s)) for n in ("train", "val", "test"))
    print(f"{args.log}: W={W} instances train={len(tr[0])} val={len(va[0])} test={len(te[0])} classes={C}")

    set_seed(args.seed)
    model = new_model(C).to(device)
    opt = torch.optim.NAdam(model.parameters(), lr=HP["lr"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=HP["plateau_factor"],
                                                       patience=HP["plateau_patience"], threshold=1e-4,
                                                       threshold_mode="rel", cooldown=0, min_lr=0, eps=1e-8)
    train_model(model=model, optimizer=opt, train_dataset=TensorDataset(*tr), val_dataset=TensorDataset(*va),
                start_epoch=0, num_epochs=args.epochs, num_classes=C, batch_interval=1600, path_name=str(folder),
                mean_std_ttne=s["ttne"], batch_size=HP["batch_size"], lr_scheduler_present=True,
                lstm_scheduler=True, lr_scheduler=sched)

    df = pd.read_csv(folder / "backup_results.csv")
    epoch = int(df.loc[df["composite validation loss"].idxmin(), "epoch"])
    model = new_model(C).to(device)
    model.load_state_dict(torch.load(folder / f"model_epoch_{epoch}.pt", map_location=device)["model_state_dict"])
    logits, dt_hat = predict(model, te, s["ttne"], device)

    rows = target_rows(test)
    assert list(zip(rows["case"], rows["pos"])) == raw["test"][1], "instance order differs from test targets"
    full = torch.softmax(logits.masked_fill(torch.arange(C) == 0, -1e9), -1)
    rows["p_act"] = torch.softmax(logits[:, 1:T + 1], -1)
    rows["time"] = dt_hat.unsqueeze(1)
    torch.save(rows, folder / "test_pred.pt")
    cfg = {"variant": "SEPLSTM", "log": args.log, "seed": str(args.seed), "n_mix": 0, "vocab": vocab,
           "split_hashes": json.loads((OUT_DIR / "hashes.json").read_text())[args.log],
           "sutran_commit": "f120e6f7067302b0b5ef26f043322ee2619e07a4", "hp": HP, "epochs": args.epochs,
           "window": W, "best_epoch": epoch, "train_stats": s, "mean_end_mass": float(full[:, eos].mean())}
    (folder / "config.json").write_text(json.dumps(cfg, indent=1))
    print(f"best epoch {epoch}; mean END mass {cfg['mean_end_mass']:.4f} -> {folder}")


if __name__ == "__main__":
    main()
