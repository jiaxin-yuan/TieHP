"""Train one variant (A0-A3, A3J joint, A3eps = A3-ε, HA1/HA3 hurdle, A3M tie-block b′, A3B tie-block b, A3C = A3 + CE at at-risk contexts) on one prepared log.

  python -m train.train --log g1_batch30_s0 --variant A3 --seed 0
Writes runs/<log>/<variant>_s<seed>/: config.json, history.csv, last.pt, model.pt, test_pred.pt.
Loss per batch = (L_act + λ L_time) / #targets; A3J requires λ=1 for the joint likelihood.
Optimiser, clipping, batch size, epochs and
patience as THP-M run_thp.sh / trainer/train.py; deviations (README Training): min_delta, schedule plateau.
An interrupted run resumes from last.pt when started again with the same arguments.
--replicas N > 1: seeds seed..seed+N-1 trained together in one vmapped group (model.stacking_local, unedited);
one ordinary run folder per seed; shared batch order and train seed, so not identical to N serial runs; no resume.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
import random
import subprocess
import time
from pathlib import Path
import platform

import numpy as np
import torch

from model.losses import block_set_nll, block_start, ce_nll, pre_block_set_nll, eps_nll, joint_set_nll, risk_set_nll, set_nll, thp_nll, zero_atom_nll
from model.model import Transformer
from train.collapse_risk import risky, targets
from train.dataset import OUT_DIR, load_log

REPO = Path(__file__).resolve().parent.parent
# variant -> (model_type, activity loss); time loss follows the model type (A3eps overrides to eps_nll)
VARIANTS = {"A0": ("baseline_act", ce_nll), "A1": ("mixture", ce_nll),
            "A2": ("baseline_act", set_nll), "A3": ("mixture", set_nll),
            "A3eps": ("mixture", set_nll),  # A3-ε: A3's model, time loss eps_nll; its tie head gets no gradient
            "HA1": ("mixture", ce_nll), "HA3": ("mixture", set_nll),
            "A3J": ("mixture_joint", set_nll),  # joint likelihood; act/time decomposition in batch_loss
            "A3M": ("mixture", block_set_nll),  # A3 with activity loss on Δt > 0 targets only (tie-block design b′)
            "A3B": ("mixture", pre_block_set_nll),  # A3, each target predicted from the state before its block (design b)
            "A3C": ("mixture", risk_set_nll)}  # A3 with CE at at-risk contexts (train/collapse_risk.py risky, train-fixed)
HURDLE = {"HA1", "HA3"}  # plan_exp.md 2.1: Δt = 0 targets dropped from both losses in train and val; test keeps all
EPS = 1e-5               # days (≈ 0.9 s); A3-ε floor on Δt, EasyTPP's 1e-5 (in its own unit; choice uncertain)
# plan.md §2 C2 network sizes: d_model, d_inner, n_layers, n_head, d_k (= d_v)
SIZES = {"2k": (24, 48, 2, 3, 8), "10k": (48, 96, 2, 4, 12)}
SIZE_SPLIT = 11_000      # train targets; ≈ geometric mean of 4.8k (2k cases) and 25.2k (10k cases)
EVAL_SEED = 12345        # fixed RNG for validation (A0/A2's Monte Carlo integral)
RNG_POLICY = "v2: isolated init; domain-derived execution/data seeds; epoch order; all-device eval isolation"
KEYS = ("types", "event_time", "dt", "R", "cause", "target_mask", "risk")
HEADER = "epoch,train_loss,train_act,train_time,train_gnorm,val_loss,val_act,val_time,lr,seconds"


def model_config(variant, size, num_types, n_mix=5, dropout=.1):
    """Resolve experiment names to the exact Transformer constructor dictionary."""
    width, inner, layers, heads, key = SIZES[size]
    return dict(num_types=num_types, d_model=width, d_inner=inner, n_layers=layers,
                n_head=heads, d_k=key, d_v=key, n_mix=n_mix, dropout=dropout,
                model_type=VARIANTS[variant][0])


def stream_seed(seed: int, role: str) -> int:
    """Stable uint32 seed per role; do not reuse the initialization RNG sequence."""
    return int.from_bytes(hashlib.sha256(f"bp26:{role}:{seed}".encode()).digest()[:4], "little")


def parse_args():
    """Command-line arguments."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", required=True)
    ap.add_argument("--variant", required=True, choices=list(VARIANTS))
    ap.add_argument("--seed", type=int, default=0, help="CPU model initialization seed")
    ap.add_argument("--train_seed", type=int, help="dropout/Monte Carlo seed; derived from --seed by default")
    ap.add_argument("--data_seed", type=int, help="epoch-order seed; derived from --seed by default")
    ap.add_argument("--eval_seed", type=int, default=EVAL_SEED)
    ap.add_argument("--size", choices=["auto", *SIZES], default="auto", help="auto: by number of train targets")
    ap.add_argument("--n_mix", type=int, default=5)
    ap.add_argument("--lam", type=float, default=1.0, help="λ, weight of the time loss")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=24)
    ap.add_argument("--min_delta", type=float, default=1e-4, help="smallest val-loss drop that counts as improvement")
    ap.add_argument("--sched", choices=["step", "plateau"], default="plateau",
                    help="step: StepLR(10, 0.5) as THP-M; plateau: halve lr after 5 epochs without improvement")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=0.002)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, default=REPO / "runs")
    ap.add_argument("--replicas", type=int, default=1, help=">1: train seeds seed..seed+N-1 as one stacked group")
    ap.add_argument("--init_mean", action="store_true", help="mixture-head biases from train statistics (off = THP-M init)")
    args = ap.parse_args()
    if args.variant == "A3J" and args.lam != 1:
        ap.error("A3J is one joint likelihood and requires --lam 1")
    return args


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch, and make torch deterministic."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")      # needed by deterministic cuBLAS
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def git_commit() -> str:
    """Current commit, with '+dirty' if the tree has uncommitted changes; 'unknown' without git."""
    try:
        def run(*a):
            return subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True, check=True).stdout
        return run("rev-parse", "HEAD").strip() + ("+dirty" if run("status", "--porcelain", "-uno").strip() else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def risk_mask(data: dict, keys: set) -> torch.Tensor:
    """[N,S] bool: targets whose context (activities before them) is in the at-risk set; all False if keys is empty."""
    risk = torch.zeros_like(data["target_mask"])
    for i, k in (data["target_mask"].nonzero().tolist() if keys else []):
        risk[i, k] = tuple(data["types"][i, :k].tolist()) in keys
    return risk


def batches(data: dict, batch_size: int, device, shuffle: bool, seed=None):
    """Yield dicts of tensors for each batch of cases."""
    n = data["types"].size(0)
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    order = torch.randperm(n, generator=generator, device="cpu") if shuffle else torch.arange(n, device="cpu")
    for s in range(0, n, batch_size):
        idx = order[s:s + batch_size]
        yield {k: data[k][idx].to(device) for k in KEYS}


def batch_loss(model, b: dict, act_loss, n_mix: int, eps=None):
    """Summed act/time components and target count; A3J time = joint − act, total at λ=1."""
    h, out = model(b["types"], b["event_time"])
    m = b["target_mask"]
    if model.model_type == "baseline_act":
        time_pred, logits = out
        l_time = thp_nll(model, h, b["event_time"], b["types"], time_pred, b["dt"], m)
    else:
        tie_logit, time_params, logits = out
        if model.model_type == "mixture_joint":
            l_act = set_nll(logits, b["R"], m)
            joint = joint_set_nll(logits, tie_logit, time_params, b["dt"], b["R"], m, n_mix)
            return l_act, joint - l_act, int(m[:, 1:].sum())
        l_time = (zero_atom_nll(tie_logit, time_params, b["dt"], m, n_mix) if eps is None
                  else eps_nll(time_params, b["dt"], m, n_mix, eps))
    l_act = (ce_nll(logits, b["types"], m) if act_loss is ce_nll
             else act_loss(logits, b["R"], m, b["dt"]) if act_loss in (block_set_nll, pre_block_set_nll)
             else act_loss(logits, b["R"], m, b["types"], b["risk"]) if act_loss is risk_set_nll
             else set_nll(logits, b["R"], m))
    return l_act, l_time, int(m[:, 1:].sum())


def run_epoch(model, data, act_loss, args, optimiser=None, *, shuffle_seed=None) -> dict:
    """One pass over a split (training if an optimiser is given); mean loss, act, time per target and mean grad norm."""
    model.train(optimiser is not None)
    act, tim, n, gnorm, steps = 0.0, 0.0, 0, 0.0, 0
    with torch.set_grad_enabled(optimiser is not None):
        for b in batches(data, args.batch_size, args.device, shuffle=optimiser is not None, seed=shuffle_seed):
            l_act, l_time, nt = batch_loss(model, b, act_loss, args.n_mix, EPS if args.variant == "A3eps" else None)
            if optimiser is not None:
                optimiser.zero_grad()
                ((l_act + args.lam * l_time) / max(nt, 1)).backward()
                gnorm += float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))   # norm before clipping
                steps += 1
                optimiser.step()
            act, tim, n = act + l_act.item(), tim + l_time.item(), n + nt
    act, tim = act / max(n, 1), tim / max(n, 1)
    return {"loss": act + args.lam * tim, "act": act, "time": tim, "gnorm": gnorm / max(steps, 1)}


def evaluate(model, data, act_loss, args) -> dict:
    """Validation pass with a fixed RNG, so the same weights always get the same loss; training RNG untouched."""
    with evaluation_rng(getattr(args, "eval_seed", EVAL_SEED)):
        return run_epoch(model, data, act_loss, args)


@torch.no_grad()
def predict(model, data, args) -> dict:
    """Per test target: activity probabilities and time-head outputs from the position before it, plus labels."""
    model.eval()
    rows = {k: [] for k in ("case", "pos", "type", "dt", "R", "cause", "p_act", "time")}
    cases = data["case"]
    for s in range(0, len(cases), args.batch_size):
        b = {k: data[k][s:s + args.batch_size].to(args.device) for k in KEYS}
        h, out = model(b["types"], b["event_time"])
        if model.model_type == "baseline_act":
            # Δt point prediction, intensity pre-activation linear(h) per activity, prefix end time
            time_out = torch.cat([out[0], model.linear(h), b["event_time"].unsqueeze(-1)], -1)
        else:
            # A3J keeps an activity axis: [B,S,T,1+3K]; ordinary mixtures: [B,S,1+3K].
            time_out = torch.cat([out[0].unsqueeze(-1), out[1]], -1)
        logits = out[-1]
        start = block_start(b["dt"])
        for i, j in b["target_mask"].nonzero().tolist():               # target at j, predicted from j - 1
            rows["case"].append(cases[s + i])
            rows["pos"].append(j)
            rows["type"].append(int(b["types"][i, j]))
            rows["dt"].append(float(b["dt"][i, j]))
            rows["R"].append(b["R"][i, j].cpu())
            rows["cause"].append(int(b["cause"][i, j]))
            p = torch.softmax(logits[i, j - 1], -1)
            if getattr(args, "variant", None) == "A3B":
                p = pre_block_mixture(p, logits[i], out[0][i, j - 1], b["types"][i], int(start[i, j - 1]), j)
            rows["p_act"].append(p.cpu())
            rows["time"].append(time_out[i, j - 1].cpu())
    for k in ("R", "p_act", "time"):
        rows[k] = torch.stack(rows[k])
    if model.model_type == "baseline_act":
        rows["alpha"], rows["beta"] = float(model.alpha), float(model.beta)
    return rows


def pre_block_mixture(p_new, logits, tie_logit, types, s, j):
    """A3B test activity: π₀·p(· | state before the block of event j−1, its emitted members removed) + (1−π₀)·p_new (no loops assumed)."""
    p_in = torch.softmax(logits[max(s - 1, 0)], -1)
    p_in[types[s:j] - 1] = 0
    p_in = p_in / p_in.sum() if p_in.sum() > 0 else p_new
    pi0 = torch.sigmoid(tie_logit)
    return pi0 * p_in + (1 - pi0) * p_new


def save(obj, path: Path) -> None:
    """torch.save via a temporary file and an atomic rename, so a kill never leaves a half-written file."""
    torch.save(obj, path.with_suffix(".tmp"))
    os.replace(path.with_suffix(".tmp"), path)


def rng_state() -> dict:
    """All RNG states, for an exact resume."""
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "mps": torch.mps.get_rng_state() if torch.backends.mps.is_available() else None}


def set_rng_state(s: dict) -> None:
    """Restore the RNG states saved by rng_state."""
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"].cpu())
    if s["cuda"]:
        torch.cuda.set_rng_state_all([state.cpu() for state in s["cuda"]])
    if s.get("mps") is not None:
        torch.mps.set_rng_state(s["mps"].cpu())


@contextmanager
def evaluation_rng(seed=EVAL_SEED):
    """Fixed evaluation draws; restore Python/NumPy/CPU/CUDA/MPS even on failure."""
    state = rng_state()
    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        yield
    finally:
        set_rng_state(state)


def head_init(train: dict) -> dict:
    """Train-target statistics for --init_mean: zero share, mean and std of log Δt over Δt > 0 targets."""
    dt = train["dt"][train["target_mask"]]
    log_pos = dt[dt > 0].log()
    if log_pos.numel() < 2:
        raise ValueError("--init_mean requires at least two positive training targets")
    return {"pi0": float((dt == 0).float().mean()), "log_mean": float(log_pos.mean()), "log_std": float(log_pos.std())}


@torch.no_grad()
def init_mean(model, stats: dict) -> None:
    """--init_mean: biases so the initial output ≈ train statistics: π₀ = π̄₀, equal weights, μ_k = mean log Δt, σ_k = std log Δt."""
    if model.model_type not in ("mixture", "mixture_joint"):
        return                                                     # A0/A2 heads are bias-free THP Predictors: unchanged
    K, pi0 = model.n_mix, min(max(stats["pi0"], 1e-4), 1 - 1e-4)   # clamp: g1_none has no zero gap
    model.tie_head.bias.fill_(math.log(pi0 / (1 - pi0)))
    b = model.time_head.bias.view(-1, 3 * K)                        # one row per activity for A3J
    b[:, :K], b[:, K:2 * K] = 0.0, stats["log_mean"]
    b[:, 2 * K:] = math.log(math.expm1(max(stats["log_std"] - 1e-4, 1e-3)))


class Objective(torch.nn.Module):
    """One replica for StackedModels: (activity, time) loss of a batch from batch_loss, as a [2] tensor."""

    def __init__(self, model, act_loss, n_mix: int, eps=None):
        super().__init__()
        self.model, self.act_loss, self.n_mix, self.eps = model, act_loss, n_mix, eps

    def forward(self, batch: dict, prediction: bool = False):
        l_act, l_time, _ = batch_loss(self.model, batch, self.act_loss, self.n_mix, self.eps)
        return torch.stack((l_act, l_time))


def stacked_epoch(group, data: dict, args, seed=None) -> torch.Tensor:
    """One pass over a split for all replicas (training if seed is given); [N, 2] act and time loss per target."""
    group.train(seed is not None)
    total, n = 0.0, 0
    with torch.set_grad_enabled(seed is not None):
        for b in batches(data, args.batch_size, args.device, shuffle=seed is not None, seed=seed):
            nt = int(b["target_mask"][:, 1:].sum())
            total = total + (group.step(b, nt, args.lam) if seed is not None else group.losses(b).detach())
            n += nt
    return (total / max(n, 1)).cpu()


def train_stacked(args, architecture: dict, splits: tuple, act_loss, config: dict) -> None:
    """--replicas > 1: per-replica Adam, clipping, plateau LR, early stopping and best weights; then one run folder per seed."""
    from model.stacking_local.run import Selection          # lazy: stacking_local imports train.train
    from model.stacking_local.stacked import StackedModels
    assert args.variant != "A3eps", "A3eps: unused tie head has no gradient, which PackedAdam rejects"
    assert args.sched == "plateau", "stacked groups implement the plateau schedule only"
    train, val, test = splits
    seeds = list(range(args.seed, args.seed + args.replicas))
    outs = [args.out / args.log / f"{args.variant}_s{s}" for s in seeds]
    for o in outs:
        assert not any((o / f).exists() for f in ("last.pt", "model.pt", "test_pred.pt")), f"{o}: run exists"
        o.mkdir(parents=True, exist_ok=True)
    models = [Transformer.from_config(architecture, seed=s) for s in seeds]
    for m in models if "head_init" in config else []:
        init_mean(m, config["head_init"])
    group = StackedModels([Objective(m, act_loss, args.n_mix).to(args.device) for m in models])
    sel = Selection(len(seeds), args.patience, args.min_delta)
    sel.lr = [args.lr] * len(seeds)
    group.configure(sel.lr, sel.active)
    config = config | {"replicas": args.replicas, "replica_group": seeds, "git_commit": git_commit()}
    for s, o in zip(seeds, outs):
        (o / "config.json").write_text(json.dumps(config | {"seed": str(s)}, indent=1))
    hists = [(o / "history.csv").open("w", buffering=1) for o in outs]
    for h in hists:
        h.write(HEADER + "\n")
    best, t0 = [None] * len(seeds), time.time()
    for epoch in range(1, args.epochs + 1):
        if not any(sel.active):
            break
        was_active, lr = list(sel.active), list(sel.lr)
        tr = stacked_epoch(group, train, args, seed=args.data_seed + 1000 * epoch)
        with evaluation_rng(args.eval_seed):
            va = stacked_epoch(group, val, args)
        for i in sel.update(epoch, (va[:, 0] + args.lam * va[:, 1]).tolist()):
            best[i] = group.state(i)
        group.configure(sel.lr, sel.active)
        for i, h in enumerate(hists):
            if was_active[i]:                                       # gnorm not returned by StackedModels: nan
                h.write(f"{epoch},{tr[i, 0] + args.lam * tr[i, 1]:.6f},{tr[i, 0]:.6f},{tr[i, 1]:.6f},nan,"
                        f"{va[i, 0] + args.lam * va[i, 1]:.6f},{va[i, 0]:.6f},{va[i, 1]:.6f},{lr[i]:.2e},"
                        f"{time.time() - t0:.1f}\n")
        print(f"epoch {epoch:3d}  val {[round(float(v), 4) for v in va[:, 0] + args.lam * va[:, 1]]}  "
              f"active {sum(sel.active)}/{len(seeds)}", flush=True)
    for h in hists:
        h.close()
    for i, (s, o) in enumerate(zip(seeds, outs)):
        ck = {"model_config": architecture, "model": best[i]}
        save(ck, o / "model.pt")
        save(predict(Transformer.from_checkpoint(ck, device=args.device), test, args), o / "test_pred.pt")
        print(f"{args.log} {args.variant} seed {s}: best val {sel.best[i]:.4f} @ epoch {sel.best_epoch[i]} -> {o}")
    print(f"group of {len(seeds)}: {time.time() - t0:.0f} s", flush=True)


def main():
    """Train with early stopping on validation loss (resuming from last.pt if present), then save test predictions."""
    args = parse_args()
    args.train_seed = stream_seed(args.seed, "train") if args.train_seed is None else args.train_seed
    args.data_seed = stream_seed(args.seed, "data") if args.data_seed is None else args.data_seed
    set_seed(args.train_seed)
    model_type, act_loss = VARIANTS[args.variant]
    train, val, test, vocab = load_log(args.log)
    keys = risky(targets(args.log, "train", vocab)) if args.variant == "A3C" else set()   # from train only
    for d in (train, val, test):
        d["risk"] = risk_mask(d, keys)
    n_targets = int(train["target_mask"].sum())
    size = args.size if args.size != "auto" else ("2k" if n_targets < SIZE_SPLIT else "10k")  # all targets: H-M = M's size
    stats = head_init(train) if args.init_mean else None           # before the hurdle mask: π̄₀ over all targets
    if args.variant in HURDLE:
        pi0_bar = float((train["dt"][train["target_mask"]] == 0).float().mean())   # train zero share, all targets
        for d in (train, val):
            d["target_mask"] &= d["dt"] > 0
    architecture = model_config(args.variant, size, len(vocab), args.n_mix)
    model = Transformer.from_config(architecture, seed=args.seed)
    if stats:
        init_mean(model, stats)
    model = model.to(args.device)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99), eps=1e-5)
    scheduler = (torch.optim.lr_scheduler.StepLR(optimiser, step_size=10, gamma=0.5) if args.sched == "step" else
                 torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, factor=0.5, patience=5,
                                                            threshold=args.min_delta, threshold_mode="abs"))

    out = args.out / args.log / f"{args.variant}_s{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    config = {k: str(v) for k, v in vars(args).items() if k not in ("device", "replicas", "init_mean")} | {
        "size_used": size, "train_targets": n_targets, "model_type": model_type, "act_loss": act_loss.__name__,
        "model_config": architecture, "rng_policy": RNG_POLICY, "device": str(args.device),
        "torch_version": torch.__version__, "python_version": platform.python_version(),
        "numpy_version": np.__version__, "cpu_threads": torch.get_num_threads(), "determinism_warn_only": True,
        "vocab": vocab, "params": sum(p.numel() for p in model.parameters()),
        "split_hashes": json.loads((OUT_DIR / "hashes.json").read_text())[args.log]}
    if args.variant == "A3eps":
        config["eps"] = EPS                                            # only here: older configs stay resumable
    if args.variant in HURDLE:
        config["train_targets_pos"] = int(train["target_mask"].sum())  # only here, as eps
        config["pi0_bar"] = pi0_bar                                    # π̄₀ of the hurdle forecast (train/evaluate.py)
    if args.variant == "A3C":                                          # only here, as eps
        config["risk_contexts"] = len(keys)
        config["risk_targets_train"] = int(train["risk"].sum())
    if stats:
        config["head_init"] = stats                                    # only here: older configs stay resumable
    sources = [*sorted((REPO / "model").glob("*.py")), Path(__file__), REPO / "train/dataset.py"]
    if args.variant == "A3C":
        sources.append(REPO / "train/collapse_risk.py")              # the at-risk rule
    config["source_sha256"] = {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in sources}
    if args.replicas > 1:
        return train_stacked(args, architecture, (train, val, test), act_loss, config)

    start, best, best_epoch, elapsed = 1, float("inf"), 0, 0.0
    if (out / "last.pt").exists():                                     # resume an interrupted run
        old = json.loads((out / "config.json").read_text())
        same = {k: v for k, v in old.items() if k != "git_commit"} == config
        assert same, f"{out}: config differs from the interrupted run; use a new --out directory"
        ck = torch.load(out / "last.pt", map_location="cpu", weights_only=False)   # own file: RNG states
        model.load_state_dict(ck["model"])
        optimiser.load_state_dict(ck["optimiser"])
        scheduler.load_state_dict(ck["scheduler"])
        set_rng_state(ck["rng"])
        start, best, best_epoch, elapsed = ck["epoch"] + 1, ck["best"], ck["best_epoch"], ck["seconds"]
        rows = (out / "history.csv").read_text().splitlines()
        rows = [rows[0]] + [r for r in rows[1:] if int(r.split(",")[0]) < start]   # drop rows after last.pt
        (out / "history.csv").write_text("\n".join(rows) + "\n")
        print(f"resume at epoch {start} (best {best:.4f} @ {best_epoch})", flush=True)
    else:
        config["git_commit"] = git_commit()
        (out / "config.json").write_text(json.dumps(config, indent=1))
        (out / "history.csv").write_text(HEADER + "\n")

    t0 = time.time() - elapsed
    with (out / "history.csv").open("a", buffering=1) as hist:         # line-buffered: each epoch on disk at once
        for epoch in range(start, args.epochs + 1):
            if epoch - best_epoch >= args.patience:                    # resumed after the stopping epoch
                break
            tr = run_epoch(model, train, act_loss, args, optimiser, shuffle_seed=args.data_seed + 1000 * epoch)
            va = evaluate(model, val, act_loss, args)
            lr = optimiser.param_groups[0]["lr"]
            scheduler.step() if args.sched == "step" else scheduler.step(va["loss"])
            if va["loss"] < best - args.min_delta:
                best, best_epoch = va["loss"], epoch
                save(model.checkpoint(), out / "model.pt")
            hist.write(f"{epoch},{tr['loss']:.6f},{tr['act']:.6f},{tr['time']:.6f},{tr['gnorm']:.4f},"
                       f"{va['loss']:.6f},{va['act']:.6f},{va['time']:.6f},{lr:.2e},{time.time() - t0:.1f}\n")
            save(model.checkpoint() | {"optimiser": optimiser.state_dict(),
                        "scheduler": scheduler.state_dict(), "rng": rng_state(), "epoch": epoch, "best": best,
                        "best_epoch": best_epoch, "seconds": time.time() - t0}, out / "last.pt")
            print(f"epoch {epoch:3d}  train {tr['loss']:.4f} (act {tr['act']:.4f} time {tr['time']:.4f} "
                  f"gnorm {tr['gnorm']:.2f})  val {va['loss']:.4f} (act {va['act']:.4f} time {va['time']:.4f})  "
                  f"lr {lr:.1e}  best {best:.4f} @ {best_epoch}", flush=True)
            if epoch - best_epoch >= args.patience:
                break

    model = Transformer.from_checkpoint(torch.load(out / "model.pt", map_location="cpu", weights_only=True),
                                        device=args.device)
    save(predict(model, test, args), out / "test_pred.pt")
    print(f"{args.log} {args.variant} seed {args.seed}: best val {best:.4f} @ epoch {best_epoch}, "
          f"{time.time() - t0:.0f} s, params {config['params']} -> {out}")


if __name__ == "__main__":
    main()
