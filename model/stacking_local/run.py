"""Train replicas or compare serial/stacked updates. Run as a Python module."""
import argparse
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import platform
import statistics
import time

import torch
import numpy as np

from train import dataset
from train.train import (EVAL_SEED, KEYS, RNG_POLICY, SIZES, evaluation_rng,
                         rng_state, save, set_rng_state, set_seed, stream_seed)
from .stacked import STACKABLE_VARIANTS, StackedModels, make_models


class SerialModels:
    """Independent PyTorch Adam runs sharing the same uploaded minibatch."""

    def __init__(self, models):
        self.models, self.n = models, len(models)
        self.optimisers = [torch.optim.Adam(m.parameters(), lr=.002, betas=(.9, .99),
                                             eps=1e-5) for m in models]
        self.active = [True] * self.n

    def train(self, enabled):
        for model in self.models:
            model.train(enabled)

    def losses(self, batch):
        return torch.stack([model(batch) for model in self.models])

    def predictions(self, batch):
        outputs = [model(batch, True) for model in self.models]
        return tuple(torch.stack(parts) for parts in zip(*outputs))

    def step(self, batch, targets, lam=1.):
        losses = []
        for model, optimiser, active in zip(self.models, self.optimisers, self.active):
            value = model(batch)
            if active:
                optimiser.zero_grad()
                ((value[0] + lam * value[1]) / max(targets, 1)).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimiser.step()
            losses.append(value.detach())
        return torch.stack(losses)

    def configure(self, learning_rates, active):
        self.active = list(active)
        for optimiser, lr in zip(self.optimisers, learning_rates):
            optimiser.param_groups[0]["lr"] = lr

    def state(self, replica=None):
        if replica is None:
            states = [self.state(i) for i in range(self.n)]
            return {name: torch.stack([state[name] for state in states]) for name in states[0]}
        return {name: value.detach().cpu().clone()
                for name, value in self.models[replica].model.state_dict().items()}

    def restore(self, states):
        if isinstance(states, dict):
            states = [{name: value[i] for name, value in states.items()} for i in range(self.n)]
        for model, state in zip(self.models, states):
            model.model.load_state_dict(state)

    def checkpoint(self):
        return {"model": self.state(), "optimisers": [opt.state_dict() for opt in self.optimisers]}

    def load_checkpoint(self, packet):
        self.restore(packet["model"])
        for opt, state in zip(self.optimisers, packet["optimisers"], strict=True):
            opt.load_state_dict(state)


class Selection:
    """Per-replica best weights, patience 24 and absolute-threshold plateau LR.

    The LR rule matches ReduceLROnPlateau(factor=.5, patience=5,
    threshold=1e-4, threshold_mode='abs', eps=1e-8).
    """

    def __init__(self, n, patience=24, min_delta=1e-4):
        self.best, self.best_epoch = [float("inf")] * n, [0] * n
        self.lr, self.active, self.bad_epochs = [.002] * n, [True] * n, [0] * n
        self.patience, self.min_delta = patience, min_delta

    def update(self, epoch, values):
        improved = []
        for i, value in enumerate(values):
            if not self.active[i]:
                continue
            if value < self.best[i] - self.min_delta:
                self.best[i], self.best_epoch[i], self.bad_epochs[i] = value, epoch, 0
                improved.append(i)
            else:
                self.bad_epochs[i] += 1
            if self.bad_epochs[i] > 5:
                if self.lr[i] / 2 > 1e-8:
                    self.lr[i] /= 2
                self.bad_epochs[i] = 0
            self.active[i] = epoch - self.best_epoch[i] < self.patience
        return improved


def load_data(path, log, evaluation_split="test"):
    # The existing loader has a module-level path; confine the override here.
    previous = dataset.OUT_DIR
    try:
        dataset.OUT_DIR = path
        dataset.check_hashes(log)
        vocab = dataset.vocabulary(log)
        train = dataset.load_split(log, "train", vocab)
        val = dataset.load_split(log, "val", vocab)
        held_out = val if evaluation_split == "val" else dataset.load_split(log, "test", vocab)
        return train, val, held_out, vocab
    finally:
        dataset.OUT_DIR = previous


def batches(data, batch_size, device, seed=None):
    n = len(data["case"])
    order = (torch.arange(n, device="cpu") if seed is None else
             torch.randperm(n, generator=torch.Generator().manual_seed(seed), device="cpu"))
    for indices in order.split(batch_size):
        batch = {key: data[key][indices] for key in KEYS}
        targets = int(batch["target_mask"][:, 1:].sum())
        yield {key: value.to(device) for key, value in batch.items()}, targets


def measure(call, device):
    if device == "mps":
        torch.mps.synchronize()
    start = time.perf_counter()
    result = call()
    if device == "mps":
        torch.mps.synchronize()
    return result, time.perf_counter() - start


def build(models, mode, device):
    copies = [deepcopy(model).to(device) for model in models]
    return StackedModels(copies) if mode == "stacked" else SerialModels(copies)


def epoch(executor, data, args, training=False, seed=None):
    executor.train(training)
    total = torch.zeros(executor.n, 2, device=args.device)
    count = 0
    with torch.set_grad_enabled(training):
        for batch, targets in batches(data, args.batch_size, args.device, seed):
            total += (executor.step(batch, targets, args.lam) if training else
                      executor.losses(batch))
            count += targets
    result = (total / max(count, 1)).cpu()
    if not torch.isfinite(result).all():
        raise FloatingPointError("Nonfinite training or validation loss")
    return result


@torch.no_grad()
def predict(executor, data, args):
    executor.train(False)
    probabilities, times = [], []
    for batch, _ in batches(data, args.batch_size, args.device):
        p, t = executor.predictions(batch)
        mask = batch["target_mask"][:, 1:].cpu()
        probabilities.append(p[:, :, :-1].cpu()[:, mask])
        times.append(t[:, :, :-1].cpu()[:, mask])
    mask = data["target_mask"]
    positions = mask.nonzero().tolist()
    return {"p_act": torch.cat(probabilities, 1), "time": torch.cat(times, 1),
            "type": data["types"][mask], "dt": data["dt"][mask], "R": data["R"][mask],
            "cause": data["cause"][mask], "case": [data["case"][i] for i, j in positions],
            "pos": [j for i, j in positions]}


def validation_report(executor, data, args, selection, components):
    """Descriptive seed variation on validation-selected weights, without test scoring."""
    with evaluation_rng(args.eval_seed):
        checked = epoch(executor, data, args)
        predictions = predict(executor, data, args)
    torch.testing.assert_close(checked, components, atol=3e-5, rtol=3e-4)
    p = predictions["p_act"]
    chosen = p.argmax(-1)
    accuracy = chosen + 1 == predictions["type"]
    ambiguous = predictions["R"].sum(-1) > 1
    hit = predictions["R"].unsqueeze(0).expand(executor.n, -1, -1).gather(
        -1, chosen.unsqueeze(-1)).squeeze(-1)
    metrics = {"validation_loss": checked[:, 0] + args.lam * checked[:, 1],
               "activity_loss": checked[:, 0], "time_loss": checked[:, 1],
               "file_accuracy": accuracy.float().mean(1), "hit_R": hit.float().mean(1)}
    if ambiguous.any():
        metrics["ambiguous_file_accuracy"] = accuracy[:, ambiguous].float().mean(1)
        metrics["ambiguous_hit_R"] = hit[:, ambiguous].float().mean(1)
    if args.variant in ("A1", "A3", "A3J"):
        zero = predictions["time"][..., 0].sigmoid()
        if args.variant == "A3J":
            zero = (p * zero).sum(-1)
        metrics["zero_brier"] = (zero - (predictions["dt"] == 0).float()).square().mean(1)
        metrics["mean_zero_probability"] = zero.mean(1)
    rows = [{"seed": args.seed + i, "best_epoch": selection.best_epoch[i],
             **{name: value[i].item() for name, value in metrics.items()}} for i in range(executor.n)]
    with (args.out / "validation.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    summary = {name: {"mean": value.double().mean().item(),
                      "std": value.double().std().item() if executor.n > 1 else 0.,
                      "median": value.double().quantile(.5).item(),
                      "min": value.min().item(), "max": value.max().item()}
               for name, value in metrics.items()}
    return {"split": "val", "seeds": executor.n, "per_seed": rows, "summary": summary,
            "targets": len(predictions["type"]), "observed_zero_fraction": (predictions["dt"] == 0).float().mean().item(),
            "scope": "validation-selected checkpoints; seed variation conditional on one generated log and shared batch order; file labels are observed order, not latent truth"}


def train(args, models, splits, config):
    training, validation, held_out = splits
    executor = build(models, args.mode, args.device)
    selection = Selection(args.replicas, args.patience, args.min_delta)
    best, components, start = None, torch.full((args.replicas, 2), float("inf")), 1
    if args.resume:
        packet = torch.load(args.out / "last.pt", map_location="cpu", weights_only=False)
        if packet["config"] != config:
            raise ValueError("Checkpoint configuration differs; use a new output directory")
        executor.load_checkpoint(packet["executor"])
        selection.__dict__.update(packet["selection"])
        executor.configure(selection.lr, selection.active)
        best, components, start = packet["best"], packet["best_components"], packet["epoch"] + 1
        rows = [json.loads(line) for line in (args.out / "history.jsonl").read_text().splitlines()]
        (args.out / "history.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows if row["epoch"] < start))
        set_rng_state(packet["rng"])
        print(f"resume at epoch {start}", flush=True)
    for number in range(start, args.epochs + 1):
        if not any(selection.active):
            break
        losses, train_seconds = measure(
            lambda: epoch(executor, training, args, True, args.data_seed + 1000 * number), args.device)
        with evaluation_rng(args.eval_seed):
            val, val_seconds = measure(lambda: epoch(executor, validation, args), args.device)
        values = (val[:, 0] + args.lam * val[:, 1]).tolist()
        improved = selection.update(number, values)
        if improved:
            current = executor.state()
            if best is None:
                best = current
            else:
                for name in best:
                    best[name][improved] = current[name][improved]
            components[improved] = val[improved]
        executor.configure(selection.lr, selection.active)
        row = {"epoch": number, "train": losses.tolist(), "validation": val.tolist(),
               "train_seconds": train_seconds, "validation_seconds": val_seconds,
               "lr": selection.lr, "active": selection.active, "best_epoch": selection.best_epoch}
        with (args.out / "history.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        if number == 1 or number % args.checkpoint_every == 0 or number == args.epochs or not any(selection.active):
            save({"config": config, "epoch": number, "executor": executor.checkpoint(),
                  "selection": deepcopy(vars(selection)), "best": best, "best_components": components,
                  "rng": rng_state()}, args.out / "last.pt")
        print(f"epoch {number:3d} train {statistics.mean([a + args.lam * t for a, t in losses.tolist()]):.5f} "
              f"val {statistics.mean(values):.5f} std {statistics.stdev(values) if len(values) > 1 else 0.:.5f} "
              f"active {sum(selection.active)}/{executor.n} train {train_seconds:.2f}s val {val_seconds:.2f}s", flush=True)
        if not any(selection.active):
            break
    save({"model_config": models[0].model.config, "model": best,
          "seeds": config["randomness"]["initialization_seeds"], "best_epoch": selection.best_epoch,
          "validation_components": components, "vocab": config["vocab"],
          "split_hashes": config["split_hashes"]}, args.out / "best.pt")
    executor.restore(best)
    if args.evaluation_split == "val":
        return validation_report(executor, validation, args, selection, components)
    predictions = predict(executor, held_out, args)
    if args.variant in ("A0", "A2"):
        for key in ("alpha", "beta"):
            predictions[key] = best[key]
    save(predictions, args.out / "predictions.pt")
    accuracy = (predictions["p_act"].argmax(-1) + 1 == predictions["type"]).float().mean(1)
    return {"accuracy": accuracy.tolist(), "mean": accuracy.mean().item(),
            "std": accuracy.std().item() if args.replicas > 1 else None,
            "best_validation": selection.best, "best_epoch": selection.best_epoch}


def benchmark(args, models, training):
    batch, targets = next(batches(training, args.batch_size, args.device))
    samples = []
    for repeat in range(args.repeats):
        for mode in (("serial", "stacked") if repeat % 2 == 0 else ("stacked", "serial")):
            executor = build(models, mode, args.device)  # fresh weights AND Adam state
            executor.train(True)
            set_seed(args.train_seed)
            for _ in range(5):
                executor.step(batch, targets, args.lam)

            def updates(executor=executor):
                for _ in range(args.steps):
                    loss = executor.step(batch, targets, args.lam)
                return loss

            loss, seconds = measure(updates, args.device)
            if not torch.isfinite(loss).all():
                raise FloatingPointError("Nonfinite benchmark loss")
            samples.append({"repeat": repeat, "mode": mode, "seconds_per_step": seconds / args.steps})
            del updates, executor
    medians = {mode: statistics.median(s["seconds_per_step"] for s in samples if s["mode"] == mode)
               for mode in ("serial", "stacked")}
    return {"samples": samples, "median_seconds_per_step": medians,
            "speedup": medians["serial"] / medians["stacked"],
            "scope": "resident batch; updates only; five warmups excluded; fresh state per repeat"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("train", "bench"))
    parser.add_argument("--log", required=True)
    parser.add_argument("--data", type=Path, default=dataset.OUT_DIR)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--variant", choices=STACKABLE_VARIANTS, default="A3")
    parser.add_argument("--size", choices=SIZES, default="2k")
    parser.add_argument("--model-config", type=Path, help="JSON dictionary of exact Transformer constructor arguments")
    parser.add_argument("--mode", choices=("serial", "stacked"), default="stacked")
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--replicas", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=24)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--evaluation-split", choices=("val", "test"), default="val")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0, help="first CPU initialization seed; replicas use seed+i")
    parser.add_argument("--train-seed", type=int, help="group dropout/MC stream; derived from --seed by default")
    parser.add_argument("--data-seed", type=int, help="shared batch-order stream; derived from --seed by default")
    parser.add_argument("--eval-seed", type=int, default=EVAL_SEED)
    parser.add_argument("--lam", type=float, default=1.)
    args = parser.parse_args()
    if args.variant == "A3J" and args.lam != 1:
        parser.error("A3J is one joint likelihood and requires --lam 1")
    args.train_seed = stream_seed(args.seed, "train") if args.train_seed is None else args.train_seed
    args.data_seed = stream_seed(args.seed, "data") if args.data_seed is None else args.data_seed
    if min(args.replicas, args.batch_size, args.epochs, args.steps, args.repeats, args.patience, args.checkpoint_every) < 1:
        parser.error("counts must be positive")
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is unavailable; no CPU substitution")
    if args.resume and (args.action != "train" or not (args.out / "last.pt").exists()):
        parser.error("--resume requires an existing training last.pt")
    args.out.mkdir(parents=True, exist_ok=args.resume)
    torch.set_num_threads(1)
    set_seed(args.train_seed)
    sources = [Path(__file__), Path(__file__).with_name("stacked.py"),
               *sorted(Path("model").glob("*.py")), Path("train/train.py"), Path("train/dataset.py")]
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items() if key != "resume"} | {
                          "torch": torch.__version__, "python": platform.python_version(), "numpy": np.__version__,
                          "dtype": "float32", "cpu_threads": 1, "rng_policy": RNG_POLICY,
                          "determinism_warn_only": True,
                          "randomness": {"initialization_seeds": list(range(args.seed, args.seed + args.replicas)),
                                         "execution_seed": args.train_seed, "evaluation_seed": args.eval_seed,
                                         "batch_order": "local CPU Generator(data_seed + 1000*epoch)",
                                         "vmap_randomness": "different; no chunking; stopped rows retained",
                                         "scope": "fixed mode/device/group size/order; no per-replica private streams"},
                          "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
    try:
        tr, va, te, vocab = load_data(args.data, args.log, args.evaluation_split)
        config["vocab"] = vocab
        config["split_hashes"] = json.loads((args.data / "hashes.json").read_text())[args.log]
        architecture = json.loads(args.model_config.read_text()) if args.model_config else None
        models = make_models(args.variant, args.size, range(args.seed, args.seed + args.replicas), len(vocab), config=architecture)
        config["model_config"] = models[0].model.config
        config["parameters"] = sum(p.numel() for p in models[0].parameters())
        config["train_targets"] = int(tr["target_mask"].sum())
        config["validation_targets"] = int(va["target_mask"].sum())
        if args.resume and json.loads((args.out / "config.json").read_text()) != config:
            raise ValueError("Configuration/source/data/backend differs; resume refused without changing artifacts")
        (args.out / "config.json").write_text(json.dumps(config, default=str, indent=2) + "\n")
        result = (train(args, models, (tr, va, te), config) if args.action == "train" else
                  benchmark(args, models, tr))
        result["status"] = "completed"
    except Exception as error:
        if not args.resume:
            (args.out / "result.json").write_text(json.dumps({"status": "failed", "error": repr(error)}))
        raise
    (args.out / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "per_seed"}), flush=True)


if __name__ == "__main__":
    main()
