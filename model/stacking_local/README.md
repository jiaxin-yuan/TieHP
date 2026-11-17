# Independent-model stacking

Two Python files:

- `stacked.py`: original model/loss adapter, packed independent Adam, and vmap.
- `run.py`: serial reference, training, validation selection, export and benchmark CLI.

A0–A3 and A3J reuse `model/` and `train/dataset.py`. Each replica owns all weights,
clipping, Adam moments/clock, LR, stopping and validation-selected weights.
Losses are normalized per target and **summed** across replicas. Predictions
retain a model axis; each replica is scored separately.

A3J uses the trainer's joint objective (λ=1), with an activity axis in its time
predictions and a marginal zero probability in the validation report. The
standalone CLI accepts A0–A3/A3J; use `train.train` for the hurdle variants and
serial A3eps. `Objective` delegates loss calculation to the trainer, so both
entry points use the same target alignment and loss decomposition.

## Run from the repository root

Prepare inputs with `python data/prepare_data.py`, then `python data/tie_meta.py`.

```sh
PYTORCH_ENABLE_MPS_FALLBACK=0 python -B -m model.stacking_local.run train \
  --data data/out --log g1_batch30_s0 \
  --device mps --replicas 3 --out runs/stack_train

PYTORCH_ENABLE_MPS_FALLBACK=0 python -B -m model.stacking_local.run bench \
  --data data/out --log g1_batch30_s0 \
  --device mps --replicas 3 --out runs/stack_bench
```

Training supports
`--mode serial|stacked`; benchmarking compares both. Every repeat starts with
fresh identical weights and optimizer state, excludes five warmups, alternates
method order and synchronizes timing boundaries. The benchmark measures
resident-batch updates. Serial execution shares the uploaded batch too.

Outputs use a **new directory**: configuration/source/split hashes, result JSON,
and (for training) epoch history, grouped checkpoints and a validation report.
Multi-GPU execution, mixed precision,
time-distribution scoring and independent per-replica data order are outside
this prototype. Adam settings match the existing trainer. Stochastic draws
differ between serial and vmap execution; identical trajectories are not claimed.

`--seed` gives initialization seeds `seed+i`. `--train-seed` controls the group
dropout/Monte Carlo stream; `--data-seed` controls shared batch order. Both default
to separate stable seeds derived from `--seed`. `--eval-seed` defaults to 12345;
the shared trainer RNG context preserves Python/NumPy/CPU/CUDA/MPS training state.
Configuration records all resolved seeds and the ordered replica group. Repeating
the same mode, backend, grouping, batch order and software is the reproducibility
scope. `vmap(randomness="different")` supplies different draws per replica, without
private seed-addressable streams: regrouping/reordering, changing group size or
switching serial/stacked execution changes per-replica draws. Stopped rows remain
in the group. Historical runs use the previous policy and remain unchanged.

Training defaults to validation reporting (`--evaluation-split val`); test evaluation
requires `--evaluation-split test`. `validation.csv` has one row per seed at its own
selected epoch; `result.json` includes mean, sample standard deviation, median and
range. These describe initialization/execution variation on one generated log and
shared batch order. File-label accuracy uses observed order; hit@R uses the candidate
set. Neither measures recovery of the simulator's latent true order.

`--model-config architecture.json` supplies the exact `Transformer` constructor
dictionary, checked against the variant and vocabulary. The original size presets
remain available. `--epochs 100 --patience 101` keeps every seed training for all
100 epochs; learning-rate schedules and best selection remain independent.

`last.pt` contains current grouped weights, Adam state, selection, best weights,
configuration, epoch and all-device RNG states. It is replaced atomically at epoch 1,
every `--checkpoint-every` epochs (default 5), and the final epoch. Resume with the
identical command plus `--resume`; changed source/data/software/backend/grouping
is rejected. History after the last durable checkpoint is discarded on resume.
`best.pt` is written once at completion and holds each seed's selected weights in
named tensors with a leading replica axis. Only these two checkpoint files remain;
no per-epoch or per-seed checkpoint archive is created.

Extract one ordinary inference model from the group:

```python
packet = torch.load("runs/stack_train/best.pt", map_location="cpu", weights_only=True)
i = packet["seeds"].index(17)
model = Transformer.from_checkpoint({
    "model_config": packet["model_config"],
    "model": {name: value[i] for name, value in packet["model"].items()},
}).eval()
```

The implementation follows the version-matched
[functional-call contract](https://docs.pytorch.org/docs/2.9/generated/torch.func.functional_call.html),
[vmap randomness contract](https://docs.pytorch.org/docs/2.9/generated/torch.vmap.html)
and [Adam update](https://docs.pytorch.org/docs/2.9/generated/torch.optim.Adam.html).
