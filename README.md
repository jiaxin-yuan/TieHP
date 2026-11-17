# TieHP

Code, trained runs and raw results for the paper *TieHP* (CAiSE 2027 submission). TieHP is a marked temporal point
process for next-event prediction when several events share one timestamp. These events form a **tie block**, and the
log does not record their order. TieHP trains activity prediction with a set likelihood over the remaining block
members. At at-risk activity prefixes, where the set likelihood can collapse, it uses cross-entropy instead.

The repository has two guides: [Guide 1](#guide-1-use-tiehp-on-your-own-log) shows how to apply TieHP to your own
event log, and [Guide 2](#guide-2-reproduce-the-paper) shows how to reproduce every table and figure in the paper.

## Contents

```
environment.yml, .devcontainer/        pinned environment (conda or dev container)
sim/simulate_g1.py                     simulated G1 log + truth file (true order) with one zero-gap mechanism
data/prepare_data.py                   strict temporal split + prefixes of sim/out/*.csv -> data/out/
data/tie_meta.py                       tie metadata (block, remaining members R, ambiguous, cause) + split hashes
data/prepare_real.py                   BPIC15_1-5 XES -> same split + tie metadata -> data/out/
model/                                 THP-M backbone (model.py, Layers.py) and all losses (losses.py)
model/stacking_local/                  trains several initialization seeds as one vmapped group (--replicas)
train/train.py                         one run: one log, one variant, one seed
train/launch.py                        grid of runs (logs x variants x seeds), skips finished runs
train/evaluate.py                      test metrics per run and subset -> results*/<log>/metrics.csv
train/stats.py                         paired case bootstrap (95% CI) of the comparisons -> results*/stats_<config>.csv
train/all0.py                          all-0 baseline (Δ = 0, train-frequency activity)
train/collapse_risk.py, leakage.py     collapse analysis of THP-M+Set (gen seeds 0-2)
train/collapse_table.py                Table 1 (collapse, gen seeds 3-5)
train/table_real_gains.py              Table 2 (BPIC15 gains)
train/plot_results.py                  Fig. 6 (simulated logs)
baselines/run_{sutran,seplstm,processtransformer}.py   PPM baselines on our split files
eda/                                   zero-gap and tie-block statistics of real logs, Fig. 1 and Fig. 4
runs_v3/, runs_a3c_10k/, runs_real/    all trained runs used in the paper (see Guide 2)
results/, results_real/, benchmark-results/   raw results behind every table and figure
data/out/hashes.json                   SHA-256 of every split file used in the paper
```

### Variant names: code vs paper

| code | paper | activity loss |
|---|---|---|
| `A0` | THP | THP (with LSTM) |
| `A1` | THP-M (no LSTM) | cross-entropy |
| `A3` | THP-M+Set | set likelihood |
| `HA3` | Hurdle+Set | set likelihood + hurdle on P(Δ=0) |
| `A3C` | **TieHP** | set likelihood, cross-entropy at at-risk activity prefixes |
| `ALL0`, `SEPLSTM`, `PT`, `SUTRAN` | all-0, SEP-LSTM, ProcessTransformer, SuTraN | baselines |

`train/train.py` also has variants that the paper does not use (`A2`, `A3J`, `A3eps`, `HA1`, `A3M`, `A3B`).

## Setup

Either use conda:

```bash
conda env create -f environment.yml     # env "bp26": Python 3.11, versions pinned in .devcontainer/requirements.txt
conda activate bp26
```

Or open the folder in VS Code and choose *Reopen in Container* (`.devcontainer/`, GPU variant in
`.devcontainer/gpu-server/`). Every command below runs from the repository root.
The paper's runs used Python 3.11.10, torch 2.5.1+cu124 and numpy 2.4.2 (recorded in each run's `config.json`).
A CPU is enough for Guide 1. The paper's grids were trained on a GPU.

Tests of the losses: `python -m unittest model.test_block_set model.test_pre_block`.

## Guide 1: use TieHP on your own log

1. **Log format.** Use one CSV with the columns `CaseID, Activity, Timestamp` (any format pandas can parse). Inside a
   tie block, the rows stay in file order. Copy it to `sim/out/<name>.csv`. The pipeline reads every `*.csv` in
   `sim/out/` except `*_truth.csv`, so leave out logs you do not need.
2. **Split and tie metadata.**
   ```bash
   python data/prepare_data.py     # strict temporal split (last 20% of cases by start = test) -> data/out/
   python data/tie_meta.py         # adds block_id, remaining_block_members (R), ambiguous, zero_gap_cause; writes hashes
   ```
3. **Train TieHP.**
   ```bash
   python -m train.train --log <name> --variant A3C --seed 0 --out runs
   ```
   Training detects the at-risk activity prefixes from the training split. They are stored as `risk_contexts` in
   `config.json`. Use `--variant A1` for THP-M and `--variant A0` for THP. `--replicas 3` trains seeds 0-2 together.
4. **Predict and score.**
   ```bash
   python -m train.evaluate runs <name> --out results_own    # -> results_own/<name>/metrics.csv
   ```
   Without `sim/out/<name>_truth.csv`, the recorded next activity is the label, so the `*_true` columns are scored
   against the logged label.

Output in `runs/<name>/A3C_s0/`: `config.json` (all settings, split hashes, versions), `history.csv`, `train.log`,
`model.pt` (best validation epoch) and `test_pred.pt`. The keys of `test_pred.pt` are one entry per test target, in a
fixed order:
`case`, `pos`, `type` (logged next activity), `dt` (logged Δ), `R` (remaining block members, multi-hot), `cause`,
`p_act` (predicted activity distribution) and `time` (predicted Δ quantities).

## Guide 2: reproduce the paper

All trained runs are included, so every table and figure can be rebuilt from step 3 on without retraining. Steps
1-2 recreate the inputs. Each run's `config.json` stores the SHA-256 of its split files, and `train.dataset` checks
these hashes before training.

### 1. Data

Simulated logs G1 (10,000 cases each): `none`, `batch10/30/50` and `date10/30/50` with generation seeds 0-2.
`none`, `batch30/50` and `date30/50` also use seeds 3-5.

```bash
for s in 0 1 2; do
  python sim/simulate_g1.py --mech none --cases 10000 --seed $s
  for m in batch date; do for q in 0.1 0.3 0.5; do
    python sim/simulate_g1.py --mech $m --share $q --cases 10000 --seed $s; done; done
done
for s in 3 4 5; do
  python sim/simulate_g1.py --mech none --cases 10000 --seed $s
  for m in batch date; do for q in 0.3 0.5; do
    python sim/simulate_g1.py --mech $m --share $q --cases 10000 --seed $s; done; done
done
python data/prepare_data.py && python data/tie_meta.py
```

Real logs: download BPIC15_1-5 (https://doi.org/10.4121/uuid:31a308ef-c844-48da-948c-305d167a0ec1) as
`raw_datasets/BPIC15_{1..5}.xes`, then run `python data/prepare_real.py`. Run it **after** `tie_meta.py`, because it
merges its hashes into `data/out/hashes.json` and `tie_meta.py` rewrites that file. Compare the result with the
`data/out/hashes.json` shipped here.

### 2. Training (optional; the runs are included)

All TieHP/THP runs use the same settings (identical in every `config.json`): Adam lr 0.002, batch 32, ≤ 300 epochs,
early stopping with patience 24 (min Δ 1e-4), plateau scheduler, λ = 1, 5 mixture components, 2 layers, 4 heads,
d_model 48, d_inner 96, d_k = d_v = 12, dropout 0.1, evaluation seed 12345, initialization seeds 0-2.

```bash
L02=$(for c in none batch10 batch30 batch50 date10 date30 date50; do for s in 0 1 2; do echo -n "g1_${c}_10k_s$s "; done; done)
H02=$(for c in batch30 batch50 date30 date50; do for s in 0 1 2; do echo -n "g1_${c}_10k_s$s "; done; done)
T35=$(for c in batch30 batch50 date30 date50; do for s in 3 4 5; do echo -n "g1_${c}_10k_s$s "; done; done)
N35="g1_none_10k_s3 g1_none_10k_s4 g1_none_10k_s5"
REAL="BPIC15_1 BPIC15_2 BPIC15_3 BPIC15_4 BPIC15_5"

# gen seeds 0-2 (Fig. 6, collapse analysis): 288 runs
python -m train.launch --logs $L02 --variants A0 A1 A3 A3C --seeds 0 --out runs_v3 -- --replicas 3
python -m train.launch --logs $H02 --variants HA3        --seeds 0 --out runs_v3 -- --replicas 3
# gen seeds 3-5 (Table 1, Q2): 126 runs
python -m train.launch --logs $T35 --variants A1 A3 A3C --seeds 0 --out runs_a3c_10k -- --replicas 3
python -m train.launch --logs $N35 --variants A0 A3C    --seeds 0 --out runs_a3c_10k -- --replicas 3
# BPIC15 (Table 2): 80 runs
python -m train.launch --logs $REAL --variants A0 A3C --seeds 0 1 2 --out runs_real
python -m train.all0 --out runs_real $REAL
for l in $REAL; do for s in 0 1 2; do for b in sutran seplstm processtransformer; do
  python -m baselines.run_$b --log $l --seed $s --out runs_real; done; done; done
```

Baselines run their official code (SuTraN, and SEP-LSTM from the SuTraN repository) or a PyTorch port of it
(ProcessTransformer) on our split files, with the authors' settings. Each script's header lists its deviations.
Clone the pinned code first:

```bash
git clone https://github.com/BrechtWts/SuffixTransformerNetwork baselines/ext/SuffixTransformerNetwork
git -C baselines/ext/SuffixTransformerNetwork checkout f120e6f7067302b0b5ef26f043322ee2619e07a4
git clone https://github.com/Zaharah/processtransformer baselines/ext/processtransformer
git -C baselines/ext/processtransformer checkout 3f041e4eae9a2efcea0da02166e98efaa953801c
```

### 3. Metrics and statistics

```bash
python -m train.evaluate runs_v3 $L02                                    # -> results/<log>/metrics.csv
python -m train.evaluate runs_a3c_10k $T35 $N35 --out results/a3c_10k
python -m train.evaluate runs_real $REAL --out results_real
for c in none batch10 batch30 batch50 date10 date30 date50; do
  python -m train.stats runs_v3 g1_${c}_10k_s0 g1_${c}_10k_s1 g1_${c}_10k_s2; done
for c in batch30 batch50 date30 date50 none; do
  python -m train.stats runs_a3c_10k g1_${c}_10k_s3 g1_${c}_10k_s4 g1_${c}_10k_s5 --out results/a3c_10k; done
for l in $REAL; do python -m train.stats runs_real $l --out results_real; done
```

`train.stats` runs a paired case bootstrap. All targets of a test case are resampled together, every method is
scored on the same resamples, and the output gives the gain with its 95% CI.

### 4. Paper item -> file -> command

| paper | file | command |
|---|---|---|
| Fig. 1 (zero gaps in real logs) | `benchmark-results/fig_intro_zero_gaps.pdf` from `t1_zero_gap_causes.csv`, `t5_tie_blocks.csv` | `python eda/fig_intro_zero_gaps.py --fig benchmark-results/fig_intro_zero_gaps.pdf` |
| Fig. 4 (collapse example) | `benchmark-results/fig_collapse_batch.pdf` | `python eda/fig_collapse_batch.py --fig benchmark-results/fig_collapse_batch.pdf` |
| Fig. 6 (simulated logs, gen seeds 0-2) | `results/fig_g1_10k_paper.png`, `results/stats_g1_*_10k.csv` | `python train/plot_results.py --logs $L02 --variants A0 A1 A3 A3C HA3 --paper --fig results/fig_g1_10k_paper.png` |
| Q2 collapse, gen seeds 0-2 | `results/collapse_risk_10k.txt` | `python -m train.collapse_risk runs_v3 g1_{batch,date}{10,30,50}_10k_s{0,1,2}` (18 logs with tie blocks) |
| Table 1 (collapse, gen seeds 3-5) | `results/a3c_10k/collapse_table.txt`; Brier columns = mean `brier_true` (subset `ambiguous`) in `results/a3c_10k/g1_*/metrics.csv` | `python -m train.collapse_table` |
| Table 2 (BPIC15 gains) | `results_real/table_real_gains.tex` from `results_real/stats_BPIC15_*.csv` | `python -m train.table_real_gains` |
| Per-log results (all metrics, subsets) | `results*/<log>/metrics.csv` | step 3 |

`t1_zero_gap_causes.csv` and `t5_tie_blocks.csv` come from `eda/zero_gap_causes.py` and `eda/tie_blocks.py`, run on
public logs placed in `raw_datasets/` (BPIC12, BPIC15_1-5, BPIC17, BPIC20, Sepsis, Helpdesk, ...). Fig. 4 uses a
2,000-case log, which you create with `python sim/simulate_g1.py --mech batch --share 0.3 --seed 0`.

### Known limitations

- Table 1, `train.collapse_risk` and `train.evaluate` need the truth files `sim/out/<log>_truth.csv` (Guide 2, step 1).
  Without them, the true next activity silently falls back to the logged label, which gives different collapse shares.
- Bitwise equality across hardware or torch versions is not guaranteed. The metrics in `results*/` come from the
  included `test_pred.pt` files.
