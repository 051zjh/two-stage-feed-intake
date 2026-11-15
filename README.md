# Two-stage feed-intake estimation

Estimate individual pig feed intake by combining depth-based feed-weight estimation with virtual feed-pan reconstruction. This repository provides the Python implementation, depth-image samples, de-identified feeding-event data, and reference predictions for running the analysis.

## How it works

1. **Depth-based feed-weight estimation.** Stage I uses depth differences relative to an empty-hopper baseline to estimate the weight of visible feed.
2. **Virtual feed-pan reconstruction.** Stage II tracks residual feed between visits and combines feed delivery, visit duration, and a feeding-rate prior to estimate intake for each visit.
3. **Simulated linkage.** Repeated simulations propagate empirical Stage-I measurement errors through Stage II to evaluate their effect on intake estimates.

The two stages are evaluated on separate datasets; simulated linkage connects the visual measurement errors with the event-level intake model.

## Quick start

Use Python 3.10 or newer. The main analysis runs on CPU with the dependencies in `requirements.txt`; PyTorch is needed only for Stage-I model training and inference.

```bash
git clone https://github.com/051zjh/two-stage-feed-intake.git
cd two-stage-feed-intake
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python src/reproduce_all.py
```

The last command performs three steps:

- Calculates Stage-I metrics from the supplied out-of-fold (OOF) predictions and checks sample paths and five-fold assignments.
- Evaluates 48 Stage-II rate-prior candidates on the validation split, then reproduces the selected model and baseline predictions.
- Runs 1,000 simulated-linkage replicates with seed `20261008` and compares the results with the reference outputs.

A successful run prints `"status": "PASS"`. Generated files are saved to:

| Path | Output |
|---|---|
| `results/verification/verification.json` | Metrics and reference-comparison checks |
| `results/verification/reproduced_simulation_metrics.csv` | MAE for each simulated-linkage replicate |
| `results/verification/reproduced_mean_predictions.csv` | Event-level mean predictions and simulation percentiles |
| `results/stage2_reanalysis/` | Candidate selection and Stage-II prediction outputs |

## Reference results

| Analysis | Evaluation data | MAE (kg) |
|---|---|---:|
| Stage I: feed-weight estimation | 239 OOF samples | 0.7637 |
| Stage II: visit-level intake estimation | 1,298 temporal test visits | 0.0514 |
| Simulated linkage: average run-level MAE | 1,000 replicates over the test visits | 0.0702 |
| Simulated linkage: MAE of mean predictions | Predictions averaged per visit across replicates | 0.0577 |

The linkage rows summarize the error of individual simulation runs and the error after averaging predictions for each visit, respectively.

## Repository layout

```text
data/
  stage1/                 Depth frames, empty-hopper baseline, labels, OOF predictions
  stage2/
    splits/               Chronological feeding-event splits
    frozen_predictions.csv.gz
  linkage/                Reference simulation metrics and mean predictions
src/
  stage1/                 Depth-model training, OOF export, metric calculation
  stage2/                 Virtual feed-pan model, rate selection, baselines
  linkage/                Measurement-error propagation and intake simulation
  reproduce_all.py        Main analysis entry point
results/
  verification/           Reference verification reports and reproduced outputs
requirements.txt          Dependencies for the main analysis
requirements-stage1.txt   Additional dependencies for the depth model
```

## Run individual components

Run these commands from the repository root after installing `requirements.txt`.

### Stage-I prediction metrics

```bash
python src/stage1/verify_oof_metrics.py
```

This prints the sample count, R2, MAE, RMSE, and bias calculated from `data/stage1/oof_predictions.csv`.

### Stage-II model selection

```bash
python src/stage2/reproduce_selection.py \
  --data-dir data/stage2/splits \
  --out-dir results/stage2_reanalysis
```

Feeding-rate priors are estimated from training events, and the quantile and multiplier are selected by validation MAE. Test intake labels are used for evaluation. The primary model uses a 30-minute residual-state decay half-life.

### Stage-I model training

```bash
python -m pip install -r requirements-stage1.txt
python src/stage1/train_kfold_diff.py
```

The training script uses the supplied 239 depth frames, feed-weight labels, and empty-hopper baseline. It defaults to a single-channel ResNet-18 and five-fold cross-validation, saving new checkpoints and fold results to `results/stage1_retraining/`.

Custom data and output locations can be set through `FCR_LABEL_CSV`, `FCR_IMG_ROOT`, `FCR_BASELINE_PATH`, and `FCR_KFOLD_DIR`. The original trained checkpoints are not bundled; the main analysis calculates Stage-I reference metrics from the supplied OOF predictions, while this command trains new models.

## Feeding-event data

The Stage-II dataset contains 13,136 events in four chronological splits:

| File in `data/stage2/splits/` | Events |
|---|---:|
| `train.csv` | 9,074 |
| `val.csv` | 2,592 |
| `test.csv` | 1,298 |
| `val_final.csv` | 172 |

The source column names are retained for compatibility with the analysis scripts:

| CSV column | Meaning |
|---|---|
| 测定站 | Pseudonymous station identifier |
| 电子耳牌 | Pseudonymous animal identifier |
| 日期 | Shifted visit date |
| 开始时间 / 结束时间 | Visit start/end clock time |
| 料槽初重 / 料槽末重 | Recorded starting/ending feed weight (kg) |
| 采食量 | Reference intake (kg) |
| 体重 | Recorded body weight (kg) |
| 投料量 | Event delivery (kg) |

Animal and station identifiers are replaced with pseudonyms, and a common date shift preserves event order, visit durations, and gaps between visits. Original identifiers, identifier mappings, and unshifted calendar dates are not distributed.

## License

A software and data license has not yet been specified.
