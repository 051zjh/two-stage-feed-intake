# Two-stage feed-intake estimation: reproducibility package

This package supports the manuscript "Estimating individual pig feed intake through depth vision and virtual feed-pan reconstruction", prepared for submission to *The Innovation*. It contains de-identified analysis inputs, frozen out-of-fold and event-level predictions, and reproducible analysis code.

The two stages were validated on separate datasets. The linkage is controlled propagation of empirical Stage-I endpoint errors through the Stage-II recursion, not a synchronized end-to-end visual feeding trial. Oracle/teacher-forcing outputs are excluded.

## Contents

- `data/stage1/`: 239 representative depth frames, labels, and locked OOF predictions.
- `data/stage2/splits/`: de-identified train/validation/test event tables.
- `data/stage2/frozen_predictions.csv.gz`: selected current-prior and baseline predictions.
- `data/linkage/`: the 1,000 reference simulation errors and 1,298 event-level mean predictions used in Figure 5.
- `src/stage1/verify_oof_metrics.py`: verifies locked Stage-I metrics.
- `src/stage2/reproduce_selection.py`: reruns the Stage-II rate-prior grid.
- `src/linkage/propagation_core.py`: the source numerical recursion with relative data paths.
- `src/reproduce_all.py`: reruns validation selection and all 1,000 linkage replicates, checks them against the locked source outputs, and writes a verification report.

## Reproduce the main results

Use Python 3.10 or newer. The numerical verification does not require a GPU or PyTorch.

```bash
python -m pip install -r requirements.txt
python src/reproduce_all.py
```

For the originating project environment, `bash reproduce.sh` defaults to the project's Python interpreter. Other environments can use `PYTHON=python bash reproduce.sh`.

The script checks the Stage-I OOF metrics and fold assignments, reruns the 48-candidate Stage-II rate-prior search using validation events only, and reruns the Figure-5 simulated linkage with seed 20261008 and 1,000 replicates. It verifies all 13,136 primary predictions against frozen outputs and checks the linked prediction errors against the supplied source data. Reports are written to `results/verification/`.

Expected results are Stage-I MAE 0.7637 kg (239 samples), Stage-II test MAE 0.0514 kg (1,298 visits), simulated-linkage mean MAE 0.0702 kg, and Monte Carlo mean-prediction MAE 0.0577 kg. The two linkage MAEs summarize different quantities.

## Stage-I model code

`src/stage1/train_kfold_diff.py` supplies the source training implementation with relative paths and the original hyperparameters. The 239 supplied frames are the same representative frames used by the original sample construction. The supplied label order reproduces the locked five-fold assignment (seed 42). `export_locked_kfold_oof.py` can export predictions when the five original model checkpoints and fold-score report are supplied.

```bash
python -m pip install -r requirements-stage1.txt
python src/stage1/train_kfold_diff.py
```

The five original model checkpoints are not included. Training is stochastic; the verification uses the supplied frozen OOF predictions and does not claim to have retrained the network or re-created the exact checkpoint weights.

## Event data fields

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

The chronological splits contain 9,074 training, 2,592 validation, 1,298 test, and 172 additional-validation events. Original animal and station identifiers are replaced by ordered pseudonyms. Dates are shifted by the same offset to a neutral origin, preserving durations, event gaps, split membership, and sequence ordering. The identifier map and original calendar dates are excluded. These transformations support de-identification; they are not a claim of formal irreversible anonymization.

## Publication status

This directory is prepared locally. It has not yet been uploaded to GitHub. No public repository URL, manuscript DOI, or release DOI is asserted. The package excludes identifiable raw records, manuscripts, copyrighted reference PDFs, and Oracle/teacher-forcing models. An open-source/data license has not been selected; no additional license grant is implied.
