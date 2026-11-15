#!/usr/bin/env python3
"""Export five-fold out-of-fold predictions from the locked DiffResNet checkpoints."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
import torch
from torch.utils.data import DataLoader

from train_kfold_diff import DiffResNet, FeedDiffDataset, prepare_all_samples


EXPERIMENTS = ["Exp01", "Exp02", "Exp03"]
TARGET_SCALE_KG = 50.0


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    return {
        "n": int(len(y_true)),
        "r": float(pearsonr(y_true, y_pred).statistic),
        "R2": float(r2_score(y_true, y_pred)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "Bias": float(np.mean(y_pred - y_true)),
    }


def reported_fold_r2(path: Path) -> np.ndarray:
    match = re.search(r"Fold Results:\s*\[([^]]+)\]", path.read_text(encoding="utf-8"))
    if match is None:
        raise ValueError(f"Could not parse fold results from {path}")
    return np.fromstring(match.group(1), sep=" ")


def bag_from_image_path(path: str) -> str:
    stem = Path(path).stem
    return re.sub(r"_\d+_depth$", "", stem)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-csv", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--fold-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.label_csv = args.label_csv.resolve()
    args.image_root = args.image_root.resolve()
    args.baseline = args.baseline.resolve()
    args.checkpoint_dir = args.checkpoint_dir.resolve()
    args.fold_results = args.fold_results.resolve()
    args.output_dir = args.output_dir.resolve()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    labels = pd.read_csv(args.label_csv)
    samples = prepare_all_samples(labels, args.image_root, EXPERIMENTS)
    if len(samples) != 239:
        raise ValueError(f"Expected 239 formal samples, found {len(samples)}")

    baseline = np.load(args.baseline)
    expected_r2 = reported_fold_r2(args.fold_results)
    if len(expected_r2) != 5:
        raise ValueError(f"Expected five reported fold scores, found {len(expected_r2)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splitter = KFold(n_splits=5, shuffle=True, random_state=args.seed)
    rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []

    for fold, (_, val_idx) in enumerate(splitter.split(samples), start=1):
        val_samples = [samples[index] for index in val_idx]
        loader = DataLoader(
            FeedDiffDataset(val_samples, baseline, is_train=False),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        checkpoint = args.checkpoint_dir / f"best_model_fold{fold}.pth"
        model = DiffResNet().to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        model.eval()

        predictions: list[float] = []
        observations: list[float] = []
        with torch.no_grad():
            for images, targets in loader:
                output = model(images.to(device)) * TARGET_SCALE_KG
                predictions.extend(output.cpu().numpy().astype(float).tolist())
                observations.extend((targets * TARGET_SCALE_KG).numpy().astype(float).tolist())

        y_true = np.asarray(observations)
        y_pred = np.asarray(predictions)
        fold_metric = metrics(y_true, y_pred)
        difference = float(fold_metric["R2"]) - float(expected_r2[fold - 1])
        if abs(difference) > 5e-5:
            raise RuntimeError(
                f"Fold {fold} R2 mismatch: exported={fold_metric['R2']:.8f}, "
                f"reported={expected_r2[fold - 1]:.8f}"
            )
        fold_rows.append(
            {
                "fold": fold,
                **fold_metric,
                "reported_R2": float(expected_r2[fold - 1]),
                "R2_difference": difference,
                "checkpoint": str(checkpoint),
            }
        )

        for sample_index, sample, observed, predicted in zip(
            val_idx, val_samples, observations, predictions
        ):
            image_path, _ = sample
            bag = bag_from_image_path(image_path)
            experiment = next(exp for exp in EXPERIMENTS if f"_{exp}_" in bag)
            rows.append(
                {
                    "sample_index": int(sample_index),
                    "fold": fold,
                    "bag": bag,
                    "experiment": experiment,
                    "image_path": image_path,
                    "y_true_kg": observed,
                    "y_pred_kg": predicted,
                    "residual_kg": predicted - observed,
                }
            )

    oof = pd.DataFrame(rows).sort_values("sample_index").reset_index(drop=True)
    if len(oof) != len(samples) or oof["sample_index"].nunique() != len(samples):
        raise RuntimeError("OOF export did not produce exactly one prediction per formal sample")

    fold_metrics = pd.DataFrame(fold_rows)
    pooled = metrics(oof["y_true_kg"].to_numpy(), oof["y_pred_kg"].to_numpy())
    pooled.update(
        {
            "evaluation": "five-fold out-of-fold prediction",
            "fold_R2_mean": float(fold_metrics["R2"].mean()),
            "fold_R2_sd": float(fold_metrics["R2"].std(ddof=0)),
            "fold_assignment_seed": args.seed,
            "checkpoints_retrained": False,
        }
    )

    oof.to_csv(args.output_dir / "oof_predictions.csv", index=False)
    fold_metrics.to_csv(args.output_dir / "fold_metrics.csv", index=False)
    (args.output_dir / "oof_metrics.json").write_text(
        json.dumps(pooled, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(pooled, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
