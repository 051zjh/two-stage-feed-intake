#!/usr/bin/env python3
"""Evaluate a leakage-free state model with training-history rate priors.

The rate prior is estimated once from training events for each pig. Validation
selects the training-rate quantile and a multiplier; test events never update
the prior or any rate history. The paper-facing model uses only the
rate-duration bound and mass-balanced residual state; fixed capacity and
body-weight rate caps are intentionally excluded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from virtual_pool_core import load_splits, pearsonr


PRIMARY_DECAY_HOURS = 0.5
DECAY_SENSITIVITY_HOURS = (0.25, 0.75, 1.0, 1.5, 2.0)
DECAY_SENSITIVITY_GRID_MINUTES = tuple(
    int(round(hours * 60))
    for hours in sorted((PRIMARY_DECAY_HOURS, *DECAY_SENSITIVITY_HOURS))
)


def metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    y = frame["fi_true_kg"].to_numpy(float)
    p = frame["fi_pred_kg"].to_numpy(float)
    return {
        "r": float(pearsonr(y, p).statistic),
        "R2": float(r2_score(y, p)),
        "MAE": float(mean_absolute_error(y, p)),
        "RMSE": float(np.sqrt(mean_squared_error(y, p))),
        "Bias": float(np.mean(p - y)),
        "n": int(len(frame)),
    }


def training_rate_priors(train: pd.DataFrame, quantile: float) -> tuple[pd.Series, float]:
    work = train.copy()
    work["rate_kg_s"] = work["采食量"] / work["duration_s"]
    work = work.replace([np.inf, -np.inf], np.nan).dropna(subset=["rate_kg_s"])
    priors = work.groupby("电子耳牌")["rate_kg_s"].quantile(quantile)
    fallback = float(work["rate_kg_s"].quantile(quantile))
    return priors, fallback


def simulate(
    frame: pd.DataFrame,
    priors: pd.Series,
    fallback: float,
    multiplier: float,
    inherit_state: bool = True,
    decay_half_life_s: float | None = 1800.0,
    idle_reset_s: float | None = None,
) -> pd.DataFrame:
    rows: list[dict] = []
    for station, station_df in frame.groupby("测定站", sort=True):
        state = 0.0
        last_end = None
        ordered = station_df.sort_values(["start_dt", "end_dt", "event_id"], kind="mergesort")
        for event in ordered.itertuples(index=False):
            if not inherit_state:
                state = 0.0
            idle_s = 0.0 if last_end is None else max(0.0, (event.start_dt - last_end).total_seconds())
            state_pre_adjust = state
            decay_factor = 1.0
            reset_applied = False
            if last_end is not None:
                if decay_half_life_s is not None:
                    decay_factor = 2.0 ** (-idle_s / decay_half_life_s)
                    state *= decay_factor
                if idle_reset_s is not None and idle_s > idle_reset_s:
                    state = 0.0
                    reset_applied = True

            pig = int(event.电子耳牌)
            prior = float(priors.get(pig, fallback)) * multiplier
            k = prior
            state_before = state
            available = state_before + float(event.投料量)
            prediction = min(available, k * float(event.duration_s))
            state = max(0.0, available - prediction) if inherit_state else 0.0
            mass_balance_error = available - prediction - state
            rows.append({
                "event_id": event.event_id,
                "split": event.split,
                "station": int(station),
                "pig_id": pig,
                "start_dt": event.start_dt,
                "end_dt": event.end_dt,
                "duration_s": float(event.duration_s),
                "weight_kg": float(event.体重) if pd.notna(event.体重) else np.nan,
                "drop_kg": float(event.投料量),
                "fi_true_kg": float(event.采食量),
                "fi_pred_kg": prediction,
                "abs_error_kg": abs(prediction - float(event.采食量)),
                "state_pre_adjust_kg": state_pre_adjust,
                "state_before_kg": state_before,
                "state_after_kg": state,
                "available_kg": available,
                "k_dynamic_kg_s": k,
                "rate_prior_kg_s": prior,
                "idle_s": idle_s,
                "decay_factor": decay_factor,
                "reset_applied": reset_applied,
                "inheritance_active": bool(state_before > 1e-12),
                "mass_balance_error_kg": mass_balance_error,
                "config": ("state" if inherit_state else "current_only") + "_train_rate",
            })
            last_end = event.end_dt
    return pd.DataFrame(rows).sort_values(["start_dt", "station", "event_id"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/stage2/splits"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/rate_prior_candidates"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    splits = load_splits(args.data_dir)
    train = splits["train"]
    candidates = []
    predictions_by_candidate = {}
    for quantile in [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
        priors, fallback = training_rate_priors(train, quantile)
        for multiplier in [0.9, 1.0, 1.1, 1.2, 1.3, 1.4]:
            val = simulate(
                splits["val"],
                priors,
                fallback,
                multiplier,
                decay_half_life_s=PRIMARY_DECAY_HOURS * 3600.0,
            )
            result = metrics(val)
            result.update({"quantile": quantile, "multiplier": multiplier})
            candidates.append(result)
            predictions_by_candidate[(quantile, multiplier)] = (priors, fallback)

    selection = min(candidates, key=lambda x: x["MAE"])
    q = float(selection["quantile"])
    multiplier = float(selection["multiplier"])
    priors, fallback = predictions_by_candidate[(q, multiplier)]

    all_frames = []
    current_name = "current_only_train_rate_q%.2f_x%.1f" % (q, multiplier)
    primary_name = "state_train_rate_q%.2f_x%.1f_decay_%gh" % (
        q,
        multiplier,
        PRIMARY_DECAY_HOURS,
    )
    hard_reset_name = "state_train_rate_q%.2f_x%.1f_reset30m" % (q, multiplier)
    no_decay_name = "state_train_rate_q%.2f_x%.1f_no_decay_reset" % (q, multiplier)
    global_rate_name = "state_global_rate_q%.2f_x%.1f_decay_0.5h" % (q, multiplier)
    decay_sensitivity_hours = DECAY_SENSITIVITY_HOURS
    decay_sensitivity_names = [
        "state_train_rate_q%.2f_x%.1f_decay_%gh" % (q, multiplier, hours)
        for hours in decay_sensitivity_hours
    ]
    for split, frame in splits.items():
        out = simulate(
            frame,
            priors,
            fallback,
            multiplier,
            decay_half_life_s=PRIMARY_DECAY_HOURS * 3600.0,
            idle_reset_s=None,
        )
        out["config"] = primary_name
        out["rate_quantile"] = q
        out["rate_multiplier"] = multiplier
        all_frames.append(out)
        current = simulate(frame, priors, fallback, multiplier, inherit_state=False)
        current["config"] = current_name
        current["rate_quantile"] = q
        current["rate_multiplier"] = multiplier
        all_frames.append(current)
        hard_reset = simulate(frame, priors, fallback, multiplier, decay_half_life_s=None, idle_reset_s=1800.0)
        hard_reset["config"] = hard_reset_name
        hard_reset["rate_quantile"] = q
        hard_reset["rate_multiplier"] = multiplier
        all_frames.append(hard_reset)
        no_decay = simulate(frame, priors, fallback, multiplier, decay_half_life_s=None, idle_reset_s=None)
        no_decay["config"] = no_decay_name
        no_decay["rate_quantile"] = q
        no_decay["rate_multiplier"] = multiplier
        all_frames.append(no_decay)
        global_rate = simulate(
            frame,
            pd.Series(dtype=float),
            fallback,
            multiplier,
            decay_half_life_s=PRIMARY_DECAY_HOURS * 3600.0,
            idle_reset_s=None,
        )
        global_rate["config"] = global_rate_name
        global_rate["rate_quantile"] = q
        global_rate["rate_multiplier"] = multiplier
        all_frames.append(global_rate)
        for hours, decay_name in zip(decay_sensitivity_hours, decay_sensitivity_names):
            decay = simulate(
                frame,
                priors,
                fallback,
                multiplier,
                decay_half_life_s=float(hours * 3600),
                idle_reset_s=None,
            )
            decay["config"] = decay_name
            decay["rate_quantile"] = q
            decay["rate_multiplier"] = multiplier
            all_frames.append(decay)
    all_predictions = pd.concat(all_frames, ignore_index=True)
    all_predictions.to_csv(args.out_dir / "selected_predictions.csv.gz", index=False, compression="gzip")
    pd.DataFrame(candidates).sort_values("MAE").to_csv(args.out_dir / "validation_rate_selection.csv", index=False)

    metrics_rows = []
    for (config, split), group in all_predictions.groupby(["config", "split"], sort=False):
        metrics_rows.append({"config": config, "split": split, **metrics(group)})
    pd.DataFrame(metrics_rows).to_csv(args.out_dir / "selected_metrics.csv", index=False)
    selected_name = primary_name
    (args.out_dir / "selection.json").write_text(json.dumps({
        "primary_config": selected_name,
        "current_event_config": current_name,
        "selection_split": "val",
        "criterion": "minimum_validation_MAE_for_rate_grid_at_locked_decay",
        "selection_scope": "rate_quantile_and_multiplier_conditional_on_locked_decay",
        "rate_prior_source": "training_split_only",
        "rate_quantile": q,
        "rate_multiplier": multiplier,
        "decay_half_life_s": PRIMARY_DECAY_HOURS * 3600.0,
        "decay_sensitivity_grid_minutes": list(DECAY_SENSITIVITY_GRID_MINUTES),
        "decay_selection_source": "earlier_validation_only_fixed_rate_sweep",
        "decay_locked_before_rate_prior_selection": True,
        "joint_decay_rate_grid_searched": False,
        "test_labels_update_rate": False,
        "test_labels_update_state": False,
        "uses_body_weight_rate_cap": False,
        "uses_fixed_capacity_bound": False,
        "component_ablation_configs": [
            current_name,
            global_rate_name,
        ],
        "state_rule_sensitivity_configs": [hard_reset_name, no_decay_name],
        "decay_sensitivity_configs": decay_sensitivity_names,
    }, indent=2), encoding="utf-8")
    primary_test = next(
        row for row in metrics_rows
        if row["config"] == primary_name and row["split"] == "test"
    )
    print(json.dumps({"selection": selection, "primary_test": primary_test}, indent=2))


if __name__ == "__main__":
    main()
