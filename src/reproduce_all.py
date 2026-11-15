"""Reproduce the locked metrics, rate selection, and 1,000-run linkage."""
from pathlib import Path
import json
import subprocess
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/linkage'))
sys.path.insert(0, str(ROOT / 'src/stage2'))
import propagation_core as linkage
from virtual_pool_core import load_splits
from baseline_predictions import baseline_predictions


def fit(y, p):
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    return {
        'n': int(len(y)), 'R2': float(r2_score(y, p)),
        'MAE': float(mean_absolute_error(y, p)),
        'RMSE': float(np.sqrt(mean_squared_error(y, p))),
        'Bias': float((p - y).mean()),
    }


def main():
    out = ROOT / 'results/verification'
    out.mkdir(parents=True, exist_ok=True)
    oof = pd.read_csv(ROOT / 'data/stage1/oof_predictions.csv').sort_values('sample_index')
    stage1 = fit(oof.y_true_kg, oof.y_pred_kg)
    assert stage1['n'] == 239
    assert abs(stage1['MAE'] - 0.7637209225773313) < 1e-12
    assert abs(stage1['R2'] - 0.9913187362687358) < 1e-12
    for fold, (_, indices) in enumerate(KFold(5, shuffle=True, random_state=42).split(oof), 1):
        assert np.array_equal(indices, np.flatnonzero(oof.fold.to_numpy() == fold))
    labels = pd.read_csv(ROOT / 'data/stage1/reference_labels.csv')
    assert labels.image.str.removesuffix('.bag').tolist() == oof.bag.tolist()
    for relative in oof.image_path:
        assert (ROOT / 'data/stage1' / relative).is_file()
    print('Stage I: 239 frozen OOF samples, fold assignments and frame paths verified.', flush=True)

    selection_out = ROOT / 'results/stage2_reanalysis'
    subprocess.run([
        sys.executable, str(ROOT / 'src/stage2/reproduce_selection.py'),
        '--data-dir', str(ROOT / 'data/stage2/splits'),
        '--out-dir', str(selection_out),
    ], cwd=ROOT, check=True)
    selection = json.loads((selection_out / 'selection.json').read_text())
    assert selection['rate_quantile'] == 0.60 and selection['rate_multiplier'] == 1.1
    assert selection['test_labels_update_rate'] is False
    fresh = pd.read_csv(selection_out / 'selected_predictions.csv.gz', low_memory=False)
    frozen = pd.read_csv(ROOT / 'data/stage2/frozen_predictions.csv.gz', low_memory=False)
    a = fresh[fresh.config == linkage.PRIMARY]
    b = frozen[frozen.config == linkage.PRIMARY]
    matched = a.merge(b, on=['event_id', 'split'], suffixes=('_new', '_locked'), validate='one_to_one')
    assert len(matched) == 13136
    max_primary_difference = float(np.abs(matched.fi_pred_kg_new - matched.fi_pred_kg_locked).max())
    assert max_primary_difference < 1e-12
    baseline = baseline_predictions(load_splits(ROOT / 'data/stage2/splits'))
    baseline_ref = frozen[frozen.config.isin(['baseline_drop', 'baseline_time'])]
    baseline_match = baseline.merge(baseline_ref, on=['config', 'split', 'event_id'],
                                    suffixes=('_new', '_locked'), validate='one_to_one')
    assert len(baseline_match) == 26272
    max_baseline_difference = float(np.abs(baseline_match.fi_pred_kg_new - baseline_match.fi_pred_kg_locked).max())
    assert max_baseline_difference < 1e-12
    primary = b[b.split == 'test']
    primary_metrics = fit(primary.fi_true_kg, primary.fi_pred_kg)
    print('Stage II: 48-candidate validation selection and all 13,136 primary predictions reproduced.', flush=True)

    data = linkage.load_stage2('test')
    zero = linkage.level0_check(data)
    residuals = oof.residual_kg.to_numpy(float)
    generator = np.random.default_rng(20261008)
    predictions = np.empty((1000, data.n_events), dtype=float)
    max_balance_error = 0.0
    observed = data.flat.fi_true_kg.to_numpy(float)
    for rep in range(1000):
        errors = []
        for station in data.stations:
            endpoints = generator.choice(residuals, len(station) + 1, replace=True)
            errors.append(endpoints[:-1] - endpoints[1:])
        cols = linkage.simulate_arrays(data, errors)
        predictions[rep] = cols['noisy_pred']
        max_balance_error = max(max_balance_error, float(np.abs(cols['mass_balance_error']).max()))
        assert not cols['bound_violated'].any()
        assert not cols['y_exceeds_p'].any()
        assert (cols['noisy_state'] >= -1e-12).all()
    maes = np.abs(predictions - observed[None, :]).mean(axis=1)
    reference_maes = pd.read_csv(ROOT / 'data/linkage/reference_simulation_metrics.csv')
    max_mae_difference = float(np.abs(maes - reference_maes.simulated_linkage_MAE_kg).max())
    assert max_mae_difference < 1e-12
    simulated = pd.DataFrame({
        'event_id': data.flat.event_id, 'observed_intake_kg': observed,
        'mean_simulated_intake_kg': predictions.mean(axis=0),
        'simulation_p2_5_kg': np.percentile(predictions, 2.5, axis=0),
        'simulation_p97_5_kg': np.percentile(predictions, 97.5, axis=0),
    })
    reference = pd.read_csv(ROOT / 'data/linkage/reference_mean_predictions.csv')
    matched_mean = simulated.merge(reference, on='event_id', suffixes=('_new', '_locked'), validate='one_to_one')
    assert len(matched_mean) == 1298
    max_mean_difference = float(np.abs(
        matched_mean.mean_simulated_intake_kg_new - matched_mean.mean_simulated_intake_kg_locked
    ).max())
    assert max_mean_difference < 1e-12
    mean_metrics = fit(observed, predictions.mean(axis=0))
    simulated.to_csv(out / 'reproduced_mean_predictions.csv', index=False)
    pd.DataFrame({'replicate': np.arange(1, 1001), 'MAE_kg': maes}).to_csv(
        out / 'reproduced_simulation_metrics.csv', index=False)
    report = {
        'stage1_frozen_OOF': stage1,
        'stage1_retraining': 'Code and representative frames supplied; checkpoints and exact retraining not included in this verification',
        'stage2_selected_quantile': 0.60, 'stage2_selected_multiplier': 1.1,
        'stage2_temporal_test': primary_metrics,
        'stage2_all_primary_prediction_max_difference': max_primary_difference,
        'stage2_all_baseline_prediction_max_difference': max_baseline_difference,
        'linkage': {
            'seed': 20261008, 'replicates': 1000, 'mean_MAE_kg': float(maes.mean()),
            'MAE_p2_5_kg': float(np.percentile(maes, 2.5)),
            'MAE_p97_5_kg': float(np.percentile(maes, 97.5)),
            'mean_prediction_metrics': mean_metrics,
            'reference_replicate_MAE_max_difference': max_mae_difference,
            'reference_mean_prediction_max_difference': max_mean_difference,
            'max_mass_balance_error': max_balance_error,
        },
        'zero_error_controls': zero, 'status': 'PASS',
    }
    (out / 'verification.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
