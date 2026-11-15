#!/usr/bin/env python3
"""Stage-I -> Stage-II uncertainty propagation: data preparation and sequence simulator.

Module 1 (data preparation) and Module 2 (one full station-sequence simulation).

The Stage-II recursion reproduced here is identical to
Project08.FCR/03.scr/system/02.virtual_tray/run_train_history_rate_reanalysis.py::simulate
for the primary configuration (inherit_state=True, decay_half_life_s=1800, idle_reset_s=None).
Level-0 verification (eta == 0) must reproduce the locked manuscript values
0.0514 / 0.0554 / 0.0581 / 0.0673 kg before any noise scenario is interpreted.

Two entry points share one numerical core:
  simulate_arrays() -> flat numpy dict   (fast; used inside Monte-Carlo loops)
  simulate()        -> pandas DataFrame  (event-level export for key scenarios)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PREDICTIONS_CSV = ROOT / "data/stage2/frozen_predictions.csv.gz"

PRIMARY = "state_train_rate_q0.60_x1.1_decay_0.5h"
CURRENT_EVENT = "current_only_train_rate_q0.60_x1.1"
DROP = "baseline_drop"
GLOBAL_RATE = "state_global_rate_q0.60_x1.1_decay_0.5h"
HALF_LIFE_S = 1800.0

REF_MAE = {"primary": 0.0514, "current_event": 0.0554, "drop": 0.0581, "global_rate": 0.0673}

DURATION_BINS = [0, 200, 400, 600, 1000, np.inf]
DURATION_LABELS = ["0-200 s", "200-400 s", "400-600 s", "600-1000 s", ">1000 s"]
DELIVERY_LABELS = ["Q1 (smallest)", "Q2", "Q3", "Q4 (largest)"]

ARRAY_COLUMNS = [
    "seq_index", "decay_factor", "true_delivery", "perturbed_delivery", "clipped",
    "rate_x_duration", "clean_available", "clean_pred", "clean_state",
    "noisy_available", "noisy_pred", "noisy_state", "cur_clean_pred", "cur_noisy_pred",
    "drop_clean_pred", "drop_noisy_pred", "actual_intake", "delta_state", "delta_pred",
    "state_error_bound", "mass_balance_error", "y_exceeds_p", "bound_violated", "eta",
]
META_COLUMNS = ["station", "pig_id", "start_dt", "event_id", "duration_s", "idle_s"]


@dataclass
class StationSeq:
    station: int
    event_id: np.ndarray
    pig_id: np.ndarray
    start_dt: np.ndarray
    seq_index: np.ndarray
    idle_s: np.ndarray
    delivery: np.ndarray
    duration_s: np.ndarray
    rate_prior: np.ndarray
    y_true: np.ndarray
    decay: np.ndarray = field(init=False)
    rate_x_duration: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.decay = 2.0 ** (-self.idle_s / HALF_LIFE_S)
        self.rate_x_duration = self.rate_prior * self.duration_s

    def __len__(self) -> int:
        return len(self.delivery)


@dataclass
class StageIIData:
    stations: list[StationSeq]
    flat: pd.DataFrame
    drop_coef: tuple[float, float]
    n_events: int
    train_events_per_pig: dict = field(default_factory=dict)
    fallback_pigs: set = field(default_factory=set)
    hist_mean_delivery: dict = field(default_factory=dict)
    global_mean_delivery: float = 0.0
    delivery_quartiles: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.delivery_quartiles = np.quantile(self.flat.drop_kg.to_numpy(float), [0.25, 0.5, 0.75])

    def zeros_eta(self) -> list[np.ndarray]:
        return [np.zeros(len(s)) for s in self.stations]


def _station_seq(station, df: pd.DataFrame) -> StationSeq:
    return StationSeq(
        station=int(station), event_id=df.event_id.to_numpy(), pig_id=df.pig_id.to_numpy(),
        start_dt=df.start_dt.to_numpy(), seq_index=np.arange(len(df)),
        idle_s=df.idle_s.to_numpy(float), delivery=df.drop_kg.to_numpy(float),
        duration_s=df.duration_s.to_numpy(float), rate_prior=df.rate_prior_kg_s.to_numpy(float),
        y_true=df.fi_true_kg.to_numpy(float),
    )


def drop_coefficients(raw: pd.DataFrame) -> tuple[float, float]:
    """OLS intercept/slope of the Drop baseline, fitted on the TRAINING split only."""
    tr = raw[(raw.split == "train") & (raw.config == DROP)]
    slope, intercept = np.polyfit(tr.drop_kg.to_numpy(float), tr.fi_true_kg.to_numpy(float), 1)
    return float(intercept), float(slope)


def load_stage2(split: str = "test", predictions_csv: Path = PREDICTIONS_CSV) -> StageIIData:
    raw = pd.read_csv(predictions_csv, low_memory=False)
    a, b = drop_coefficients(raw)
    tr = raw[(raw.split == "train") & (raw.config == DROP)].drop_duplicates("event_id")
    train_events_per_pig = {int(k): int(v) for k, v in tr.groupby("pig_id").size().items()}

    s = raw[(raw.split == split) & (raw.config == PRIMARY)].copy()
    s = s.sort_values(["station", "start_dt", "end_dt", "event_id"], kind="mergesort")
    stations, frames = [], []
    for station, g in s.groupby("station", sort=True):
        g = g.reset_index(drop=True)
        stations.append(_station_seq(station, g))
        frames.append(g)
    flat = pd.concat(frames, ignore_index=True)
    fallback = {int(p) for p in flat.pig_id.unique() if int(p) not in train_events_per_pig}
    # Station-level delivery history from the TRAINING split only (leakage-free controls).
    hist_mean = {int(k): float(v) for k, v in tr.groupby("station").drop_kg.mean().items()}
    glob_mean = float(tr.drop_kg.mean())
    if not hist_mean or glob_mean <= 0:
        raise RuntimeError("training-split delivery history is empty; no-vision controls invalid")
    return StageIIData(stations=stations, flat=flat, drop_coef=(a, b), n_events=len(flat),
                       train_events_per_pig=train_events_per_pig, fallback_pigs=fallback,
                       hist_mean_delivery=hist_mean, global_mean_delivery=glob_mean)


def simulate_arrays(data: StageIIData, eta: list[np.ndarray]) -> dict[str, np.ndarray]:
    """Fast numerical core. Returns flat arrays in station-then-time order."""
    if len(eta) != len(data.stations):
        raise ValueError(f"eta has {len(eta)} station blocks, expected {len(data.stations)}")
    a_drop, b_drop = data.drop_coef
    n_tot = data.n_events
    cols = {c: np.empty(n_tot) for c in ARRAY_COLUMNS}
    meta = {c: np.empty(n_tot, dtype=object) for c in META_COLUMNS[:4]}
    meta["duration_s"] = np.empty(n_tot)
    meta["idle_s"] = np.empty(n_tot)
    o = 0

    for seq, e in zip(data.stations, eta):
        n = len(seq)
        if len(e) != n:
            raise ValueError(f"station {seq.station}: eta length {len(e)} != {n}")
        d = seq.delivery
        d_pert = np.maximum(0.0, d + e)
        kT = seq.rate_x_duration
        a = seq.decay

        g_clean = np.empty(n); y_clean = np.empty(n); p_clean = np.empty(n)
        g_noisy = np.empty(n); y_noisy = np.empty(n); p_noisy = np.empty(n)
        bound = np.empty(n)
        gc = gn = bd = 0.0
        for i in range(n):
            ai = a[i]
            pc = ai * gc + d[i]
            yc = pc if pc < kT[i] else kT[i]
            gc = pc - yc
            pn = ai * gn + d_pert[i]
            yn = pn if pn < kT[i] else kT[i]
            gn = pn - yn
            p_clean[i] = pc; y_clean[i] = yc; g_clean[i] = gc
            p_noisy[i] = pn; y_noisy[i] = yn; g_noisy[i] = gn
            bd = ai * bd + abs(e[i])
            bound[i] = bd

        sl = slice(o, o + n)
        cols["seq_index"][sl] = seq.seq_index
        cols["decay_factor"][sl] = a
        cols["true_delivery"][sl] = d
        cols["perturbed_delivery"][sl] = d_pert
        cols["clipped"][sl] = (d_pert == 0.0).astype(float)
        cols["rate_x_duration"][sl] = kT
        cols["clean_available"][sl] = p_clean
        cols["clean_pred"][sl] = y_clean
        cols["clean_state"][sl] = g_clean
        cols["noisy_available"][sl] = p_noisy
        cols["noisy_pred"][sl] = y_noisy
        cols["noisy_state"][sl] = g_noisy
        cols["cur_clean_pred"][sl] = np.minimum(d, kT)
        cols["cur_noisy_pred"][sl] = np.minimum(d_pert, kT)
        cols["drop_clean_pred"][sl] = a_drop + b_drop * d
        cols["drop_noisy_pred"][sl] = a_drop + b_drop * d_pert
        cols["actual_intake"][sl] = seq.y_true
        cols["delta_state"][sl] = g_noisy - g_clean
        cols["delta_pred"][sl] = y_noisy - y_clean
        cols["state_error_bound"][sl] = bound
        cols["mass_balance_error"][sl] = p_noisy - y_noisy - g_noisy
        cols["y_exceeds_p"][sl] = (y_noisy > p_noisy + 1e-12).astype(float)
        cols["bound_violated"][sl] = (np.abs(g_noisy - g_clean) > bound + 1e-9).astype(float)
        cols["eta"][sl] = e
        meta["station"][sl] = seq.station
        meta["pig_id"][sl] = seq.pig_id
        meta["start_dt"][sl] = seq.start_dt
        meta["event_id"][sl] = seq.event_id
        meta["duration_s"][sl] = seq.duration_s
        meta["idle_s"][sl] = seq.idle_s
        o += n

    cols.update(meta)
    return cols


def simulate(data: StageIIData, eta: list[np.ndarray]) -> pd.DataFrame:
    """Event-level table for one replicate; use for export and stratified analysis."""
    cols = simulate_arrays(data, eta)
    ev = pd.DataFrame({c: cols[c] for c in ARRAY_COLUMNS + META_COLUMNS})
    for tag, pcol in [("clean", "clean_pred"), ("noisy", "noisy_pred"),
                      ("cur_clean", "cur_clean_pred"), ("cur_noisy", "cur_noisy_pred"),
                      ("drop_clean", "drop_clean_pred"), ("drop_noisy", "drop_noisy_pred")]:
        ev[f"{tag}_error"] = ev[pcol] - ev.actual_intake
        ev[f"{tag}_abs_error"] = np.abs(ev[f"{tag}_error"])
    return ev


MODELS = [("primary_clean", "clean_pred"), ("primary_noisy", "noisy_pred"),
          ("current_clean", "cur_clean_pred"), ("current_noisy", "cur_noisy_pred"),
          ("drop_clean", "drop_clean_pred"), ("drop_noisy", "drop_noisy_pred")]


def _fit(err: np.ndarray, y: np.ndarray, pred: np.ndarray) -> dict:
    ss = float(((y - y.mean()) ** 2).sum())
    return dict(MAE=float(np.abs(err).mean()), RMSE=float(np.sqrt((err ** 2).mean())),
                bias=float(err.mean()),
                R2=float(1.0 - (err ** 2).sum() / ss) if ss > 0 else float("nan"),
                r=float(np.corrcoef(y, pred)[0, 1]))


def summarise_arrays(cols: dict[str, np.ndarray]) -> dict:
    """Headline metrics for one replicate, computed directly on the fast arrays."""
    y = cols["actual_intake"]
    out: dict = {}
    for tag, pcol in MODELS:
        out[tag] = _fit(cols[pcol] - y, y, cols[pcol])
    pn, dn, cn = out["primary_noisy"]["MAE"], out["drop_noisy"]["MAE"], out["current_noisy"]["MAE"]
    pc, dc = out["primary_clean"]["MAE"], out["drop_clean"]["MAE"]
    out.update(
        degradation_vs_clean_primary=pn - pc,
        adv_vs_noisy_drop=pn - dn, adv_vs_noisy_current=pn - cn,
        adv_vs_clean_drop=pn - dc,
        below_clean_drop=bool(pn < REF_MAE["drop"]),
        below_clean_current=bool(pn < REF_MAE["current_event"]),
        better_than_noisy_drop=bool(pn < dn),
        clip_rate=float(cols["clipped"].mean()),
        max_abs_mass_balance_error=float(np.abs(cols["mass_balance_error"]).max()),
        n_y_exceeds_p=int(cols["y_exceeds_p"].sum()),
        n_negative_state=int((cols["noisy_state"] < -1e-12).sum()),
        bound_violations=int(cols["bound_violated"].sum()),
        max_abs_delta_state=float(np.abs(cols["delta_state"]).max()),
        mean_abs_delta_state=float(np.abs(cols["delta_state"]).mean()),
        max_abs_delta_pred=float(np.abs(cols["delta_pred"]).max()),
        within_0p05=float((np.abs(cols["noisy_pred"] - y) <= 0.05).mean()),
        within_0p10=float((np.abs(cols["noisy_pred"] - y) <= 0.10).mean()),
        within_0p05_clean=float((np.abs(cols["clean_pred"] - y) <= 0.05).mean()),
        within_0p10_clean=float((np.abs(cols["clean_pred"] - y) <= 0.10).mean()),
    )
    return out


def summarise(ev: pd.DataFrame) -> dict:
    """Same metrics as summarise_arrays(), from an event-level DataFrame."""
    return summarise_arrays({c: ev[c].to_numpy() for c in ARRAY_COLUMNS})


def level0_check(data: StageIIData, tol: float = 5e-4) -> dict:
    """Assert the simulator reproduces the four locked manuscript values at eta == 0."""
    s = summarise_arrays(simulate_arrays(data, data.zeros_eta()))
    got = {"primary": s["primary_clean"]["MAE"], "current_event": s["current_clean"]["MAE"],
           "drop": s["drop_clean"]["MAE"]}
    raw = pd.read_csv(PREDICTIONS_CSV, low_memory=False)
    g = raw[(raw.split == "test") & (raw.config == GLOBAL_RATE)]
    got["global_rate"] = float(np.abs(g.fi_pred_kg - g.fi_true_kg).mean())
    rep = {k: dict(got=round(v, 6), expected=REF_MAE[k], abs_diff=round(abs(v - REF_MAE[k]), 6),
                   ok=bool(abs(v - REF_MAE[k]) < tol)) for k, v in got.items()}
    rep["all_ok"] = all(v["ok"] for v in rep.values() if isinstance(v, dict))
    if not rep["all_ok"]:
        raise AssertionError(f"level-0 reproduction failed: {rep}")
    return rep


def add_strata(ev: pd.DataFrame, data: StageIIData) -> pd.DataFrame:
    ev = ev.copy()
    ev["duration_stratum"] = pd.cut(ev.duration_s, DURATION_BINS, labels=DURATION_LABELS,
                                    right=False).astype(str)
    q = data.delivery_quartiles
    ev["delivery_stratum"] = pd.cut(ev.true_delivery, [-np.inf, *q, np.inf],
                                    labels=DELIVERY_LABELS).astype(str)
    n_hist = ev.pig_id.map(data.train_events_per_pig).fillna(-1).astype(int)
    ev["n_train_events"] = n_hist
    ev["identity_stratum"] = np.where(
        n_hist < 0, "no training history (global fallback prior)",
        np.where(n_hist < 20, "<20 training events",
                 np.where(n_hist < 40, "20-39 training events", ">=40 training events")))
    return ev
