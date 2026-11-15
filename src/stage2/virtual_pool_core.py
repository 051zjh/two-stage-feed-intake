from __future__ import annotations
from pathlib import Path
import pandas as pd
from scipy.stats import pearsonr

REQUIRED_COLUMNS = [
    "测定站", "电子耳牌", "日期", "开始时间", "结束时间", "采食量", "体重", "投料量"
]


def parse_events(df: pd.DataFrame, split: str | None = None) -> pd.DataFrame:
    """Normalize event fields and retain model-eligible positive events."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df.copy()
    for col in ["测定站", "电子耳牌", "采食量", "体重", "投料量"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    date = out["日期"].astype(str).str.strip()
    start = pd.to_datetime(date + " " + out["开始时间"].astype(str).str.strip(), errors="coerce")
    end = pd.to_datetime(date + " " + out["结束时间"].astype(str).str.strip(), errors="coerce")
    duration = (end - start).dt.total_seconds()
    end = end.where(duration >= 0, end + pd.Timedelta(days=1))
    duration = (end - start).dt.total_seconds()
    out["start_dt"] = start
    out["end_dt"] = end
    out["duration_s"] = duration
    if split is not None:
        out["split"] = split
    elif "split" not in out.columns:
        out["split"] = "all"

    out = out.dropna(subset=["测定站", "电子耳牌", "采食量", "投料量", "start_dt", "end_dt", "duration_s"])
    out = out[(out["采食量"] > 0) & (out["投料量"] > 0) & (out["duration_s"] > 0)].copy()
    out["测定站"] = out["测定站"].astype(int)
    out["电子耳牌"] = out["电子耳牌"].astype("int64")
    out["event_id"] = (
        out["测定站"].astype(str) + "|" + out["电子耳牌"].astype(str) + "|"
        + out["start_dt"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    )
    return out.reset_index(drop=True)


def load_splits(data_dir: Path) -> dict[str, pd.DataFrame]:
    result: dict[str, pd.DataFrame] = {}
    for split in ["train", "val", "test", "val_final"]:
        path = data_dir / f"{split}.csv"
        if path.exists():
            result[split] = parse_events(pd.read_csv(path), split=split)
    if "train" not in result:
        raise FileNotFoundError(f"Missing training split under {data_dir}")
    return result

