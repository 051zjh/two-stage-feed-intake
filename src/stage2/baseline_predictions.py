"""Training-only baseline fits copied from the source analysis."""
import pandas as pd
from sklearn.linear_model import LinearRegression


def baseline_predictions(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    train = splits["train"]
    drop_model = LinearRegression().fit(train[["投料量"]], train["采食量"])
    weight_median = float(train["体重"].median())
    time_x = train[["duration_s", "体重"]].copy()
    time_x["体重"] = time_x["体重"].fillna(weight_median)
    time_model = LinearRegression().fit(time_x, train["采食量"])
    outputs = []
    for split, df in splits.items():
        common = pd.DataFrame({
            "event_id": df["event_id"], "split": split, "station": df["测定站"],
            "pig_id": df["电子耳牌"], "start_dt": df["start_dt"],
            "duration_s": df["duration_s"], "weight_kg": df["体重"],
            "drop_kg": df["投料量"], "fi_true_kg": df["采食量"],
        })
        drop = common.copy()
        drop["fi_pred_kg"] = drop_model.predict(df[["投料量"]])
        drop["config"] = "baseline_drop"
        outputs.append(drop)
        time_x = df[["duration_s", "体重"]].copy()
        time_x["体重"] = time_x["体重"].fillna(weight_median)
        time = common.copy()
        time["fi_pred_kg"] = time_model.predict(time_x)
        time["config"] = "baseline_time"
        outputs.append(time)
    return pd.concat(outputs, ignore_index=True)
