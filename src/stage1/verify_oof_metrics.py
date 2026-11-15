from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
d = pd.read_csv(Path("data/stage1/oof_predictions.csv"))
y, p = d.y_true_kg.to_numpy(), d.y_pred_kg.to_numpy()
print({"n": len(d), "R2": r2_score(y,p), "MAE": mean_absolute_error(y,p), "RMSE": np.sqrt(mean_squared_error(y,p)), "Bias": float((p-y).mean())})
