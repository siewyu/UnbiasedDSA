# evaluate.py
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np


def evaluate_predictions(labels_csv="../data/labels.csv", preds_csv="../data/predictions.csv"):
    labels = pd.read_csv(labels_csv)
    preds = pd.read_csv(preds_csv)

    merged = labels.merge(preds, on="image_id", suffixes=("_true", "_pred")).rename(columns={"hgb": "hgb_pred"})
    merged["error"] = (merged["hgb_true"] - merged["hgb_pred"]).abs()

    mae = mean_absolute_error(merged["hgb_true"], merged["hgb_pred"])
    rmse = np.sqrt(mean_squared_error(merged["hgb_true"], merged["hgb_pred"]))
    r2 = r2_score(merged["hgb_true"], merged["hgb_pred"])

    print("=" * 70)
    print("MODEL EVALUATION RESULTS")
    print("=" * 70)
    print("\nSample Predictions:")
    print(merged[["image_id", "hgb_true", "hgb_pred", "error"]].head(10).to_string(index=False))
    print("\n" + "=" * 70)
    print("PERFORMANCE METRICS")
    print("=" * 70)
    print(f"Mean Absolute Error (MAE):  {mae:.4f} g/dL")
    print(f"Root Mean Squared Error:    {rmse:.4f} g/dL")
    print(f"R² Score:                   {r2:.4f}")
    print(f"Median Absolute Error:      {merged['error'].median():.4f} g/dL")
    print(f"Std of Absolute Error:      {merged['error'].std():.4f} g/dL")
    print(f"Min / Max Abs Error:        {merged['error'].min():.4f} / {merged['error'].max():.4f} g/dL")
    print("=" * 70)

    bad = merged.sort_values("error", ascending=False).head(10)
    print("\nTop 10 worst cases:")
    print(bad[["image_id","hgb_true","hgb_pred","error"]].to_string(index=False))

    merged.to_csv("evaluation_results.csv", index=False)
    print("\nDetailed results saved to: evaluation_results.csv")


if __name__ == "__main__":
    evaluate_predictions()
