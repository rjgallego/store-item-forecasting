from src.models.catboost_baseline import run_catboost_experiment
import joblib
import json
from pathlib import Path

def main():
    model, metrics, series_mae = run_catboost_experiment()

    models_path = Path("models")
    metrics_path = Path("reports")

    joblib.dump(model, models_path / "baseline_catboost.pkl")

    series_mae.to_csv(metrics_path / "catboost_series_mae.csv", header=["mae"])

    metrics_path = metrics_path / "catboost_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    
    print(metrics)

if __name__ == "__main__":
    main()