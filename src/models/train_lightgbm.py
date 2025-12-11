from src.models.lightgbm_baseline import run_baseline_experiment
import joblib
import json
from pathlib import Path

def main():
    model, metrics, series_mae = run_baseline_experiment()
    joblib.dump(model, "models/baseline_lightgbm.pkl")

    metrics_path = Path("reports/baseline_metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(metrics)

if __name__ == "__main__":
    main()