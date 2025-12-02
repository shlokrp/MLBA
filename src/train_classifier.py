import json
from pathlib import Path
from data_loader import load_data
from preprocess import clean_data, split_for_tasks, scale_features
from models.baseline_linear import build_linear
from models.knearest import build_knn_regressor
from models.random_forest import build_rf_regressor
from models.gradient_boosting import build_gb_regressor
from models.xgboost_model import build_xgb_regressor
from evaluation.regression_metrics import evaluate_regressor

def main():
    df = load_data("data/ttc-bus-delay-data-2023.xlsx")
    df = clean_data(df)

    X_train, X_test, _, _, y_train, y_test = split_for_tasks(df)

    X_train, X_test, scaler = scale_features(X_train, X_test)

    models = {
        "linear_regression": build_linear(),
        "knn_regressor": build_knn_regressor(),
        "random_forest": build_rf_regressor(),
        "gradient_boosting": build_gb_regressor(),
        "xgboost_regressor": build_xgb_regressor(),
    }

    results_dir = Path("results/regression")
    results_dir.mkdir(parents=True, exist_ok=True)

    final_results = {}

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)

        metrics = evaluate_regressor(model, X_test, y_test)
        final_results[name] = metrics

    with open(results_dir / "regression_summary.json", "w") as f:
        json.dump(final_results, f, indent=4)

if __name__ == "__main__":
    main()
