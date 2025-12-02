import json
from pathlib import Path
from data_loader import load_data
from preprocess import clean_data, split_for_tasks, scale_features
from models.baseline_logistic import build_logistic
from models.knearest import build_knn_classifier
from models.random_forest import build_rf_classifier
from models.gradient_boosting import build_gb_classifier
from models.xgboost_model import build_xgb_classifier
from evaluation.classification_metrics import evaluate_classifier
from evaluation.plotting import plot_confusion_matrix, plot_roc, plot_pr

def main():
    df = load_data("data/ttc-bus-delay-data-2023.xlsx")
    df = clean_data(df)

    X_train, X_test, y_train, y_test, _, _ = split_for_tasks(df)

    X_train, X_test, scaler = scale_features(X_train, X_test)

    models = {
        "logistic_regression": build_logistic(),
        "knn": build_knn_classifier(),
        "random_forest": build_rf_classifier(),
        "gradient_boosting": build_gb_classifier(),
        "xgboost": build_xgb_classifier(),
    }

    results_dir = Path("results/classification")
    results_dir.mkdir(parents=True, exist_ok=True)

    final_results = {}

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)

        metrics = evaluate_classifier(model, X_test, y_test)
        final_results[name] = metrics

        plot_confusion_matrix(
            metrics["confusion_matrix"],
            results_dir / f"{name}_confusion_matrix.png"
        )
        plot_roc(model, X_test, y_test, results_dir / f"{name}_roc_curve.png")
        plot_pr(model, X_test, y_test, results_dir / f"{name}_pr_curve.png")

    with open(results_dir / "metrics_summary.json", "w") as f:
        json.dump(final_results, f, indent=4)

if __name__ == "__main__":
    main()
