"""Train the closure-risk model using evaluation fit for an imbalanced classifier.

The output is a calibrated *probability* of road closure.  The operating
threshold is selected on validation data to maximise F1, rather than assuming
0.50 or reporting accuracy as the main result.
"""

import json
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    r2_score,
)


DATA_PATH = "gridlock_dataset.csv"
MODEL_PATH = "event_closure_model.cbm"
METRICS_PATH = "model_metrics.json"
FEATURES = [
    "event_type", "event_cause", "veh_type", "hour_of_day", "is_weekend",
    "corridor", "zone", "junction",
]
CATEGORICAL_FEATURES = ["event_type", "event_cause", "veh_type", "corridor", "zone", "junction"]


def prepare_data(path):
    df = pd.read_csv(path)
    df["start_datetime"] = pd.to_datetime(df["start_datetime"], errors="coerce", utc=True)
    df = df.dropna(subset=["start_datetime"]).sort_values("start_datetime").reset_index(drop=True)
    df["hour_of_day"] = df["start_datetime"].dt.hour.astype(int)
    df["is_weekend"] = df["start_datetime"].dt.dayofweek.isin([5, 6]).astype(int)

    for column in CATEGORICAL_FEATURES:
        df[column] = df[column].fillna("Missing").astype(str)

    # Explicit conversion protects training if the CSV is read as text on a new machine.
    df["requires_road_closure"] = (
        df["requires_road_closure"].astype(str).str.strip().str.lower().map({"true": 1, "false": 0}).fillna(0).astype(int)
    )
    return df


def best_f1_threshold(y_true, probabilities):
    thresholds = np.arange(0.05, 0.96, 0.01)
    scores = [f1_score(y_true, probabilities >= threshold, zero_division=0) for threshold in thresholds]
    index = int(np.argmax(scores))
    return float(thresholds[index]), float(scores[index])


def classification_metrics(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)
    return {
        "f1": round(float(f1_score(y_true, predictions, zero_division=0)), 4),
        "precision": round(float(precision_score(y_true, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, predictions, zero_division=0)), 4),
        "pr_auc": round(float(average_precision_score(y_true, probabilities)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, probabilities)), 4),
        "brier_score": round(float(brier_score_loss(y_true, probabilities)), 4),
        # R² is not a classifier-selection metric. It is retained only because
        # probability predictions can technically be compared against 0/1 labels.
        "probability_r2_supplemental": round(float(r2_score(y_true, probabilities)), 4),
    }


def main():
    print("Loading and time-ordering historical events...")
    df = prepare_data(DATA_PATH)
    target = df["requires_road_closure"]

    # Chronological split: train on the past, tune on the next period, report on the future.
    train_end = int(len(df) * 0.70)
    validation_end = int(len(df) * 0.85)
    train, validation, test = df.iloc[:train_end], df.iloc[train_end:validation_end], df.iloc[validation_end:]
    print(f"Rows: train={len(train)}, validation={len(validation)}, test={len(test)}")
    print(f"Closure prevalence: {target.mean():.1%}")

    # Balancing improves recall for the scarce closure class; threshold tuning then
    # chooses the precision/recall trade-off that maximises F1.
    negatives = int((train["requires_road_closure"] == 0).sum())
    positives = int((train["requires_road_closure"] == 1).sum())
    positive_weight = max(1.0, negatives / max(positives, 1))
    model = CatBoostClassifier(
        iterations=700,
        learning_rate=0.04,
        depth=7,
        loss_function="Logloss",
        # PR-AUC is threshold-independent and suited to rare closures. F1 is
        # optimized afterwards on the validation set at the best threshold.
        eval_metric="PRAUC:type=Classic",
        cat_features=CATEGORICAL_FEATURES,
        class_weights=[1.0, positive_weight],
        random_seed=42,
        verbose=100,
        train_dir="catboost_training_run",
        od_type="Iter",
        od_wait=80,
    )

    print("Training CatBoost closure-risk model...")
    model.fit(
        train[FEATURES],
        train["requires_road_closure"],
        eval_set=(validation[FEATURES], validation["requires_road_closure"]),
        use_best_model=True,
    )

    validation_probabilities = model.predict_proba(validation[FEATURES])[:, 1]
    threshold, validation_f1 = best_f1_threshold(validation["requires_road_closure"].to_numpy(), validation_probabilities)
    test_probabilities = model.predict_proba(test[FEATURES])[:, 1]
    metrics = classification_metrics(test["requires_road_closure"].to_numpy(), test_probabilities, threshold)
    metrics.update(
        {
            "selected_threshold": threshold,
            "validation_f1_at_selected_threshold": round(validation_f1, 4),
            "test_rows": int(len(test)),
            "test_closure_prevalence": round(float(test["requires_road_closure"].mean()), 4),
            "evaluation_note": "F1, PR-AUC, precision and recall are primary. R2 is supplemental only; R2 is not appropriate for selecting a binary classifier.",
        }
    )

    model.save_model(MODEL_PATH)
    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print("\nFinal future-period test metrics")
    for name, value in metrics.items():
        if name not in {"evaluation_note"}:
            print(f"{name}: {value}")
    print(f"\nSaved model to {MODEL_PATH} and metrics to {METRICS_PATH}")


if __name__ == "__main__":
    main()
