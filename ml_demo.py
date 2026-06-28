from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).parent
DATA_PATH = ROOT / "synthetic_ml_data.csv"
METRICS_PATH = ROOT / "metrics.json"
FEATURES = ["study_hours_week", "practice_tests", "attendance_rate", "prior_score"]
TARGET = "ready_for_assessment"


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -35, 35)
    return 1 / (1 + np.exp(-values))


def stratified_split(
    labels: np.ndarray, test_size: float = 0.25, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_indices: list[int] = []
    test_indices: list[int] = []
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        cutoff = max(1, round(len(indices) * test_size))
        test_indices.extend(indices[:cutoff].tolist())
        train_indices.extend(indices[cutoff:].tolist())
    rng.shuffle(train_indices)
    rng.shuffle(test_indices)
    return np.array(train_indices), np.array(test_indices)


def train_logistic_regression(
    features: np.ndarray,
    labels: np.ndarray,
    learning_rate: float = 0.08,
    iterations: int = 4000,
    l2_penalty: float = 0.01,
) -> tuple[np.ndarray, float]:
    weights = np.zeros(features.shape[1], dtype=float)
    bias = 0.0
    for _ in range(iterations):
        probabilities = sigmoid(features @ weights + bias)
        errors = probabilities - labels
        weight_gradient = (features.T @ errors) / len(labels) + l2_penalty * weights
        bias_gradient = float(errors.mean())
        weights -= learning_rate * weight_gradient
        bias -= learning_rate * bias_gradient
    return weights, bias


def classification_metrics(
    labels: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray
) -> dict:
    true_positive = int(((labels == 1) & (predictions == 1)).sum())
    true_negative = int(((labels == 0) & (predictions == 0)).sum())
    false_positive = int(((labels == 0) & (predictions == 1)).sum())
    false_negative = int(((labels == 1) & (predictions == 0)).sum())

    def ratio(numerator: float, denominator: float) -> float:
        return float(numerator / denominator) if denominator else 0.0

    ranks = pd.Series(probabilities).rank(method="average").to_numpy()
    positives = labels == 1
    positive_count = int(positives.sum())
    negative_count = int((~positives).sum())
    auc = ratio(
        ranks[positives].sum() - positive_count * (positive_count + 1) / 2,
        positive_count * negative_count,
    )
    precision = ratio(true_positive, true_positive + false_positive)
    recall = ratio(true_positive, true_positive + false_negative)
    return {
        "accuracy": ratio(true_positive + true_negative, len(labels)),
        "precision": precision,
        "recall": recall,
        "f1": ratio(2 * precision * recall, precision + recall),
        "roc_auc": auc,
        "confusion_matrix": [[true_negative, false_positive], [false_negative, true_positive]],
    }


def main() -> None:
    frame = pd.read_csv(DATA_PATH)
    features = frame[FEATURES].to_numpy(dtype=float)
    labels = frame[TARGET].to_numpy(dtype=int)
    train_indices, test_indices = stratified_split(labels)
    x_train, x_test = features[train_indices], features[test_indices]
    y_train, y_test = labels[train_indices], labels[test_indices]

    means = x_train.mean(axis=0)
    standard_deviations = x_train.std(axis=0)
    standard_deviations[standard_deviations == 0] = 1
    x_train_scaled = (x_train - means) / standard_deviations
    x_test_scaled = (x_test - means) / standard_deviations

    weights, bias = train_logistic_regression(x_train_scaled, y_train)
    probabilities = sigmoid(x_test_scaled @ weights + bias)
    predictions = (probabilities >= 0.5).astype(int)
    calculated = classification_metrics(y_test, predictions, probabilities)

    metrics = {
        "dataset_rows": int(len(frame)),
        "test_rows": int(len(x_test)),
        "positive_rate": round(float(frame[TARGET].mean()), 4),
        "accuracy": round(calculated["accuracy"], 4),
        "precision": round(calculated["precision"], 4),
        "recall": round(calculated["recall"], 4),
        "f1": round(calculated["f1"], 4),
        "roc_auc": round(calculated["roc_auc"], 4),
        "confusion_matrix": calculated["confusion_matrix"],
        "standardized_feature_coefficients": {
            feature: round(float(coefficient), 4)
            for feature, coefficient in zip(FEATURES, weights)
        },
        "intercept": round(float(bias), 4),
        "implementation": "Logistic regression trained with NumPy gradient descent.",
        "interpretation_limit": (
            "Synthetic methodological demo only; not valid for decisions about real learners."
        ),
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Saved {METRICS_PATH.name}")


if __name__ == "__main__":
    main()
