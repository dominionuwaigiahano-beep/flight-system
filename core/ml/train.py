"""
train.py
Trains and evaluates the two models, then saves them as .pkl for runtime use.
Prints the metrics that are quoted in Chapter Four (accuracy, precision, recall,
F1, confusion matrix). Uses a held-out test split so the metrics are honest.
"""
import os
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

HERE = os.path.dirname(__file__)
DATA_DIR = os.path.join(HERE, "..", "..", "data")
MODEL_DIR = os.path.join(HERE, "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

def train_one(csv_name, label_col, model_name, title):
    df = pd.read_csv(os.path.join(DATA_DIR, csv_name))
    X = df.drop(columns=[label_col])
    y = df[label_col]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y)

    clf = RandomForestClassifier(n_estimators=120, max_depth=8,
                                 random_state=42, class_weight="balanced")
    clf.fit(X_tr, y_tr)
    pred = clf.predict(X_te)

    metrics = {
        "model": title,
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "accuracy": round(accuracy_score(y_te, pred), 4),
        "precision": round(precision_score(y_te, pred), 4),
        "recall": round(recall_score(y_te, pred), 4),
        "f1": round(f1_score(y_te, pred), 4),
        "confusion_matrix": confusion_matrix(y_te, pred).tolist(),
        "features": list(X.columns),
        "importances": {c: round(float(i), 4)
                        for c, i in zip(X.columns, clf.feature_importances_)},
    }

    joblib.dump(clf, os.path.join(MODEL_DIR, model_name))
    print(f"\n=== {title} ===")
    print(f"train/test: {metrics['n_train']}/{metrics['n_test']}")
    print(f"accuracy={metrics['accuracy']}  precision={metrics['precision']}  "
          f"recall={metrics['recall']}  f1={metrics['f1']}")
    print("confusion matrix [[TN FP][FN TP]]:", metrics["confusion_matrix"])
    print("feature importances:", metrics["importances"])
    print(classification_report(y_te, pred, digits=3))
    return metrics

if __name__ == "__main__":
    all_metrics = {}
    all_metrics["missed_boarding"] = train_one(
        "missed_boarding.csv", "missed_boarding",
        "missed_boarding_model.pkl", "Missed-Boarding Risk Model")
    all_metrics["flight_delay"] = train_one(
        "flight_delay.csv", "delayed",
        "flight_delay_model.pkl", "Flight-Delay Prediction Model")

    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)
    print("\nModels + metrics.json saved to", os.path.abspath(MODEL_DIR))
