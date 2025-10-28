import os
import json
import argparse
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
)
import joblib
import matplotlib.pyplot as plt

RAW_PROCESSED_CSV = "processedtest4.csv"
OUT_MODEL_DIR = "models"
OUT_PLOTS_DIR = "plots"
OUT_RESULTS_DIR = "results"

DEFAULT_MODEL = "logistic" 
DEFAULT_TRAIN_FRAC = 0.8
RANDOM_SEED = 42

def ensure_dirs():
    os.makedirs(OUT_MODEL_DIR, exist_ok=True)
    os.makedirs(OUT_PLOTS_DIR, exist_ok=True)
    os.makedirs(OUT_RESULTS_DIR, exist_ok=True)

def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Processed CSV not found: {path}. Run feature_extraction.py first.")
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    df = df.sort_index()
    return df

def time_train_test_split(df, feature_cols, train_frac):
    n = len(df)
    train_end = int(n * train_frac)
    train = df.iloc[:train_end].copy()
    test = df.iloc[train_end:].copy()

    X_train = train[feature_cols].values
    y_train = train["label"].values
    X_test = test[feature_cols].values
    y_test = test["label"].values

    test_dates = test.index

    return X_train, X_test, y_train, y_test, test_dates, train, test

def save_metrics(metrics, path):
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)

def save_classification_report(report_text, path):
    with open(path, "w") as f:
        f.write(report_text)

def plot_confusion_matrix(cm, labels, outpath):
    plt.figure(figsize=(6,5))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(int(cm[i, j]), "d"),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black"
            )
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def main(args):
    ensure_dirs()
    print("Loading data from:", RAW_PROCESSED_CSV)
    df = load_data(RAW_PROCESSED_CSV)

    reserved = {"Adj Close", "return", "label", "Date"}
    feature_cols = [c for c in df.columns if c not in reserved]
    if len(feature_cols) == 0:
        raise RuntimeError("No feature columns found in processed CSV. Check feature_extraction.py output.")
    print("Using feature columns:", feature_cols)

    X_train, X_test, y_train, y_test, test_dates, train_df, test_df = time_train_test_split(
        df, feature_cols, args.train_frac
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model_name = args.model.lower()
    if model_name == "logistic":
        if args.balance_classes:
            clf = LogisticRegression(multi_class="multinomial", max_iter=2000, class_weight="balanced", random_state=RANDOM_SEED)
        else:
            clf = LogisticRegression(multi_class="multinomial", max_iter=2000, random_state=RANDOM_SEED)
        out_model_path = os.path.join(OUT_MODEL_DIR, "logistic_baseline.pkl")
    elif model_name in ("rf", "random_forest", "randomforest"):
        if args.balance_classes:
            clf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED, class_weight="balanced")
        else:
            clf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED)
        out_model_path = os.path.join(OUT_MODEL_DIR, "rf_baseline.pkl")
    else:
        raise ValueError("Unknown model: choose 'logistic' or 'rf'")

    print(f"Training {model_name} model...")
    clf.fit(X_train_scaled, y_train)

    y_pred = clf.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
    cls_report_text = classification_report(y_test, y_pred, labels=clf.classes_, zero_division=0)

    # additional numeric metrics: per-class precision/recall/f1
    prec, rec, f1, sup = precision_recall_fscore_support(y_test, y_pred, labels=clf.classes_, zero_division=0)

    metrics = {
        "model": model_name,
        "train_fraction": args.train_frac,
        "num_features": len(feature_cols),
        "num_train_rows": len(train_df),
        "num_test_rows": len(test_df),
        "accuracy_test": float(acc),
        "classes": list(clf.classes_),
        "per_class_metrics": {
            cls: {"precision": float(p), "recall": float(r), "f1": float(f), "support": int(s)}
            for cls, p, r, f, s in zip(clf.classes_, prec, rec, f1, sup)
        },
        "datetime": datetime.utcnow().isoformat() + "Z"
    }

    joblib.dump({"model": clf, "scaler": scaler, "feature_cols": feature_cols}, out_model_path)
    print("Saved model to:", out_model_path)

    cm_path = os.path.join(OUT_PLOTS_DIR, "confusion_matrix.png")
    plot_confusion_matrix(cm, labels=list(clf.classes_), outpath=cm_path)
    print("Saved confusion matrix to:", cm_path)

    metrics_path = os.path.join(OUT_RESULTS_DIR, "metrics.json")
    save_metrics(metrics, metrics_path)
    print("Saved metrics to:", metrics_path)

    report_path = os.path.join(OUT_RESULTS_DIR, "classification_report.txt")
    save_classification_report(cls_report_text, report_path)
    print("Saved classification report to:", report_path)

    print("\n--- Test accuracy:", acc)
    print("\nClassification report:\n")
    print(cls_report_text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train baseline classifier on processed AAPL data")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, choices=["logistic", "rf"], help="Model type")
    parser.add_argument("--train-frac", type=float, default=DEFAULT_TRAIN_FRAC, help="Fraction of data for training (time-based)")
    parser.add_argument("--balance-classes", action="store_true", help="Use class_weight='balanced' to handle class imbalance")
    args = parser.parse_args()
    main(args)
