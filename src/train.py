"""Train and evaluate the phishing email classifier.

Pipeline = build_features() (TF-IDF + engineered signals) -> LogisticRegression.
Logistic regression is a strong, fast, well-calibrated baseline for sparse text
features and its coefficients stay interpretable, which matters for a security
tool where you want to justify why an email was flagged.

Run:  python -m src.train --data data/emails.csv --out models/phishing_model.joblib
"""

from __future__ import annotations

import argparse
import os
import sys

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

# Allow `python -m src.train` and `python src/train.py` to both work.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.features import build_features  # noqa: E402


def _row_to_text(row: pd.Series) -> str:
    """Assemble one email string from whatever columns are available.

    Supports both the simple synthetic format (a single `text` column) and
    real corpora like CEAS-2008 (`sender` / `subject` / `body` columns).
    """
    if "text" in row and pd.notna(row.get("text")):
        return str(row["text"])
    parts: list[str] = []
    if pd.notna(row.get("sender")):
        parts.append(f"From: {row['sender']}")
    if pd.notna(row.get("subject")):
        parts.append(f"Subject: {row['subject']}")
    if pd.notna(row.get("body")):
        parts.append(str(row["body"]))
    return "\n".join(parts)


def load_data(path: str) -> tuple[list[str], list[int]]:
    if not os.path.exists(path):
        raise SystemExit(
            f"Dataset not found: {path}\n"
            f"Generate a synthetic one:  python data/make_dataset.py --out {path}"
        )
    df = pd.read_csv(path)
    if "label" not in df.columns:
        raise SystemExit(f"Dataset must have a 'label' column. Found: {list(df.columns)}")
    df = df.dropna(subset=["label"])
    # label may be a string like "1"/"0"; coerce and drop anything non-binary.
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df[df["label"].isin([0, 1])]
    texts = df.apply(_row_to_text, axis=1).tolist()
    labels = df["label"].astype(int).tolist()
    return texts, labels


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("features", build_features()),
        ("clf", LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced")),
    ])


def main() -> None:
    ap = argparse.ArgumentParser(description="Train phishing detector")
    ap.add_argument("--data", default="data/emails.csv")
    ap.add_argument("--out", default="models/phishing_model.joblib")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    X, y = load_data(args.data)
    print(f"Loaded {len(X)} emails "
          f"({sum(y)} phishing / {len(y) - sum(y)} legitimate)\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    pipe = build_pipeline()

    # Cross-validated ROC-AUC on the training split for a stability check.
    print("Running 5-fold cross-validation (ROC-AUC) ...")
    cv = cross_val_score(pipe, X_train, y_train, cv=5, scoring="roc_auc")
    print(f"  CV ROC-AUC: {cv.mean():.4f} +/- {cv.std():.4f}\n")

    print("Fitting final model on the training split ...")
    pipe.fit(X_train, y_train)

    # Held-out evaluation.
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]

    print("\n=== Held-out test performance ===")
    print(classification_report(
        y_test, y_pred, target_names=["legitimate", "phishing"], digits=4))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print("\nConfusion matrix (test):")
    print(f"  true legit -> pred legit: {tn:5d} | pred phish: {fp:5d}")
    print(f"  true phish -> pred legit: {fn:5d} | pred phish: {tp:5d}")

    # Refit on ALL data before saving so the shipped model uses every example.
    print("\nRefitting on the full dataset and saving ...")
    pipe.fit(X, y)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    joblib.dump(pipe, args.out)
    print(f"Saved model -> {args.out}")


if __name__ == "__main__":
    main()
