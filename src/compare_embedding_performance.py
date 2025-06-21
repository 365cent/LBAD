#!/usr/bin/env python3
"""Compare classification performance of FastText, LogBERT and KnowLog embeddings."""

from pathlib import Path
import pickle

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

EMB_DIR = Path("embeddings")


def load_embedding(path_prefix: Path):
    with open(path_prefix / "log_embeddings.pkl", "rb") as f:
        X = pickle.load(f)
    with open(path_prefix / "labels.json", "r") as f:
        y = pickle.load(f)
    if not isinstance(X, np.ndarray):
        X = np.array(X)
    return X, y


def evaluate(name: str, X: np.ndarray, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    score = f1_score(y_test, preds, average="weighted")
    print(f"{name} F1-score: {score:.4f}")
    return score


def main():
    results = {}
    for name in ["fasttext", "Spark_Log_Analysis-logbert", "knowlog-bert"]:
        prefix = EMB_DIR / name if name != "fasttext" else EMB_DIR
        try:
            X, y = load_embedding(prefix)
        except FileNotFoundError:
            print(f"Embeddings for {name} not found, skipping")
            continue
        results[name] = evaluate(name, X, y)
    with open("results/embedding_comparison.json", "w") as f:
        import json
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
