#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Supervised Binary Baseline with SMOTE

- Converts multi-label labels into binary: malicious (any attack) vs normal (no attack)
- Applies SMOTE on the training set to control minority ratio
- Trains LR, RF, and XGBoost classifiers and evaluates on a held-out test set
- Saves reports and metrics under results/binary_<log_type>/
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pickle

# Ensure Matplotlib can write cache/config on HPC
try:
    if "MPLCONFIGDIR" not in os.environ:
        _mpl_dir = Path.cwd() / ".mplconfig"
        _mpl_dir.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(_mpl_dir)
except Exception:
    pass
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    from imblearn.over_sampling import SMOTE
except Exception:
    SMOTE = None

ROOT = Path(__file__).resolve().parent.parent
EMBEDDINGS_DIR = ROOT / "embeddings"
RESULTS_ROOT = ROOT / "results"


def load_embeddings_labels(log_type: str, embedding_type: str = None):
    """Load embeddings and labels, supporting method-specific subfolders.

    Search order:
    1) embeddings/<embedding_type>/<log_type>/log_<log_type>.pkl, label_<log_type>.pkl (if embedding_type given)
    2) embeddings/<log_type>/log_<log_type>.pkl, label_<log_type>.pkl (legacy per-type)
    3) embeddings/<log_type>/embeddings.pkl, labels.pkl (older legacy)
    """
    candidates = []
    if embedding_type:
        base = EMBEDDINGS_DIR / embedding_type / log_type
        candidates.append((base / f"log_{log_type}.pkl", base / f"label_{log_type}.pkl"))
    # Legacy per-type
    base_legacy = EMBEDDINGS_DIR / log_type
    candidates.append((base_legacy / f"log_{log_type}.pkl", base_legacy / f"label_{log_type}.pkl"))
    # Older legacy names
    candidates.append((base_legacy / "embeddings.pkl", base_legacy / "labels.pkl"))

    X = Y = classes = None
    last_err = None
    for x_path, y_path in candidates:
        try:
            if x_path.exists() and y_path.exists():
                with open(x_path, "rb") as f:
                    X = pickle.load(f)
                with open(y_path, "rb") as f:
                    data = pickle.load(f)
                break
        except Exception as e:
            last_err = e
            continue
    if X is None:
        raise FileNotFoundError(f"Embeddings not found for '{log_type}'. Tried: " + ", ".join([str(p[0].parent) for p in candidates]))

    if isinstance(data, dict) and "vectors" in data:
        Y = data["vectors"]
        classes = data.get("classes", [])
    else:
        Y = data
        classes = []
    if not isinstance(X, np.ndarray):
        X = np.array(X, dtype=np.float32)
    else:
        X = X.astype(np.float32)
    if not isinstance(Y, np.ndarray):
        Y = np.array(Y, dtype=np.int8)
    else:
        Y = Y.astype(np.int8)
    return X, Y, classes


def to_binary_labels(Y_multi: np.ndarray) -> np.ndarray:
    return (Y_multi.sum(axis=1) > 0).astype(np.int32)


def train_and_eval(X, y, pos_ratio=0.5, test_size=0.2, random_state=42, models=("lr", "rf", "xgb")):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # SMOTE on training set only
    if SMOTE is not None:
        # sampling_strategy is the desired ratio minority/majority
        # Convert desired positive ratio r = pos/(pos+neg) to minority/majority ratio: r/(1-r)
        r = float(pos_ratio)
        r = min(max(r, 0.05), 0.95)
        sampling_strategy = r / (1.0 - r)
        smote = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state, k_neighbors=5)
        X_train_s, y_train = smote.fit_resample(X_train_s, y_train)

    results = {}

    if "lr" in models:
        lr = LogisticRegression(max_iter=2000, class_weight="balanced")
        lr.fit(X_train_s, y_train)
        y_pred = lr.predict(X_test_s)
        y_prob = lr.predict_proba(X_test_s)[:, 1]
        results["lr"] = compute_metrics(y_test, y_pred, y_prob)

    if "rf" in models:
        rf = RandomForestClassifier(n_estimators=300, max_depth=None, n_jobs=-1, class_weight="balanced_subsample", random_state=random_state)
        rf.fit(X_train_s, y_train)
        y_pred = rf.predict(X_test_s)
        y_prob = rf.predict_proba(X_test_s)[:, 1]
        results["rf"] = compute_metrics(y_test, y_pred, y_prob)

    if "xgb" in models and HAS_XGB:
        xgb = XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            n_jobs=-1,
            tree_method="hist",
            eval_metric="logloss",
            random_state=random_state,
        )
        xgb.fit(X_train_s, y_train)
        y_pred = xgb.predict(X_test_s)
        y_prob = xgb.predict_proba(X_test_s)[:, 1]
        results["xgb"] = compute_metrics(y_test, y_pred, y_prob)

    return results, {"test_size": test_size, "pos_ratio": pos_ratio}


def compute_metrics(y_true, y_pred, y_prob=None):
    m = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
    }
    try:
        if y_prob is not None and len(np.unique(y_true)) > 1:
            m["roc_auc"] = roc_auc_score(y_true, y_prob)
    except Exception:
        pass
    return m


def main():
    parser = argparse.ArgumentParser(description="Supervised binary baseline with SMOTE")
    parser.add_argument("--log-type", type=str, required=True, help="Log type to process")
    parser.add_argument("--pos-ratio", type=float, default=0.5, help="Desired positive ratio after SMOTE (0-1)")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test size fraction")
    parser.add_argument("--models", nargs="*", default=["lr", "rf", "xgb"], help="Models to run")
    parser.add_argument("--embedding-type", type=str, default=None, help="Embedding subfolder (fasttext|logbert|word2vec)")
    args = parser.parse_args()

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    X, Y_multi, classes = load_embeddings_labels(args.log_type, args.embedding_type)
    y_bin = to_binary_labels(Y_multi)

    results, meta = train_and_eval(X, y_bin, pos_ratio=args.pos_ratio, test_size=args.test_size, models=args.models)

    out_dir = RESULTS_ROOT / f"binary_{args.log_type}"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"results": results, "meta": meta, "n_samples": int(len(X))}, f, indent=2)

    print(f"Saved binary baseline results to {out_dir}")
    for name, m in results.items():
        print(f"{name.upper()}: F1={m['f1']:.3f} Acc={m['accuracy']:.3f} Prec={m['precision']:.3f} Rec={m['recall']:.3f} " + (f"AUC={m.get('roc_auc', float('nan')):.3f}" if 'roc_auc' in m else ""))


if __name__ == "__main__":
    main()