#!/usr/bin/env python3
"""Lightweight classical baselines aligned with the transformer evaluation pipeline."""

from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, jaccard_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:  # Optional dependency for gradient boosted trees
    from xgboost import XGBClassifier

    HAS_XGB = True
except ImportError:  # pragma: no cover - optional runtime dependency
    HAS_XGB = False


###############################################################################
# Embedding loading utilities (reused by the transformer pipeline)
###############################################################################


def safe_load(path: Path) -> Optional[np.ndarray]:
    """Loads a pickle file safely, handling truncated/corrupted files."""

    try:
        with path.open("rb") as handle:
            return pickle.load(handle)
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"✗ {path.name}: {exc}")
        return None


def _resolve_embedding_roots(embedding_type: str) -> Tuple[List[Path], List[Path]]:
    """Returns candidate roots for a given embedding type."""

    mapping = {
        "fasttext": ["fasttext", "fasttext_embeddings", ""],
        "word2vec": ["word2vec", "word2vec_embeddings"],
        "logbert": ["logbert", "logbert_embeddings"],
    }

    subdirs = mapping.get(embedding_type, [""])
    search_order: List[Path] = []
    seen: set[str] = set()

    for subdir in subdirs:
        candidate = Path("embeddings") / subdir if subdir else Path("embeddings")
        key = str(candidate.resolve()) if candidate.exists() else str(candidate)
        if key in seen:
            continue
        search_order.append(candidate)
        seen.add(key)

    existing = [path for path in search_order if path.exists()]
    return (existing if existing else search_order, search_order)


def _iter_embedding_dirs(candidate_roots: Iterable[Path]) -> Iterable[Path]:
    """Yields directories that may contain serialized embeddings."""

    for root in candidate_roots:
        if not root.exists() or not root.is_dir():
            continue
        for child in sorted(root.iterdir()):
            if child.is_dir():
                yield child


def _load_embeddings_from_roots(
    embedding_type: str,
    candidate_roots: Iterable[Path],
    target_log_type: Optional[str],
) -> Tuple[Dict[str, np.ndarray], Dict[str, dict]]:
    embeddings: Dict[str, np.ndarray] = {}
    labels: Dict[str, dict] = {}

    def load_single(directory: Path, logical_name: str) -> None:
        if logical_name in embeddings:
            return
        log_pkl = directory / f"log_{logical_name}.pkl"
        label_pkl = directory / f"label_{logical_name}.pkl"
        if not (log_pkl.exists() and label_pkl.exists()):
            return
        print(f"Loading {logical_name} ({embedding_type})...", end=" ")
        log_array = safe_load(log_pkl)
        label_obj = safe_load(label_pkl)
        if isinstance(log_array, np.ndarray) and isinstance(label_obj, dict):
            embeddings[logical_name] = log_array.astype(np.float32)
            labels[logical_name] = label_obj
            print("✓")
        else:
            print("✗")

    if target_log_type:
        slug = target_log_type
        for root in candidate_roots:
            target_dir = root / slug
            if target_dir.exists():
                load_single(target_dir, slug)
        return embeddings, labels

    for directory in _iter_embedding_dirs(candidate_roots):
        load_single(directory, directory.name)

    return embeddings, labels


def _load_embedding_dispatch(embedding_type: str, target_log_type: Optional[str]):
    roots, search_order = _resolve_embedding_roots(embedding_type)
    embeddings, labels = _load_embeddings_from_roots(embedding_type, roots, target_log_type)

    if target_log_type and target_log_type not in embeddings:
        searched = [str((root / target_log_type).resolve()) for root in search_order]
        raise FileNotFoundError(
            f"Embeddings for '{target_log_type}' not found for {embedding_type}. Checked: {', '.join(searched)}"
        )

    if not embeddings:
        searched = ", ".join(str(path.resolve()) if path.exists() else str(path) for path in search_order)
        hint = {
            "fasttext": "python src/fasttext_embedding.py --output-subdir fasttext",
            "word2vec": "python src/word2vec_embedding_thesis.py --output-subdir word2vec",
            "logbert": "python src/logbert_embeddings.py --output-subdir logbert",
        }.get(embedding_type, "<embedding script>")
        raise FileNotFoundError(
            f"No valid {embedding_type} embeddings located. Checked directories: {searched}.\n"
            f"Generate embeddings via:\n  {hint}"
        )

    print(f"Loaded {len(embeddings)} log types for {embedding_type}")
    return embeddings, labels


def load_fasttext_embeddings(target_log_type: Optional[str] = None):
    return _load_embedding_dispatch("fasttext", target_log_type)


def load_word2vec_embeddings(target_log_type: Optional[str] = None):
    return _load_embedding_dispatch("word2vec", target_log_type)


def load_logbert_embeddings(target_log_type: Optional[str] = None):
    return _load_embedding_dispatch("logbert", target_log_type)


###############################################################################
# Dataset preparation (no hierarchy, SMOTE, or contamination adjustments)
###############################################################################


@dataclass(frozen=True)
class BaselineConfig:
    """Configuration for classical baselines."""

    test_ratio: float = 0.2
    random_state: int = 42
    include_logistic: bool = True
    include_random_forest: bool = True
    include_xgb: bool = True
    include_linear: bool = True
    n_estimators: int = 200
    max_depth: Optional[int] = None
    xgb_learning_rate: float = 0.1
    xgb_estimators: int = 200
    xgb_max_depth: int = 6


@dataclass
class DatasetBundle:
    """Container for dataset splits and metadata."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    label_names: List[str]


@lru_cache(maxsize=128)
def _memoize_label_names(class_names: Tuple[str, ...]) -> List[str]:
    """Memoizes label names for repeated use (dynamic programming via caching)."""

    return [str(name) for name in class_names]


def _extract_label_matrix(label_dict: Mapping[str, np.ndarray], num_samples: int) -> Tuple[np.ndarray, List[str]]:
    """Returns the label matrix and associated names for a log type."""

    vectors = label_dict.get("vectors") if isinstance(label_dict, Mapping) else None
    classes = label_dict.get("classes") if isinstance(label_dict, Mapping) else None

    if isinstance(vectors, np.ndarray):
        matrix = vectors.astype(np.float32)
        if isinstance(classes, Sequence) and classes:
            label_names = _memoize_label_names(tuple(str(cls) for cls in classes))
        else:
            label_names = [f"label_{idx}" for idx in range(matrix.shape[1])]
        return matrix, label_names

    # Fallback: create a dummy normal label when none are provided
    return np.zeros((num_samples, 1), dtype=np.float32), ["normal"]


def prepare_dataset(
    vectors: np.ndarray,
    label_dict: Mapping[str, np.ndarray],
    config: BaselineConfig,
    sample_size: Optional[int] = None,
) -> DatasetBundle:
    """Splits embeddings into train/test sets without additional balancing."""

    features = vectors.astype(np.float32)
    num_samples = features.shape[0]

    if sample_size and 0 < sample_size < num_samples:
        rng = np.random.default_rng(config.random_state)
        indices = rng.choice(num_samples, size=sample_size, replace=False)
        features = features[indices]
        label_matrix, label_names = _extract_label_matrix(label_dict, sample_size)
        label_matrix = label_matrix[indices]
    else:
        label_matrix, label_names = _extract_label_matrix(label_dict, num_samples)

    label_matrix = label_matrix.astype(np.float32, copy=False)
    if label_matrix.shape[0] != features.shape[0]:
        raise ValueError("Label matrix row count does not match feature matrix")

    stratify_labels = None
    if label_matrix.size > 0:
        row_sums = label_matrix.sum(axis=1)
        if not np.allclose(row_sums, row_sums[0]):  # ensure more than one class state
            stratify_labels = (row_sums > 0).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        label_matrix,
        test_size=config.test_ratio,
        random_state=config.random_state,
        stratify=stratify_labels if stratify_labels is not None and stratify_labels.sum() not in {0, len(stratify_labels)} else None,
    )

    return DatasetBundle(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        label_names=label_names,
    )


###############################################################################
# Model training and evaluation
###############################################################################


def _build_models(config: BaselineConfig) -> Dict[str, object]:
    """Constructs the set of baseline models based on config flags."""

    models: Dict[str, object] = {}

    if config.include_logistic:
        logistic = LogisticRegression(
            penalty="l2",
            solver="saga",
            max_iter=500,
            tol=1e-3,
            C=1.0,
            n_jobs=1,
            random_state=config.random_state,
        )
        models["logistic_regression"] = Pipeline(
            steps=[
                ("scale", StandardScaler()),
                ("clf", MultiOutputClassifier(logistic, n_jobs=1)),
            ]
        )

    if config.include_linear:
        linear = LinearRegression(n_jobs=None) if hasattr(LinearRegression, "n_jobs") else LinearRegression()
        models["linear_regression"] = Pipeline(
            steps=[
                ("scale", StandardScaler()),
                ("reg", MultiOutputRegressor(linear)),
            ]
        )

    if config.include_random_forest:
        rf = RandomForestClassifier(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            n_jobs=-1,
            random_state=config.random_state,
        )
        models["random_forest"] = MultiOutputClassifier(rf, n_jobs=-1)

    if config.include_xgb and HAS_XGB:
        xgb = XGBClassifier(
            learning_rate=config.xgb_learning_rate,
            n_estimators=config.xgb_estimators,
            max_depth=config.xgb_max_depth,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            reg_alpha=0.0,
            tree_method="hist",
            n_jobs=-1,
            random_state=config.random_state,
            objective="binary:logistic",
            eval_metric="logloss",
        )
        models["xgboost"] = MultiOutputClassifier(xgb, n_jobs=-1)

    return models


def _ensure_binary(predictions: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Converts real-valued predictions into binary decisions."""

    if predictions.dtype.kind in {"i", "u", "b"}:
        return predictions.astype(np.int32, copy=False)
    return (predictions >= threshold).astype(np.int32)


def _report_metrics(
    model_name: str,
    log_type: str,
    embedding_type: str,
    label_names: Sequence[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dir: Path,
) -> Dict[str, float]:
    """Prints metrics to console and persists a report mirroring transformer output."""

    y_true = y_true.astype(np.int32, copy=False)
    y_pred = _ensure_binary(y_pred)

    print(f"\nModel: {model_name}")
    print("=== Per-Class Metrics ===")
    print(f"{'Class':<25} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 67)

    class_metrics: List[Tuple[float, float, float]] = []
    for idx, name in enumerate(label_names):
        if idx >= y_true.shape[1]:
            break
        y_true_col = y_true[:, idx]
        y_pred_col = y_pred[:, idx]
        if y_true_col.sum() == 0:
            continue
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true_col,
            y_pred_col,
            average="binary",
            zero_division=0,
        )
        class_metrics.append((prec, rec, f1))
        support = int(y_true_col.sum())
        print(f"{name:<25} {prec:>10.3f} {rec:>10.3f} {f1:>10.3f} {support:>10d}")

    print("\n=== Overall Metrics ===")
    micro_prec, micro_rec, micro_f1, _ = precision_recall_fscore_support(
        y_true.ravel(), y_pred.ravel(), average="micro", zero_division=0
    )
    print(f"Micro-averaged: Precision={micro_prec:.3f}, Recall={micro_rec:.3f}, F1={micro_f1:.3f}")

    macro_prec, macro_rec, macro_f1 = (0.0, 0.0, 0.0)
    if class_metrics:
        macro_prec, macro_rec, macro_f1 = np.mean(class_metrics, axis=0)
        print(f"Macro-averaged: Precision={macro_prec:.3f}, Recall={macro_rec:.3f}, F1={macro_f1:.3f}")

    jaccard = jaccard_score(y_true, y_pred, average="samples", zero_division=0)
    print(f"Jaccard Score (samples): {jaccard:.3f}")

    any_attack_true = (y_true.sum(axis=1) > 0).astype(int)
    any_attack_pred = (y_pred.sum(axis=1) > 0).astype(int)
    if len(np.unique(any_attack_true)) > 1:
        anomaly_prec, anomaly_rec, anomaly_f1, _ = precision_recall_fscore_support(
            any_attack_true, any_attack_pred, average="binary", zero_division=0
        )
        print(
            f"\nAnomaly Detection: Precision={anomaly_prec:.3f}, "
            f"Recall={anomaly_rec:.3f}, F1={anomaly_f1:.3f}"
        )
    else:
        anomaly_prec = anomaly_rec = anomaly_f1 = 0.0

    output_dir.mkdir(exist_ok=True, parents=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    embedding_suffix = f"_{embedding_type}" if embedding_type else ""
    report_path = output_dir / f"baseline_{model_name}_{log_type}{embedding_suffix}_evaluation_{timestamp}.txt"

    with report_path.open("w") as handle:
        handle.write("Baseline Evaluation Report\n")
        handle.write(f"Log Type: {log_type}\n")
        if embedding_type:
            handle.write(f"Embedding Type: {embedding_type}\n")
        handle.write(f"Model: {model_name}\n")
        handle.write(f"Timestamp: {timestamp}\n")
        handle.write(f"Dataset: {len(y_true)} test samples\n")
        handle.write("=" * 50 + "\n\n")

        handle.write("Per-Class Metrics:\n")
        handle.write(f"{'Class':<25} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}\n")
        handle.write("-" * 67 + "\n")
        for idx, name in enumerate(label_names):
            if idx >= y_true.shape[1]:
                break
            y_true_col = y_true[:, idx]
            y_pred_col = y_pred[:, idx]
            if y_true_col.sum() == 0:
                continue
            prec, rec, f1, _ = precision_recall_fscore_support(
                y_true_col,
                y_pred_col,
                average="binary",
                zero_division=0,
            )
            support = int(y_true_col.sum())
            handle.write(f"{name:<25} {prec:>10.3f} {rec:>10.3f} {f1:>10.3f} {support:>10d}\n")

        handle.write("\nOverall Metrics:\n")
        handle.write(
            f"Micro-averaged: Precision={micro_prec:.3f}, Recall={micro_rec:.3f}, F1={micro_f1:.3f}\n"
        )
        if class_metrics:
            handle.write(
                f"Macro-averaged: Precision={macro_prec:.3f}, Recall={macro_rec:.3f}, F1={macro_f1:.3f}\n"
            )
        handle.write(f"Jaccard Score: {jaccard:.3f}\n")
        if len(np.unique(any_attack_true)) > 1:
            handle.write(
                f"Anomaly Detection: Precision={anomaly_prec:.3f}, "
                f"Recall={anomaly_rec:.3f}, F1={anomaly_f1:.3f}\n"
            )

    print(f"\nEvaluation report saved: {report_path}")

    return {
        "micro_precision": float(micro_prec),
        "micro_recall": float(micro_rec),
        "micro_f1": float(micro_f1),
        "macro_precision": float(macro_prec),
        "macro_recall": float(macro_rec),
        "macro_f1": float(macro_f1),
        "jaccard": float(jaccard),
    }


def train_and_evaluate_models(
    bundle: DatasetBundle,
    config: BaselineConfig,
    log_type: str,
    embedding_type: str,
    output_dir: Path,
) -> Dict[str, Dict[str, float]]:
    """Trains all configured models and returns their evaluation metrics."""

    models = _build_models(config)
    results: Dict[str, Dict[str, float]] = {}

    for name, estimator in models.items():
        print(f"\nTraining {name}...")
        estimator.fit(bundle.X_train, bundle.y_train)
        predictions = estimator.predict(bundle.X_test)
        metrics = _report_metrics(
            model_name=name,
            log_type=log_type,
            embedding_type=embedding_type,
            label_names=bundle.label_names,
            y_true=bundle.y_test,
            y_pred=predictions,
            output_dir=output_dir,
        )
        results[name] = metrics

    return results


###############################################################################
# Entry point
###############################################################################


def run_baseline(
    embedding_type: str,
    embeddings: Mapping[str, np.ndarray],
    labels: Mapping[str, Mapping[str, np.ndarray]],
    config: BaselineConfig,
    output_dir: Path,
    target_log_type: Optional[str] = None,
    sample_size: Optional[int] = None,
) -> None:
    """Executes the baseline workflow for all available log types."""

    for log_type, vectors in embeddings.items():
        if target_log_type and log_type != target_log_type:
            continue

        label_dict = labels.get(log_type, {})

        print("\n" + "=" * 60)
        print(f"Processing {log_type} with {embedding_type} embeddings")
        print("=" * 60)

        bundle = prepare_dataset(vectors, label_dict, config, sample_size=sample_size)

        total_samples = vectors.shape[0]
        anomaly_count = int((bundle.y_train.sum(axis=1) > 0).sum() + (bundle.y_test.sum(axis=1) > 0).sum())
        print("Dataset summary:")
        print(f"  Total samples : {total_samples}")
        print(f"  Train samples : {bundle.X_train.shape[0]}")
        print(f"  Test samples  : {bundle.X_test.shape[0]}")
        print(f"  Label count   : {len(bundle.label_names)}")
        print(f"  Anomaly rows  : {anomaly_count}")

        train_and_evaluate_models(bundle, config, log_type, embedding_type, output_dir)


def parse_arguments() -> argparse.Namespace:
    """Parses CLI arguments."""

    parser = argparse.ArgumentParser(description="Classical baselines for transformer comparison")
    parser.add_argument(
        "--embedding-type",
        type=str,
        default="fasttext",
        choices=["fasttext", "word2vec", "logbert", "all"],
        help="Embedding type to evaluate (default: fasttext)",
    )
    parser.add_argument(
        "--log-type",
        type=str,
        default=None,
        help="Optional log type to restrict evaluation",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional subsample size (processes full dataset if not provided)",
    )
    parser.add_argument(
        "--skip-xgb",
        action="store_true",
        help="Disable XGBoost baseline",
    )
    parser.add_argument(
        "--skip-rf",
        action="store_true",
        help="Disable Random Forest baseline",
    )
    parser.add_argument(
        "--skip-logistic",
        action="store_true",
        help="Disable Logistic Regression baseline",
    )
    parser.add_argument(
        "--skip-linear",
        action="store_true",
        help="Disable Linear Regression baseline",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Test set ratio (default: 0.2)",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=200,
        help="Number of estimators for tree-based models",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Maximum depth for tree-based models",
    )
    parser.add_argument(
        "--xgb-estimators",
        type=int,
        default=200,
        help="Number of trees for XGBoost",
    )
    parser.add_argument(
        "--xgb-learning-rate",
        type=float,
        default=0.1,
        help="Learning rate for XGBoost",
    )
    parser.add_argument(
        "--xgb-max-depth",
        type=int,
        default=6,
        help="Tree depth for XGBoost",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    config = BaselineConfig(
        test_ratio=args.test_ratio,
        random_state=args.random_state,
        include_xgb=not args.skip_xgb and HAS_XGB,
        include_random_forest=not args.skip_rf,
        include_logistic=not args.skip_logistic,
        include_linear=not args.skip_linear,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        xgb_estimators=args.xgb_estimators,
        xgb_learning_rate=args.xgb_learning_rate,
        xgb_max_depth=args.xgb_max_depth,
    )

    embedding_order = (
        ["fasttext", "word2vec", "logbert"]
        if args.embedding_type == "all"
        else [args.embedding_type]
    )

    results_dir = Path("results")
    loaders = {
        "fasttext": load_fasttext_embeddings,
        "word2vec": load_word2vec_embeddings,
        "logbert": load_logbert_embeddings,
    }

    for embedding_type in embedding_order:
        print("\n" + "=" * 60)
        print(f"Evaluating embedding: {embedding_type}")
        print("=" * 60)

        loader = loaders.get(embedding_type)
        if loader is None:
            print(f"Skipping unsupported embedding type: {embedding_type}")
            continue

        try:
            embeddings, labels = loader(args.log_type)
        except FileNotFoundError as exc:
            print(f"Skipping {embedding_type}: {exc}")
            continue

        run_baseline(
            embedding_type=embedding_type,
            embeddings=embeddings,
            labels=labels,
            config=config,
            output_dir=results_dir,
            target_log_type=args.log_type,
            sample_size=args.sample_size,
        )


if __name__ == "__main__":
    main()
