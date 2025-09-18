#!/usr/bin/env python3
"""Classical multi-label baselines for log analysis.

This module provides lightweight, high-performance baselines that mirror the
data handling used by the transformer pipeline. It supports training and
evaluating multiple traditional ML models on pre-computed embeddings with
hierarchy-aware SMOTE-style augmentation to improve rare-class recall and
precision.
"""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    hamming_loss,
    jaccard_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:  # Optional dependency for gradient boosted trees
    from xgboost import XGBClassifier

    HAS_XGB = True
except ImportError:  # pragma: no cover - optional runtime dependency
    HAS_XGB = False


###############################################################################
# Hierarchy definition and helper utilities
###############################################################################

hierarchy: Mapping[str, Mapping[str, Sequence[str]]] = {
    "foothold": {"attacker_http": ["dirb", "webshell_cmd", "webshell_upload"]},
    "escalate": {
        "escalated_command": ["escalated_sudo_session"],
        "attacker_change_user": [],
        "reverse_shell": [],
    },
    "attacker_vpn": {},
    "dnsteal": {
        "dnsteal-received": [],
        "dnsteal-dropped": [],
        "exfiltration-service": [],
    },
}


@dataclass(frozen=True)
class BaselineConfig:
    """Configuration for classical baselines."""

    contamination: float = 0.2
    min_train_anomalies: int = 512
    rare_class_threshold: int = 512
    min_synthetic_target: int = 64
    max_synthetic_multiplier: float = 2.0
    hard_negative_multiplier: float = 1.5
    test_ratio: float = 0.15
    val_ratio: float = 0.15
    random_state: int = 42
    include_logistic: bool = True
    include_random_forest: bool = True
    include_xgb: bool = True
    n_estimators: int = 200
    max_depth: Optional[int] = None
    xgb_learning_rate: float = 0.1
    xgb_estimators: int = 200
    report_top_k: int = 15


@dataclass
class DatasetBundle:
    """Container for dataset splits and metadata."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    label_names: List[str]
    smote_summary: Dict[str, Dict[str, float]]


@dataclass
class ModelMetrics:
    """Evaluation metrics for a baseline model."""

    model_name: str
    micro_f1: float
    macro_f1: float
    weighted_f1: float
    micro_precision: float
    micro_recall: float
    macro_precision: float
    macro_recall: float
    subset_accuracy: float
    jaccard_micro: float
    jaccard_macro: float
    hamming: float
    report: str


@dataclass
class RunSummary:
    """Summary of a full baseline run for a log type and embedding."""

    log_type: str
    embedding_type: str
    metrics: List[ModelMetrics] = field(default_factory=list)
    dataset_stats: Dict[str, float] = field(default_factory=dict)


###############################################################################
# Embedding loading utilities (mirrors transformer.py)
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
# Dataset preparation
###############################################################################


def flatten_hierarchy(tree: Mapping[str, Mapping[str, Sequence[str]]]):
    """Flattens the hierarchy into a generator of node names."""

    for node, children in tree.items():
        yield node
        if isinstance(children, Mapping):
            for child, leaves in children.items():
                yield child
                for leaf in leaves:
                    yield leaf


def create_multilabel_targets(label_dict: Mapping[str, np.ndarray]) -> Optional[np.ndarray]:
    """Aligns label vectors with the flattened hierarchy."""

    vectors = label_dict.get("vectors")
    if vectors is None:
        return None

    node_list = list(flatten_hierarchy(hierarchy))
    target = np.zeros((vectors.shape[0], len(node_list)), dtype=np.float32)
    width = min(vectors.shape[1], target.shape[1])
    target[:, :width] = vectors[:, :width]
    return target


def _concatenate_feature_blocks(vectors: np.ndarray) -> np.ndarray:
    """Returns a flattened feature representation across embedding segments."""

    return vectors.astype(np.float32, copy=False)


def build_parent_lookup(tree: Mapping[str, Mapping[str, Sequence[str]]]) -> Dict[str, Optional[str]]:
    """Creates a lookup from each node to its parent in the hierarchy."""

    parents: Dict[str, Optional[str]] = {}

    def recurse(current: Mapping[str, Mapping[str, Sequence[str]]], parent: Optional[str]) -> None:
        for node, children in current.items():
            parents.setdefault(node, parent)
            if isinstance(children, Mapping):
                recurse(children, node)
                for intermediate, leaves in children.items():
                    parents.setdefault(intermediate, node)
                    for leaf in leaves:
                        parents.setdefault(str(leaf), intermediate)

    recurse(tree, None)
    return parents


def _apply_hierarchy_aware_smote(
    X: np.ndarray,
    y: np.ndarray,
    label_names: List[str],
    config: BaselineConfig,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Dict[str, float]]]:
    """Generates synthetic samples to improve minority coverage."""

    rng = np.random.default_rng(config.random_state)
    parent_lookup = build_parent_lookup(hierarchy)
    name_to_index = {name: idx for idx, name in enumerate(label_names)}

    adaptive_threshold = int(0.05 * len(y)) if len(y) > 0 else config.rare_class_threshold
    rare_threshold = max(config.min_synthetic_target, min(config.rare_class_threshold, adaptive_threshold))

    global_std = np.std(X, axis=0, keepdims=True)
    global_std[global_std < 1e-6] = 1.0
    global_std = global_std.astype(np.float32)

    synthetic_features: List[np.ndarray] = []
    synthetic_targets: List[np.ndarray] = []
    per_class_stats: Dict[str, Dict[str, float]] = {}

    for idx, name in enumerate(label_names):
        positives = np.where(y[:, idx] == 1)[0]
        pos_count = positives.size
        per_class_stats[name] = {"original": float(pos_count)}

        if pos_count == 0 or pos_count > rare_threshold or pos_count < 2:
            continue

        target_cap = min(
            max(pos_count + config.min_synthetic_target, int(pos_count * config.max_synthetic_multiplier)),
            rare_threshold,
        )
        synth_needed = max(0, target_cap - pos_count)
        if synth_needed == 0:
            continue

        pair_idx = rng.choice(positives, size=(synth_needed, 2), replace=True)
        base_a = X[pair_idx[:, 0]]
        base_b = X[pair_idx[:, 1]]
        lam = rng.random(synth_needed, dtype=np.float32)[:, None]

        class_std = np.std(X[positives], axis=0, keepdims=True)
        class_std = np.where(class_std < 1e-6, global_std, class_std).astype(np.float32)
        noise = rng.normal(0.0, 1.0, size=base_a.shape).astype(np.float32) * class_std * 0.02

        X_new = base_a + lam * (base_b - base_a) + noise
        Y_new = np.maximum(y[pair_idx[:, 0]], y[pair_idx[:, 1]]).astype(np.float32)

        synthetic_features.append(X_new)
        synthetic_targets.append(Y_new)

        parent_name = parent_lookup.get(name)
        if parent_name and parent_name in name_to_index:
            parent_idx = name_to_index[parent_name]
            parent_only = np.where((y[:, parent_idx] == 1) & (y[:, idx] == 0))[0]
            if parent_only.size > 0:
                neg_quota = min(parent_only.size, int(synth_needed * config.hard_negative_multiplier))
                if neg_quota > 0:
                    neg_idx = rng.choice(parent_only, size=neg_quota, replace=False)
                    synthetic_features.append(X[neg_idx])
                    synthetic_targets.append(y[neg_idx].astype(np.float32))

        per_class_stats[name]["after_synth"] = per_class_stats[name]["original"] + float(synth_needed)

    if synthetic_features:
        X_aug = np.vstack([X] + synthetic_features)
        y_aug = np.vstack([y] + synthetic_targets)
    else:
        X_aug, y_aug = X, y

    return X_aug, y_aug, per_class_stats


def _rebalance_contamination(
    X: np.ndarray,
    y: np.ndarray,
    config: BaselineConfig,
    label_names: List[str],
    per_class_stats: Dict[str, Dict[str, float]],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Dict[str, float]]]:
    """Matches the desired contamination ratio via oversampling/undersampling."""

    rng = np.random.default_rng(config.random_state + 1)

    normal_mask = y.sum(axis=1) == 0
    anomaly_mask = ~normal_mask
    X_normals, Y_normals = X[normal_mask], y[normal_mask]
    X_anomalies, Y_anomalies = X[anomaly_mask], y[anomaly_mask]

    desired_total = max(len(X), len(X_anomalies) + len(X_normals))
    desired_anomaly = max(config.min_train_anomalies, int(desired_total * config.contamination))
    desired_anomaly = min(desired_anomaly, len(X_anomalies)) if len(X_anomalies) > 0 else 0
    desired_normal = max(1, desired_total - desired_anomaly)

    if len(X_normals) >= desired_normal:
        idx = rng.choice(len(X_normals), size=desired_normal, replace=False)
        X_normals_bal, Y_normals_bal = X_normals[idx], Y_normals[idx]
    else:
        deficit = desired_normal - len(X_normals)
        replicated = X_normals[rng.choice(len(X_normals), size=deficit, replace=True)] if len(X_normals) > 0 else np.zeros((0, X.shape[1]), dtype=np.float32)
        replicated_y = Y_normals[rng.choice(len(Y_normals), size=deficit, replace=True)] if len(Y_normals) > 0 else np.zeros((0, y.shape[1]), dtype=np.float32)
        X_normals_bal = np.vstack([X_normals, replicated]) if len(X_normals) else replicated
        Y_normals_bal = np.vstack([Y_normals, replicated_y]) if len(Y_normals) else replicated_y

    if desired_anomaly == 0 or len(X_anomalies) == 0:
        X_anomalies_bal = np.zeros((0, X.shape[1]), dtype=np.float32)
        Y_anomalies_bal = np.zeros((0, y.shape[1]), dtype=np.float32)
    elif len(X_anomalies) <= desired_anomaly:
        X_anomalies_bal, Y_anomalies_bal = X_anomalies, Y_anomalies
    else:
        idx = rng.choice(len(X_anomalies), size=desired_anomaly, replace=False)
        X_anomalies_bal, Y_anomalies_bal = X_anomalies[idx], Y_anomalies[idx]

    X_balanced = np.vstack([X_normals_bal, X_anomalies_bal]).astype(np.float32)
    y_balanced = np.vstack([Y_normals_bal, Y_anomalies_bal]).astype(np.float32)

    before_counts = y.sum(axis=0)
    after_counts = y_balanced.sum(axis=0)
    for idx, name in enumerate(label_names):
        stats = per_class_stats.setdefault(name, {})
        stats.setdefault("original", float(before_counts[idx]))
        stats["balanced"] = float(after_counts[idx])

    return X_balanced, y_balanced, per_class_stats


def prepare_training_bundle(
    vectors: np.ndarray,
    labels: Optional[np.ndarray],
    config: BaselineConfig,
) -> DatasetBundle:
    """Splits embeddings into train/val/test sets and balances the training set."""

    features = _concatenate_feature_blocks(vectors)
    if labels is None:
        labels = np.zeros((features.shape[0], len(list(flatten_hierarchy(hierarchy)))), dtype=np.float32)

    label_names = list(flatten_hierarchy(hierarchy))[: labels.shape[1]]

    X_temp, X_test, y_temp, y_test = train_test_split(
        features,
        labels,
        test_size=config.test_ratio,
        random_state=config.random_state,
        stratify=(labels.sum(axis=1) > 0).astype(int) if labels.sum() > 0 else None,
    )

    val_size = config.val_ratio / (1.0 - config.test_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=val_size,
        random_state=config.random_state,
        stratify=(y_temp.sum(axis=1) > 0).astype(int) if y_temp.sum() > 0 else None,
    )

    X_aug, y_aug, per_class = _apply_hierarchy_aware_smote(X_train, y_train, label_names, config)
    X_balanced, y_balanced, per_class = _rebalance_contamination(X_aug, y_aug, config, label_names, per_class)

    smote_summary = {
        name: {
            key: float(value)
            for key, value in stats.items()
            if value is not None
        }
        for name, stats in per_class.items()
    }

    return DatasetBundle(
        X_train=X_balanced,
        y_train=y_balanced,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        label_names=label_names,
        smote_summary=smote_summary,
    )


###############################################################################
# Model training and evaluation
###############################################################################


def _build_models(config: BaselineConfig) -> Dict[str, MultiOutputClassifier]:
    """Constructs the set of baseline models based on config flags."""

    models: Dict[str, MultiOutputClassifier] = {}

    if config.include_logistic:
        lr = LogisticRegression(
            penalty="l2",
            solver="saga",
            max_iter=200,
            tol=1e-3,
            C=1.0,
            n_jobs=-1,
            random_state=config.random_state,
        )
        models["logistic"] = MultiOutputClassifier(lr, n_jobs=-1)

    if config.include_random_forest:
        rf = RandomForestClassifier(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            min_samples_leaf=2,
            n_jobs=-1,
            class_weight="balanced_subsample",
            random_state=config.random_state,
        )
        models["random_forest"] = MultiOutputClassifier(rf, n_jobs=-1)

    if config.include_xgb and HAS_XGB:
        xgb = XGBClassifier(
            learning_rate=config.xgb_learning_rate,
            n_estimators=config.xgb_estimators,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            reg_alpha=0.5,
            tree_method="hist",
            n_jobs=-1,
            random_state=config.random_state,
        )
        models["xgboost"] = MultiOutputClassifier(xgb, n_jobs=-1)

    return models


def _fit_pipeline(model: MultiOutputClassifier, X: np.ndarray, y: np.ndarray) -> Pipeline:
    """Creates and fits a standardised pipeline wrapping the estimator."""

    pipeline = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("clf", model),
        ]
    )
    pipeline.fit(X, y)
    return pipeline


def _evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str,
) -> Tuple[float, float, float]:
    """Returns precision, recall, f1 for the requested averaging scheme."""

    precision = precision_score(y_true, y_pred, average=average, zero_division=0)
    recall = recall_score(y_true, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
    return precision, recall, f1


def train_and_evaluate_models(bundle: DatasetBundle, config: BaselineConfig) -> List[ModelMetrics]:
    """Trains all configured models and returns their evaluation metrics."""

    models = _build_models(config)
    metrics: List[ModelMetrics] = []

    for name, estimator in models.items():
        print(f"\nTraining {name}...")
        pipeline = _fit_pipeline(estimator, bundle.X_train, bundle.y_train)
        y_pred = pipeline.predict(bundle.X_test)

        micro_prec, micro_rec, micro_f1 = _evaluate_predictions(bundle.y_test, y_pred, "micro")
        macro_prec, macro_rec, macro_f1 = _evaluate_predictions(bundle.y_test, y_pred, "macro")
        weighted_f1 = f1_score(bundle.y_test, y_pred, average="weighted", zero_division=0)

        report = classification_report(
            bundle.y_test,
            y_pred,
            target_names=bundle.label_names[: bundle.y_test.shape[1]],
            zero_division=0,
        )

        metrics.append(
            ModelMetrics(
                model_name=name,
                micro_f1=micro_f1,
                macro_f1=macro_f1,
                weighted_f1=weighted_f1,
                micro_precision=micro_prec,
                micro_recall=micro_rec,
                macro_precision=macro_prec,
                macro_recall=macro_rec,
                subset_accuracy=accuracy_score(bundle.y_test, y_pred),
                jaccard_micro=jaccard_score(bundle.y_test, y_pred, average="micro", zero_division=0),
                jaccard_macro=jaccard_score(bundle.y_test, y_pred, average="macro", zero_division=0),
                hamming=hamming_loss(bundle.y_test, y_pred),
                report=report,
            )
        )

    return metrics


###############################################################################
# Reporting utilities
###############################################################################


def _write_report(summary: RunSummary, output_dir: Path) -> None:
    """Persists evaluation metrics to a timestamped report."""

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    report_path = output_dir / f"baselines_{summary.log_type}_{summary.embedding_type}_{timestamp}.txt"

    with report_path.open("w") as handle:
        handle.write(f"Baseline Evaluation Report\n")
        handle.write(f"Log Type: {summary.log_type}\n")
        handle.write(f"Embedding: {summary.embedding_type}\n")
        handle.write(f"Timestamp: {timestamp}\n")
        handle.write("=" * 60 + "\n\n")

        handle.write("Overall Dataset Statistics:\n")
        for key, value in summary.dataset_stats.items():
            handle.write(f"  {key}: {value}\n")
        handle.write("\n")

        for metric in summary.metrics:
            handle.write(f"Model: {metric.model_name}\n")
            handle.write(f"  Micro F1       : {metric.micro_f1:.4f}\n")
            handle.write(f"  Macro F1       : {metric.macro_f1:.4f}\n")
            handle.write(f"  Weighted F1    : {metric.weighted_f1:.4f}\n")
            handle.write(f"  Micro Precision: {metric.micro_precision:.4f}\n")
            handle.write(f"  Micro Recall   : {metric.micro_recall:.4f}\n")
            handle.write(f"  Macro Precision: {metric.macro_precision:.4f}\n")
            handle.write(f"  Macro Recall   : {metric.macro_recall:.4f}\n")
            handle.write(f"  Subset Accuracy: {metric.subset_accuracy:.4f}\n")
            handle.write(f"  Jaccard Micro  : {metric.jaccard_micro:.4f}\n")
            handle.write(f"  Jaccard Macro  : {metric.jaccard_macro:.4f}\n")
            handle.write(f"  Hamming Loss   : {metric.hamming:.4f}\n\n")
            handle.write("Per-class report:\n")
            handle.write(metric.report)
            handle.write("\n" + "-" * 60 + "\n\n")

    print(f"Saved baseline report: {report_path}")


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
) -> None:
    """Executes the baseline workflow for all available log types."""

    for log_type, vectors in embeddings.items():
        if target_log_type and log_type != target_log_type:
            continue

        print("\n" + "=" * 60)
        print(f"Processing {log_type} with {embedding_type} embeddings")
        print("=" * 60)

        label_target = create_multilabel_targets(labels.get(log_type, {}))
        bundle = prepare_training_bundle(vectors, label_target, config)

        total_samples = vectors.shape[0]
        anomaly_count = int((label_target.sum(axis=1) > 0).sum()) if label_target is not None else 0
        normal_count = total_samples - anomaly_count

        dataset_stats = {
            "original_samples": total_samples,
            "original_normals": normal_count,
            "original_anomalies": anomaly_count,
            "train_samples": int(bundle.X_train.shape[0]),
            "validation_samples": int(bundle.X_val.shape[0]),
            "test_samples": int(bundle.X_test.shape[0]),
        }

        print("Balanced dataset distribution (post-SMOTE):")
        print(f"  Train samples : {bundle.X_train.shape[0]}")
        print(f"  Validation    : {bundle.X_val.shape[0]}")
        print(f"  Test          : {bundle.X_test.shape[0]}")

        top_k = config.report_top_k
        print("  Per-class adjustments (top changes):")
        adjustments = []
        for name, stats in bundle.smote_summary.items():
            original = stats.get("original", 0.0)
            balanced = stats.get("balanced", original)
            delta = balanced - original
            adjustments.append((name, original, balanced, delta))
        adjustments.sort(key=lambda item: abs(item[3]), reverse=True)
        for name, original, balanced, delta in adjustments[:top_k]:
            trend = "++" if delta > 0 else "--" if delta < 0 else "=="
            print(f"    {name:<25} {balanced:>6.0f} ({trend} {delta:+.0f})")

        metrics = train_and_evaluate_models(bundle, config)
        summary = RunSummary(
            log_type=log_type,
            embedding_type=embedding_type,
            metrics=metrics,
            dataset_stats=dataset_stats,
        )
        _write_report(summary, output_dir)


def parse_arguments() -> argparse.Namespace:
    """Parses CLI arguments."""

    parser = argparse.ArgumentParser(description="Classical multi-label baselines")
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
        "--skip-lr",
        action="store_true",
        help="Disable Logistic Regression baseline",
    )
    parser.add_argument(
        "--contamination",
        type=float,
        default=0.2,
        help="Target contamination ratio for training set balancing",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    config = BaselineConfig(
        contamination=args.contamination,
        include_xgb=not args.skip_xgb and HAS_XGB,
        include_random_forest=not args.skip_rf,
        include_logistic=not args.skip_lr,
        random_state=args.random_state,
    )

    embedding_order = [
        "fasttext",
        "word2vec",
        "logbert",
    ] if args.embedding_type == "all" else [args.embedding_type]

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

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
        )


if __name__ == "__main__":
    import pickle  # Local import to avoid unused when module imported

    main()
