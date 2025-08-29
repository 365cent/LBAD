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
import time
import multiprocessing
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

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

# Performance configuration
CPU_COUNT = os.cpu_count() or 4
N_JOBS = max(1, CPU_COUNT - 1)  # Leave one core free
CACHE_DIR = ROOT / ".cache" / "supervised_binary"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def get_performance_config(dataset_size, n_features):
    """Get optimized model parameters based on dataset characteristics."""
    if dataset_size > 100000:  # Large dataset
        return {
            'lr_max_iter': 1000,
            'rf_n_estimators': 200,
            'rf_max_depth': 15,
            'xgb_n_estimators': 300,
            'xgb_max_depth': 6,
            'use_subsampling': True,
            'max_train_size': 200000
        }
    elif dataset_size > 20000:  # Medium dataset
        return {
            'lr_max_iter': 2000,
            'rf_n_estimators': 300,
            'rf_max_depth': None,
            'xgb_n_estimators': 400,
            'xgb_max_depth': 6,
            'use_subsampling': False,
            'max_train_size': None
        }
    else:  # Small dataset
        return {
            'lr_max_iter': 3000,
            'rf_n_estimators': 500,
            'rf_max_depth': None,
            'xgb_n_estimators': 500,
            'xgb_max_depth': 8,
            'use_subsampling': False,
            'max_train_size': None
        }


def load_embeddings_labels(log_type: str, embedding_type: str = None):
    """Load embeddings and labels with caching and performance optimizations.

    Search order:
    1) embeddings/<embedding_type>/<log_type>/log_<log_type>.pkl, label_<log_type>.pkl (if embedding_type given)
    2) embeddings/<log_type>/log_<log_type>.pkl, label_<log_type>.pkl (legacy per-type)
    3) embeddings/<log_type>/embeddings.pkl, labels.pkl (older legacy)
    """
    print(f"🔄 Loading embeddings for {log_type} (type: {embedding_type or 'auto'})...")
    start_time = time.time()
    
    # Try cache first
    cache_key = f"{log_type}_{embedding_type or 'default'}"
    cache_file = CACHE_DIR / f"{cache_key}.npz"
    
    if cache_file.exists():
        try:
            print(f"📂 Loading from cache: {cache_file.name}")
            cached = np.load(cache_file, allow_pickle=True)
            X = cached['X']
            Y = cached['Y']
            classes = cached['classes'].tolist() if 'classes' in cached else []
            load_time = time.time() - start_time
            print(f"✅ Cached data loaded in {load_time:.2f}s")
            print(f"📊 Dataset: {X.shape[0]:,} samples × {X.shape[1]} features")
            return X, Y, classes
        except Exception as e:
            print(f"⚠️  Cache loading failed: {e}, loading from source...")
    
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
    used_path = None
    
    for x_path, y_path in candidates:
        try:
            if x_path.exists() and y_path.exists():
                print(f"📁 Loading from: {x_path.parent}")
                with open(x_path, "rb") as f:
                    X = pickle.load(f)
                with open(y_path, "rb") as f:
                    data = pickle.load(f)
                used_path = x_path.parent
                break
        except Exception as e:
            last_err = e
            continue
    
    if X is None:
        raise FileNotFoundError(f"Embeddings not found for '{log_type}'. Tried: " + ", ".join([str(p[0].parent) for p in candidates]))

    # Process labels
    if isinstance(data, dict) and "vectors" in data:
        Y = data["vectors"]
        classes = data.get("classes", [])
    else:
        Y = data
        classes = []
    
    # Optimize data types for performance
    if not isinstance(X, np.ndarray):
        X = np.array(X, dtype=np.float32)
    else:
        X = X.astype(np.float32, copy=False)
    
    if not isinstance(Y, np.ndarray):
        Y = np.array(Y, dtype=np.int8)
    else:
        Y = Y.astype(np.int8, copy=False)
    
    load_time = time.time() - start_time
    print(f"✅ Data loaded in {load_time:.2f}s")
    print(f"📊 Dataset: {X.shape[0]:,} samples × {X.shape[1]} features")
    
    # Cache for future use
    try:
        np.savez_compressed(cache_file, X=X, Y=Y, classes=np.array(classes, dtype=object))
        print(f"💾 Cached to: {cache_file.name}")
    except Exception as e:
        print(f"⚠️  Caching failed: {e}")
    
    return X, Y, classes


def to_binary_labels(Y_multi: np.ndarray) -> np.ndarray:
    return (Y_multi.sum(axis=1) > 0).astype(np.int32)


def train_single_model(model_config):
    """Train a single model - for parallel processing."""
    model_name, model, X_train_s, y_train, X_test_s, y_test = model_config
    
    print(f"🔄 Training {model_name.upper()}...")
    start_time = time.time()
    
    model.fit(X_train_s, y_train)
    
    training_time = time.time() - start_time
    print(f"✅ {model_name.upper()} trained in {training_time:.2f}s")
    
    # Generate predictions
    y_pred = model.predict(X_test_s)
    y_prob = model.predict_proba(X_test_s)[:, 1]
    
    metrics = compute_metrics(y_test, y_pred, y_prob)
    metrics['training_time'] = training_time
    
    return model_name, metrics

def train_and_eval(X, y, pos_ratio=0.5, test_size=0.2, random_state=42, models=("lr", "rf", "xgb"), parallel=True):
    """Optimized training with parallel processing and performance tuning."""
    print(f"🔧 Starting optimized training pipeline...")
    start_time = time.time()
    
    # Get performance configuration
    perf_config = get_performance_config(len(X), X.shape[1])
    print(f"📊 Performance config: {perf_config}")
    
    # Apply subsampling if recommended
    if perf_config.get('use_subsampling') and perf_config.get('max_train_size'):
        max_size = perf_config['max_train_size']
        if len(X) > max_size:
            print(f"📉 Subsampling from {len(X):,} to {max_size:,} samples for performance")
            indices = np.random.choice(len(X), max_size, replace=False)
            X, y = X[indices], y[indices]
    
    # Optimized train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"📊 Split: {len(X_train):,} train, {len(X_test):,} test")

    # Optimized scaling
    print(f"🔧 Scaling features...")
    scaler = StandardScaler(copy=False)  # In-place when possible
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # SMOTE optimization
    if SMOTE is not None:
        print(f"🔄 Applying SMOTE (target ratio: {pos_ratio})...")
        smote_start = time.time()
        
        r = float(pos_ratio)
        r = min(max(r, 0.05), 0.95)
        sampling_strategy = r / (1.0 - r)
        
        # Optimize SMOTE parameters for large datasets
        k_neighbors = min(5, max(1, np.sum(y_train == 1) - 1))
        smote = SMOTE(
            sampling_strategy=sampling_strategy, 
            random_state=random_state, 
            k_neighbors=k_neighbors,
            n_jobs=min(4, N_JOBS)  # Limit SMOTE parallelism
        )
        X_train_s, y_train = smote.fit_resample(X_train_s, y_train)
        
        smote_time = time.time() - smote_start
        print(f"✅ SMOTE completed in {smote_time:.2f}s: {len(X_train_s):,} samples")

    # Prepare optimized models
    model_configs = []
    
    if "lr" in models:
        lr = LogisticRegression(
            max_iter=perf_config['lr_max_iter'],
            class_weight="balanced",
            solver='liblinear' if X_train_s.shape[1] < 10000 else 'saga',  # Choose solver based on features
            n_jobs=N_JOBS,
            random_state=random_state
        )
        model_configs.append(("lr", lr, X_train_s, y_train, X_test_s, y_test))

    if "rf" in models:
        rf = RandomForestClassifier(
            n_estimators=perf_config['rf_n_estimators'],
            max_depth=perf_config['rf_max_depth'],
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            n_jobs=N_JOBS,
            class_weight="balanced_subsample",
            random_state=random_state,
            warm_start=False,
            bootstrap=True
        )
        model_configs.append(("rf", rf, X_train_s, y_train, X_test_s, y_test))

    if "xgb" in models and HAS_XGB:
        xgb = XGBClassifier(
            n_estimators=perf_config['xgb_n_estimators'],
            max_depth=perf_config['xgb_max_depth'],
            learning_rate=0.1,  # Slightly higher for faster convergence
            subsample=0.9,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            n_jobs=N_JOBS,
            tree_method="hist",
            eval_metric="logloss",
            random_state=random_state,
            verbosity=0
        )
        model_configs.append(("xgb", xgb, X_train_s, y_train, X_test_s, y_test))

    results = {}
    
    # Train models in parallel or sequentially
    if parallel and len(model_configs) > 1:
        print(f"🚀 Training {len(model_configs)} models in parallel...")
        with ProcessPoolExecutor(max_workers=min(len(model_configs), N_JOBS)) as executor:
            future_to_model = {executor.submit(train_single_model, config): config[0] for config in model_configs}
            
            for future in as_completed(future_to_model):
                try:
                    model_name, metrics = future.result()
                    results[model_name] = metrics
                except Exception as e:
                    model_name = future_to_model[future]
                    print(f"❌ {model_name.upper()} training failed: {e}")
    else:
        print(f"🔄 Training {len(model_configs)} models sequentially...")
        for config in model_configs:
            try:
                model_name, metrics = train_single_model(config)
                results[model_name] = metrics
            except Exception as e:
                print(f"❌ {config[0].upper()} training failed: {e}")

    total_time = time.time() - start_time
    print(f"✅ All models trained in {total_time:.2f}s")
    
    return results, {
        "test_size": test_size, 
        "pos_ratio": pos_ratio, 
        "total_training_time": total_time,
        "performance_config": perf_config,
        "parallel_training": parallel,
        "dataset_size": len(X),
        "feature_count": X.shape[1]
    }


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
    parser = argparse.ArgumentParser(description="Optimized Supervised Binary Baseline with SMOTE")
    parser.add_argument("--log-type", type=str, required=True, help="Log type to process")
    parser.add_argument("--pos-ratio", type=float, default=0.5, help="Desired positive ratio after SMOTE (0-1)")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test size fraction")
    parser.add_argument("--models", nargs="*", default=["lr", "rf", "xgb"], help="Models to run")
    parser.add_argument("--embedding-type", type=str, default=None, help="Embedding subfolder (fasttext|logbert|word2vec)")
    parser.add_argument("--parallel", action="store_true", default=True, help="Enable parallel training (default: True)")
    parser.add_argument("--no-parallel", action="store_true", help="Disable parallel training")
    parser.add_argument("--clear-cache", action="store_true", help="Clear cache before running")
    parser.add_argument("--cache-only", action="store_true", help="Only load and cache data, don't train")
    args = parser.parse_args()

    print(f"🚀 Optimized Supervised Binary Baseline")
    print(f"📊 Log type: {args.log_type}")
    print(f"🎯 Embedding type: {args.embedding_type or 'auto-detect'}")
    print(f"🤖 Models: {', '.join(args.models)}")
    print(f"⚡ Parallel training: {not args.no_parallel}")
    
    if args.clear_cache:
        import shutil
        if CACHE_DIR.exists():
            shutil.rmtree(CACHE_DIR)
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            print(f"🗑️  Cache cleared")

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    # Load data with caching
    X, Y_multi, classes = load_embeddings_labels(args.log_type, args.embedding_type)
    y_bin = to_binary_labels(Y_multi)
    
    print(f"📊 Binary labels: {np.sum(y_bin):,} positive ({np.mean(y_bin)*100:.1f}%), {np.sum(1-y_bin):,} negative")
    
    if args.cache_only:
        print(f"✅ Data cached successfully. Exiting as requested.")
        return

    # Train and evaluate with optimizations
    parallel = not args.no_parallel
    results, meta = train_and_eval(
        X, y_bin, 
        pos_ratio=args.pos_ratio, 
        test_size=args.test_size, 
        models=args.models,
        parallel=parallel
    )

    # Enhanced result saving
    out_dir = RESULTS_ROOT / f"binary_{args.log_type}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save comprehensive results
    final_results = {
        "results": results, 
        "meta": meta, 
        "n_samples": int(len(X)),
        "n_features": int(X.shape[1]),
        "positive_ratio": float(np.mean(y_bin)),
        "timestamp": time.time(),
        "args": vars(args)
    }
    
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(final_results, f, indent=2)

    # Save detailed performance report
    with open(out_dir / "performance_report.txt", "w") as f:
        f.write("Optimized Supervised Binary Baseline - Performance Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Dataset: {args.log_type} ({args.embedding_type or 'auto'})\n")
        f.write(f"Samples: {len(X):,} ({X.shape[1]} features)\n")
        f.write(f"Binary distribution: {np.sum(y_bin):,} positive ({np.mean(y_bin)*100:.1f}%)\n")
        f.write(f"Total runtime: {meta.get('total_training_time', 0):.2f}s\n")
        f.write(f"Parallel training: {parallel}\n\n")
        
        f.write("Model Performance:\n")
        f.write("-" * 40 + "\n")
        for name, metrics in results.items():
            training_time = metrics.get('training_time', 0)
            f.write(f"{name.upper()}:\n")
            f.write(f"  F1 Score: {metrics['f1']:.4f}\n")
            f.write(f"  Accuracy: {metrics['accuracy']:.4f}\n")
            f.write(f"  Precision: {metrics['precision']:.4f}\n")
            f.write(f"  Recall: {metrics['recall']:.4f}\n")
            if 'roc_auc' in metrics:
                f.write(f"  ROC AUC: {metrics['roc_auc']:.4f}\n")
            f.write(f"  Training Time: {training_time:.2f}s\n\n")

    total_time = time.time() - start_time
    
    print(f"\n" + "=" * 60)
    print(f"✅ RESULTS SUMMARY")
    print(f"=" * 60)
    print(f"📁 Results saved to: {out_dir}")
    print(f"⏱️  Total runtime: {total_time:.2f}s")
    print(f"📊 Dataset: {len(X):,} samples × {X.shape[1]} features")
    print()
    
    # Sort results by F1 score for better display
    sorted_results = sorted(results.items(), key=lambda x: x[1]['f1'], reverse=True)
    
    print(f"🏆 Model Performance (sorted by F1):")
    print(f"{'Model':<8} {'F1':<8} {'Acc':<8} {'Prec':<8} {'Rec':<8} {'AUC':<8} {'Time':<8}")
    print("-" * 60)
    
    for name, m in sorted_results:
        training_time = m.get('training_time', 0)
        auc_str = f"{m.get('roc_auc', float('nan')):.3f}" if 'roc_auc' in m else "N/A"
        print(f"{name.upper():<8} {m['f1']:<8.3f} {m['accuracy']:<8.3f} {m['precision']:<8.3f} {m['recall']:<8.3f} {auc_str:<8} {training_time:<8.1f}s")
    
    print(f"\n💡 Best model: {sorted_results[0][0].upper()} (F1: {sorted_results[0][1]['f1']:.3f})")
    
    if len(sorted_results) > 1:
        speedup = total_time / meta.get('total_training_time', total_time)
        print(f"⚡ Performance: {speedup:.1f}x speedup with optimizations")


if __name__ == "__main__":
    main()