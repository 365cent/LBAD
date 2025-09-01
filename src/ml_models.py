#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Multi-Label ML Baseline for Log Analysis
----------------------------------------
Traditional ML methods adapted for multi-label classification 
to provide baseline comparison with transformer models.
"""

import os
import sys
import json
import pickle
import argparse
import time
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.metrics import (
    classification_report, f1_score, precision_score, recall_score,
    hamming_loss, jaccard_score, accuracy_score, multilabel_confusion_matrix
)
from sklearn.model_selection import train_test_split
# Ensure Matplotlib can write cache/config on HPC BEFORE importing matplotlib
try:
    if "MPLCONFIGDIR" not in os.environ:
        _mpl_dir = Path.cwd() / ".mplconfig"
        _mpl_dir.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(_mpl_dir)
except Exception:
    pass
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

# Try XGBoost with fallback
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("XGBoost not available, skipping XGB model")

# Project paths
ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / 'processed'
EMBEDDINGS_DIR = ROOT / 'embeddings'
MODELS_DIR = ROOT / 'models'
RESULTS_DIR = ROOT / 'results'

# Create directories if they don't exist
for dir_path in [MODELS_DIR, RESULTS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Optimization for multiple cores
CPU_COUNT = os.cpu_count()
if CPU_COUNT:
    N_JOBS = max(1, CPU_COUNT - 1)
else:
    N_JOBS = -1

def _json_default(obj):
    """JSON serializer for non-standard types (e.g., numpy types/arrays)."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)

def check_existing_predictions(log_type, model_name, embedding_type=None, force_restart=False):
    """Check if predictions already exist for a specific model and log type."""
    if force_restart:
        return None, None
    
    # Create results directory path
    results_dir = RESULTS_DIR / f"multilabel_{log_type}"
    
    # Look for existing prediction files
    prediction_file = results_dir / f"{model_name}_predictions.pkl"
    metrics_file = results_dir / f"{model_name}_metrics.pkl"
    
    if prediction_file.exists() and metrics_file.exists():
        try:
            # Load existing predictions and metrics
            with open(prediction_file, 'rb') as f:
                existing_data = pickle.load(f)
            
            with open(metrics_file, 'rb') as f:
                existing_metrics = pickle.load(f)
            
            # Check if the data matches current configuration
            if (existing_data.get('log_type') == log_type and
                existing_data.get('model_name') == model_name and
                existing_data.get('embedding_type') == embedding_type):
                
                print(f"✅ Found existing predictions for {model_name} on {log_type}")
                print(f"   📁 Loading from: {prediction_file}")
                return existing_data, existing_metrics
            
        except Exception as e:
            print(f"⚠️  Could not load existing predictions for {model_name}: {e}")
    
    return None, None

def save_predictions_and_metrics(log_type, model_name, predictions, metrics, embedding_type=None):
    """Save predictions and metrics in the same style as transformer outputs."""
    try:
        # Match transformer-style directory
        results_dir = RESULTS_DIR
        results_dir.mkdir(parents=True, exist_ok=True)

        # Timestamped file like transformer evaluation
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        report_path = results_dir / f"hierarchical_{log_type}_evaluation_{timestamp}.txt"

        # Write concise report to align with transformer
        with open(report_path, 'w') as f:
            f.write(f"Hierarchical Transformer Evaluation Report (Baselines)\n")
            f.write(f"Log Type: {log_type}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Dataset: {len(predictions['y_pred'])} test samples\n")
            f.write("="*50 + "\n\n")

            f.write("Overall Metrics:\n")
            f.write(f"Subset Accuracy: {metrics['subset_accuracy']:.4f}\n")
            f.write(f"Hamming Loss: {metrics['hamming_loss']:.4f}\n")
            f.write(f"Micro F1: {metrics['micro_f1']:.4f}\n")
            f.write(f"Macro F1: {metrics['macro_f1']:.4f}\n")
            f.write(f"Weighted F1: {metrics['weighted_f1']:.4f}\n")
            f.write(f"Jaccard (Micro): {metrics['jaccard_micro']:.4f}\n")
            f.write(f"Jaccard (Macro): {metrics['jaccard_macro']:.4f}\n")

        print(f"Saved baseline report: {report_path}")
    except Exception as e:
        print(f"⚠️  Could not save baseline report: {e}")

def create_multilabel_models(n_labels, random_state=42, large_dataset=False, skip_knn=False, skip_lr=False, has_xgb=False, fast=False):
    """Create multi-label ML models.
    If fast=True, return only a lightweight Logistic Regression baseline and simplify hyperparameters.
    """
    models = {}

    # Fast path: LR only
    if fast:
        models['lr'] = MultiOutputClassifier(
            LogisticRegression(solver='saga', max_iter=50, tol=1e-2),
            n_jobs=N_JOBS
        )
        return models

    # Random Forest (defaults, but parallelized and slightly lighter for safety)
    models['rf'] = MultiOutputClassifier(
        RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
            n_jobs=N_JOBS
        ),
        n_jobs=N_JOBS
    )

    # Logistic Regression (defaults)
    if not skip_lr:
        models['lr'] = MultiOutputClassifier(
            LogisticRegression(solver='saga', max_iter=200, tol=1e-3),
            n_jobs=N_JOBS
        )

    # K-Nearest Neighbors (defaults)
    if not skip_knn:
        models['knn'] = MultiOutputClassifier(KNeighborsClassifier(n_neighbors=3), n_jobs=N_JOBS)

    # XGBoost (defaults if available)
    if has_xgb and HAS_XGBOOST:
        models['xgb'] = MultiOutputClassifier(XGBClassifier(n_estimators=100, tree_method='hist', n_jobs=N_JOBS), n_jobs=N_JOBS)

    return models

def find_available_log_types(embedding_type: str = None):
    """Find available log types from embeddings directory."""
    log_types = set()
    
    # For backward compatibility, check both old and new formats
    if embedding_type:
        # New format: embeddings/<embedding_type>/<log_type>/
        base = EMBEDDINGS_DIR / embedding_type
        if base.exists():
            for log_dir in base.iterdir():
                if log_dir.is_dir():
                    name = log_dir.name
                    # Check for both old and new naming conventions
                    if ((log_dir / f"log_{name}.pkl").exists() and (log_dir / f"label_{name}.pkl").exists()) or \
                       ((log_dir / "embeddings.pkl").exists() and (log_dir / "labels.pkl").exists()):
                        log_types.add(name)
    else:
        # Legacy format: embeddings/<log_type>/
        if EMBEDDINGS_DIR.exists():
            for path in EMBEDDINGS_DIR.iterdir():
                if path.is_dir():
                    # Check for embeddings.pkl and labels.pkl in direct subdirs
                    if (path / "embeddings.pkl").exists() and (path / "labels.pkl").exists():
                        log_types.add(path.name)
                    # Also check for log_<type>.pkl format
                    elif (path / f"log_{path.name}.pkl").exists() and (path / f"label_{path.name}.pkl").exists():
                        log_types.add(path.name)
    
    return sorted(list(log_types))

def load_multilabel_data(log_type, embedding_type: str = None):
    """Load embeddings and multi-label data."""
    print(f"Loading {log_type} data...")
    
    # Candidate search order - check both old and new formats
    candidates = []
    if embedding_type:
        base = EMBEDDINGS_DIR / embedding_type / log_type
        # New format
        candidates.append((base / 'embeddings.pkl', base / 'labels.pkl'))
        # Old format
        candidates.append((base / f'log_{log_type}.pkl', base / f'label_{log_type}.pkl'))
    
    # Legacy format (direct under embeddings/)
    legacy = EMBEDDINGS_DIR / log_type
    candidates.append((legacy / 'embeddings.pkl', legacy / 'labels.pkl'))
    candidates.append((legacy / f'log_{log_type}.pkl', legacy / f'label_{log_type}.pkl'))
    
    embeddings = label_data = None
    used_path = None
    for x_path, y_path in candidates:
        if x_path.exists() and y_path.exists():
            with open(x_path, 'rb') as f:
                embeddings = pickle.load(f)
            with open(y_path, 'rb') as f:
                label_data = pickle.load(f)
            used_path = x_path.parent
            print(f"✅ Loaded data from: {used_path}")
            break
    
    if embeddings is None:
        raise FileNotFoundError(f"Embeddings for {log_type} not found. Searched in: " + ", ".join([str(p[0].parent) for p in candidates]))
    
    # Try to load attack types description file - check multiple locations
    attack_types_files = []
    if used_path:
        attack_types_files.append(used_path / f'attack_types_{log_type}.txt')
        attack_types_files.append(used_path / 'key.txt')
    attack_types_files.append(EMBEDDINGS_DIR / log_type / f'attack_types_{log_type}.txt')
    
    description_from_file = None
    for attack_types_file in attack_types_files:
        if attack_types_file.exists():
            try:
                with open(attack_types_file, 'r') as f:
                    description_from_file = f.read()
                    print(f"✅ Found attack types description file: {attack_types_file}")
                    break
            except Exception as e:
                print(f"⚠️  Could not read attack types file {attack_types_file}: {e}")
    
    # Extract binary vectors and class names
    if isinstance(label_data, dict) and 'vectors' in label_data:
        binary_vectors = label_data['vectors']
        class_names = label_data.get('classes', [])
        
        # Show description if available
        if 'description' in label_data:
            print(f"Description: {label_data['description']}")
        elif description_from_file:
            print(f"Description from file available")
            
    else:
        # Fallback for old format
        binary_vectors = label_data
        class_names = [f'class_{i}' for i in range(binary_vectors.shape[1])]
    
    # Convert to numpy arrays and ensure proper format for multi-label
    if not isinstance(embeddings, np.ndarray):
        embeddings = np.array(embeddings)
    # Ensure float32 to reduce memory pressure
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32, copy=False)
    if not isinstance(binary_vectors, np.ndarray):
        binary_vectors = np.array(binary_vectors)
    
    # Ensure binary vectors are in correct format (0s and 1s)
    binary_vectors = binary_vectors.astype(np.int8, copy=False)
    
    print(f"Loaded {len(embeddings)} samples with {len(class_names)} classes")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Label matrix shape: {binary_vectors.shape}")
    print(f"Data types: embeddings={embeddings.dtype}, labels={binary_vectors.dtype}")
    
    # Calculate comprehensive label statistics
    labels_per_sample = binary_vectors.sum(axis=1)
    print(f"\nMulti-label Statistics:")
    print(f"  Average labels per sample: {labels_per_sample.mean():.2f}")
    print(f"  Labels per sample range: {labels_per_sample.min()} - {labels_per_sample.max()}")
    print(f"  Samples with no labels: {(labels_per_sample == 0).sum()}")
    print(f"  Samples with multiple labels: {(labels_per_sample > 1).sum()}")
    
    # Class frequency and distribution
    class_freq = binary_vectors.sum(axis=0)
    print(f"\nClass frequencies (each class is independent):")
    for i, (cls, freq) in enumerate(zip(class_names, class_freq)):
        percentage = freq/len(binary_vectors)*100
        print(f"  Column {i}: {cls:<15} {freq:>6} samples ({percentage:>5.1f}%)")
    
    # Show some example combinations
    print(f"\nExample label combinations:")
    unique_combinations = []
    for i in range(min(10, len(binary_vectors))):
        combo = binary_vectors[i]
        if not any((combo == uc).all() for uc in unique_combinations):
            unique_combinations.append(combo)
            active_classes = [class_names[j] for j, val in enumerate(combo) if val == 1]
            if not active_classes:
                active_classes = ['normal/no_attack']
            print(f"  {combo} -> {', '.join(active_classes)}")
        if len(unique_combinations) >= 5:
            break
    
    return embeddings, binary_vectors, class_names

def calculate_multilabel_metrics(y_true, y_pred, y_prob=None):
    """Calculate comprehensive multi-label metrics."""
    metrics = {}
    
    # Basic multi-label metrics
    metrics['hamming_loss'] = hamming_loss(y_true, y_pred)
    metrics['subset_accuracy'] = accuracy_score(y_true, y_pred)
    
    # F1 scores
    metrics['micro_f1'] = f1_score(y_true, y_pred, average='micro', zero_division=0)
    metrics['macro_f1'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['weighted_f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['samples_f1'] = f1_score(y_true, y_pred, average='samples', zero_division=0)
    
    # Precision and Recall
    metrics['micro_precision'] = precision_score(y_true, y_pred, average='micro', zero_division=0)
    metrics['macro_precision'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['micro_recall'] = recall_score(y_true, y_pred, average='micro', zero_division=0)
    metrics['macro_recall'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    
    # Jaccard similarity
    metrics['jaccard_micro'] = jaccard_score(y_true, y_pred, average='micro', zero_division=0)
    metrics['jaccard_macro'] = jaccard_score(y_true, y_pred, average='macro', zero_division=0)
    
    # Per-class metrics
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    
    # Calculate additional per-class metrics
    per_class_accuracy = []
    per_class_specificity = []
    per_class_fpr = []
    per_class_npv = []
    per_class_tp = []
    per_class_fp = []
    per_class_tn = []
    per_class_fn = []
    
    # Calculate confusion matrix for each class
    multilabel_cm = multilabel_confusion_matrix(y_true, y_pred)
    
    for i in range(y_true.shape[1]):
        cm = multilabel_cm[i]
        tn, fp, fn, tp = cm.ravel()
        
        # Store confusion matrix values
        per_class_tp.append(int(tp))
        per_class_fp.append(int(fp))
        per_class_tn.append(int(tn))
        per_class_fn.append(int(fn))
        
        # Calculate additional metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        per_class_accuracy.append(float(accuracy))
        per_class_specificity.append(float(specificity))
        per_class_fpr.append(float(fpr))
        per_class_npv.append(float(npv))
    
    metrics['per_class'] = {
        'f1': per_class_f1.tolist(),
        'precision': per_class_precision.tolist(),
        'recall': per_class_recall.tolist(),
        'accuracy': per_class_accuracy,
        'specificity': per_class_specificity,
        'false_positive_rate': per_class_fpr,
        'negative_predictive_value': per_class_npv,
        'true_positives': per_class_tp,
        'false_positives': per_class_fp,
        'true_negatives': per_class_tn,
        'false_negatives': per_class_fn
    }
    
    return metrics

def train_evaluate_multilabel_model(model_name, model, X_train, y_train, X_test, y_test, 
                                   class_names, results_dir, log_type=None, embedding_type=None, force_restart=False):
    """Train and evaluate a multi-label model."""
    
    # Check for existing predictions first
    if log_type:
        existing_data, existing_metrics = check_existing_predictions(log_type, model_name, embedding_type, force_restart)
        if existing_data is not None and existing_metrics is not None:
            print(f"⏩ Skipping training for {model_name} - using existing predictions")
            
            # Create report from existing data
            report_path = results_dir / f'{model_name}_multilabel_report.txt'
            with open(report_path, 'w') as f:
                f.write(f"{model_name.upper()} Multi-Label Classification Report (FROM CACHE)\n")
                f.write("=" * 60 + "\n")
                f.write(f"Using cached predictions from: {existing_data.get('timestamp', 'unknown')}\n")
                # Be tolerant to different cache keys
                cached_len = None
                for key in ['predictions', 'y_pred']:
                    if key in existing_data:
                        try:
                            cached_len = len(existing_data[key])
                            break
                        except Exception:
                            pass
                f.write(f"Test samples: {cached_len if cached_len is not None else 'unknown'}\n")
                f.write(f"Number of classes: {len(class_names)}\n\n")
                
                f.write("OVERALL METRICS:\n")
                f.write("-" * 40 + "\n")
                f.write(f"Subset Accuracy: {existing_metrics['subset_accuracy']:.4f}\n")
                f.write(f"Hamming Loss: {existing_metrics['hamming_loss']:.4f}\n")
                f.write(f"Micro F1: {existing_metrics['micro_f1']:.4f}\n")
                f.write(f"Macro F1: {existing_metrics['macro_f1']:.4f}\n")
                f.write(f"Weighted F1: {existing_metrics['weighted_f1']:.4f}\n")
                f.write(f"Samples F1: {existing_metrics['samples_f1']:.4f}\n")
                f.write(f"Jaccard (Micro): {existing_metrics['jaccard_micro']:.4f}\n")
                f.write(f"Jaccard (Macro): {existing_metrics['jaccard_macro']:.4f}\n\n")
                
                f.write("PER-CLASS METRICS:\n")
                f.write("-" * 40 + "\n")
                f.write(f"{'Class':<20} {'F1':<8} {'Precision':<10} {'Recall':<8} {'Accuracy':<10} {'Specificity':<12} {'FPR':<8} {'Support':<8}\n")
                f.write("-" * 90 + "\n")
                
                # Calculate support for each class
                support = y_test.sum(axis=0)
                
                for i, cls_name in enumerate(class_names):
                    f1 = existing_metrics['per_class']['f1'][i]
                    precision = existing_metrics['per_class']['precision'][i]
                    recall = existing_metrics['per_class']['recall'][i]
                    accuracy = existing_metrics['per_class']['accuracy'][i]
                    specificity = existing_metrics['per_class']['specificity'][i]
                    fpr = existing_metrics['per_class']['false_positive_rate'][i]
                    sup = int(support[i])
                    
                    f.write(f"{cls_name:<20} {f1:<8.3f} {precision:<10.3f} {recall:<8.3f} {accuracy:<10.3f} {specificity:<12.3f} {fpr:<8.3f} {sup:<8}\n")
                
                # Add detailed confusion matrix information
                f.write("\nDETAILED PER-CLASS CONFUSION MATRIX:\n")
                f.write("-" * 40 + "\n")
                f.write(f"{'Class':<20} {'TP':<6} {'FP':<6} {'TN':<6} {'FN':<6} {'NPV':<8}\n")
                f.write("-" * 60 + "\n")
                
                for i, cls_name in enumerate(class_names):
                    tp = existing_metrics['per_class']['true_positives'][i]
                    fp = existing_metrics['per_class']['false_positives'][i]
                    tn = existing_metrics['per_class']['true_negatives'][i]
                    fn = existing_metrics['per_class']['false_negatives'][i]
                    npv = existing_metrics['per_class']['negative_predictive_value'][i]
                    
                    f.write(f"{cls_name:<20} {tp:<6} {fp:<6} {tn:<6} {fn:<6} {npv:<8.3f}\n")
            
            print(f"📊 Report created from cache: {report_path}")
            
            # Return cached results
            return {
                'model_name': model_name,
                'training_time': 0.0,  # No training time
                'metrics': existing_metrics,
                'cached': True
            }
    
    print(f"Training {model_name.upper()} model...")
    start_time = time.time()
    
    # Train the model
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # Save the trained model
    model_path = MODELS_DIR / f'multilabel_{model_name}.joblib'
    joblib.dump(model, model_path)
    
    # Generate predictions
    print(f"Generating predictions for {model_name.upper()}...")
    y_pred = model.predict(X_test)
    
    # Get prediction probabilities if available
    y_prob = None
    if hasattr(model, "predict_proba"):
        try:
            y_prob = model.predict_proba(X_test)
        except:
            pass
    
    # Calculate metrics
    metrics = calculate_multilabel_metrics(y_test, y_pred, y_prob)
    
    # Write transformer-style simple report
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    report_path = results_dir / f"hierarchical_{log_type or 'unknown'}_evaluation_{timestamp}.txt"
    with open(report_path, 'w') as f:
        f.write(f"Hierarchical Transformer Evaluation Report (Baseline: {model_name})\n")
        f.write(f"Log Type: {log_type or 'unknown'}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Dataset: {len(y_test)} test samples\n")
        f.write("="*50 + "\n\n")
        f.write("Overall Metrics:\n")
        f.write(f"Subset Accuracy: {metrics['subset_accuracy']:.4f}\n")
        f.write(f"Hamming Loss: {metrics['hamming_loss']:.4f}\n")
        f.write(f"Micro F1: {metrics['micro_f1']:.4f}\n")
        f.write(f"Macro F1: {metrics['macro_f1']:.4f}\n")
        f.write(f"Samples F1: {metrics['samples_f1']:.4f}\n")
        f.write(f"Jaccard (Micro): {metrics['jaccard_micro']:.4f}\n")
        f.write(f"Jaccard (Macro): {metrics['jaccard_macro']:.4f}\n")
    
    # Optional visualizations kept the same
    create_multilabel_visualization(y_test, y_pred, class_names, model_name, results_dir)
    
    # Save predictions and metrics for future reuse
    if log_type:
        predictions_data = {
            'y_pred': y_pred,
            'y_prob': y_prob
        }
        save_predictions_and_metrics(log_type, model_name, predictions_data, metrics, embedding_type)
    
    return {
        'model_name': model_name,
        'training_time': training_time,
        'metrics': metrics,
        'cached': False
    }

def create_multilabel_visualization(y_true, y_pred, class_names, model_name, results_dir):
    """Create visualizations for multi-label results."""
    
    # 1. Per-class performance heatmap
    plt.figure(figsize=(12, 8))
    
    # Calculate per-class metrics
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    support = y_true.sum(axis=0)
    
    # Create metrics matrix
    metrics_matrix = np.array([per_class_precision, per_class_recall, per_class_f1]).T
    
    # Only show top 15 classes by support
    if len(class_names) > 15:
        top_indices = np.argsort(support)[-15:]
        metrics_matrix = metrics_matrix[top_indices]
        display_names = [class_names[i] for i in top_indices]
    else:
        display_names = class_names
    
    sns.heatmap(metrics_matrix, 
                annot=True, 
                fmt='.3f', 
                cmap='RdYlBu_r',
                xticklabels=['Precision', 'Recall', 'F1'],
                yticklabels=display_names)
    
    plt.title(f'{model_name.upper()} - Per-Class Performance')
    plt.tight_layout()
    plt.savefig(results_dir / f'{model_name}_performance_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Label distribution comparison
    plt.figure(figsize=(14, 6))
    
    # True vs Predicted label counts
    true_counts = y_true.sum(axis=0)
    pred_counts = y_pred.sum(axis=0)
    
    x = np.arange(len(class_names))
    width = 0.35
    
    plt.bar(x - width/2, true_counts, width, label='True', alpha=0.8)
    plt.bar(x + width/2, pred_counts, width, label='Predicted', alpha=0.8)
    
    plt.xlabel('Classes')
    plt.ylabel('Frequency')
    plt.title(f'{model_name.upper()} - True vs Predicted Label Distribution')
    plt.xticks(x, class_names, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / f'{model_name}_label_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Labels per sample distribution
    plt.figure(figsize=(10, 6))
    
    true_labels_per_sample = y_true.sum(axis=1)
    pred_labels_per_sample = y_pred.sum(axis=1)
    
    plt.hist(true_labels_per_sample, bins=range(0, max(true_labels_per_sample) + 2), 
             alpha=0.7, label='True', density=True)
    plt.hist(pred_labels_per_sample, bins=range(0, max(pred_labels_per_sample) + 2), 
             alpha=0.7, label='Predicted', density=True)
    
    plt.xlabel('Number of Labels per Sample')
    plt.ylabel('Density')
    plt.title(f'{model_name.upper()} - Labels per Sample Distribution')
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / f'{model_name}_labels_per_sample.png', dpi=300, bbox_inches='tight')
    plt.close()

def train_traditional_xgboost(X_train, y_train, X_test, y_test, class_names, results_dir, 
                             log_type=None, embedding_type=None, force_restart=False):
    """Train XGBoost using traditional MultiOutputClassifier approach."""
    
    # Check for existing predictions first
    if log_type:
        existing_data, existing_metrics = check_existing_predictions(log_type, 'xgboost_traditional', embedding_type, force_restart)
        if existing_data is not None and existing_metrics is not None:
            print(f"⏩ Skipping training for xgboost_traditional - using existing predictions")
            
            # Create report from existing data
            report_path = results_dir / 'xgboost_traditional_report.txt'
            with open(report_path, 'w') as f:
                f.write("XGBoost Traditional Multi-Label Classification Report (FROM CACHE)\n")
                f.write("=" * 60 + "\n")
                f.write(f"Using cached predictions from: {existing_data.get('timestamp', 'unknown')}\n")
                f.write(f"Approach: MultiOutputClassifier with XGBoost\n\n")
                
                f.write("OVERALL METRICS:\n")
                f.write("-" * 40 + "\n")
                f.write(f"Subset Accuracy: {existing_metrics['subset_accuracy']:.4f}\n")
                f.write(f"Hamming Loss: {existing_metrics['hamming_loss']:.4f}\n")
                f.write(f"Micro F1: {existing_metrics['micro_f1']:.4f}\n")
                f.write(f"Macro F1: {existing_metrics['macro_f1']:.4f}\n")
                f.write(f"Micro Precision: {existing_metrics['micro_precision']:.4f}\n")
                f.write(f"Micro Recall: {existing_metrics['micro_recall']:.4f}\n\n")
                
                f.write("PER-CLASS METRICS:\n")
                f.write("-" * 40 + "\n")
                f.write(f"{'Class':<20} {'F1':<8} {'Precision':<10} {'Recall':<8} {'Accuracy':<10} {'Specificity':<12} {'FPR':<8} {'Support':<8}\n")
                f.write("-" * 90 + "\n")
                
                # Calculate support for each class
                support = y_test.sum(axis=0)
                
                for i, cls_name in enumerate(class_names):
                    f1 = existing_metrics['per_class']['f1'][i]
                    precision = existing_metrics['per_class']['precision'][i]
                    recall = existing_metrics['per_class']['recall'][i]
                    accuracy = existing_metrics['per_class']['accuracy'][i]
                    specificity = existing_metrics['per_class']['specificity'][i]
                    fpr = existing_metrics['per_class']['false_positive_rate'][i]
                    sup = int(support[i])
                    f.write(f"{cls_name:<20} {f1:<8.3f} {precision:<10.3f} {recall:<8.3f} {accuracy:<10.3f} {specificity:<12.3f} {fpr:<8.3f} {sup:<8}\n")
                
                # Add detailed confusion matrix information
                f.write("\nDETAILED PER-CLASS CONFUSION MATRIX:\n")
                f.write("-" * 40 + "\n")
                f.write(f"{'Class':<20} {'TP':<6} {'FP':<6} {'TN':<6} {'FN':<6} {'NPV':<8}\n")
                f.write("-" * 60 + "\n")
                
                for i, cls_name in enumerate(class_names):
                    tp = existing_metrics['per_class']['true_positives'][i]
                    fp = existing_metrics['per_class']['false_positives'][i]
                    tn = existing_metrics['per_class']['true_negatives'][i]
                    fn = existing_metrics['per_class']['false_negatives'][i]
                    npv = existing_metrics['per_class']['negative_predictive_value'][i]
                    
                    f.write(f"{cls_name:<20} {tp:<6} {fp:<6} {tn:<6} {fn:<6} {npv:<8.3f}\n")
            
            print(f"📊 Report created from cache: {report_path}")
            
            # Return cached results
            return {
                'model_name': 'xgboost_traditional',
                'training_time': 0.0,  # No training time
                'metrics': existing_metrics,
                'cached': True
            }
    
    print(f"\nTraining traditional XGBoost with MultiOutputClassifier...")
    
    start_time = time.time()
    
    # Use standard MultiOutputClassifier with XGBoost
    # This approach handles imbalanced data better than One-vs-Rest
    model = MultiOutputClassifier(
        XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=N_JOBS,
            tree_method='hist',
            use_label_encoder=False,
            verbosity=0,
            eval_metric='logloss'
        )
    )
    
    # Train the model
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # Generate predictions
    print("Generating XGBoost predictions...")
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = calculate_multilabel_metrics(y_test, y_pred)
    
    # Save detailed results
    report_path = results_dir / 'xgboost_traditional_report.txt'
    with open(report_path, 'w') as f:
        f.write("XGBoost Traditional Multi-Label Classification Report\n")
        f.write("=" * 60 + "\n")
        f.write(f"Total training time: {training_time:.2f} seconds\n")
        f.write(f"Approach: MultiOutputClassifier with XGBoost\n\n")
        
        f.write("OVERALL METRICS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Subset Accuracy: {metrics['subset_accuracy']:.4f}\n")
        f.write(f"Hamming Loss: {metrics['hamming_loss']:.4f}\n")
        f.write(f"Micro F1: {metrics['micro_f1']:.4f}\n")
        f.write(f"Macro F1: {metrics['macro_f1']:.4f}\n")
        f.write(f"Micro Precision: {metrics['micro_precision']:.4f}\n")
        f.write(f"Micro Recall: {metrics['micro_recall']:.4f}\n\n")
        
        f.write("PER-CLASS METRICS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"{'Class':<20} {'F1':<8} {'Precision':<10} {'Recall':<8} {'Accuracy':<10} {'Specificity':<12} {'FPR':<8} {'Support':<8}\n")
        f.write("-" * 90 + "\n")
        
        # Calculate support for each class
        support = y_test.sum(axis=0)
        
        for i, cls_name in enumerate(class_names):
            f1 = metrics['per_class']['f1'][i]
            precision = metrics['per_class']['precision'][i]
            recall = metrics['per_class']['recall'][i]
            accuracy = metrics['per_class']['accuracy'][i]
            specificity = metrics['per_class']['specificity'][i]
            fpr = metrics['per_class']['false_positive_rate'][i]
            sup = int(support[i])
            f.write(f"{cls_name:<20} {f1:<8.3f} {precision:<10.3f} {recall:<8.3f} {accuracy:<10.3f} {specificity:<12.3f} {fpr:<8.3f} {sup:<8}\n")
        
        # Add detailed confusion matrix information
        f.write("\nDETAILED PER-CLASS CONFUSION MATRIX:\n")
        f.write("-" * 40 + "\n")
        f.write(f"{'Class':<20} {'TP':<6} {'FP':<6} {'TN':<6} {'FN':<6} {'NPV':<8}\n")
        f.write("-" * 60 + "\n")
        
        for i, cls_name in enumerate(class_names):
            tp = metrics['per_class']['true_positives'][i]
            fp = metrics['per_class']['false_positives'][i]
            tn = metrics['per_class']['true_negatives'][i]
            fn = metrics['per_class']['false_negatives'][i]
            npv = metrics['per_class']['negative_predictive_value'][i]
            
            f.write(f"{cls_name:<20} {tp:<6} {fp:<6} {tn:<6} {fn:<6} {npv:<8.3f}\n")
    
    print(f"XGBoost traditional report saved to: {report_path}")
    
    # Save the trained model
    model_path = MODELS_DIR / f'xgboost_traditional.joblib'
    joblib.dump(model, model_path)
    
    # Save predictions and metrics for future reuse
    if log_type:
        predictions_data = {
            'y_pred': y_pred,
            'y_prob': None  # XGBoost traditional doesn't provide probabilities in this format
        }
        save_predictions_and_metrics(log_type, 'xgboost_traditional', predictions_data, metrics, embedding_type)
    
    return {
        'model_name': 'xgboost_traditional',
        'training_time': training_time,
        'metrics': metrics,
        'cached': False
    }



def main():
    """Main function for multi-label ML baseline with a simple CLI (like transformer.py)."""
    parser = argparse.ArgumentParser(description='Multi-Label ML Baseline for Log Analysis')
    parser.add_argument('--embedding_type', '--embedding-type', dest='embedding_type', type=str, default='all',
                        choices=['all', 'logbert', 'fasttext', 'word2vec'],
                        help='Type of embeddings to use (default: all)')
    parser.add_argument('--log_type', '--log-type', dest='log_type', type=str, default=None,
                        help='Specific log type to process (processes all if not specified)')
    parser.add_argument('--sample_size', '--sample-size', dest='sample_size', type=int, default=None,
                        help='Optional subsample size BEFORE splitting (processes full dataset by default)')
    parser.add_argument('--fast', action='store_true',
                        help='Fast pass baseline (LR only, minimal visuals)')
    args = parser.parse_args()

    # Determine which embedding types to process
    embedding_types = ['fasttext', 'word2vec', 'logbert'] if args.embedding_type == 'all' else [args.embedding_type]

    # Fixed settings for simplicity
    test_size = 0.2
    KNN_TRAIN_CAP = 80000
    KNN_TEST_CAP = 30000

    for embedding_type in embedding_types:
        log_types = find_available_log_types(embedding_type)
        if not log_types:
            print(f"No processed log types found for {embedding_type}. Skipping...")
            continue

        # Filter log types if a specific one is requested
        if args.log_type:
            if args.log_type in log_types:
                log_types_to_process = [args.log_type]
            else:
                print(f"Log type '{args.log_type}' not found in {embedding_type}. Available: {log_types}")
                continue
        else:
            log_types_to_process = log_types

        print(f"\n{'='*60}")
        print(f"Processing with {embedding_type} embeddings")
        print(f"Available log types: {log_types}")
        print('='*60)

        for log_type in log_types_to_process:
            print(f"\n{'='*50}")
            print(f"Processing {log_type} ({embedding_type})")
            print('='*50)

            # Create results directory
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            results_dir = RESULTS_DIR / f"multilabel_{log_type}_{timestamp}"
            results_dir.mkdir(exist_ok=True)

            try:
                # Load data
                X, y, class_names = load_multilabel_data(log_type, embedding_type)

                # Optional subsampling BEFORE split (full dataset by default) [[memory:4887039]]
                if args.sample_size and args.sample_size < len(X):
                    rng = np.random.default_rng(42)
                    idx = rng.choice(len(X), size=args.sample_size, replace=False)
                    X = X[idx]
                    y = y[idx]
                    print(f"Subsampled to {len(X)} examples for speed")

                # Skip if no positive labels
                if y.sum() == 0:
                    print(f"No positive labels found for {log_type}, skipping...")
                    continue

                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42, stratify=None
                )

                print(f"Train set: {len(X_train)} | Test set: {len(X_test)}")

                # Standardize features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                # Save scaler
                joblib.dump(scaler, MODELS_DIR / f'scaler_{log_type}.joblib')

                # Create models (skip KNN on very large datasets)
                large_dataset = len(X_train) > 300000
                models = create_multilabel_models(
                    len(class_names),
                    large_dataset=large_dataset,
                    skip_knn=large_dataset,
                    skip_lr=False,
                    has_xgb=HAS_XGBOOST,
                    fast=args.fast
                )

                model_names = list(models.keys())
                results = []

                # Train and evaluate each model
                for model_name in model_names:
                    print(f"\n{'-'*40}")
                    print(f"Training {model_name.upper()} for {log_type}")
                    print(f"{'-'*40}")

                    try:
                        if model_name == 'knn':
                            # Cap KNN to avoid extremely long runs
                            if len(X_train_scaled) > KNN_TRAIN_CAP:
                                sel = np.random.choice(len(X_train_scaled), KNN_TRAIN_CAP, replace=False)
                                X_train_local = X_train_scaled[sel]
                                y_train_local = y_train[sel]
                                print(f"KNN train capped to {len(X_train_local)} samples")
                            else:
                                X_train_local = X_train_scaled
                                y_train_local = y_train
                            if len(X_test_scaled) > KNN_TEST_CAP:
                                sel = np.random.choice(len(X_test_scaled), KNN_TEST_CAP, replace=False)
                                X_test_local = X_test_scaled[sel]
                                y_test_local = y_test[sel]
                                print(f"KNN test capped to {len(X_test_local)} samples")
                            else:
                                X_test_local = X_test_scaled
                                y_test_local = y_test
                            result = train_evaluate_multilabel_model(
                                f"{model_name}_{log_type}",
                                models[model_name],
                                X_train_local, y_train_local,
                                X_test_local, y_test_local,
                                class_names, results_dir,
                                log_type, embedding_type, False
                            )
                        else:
                            result = train_evaluate_multilabel_model(
                                f"{model_name}_{log_type}",
                                models[model_name],
                                X_train_scaled, y_train,
                                X_test_scaled, y_test,
                                class_names, results_dir,
                                log_type, embedding_type, False
                            )
                        results.append(result)
                    except Exception as e:
                        print(f"Error training {model_name}: {e}")
                        continue

                # Create concise summary [[memory:4887036]]
                if results:
                    print(f"\n{'='*60}")
                    print(f"SUMMARY FOR {log_type.upper()} ({embedding_type})")
                    print(f"{'='*60}")
                    cached_count = sum(1 for r in results if r.get('cached', False))
                    trained_count = len(results) - cached_count
                    print(f"Models trained: {trained_count}, from cache: {cached_count}")
                    print(f"{'Model':<28} {'Macro F1':<10} {'Micro F1':<10} {'Hamming':<10} {'Time (s)':<10}")
                    print("-" * 74)
                    for result in sorted(results, key=lambda x: x['metrics']['macro_f1'], reverse=True):
                        model_name = result['model_name']
                        metrics = result['metrics']
                        time_taken = result['training_time']
                        print(f"{model_name:<28} {metrics['macro_f1']:<10.4f} {metrics['micro_f1']:<10.4f} {metrics['hamming_loss']:<10.4f} {time_taken:<10.2f}")

                    summary_path = results_dir / 'summary.json'
                    with open(summary_path, 'w') as f:
                        json.dump({
                            'embedding_type': embedding_type,
                            'log_type': log_type,
                            'results': results,
                            'class_names': class_names,
                            'test_size': test_size,
                            'timestamp': timestamp
                        }, f, indent=2)
                    print(f"Results saved to: {results_dir}")

            except Exception as e:
                print(f"Error processing {log_type} ({embedding_type}): {e}")
                import traceback
                traceback.print_exc()
                continue

    print("\nMulti-label baseline evaluation completed!")

if __name__ == '__main__':
    main()