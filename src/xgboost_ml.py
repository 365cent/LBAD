#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
XGBoost Multi-Label Model for Log Anomaly Detection
------------------------------------------------------
Uses a One-vs-Rest (OvR) strategy with XGBoost to predict multi-label
binary vectors (e.g., [0, 1, 0, 0, 1]) indicating active attack types.
Can process either combined embeddings or embeddings for a specific log type.
"""

import os
import pickle
import argparse
import time
import numpy as np
import pandas as pd
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, multilabel_confusion_matrix, hamming_loss
import joblib # For saving/loading models
import json # For potentially loading attack names if not hardcoded

# --- Project Paths & Configuration ---
ROOT = Path(__file__).resolve().parent.parent # Script is in src/, so go up one level for project root

EMBEDDINGS_DIR = ROOT / 'embeddings'
MODELS_BASE_DIR = ROOT / 'models' / 'xgboost_ml' 
RESULTS_BASE_DIR = ROOT / 'results' / 'xgboost_ml'

for dir_path in [MODELS_BASE_DIR, RESULTS_BASE_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# For Apple Silicon optimization: tune thread counts
CPU_COUNT = os.cpu_count()
N_JOBS = max(1, CPU_COUNT - 1) if CPU_COUNT else -1

# --- Attack Definitions (essential for mapping vector indices to attack names) ---
# This dictionary should be consistent with LOG_TYPE_ATTACKS in fasttext_embedding.py
LOG_TYPE_ATTACKS_BASELINE = {
    'dns': ['traceroute', 'dns_scan', 'service_scan', 'dnsteal-received', 'dnsteal-dropped'],
    'network': ['traceroute', 'network_scan', 'dns_scan', 'service_scan', 'port_scan', 'ddos', 'tcp_syn_flood', 'udp_flood', 'icmp_flood'],
    'web': ['webshell_cmd', 'webshell_upload', 'dirb', 'wordpress_database_dump', 'wordpress_scan', 'sql_injection', 'xss', 'directory_traversal', 'command_injection'],
    'error': ['dirb', 'wordpress_scan', 'sql_injection_error', 'auth_failure_reflected_in_error'],
    'monitoring': ['password_cracking', 'malware_execution', 'suspicious_process'],
    'auth': ['login_as_system_user', 'reverse_shell', 'ssh_bruteforce', 'ftp_bruteforce', 'password_cracking'],
    'audit': ['root_command_execution', 'login_as_system_user', 'dnsteal-received', 'dnsteal-dropped', 'privilege_escalation', 'unauthorized_file_access'],
    'ids': ['port_scan_alert', 'web_attack_alert', 'malware_signature_detected', 'ddos_alert', 'bruteforce_alert', 'policy_violation', 'exploit_detected']
}
DEFAULT_ALL_POSSIBLE_LABELS = sorted(list(set(attack for attacks in LOG_TYPE_ATTACKS_BASELINE.values() for attack in attacks)))

def load_data(embeddings_file_x, labels_file_y, num_expected_labels, test_size=0.2, random_state=42):
    """
    Loads embeddings and multi-label binary indicators based on provided paths.
    Performs a train/test split and validates label dimensions.
    """
    print(f"Loading embeddings from: {embeddings_file_x}")
    print(f"Loading multi-label binary indicators from: {labels_file_y}")

    try:
        with open(embeddings_file_x, 'rb') as f:
            X = pickle.load(f)
        with open(labels_file_y, 'rb') as f:
            Y_multi = pickle.load(f)
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        print(f"Please ensure the specified embedding and label files exist for the chosen context.")
        raise

    if not isinstance(X, np.ndarray):
        X = np.array(X)
    if not isinstance(Y_multi, np.ndarray):
        Y_multi = np.array(Y_multi)

    if Y_multi.ndim == 1: 
        print("Warning: Y_multi loaded as 1D array. Attempting to stack into 2D if it contains lists/arrays.")
        try:
            Y_multi = np.array([np.array(row) for row in Y_multi])
            if Y_multi.ndim == 1: 
                raise ValueError("Y_multi could not be converted to a 2D array of labels.")
        except Exception as stack_e:
            print(f"Error converting Y_multi to 2D: {stack_e}")
            raise ValueError(f"Loaded Y_multi (shape {Y_multi.shape}) is not a 2D numpy array of label vectors.")
    
    if Y_multi.ndim != 2:
         raise ValueError(f"Loaded Y_multi (shape {Y_multi.shape}) is not a 2D numpy array as expected for multi-label tasks.")

    if Y_multi.shape[1] != num_expected_labels:
        print(f"CRITICAL Warning: Loaded labels matrix has {Y_multi.shape[1]} columns, but current context expects {num_expected_labels} labels.")
        print(f"  Expected labels for context: {DEFAULT_ALL_POSSIBLE_LABELS if num_expected_labels == len(DEFAULT_ALL_POSSIBLE_LABELS) else 'context-specific list'}")
        print("Ensure that the label file corresponds to the chosen context and its attack definitions in this script.")
        raise ValueError("Label dimension mismatch based on context.")

    print(f"Loaded X shape: {X.shape}")
    print(f"Loaded Y_multi shape: {Y_multi.shape}")

    X_train, X_test, Y_train_multi, Y_test_multi = train_test_split(
        X, Y_multi, test_size=test_size, random_state=random_state, 
        stratify=Y_multi.sum(axis=1) > 0 if Y_multi.shape[0] > 1 and Y_multi.shape[1] > 0 else None 
    )
    print(f"X_train shape: {X_train.shape}, Y_train_multi shape: {Y_train_multi.shape}")
    print(f"X_test shape: {X_test.shape}, Y_test_multi shape: {Y_test_multi.shape}")
    
    return X_train, X_test, Y_train_multi, Y_test_multi

def train_evaluate_ovr_xgboost(X_train, Y_train_multi, X_test, Y_test_multi, current_labels_for_run, results_dir, models_dir):
    """
    Trains one XGBoost binary classifier per label (One-vs-Rest) for the current context.
    Evaluates the multi-label predictions.
    """
    num_labels_for_context = len(current_labels_for_run)
    
    trained_models = []
    training_times = []

    print(f"\nStarting One-vs-Rest XGBoost training for {num_labels_for_context} labels in current context...")

    for i in range(num_labels_for_context):
        label_name = current_labels_for_run[i]
        print(f"  Training model {i+1}/{num_labels_for_context} for label: '{label_name}'...")
        
        y_train_single_label = Y_train_multi[:, i]

        # Check for sufficient samples of both classes for this specific label
        unique_classes, class_counts = np.unique(y_train_single_label, return_counts=True)
        if len(unique_classes) < 2 or np.min(class_counts) < 2: # Need at least 2 samples of each class
            print(f"    Skipping training for '{label_name}' as it has too few samples of one class in the training data for this label.")
            trained_models.append(None) 
            training_times.append(0)
            continue
        
        model_start_time = time.time()
        model = XGBClassifier(
            n_estimators=100, 
            max_depth=7,      
            learning_rate=0.1,
            gamma=0.1,
            random_state=42,
            n_jobs=N_JOBS,
            tree_method='hist',
            use_label_encoder=False, 
            verbosity=0
        )
        
        model.fit(X_train, y_train_single_label)
        current_training_time = time.time() - model_start_time
        training_times.append(current_training_time)
        trained_models.append(model)
        print(f"    Model for '{label_name}' trained in {current_training_time:.2f}s.")

    total_training_time = sum(training_times)
    print(f"Total training time for all OvR models: {total_training_time:.2f}s")

    models_save_path = models_dir / "ovr_xgboost_models.joblib"
    joblib.dump(trained_models, models_save_path)
    print(f"Saved all OvR XGBoost models to: {models_save_path}")

    print("\nPredicting on the test set...")
    Y_pred_multi = np.zeros_like(Y_test_multi, dtype=int)

    for i in range(num_labels_for_context):
        model = trained_models[i]
        if model is not None: 
            Y_pred_multi[:, i] = model.predict(X_test)
        else: 
             Y_pred_multi[:, i] = 0 # Predict 0 if model was skipped

    print("\nEvaluating multi-label predictions...")
    report_path = results_dir / "classification_report.txt"
    metrics_summary_path = results_dir / "metrics_summary.json"

    f1_micro = f1_score(Y_test_multi, Y_pred_multi, average='micro', zero_division=0)
    f1_macro = f1_score(Y_test_multi, Y_pred_multi, average='macro', zero_division=0)
    f1_samples = f1_score(Y_test_multi, Y_pred_multi, average='samples', zero_division=0)
    precision_micro = precision_score(Y_test_multi, Y_pred_multi, average='micro', zero_division=0)
    recall_micro = recall_score(Y_test_multi, Y_pred_multi, average='micro', zero_division=0)
    hamming = hamming_loss(Y_test_multi, Y_pred_multi)

    print(f"  F1 Score (Micro): {f1_micro:.4f}")
    print(f"  F1 Score (Macro): {f1_macro:.4f}")
    print(f"  F1 Score (Samples): {f1_samples:.4f}")
    print(f"  Precision (Micro): {precision_micro:.4f}")
    print(f"  Recall (Micro): {recall_micro:.4f}")
    print(f"  Hamming Loss: {hamming:.4f}")
    
    report = classification_report(Y_test_multi, Y_pred_multi, target_names=current_labels_for_run, zero_division=0)
    print("\nPer-Label Classification Report:")
    print(report)

    context_name_for_report = results_dir.name.replace(f"run_{results_dir.parent.name}_", "").split('_')[0] \
                              if "run_" in results_dir.name else results_dir.name

    with open(report_path, 'w') as f:
        f.write(f"XGBoost Multi-Label (One-vs-Rest) Classification Report\n")
        f.write(f"Context: {context_name_for_report}\n")
        f.write("="*60 + "\n")
        f.write(f"Total training time: {total_training_time:.2f}s\n\n")
        f.write(f"Overall Metrics:\n")
        f.write(f"  F1 Score (Micro): {f1_micro:.4f}\n")
        f.write(f"  F1 Score (Macro): {f1_macro:.4f}\n")
        f.write(f"  F1 Score (Samples): {f1_samples:.4f}\n")
        f.write(f"  Precision (Micro): {precision_micro:.4f}\n")
        f.write(f"  Recall (Micro): {recall_micro:.4f}\n")
        f.write(f"  Hamming Loss: {hamming:.4f}\n\n")
        f.write("Per-Label Classification Report:\n")
        f.write(report)

    metrics_summary = {
        'context': context_name_for_report,
        'total_training_time_seconds': total_training_time,
        'f1_micro': f1_micro, 'f1_macro': f1_macro, 'f1_samples': f1_samples,
        'precision_micro': precision_micro,
        'recall_micro': recall_micro,
        'hamming_loss': hamming
    }
    with open(metrics_summary_path, 'w') as f:
        json.dump(metrics_summary, f, indent=4)
    
    print(f"\nFull classification report saved to: {report_path}")
    print(f"Metrics summary saved to: {metrics_summary_path}")

def main():
    parser = argparse.ArgumentParser(description="XGBoost Multi-Label Model for Log Anomaly Detection.")
    parser.add_argument(
        '--context', type=str, default='all_combined',
        choices=list(LOG_TYPE_ATTACKS_BASELINE.keys()) + ['all_combined'], # Ensure valid choices
        help="Context for embeddings: a specific log type (e.g., 'web', 'dns') or 'all_combined'."
    )
    parser.add_argument(
        '--test_size', type=float, default=0.2,
        help="Proportion of the dataset to include in the test split."
    )
    parser.add_argument(
        '--random_state', type=int, default=42,
        help="Random state for reproducibility."
    )
    args = parser.parse_args()

    print(f"--- XGBoost Multi-Label Model (Context: {args.context}) ---")
    overall_start_time = time.time()

    current_labels_list_for_run = []
    num_labels_for_run = 0
    embeddings_file_x_path = None
    labels_file_y_path = None

    context_arg_lower = args.context.lower()

    if context_arg_lower == 'all_combined':
        current_labels_list_for_run = DEFAULT_ALL_POSSIBLE_LABELS
        embeddings_file_x_path = EMBEDDINGS_DIR / "embeddings_all_combined.pkl"
        labels_file_y_path = EMBEDDINGS_DIR / "labels_all_combined.pkl"
        print(f"Using ALL_COMBINED context. Expecting {len(current_labels_list_for_run)} labels: {current_labels_list_for_run}")
    elif context_arg_lower in LOG_TYPE_ATTACKS_BASELINE:
        current_labels_list_for_run = LOG_TYPE_ATTACKS_BASELINE[context_arg_lower]
        # The order in LOG_TYPE_ATTACKS_BASELINE is assumed to be the order for labels.pkl
        embeddings_file_x_path = EMBEDDINGS_DIR / context_arg_lower / "embeddings.pkl"
        labels_file_y_path = EMBEDDINGS_DIR / context_arg_lower / "labels.pkl"
        print(f"Using context for log type: '{context_arg_lower}'. Expecting {len(current_labels_list_for_run)} labels: {current_labels_list_for_run}")
    else:
        # This case should not be reached if choices in argparse are set correctly
        print(f"Error: Invalid context '{args.context}'. This should have been caught by argparse.")
        return 
    
    num_labels_for_run = len(current_labels_list_for_run)
    if num_labels_for_run == 0:
        print(f"Error: No labels defined for context '{args.context}'. Cannot proceed.")
        return

    if not embeddings_file_x_path.exists() or not labels_file_y_path.exists():
        print(f"Error: Required input files not found for context '{args.context}'.")
        print(f"  Expected X: {embeddings_file_x_path}")
        print(f"  Expected Y: {labels_file_y_path}")
        print("Please ensure these files exist, possibly by running fasttext_embedding.py with the appropriate options.")
        return

    X_train, X_test, Y_train_multi, Y_test_multi = load_data(
        embeddings_file_x_path, labels_file_y_path, num_labels_for_run, 
        args.test_size, args.random_state
    )

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_context_name = context_arg_lower.replace('/', '_') 
    current_run_results_dir = RESULTS_BASE_DIR / f"run_{run_context_name}_{timestamp}"
    current_run_models_dir = MODELS_BASE_DIR / f"run_{run_context_name}_{timestamp}"
    current_run_results_dir.mkdir(parents=True, exist_ok=True)
    current_run_models_dir.mkdir(parents=True, exist_ok=True)

    train_evaluate_ovr_xgboost(
        X_train, Y_train_multi, X_test, Y_test_multi,
        current_labels_list_for_run,
        current_run_results_dir,
        current_run_models_dir
    )

    overall_end_time = time.time()
    print(f"\nTotal script execution time: {overall_end_time - overall_start_time:.2f} seconds.")
    print(f"Results and models saved in subdirectories under: {RESULTS_BASE_DIR} and {MODELS_BASE_DIR}")

if __name__ == '__main__':
    main() 