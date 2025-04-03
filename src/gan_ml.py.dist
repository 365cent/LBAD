#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Machine Learning for Log-Based Attack Detection
----------------------------------------------
Uses FastText embeddings to train and evaluate machine learning models.
Optimized for Apple Silicon (M1/M2/M3) processors.
"""

import sys
import json
import pickle
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Project paths
ROOT = Path(__file__).resolve().parent.parent
EMB, MOD, RES = ROOT / 'embeddings', ROOT / 'models', ROOT / 'results'
[d.mkdir(exist_ok=True) for d in (EMB, MOD, RES)]

# For Apple Silicon optimization: use appropriate thread counts
CPU_COUNT = os.cpu_count()
if CPU_COUNT:
    N_JOBS = max(1, CPU_COUNT - 1)  # Leave one core free
else:
    N_JOBS = -1  # Use all cores

# Model definitions with Apple Silicon optimizations
MODELS = {
    'rf': RandomForestClassifier(
        n_estimators=100, 
        random_state=42, 
        n_jobs=N_JOBS,
        verbose=0
    ),
    'xgb': XGBClassifier(
        n_estimators=100, 
        learning_rate=0.1, 
        random_state=42, 
        n_jobs=N_JOBS,
        tree_method='hist',  # Better for Apple Silicon
        enable_categorical=True,
        use_label_encoder=False  # We'll handle encoding manually
    ),
    'svm': SVC(
        kernel='rbf', 
        probability=True, 
        random_state=42,
        cache_size=2000  # Allocate more cache for faster training
    )
}

def parse_labels(label_strings):
    """Convert JSON label strings to classification labels."""
    parsed = []
    for label in label_strings:
        try:
            data = json.loads(label)
            if isinstance(data, list):
                parsed.append("normal" if not data else data[0])
            else:
                parsed.append("unknown")
        except:
            parsed.append("unknown")
    return parsed

def load_data():
    """Load embeddings and labels from separate pickle files."""
    print("Loading data...")
    
    # Load embeddings
    try:
        with open(EMB / 'train_embeddings.pkl', 'rb') as f: 
            X_train = pickle.load(f)
        with open(EMB / 'test_embeddings.pkl', 'rb') as f: 
            X_test = pickle.load(f)
    except FileNotFoundError:
        print("Error: Embedding files not found. Run fasttext_embedding.py first.")
        sys.exit(1)
    
    # Load and parse labels
    try:
        with open(EMB / 'train_labels.pkl', 'rb') as f:
            y_train_raw = pickle.load(f)
        with open(EMB / 'test_labels.pkl', 'rb') as f:
            y_test_raw = pickle.load(f)
        
        y_train = parse_labels(y_train_raw)
        y_test = parse_labels(y_test_raw)
    except FileNotFoundError:
        print("Error: Label files not found. Run fasttext_embedding.py first.")
        sys.exit(1)
    
    print(f"Loaded {len(X_train)} training samples, {len(X_test)} test samples")
    all_labels = y_train + y_test
    print(f"Label distribution: {pd.Series(all_labels).value_counts().to_dict()}")
    
    return X_train, y_train, X_test, y_test

def main():
    """Train and evaluate models with console output."""
    parser = argparse.ArgumentParser(description='ML models for log analysis')
    parser.add_argument('--model', choices=['rf', 'xgb', 'svm', 'all'], default='all',
                      help='Model to train (default: all)')
    parser.add_argument('--evaluate-only', action='store_true',
                      help='Only evaluate existing models (no training)')
    args = parser.parse_args()

    # Load data
    X_train, y_train, X_test, y_test = load_data()
    
    # Create label encoder to convert string labels to integers
    le = LabelEncoder()
    # Fit on all unique labels to ensure test set labels are handled
    all_labels = np.unique(np.concatenate([y_train, y_test]))
    le.fit(all_labels)
    
    # Save the label encoder for future use
    joblib.dump(le, MOD / 'label_encoder.joblib')
    
    # Convert labels to numerical format (needed for XGBoost)
    y_train_encoded = le.transform(y_train)
    y_test_encoded = le.transform(y_test)
    
    # Determine which models to use
    models = ['rf', 'xgb', 'svm'] if args.model == 'all' else [args.model]
    
    for m in models:
        model_path = MOD / f'{m}.joblib'
        print(f"\nProcessing {m.upper()} model...")
        
        # Either load or train the model
        if model_path.exists() and args.evaluate_only:
            print(f"Loading existing model...")
            model = joblib.load(model_path)
        else:
            print(f"Training model...")
            # For XGBoost, use encoded labels, for others use original strings
            if m == 'xgb':
                model = MODELS[m].fit(X_train, y_train_encoded)
            else:
                model = MODELS[m].fit(X_train, y_train)
            
            # Save the trained model
            joblib.dump(model, model_path)
        
        # Evaluate model
        print(f"\n{m.upper()} Model Evaluation:")
        print("-" * 40)
        
        # For prediction, handle XGBoost separately
        if m == 'xgb':
            y_pred_encoded = model.predict(X_test)
            y_pred = le.inverse_transform(y_pred_encoded)
        else:
            y_pred = model.predict(X_test)
        
        # Generate classification report
        report = classification_report(y_test, y_pred, zero_division=0)
        print(report)
        
        # Save classification report to file
        with open(RES / f'{m}_classification_report.txt', 'w') as f:
            f.write(f"{m.upper()} Classification Report\n")
            f.write("-" * 40 + "\n")
            f.write(report)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_test, y_pred)
        
        # Normalize confusion matrix
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)  # Replace NaN with 0
        
        # Get unique sorted labels for the matrix
        unique_labels = np.unique(np.concatenate([y_test, y_pred]))
        
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=unique_labels, 
                   yticklabels=unique_labels)
        plt.title(f'{m.upper()} - Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(RES / f'{m}_cm.png')
        plt.close()
        
        # Output summary statistics
        correct = np.sum(np.array(y_pred) == np.array(y_test))
        total = len(y_test)
        accuracy = correct / total
        
        print(f"\nSummary Statistics:")
        print(f"  Accuracy: {accuracy:.4f} ({correct}/{total})")
        
        # Calculate class-wise metrics
        label_stats = pd.DataFrame(columns=['Support', 'Correct', 'Accuracy'])
        
        # Convert to numpy arrays if they aren't already
        y_test_array = np.array(y_test)
        y_pred_array = np.array(y_pred)
        
        for label in unique_labels:
            # Create boolean mask as numpy array
            mask = (y_test_array == label)
            support = np.sum(mask)
            correct_pred = np.sum((y_pred_array == label) & mask)
            label_accuracy = correct_pred / support if support > 0 else 0
            
            label_stats.loc[label] = [support, correct_pred, label_accuracy]
        
        # Sort by support count (descending)
        label_stats = label_stats.sort_values('Support', ascending=False)
        
        # Print top 5 classes by support
        print("\nTop 5 classes by frequency:")
        print(label_stats.head(5).to_string())
        
        # Save complete stats to file
        label_stats.to_csv(RES / f'{m}_class_statistics.csv')

    print(f"\nAll models evaluated. Results saved to {RES}")

if __name__ == '__main__':
    main()