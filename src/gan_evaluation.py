#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
VAE-GAN Augmentation Evaluation
-------------------------------
Evaluates the effectiveness of VAE-GAN data augmentation using XGBoost.
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

# Project paths
ROOT = Path(__file__).resolve().parent.parent
EMB = ROOT / 'embeddings'
AUG = ROOT / 'augmented'
RES = ROOT / 'results'
RES.mkdir(exist_ok=True)

# For Apple Silicon optimization
N_JOBS = max(1, os.cpu_count() - 1) if os.cpu_count() else -1
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def parse_label(label):
    """Parse JSON-formatted label string."""
    try:
        data = json.loads(label) if isinstance(label, str) else label
        return "normal" if not data or not isinstance(data, list) else data[0]
    except:
        return "unknown"

def load_data():
    """Load original test data and augmented training data."""
    print("Loading data...")
    
    # Load augmented training data
    with open(AUG / "synthetic_embeddings.pkl", 'rb') as f:
        synthetic_embeddings = pickle.load(f)
    with open(AUG / "synthetic_labels.pkl", 'rb') as f:
        synthetic_labels = pickle.load(f)
    
    # Load original training data
    with open(EMB / 'train_embeddings.pkl', 'rb') as f:
        orig_train_embeddings = pickle.load(f)
    with open(EMB / 'train_labels.pkl', 'rb') as f:
        orig_train_labels = [parse_label(label) for label in pickle.load(f)]
    
    # Load test data
    with open(EMB / 'test_embeddings.pkl', 'rb') as f:
        test_embeddings = pickle.load(f)
    with open(EMB / 'test_labels.pkl', 'rb') as f:
        test_labels = [parse_label(label) for label in pickle.load(f)]
    
    # Combine original and synthetic data for augmented training set
    augmented_embeddings = np.vstack([orig_train_embeddings, synthetic_embeddings])
    augmented_labels = np.concatenate([orig_train_labels, synthetic_labels])
    
    print(f"Original training samples: {len(orig_train_embeddings)}")
    print(f"Synthetic samples: {len(synthetic_embeddings)}")
    print(f"Augmented training samples: {len(augmented_embeddings)}")
    print(f"Test samples: {len(test_embeddings)}")
    
    # Print class distribution in original vs augmented
    orig_dist = pd.Series(orig_train_labels).value_counts()
    aug_dist = pd.Series(augmented_labels).value_counts()
    
    print("\nClass distribution (original vs augmented):")
    all_classes = sorted(set(orig_dist.index) | set(aug_dist.index))
    for cls in all_classes:
        orig_count = orig_dist.get(cls, 0)
        aug_count = aug_dist.get(cls, 0)
        print(f"  {cls}: {orig_count} → {aug_count} (+{aug_count-orig_count})")
    
    return (orig_train_embeddings, orig_train_labels, 
            augmented_embeddings, augmented_labels,
            test_embeddings, test_labels)

def evaluate_with_xgboost(data_tuple):
    """Evaluate both original and augmented data with XGBoost."""
    (orig_train_X, orig_train_y, 
     aug_train_X, aug_train_y,
     test_X, test_y) = data_tuple
    
    # Initialize label encoder
    le = LabelEncoder()
    all_labels = np.unique(np.concatenate([orig_train_y, aug_train_y, test_y]))
    le.fit(all_labels)
    
    # Create XGBoost classifier
    xgb = XGBClassifier(
        n_estimators=100, 
        learning_rate=0.1, 
        random_state=RANDOM_SEED, 
        n_jobs=N_JOBS,
        tree_method='hist',
        enable_categorical=True,
        use_label_encoder=False
    )
    
    # Train and evaluate with original data
    print("\nTraining XGBoost on original data...")
    orig_train_y_enc = le.transform(orig_train_y)
    xgb.fit(orig_train_X, orig_train_y_enc)
    
    orig_pred_enc = xgb.predict(test_X)
    orig_pred = le.inverse_transform(orig_pred_enc)
    
    # Generate classification report for original data
    orig_report = classification_report(test_y, orig_pred)
    print("\nOriginal Data Classification Report:")
    print(orig_report)
    
    # Save original report to file
    with open(RES / "xgb_original_report.txt", 'w') as f:
        f.write("XGBoost - Original Data Classification Report\n")
        f.write("-" * 50 + "\n")
        f.write(orig_report)
    
    # Save original confusion matrix
    plt.figure(figsize=(10, 8))
    cm_orig = confusion_matrix(test_y, orig_pred)
    sns.heatmap(cm_orig, annot=True, fmt='d', cmap='Blues')
    plt.title('XGBoost - Original Data - Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(RES / "xgb_original_cm.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Train and evaluate with augmented data
    print("\nTraining XGBoost on augmented data...")
    xgb = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        random_state=RANDOM_SEED,
        n_jobs=N_JOBS,
        tree_method='hist',
        enable_categorical=True,
        use_label_encoder=False
    )
    
    aug_train_y_enc = le.transform(aug_train_y)
    xgb.fit(aug_train_X, aug_train_y_enc)
    
    aug_pred_enc = xgb.predict(test_X)
    aug_pred = le.inverse_transform(aug_pred_enc)
    
    # Generate classification report for augmented data
    aug_report = classification_report(test_y, aug_pred)
    print("\nAugmented Data Classification Report:")
    print(aug_report)
    
    # Save augmented report to file
    with open(RES / "xgb_augmented_report.txt", 'w') as f:
        f.write("XGBoost - Augmented Data Classification Report\n")
        f.write("-" * 50 + "\n")
        f.write(aug_report)
    
    # Save augmented confusion matrix
    plt.figure(figsize=(10, 8))
    cm_aug = confusion_matrix(test_y, aug_pred)
    sns.heatmap(cm_aug, annot=True, fmt='d', cmap='Blues')
    plt.title('XGBoost - Augmented Data - Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(RES / "xgb_augmented_cm.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Calculate per-class recall comparison
    classes = np.unique(np.concatenate([test_y, orig_pred, aug_pred]))
    recall_comparison = []
    
    for cls in classes:
        # Calculate recall for original model
        true_cls = np.array(test_y) == cls
        orig_pred_cls = np.array(orig_pred) == cls
        orig_recall = np.sum(true_cls & orig_pred_cls) / max(np.sum(true_cls), 1)
        
        # Calculate recall for augmented model
        aug_pred_cls = np.array(aug_pred) == cls
        aug_recall = np.sum(true_cls & aug_pred_cls) / max(np.sum(true_cls), 1)
        
        # Add to comparison
        recall_comparison.append({
            'Class': cls,
            'Original': orig_recall,
            'Augmented': aug_recall,
            'Improvement': aug_recall - orig_recall,
            'Support': np.sum(true_cls)
        })
    
    # Create dataframe for visualization
    recall_df = pd.DataFrame(recall_comparison)
    
    # Sort by improvement (descending)
    recall_df = recall_df.sort_values('Improvement', ascending=False)
    
    # Save recall comparison to CSV
    recall_df.to_csv(RES / "xgb_recall_comparison.csv", index=False)
    
    # Plot top 10 classes with most improvement
    plt.figure(figsize=(12, 8))
    top10 = recall_df.head(10)
    
    plt.barh(top10['Class'], top10['Improvement'], color='green')
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    plt.title('Top 10 Classes with Highest Recall Improvement')
    plt.xlabel('Recall Improvement')
    plt.tight_layout()
    plt.savefig(RES / "xgb_recall_improvement.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Extract overall metrics from classification reports
    orig_metrics = {
        'accuracy': float(orig_report.split('\n')[-2].split()[-1]),
        'macro_f1': float(orig_report.split('\n')[-4].split()[-2]),
        'weighted_f1': float(orig_report.split('\n')[-3].split()[-2])
    }
    
    aug_metrics = {
        'accuracy': float(aug_report.split('\n')[-2].split()[-1]),
        'macro_f1': float(aug_report.split('\n')[-4].split()[-2]),
        'weighted_f1': float(aug_report.split('\n')[-3].split()[-2])
    }
    
    # Create summary visualization
    plt.figure(figsize=(10, 6))
    metrics = ['Accuracy', 'Macro F1', 'Weighted F1']
    original = [orig_metrics['accuracy'], orig_metrics['macro_f1'], orig_metrics['weighted_f1']]
    augmented = [aug_metrics['accuracy'], aug_metrics['macro_f1'], aug_metrics['weighted_f1']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.bar(x - width/2, original, width, label='Original', color='lightblue')
    plt.bar(x + width/2, augmented, width, label='Augmented', color='orange')
    
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('XGBoost Performance: Original vs Augmented Data')
    plt.xticks(x, metrics)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(RES / "xgb_summary.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Return metrics dictionary
    return {
        'original': orig_metrics,
        'augmented': aug_metrics
    }

def main():
    """Main function to evaluate VAE-GAN augmentation with XGBoost."""
    # Load data
    data_tuple = load_data()
    
    # Evaluate with XGBoost
    results = evaluate_with_xgboost(data_tuple)
    
    # Print summary
    print("\nEvaluation Summary:")
    print(f"Original accuracy: {results['original']['accuracy']:.4f}")
    print(f"Augmented accuracy: {results['augmented']['accuracy']:.4f}")
    print(f"Improvement: {results['augmented']['accuracy'] - results['original']['accuracy']:.4f}")
    
    print(f"\nOriginal weighted F1: {results['original']['weighted_f1']:.4f}")
    print(f"Augmented weighted F1: {results['augmented']['weighted_f1']:.4f}")
    print(f"Improvement: {results['augmented']['weighted_f1'] - results['original']['weighted_f1']:.4f}")
    
    print(f"\nEvaluation complete. Results saved to {RES}")

if __name__ == "__main__":
    main()