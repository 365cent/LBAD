#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate Anomaly Detection Results Against Original Labels

This script compares the transformer anomaly detection results with the original
multi-label attack classifications to assess performance and provide detailed analysis.
"""

import os
import sys
import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, precision_recall_fscore_support,
    f1_score, accuracy_score, roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve
)
import argparse
from typing import Dict, List, Tuple, Any

def load_original_labels(log_type: str) -> Tuple[np.ndarray, List[str]]:
    """Load original multi-label attack classifications"""
    label_file = Path("embeddings") / log_type / f"label_{log_type}.pkl"
    
    with open(label_file, 'rb') as f:
        label_data = pickle.load(f)
    
    vectors = label_data['vectors']
    classes = label_data['classes']
    
    print(f"📊 Original Labels:")
    print(f"   Shape: {vectors.shape}")
    print(f"   Classes: {classes}")
    print(f"   Total samples: {len(vectors):,}")
    
    # Calculate attack statistics
    attack_counts = vectors.sum(axis=1)
    normal_samples = (attack_counts == 0).sum()
    attack_samples = (attack_counts > 0).sum()
    
    print(f"   Normal samples (no attacks): {normal_samples:,} ({normal_samples/len(vectors)*100:.1f}%)")
    print(f"   Attack samples (any attack): {attack_samples:,} ({attack_samples/len(vectors)*100:.1f}%)")
    
    # Per-class statistics
    print(f"\n📈 Per-class attack distribution:")
    for i, class_name in enumerate(classes):
        count = vectors[:, i].sum()
        percentage = count / len(vectors) * 100
        print(f"   {class_name}: {count:,} samples ({percentage:.1f}%)")
    
    return vectors, classes

def load_anomaly_results(log_type: str) -> Tuple[np.ndarray, np.ndarray, float]:
    """Load anomaly detection results"""
    # Load enhanced embeddings with anomaly scores
    enhanced_file = Path("results") / "embeddings" / f"{log_type}_anomaly" / f"log_{log_type}_anomaly_scores.pkl"
    
    with open(enhanced_file, 'rb') as f:
        enhanced_embeddings = pickle.load(f)
    
    # Extract anomaly scores (last column)
    anomaly_scores = enhanced_embeddings[:, -1]
    
    # Load binary predictions
    binary_file = Path("results") / "embeddings" / f"{log_type}_anomaly" / f"log_{log_type}_anomaly.pkl"
    
    with open(binary_file, 'rb') as f:
        binary_embeddings = pickle.load(f)
    
    binary_predictions = binary_embeddings[:, -1]
    
    # Load threshold from metadata
    metadata_file = Path("results") / "embeddings" / f"{log_type}_anomaly" / f"metadata_{log_type}_anomaly.json"
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    threshold = metadata['detection_threshold']
    
    print(f"🔍 Anomaly Detection Results:")
    print(f"   Total samples: {len(anomaly_scores):,}")
    print(f"   Threshold: {threshold:.6f}")
    print(f"   Normal predictions: {(binary_predictions == 0).sum():,} ({(binary_predictions == 0).sum()/len(binary_predictions)*100:.1f}%)")
    print(f"   Anomaly predictions: {(binary_predictions == 1).sum():,} ({(binary_predictions == 1).sum()/len(binary_predictions)*100:.1f}%)")
    print(f"   Mean anomaly score: {anomaly_scores.mean():.4f}")
    print(f"   Std anomaly score: {anomaly_scores.std():.4f}")
    
    return anomaly_scores, binary_predictions, threshold

def create_ground_truth_labels(original_vectors: np.ndarray) -> np.ndarray:
    """Create binary ground truth: 0 = normal (no attacks), 1 = anomaly (any attack)"""
    # Sum across all attack types - if any attack is present, it's an anomaly
    attack_sums = original_vectors.sum(axis=1)
    ground_truth = (attack_sums > 0).astype(int)
    
    return ground_truth

def evaluate_binary_classification(ground_truth: np.ndarray, predictions: np.ndarray, 
                                 anomaly_scores: np.ndarray, log_type: str) -> Dict[str, Any]:
    """Evaluate binary anomaly detection performance"""
    
    print(f"\n🎯 Binary Classification Evaluation:")
    print(f"   Ground truth - Normal: {(ground_truth == 0).sum():,} ({(ground_truth == 0).sum()/len(ground_truth)*100:.1f}%)")
    print(f"   Ground truth - Anomaly: {(ground_truth == 1).sum():,} ({(ground_truth == 1).sum()/len(ground_truth)*100:.1f}%)")
    print(f"   Predictions - Normal: {(predictions == 0).sum():,} ({(predictions == 0).sum()/len(predictions)*100:.1f}%)")
    print(f"   Predictions - Anomaly: {(predictions == 1).sum():,} ({(predictions == 1).sum()/len(predictions)*100:.1f}%)")
    
    # Calculate metrics
    accuracy = accuracy_score(ground_truth, predictions)
    precision, recall, f1, support = precision_recall_fscore_support(ground_truth, predictions, average='binary', zero_division=0)
    
    # ROC and PR metrics using anomaly scores
    roc_auc = roc_auc_score(ground_truth, anomaly_scores)
    pr_auc = average_precision_score(ground_truth, anomaly_scores)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(ground_truth, predictions).ravel()
    
    # Additional metrics
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False positive rate
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False negative rate
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # True positive rate (recall)
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0  # True negative rate
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'fpr': fpr,
        'fnr': fnr,
        'tpr': tpr,
        'tnr': tnr,
        'support': support
    }
    
    print(f"\n📊 Performance Metrics:")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall: {recall:.4f}")
    print(f"   F1 Score: {f1:.4f}")
    print(f"   ROC AUC: {roc_auc:.4f}")
    print(f"   PR AUC: {pr_auc:.4f}")
    print(f"   True Positives: {tp:,}")
    print(f"   True Negatives: {tn:,}")
    print(f"   False Positives: {fp:,}")
    print(f"   False Negatives: {fn:,}")
    print(f"   False Positive Rate: {fpr:.4f}")
    print(f"   False Negative Rate: {fnr:.4f}")
    
    return metrics

def analyze_per_class_performance(original_vectors: np.ndarray, predictions: np.ndarray, 
                                classes: List[str], log_type: str) -> Dict[str, Any]:
    """Analyze performance for each attack type individually"""
    
    print(f"\n🔍 Per-Class Analysis:")
    
    class_metrics = {}
    
    for i, class_name in enumerate(classes):
        # Ground truth for this class
        class_ground_truth = original_vectors[:, i]
        
        # Calculate metrics for this class
        precision, recall, f1, support = precision_recall_fscore_support(
            class_ground_truth, predictions, average='binary', zero_division=0
        )
        
        # Confusion matrix for this class
        tn, fp, fn, tp = confusion_matrix(class_ground_truth, predictions).ravel()
        
        class_metrics[class_name] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'support': support,
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'total_samples': int(support) if support is not None else 0
        }
        
        print(f"   {class_name}:")
        print(f"     Total samples: {support:,}" if support is not None else "     Total samples: 0")
        print(f"     Precision: {precision:.4f}")
        print(f"     Recall: {recall:.4f}")
        print(f"     F1 Score: {f1:.4f}")
        print(f"     TP: {tp:,}, TN: {tn:,}, FP: {fp:,}, FN: {fn:,}")
    
    return class_metrics

def create_visualizations(ground_truth: np.ndarray, predictions: np.ndarray, 
                         anomaly_scores: np.ndarray, original_vectors: np.ndarray,
                         classes: List[str], metrics: Dict[str, Any], 
                         log_type: str, output_dir: Path):
    """Create comprehensive visualizations"""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Confusion Matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(ground_truth, predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Anomaly'], 
                yticklabels=['Normal', 'Anomaly'])
    plt.title(f'Confusion Matrix - {log_type}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_dir / f'confusion_matrix_{log_type}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. ROC Curve
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    fpr, tpr, _ = roc_curve(ground_truth, anomaly_scores)
    plt.plot(fpr, tpr, label=f'ROC (AUC = {metrics["roc_auc"]:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Precision-Recall Curve
    plt.subplot(1, 2, 2)
    precision, recall, _ = precision_recall_curve(ground_truth, anomaly_scores)
    plt.plot(recall, precision, label=f'PR (AUC = {metrics["pr_auc"]:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'roc_pr_curves_{log_type}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Anomaly Score Distribution by True Label
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    normal_scores = anomaly_scores[ground_truth == 0]
    anomaly_scores_gt = anomaly_scores[ground_truth == 1]
    
    if len(normal_scores) > 0:
        plt.hist(normal_scores, bins=50, alpha=0.7, label='Normal', color='blue', edgecolor='black')
    if len(anomaly_scores_gt) > 0:
        plt.hist(anomaly_scores_gt, bins=50, alpha=0.7, label='Anomaly', color='red', edgecolor='black')
    
    plt.xlabel('Anomaly Score')
    plt.ylabel('Frequency')
    plt.title(f'Anomaly Score Distribution - {log_type}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. Per-class anomaly score distributions
    plt.subplot(1, 2, 2)
    for i, class_name in enumerate(classes):
        class_mask = original_vectors[:, i] == 1
        if class_mask.sum() > 0:
            class_scores = anomaly_scores[class_mask]
            plt.hist(class_scores, bins=30, alpha=0.6, label=class_name, edgecolor='black')
    
    plt.xlabel('Anomaly Score')
    plt.ylabel('Frequency')
    plt.title(f'Anomaly Scores by Attack Type - {log_type}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'anomaly_score_distributions_{log_type}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Performance comparison table
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table data
    table_data = [
        ['Metric', 'Value'],
        ['Accuracy', f"{metrics['accuracy']:.4f}"],
        ['Precision', f"{metrics['precision']:.4f}"],
        ['Recall', f"{metrics['recall']:.4f}"],
        ['F1 Score', f"{metrics['f1_score']:.4f}"],
        ['ROC AUC', f"{metrics['roc_auc']:.4f}"],
        ['PR AUC', f"{metrics['pr_auc']:.4f}"],
        ['True Positives', f"{metrics['true_positives']:,}"],
        ['True Negatives', f"{metrics['true_negatives']:,}"],
        ['False Positives', f"{metrics['false_positives']:,}"],
        ['False Negatives', f"{metrics['false_negatives']:,}"],
        ['False Positive Rate', f"{metrics['fpr']:.4f}"],
        ['False Negative Rate', f"{metrics['fnr']:.4f}"]
    ]
    
    table = ax.table(cellText=table_data, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    plt.title(f'Anomaly Detection Performance Summary - {log_type}', fontsize=14, pad=20)
    plt.savefig(output_dir / f'performance_summary_{log_type}.png', dpi=300, bbox_inches='tight')
    plt.close()

def save_evaluation_results(metrics: Dict[str, Any], class_metrics: Dict[str, Any], 
                          log_type: str, output_dir: Path):
    """Save evaluation results to files"""
    
    # Save main metrics
    results = {
        'log_type': log_type,
        'evaluation_timestamp': pd.Timestamp.now().isoformat(),
        'binary_classification_metrics': metrics,
        'per_class_metrics': class_metrics
    }
    
    with open(output_dir / f'evaluation_results_{log_type}.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save detailed classification report
    with open(output_dir / f'classification_report_{log_type}.txt', 'w') as f:
        f.write(f"Anomaly Detection Evaluation Report - {log_type}\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("Binary Classification Metrics:\n")
        f.write("-" * 30 + "\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
        
        f.write("\nPer-Class Analysis:\n")
        f.write("-" * 20 + "\n")
        for class_name, class_metric in class_metrics.items():
            f.write(f"\n{class_name}:\n")
            for key, value in class_metric.items():
                f.write(f"  {key}: {value}\n")

def main():
    parser = argparse.ArgumentParser(description='Evaluate Anomaly Detection Results')
    parser.add_argument('--log-type', type=str, required=True, 
                       help='Log type to evaluate (e.g., wp-error)')
    
    args = parser.parse_args()
    
    print(f"🔍 Evaluating Anomaly Detection for {args.log_type}")
    print("=" * 60)
    
    # Load data
    original_vectors, classes = load_original_labels(args.log_type)
    anomaly_scores, predictions, threshold = load_anomaly_results(args.log_type)
    
    # Create ground truth
    ground_truth = create_ground_truth_labels(original_vectors)
    
    # Evaluate binary classification
    metrics = evaluate_binary_classification(ground_truth, predictions, anomaly_scores, args.log_type)
    
    # Analyze per-class performance
    class_metrics = analyze_per_class_performance(original_vectors, predictions, classes, args.log_type)
    
    # Create output directory
    output_dir = Path("results") / "anomaly_evaluation" / args.log_type
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create visualizations
    create_visualizations(ground_truth, predictions, anomaly_scores, original_vectors,
                         classes, metrics, args.log_type, output_dir)
    
    # Save results
    save_evaluation_results(metrics, class_metrics, args.log_type, output_dir)
    
    print(f"\n✅ Evaluation completed! Results saved to {output_dir}")
    
    # Print summary
    print(f"\n📋 Summary:")
    print(f"   Log type: {args.log_type}")
    print(f"   Total samples: {len(ground_truth):,}")
    print(f"   Ground truth anomaly rate: {ground_truth.mean():.1%}")
    print(f"   Predicted anomaly rate: {predictions.mean():.1%}")
    print(f"   Accuracy: {metrics['accuracy']:.4f}")
    print(f"   F1 Score: {metrics['f1_score']:.4f}")
    print(f"   ROC AUC: {metrics['roc_auc']:.4f}")
    
    if metrics['accuracy'] < 0.5:
        print(f"\n⚠️  WARNING: Low accuracy detected. This might indicate:")
        print(f"   - The anomaly detection threshold needs adjustment")
        print(f"   - The model needs more training on normal patterns")
        print(f"   - The data distribution is highly imbalanced")
    else:
        print(f"\n✅ Good performance detected!")

if __name__ == "__main__":
    main() 