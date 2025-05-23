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
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.model_selection import StratifiedKFold
from scipy import stats
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

def plot_roc_curves(y_test, orig_probs, aug_probs, le, top_n=5):
    """Plot ROC curves for top N classes."""
    plt.figure(figsize=(12, 10))
    
    # Get classes with most improvement in AUC
    aucs = {}
    for i, class_name in enumerate(le.classes_):
        if class_name == 'normal':
            continue
        
        y_true = (np.array(y_test) == class_name).astype(int)
        fpr_orig, tpr_orig, _ = roc_curve(y_true, orig_probs[:, i])
        fpr_aug, tpr_aug, _ = roc_curve(y_true, aug_probs[:, i])
        
        auc_orig = auc(fpr_orig, tpr_orig)
        auc_aug = auc(fpr_aug, tpr_aug)
        
        aucs[class_name] = {
            'orig': auc_orig,
            'aug': auc_aug,
            'diff': auc_aug - auc_orig
        }
    
    # Sort by improvement
    top_classes = sorted(aucs.items(), key=lambda x: x[1]['diff'], reverse=True)[:top_n]
    
    # Plot curves for top classes
    for class_name, auc_values in top_classes:
        y_true = (np.array(y_test) == class_name).astype(int)
        
        # Get class index
        class_idx = np.where(le.classes_ == class_name)[0][0]
        
        # Original model
        fpr_orig, tpr_orig, _ = roc_curve(y_true, orig_probs[:, class_idx])
        auc_orig = auc(fpr_orig, tpr_orig)
        
        # Augmented model
        fpr_aug, tpr_aug, _ = roc_curve(y_true, aug_probs[:, class_idx])
        auc_aug = auc(fpr_aug, tpr_aug)
        
        plt.plot(fpr_orig, tpr_orig, linestyle='--', 
                 label=f'{class_name} - Original (AUC = {auc_orig:.3f})')
        plt.plot(fpr_aug, tpr_aug, 
                 label=f'{class_name} - Augmented (AUC = {auc_aug:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Top Classes with Most Improvement')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RES / "roc_curves.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_precision_recall_curves(y_test, orig_probs, aug_probs, le, top_n=5):
    """Plot precision-recall curves for top N classes."""
    plt.figure(figsize=(12, 10))
    
    # Get classes with most improvement in Average Precision
    avg_precs = {}
    for i, class_name in enumerate(le.classes_):
        if class_name == 'normal':
            continue
        
        y_true = (np.array(y_test) == class_name).astype(int)
        
        avg_prec_orig = average_precision_score(y_true, orig_probs[:, i])
        avg_prec_aug = average_precision_score(y_true, aug_probs[:, i])
        
        avg_precs[class_name] = {
            'orig': avg_prec_orig,
            'aug': avg_prec_aug,
            'diff': avg_prec_aug - avg_prec_orig
        }
    
    # Sort by improvement
    top_classes = sorted(avg_precs.items(), key=lambda x: x[1]['diff'], reverse=True)[:top_n]
    
    # Plot curves for top classes
    for class_name, ap_values in top_classes:
        y_true = (np.array(y_test) == class_name).astype(int)
        
        # Get class index
        class_idx = np.where(le.classes_ == class_name)[0][0]
        
        # Original model
        precision_orig, recall_orig, _ = precision_recall_curve(y_true, orig_probs[:, class_idx])
        ap_orig = average_precision_score(y_true, orig_probs[:, class_idx])
        
        # Augmented model
        precision_aug, recall_aug, _ = precision_recall_curve(y_true, aug_probs[:, class_idx])
        ap_aug = average_precision_score(y_true, aug_probs[:, class_idx])
        
        plt.plot(recall_orig, precision_orig, linestyle='--', 
                 label=f'{class_name} - Original (AP = {ap_orig:.3f})')
        plt.plot(recall_aug, precision_aug, 
                 label=f'{class_name} - Augmented (AP = {ap_aug:.3f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves for Top Classes with Most Improvement')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RES / "precision_recall_curves.png", dpi=300, bbox_inches='tight')
    plt.close()

def statistical_significance_test(data_tuple, n_folds=5):
    """Perform statistical significance testing using cross-validation."""
    (orig_train_X, orig_train_y, 
     augmented_train_X, augmented_train_y,
     test_X, test_y) = data_tuple
    
    # Initialize label encoder and encode all labels
    le = LabelEncoder()
    all_labels = np.unique(np.concatenate([orig_train_y, augmented_train_y, test_y]))
    le.fit(all_labels)
    
    # Convert to numpy arrays for consistency
    orig_train_X = np.array(orig_train_X)
    orig_train_y_enc = le.transform(orig_train_y)
    augmented_train_X = np.array(augmented_train_X)
    augmented_train_y_enc = le.transform(augmented_train_y)
    test_X = np.array(test_X)
    test_y_enc = le.transform(test_y)
    
    # Store scores for each fold
    orig_scores = []
    augmented_scores = []
    
    # Define stratified cross-validation
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    
    print(f"\nPerforming statistical significance test with {n_folds}-fold CV...")
    
    # Perform cross-validation
    for fold, (train_idx, val_idx) in enumerate(skf.split(test_X, test_y_enc)):
        X_val, y_val = test_X[val_idx], test_y_enc[val_idx]
        
        # Train and evaluate original model
        orig_xgb = XGBClassifier(
            n_estimators=100, 
            learning_rate=0.1, 
            random_state=RANDOM_SEED, 
            n_jobs=N_JOBS,
            tree_method='hist',
            enable_categorical=True,
            use_label_encoder=False
        )
        orig_xgb.fit(orig_train_X, orig_train_y_enc)
        orig_score = orig_xgb.score(X_val, y_val)
        orig_scores.append(orig_score)
        
        # Train and evaluate augmented model
        augmented_xgb = XGBClassifier(
            n_estimators=100, 
            learning_rate=0.1, 
            random_state=RANDOM_SEED, 
            n_jobs=N_JOBS,
            tree_method='hist',
            enable_categorical=True,
            use_label_encoder=False
        )
        augmented_xgb.fit(augmented_train_X, augmented_train_y_enc)
        augmented_score = augmented_xgb.score(X_val, y_val)
        augmented_scores.append(augmented_score)
        
        print(f"  Fold {fold+1}: Original={orig_score:.4f}, Augmented={augmented_score:.4f}, Diff={augmented_score-orig_score:.4f}")
    
    # Perform paired t-test
    t_stat, p_value = stats.ttest_rel(augmented_scores, orig_scores)
    
    print(f"\nStatistical Significance Test Results:")
    print(f"  Original mean accuracy: {np.mean(orig_scores):.4f} ± {np.std(orig_scores):.4f}")
    print(f"  Augmented mean accuracy: {np.mean(augmented_scores):.4f} ± {np.std(augmented_scores):.4f}")
    print(f"  Mean improvement: {np.mean(augmented_scores) - np.mean(orig_scores):.4f}")
    print(f"  T-statistic: {t_stat:.4f}")
    print(f"  P-value: {p_value:.6f}")
    print(f"  Statistically significant: {'Yes' if p_value < 0.05 else 'No'}")
    
    # Save results to file
    with open(RES / "statistical_significance.txt", 'w') as f:
        f.write("Statistical Significance Test Results\n")
        f.write("-" * 50 + "\n")
        f.write(f"Original mean accuracy: {np.mean(orig_scores):.4f} ± {np.std(orig_scores):.4f}\n")
        f.write(f"Augmented mean accuracy: {np.mean(augmented_scores):.4f} ± {np.std(augmented_scores):.4f}\n")
        f.write(f"Mean improvement: {np.mean(augmented_scores) - np.mean(orig_scores):.4f}\n")
        f.write(f"T-statistic: {t_stat:.4f}\n")
        f.write(f"P-value: {p_value:.6f}\n")
        f.write(f"Statistically significant: {'Yes' if p_value < 0.05 else 'No'}\n")
    
    return {
        'orig_mean': float(np.mean(orig_scores)),
        'augmented_mean': float(np.mean(augmented_scores)),
        'improvement': float(np.mean(augmented_scores) - np.mean(orig_scores)),
        'p_value': float(p_value),
        'significant': bool(p_value < 0.05)
    }

def plot_class_improvement_heatmap(recall_df, title="Recall Improvement by Class"):
    """Create a heatmap visualization of improvements for each class."""
    # Sort by improvement (descending)
    recall_df = recall_df.sort_values('Improvement', ascending=False)
    
    # Create a pivot table for visualization
    plt.figure(figsize=(12, 8))
    
    # Create a heatmap with better color scaling
    heatmap_data = pd.DataFrame({
        'Class': recall_df['Class'],
        'Original': recall_df['Original'],
        'Augmented': recall_df['Augmented'],
        'Improvement': recall_df['Improvement']
    }).set_index('Class')
    
    # Plot the heatmap
    ax = sns.heatmap(heatmap_data, annot=True, cmap='RdYlGn', fmt='.3f',
                    cbar_kws={'label': 'Value'}, linewidths=.5)
    
    plt.title(title)
    plt.tight_layout()
    plt.savefig(RES / "class_improvement_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()

def report_class_counts(orig_train_y, augmented_train_y, test_y):
    """Generate a detailed report of class counts in original and augmented training sets."""
    # Count samples in each set
    orig_counts = pd.Series(orig_train_y).value_counts().to_dict()
    augmented_counts = pd.Series(augmented_train_y).value_counts().to_dict()
    test_counts = pd.Series(test_y).value_counts().to_dict()
    
    # Convert any numpy types to Python native types
    for key in orig_counts:
        orig_counts[key] = int(orig_counts[key])
    for key in augmented_counts:
        augmented_counts[key] = int(augmented_counts[key])
    for key in test_counts:
        test_counts[key] = int(test_counts[key])
    
    # Get all unique classes
    all_classes = sorted(set(list(orig_counts.keys()) + 
                            list(augmented_counts.keys()) + 
                            list(test_counts.keys())))
    
    # Create dataframe with all counts
    count_data = []
    for cls in all_classes:
        count_data.append({
            'Class': cls,
            'Original Training': orig_counts.get(cls, 0),
            'Augmented Training': augmented_counts.get(cls, 0),
            'Synthetic Added': augmented_counts.get(cls, 0) - orig_counts.get(cls, 0),
            'Test Set': test_counts.get(cls, 0)
        })
    
    count_df = pd.DataFrame(count_data)
    
    # Print the table
    print("\n=== CLASS DISTRIBUTION REPORT ===")
    print("Note: 'Support' in classification reports refers to test set counts, not training counts")
    print("\nClass counts across datasets:")
    
    # Sort by number of synthetic samples added (descending)
    count_df = count_df.sort_values('Synthetic Added', ascending=False)
    
    # Format as a nice table
    pd.set_option('display.max_rows', None)
    print(count_df.to_string(index=False))
    
    # Save to CSV
    count_df.to_csv(RES / "class_distribution_counts.csv", index=False)
    
    # Return the dataframe for possible further use
    return count_df

def evaluate_with_xgboost(data_tuple):
    """Evaluate both original and augmented data with XGBoost."""
    (orig_train_X, orig_train_y, 
     augmented_train_X, augmented_train_y,
     test_X, test_y) = data_tuple
    
    # Generate and save detailed class count report
    report_class_counts(orig_train_y, augmented_train_y, test_y)
    
    # Initialize label encoder
    le = LabelEncoder()
    all_labels = np.unique(np.concatenate([orig_train_y, augmented_train_y, test_y]))
    le.fit(all_labels)
    
    # Create XGBoost classifier for original data
    print("\nTraining XGBoost on original data only...")
    xgb_orig = XGBClassifier(
        n_estimators=100, 
        learning_rate=0.1, 
        random_state=RANDOM_SEED, 
        n_jobs=N_JOBS,
        tree_method='hist',
        enable_categorical=True,
        use_label_encoder=False,
        eval_metric='mlogloss'  # Add evaluation metric for better performance
    )
    
    # Train with original data only
    orig_train_y_enc = le.transform(orig_train_y)
    xgb_orig.fit(orig_train_X, orig_train_y_enc)
    
    # Predict with original model
    orig_pred_enc = xgb_orig.predict(test_X)
    orig_pred = le.inverse_transform(orig_pred_enc)
    
    # Save prediction probabilities for ROC curves
    orig_probs = xgb_orig.predict_proba(test_X)
    
    # Generate classification report for original data
    orig_report = classification_report(test_y, orig_pred)
    print("\nModel trained on original data only - Classification Report:")
    print(orig_report)
    print("Note: 'Support' column shows TEST data counts, not training data")
    
    # Save original report to file
    with open(RES / "xgb_original_report.txt", 'w') as f:
        f.write("XGBoost - Model trained on original data only - Classification Report\n")
        f.write("-" * 50 + "\n")
        f.write(orig_report)
        f.write("\nNote: 'Support' column shows TEST data counts, not training data\n")
    
    # Save original confusion matrix
    plt.figure(figsize=(10, 8))
    cm_orig = confusion_matrix(test_y, orig_pred)
    sns.heatmap(cm_orig, annot=True, fmt='d', cmap='Blues')
    plt.title('XGBoost - Model trained on original data only - Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(RES / "xgb_original_cm.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create XGBoost classifier for augmented data (original + synthetic)
    print("\nTraining XGBoost on augmented data (original + synthetic)...")
    xgb_augmented = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        random_state=RANDOM_SEED,
        n_jobs=N_JOBS,
        tree_method='hist',
        enable_categorical=True,
        use_label_encoder=False,
        eval_metric='mlogloss'  # Add evaluation metric for better performance
    )
    
    # Train with augmented data (original + synthetic)
    augmented_train_y_enc = le.transform(augmented_train_y)
    xgb_augmented.fit(augmented_train_X, augmented_train_y_enc)
    
    # Predict with augmented model
    augmented_pred_enc = xgb_augmented.predict(test_X)
    augmented_pred = le.inverse_transform(augmented_pred_enc)
    
    # Save prediction probabilities for ROC curves
    augmented_probs = xgb_augmented.predict_proba(test_X)
    
    # Generate classification report for augmented data
    augmented_report = classification_report(test_y, augmented_pred, output_dict=True)
    augmented_report_str = classification_report(test_y, augmented_pred)
    print("\nModel trained on augmented data - Classification Report:")
    print(augmented_report_str)
    print("Note: 'Support' column shows TEST data counts, not training data")
    
    # Save per-class F1 scores for detailed analysis
    class_f1_scores = {}
    for cls in augmented_report:
        if cls not in ['accuracy', 'macro avg', 'weighted avg']:
            # Convert all values to native Python types for JSON serialization
            orig_count = pd.Series(orig_train_y).value_counts().get(cls, 0)
            aug_count = pd.Series(augmented_train_y).value_counts().get(cls, 0)
            
            class_f1_scores[cls] = {
                'f1': float(augmented_report[cls]['f1-score']),
                'precision': float(augmented_report[cls]['precision']),
                'recall': float(augmented_report[cls]['recall']),
                'support': int(augmented_report[cls]['support']),
                'training_count_original': int(orig_count),
                'training_count_augmented': int(aug_count)
            }
    
    # Save class-level F1 scores to file
    with open(RES / "class_f1_scores.json", 'w') as f:
        json.dump(class_f1_scores, f, indent=2)
    
    # Save augmented report to file
    with open(RES / "xgb_augmented_report.txt", 'w') as f:
        f.write("XGBoost - Model trained on augmented data - Classification Report\n")
        f.write("-" * 50 + "\n")
        f.write(augmented_report_str)
        f.write("\nNote: 'Support' column shows TEST data counts, not training data\n")
    
    # Create a detailed comparison table
    compare_data = []
    for cls in np.unique(np.concatenate([orig_train_y, augmented_train_y, test_y])):
        orig_count = int(pd.Series(orig_train_y).value_counts().get(cls, 0))
        aug_count = int(pd.Series(augmented_train_y).value_counts().get(cls, 0))
        test_count = int(pd.Series(test_y).value_counts().get(cls, 0))
        
        orig_f1 = float(class_f1_scores[cls]['f1']) if cls in class_f1_scores else 0.0
        aug_f1 = float(class_f1_scores[cls]['f1']) if cls in class_f1_scores else 0.0
        
        compare_data.append({
            'Class': cls,
            'Original Training Count': orig_count,
            'Augmented Training Count': aug_count,
            'Test Count': test_count,
            'Original F1': orig_f1,
            'Augmented F1': aug_f1,
            'F1 Improvement': aug_f1 - orig_f1
        })
    
    compare_df = pd.DataFrame(compare_data)
    
    # Save the comparison table
    compare_df.to_csv(RES / "training_vs_test_counts.csv", index=False)
    
    # Plot ROC curves and Precision-Recall curves for key classes
    plot_roc_curves(test_y, orig_probs, augmented_probs, le)
    plot_precision_recall_curves(test_y, orig_probs, augmented_probs, le)
    
    # Calculate per-class recall and F1 comparison
    classes = np.unique(np.concatenate([test_y, orig_pred, augmented_pred]))
    recall_comparison = []
    
    for cls in classes:
        # Get true positive, false positive, etc. counts for each class
        true_cls = np.array(test_y) == cls
        orig_pred_cls = np.array(orig_pred) == cls
        augmented_pred_cls = np.array(augmented_pred) == cls
        
        # Calculate precision, recall and F1 for original model
        orig_tp = np.sum(true_cls & orig_pred_cls)
        orig_fp = np.sum((~true_cls) & orig_pred_cls)
        orig_fn = np.sum(true_cls & (~orig_pred_cls))
        
        orig_precision = orig_tp / max(orig_tp + orig_fp, 1)
        orig_recall = orig_tp / max(orig_tp + orig_fn, 1)
        orig_f1 = 2 * orig_precision * orig_recall / max(orig_precision + orig_recall, 1e-10)
        
        # Calculate precision, recall and F1 for augmented model
        augmented_tp = np.sum(true_cls & augmented_pred_cls)
        augmented_fp = np.sum((~true_cls) & augmented_pred_cls)
        augmented_fn = np.sum(true_cls & (~augmented_pred_cls))
        
        augmented_precision = augmented_tp / max(augmented_tp + augmented_fp, 1)
        augmented_recall = augmented_tp / max(augmented_tp + augmented_fn, 1)
        augmented_f1 = 2 * augmented_precision * augmented_recall / max(augmented_precision + augmented_recall, 1e-10)
        
        # Add to comparison
        recall_comparison.append({
            'Class': cls,
            'Original Precision': orig_precision,
            'Augmented Precision': augmented_precision,
            'Original Recall': orig_recall,
            'Augmented Recall': augmented_recall,
            'Original F1': orig_f1,
            'Augmented F1': augmented_f1,
            'Precision Improvement': augmented_precision - orig_precision,
            'Recall Improvement': augmented_recall - orig_recall,
            'F1 Improvement': augmented_f1 - orig_f1,
            'Support': np.sum(true_cls)
        })
    
    # Create dataframe for visualization
    recall_df = pd.DataFrame(recall_comparison)
    
    # Sort by F1 improvement (descending)
    recall_df = recall_df.sort_values('F1 Improvement', ascending=False)
    
    # Save detailed metrics comparison to CSV
    recall_df.to_csv(RES / "class_metrics_comparison.csv", index=False)
    
    # Create a simplified version for plotting
    plot_df = pd.DataFrame({
        'Class': recall_df['Class'],
        'Original': recall_df['Original F1'],
        'Augmented': recall_df['Augmented F1'],
        'Improvement': recall_df['F1 Improvement']
    })
    
    # Plot class improvement heatmap
    plot_class_improvement_heatmap(plot_df, title="F1 Score Improvement by Class")
    
    # Plot top 10 classes with most F1 improvement
    plt.figure(figsize=(12, 8))
    top10 = recall_df.head(10)
    
    plt.barh(top10['Class'], top10['F1 Improvement'], color='green')
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    plt.title('Top 10 Classes with Highest F1 Score Improvement')
    plt.xlabel('F1 Score Improvement')
    plt.tight_layout()
    plt.savefig(RES / "xgb_f1_improvement.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot classes with biggest decline (if any)
    bottom10 = recall_df.tail(10)
    if bottom10['F1 Improvement'].min() < 0:
        plt.figure(figsize=(12, 8))
        plt.barh(bottom10['Class'], bottom10['F1 Improvement'], color='red')
        plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        plt.title('Classes with F1 Score Decline')
        plt.xlabel('F1 Score Decline')
        plt.tight_layout()
        plt.savefig(RES / "xgb_f1_decline.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # Extract overall metrics from classification reports
    orig_metrics = {
        'accuracy': float(orig_report.split('\n')[-2].split()[-1]),
        'macro_f1': float(orig_report.split('\n')[-4].split()[-2]),
        'weighted_f1': float(orig_report.split('\n')[-3].split()[-2])
    }
    
    augmented_metrics = {
        'accuracy': augmented_report['accuracy'],
        'macro_f1': augmented_report['macro avg']['f1-score'],
        'weighted_f1': augmented_report['weighted avg']['f1-score']
    }
    
    # Create summary visualization
    plt.figure(figsize=(10, 6))
    metrics = ['Accuracy', 'Macro F1', 'Weighted F1']
    original = [orig_metrics['accuracy'], orig_metrics['macro_f1'], orig_metrics['weighted_f1']]
    augmented = [augmented_metrics['accuracy'], augmented_metrics['macro_f1'], augmented_metrics['weighted_f1']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.bar(x - width/2, original, width, label='Original only', color='lightblue')
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
        'augmented': augmented_metrics,
        'probabilities': {
            'original': orig_probs,
            'augmented': augmented_probs
        },
        'label_encoder': le
    }

def plot_class_distribution_comparison(orig_labels, augmented_labels):
    """Plot class distribution before and after augmentation."""
    orig_counts = pd.Series(orig_labels).value_counts()
    augmented_counts = pd.Series(augmented_labels).value_counts()
    
    # Combine into a dataframe
    combined_df = pd.DataFrame({
        'Original': orig_counts,
        'Augmented': augmented_counts
    }).fillna(0).sort_values('Augmented', ascending=False)
    
    # Calculate increase
    combined_df['Increase'] = combined_df['Augmented'] - combined_df['Original']
    combined_df['Increase_Pct'] = (combined_df['Increase'] / combined_df['Original'] * 100).replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Plot top 20 classes
    plt.figure(figsize=(15, 10))
    top20 = combined_df.head(20)
    
    # Create a grouped bar chart
    x = np.arange(len(top20))
    width = 0.4
    
    # Plot bars
    plt.bar(x - width/2, top20['Original'], width, label='Original', color='steelblue')
    plt.bar(x + width/2, top20['Augmented'], width, label='Augmented', color='darkorange')
    
    # Add labels and title
    plt.xlabel('Class')
    plt.ylabel('Count (log scale)')
    plt.title('Class Distribution: Original vs Augmented Training Data')
    plt.xticks(x, top20.index, rotation=45, ha='right')
    plt.yscale('log')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(RES / "class_distribution_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save table to CSV
    combined_df.to_csv(RES / "class_distribution_comparison.csv")
    
    return combined_df

def main():
    """Main function to evaluate VAE-GAN augmentation with XGBoost."""
    # Load data
    data_tuple = load_data()
    
    # Plot class distribution comparison
    orig_train_labels = data_tuple[1]
    augmented_train_labels = data_tuple[3]
    plot_class_distribution_comparison(orig_train_labels, augmented_train_labels)
    
    # Evaluate with XGBoost
    results = evaluate_with_xgboost(data_tuple)
    
    # Perform statistical significance testing
    sig_results = statistical_significance_test(data_tuple)
    
    # Print summary
    print("\nEvaluation Summary:")
    print(f"Original model accuracy: {results['original']['accuracy']:.4f}")
    print(f"Augmented model accuracy: {results['augmented']['accuracy']:.4f}")
    print(f"Improvement: {results['augmented']['accuracy'] - results['original']['accuracy']:.4f}")
    
    print(f"\nOriginal model weighted F1: {results['original']['weighted_f1']:.4f}")
    print(f"Augmented model weighted F1: {results['augmented']['weighted_f1']:.4f}")
    print(f"Improvement: {results['augmented']['weighted_f1'] - results['original']['weighted_f1']:.4f}")
    
    print(f"\nStatistical significance: {'Yes' if sig_results['significant'] else 'No'} (p={sig_results['p_value']:.6f})")
    
    print(f"\nEvaluation complete. Results saved to {RES}")

if __name__ == "__main__":
    main()