#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GAN Evaluation for Log Data Augmentation
----------------------------------------
Evaluates the effectiveness of GAN-based data augmentation by comparing
model performance with and without augmented data.
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
from sklearn.metrics import classification_report, confusion_matrix, recall_score, precision_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from tqdm import tqdm

# Project paths
ROOT = Path(__file__).resolve().parent.parent
EMB = ROOT / 'embeddings'
MOD = ROOT / 'models'
RES = ROOT / 'results'
AUG = ROOT / 'augmented'

# For Apple Silicon optimization
CPU_COUNT = os.cpu_count()
if CPU_COUNT:
    N_JOBS = max(1, CPU_COUNT - 1)  # Leave one core free
else:
    N_JOBS = -1

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Model definitions
MODELS = {
    'rf': RandomForestClassifier(
        n_estimators=100, 
        random_state=RANDOM_SEED, 
        n_jobs=N_JOBS,
        verbose=0
    ),
    'xgb': XGBClassifier(
        n_estimators=100, 
        learning_rate=0.1, 
        random_state=RANDOM_SEED, 
        n_jobs=N_JOBS,
        tree_method='hist',
        enable_categorical=True,
        use_label_encoder=False
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

def load_data(use_augmented=False):
    """Load original or augmented embeddings and labels."""
    print(f"Loading {'augmented' if use_augmented else 'original'} data...")
    
    if use_augmented:
        # Check if augmented data exists
        if not (AUG / "augmented_train_embeddings.pkl").exists():
            print("Error: Augmented files not found. Run gan_augmentation.py first.")
            sys.exit(1)
            
        # Load augmented data
        try:
            with open(AUG / 'augmented_train_embeddings.pkl', 'rb') as f:
                X_train = pickle.load(f)
            with open(AUG / 'augmented_train_labels.pkl', 'rb') as f:
                y_train = pickle.load(f)
        except FileNotFoundError:
            print("Error: Augmented files not found. Run gan_augmentation.py first.")
            sys.exit(1)
    else:
        # Load original data
        try:
            with open(EMB / 'train_embeddings.pkl', 'rb') as f:
                X_train = pickle.load(f)
            with open(EMB / 'train_labels.pkl', 'rb') as f:
                y_train_raw = pickle.load(f)
                y_train = parse_labels(y_train_raw)
        except FileNotFoundError:
            print("Error: Original embedding files not found. Run embedding script first.")
            sys.exit(1)
    
    # Always load the original test set
    try:
        with open(EMB / 'test_embeddings.pkl', 'rb') as f:
            X_test = pickle.load(f)
        with open(EMB / 'test_labels.pkl', 'rb') as f:
            y_test_raw = pickle.load(f)
            y_test = parse_labels(y_test_raw)
    except FileNotFoundError:
        print("Error: Test files not found. Run embedding script first.")
        sys.exit(1)
    
    print(f"Loaded {len(X_train)} training samples, {len(X_test)} test samples")
    
    # Display class distribution
    train_dist = pd.Series(y_train).value_counts()
    print("\nTraining class distribution:")
    for class_name, count in train_dist.items():
        print(f"  {class_name}: {count}")
    
    return X_train, y_train, X_test, y_test

def evaluate_model(model, X_train, y_train, X_test, y_test, model_name, data_type):
    """Train and evaluate model with given data."""
    print(f"\nTraining {model_name} on {data_type} data...")
    
    # Create label encoder for XGBoost
    le = LabelEncoder()
    all_labels = np.unique(np.concatenate([y_train, y_test]))
    le.fit(all_labels)
    
    # Train the model
    if model_name == 'xgb':
        # For XGBoost, use encoded labels
        y_train_encoded = le.transform(y_train)
        model.fit(X_train, y_train_encoded)
        
        # Predict
        y_pred_encoded = model.predict(X_test)
        y_pred = le.inverse_transform(y_pred_encoded)
    else:
        # For other models
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
    # Calculate metrics
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    # Calculate class-specific metrics
    class_metrics = {}
    for label in np.unique(np.concatenate([y_test, y_pred])):
        class_metrics[label] = {
            'precision': precision_score(y_test, y_pred, labels=[label], average=None, zero_division=0)[0],
            'recall': recall_score(y_test, y_pred, labels=[label], average=None, zero_division=0)[0],
            'f1': f1_score(y_test, y_pred, labels=[label], average=None, zero_division=0)[0],
            'support': np.sum(np.array(y_test) == label)
        }
    
    # Generate classification report
    report = classification_report(y_test, y_pred, zero_division=0)
    print("\nClassification Report:")
    print(report)
    
    # Save report to file
    with open(RES / f"{model_name}_{data_type}_report.txt", 'w') as f:
        f.write(f"{model_name.upper()} - {data_type.capitalize()} Data\n")
        f.write("-" * 50 + "\n")
        f.write(report)
    
    # Create confusion matrix visualization
    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)  # Replace NaN with 0
    
    # Get unique labels
    unique_labels = np.unique(np.concatenate([y_test, y_pred]))
    
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
               xticklabels=unique_labels, 
               yticklabels=unique_labels)
    plt.title(f'{model_name.upper()} - {data_type.capitalize()} Data - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(RES / f"{model_name}_{data_type}_cm.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'class_metrics': class_metrics
    }

def compare_metrics(original_metrics, augmented_metrics, model_name):
    """Compare and visualize metrics between original and augmented data."""
    # Get class metrics
    orig_class = original_metrics['class_metrics']
    aug_class = augmented_metrics['class_metrics']
    
    # Get all unique classes
    all_classes = sorted(set(list(orig_class.keys()) + list(aug_class.keys())))
    
    # Calculate improvement for each class
    improvements = {}
    for cls in all_classes:
        orig = orig_class.get(cls, {'f1': 0, 'precision': 0, 'recall': 0, 'support': 0})
        aug = aug_class.get(cls, {'f1': 0, 'precision': 0, 'recall': 0, 'support': 0})
        
        improvements[cls] = {
            'f1_change': aug['f1'] - orig['f1'],
            'precision_change': aug['precision'] - orig['precision'],
            'recall_change': aug['recall'] - orig['recall'],
            'support': orig['support']
        }
    
    # Create dataframe for visualization
    df = pd.DataFrame(columns=['Class', 'Metric', 'Original', 'Augmented', 'Change', 'Support'])
    
    for cls in all_classes:
        orig = orig_class.get(cls, {'f1': 0, 'precision': 0, 'recall': 0, 'support': 0})
        aug = aug_class.get(cls, {'f1': 0, 'precision': 0, 'recall': 0, 'support': 0})
        support = orig['support']
        
        for metric in ['precision', 'recall', 'f1']:
            df = pd.concat([df, pd.DataFrame({
                'Class': [cls],
                'Metric': [metric],
                'Original': [orig[metric]],
                'Augmented': [aug[metric]],
                'Change': [aug[metric] - orig[metric]],
                'Support': [support]
            })], ignore_index=True)
    
    # Sort by absolute change (descending)
    df = df.sort_values('Change', key=abs, ascending=False)
    
    # Save the comparison to CSV
    df.to_csv(RES / f"{model_name}_metric_comparison.csv", index=False)
    
    # Create comparison chart for top classes with significant changes
    plt.figure(figsize=(12, 10))
    
    # Get top 10 classes with most significant changes (by absolute F1 change)
    f1_changes = df[df['Metric'] == 'f1'].sort_values('Change', key=abs, ascending=False).head(10)
    
    # Plot bar chart
    plt.barh(f1_changes['Class'], f1_changes['Change'], color=['green' if x > 0 else 'red' for x in f1_changes['Change']])
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    plt.title(f'{model_name.upper()} - F1 Score Change After GAN Augmentation')
    plt.xlabel('F1 Score Change')
    plt.ylabel('Class')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(RES / f"{model_name}_f1_changes.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a summary of overall metrics
    summary = pd.DataFrame([
        {'Metric': 'Precision', 'Original': original_metrics['precision'], 'Augmented': augmented_metrics['precision'], 
         'Change': augmented_metrics['precision'] - original_metrics['precision']},
        {'Metric': 'Recall', 'Original': original_metrics['recall'], 'Augmented': augmented_metrics['recall'], 
         'Change': augmented_metrics['recall'] - original_metrics['recall']},
        {'Metric': 'F1 Score', 'Original': original_metrics['f1'], 'Augmented': augmented_metrics['f1'], 
         'Change': augmented_metrics['f1'] - original_metrics['f1']}
    ])
    
    # Save the summary
    summary.to_csv(RES / f"{model_name}_overall_summary.csv", index=False)
    
    # Create overall metrics comparison chart
    plt.figure(figsize=(10, 6))
    
    metrics = ['Precision', 'Recall', 'F1 Score']
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.bar(x - width/2, [original_metrics['precision'], original_metrics['recall'], original_metrics['f1']], 
            width, label='Original', color='lightblue')
    plt.bar(x + width/2, [augmented_metrics['precision'], augmented_metrics['recall'], augmented_metrics['f1']], 
            width, label='Augmented', color='orange')
    
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title(f'{model_name.upper()} - Impact of GAN Augmentation on Overall Model Performance')
    plt.xticks(x, metrics)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(RES / f"{model_name}_overall_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Return the summary
    return summary

def main():
    """Main function to orchestrate GAN evaluation."""
    parser = argparse.ArgumentParser(description='Evaluate GAN-based data augmentation')
    parser.add_argument('--models', choices=['rf', 'xgb', 'all'], default='all',
                        help='Models to evaluate (default: all)')
    parser.add_argument('--data', choices=['original', 'augmented', 'both'], default='both',
                        help='Data to use for evaluation (default: both)')
    args = parser.parse_args()
    
    # Determine which models to evaluate
    model_list = list(MODELS.keys()) if args.models == 'all' else [args.models]
    
    print("=" * 80)
    print("Evaluating GAN-based Data Augmentation")
    print("=" * 80)
    
    all_summaries = {}
    
    for model_name in model_list:
        print(f"\n{'='*50}")
        print(f"Evaluating {model_name.upper()} model")
        print(f"{'='*50}")
        
        # Clone the model to ensure fresh instances
        model = MODELS[model_name]
        
        # Load test data always
        _, _, X_test, y_test = load_data(use_augmented=False)
        
        # Evaluate with original data if requested
        original_metrics = None
        if args.data in ['original', 'both']:
            X_train_orig, y_train_orig, _, _ = load_data(use_augmented=False)
            original_metrics = evaluate_model(
                model, X_train_orig, y_train_orig, X_test, y_test, 
                model_name, "original"
            )
            # Need to create a fresh model instance for the next run
            model = MODELS[model_name]
        
        # Evaluate with augmented data if requested
        augmented_metrics = None
        if args.data in ['augmented', 'both']:
            X_train_aug, y_train_aug, _, _ = load_data(use_augmented=True)
            augmented_metrics = evaluate_model(
                model, X_train_aug, y_train_aug, X_test, y_test, 
                model_name, "augmented"
            )
        
        # Compare and visualize results if both were evaluated
        if args.data == 'both' and original_metrics and augmented_metrics:
            summary = compare_metrics(original_metrics, augmented_metrics, model_name)
            all_summaries[model_name] = summary
        else:
            # If only one type of data was evaluated, print a message
            print(f"\nEvaluation complete for {args.data} data.")
            print(f"To compare performance, use --data both")
    
    # Create final summary across all models if comparing both datasets
    if args.data == 'both' and len(model_list) > 1:
        print("\nOverall comparison across all models:")
        
        # Combine summaries
        df_all = pd.DataFrame(columns=['Model', 'Metric', 'Original', 'Augmented', 'Change'])
        
        for model_name, summary in all_summaries.items():
            model_df = summary.copy()
            model_df['Model'] = model_name.upper()
            df_all = pd.concat([df_all, model_df[['Model', 'Metric', 'Original', 'Augmented', 'Change']]], 
                              ignore_index=True)
        
        # Save combined summary
        df_all.to_csv(RES / "all_models_comparison.csv", index=False)
        
        # Create visualization comparing all models
        plt.figure(figsize=(12, 8))
        
        f1_by_model = df_all[df_all['Metric'] == 'F1 Score']
        
        models = f1_by_model['Model'].unique()
        x = np.arange(len(models))
        width = 0.35
        
        plt.bar(x - width/2, f1_by_model['Original'], width, label='Original', color='lightblue')
        plt.bar(x + width/2, f1_by_model['Augmented'], width, label='Augmented', color='orange')
        
        # Add absolute values on bars
        for i, value in enumerate(f1_by_model['Original']):
            plt.text(i - width/2, value + 0.01, f'{value:.3f}', ha='center', va='bottom')
        
        for i, value in enumerate(f1_by_model['Augmented']):
            plt.text(i + width/2, value + 0.01, f'{value:.3f}', ha='center', va='bottom')
        
        plt.xlabel('Model')
        plt.ylabel('F1 Score')
        plt.title('Impact of GAN Augmentation on Model Performance')
        plt.xticks(x, models)
        plt.ylim(0, 1)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(RES / "all_models_f1_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"\nEvaluation complete! Results saved to {RES}")

if __name__ == '__main__':
    main()