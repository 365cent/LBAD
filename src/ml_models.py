#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Optimized Traditional ML for Log Analysis
-----------------------------------------
High-performance traditional ML methods (XGBoost, Random Forest, etc.)
for log-based anomaly detection. Ultra-optimized for Apple Silicon.
"""

import os
import sys
import json
import pickle
import argparse
import time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import multiprocessing
from functools import partial
from tqdm import tqdm
import tensorflow as tf

# Project paths
ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / 'processed'
EMBEDDINGS_DIR = ROOT / 'embeddings'
MODELS_DIR = ROOT / 'models'
RESULTS_DIR = ROOT / 'results'

# Create directories if they don't exist
for dir_path in [MODELS_DIR, RESULTS_DIR]:
    dir_path.mkdir(exist_ok=True)

# For Apple Silicon optimization: tune thread counts
CPU_COUNT = os.cpu_count()
if CPU_COUNT:
    N_JOBS = max(1, CPU_COUNT - 1)  # Leave one core free
else:
    N_JOBS = -1  # Use all cores

# Start timing
start_time = time.time()

# Define models with optimized parameters
def create_models(random_state=42):
    """Create ML models optimized for Apple Silicon."""
    return {
        'rf': RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=10,
            min_samples_leaf=4,
            max_features='sqrt',
            bootstrap=True,
            random_state=random_state,
            n_jobs=N_JOBS,
            verbose=0,
            class_weight='balanced'  # Handle imbalanced classes
        ),
        'xgb': XGBClassifier(
            n_estimators=100,
            max_depth=8,
            learning_rate=0.1,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1,
            random_state=random_state,
            n_jobs=N_JOBS,
            tree_method='hist',  # Much faster on Apple Silicon
            enable_categorical=False,
            use_label_encoder=False,
            verbosity=0
        ),
        'svm': SVC(
            kernel='rbf',
            C=10.0,
            gamma='scale',
            probability=True,
            random_state=random_state,
            cache_size=2000,  # Allocate more cache for faster training
            class_weight='balanced'  # Handle imbalanced classes
        ),
        'knn': KNeighborsClassifier(
            n_neighbors=5,
            weights='distance',
            algorithm='auto',
            leaf_size=30,
            n_jobs=N_JOBS
        ),
        'lr': LogisticRegression(
            penalty='l2',
            C=1.0,
            solver='saga',  # Fast solver that handles all penalties
            max_iter=200,
            random_state=random_state,
            n_jobs=N_JOBS,
            class_weight='balanced',  # Handle imbalanced classes
            verbose=0
        )
    }

def parse_example(example):
    """Parse a TensorFlow Example protocol buffer."""
    feature_description = {
        'l': tf.io.FixedLenFeature([], tf.string),  # log
        'y': tf.io.FixedLenFeature([], tf.string),  # label
    }
    return tf.io.parse_single_example(example, feature_description)

def load_embedding_data(embedding_type='fasttext'):
    """Load embeddings and labels from pickle files."""
    print(f"Loading {embedding_type} embeddings...")
    
    prefix = f"{embedding_type}_" if embedding_type != 'fasttext' else ""
    
    try:
        with open(EMBEDDINGS_DIR / f'{prefix}train_embeddings.pkl', 'rb') as f:
            X_train = pickle.load(f)
        with open(EMBEDDINGS_DIR / f'{prefix}test_embeddings.pkl', 'rb') as f:
            X_test = pickle.load(f)
        with open(EMBEDDINGS_DIR / f'{prefix}train_labels.pkl', 'rb') as f:
            y_train_raw = pickle.load(f)
        with open(EMBEDDINGS_DIR / f'{prefix}test_labels.pkl', 'rb') as f:
            y_test_raw = pickle.load(f)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"Make sure to run the embedding generation script first.")
        sys.exit(1)
    
    # Parse JSON label strings
    y_train = parse_labels(y_train_raw)
    y_test = parse_labels(y_test_raw)
    
    print(f"Loaded {len(X_train)} training samples, {len(X_test)} test samples")
    
    return X_train, y_train, X_test, y_test

def parse_labels(label_strings):
    """Convert JSON label strings to classification labels."""
    parsed = []
    for label in label_strings:
        try:
            data = json.loads(label)
            if isinstance(data, list):
                if not data:  # Empty array means "normal"
                    parsed.append("normal")
                else:
                    parsed.append(data[0])  # Use first label if multiple
            else:
                parsed.append("unknown")
        except:
            parsed.append("unknown")
    return parsed

def load_tfrecord_data(max_samples=None, vectorize=True):
    """Load data directly from TFRecord files and vectorize."""
    print("Loading data from TFRecord files...")
    tfrecord_files = list(PROCESSED_DIR.glob("**/*.tfrecord"))
    
    if not tfrecord_files:
        print(f"No TFRecord files found in {PROCESSED_DIR}")
        sys.exit(1)
    
    logs = []
    labels = []
    
    # Process each TFRecord file
    total_processed = 0
    for file_path in tfrecord_files:
        try:
            dataset = tf.data.TFRecordDataset(str(file_path), compression_type="GZIP")
            
            for raw_record in dataset:
                if max_samples and total_processed >= max_samples:
                    break
                    
                parsed = parse_example(raw_record)
                log = parsed['l'].numpy().decode('utf-8')
                label = parsed['y'].numpy().decode('utf-8')
                
                logs.append(log)
                labels.append(label)
                total_processed += 1
                
        except Exception as e:
            print(f"Error with {file_path}: {e}")
    
    print(f"Loaded {len(logs)} log entries")
    
    # Parse JSON labels
    parsed_labels = parse_labels(labels)
    
    if not vectorize:
        return logs, parsed_labels
    
    # Vectorize logs with TF-IDF (optimized for performance)
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    print("Vectorizing logs (TF-IDF)...")
    vectorizer = TfidfVectorizer(
        max_features=10000,
        min_df=2,
        max_df=0.9,
        ngram_range=(1, 2),
        sublinear_tf=True
    )
    
    X = vectorizer.fit_transform(logs)
    
    # Save vectorizer for future use
    joblib.dump(vectorizer, MODELS_DIR / 'tfidf_vectorizer.joblib')
    
    print(f"Vectorized to {X.shape[1]} features")
    
    return X, parsed_labels

def train_evaluate_model(model_name, model, X_train, y_train, X_test, y_test, label_encoder, results_dir):
    """Train and evaluate a single model."""
    model_path = MODELS_DIR / f'{model_name}.joblib'
    model_start_time = time.time()
    
    # Train the model
    print(f"Training {model_name.upper()} model...")
    model.fit(X_train, y_train)
    
    # Save the trained model
    joblib.dump(model, model_path)
    training_time = time.time() - model_start_time
    
    # Generate predictions
    y_pred = model.predict(X_test)
    
    # Convert numerical predictions back to strings for SVM and other models if needed
    if isinstance(y_pred[0], (np.integer, int)):
        y_pred = label_encoder.inverse_transform(y_pred)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    
    # Save metrics
    with open(results_dir / f'{model_name}_report.txt', 'w') as f:
        f.write(f"{model_name.upper()} Classification Report\n")
        f.write(f"Training time: {training_time:.2f} seconds\n")
        f.write("-" * 50 + "\n")
        f.write(report)
    
    # Create confusion matrix
    labels = np.unique(np.concatenate([y_test, y_pred]))
    
    # Limit to top 15 classes by frequency for better visualization
    label_counts = pd.Series(y_test).value_counts()
    top_labels = label_counts.head(15).index.tolist()
    
    # Filter confusion matrix to include only top labels
    mask_test = np.isin(y_test, top_labels)
    mask_pred = np.isin(y_pred, top_labels)
    
    # Create boolean mask for rows that have both test and pred in top labels
    combined_mask = mask_test & mask_pred
    
    # Apply mask to create filtered versions of y_test and y_pred
    y_test_filtered = np.array(y_test)[combined_mask]
    y_pred_filtered = np.array(y_pred)[combined_mask]
    
    # Generate confusion matrix
    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(y_test_filtered, y_pred_filtered, labels=top_labels)
    
    # Normalize for better visualization
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)  # Replace NaN with 0
    
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
               xticklabels=top_labels, 
               yticklabels=top_labels)
    plt.title(f'{model_name.upper()} - Confusion Matrix (Top 15 Classes)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(results_dir / f'{model_name}_cm.png', dpi=300)
    plt.close()
    
    # Calculate per-class metrics for detailed analysis
    class_metrics = {}
    
    for label in np.unique(y_test):
        # Create mask for this class
        mask = (np.array(y_test) == label)
        
        # Calculate metrics
        true_positives = np.sum((np.array(y_pred) == label) & mask)
        support = np.sum(mask)
        
        if support > 0:
            precision = true_positives / np.sum(np.array(y_pred) == label) if np.sum(np.array(y_pred) == label) > 0 else 0
            recall = true_positives / support
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            class_metrics[label] = {
                'precision': precision,
                'recall': recall,
                'f1-score': f1,
                'support': int(support)
            }
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame.from_dict(class_metrics, orient='index')
    metrics_df = metrics_df.sort_values('support', ascending=False)
    metrics_df.to_csv(results_dir / f'{model_name}_class_metrics.csv')
    
    return {
        'model_name': model_name,
        'accuracy': accuracy,
        'training_time': training_time
    }

def main():
    """Main function for training and evaluating ML models."""
    parser = argparse.ArgumentParser(description='Optimized ML analysis for log data')
    parser.add_argument('--model', choices=['rf', 'xgb', 'svm', 'knn', 'lr', 'all'], 
                        default='all', help='Model to train (default: all)')
    parser.add_argument('--data-source', choices=['embedding', 'tfrecord'], 
                        default='embedding', help='Data source (default: embedding)')
    parser.add_argument('--embedding-type', choices=['fasttext', 'word2vec', 'tfidf'], 
                        default='fasttext', help='Embedding type (default: fasttext)')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum number of samples to use (for tfrecord only)')
    parser.add_argument('--no-train', action='store_true',
                        help='Skip training and only evaluate existing models')
    args = parser.parse_args()

    # Create results directory for this run
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = RESULTS_DIR / f"run_{timestamp}"
    run_dir.mkdir(exist_ok=True)
    
    # Load data based on source
    if args.data_source == 'embedding':
        X_train, y_train, X_test, y_test = load_embedding_data(args.embedding_type)
    else:  # tfrecord
        # For TFRecord source, we'll use cross-validation since we don't have a predefined split
        X, y = load_tfrecord_data(args.max_samples)
        
        # Create a train/test split
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if len(set(y)) < len(y)/2 else None
        )
    
    # Create label encoder
    le = LabelEncoder()
    all_labels = np.unique(np.concatenate([y_train, y_test]))
    le.fit(all_labels)
    
    # Save label encoder
    joblib.dump(le, MODELS_DIR / 'label_encoder.joblib')
    
    # Convert labels for models that require numeric labels
    y_train_enc = le.transform(y_train)
    y_test_enc = le.transform(y_test)
    
    # Display dataset statistics
    print(f"\nDataset Statistics:")
    print(f"Training samples: {len(y_train)}")
    print(f"Test samples: {len(y_test)}")
    print(f"Total unique labels: {len(all_labels)}")
    
    # Display label distribution
    label_counts = pd.Series(y_train).value_counts()
    normal_count = label_counts.get('normal', 0)
    abnormal_count = len(y_train) - normal_count
    
    print(f"\nTraining Set Label Distribution:")
    print(f"Normal logs: {normal_count} ({normal_count/len(y_train)*100:.2f}%)")
    print(f"Abnormal logs: {abnormal_count} ({abnormal_count/len(y_train)*100:.2f}%)")
    print(f"Top 5 anomaly types:")
    
    for label, count in label_counts[label_counts.index != 'normal'].head(5).items():
        print(f"  - {label}: {count} ({count/len(y_train)*100:.2f}%)")
    
    # Create and train models
    models = create_models()
    model_list = list(models.keys()) if args.model == 'all' else [args.model]
    
    results = []
    
    for model_name in model_list:
        print(f"\n{'-'*50}")
        print(f"Processing {model_name.upper()} model...")
        model = models[model_name]
        
        # Check if we should skip training
        model_path = MODELS_DIR / f'{model_name}.joblib'
        if args.no_train and model_path.exists():
            print(f"Loading existing model from {model_path}...")
            model = joblib.load(model_path)
        else:
            # For XGBoost and other models that need numeric labels
            if model_name in ['xgb']:
                result = train_evaluate_model(
                    model_name, model, X_train, y_train_enc, X_test, y_test, le, run_dir
                )
            else:
                result = train_evaluate_model(
                    model_name, model, X_train, y_train, X_test, y_test, le, run_dir
                )
            
            results.append(result)
    
    # Summarize results
    if results:
        print("\nModel Performance Summary:")
        print("-" * 70)
        print(f"{'Model':<15} {'Accuracy':<10} {'Training Time (s)':<20}")
        print("-" * 70)
        
        for result in sorted(results, key=lambda x: x['accuracy'], reverse=True):
            print(f"{result['model_name'].upper():<15} {result['accuracy']:.4f}     {result['training_time']:.2f}s")
    
    # Save summary to file
    with open(run_dir / 'summary.txt', 'w') as f:
        f.write(f"Log Analysis Run Summary - {timestamp}\n")
        f.write("-" * 50 + "\n")
        f.write(f"Data source: {args.data_source}\n")
        if args.data_source == 'embedding':
            f.write(f"Embedding type: {args.embedding_type}\n")
        f.write(f"Training samples: {len(y_train)}\n")
        f.write(f"Test samples: {len(y_test)}\n")
        f.write(f"Total unique labels: {len(all_labels)}\n\n")
        
        if results:
            f.write("Model Performance Summary:\n")
            f.write("-" * 50 + "\n")
            f.write(f"{'Model':<15} {'Accuracy':<10} {'Training Time (s)':<20}\n")
            f.write("-" * 50 + "\n")
            
            for result in sorted(results, key=lambda x: x['accuracy'], reverse=True):
                f.write(f"{result['model_name'].upper():<15} {result['accuracy']:.4f}     {result['training_time']:.2f}s\n")
        
        total_time = time.time() - start_time
        f.write(f"\nTotal execution time: {total_time:.2f}s\n")
    
    # Generate label distribution visualization
    plt.figure(figsize=(12, 8))
    top_labels = label_counts.head(10)
    
    # Add an "Other" category if there are more than 10 labels
    if len(label_counts) > 10:
        other_count = label_counts[10:].sum()
        top_labels = pd.concat([top_labels, pd.Series({'Other': other_count})])
    
    # Create pie chart
    plt.pie(
        top_labels, 
        labels=top_labels.index, 
        autopct='%1.1f%%',
        explode=[0.1 if label == 'normal' else 0 for label in top_labels.index],
        shadow=True, 
        startangle=90
    )
    plt.title('Label Distribution')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(run_dir / 'label_distribution.png', dpi=300)
    plt.close()
    
    print(f"\nResults saved to {run_dir}")
    print(f"Total execution time: {time.time() - start_time:.2f}s")

if __name__ == '__main__':
    main()