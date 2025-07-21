#!/usr/bin/env python3
"""
Transformer Model Evaluation Pipeline
====================================

Automatically evaluates trained transformer models using the best available method:

1. Direct Supervised Evaluation (AUTO-DETECTED):
   - Loads predictions.pkl saved during training
   - Applies sklearn metrics directly to saved predictions
   - Fast and straightforward evaluation
   
2. Model-based Evaluation (FALLBACK):
   - Loads TFRecord dataset and generates predictions from trained model
   - Used when predictions.pkl is not available
   - Slower but comprehensive analysis

Usage:
    # Automatic evaluation (detects best method)
    python src/evaluate_transformer.py --log-type wp-error
    
    # Using specific model path
    python src/evaluate_transformer.py --log-type wp-error --model-path models/transformer_wp-error.pth
"""

import argparse
import json
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import warnings

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import (
    precision_recall_fscore_support, f1_score, accuracy_score, 
    hamming_loss, jaccard_score, precision_score, recall_score
)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Import from transformer module
import sys
sys.path.append('.')
from src.transformer import UnsupervisedMultiLabelTransformer, SystemConfig, ProgressTracker, detect_system_resources


class TransformerEvaluator:
    """Handles evaluation of trained transformer models on full TFRecord datasets"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.device = torch.device(config.device)
    
    def evaluate_direct_supervised(self, log_type: str) -> Dict[str, Any]:
        """
        Direct supervised evaluation using saved predictions file.
        Loads predictions.pkl and applies sklearn metrics directly.
        """
        print(f"🔄 Loading predictions for direct supervised evaluation of {log_type}...")
        
        # Load predictions file
        predictions_file = Path(f"results/{log_type}/predictions.pkl")
        
        if not predictions_file.exists():
            raise FileNotFoundError(f"Predictions file not found: {predictions_file}")
        
        with open(predictions_file, 'rb') as f:
            data = pickle.load(f)
        
        # Extract data
        ids = data["ids"]
        probs = data["probs"]
        preds = data["preds"]
        
        # Check if true labels are available
        if "true_labels" not in data or data["true_labels"] is None:
            print(f"❌ No true labels found in predictions file - cannot perform supervised evaluation")
            return None
        
        y_true = data["true_labels"]
        
        print(f"✅ Loaded predictions for {len(ids):,} samples")
        print(f"📊 Shape: {probs.shape} (samples × classes)")
        print(f"📋 True labels available: {y_true.shape}")
        
        # Load model to get classes information
        model_path = self._find_model_path(log_type)
        if model_path:
            try:
                checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
                classes = checkpoint['classes']
                print(f"🏷️  Classes: {len(classes)} - {classes[:5]}{'...' if len(classes) > 5 else ''}")
            except:
                classes = [f"class_{i}" for i in range(probs.shape[1])]
                print(f"⚠️  Could not load class names, using generic names")
        else:
            classes = [f"class_{i}" for i in range(probs.shape[1])]
            print(f"⚠️  Model not found, using generic class names")
        
        # Calculate comprehensive metrics using sklearn
        print(f"🔍 Computing supervised evaluation metrics...")
        
        # Overall multi-label metrics
        subset_accuracy = accuracy_score(y_true, preds)
        hamming_loss_score = hamming_loss(y_true, preds)
        micro_f1 = f1_score(y_true, preds, average='micro', zero_division=0)
        macro_f1 = f1_score(y_true, preds, average='macro', zero_division=0)
        weighted_f1 = f1_score(y_true, preds, average='weighted', zero_division=0)
        samples_f1 = f1_score(y_true, preds, average='samples', zero_division=0)
        
        # Additional metrics
        micro_precision = precision_score(y_true, preds, average='micro', zero_division=0)
        macro_precision = precision_score(y_true, preds, average='macro', zero_division=0)
        micro_recall = recall_score(y_true, preds, average='micro', zero_division=0)
        macro_recall = recall_score(y_true, preds, average='macro', zero_division=0)
        
        # Jaccard scores
        jaccard_micro = jaccard_score(y_true, preds, average='micro', zero_division=0)
        jaccard_macro = jaccard_score(y_true, preds, average='macro', zero_division=0)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, preds, average=None, zero_division=0
        )
        
        # Print results
        print(f"\n📊 DIRECT SUPERVISED EVALUATION RESULTS")
        print("=" * 60)
        print(f"Subset Accuracy:  {subset_accuracy:.4f}")
        print(f"Hamming Loss:     {hamming_loss_score:.4f}")
        print(f"Micro F1:         {micro_f1:.4f}")
        print(f"Macro F1:         {macro_f1:.4f}")
        print(f"Weighted F1:      {weighted_f1:.4f}")
        print(f"Samples F1:       {samples_f1:.4f}")
        print(f"Micro Precision:  {micro_precision:.4f}")
        print(f"Macro Precision:  {macro_precision:.4f}")
        print(f"Micro Recall:     {micro_recall:.4f}")
        print(f"Macro Recall:     {macro_recall:.4f}")
        print(f"Jaccard (Micro):  {jaccard_micro:.4f}")
        print(f"Jaccard (Macro):  {jaccard_macro:.4f}")
        print("")
        
        # Generate classification report
        print("📋 PER-CLASS CLASSIFICATION REPORT:")
        print("-" * 60)
        from sklearn.metrics import classification_report
        class_report = classification_report(
            y_true, preds,
            target_names=classes,
            zero_division=0,
            digits=3
        )
        print(class_report)
        
        # Sample distribution analysis
        labels_per_sample = preds.sum(axis=1)
        true_labels_per_sample = y_true.sum(axis=1)
        
        print("📈 SAMPLE DISTRIBUTION ANALYSIS:")
        print("-" * 60)
        print(f"Average predicted labels per sample: {labels_per_sample.mean():.3f}")
        print(f"Average true labels per sample: {true_labels_per_sample.mean():.3f}")
        print(f"Predicted labels range: {labels_per_sample.min()} - {labels_per_sample.max()}")
        print(f"True labels range: {true_labels_per_sample.min()} - {true_labels_per_sample.max()}")
        print(f"Samples with no predicted labels: {(labels_per_sample == 0).sum():,}")
        print(f"Samples with no true labels: {(true_labels_per_sample == 0).sum():,}")
        print(f"Samples with multiple predicted labels: {(labels_per_sample > 1).sum():,}")
        print(f"Samples with multiple true labels: {(true_labels_per_sample > 1).sum():,}")
        
        # Compile results
        results = {
            'metrics': {
                'subset_accuracy': float(subset_accuracy),
                'hamming_loss': float(hamming_loss_score),
                'micro_f1': float(micro_f1),
                'macro_f1': float(macro_f1),
                'weighted_f1': float(weighted_f1),
                'samples_f1': float(samples_f1),
                'micro_precision': float(micro_precision),
                'macro_precision': float(macro_precision),
                'micro_recall': float(micro_recall),
                'macro_recall': float(macro_recall),
                'jaccard_micro': float(jaccard_micro),
                'jaccard_macro': float(jaccard_macro),
                'per_class_precision': precision.tolist(),
                'per_class_recall': recall.tolist(),
                'per_class_f1': f1.tolist(),
                'per_class_support': support.tolist(),
                'classes': classes,
                'n_test_samples': len(y_true),
                'evaluation_type': 'direct_supervised',
                'avg_predicted_labels_per_sample': float(labels_per_sample.mean()),
                'avg_true_labels_per_sample': float(true_labels_per_sample.mean()),
                'samples_with_no_predicted_labels': int((labels_per_sample == 0).sum()),
                'samples_with_no_true_labels': int((true_labels_per_sample == 0).sum()),
                'samples_with_multiple_predicted_labels': int((labels_per_sample > 1).sum()),
                'samples_with_multiple_true_labels': int((true_labels_per_sample > 1).sum()),
            },
            'predictions': preds,
            'probabilities': probs,
            'true_labels': y_true,
            'ids': ids,
            'classification_report': class_report
        }
        
        return results
    
    def _find_model_path(self, log_type: str) -> Optional[Path]:
        """Find model path for a given log type"""
        models_dir = Path("models")
        patterns = [
            f"transformer_{log_type}_{self.config.node_name}_{self.config.job_id}.pth",
            f"transformer_{log_type}_*.pth",
            f"transformer_{log_type}.pth"
        ]
        
        for pattern in patterns:
            matches = list(models_dir.glob(pattern))
            if matches:
                return max(matches, key=lambda p: p.stat().st_mtime)
        return None
        
    def load_tfrecord_dataset(self, log_type: str, target_dim: int = 768) -> Tuple[Optional[np.ndarray], Optional[List[str]], Optional[np.ndarray]]:
        """
        Load full dataset from TFRecord files for comprehensive evaluation.
        """
        print(f"🔄 Loading full TFRecord dataset for {log_type}...")
        load_start_time = time.time()
        
        # Check if TFRecord files exist for this log type
        processed_dir = Path("processed")
        log_type_dir = processed_dir / log_type
        
        if not log_type_dir.exists():
            print(f"❌ No TFRecord directory found for {log_type}")
            return None, None, None
        
        tfrecord_files = list(log_type_dir.glob("*.tfrecord"))
        if not tfrecord_files:
            print(f"❌ No TFRecord files found for {log_type}")
            return None, None, None
        
        print(f"📂 Found {len(tfrecord_files)} TFRecord files")
        
        try:
            import tensorflow as tf
            
            all_logs = []
            all_labels_json = []
            
            for file_idx, file_path in enumerate(tfrecord_files):
                print(f"   Loading file {file_idx+1}/{len(tfrecord_files)}: {file_path.name}")
                
                try:
                    dataset = tf.data.TFRecordDataset(str(file_path), compression_type="GZIP")
                    
                    for raw_record in dataset:
                        feature_description = {
                            'l': tf.io.FixedLenFeature([], tf.string),
                            'y': tf.io.FixedLenFeature([], tf.string),
                        }
                        parsed = tf.io.parse_single_example(raw_record, feature_description)
                        
                        log_line = parsed['l'].numpy().decode('utf-8')
                        labels_json = parsed['y'].numpy().decode('utf-8')
                        
                        all_logs.append(log_line)
                        all_labels_json.append(labels_json)
                        
                except Exception as e:
                    print(f"   ⚠️  Error loading {file_path}: {e}")
                    continue
            
            if not all_logs:
                print(f"❌ No data loaded from TFRecord files")
                return None, None, None
            
            print(f"✅ Loaded {len(all_logs):,} log entries")
            
            # Parse labels and collect unique classes
            all_labels_parsed = []
            all_classes = set()
            
            for labels_json in all_labels_json:
                try:
                    labels = json.loads(labels_json) if labels_json.strip() else []
                    if isinstance(labels, str):
                        labels = [labels]
                    elif not isinstance(labels, list):
                        labels = []
                    all_labels_parsed.append(labels)
                    all_classes.update(labels)
                except (json.JSONDecodeError, TypeError):
                    all_labels_parsed.append([])
            
            classes = sorted(list(all_classes))
            if not classes:
                classes = ['normal']
                
            print(f"📊 Found {len(classes)} classes: {classes[:5]}{'...' if len(classes) > 5 else ''}")
            
            # Create binary label matrix
            true_labels = np.zeros((len(all_logs), len(classes)), dtype=np.float32)
            for i, labels in enumerate(all_labels_parsed):
                for label in labels:
                    if label in classes:
                        true_labels[i, classes.index(label)] = 1.0
            
            # Generate embeddings using TF-IDF
            print(f"🔄 Generating {target_dim}D embeddings for {len(all_logs):,} log entries...")
            embeddings = self._generate_tfidf_embeddings(all_logs, target_dim=target_dim)
            
            load_time = time.time() - load_start_time
            print(f"✅ Dataset loading completed in {load_time:.1f}s")
            print(f"📊 Dataset: {len(embeddings):,} samples × {embeddings.shape[1]}D embeddings")
            
            return embeddings, classes, true_labels
            
        except Exception as e:
            print(f"❌ Error loading TFRecord dataset: {e}")
            return None, None, None
    
    def _generate_tfidf_embeddings(self, logs: List[str], target_dim: int = 768) -> np.ndarray:
        """Generate TF-IDF embeddings with dimensionality reduction"""
        vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english',
            lowercase=True
        )
        
        tfidf_matrix = vectorizer.fit_transform(logs)
        
        if tfidf_matrix.shape[1] > target_dim:
            svd = TruncatedSVD(n_components=target_dim, random_state=42)
            embeddings = svd.fit_transform(tfidf_matrix)
        else:
            embeddings = tfidf_matrix.toarray()
        
        # Normalize embeddings
        embeddings = normalize(embeddings, norm='l2', axis=1).astype(np.float32)
        return embeddings
    
    def load_transformer_embeddings_and_metadata(self, log_type: str) -> Tuple[Optional[np.ndarray], Optional[List[str]], Optional[np.ndarray]]:
        """
        Load transformer embeddings generated for the full dataset.
        This uses the embeddings created by the transformer after training.
        """
        print(f"🔄 Loading transformer embeddings for full {log_type} dataset...")
        load_start_time = time.time()
        
        # Look for transformer embeddings directory
        embeddings_base_dir = Path("embeddings")
        transformer_dir = embeddings_base_dir / f"{log_type}_transformer_full"
        
        if not transformer_dir.exists():
            print(f"❌ No transformer embeddings directory found: {transformer_dir}")
            print(f"   Full dataset embeddings are no longer generated during training.")
            print(f"   Use direct evaluation instead: python src/evaluate_transformer.py --log-type {log_type} --direct")
            return None, None, None
        
        # Find the most recent transformer embeddings and metadata files
        embeddings_files = list(transformer_dir.glob(f"transformer_embeddings_{log_type}_*.pkl"))
        metadata_files = list(transformer_dir.glob(f"metadata_{log_type}_*.pkl"))
        
        if not embeddings_files:
            print(f"❌ No transformer embeddings found in {transformer_dir}")
            return None, None, None
        
        if not metadata_files:
            print(f"❌ No metadata files found in {transformer_dir}")
            return None, None, None
        
        # Use the most recent files
        embeddings_file = max(embeddings_files, key=lambda p: p.stat().st_mtime)
        metadata_file = max(metadata_files, key=lambda p: p.stat().st_mtime)
        
        try:
            # Load transformer embeddings
            print(f"📂 Loading transformer embeddings from {embeddings_file.name}...")
            with open(embeddings_file, 'rb') as f:
                transformer_embeddings = pickle.load(f)
            
            # Load metadata
            print(f"📂 Loading metadata from {metadata_file.name}...")
            with open(metadata_file, 'rb') as f:
                metadata = pickle.load(f)
            
            # Extract information from metadata
            classes = metadata['classes']
            true_labels = metadata['true_labels']
            raw_logs = metadata['raw_logs']
            
            load_time = time.time() - load_start_time
            print(f"✅ Transformer embeddings loaded in {load_time:.1f}s")
            print(f"📊 Dataset: {len(transformer_embeddings):,} samples × {transformer_embeddings.shape[1]}D transformer embeddings")
            print(f"📊 Classes: {len(classes)} - {classes}")
            print(f"📄 Original logs: {len(raw_logs):,} entries available")
            
            return transformer_embeddings, classes, true_labels
            
        except Exception as e:
            print(f"❌ Error loading transformer embeddings: {e}")
            return None, None, None
    
    def load_trained_model(self, model_path: Path, log_type: str) -> Tuple[UnsupervisedMultiLabelTransformer, List[str], int]:
        """Load a trained transformer model and return input dimension"""
        print(f"📂 Loading trained model from {model_path}")
        
        try:
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            
            # Extract model configuration
            classes = checkpoint['classes']
            config_dict = checkpoint['config']
            
            # Determine input dimension from model state
            input_dim = None
            for key, tensor in checkpoint['model_state_dict'].items():
                if 'input_projection' in key and 'weight' in key:
                    input_dim = tensor.shape[1]
                    break
            
            if input_dim is None:
                # Fallback: try to infer from first layer
                for key, tensor in checkpoint['model_state_dict'].items():
                    if 'weight' in key and len(tensor.shape) == 2:
                        input_dim = tensor.shape[1]
                        break
            
            if input_dim is None:
                raise ValueError("Could not determine input dimension from model")
            
            # Create model with correct dimensions
            model = UnsupervisedMultiLabelTransformer(
                input_dim=input_dim,
                latent_dim=512,
                n_labels=len(classes),
                n_clusters=min(8, len(classes)),
                dropout=0.1
            )
            
            # Load model weights
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            model.eval()
            
            print(f"✅ Model loaded: {input_dim}D → {len(classes)} classes")
            return model, classes, input_dim
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    def evaluate_transformer_embeddings(self, embeddings: np.ndarray, true_labels: np.ndarray, 
                                       classes: List[str]) -> Dict[str, Any]:
        """Evaluate transformer embeddings using unsupervised metrics and clustering"""
        print(f"🔍 Evaluating transformer embeddings on {len(embeddings):,} samples...")
        
        # For unsupervised evaluation, we'll use clustering and similarity-based analysis
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score, calinski_harabasz_score
        
        # Perform clustering with number of clusters = number of classes
        n_clusters = len(classes)
        
        print(f"🔄 Performing K-means clustering with {n_clusters} clusters...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Calculate clustering metrics
        silhouette = silhouette_score(embeddings, cluster_labels)
        calinski_harabasz = calinski_harabasz_score(embeddings, cluster_labels)
        
        print(f"📊 Clustering metrics:")
        print(f"   Silhouette Score: {silhouette:.4f}")
        print(f"   Calinski-Harabasz Score: {calinski_harabasz:.2f}")
        
        # Create pseudo-predictions based on cluster assignment
        # Each cluster represents a potential class
        predictions = np.zeros((len(embeddings), len(classes)), dtype=np.float32)
        
        # For each cluster, calculate similarity to each true class
        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_samples = embeddings[cluster_mask]
            
            if len(cluster_samples) == 0:
                continue
                
            # Calculate cluster centroid
            cluster_center = cluster_samples.mean(axis=0)
            
            # For each sample in cluster, assign probability based on distance to center
            for idx in np.where(cluster_mask)[0]:
                # Distance-based probability (closer to center = higher probability)
                distance = np.linalg.norm(embeddings[idx] - cluster_center)
                # Convert distance to probability (lower distance = higher probability)
                max_distance = np.max([np.linalg.norm(embeddings[i] - cluster_center) 
                                     for i in np.where(cluster_mask)[0]])
                if max_distance > 0:
                    prob = 1.0 - (distance / max_distance)
                else:
                    prob = 1.0
                
                # Assign this probability to the corresponding cluster
                predictions[idx, cluster_id % len(classes)] = prob
        
        # Normalize predictions
        row_sums = predictions.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        predictions = predictions / row_sums
        
        # Set threshold and create binary predictions
        threshold = 0.3  # Fixed threshold for unsupervised evaluation
        binary_predictions = (predictions > threshold).astype(int)
        
        # Calculate metrics
        metrics = self._calculate_unsupervised_metrics(
            true_labels, binary_predictions, predictions, classes, embeddings, cluster_labels
        )
        
        return {
            'metrics': metrics,
            'predictions': predictions,
            'binary_predictions': binary_predictions,
            'cluster_labels': cluster_labels,
            'clustering_metrics': {
                'silhouette_score': silhouette,
                'calinski_harabasz_score': calinski_harabasz
            }
        }
    
    def _optimize_thresholds(self, y_true: np.ndarray, y_prob: np.ndarray) -> np.ndarray:
        """Optimize per-class thresholds for F1 score"""
        thresholds = np.full(y_true.shape[1], 0.5)
        
        for class_idx in range(y_true.shape[1]):
            class_true = y_true[:, class_idx]
            class_prob = y_prob[:, class_idx]
            
            if class_true.sum() == 0:  # No positive samples
                thresholds[class_idx] = 0.8  # High threshold
                continue
            
            best_f1 = 0
            best_threshold = 0.5
            
            for threshold in np.arange(0.1, 0.9, 0.05):
                class_pred = (class_prob > threshold).astype(int)
                f1 = f1_score(class_true, class_pred, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
            
            thresholds[class_idx] = best_threshold
        
        return thresholds
    
    def _calculate_unsupervised_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                      y_prob: np.ndarray, classes: List[str], 
                                      embeddings: np.ndarray, cluster_labels: np.ndarray) -> Dict[str, Any]:
        """Calculate metrics for unsupervised transformer embeddings evaluation"""
        
        metrics = {}
        
        # If we have true labels, calculate supervised metrics too
        if y_true is not None:
            metrics['subset_accuracy'] = float(accuracy_score(y_true, y_pred))
            metrics['hamming_loss'] = float(hamming_loss(y_true, y_pred))
            metrics['micro_f1'] = float(f1_score(y_true, y_pred, average='micro', zero_division=0))
            metrics['macro_f1'] = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
            metrics['weighted_f1'] = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
            metrics['samples_f1'] = float(f1_score(y_true, y_pred, average='samples', zero_division=0))
            
            # Jaccard scores
            metrics['jaccard_micro'] = float(jaccard_score(y_true, y_pred, average='micro', zero_division=0))
            metrics['jaccard_macro'] = float(jaccard_score(y_true, y_pred, average='macro', zero_division=0))
            
            # Per-class metrics
            precision, recall, f1, support = precision_recall_fscore_support(
                y_true, y_pred, average=None, zero_division=0
            )
            
            metrics['per_class_precision'] = precision.tolist()
            metrics['per_class_recall'] = recall.tolist()
            metrics['per_class_f1'] = f1.tolist()
            metrics['per_class_support'] = support.tolist()
        else:
            # For unsupervised case, create placeholder metrics
            metrics['subset_accuracy'] = 0.0
            metrics['hamming_loss'] = 0.0
            metrics['micro_f1'] = 0.0
            metrics['macro_f1'] = 0.0
            metrics['weighted_f1'] = 0.0
            metrics['samples_f1'] = 0.0
            metrics['jaccard_micro'] = 0.0
            metrics['jaccard_macro'] = 0.0
            metrics['per_class_precision'] = [0.0] * len(classes)
            metrics['per_class_recall'] = [0.0] * len(classes)
            metrics['per_class_f1'] = [0.0] * len(classes)
            metrics['per_class_support'] = [0] * len(classes)
        
        # Clustering-specific metrics
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
        
        if y_true is not None:
            # Convert multi-label to single-label for clustering evaluation
            y_true_single = np.argmax(y_true, axis=1)
            
            metrics['adjusted_rand_score'] = float(adjusted_rand_score(y_true_single, cluster_labels))
            metrics['normalized_mutual_info'] = float(normalized_mutual_info_score(y_true_single, cluster_labels))
        
        # Embedding quality metrics
        metrics['embedding_dimension'] = embeddings.shape[1]
        metrics['embedding_norm_mean'] = float(np.linalg.norm(embeddings, axis=1).mean())
        metrics['embedding_norm_std'] = float(np.linalg.norm(embeddings, axis=1).std())
        
        # Prediction confidence
        metrics['prediction_confidence_mean'] = float(y_prob.mean())
        metrics['prediction_confidence_std'] = float(y_prob.std())
        
        # Sample distribution
        labels_per_sample = y_pred.sum(axis=1)
        metrics['avg_labels_per_sample'] = float(labels_per_sample.mean())
        metrics['std_labels_per_sample'] = float(labels_per_sample.std())
        metrics['min_labels_per_sample'] = int(labels_per_sample.min())
        metrics['max_labels_per_sample'] = int(labels_per_sample.max())
        
        # Counts
        metrics['samples_with_no_labels'] = int((labels_per_sample == 0).sum())
        metrics['samples_with_one_label'] = int((labels_per_sample == 1).sum())
        metrics['samples_with_multiple_labels'] = int((labels_per_sample > 1).sum())
        
        # Test samples
        metrics['n_test_samples'] = len(y_true) if y_true is not None else len(y_pred)
        metrics['evaluation_type'] = 'unsupervised_transformer_embeddings'
        metrics['classes'] = classes
        
        # Add optimal thresholds (fixed threshold for unsupervised evaluation)
        metrics['optimal_thresholds'] = [0.3] * len(classes)
        
        return metrics
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                          y_prob: np.ndarray, classes: List[str], 
                          thresholds: np.ndarray) -> Dict[str, Any]:
        """Calculate comprehensive evaluation metrics"""
        
        metrics = {}
        
        # Overall metrics
        metrics['subset_accuracy'] = float(accuracy_score(y_true, y_pred))
        metrics['hamming_loss'] = float(hamming_loss(y_true, y_pred))
        metrics['micro_f1'] = float(f1_score(y_true, y_pred, average='micro', zero_division=0))
        metrics['macro_f1'] = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
        metrics['weighted_f1'] = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
        metrics['samples_f1'] = float(f1_score(y_true, y_pred, average='samples', zero_division=0))
        
        # Jaccard scores
        metrics['jaccard_micro'] = float(jaccard_score(y_true, y_pred, average='micro', zero_division=0))
        metrics['jaccard_macro'] = float(jaccard_score(y_true, y_pred, average='macro', zero_division=0))
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        metrics['per_class_precision'] = precision.tolist()
        metrics['per_class_recall'] = recall.tolist()
        metrics['per_class_f1'] = f1.tolist()
        metrics['per_class_support'] = support.tolist()
        metrics['optimal_thresholds'] = thresholds.tolist()
        metrics['classes'] = classes
        
        # Prediction confidence
        metrics['prediction_confidence_mean'] = float(y_prob.mean())
        metrics['prediction_confidence_std'] = float(y_prob.std())
        
        # Sample distribution
        labels_per_sample = y_pred.sum(axis=1)
        metrics['avg_labels_per_sample'] = float(labels_per_sample.mean())
        metrics['std_labels_per_sample'] = float(labels_per_sample.std())
        metrics['min_labels_per_sample'] = int(labels_per_sample.min())
        metrics['max_labels_per_sample'] = int(labels_per_sample.max())
        
        # Counts
        metrics['samples_with_no_labels'] = int((labels_per_sample == 0).sum())
        metrics['samples_with_one_label'] = int((labels_per_sample == 1).sum())
        metrics['samples_with_multiple_labels'] = int((labels_per_sample > 1).sum())
        
        # Test samples
        metrics['n_test_samples'] = len(y_true)
        metrics['evaluation_type'] = 'supervised'
        
        return metrics
    
    def generate_classification_report(self, results: Dict[str, Any], log_type: str, 
                                     output_dir: Path, training_time: float = 0.0) -> str:
        """Generate comprehensive classification report"""
        
        metrics = results['metrics']
        report_lines = []
        
        report_lines.append(f"TRANSFORMER Embeddings Evaluation Report - {log_type.upper()}")
        report_lines.append("=" * 80)
        report_lines.append(f"Training time: {training_time:.2f} seconds")
        report_lines.append(f"Test samples: {metrics['n_test_samples']:,}")
        report_lines.append(f"Number of classes: {len(metrics['classes'])}")
        report_lines.append(f"Evaluation type: {metrics['evaluation_type']}")
        report_lines.append(f"Embedding dimension: {metrics.get('embedding_dimension', 'N/A')}")
        report_lines.append(f"Node: {self.config.node_name} | Job: {self.config.job_id}")
        report_lines.append("")
        
        # Add clustering metrics if available
        if 'clustering_metrics' in results:
            clustering = results['clustering_metrics']
            report_lines.append("CLUSTERING METRICS:")
            report_lines.append("-" * 40)
            report_lines.append(f"Silhouette Score: {clustering['silhouette_score']:.4f}")
            report_lines.append(f"Calinski-Harabasz Score: {clustering['calinski_harabasz_score']:.2f}")
            if 'adjusted_rand_score' in metrics:
                report_lines.append(f"Adjusted Rand Score: {metrics['adjusted_rand_score']:.4f}")
            if 'normalized_mutual_info' in metrics:
                report_lines.append(f"Normalized Mutual Info: {metrics['normalized_mutual_info']:.4f}")
            report_lines.append("")
        
        report_lines.append("OVERALL METRICS:")
        report_lines.append("-" * 40)
        report_lines.append(f"Subset Accuracy: {metrics['subset_accuracy']:.4f}")
        report_lines.append(f"Hamming Loss: {metrics['hamming_loss']:.4f}")
        report_lines.append(f"Micro F1: {metrics['micro_f1']:.4f}")
        report_lines.append(f"Macro F1: {metrics['macro_f1']:.4f}")
        report_lines.append(f"Weighted F1: {metrics['weighted_f1']:.4f}")
        report_lines.append(f"Samples F1: {metrics['samples_f1']:.4f}")
        report_lines.append(f"Jaccard (Micro): {metrics['jaccard_micro']:.4f}")
        report_lines.append(f"Jaccard (Macro): {metrics['jaccard_macro']:.4f}")
        report_lines.append("")
        
        report_lines.append("PER-CLASS METRICS:")
        report_lines.append("-" * 40)
        report_lines.append(f"{'Class':<25} {'F1':<8} {'Precision':<10} {'Recall':<8} {'Support':<8} {'Threshold':<10}")
        report_lines.append("-" * 84)
        
        for i, cls in enumerate(metrics['classes']):
            f1 = metrics['per_class_f1'][i]
            precision = metrics['per_class_precision'][i]
            recall = metrics['per_class_recall'][i]
            support = metrics['per_class_support'][i]
            threshold = metrics.get('optimal_thresholds', [0.3] * len(metrics['classes']))[i]
            
            report_lines.append(f"{cls:<25} {f1:<8.3f} {precision:<10.3f} {recall:<8.3f} {support:<8} {threshold:<10.3f}")
        
        report_lines.append("")
        report_lines.append("SAMPLE DISTRIBUTION:")
        report_lines.append("-" * 40)
        report_lines.append(f"Samples with no labels: {metrics['samples_with_no_labels']:,}")
        report_lines.append(f"Samples with one label: {metrics['samples_with_one_label']:,}")
        report_lines.append(f"Samples with multiple labels: {metrics['samples_with_multiple_labels']:,}")
        report_lines.append(f"Average labels per sample: {metrics['avg_labels_per_sample']:.3f}")
        report_lines.append(f"Std labels per sample: {metrics['std_labels_per_sample']:.3f}")
        report_lines.append(f"Labels per sample range: {metrics['min_labels_per_sample']} - {metrics['max_labels_per_sample']}")
        
        report_content = "\n".join(report_lines)
        
        # Save report
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / f"transformer_evaluation_report_{log_type}_{self.config.node_name}_{self.config.job_id}.txt"
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        print(f"📊 Classification report saved to: {report_path}")
        return str(report_path)
    
    def save_results(self, results: Dict[str, Any], log_type: str, output_dir: Path):
        """Save evaluation results"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results_path = output_dir / f"transformer_evaluation_results_{log_type}_{self.config.node_name}_{self.config.job_id}.pkl"
        with open(results_path, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"💾 Results saved to: {results_path}")
    
    def print_summary(self, results: Dict[str, Any], log_type: str, training_time: float = 0.0):
        """Print evaluation summary"""
        metrics = results['metrics']
        
        print(f"\n" + "=" * 80)
        print(f"🎯 EVALUATION SUMMARY - {log_type.upper()}")
        print("=" * 80)
        print(f"Training time: {training_time:.2f} seconds")
        print(f"Test samples: {metrics['n_test_samples']:,}")
        print(f"Classes: {len(metrics['classes'])}")
        print(f"Evaluation type: {metrics['evaluation_type']}")
        print("")
        print("KEY METRICS:")
        print(f"  Macro F1:       {metrics['macro_f1']:.4f}")
        print(f"  Micro F1:       {metrics['micro_f1']:.4f}")
        print(f"  Weighted F1:    {metrics['weighted_f1']:.4f}")
        print(f"  Subset Accuracy: {metrics['subset_accuracy']:.4f}")
        print(f"  Hamming Loss:   {metrics['hamming_loss']:.4f}")
        print("")
        
        # Top performing classes
        class_f1_pairs = list(zip(metrics['classes'], metrics['per_class_f1']))
        class_f1_pairs.sort(key=lambda x: x[1], reverse=True)
        
        print("TOP CLASSES BY F1:")
        for i, (cls, f1) in enumerate(class_f1_pairs[:5]):
            print(f"  {i+1}. {cls:<30} F1: {f1:.4f}")
        print("=" * 80)


def find_model_path(log_type: str, config: SystemConfig) -> Optional[Path]:
    """Find the most recent model file for a log type"""
    models_dir = Path("models")
    
    # Try specific patterns
    patterns = [
        f"transformer_{log_type}_{config.node_name}_{config.job_id}.pth",
        f"transformer_{log_type}_*.pth",
        f"transformer_{log_type}.pth"
    ]
    
    for pattern in patterns:
        matches = list(models_dir.glob(pattern))
        if matches:
            # Return the most recent
            return max(matches, key=lambda p: p.stat().st_mtime)
    
    return None


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained transformer model")
    parser.add_argument("--log-type", type=str, required=True, 
                       help="Log type to evaluate (e.g., wp-access, wp-error)")
    parser.add_argument("--model-path", type=str, 
                       help="Path to trained model (auto-detected if not provided)")
    parser.add_argument("--output-dir", type=str, default="results", 
                       help="Output directory for results")
    parser.add_argument("--training-time", type=float, default=0.0,
                       help="Training time in seconds (for reporting)")

    
    args = parser.parse_args()
    
    # Detect system configuration
    config = detect_system_resources()
    
    print("🚀 Transformer Model Evaluation")
    print("=" * 50)
    print(f"Log type: {args.log_type}")
    print(f"Device: {config.device}")
    print(f"Node: {config.node_name} | Job: {config.job_id}")
    
    # Initialize evaluator
    evaluator = TransformerEvaluator(config)
    
    # Auto-detect best evaluation method
    predictions_file = Path(f"results/{args.log_type}/predictions.pkl")
    use_direct = predictions_file.exists()
    
    print(f"Method: {'Direct Supervised (predictions.pkl found)' if use_direct else 'Model-based Evaluation (fallback)'}")
    print("")
    
    try:
        if use_direct:
            # Use direct supervised evaluation from predictions.pkl
            print("🎯 Using Direct Supervised Evaluation")
            print("=" * 50)
            print("Found predictions.pkl - applying sklearn metrics directly")
            print("")
            
            results = evaluator.evaluate_direct_supervised(args.log_type)
            
            if results is None:
                print(f"❌ Direct supervised evaluation failed")
                print(f"Falling back to model-based evaluation...")
                use_direct = False
            else:
                # Save results
                output_dir = Path(args.output_dir) / args.log_type
                output_dir.mkdir(parents=True, exist_ok=True)
                
                results_path = output_dir / f"direct_supervised_results_{args.log_type}_{config.node_name}_{config.job_id}.pkl"
                with open(results_path, 'wb') as f:
                    pickle.dump(results, f)
                
                print(f"\n💾 Results saved to: {results_path}")
                print(f"✅ Direct supervised evaluation completed for {args.log_type}")
        
        if not use_direct:
            # Use model-based evaluation method
            print("🔄 Using Model-based Evaluation")
            print("=" * 50)
            print("No predictions.pkl found - generating predictions from trained model")
            print("")
            
            # Find model path
            if args.model_path:
                model_path = Path(args.model_path)
            else:
                model_path = find_model_path(args.log_type, config)
            
            if not model_path or not model_path.exists():
                print(f"❌ Model not found for {args.log_type}")
                print(f"   Expected location: models/transformer_{args.log_type}_*.pth")
                print(f"   Please train the model first using: python src/transformer.py --log-type {args.log_type}")
                return
            
            # Load trained model
            model, training_classes, model_input_dim = evaluator.load_trained_model(model_path, args.log_type)
            
            # Load TFRecord dataset and generate predictions
            embeddings, classes, true_labels = evaluator.load_tfrecord_dataset(args.log_type, model_input_dim)
            
            if embeddings is None:
                print(f"❌ Could not load TFRecord dataset for {args.log_type}")
                print(f"   Please ensure TFRecord files exist in processed/{args.log_type}/")
                return
            
            # Check class compatibility
            if set(training_classes) != set(classes):
                print(f"⚠️  Class mismatch between training and evaluation data")
                print(f"   Training classes: {len(training_classes)} - {training_classes[:3]}...")
                print(f"   Evaluation classes: {len(classes)} - {classes[:3]}...")
                print(f"   Using training classes for evaluation")
                
                # Store original classes for mapping
                original_classes = classes
                classes = training_classes
                
                # Adjust true_labels to match training classes
                if true_labels is not None:
                    new_true_labels = np.zeros((len(true_labels), len(training_classes)), dtype=np.float32)
                    for i, train_cls in enumerate(training_classes):
                        if train_cls in original_classes:
                            orig_idx = original_classes.index(train_cls)
                            new_true_labels[:, i] = true_labels[:, orig_idx]
                    true_labels = new_true_labels
            
            print(f"📊 Dataset loaded: {len(embeddings):,} samples")
            
            # Generate predictions using the trained model
            print(f"🤖 Generating predictions from trained model...")
            model.eval()
            device = torch.device(config.device)
            
            predictions = []
            probs = []
            batch_size = 64
            
            with torch.no_grad():
                for i in range(0, len(embeddings), batch_size):
                    batch = torch.from_numpy(embeddings[i:i+batch_size]).float().to(device)
                    logits = model(batch)['labels']
                    batch_probs = torch.sigmoid(logits).cpu().numpy()
                    batch_preds = (batch_probs >= 0.5).astype(int)
                    
                    probs.append(batch_probs)
                    predictions.append(batch_preds)
            
            probs = np.vstack(probs)
            predictions = np.vstack(predictions)
            
            print(f"✅ Generated predictions for {len(predictions):,} samples")
            
            # Calculate metrics using the same method as direct evaluation
            if true_labels is not None:
                # We have true labels - do supervised evaluation
                results = {
                    'metrics': evaluator._calculate_metrics(true_labels, predictions, probs, classes, 
                                                          evaluator._optimize_thresholds(true_labels, probs)),
                    'predictions': predictions,
                    'probabilities': probs,
                    'true_labels': true_labels,
                    'ids': np.arange(len(predictions))
                }
                
                # Print results similar to direct evaluation
                metrics = results['metrics']
                print(f"\n📊 MODEL-BASED EVALUATION RESULTS")
                print("=" * 60)
                print(f"Subset Accuracy:  {metrics['subset_accuracy']:.4f}")
                print(f"Hamming Loss:     {metrics['hamming_loss']:.4f}")
                print(f"Micro F1:         {metrics['micro_f1']:.4f}")
                print(f"Macro F1:         {metrics['macro_f1']:.4f}")
                print(f"Weighted F1:      {metrics['weighted_f1']:.4f}")
                print(f"Samples F1:       {metrics['samples_f1']:.4f}")
                
            else:
                # No true labels - do unsupervised evaluation
                results = evaluator.evaluate_transformer_embeddings(embeddings, true_labels, classes)
            
            # Generate outputs
            output_dir = Path(args.output_dir) / args.log_type
            evaluator.generate_classification_report(results, args.log_type, output_dir, args.training_time)
            evaluator.save_results(results, args.log_type, output_dir)
            evaluator.print_summary(results, args.log_type, args.training_time)
            
            print(f"\n✅ Model-based evaluation completed for {args.log_type}")
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Evaluation interrupted")
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 