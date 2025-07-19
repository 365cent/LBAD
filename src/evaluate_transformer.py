#!/usr/bin/env python3
"""
Transformer Model Evaluation Pipeline
====================================

Evaluates trained transformer models on full TFRecord datasets.
Provides comprehensive classification performance metrics.

Usage:
    python src/evaluate_transformer.py --log-type wp-access
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
    
    def evaluate_model(self, model: UnsupervisedMultiLabelTransformer, 
                      embeddings: np.ndarray, true_labels: np.ndarray, 
                      classes: List[str]) -> Dict[str, Any]:
        """Evaluate model and return comprehensive metrics"""
        print(f"🔍 Evaluating model on {len(embeddings):,} samples...")
        
        model.eval()
        predictions = []
        batch_size = 64
        
        # Generate predictions
        with torch.no_grad():
            for i in range(0, len(embeddings), batch_size):
                batch = torch.from_numpy(embeddings[i:i+batch_size]).float().to(self.device)
                outputs = model(batch)
                probs = torch.sigmoid(outputs['labels'])
                predictions.append(probs.cpu().numpy())
        
        predictions = np.vstack(predictions)
        
        # Optimize thresholds and get binary predictions
        thresholds = self._optimize_thresholds(true_labels, predictions)
        binary_predictions = (predictions > thresholds).astype(int)
        
        # Calculate comprehensive metrics
        metrics = self._calculate_metrics(true_labels, binary_predictions, predictions, classes, thresholds)
        
        return {
            'metrics': metrics,
            'predictions': predictions,
            'binary_predictions': binary_predictions,
            'thresholds': thresholds
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
        
        report_lines.append(f"TRANSFORMER Multi-Label Classification Report - {log_type.upper()}")
        report_lines.append("=" * 80)
        report_lines.append(f"Training time: {training_time:.2f} seconds")
        report_lines.append(f"Test samples: {metrics['n_test_samples']:,}")
        report_lines.append(f"Number of classes: {len(metrics['classes'])}")
        report_lines.append(f"Evaluation type: {metrics['evaluation_type']}")
        report_lines.append(f"Node: {self.config.node_name} | Job: {self.config.job_id}")
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
            threshold = metrics['optimal_thresholds'][i]
            
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
    parser = argparse.ArgumentParser(description="Evaluate trained transformer model on full TFRecord dataset")
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
    print("")
    
    # Initialize evaluator
    evaluator = TransformerEvaluator(config)
    
    try:
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
        
        # Load full TFRecord dataset with correct embedding dimension
        embeddings, classes, true_labels = evaluator.load_tfrecord_dataset(args.log_type, target_dim=model_input_dim)
        
        if embeddings is None:
            print(f"❌ Could not load TFRecord dataset for {args.log_type}")
            print(f"   Make sure processed/{args.log_type}/*.tfrecord files exist")
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
        
        print(f"📊 Dataset loaded: {len(embeddings):,} samples vs {len(training_classes)} training classes")
        
        # Evaluate model
        results = evaluator.evaluate_model(model, embeddings, true_labels, classes)
        
        # Generate outputs
        output_dir = Path(args.output_dir) / args.log_type
        evaluator.generate_classification_report(results, args.log_type, output_dir, args.training_time)
        evaluator.save_results(results, args.log_type, output_dir)
        evaluator.print_summary(results, args.log_type, args.training_time)
        
        print(f"\n✅ Evaluation completed for {args.log_type}")
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Evaluation interrupted")
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 