#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Evaluation Script for Transformer Predictions

Compares transformer predictions with original labeled data from preprocessing.py
to calculate proper metrics: F1 score, precision, recall, accuracy, etc.
"""

import os
import sys
import json
import pickle
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    classification_report, 
    precision_recall_fscore_support,
    accuracy_score,
    hamming_loss,
    multilabel_confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TransformerEvaluator:
    """Evaluates transformer predictions against original labeled data"""
    
    def __init__(self, processed_dir: str = "processed", predictions_dir: str = "predictions"):
        self.processed_dir = Path(processed_dir)
        self.predictions_dir = Path(predictions_dir)
        
        # Log type mapping
        self.log_types = ['vpn', 'wp-access', 'wp-error', 'intranet-error', 'auth', 'audit', 'dns', 'share', 'monitor']
        
    def load_original_data(self, log_type: str) -> Tuple[List[str], List[List[str]]]:
        """Load original labeled data from TFRecord files"""
        logger.info(f"Loading original data for log type: {log_type}")
        
        type_dir = self.processed_dir / log_type
        if not type_dir.exists():
            logger.warning(f"No processed data found for {log_type}")
            return [], []
        
        all_logs = []
        all_labels = []
        
        # Load all TFRecord files for this log type
        tfrecord_files = list(type_dir.glob("*.tfrecord"))
        
        for tfrecord_file in tfrecord_files:
            logger.info(f"Loading {tfrecord_file}")
            
            try:
                dataset = tf.data.TFRecordDataset(
                    str(tfrecord_file), 
                    compression_type="GZIP"
                )
                
                for serialized_example in dataset:
                    example = tf.train.Example()
                    example.ParseFromString(serialized_example.numpy())
                    
                    # Extract features
                    log_line = example.features.feature['l'].bytes_list.value[0].decode('utf-8')
                    labels_json = example.features.feature['y'].bytes_list.value[0].decode('utf-8')
                    log_type_from_file = example.features.feature['log_type'].bytes_list.value[0].decode('utf-8')
                    
                    # Parse labels
                    try:
                        labels = json.loads(labels_json) if labels_json else []
                    except json.JSONDecodeError:
                        labels = []
                    
                    all_logs.append(log_line)
                    all_labels.append(labels)
                    
            except Exception as e:
                logger.error(f"Error loading {tfrecord_file}: {e}")
                continue
        
        logger.info(f"Loaded {len(all_logs)} log entries with {len(all_labels)} label sets for {log_type}")
        return all_logs, all_labels
    
    def load_predictions(self, log_type: str) -> Tuple[np.ndarray, List[str]]:
        """Load transformer predictions for a log type"""
        logger.info(f"Loading predictions for log type: {log_type}")
        
        # Find the most recent prediction file for this log type
        prediction_files = list(self.predictions_dir.glob(f"label_predictions_{log_type}_*.pkl"))
        
        if not prediction_files:
            logger.warning(f"No prediction files found for {log_type}")
            return np.array([]), []
        
        # Use the most recent file
        latest_file = max(prediction_files, key=lambda x: x.stat().st_mtime)
        logger.info(f"Using prediction file: {latest_file}")
        
        with open(latest_file, 'rb') as f:
            prediction_data = pickle.load(f)
        
        predictions = prediction_data['vectors']
        classes = prediction_data['classes']
        
        logger.info(f"Loaded predictions: {predictions.shape}, classes: {len(classes)}")
        return predictions, classes
    
    def create_label_mapping(self, original_labels: List[List[str]], prediction_classes: List[str]) -> np.ndarray:
        """Convert original text labels to binary format matching prediction classes"""
        logger.info(f"Creating label mapping for {len(original_labels)} samples and {len(prediction_classes)} classes")
        
        # Create mapping from text labels to indices
        class_to_idx = {cls: idx for idx, cls in enumerate(prediction_classes)}
        
        # Convert original labels to binary format
        binary_labels = np.zeros((len(original_labels), len(prediction_classes)), dtype=int)
        
        for i, labels in enumerate(original_labels):
            for label in labels:
                if label in class_to_idx:
                    binary_labels[i, class_to_idx[label]] = 1
        
        logger.info(f"Created binary labels: {binary_labels.shape}")
        logger.info(f"Label distribution: {binary_labels.sum(axis=0)}")
        
        return binary_labels
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, classes: List[str]) -> Dict[str, Any]:
        """Calculate comprehensive metrics for multi-label classification"""
        logger.info(f"Calculating metrics for {y_true.shape[0]} samples and {len(classes)} classes")
        
        metrics = {}
        
        # Overall metrics
        metrics['hamming_loss'] = hamming_loss(y_true, y_pred)
        metrics['exact_match_accuracy'] = accuracy_score(y_true, y_pred)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        # Macro averages
        metrics['macro_precision'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['macro_recall'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['macro_f1'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        # Micro averages
        metrics['micro_precision'] = precision_score(y_true, y_pred, average='micro', zero_division=0)
        metrics['micro_recall'] = recall_score(y_true, y_pred, average='micro', zero_division=0)
        metrics['micro_f1'] = f1_score(y_true, y_pred, average='micro', zero_division=0)
        
        # Weighted averages
        metrics['weighted_precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['weighted_recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['weighted_f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Per-class detailed metrics
        per_class_metrics = {}
        for i, class_name in enumerate(classes):
            per_class_metrics[class_name] = {
                'precision': precision[i],
                'recall': recall[i],
                'f1': f1[i],
                'support': support[i]
            }
        
        metrics['per_class'] = per_class_metrics
        
        # Sample-level metrics
        metrics['avg_labels_per_sample_true'] = float(y_true.sum(axis=1).mean())
        metrics['avg_labels_per_sample_pred'] = float(y_pred.sum(axis=1).mean())
        metrics['std_labels_per_sample_true'] = float(y_true.sum(axis=1).std())
        metrics['std_labels_per_sample_pred'] = float(y_pred.sum(axis=1).std())
        
        return metrics
    
    def evaluate_log_type(self, log_type: str) -> Optional[Dict[str, Any]]:
        """Evaluate predictions for a specific log type"""
        logger.info(f"Evaluating log type: {log_type}")
        
        try:
            # Load original data
            original_logs, original_labels = self.load_original_data(log_type)
            if not original_logs:
                logger.warning(f"No original data found for {log_type}")
                return None
            
            # Load predictions
            predictions, classes = self.load_predictions(log_type)
            if len(predictions) == 0:
                logger.warning(f"No predictions found for {log_type}")
                return None
            
            # Ensure we have the same number of samples
            min_samples = min(len(original_labels), len(predictions))
            logger.info(f"Using {min_samples} samples for evaluation")
            
            # Truncate to same length
            original_labels = original_labels[:min_samples]
            predictions = predictions[:min_samples]
            
            # Create binary labels from original text labels
            binary_labels = self.create_label_mapping(original_labels, classes)
            
            # Calculate metrics
            metrics = self.calculate_metrics(binary_labels, predictions, classes)
            
            # Add metadata
            metrics['log_type'] = log_type
            metrics['n_samples'] = min_samples
            metrics['n_classes'] = len(classes)
            metrics['classes'] = classes
            
            logger.info(f"Evaluation complete for {log_type}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating {log_type}: {e}")
            return None
    
    def evaluate_all(self) -> Dict[str, Any]:
        """Evaluate all available log types"""
        logger.info("Starting comprehensive evaluation")
        
        results = {}
        overall_metrics = {
            'total_samples': 0,
            'total_classes': 0,
            'macro_f1_scores': [],
            'micro_f1_scores': [],
            'hamming_losses': []
        }
        
        # Evaluate each log type
        for log_type in self.log_types:
            logger.info(f"\n{'='*50}")
            logger.info(f"Evaluating: {log_type}")
            logger.info(f"{'='*50}")
            
            metrics = self.evaluate_log_type(log_type)
            if metrics:
                results[log_type] = metrics
                
                # Aggregate overall metrics
                overall_metrics['total_samples'] += metrics['n_samples']
                overall_metrics['total_classes'] += metrics['n_classes']
                overall_metrics['macro_f1_scores'].append(metrics['macro_f1'])
                overall_metrics['micro_f1_scores'].append(metrics['micro_f1'])
                overall_metrics['hamming_losses'].append(metrics['hamming_loss'])
                
                # Print summary
                self.print_log_type_summary(log_type, metrics)
        
        # Calculate overall averages
        if overall_metrics['macro_f1_scores']:
            overall_metrics['avg_macro_f1'] = np.mean(overall_metrics['macro_f1_scores'])
            overall_metrics['avg_micro_f1'] = np.mean(overall_metrics['micro_f1_scores'])
            overall_metrics['avg_hamming_loss'] = np.mean(overall_metrics['hamming_losses'])
        
        results['overall'] = overall_metrics
        
        return results
    
    def print_log_type_summary(self, log_type: str, metrics: Dict[str, Any]):
        """Print summary for a log type"""
        print(f"\n{log_type.upper()} EVALUATION SUMMARY:")
        print(f"  Samples: {metrics['n_samples']}")
        print(f"  Classes: {metrics['n_classes']}")
        print(f"  Macro F1: {metrics['macro_f1']:.4f}")
        print(f"  Micro F1: {metrics['micro_f1']:.4f}")
        print(f"  Hamming Loss: {metrics['hamming_loss']:.4f}")
        print(f"  Exact Match Accuracy: {metrics['exact_match_accuracy']:.4f}")
        print(f"  Avg Labels/Sample (True): {metrics['avg_labels_per_sample_true']:.2f}")
        print(f"  Avg Labels/Sample (Pred): {metrics['avg_labels_per_sample_pred']:.2f}")
        
        # Print per-class F1 scores
        print(f"  Per-class F1 scores:")
        for class_name, class_metrics in metrics['per_class'].items():
            print(f"    {class_name}: {class_metrics['f1']:.4f} (support: {class_metrics['support']})")
    
    def save_results(self, results: Dict[str, Any], output_file: str = None):
        """Save evaluation results"""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"evaluation_results_{timestamp}.json"
        
        output_path = self.predictions_dir / output_file
        
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for key, value in results.items():
            if key == 'overall':
                json_results[key] = value
            else:
                json_results[key] = {}
                for metric_key, metric_value in value.items():
                    if isinstance(metric_value, np.ndarray):
                        json_results[key][metric_key] = metric_value.tolist()
                    elif isinstance(metric_value, np.integer):
                        json_results[key][metric_key] = int(metric_value)
                    elif isinstance(metric_value, np.floating):
                        json_results[key][metric_key] = float(metric_value)
                    else:
                        json_results[key][metric_key] = metric_value
        
        with open(output_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        logger.info(f"Results saved to: {output_path}")
        return output_path
    
    def print_overall_summary(self, results: Dict[str, Any]):
        """Print overall evaluation summary"""
        overall = results.get('overall', {})
        
        print(f"\n{'='*80}")
        print("OVERALL EVALUATION SUMMARY")
        print(f"{'='*80}")
        print(f"Total Samples: {overall.get('total_samples', 0)}")
        print(f"Total Classes: {overall.get('total_classes', 0)}")
        print(f"Average Macro F1: {overall.get('avg_macro_f1', 0):.4f}")
        print(f"Average Micro F1: {overall.get('avg_micro_f1', 0):.4f}")
        print(f"Average Hamming Loss: {overall.get('avg_hamming_loss', 0):.4f}")
        
        print(f"\nPer-Log-Type F1 Scores:")
        for log_type, metrics in results.items():
            if log_type != 'overall':
                print(f"  {log_type}: Macro F1 = {metrics['macro_f1']:.4f}, Micro F1 = {metrics['micro_f1']:.4f}")

def main():
    """Main evaluation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate transformer predictions against original data")
    parser.add_argument("--log-type", "-t", 
                       help="Evaluate specific log type (e.g., 'wp-error', 'wp-access')")
    parser.add_argument("--processed-dir", "-p", default="processed",
                       help="Directory containing processed TFRecord files")
    parser.add_argument("--predictions-dir", "-d", default="predictions",
                       help="Directory containing prediction files")
    parser.add_argument("--output", "-o", 
                       help="Output file for results")
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = TransformerEvaluator(args.processed_dir, args.predictions_dir)
    
    try:
        if args.log_type:
            # Evaluate specific log type
            logger.info(f"Evaluating specific log type: {args.log_type}")
            metrics = evaluator.evaluate_log_type(args.log_type)
            if metrics:
                evaluator.print_log_type_summary(args.log_type, metrics)
                results = {args.log_type: metrics}
                evaluator.save_results(results, args.output)
            else:
                logger.error(f"Failed to evaluate {args.log_type}")
        else:
            # Evaluate all log types
            logger.info("Evaluating all log types")
            results = evaluator.evaluate_all()
            evaluator.print_overall_summary(results)
            evaluator.save_results(results, args.output)
        
        logger.info("Evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 