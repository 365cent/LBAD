#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation Script for Transformer-based Generative Model

This script evaluates a trained generative transformer model using labeled test data.
It provides real metrics like F1-score, precision, recall, etc. when true labels are available.

Usage:
    python evaluate_model.py --log-type wp-error --model-path models/transformer_wp-error.pth
    python evaluate_model.py --log-type vpn --test-split 0.2
    python evaluate_model.py --log-type dns --test-split 0.3
"""

import os
import sys
import pickle
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    classification_report, precision_recall_fscore_support,
    f1_score, accuracy_score, hamming_loss, jaccard_score,
    balanced_accuracy_score, multilabel_confusion_matrix
)

# Add the src directory to path to import transformer
sys.path.append(str(Path(__file__).parent))
from transformer import (
    UnsupervisedMultiLabelTransformer, detect_system_resources,
    enhanced_adaptive_thresholding, optimize_per_class_thresholds
)

def load_trained_model(model_path: Path, device: torch.device) -> Tuple[UnsupervisedMultiLabelTransformer, Dict]:
    """Load a trained transformer model"""
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Fix for PyTorch 2.6+ weights_only=True default - set to False for compatibility
    saved_data = torch.load(model_path, map_location=device, weights_only=False)
    
    # Extract configuration from saved data
    config_dict = saved_data.get('config', {})
    classes = saved_data.get('classes', [])
    
    # Determine model parameters
    n_labels = len(classes) if classes else 1
    
    # Get input dimension from the saved model state
    model_state = saved_data['model_state_dict']
    input_dim = model_state['input_proj.0.weight'].shape[1]
    latent_dim = model_state['input_proj.0.weight'].shape[0]
    
    # Estimate n_clusters from saved state
    if 'cluster_head.0.weight' in model_state:
        n_clusters = model_state['cluster_head.0.weight'].shape[0]
    else:
        n_clusters = 1
    
    # Create model
    model = UnsupervisedMultiLabelTransformer(
        input_dim=input_dim,
        latent_dim=latent_dim,
        n_labels=n_labels,
        n_clusters=n_clusters
    )
    
    # Load state dict
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()
    
    print(f"✅ Loaded model: {input_dim}D → {latent_dim}D, {n_labels} classes")
    
    return model, saved_data

def load_test_data(log_type: str, test_split: float = 0.2, seed: int = 42) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load and split test data"""
    embeddings_dir = Path("embeddings")
    log_file = embeddings_dir / log_type / f"log_{log_type}.pkl"
    label_file = embeddings_dir / log_type / f"label_{log_type}.pkl"
    
    if not log_file.exists():
        raise FileNotFoundError(f"Embedding file not found: {log_file}")
    
    # Load embeddings
    with open(log_file, 'rb') as f:
        embeddings = pickle.load(f)
    
    # Load labels
    true_labels = None
    classes = []
    
    if label_file.exists():
        with open(label_file, 'rb') as f:
            label_data = pickle.load(f)
            if isinstance(label_data, dict):
                if 'classes' in label_data:
                    classes = label_data['classes']
                if 'vectors' in label_data:
                    true_labels = label_data['vectors']
    
    if true_labels is None:
        raise ValueError(f"No true labels found for {log_type}. Cannot evaluate without labels.")
    
    # Split into train/test
    rng = np.random.default_rng(seed)
    n = len(embeddings)
    indices = np.arange(n)
    rng.shuffle(indices)
    
    n_test = int(n * test_split)
    test_indices = indices[:n_test]
    
    embeddings_test = embeddings[test_indices]
    labels_test = true_labels[test_indices]
    
    print(f"✅ Loaded test data: {len(embeddings_test)} samples, {len(classes)} classes")
    
    return embeddings_test, labels_test, classes

def evaluate_model_with_labels(model: UnsupervisedMultiLabelTransformer, 
                             embeddings: np.ndarray, 
                             true_labels: np.ndarray,
                             classes: List[str],
                             device: torch.device,
                             val_split: float = 0.3) -> Dict[str, Any]:
    """Evaluate model with true labels"""
    
    # Split test data into val/test for threshold optimization
    n_test = len(embeddings)
    n_val = int(n_test * val_split)
    
    embeddings_val = embeddings[:n_val]
    embeddings_test = embeddings[n_val:]
    labels_val = true_labels[:n_val]
    labels_test = true_labels[n_val:]
    
    print(f"Split: {len(embeddings_val)} validation, {len(embeddings_test)} test samples")
    
    # Generate predictions for validation set (for threshold optimization)
    print("Generating validation predictions...")
    predictions_val = []
    batch_size = 64
    
    with torch.no_grad():
        for i in range(0, len(embeddings_val), batch_size):
            batch = torch.from_numpy(embeddings_val[i:i+batch_size]).float().to(device)
            outputs = model(batch)
            probs = torch.sigmoid(outputs['labels'])
            predictions_val.append(probs.cpu().numpy())
    
    predictions_val = np.vstack(predictions_val)
    
    # Optimize thresholds on validation set
    print("Optimizing thresholds...")
    optimal_thresholds = optimize_per_class_thresholds(
        labels_val, predictions_val, metric='f1', beta=1.0
    )
    
    # Generate predictions for test set
    print("Generating test predictions...")
    predictions_test = []
    
    with torch.no_grad():
        for i in range(0, len(embeddings_test), batch_size):
            batch = torch.from_numpy(embeddings_test[i:i+batch_size]).float().to(device)
            outputs = model(batch)
            probs = torch.sigmoid(outputs['labels'])
            predictions_test.append(probs.cpu().numpy())
    
    predictions_test = np.vstack(predictions_test)
    
    # Apply optimized thresholds
    binary_predictions = (predictions_test >= optimal_thresholds).astype(int)
    
    # Calculate comprehensive metrics
    print("Calculating metrics...")
    
    results = calculate_comprehensive_metrics(
        binary_predictions, classes, y_true=labels_test, probs=predictions_test
    )
    
    # Add evaluation-specific info
    results['optimal_thresholds'] = optimal_thresholds.tolist()
    results['n_test_samples'] = len(labels_test)
    results['n_val_samples'] = len(labels_val)
    
    return results, predictions_test, binary_predictions, labels_test

def calculate_comprehensive_metrics(binary_predictions: np.ndarray, classes: List[str], 
                                  y_true: np.ndarray = None, probs: np.ndarray = None) -> Dict[str, Any]:
    """
    Calculate comprehensive metrics for multi-label classification
    
    Args:
        binary_predictions: Binary predictions (n_samples, n_classes)
        classes: List of class names
        y_true: True labels if available (n_samples, n_classes)
        probs: Prediction probabilities if available (n_samples, n_classes)
    
    Returns:
        Dictionary of metrics including real sklearn metrics when true labels are available
    """
    from sklearn.metrics import (
        precision_recall_fscore_support, f1_score, accuracy_score, 
        hamming_loss, jaccard_score, balanced_accuracy_score
    )
    
    metrics = {}
    
    # Sample-level metrics
    labels_per_sample = binary_predictions.sum(axis=1)
    metrics['avg_labels_per_sample'] = float(labels_per_sample.mean())
    metrics['std_labels_per_sample'] = float(labels_per_sample.std())
    metrics['min_labels_per_sample'] = int(labels_per_sample.min())
    metrics['max_labels_per_sample'] = int(labels_per_sample.max())
    
    # Class-level metrics
    class_counts = binary_predictions.sum(axis=0)
    metrics['class_counts'] = class_counts.tolist()
    metrics['most_frequent_classes'] = []
    
    # Get top 10 most frequent classes
    if len(classes) > 0:
        class_freq_pairs = list(zip(classes, class_counts))
        class_freq_pairs.sort(key=lambda x: x[1], reverse=True)
        metrics['most_frequent_classes'] = [
            {'class': cls, 'count': int(count), 'percentage': float(count/len(binary_predictions)*100)}
            for cls, count in class_freq_pairs[:10]
        ]
    
    # Multi-label specific metrics
    metrics['samples_with_no_labels'] = int((labels_per_sample == 0).sum())
    metrics['samples_with_one_label'] = int((labels_per_sample == 1).sum())
    metrics['samples_with_multiple_labels'] = int((labels_per_sample > 1).sum())
    
    # Real metrics when true labels are available
    if y_true is not None:
        # Per-class metrics
        prec_c, rec_c, f1_c, support_c = precision_recall_fscore_support(
            y_true, binary_predictions, average=None, zero_division=0
        )
        
        # Store per-class metrics
        metrics['per_class'] = {}
        for i, cls in enumerate(classes):
            metrics['per_class'][cls] = {
                'support': int(support_c[i]),
                'precision': float(prec_c[i]),
                'recall': float(rec_c[i]),
                'f1-score': float(f1_c[i])
            }
        
        # Store per-class arrays for evaluation results
        metrics['per_class_precision'] = prec_c.tolist()
        metrics['per_class_recall'] = rec_c.tolist()
        metrics['per_class_f1'] = f1_c.tolist()
        metrics['per_class_support'] = support_c.tolist()
        
        # Overall metrics
        metrics['macro_f1'] = float(f1_score(y_true, binary_predictions, average='macro', zero_division=0))
        metrics['micro_f1'] = float(f1_score(y_true, binary_predictions, average='micro', zero_division=0))
        metrics['weighted_f1'] = float(f1_score(y_true, binary_predictions, average='weighted', zero_division=0))
        metrics['samples_f1'] = float(f1_score(y_true, binary_predictions, average='samples', zero_division=0))
        
        # Subset accuracy (exact match)
        metrics['subset_accuracy'] = float(accuracy_score(y_true, binary_predictions))
        
        # Hamming loss
        metrics['hamming_loss'] = float(hamming_loss(y_true, binary_predictions))
        
        # Jaccard score (similarity)
        metrics['jaccard_macro'] = float(jaccard_score(y_true, binary_predictions, average='macro', zero_division=0))
        metrics['jaccard_micro'] = float(jaccard_score(y_true, binary_predictions, average='micro', zero_division=0))
        
        # Per-class balanced accuracy
        balanced_acc_per_class = []
        for c in range(y_true.shape[1]):
            y_c = y_true[:, c]
            yp_c = binary_predictions[:, c]
            
            # Skip if no samples for this class
            if y_c.sum() == 0 and (1 - y_c).sum() == 0:
                balanced_acc_per_class.append(0.0)
                continue
                
            # Calculate balanced accuracy for this class
            ba = balanced_accuracy_score(y_c, yp_c)
            balanced_acc_per_class.append(float(ba))
        
        metrics['balanced_accuracy_per_class'] = balanced_acc_per_class
        metrics['mean_balanced_accuracy'] = float(np.mean(balanced_acc_per_class))
        
        # Confusion matrix counts per class
        confusion_per_class = {}
        for i, cls in enumerate(classes):
            y_c = y_true[:, i]
            yp_c = binary_predictions[:, i]
            
            tp = int(np.sum((yp_c == 1) & (y_c == 1)))
            fp = int(np.sum((yp_c == 1) & (y_c == 0)))
            fn = int(np.sum((yp_c == 0) & (y_c == 1)))
            tn = int(np.sum((yp_c == 0) & (y_c == 0)))
            
            confusion_per_class[cls] = {
                'true_positives': tp,
                'false_positives': fp,
                'false_negatives': fn,
                'true_negatives': tn
            }
        
        metrics['confusion_per_class'] = confusion_per_class
        
        # Prediction confidence metrics (if probabilities available)
        if probs is not None:
            metrics['prediction_confidence_mean'] = float(probs.mean())
            metrics['prediction_confidence_std'] = float(probs.std())
            
            # Confidence of correct predictions
            correct_mask = (binary_predictions == y_true)
            correct_confidences = probs[correct_mask]
            incorrect_confidences = probs[~correct_mask]
            
            metrics['correct_prediction_confidence_mean'] = float(correct_confidences.mean()) if len(correct_confidences) > 0 else 0.0
            metrics['incorrect_prediction_confidence_mean'] = float(incorrect_confidences.mean()) if len(incorrect_confidences) > 0 else 0.0
    else:
        # No true labels available - unsupervised metrics only
        # Set real metrics to None (these will be filtered out in printing)
        metrics['per_class_precision'] = None
        metrics['per_class_recall'] = None
        metrics['per_class_f1'] = None
        metrics['per_class_support'] = None
        metrics['macro_f1'] = None
        metrics['micro_f1'] = None
        metrics['weighted_f1'] = None
        metrics['samples_f1'] = None
        metrics['subset_accuracy'] = None
        metrics['hamming_loss'] = None
        metrics['jaccard_macro'] = None
        metrics['jaccard_micro'] = None
        metrics['mean_balanced_accuracy'] = None
        
        # Prediction confidence metrics (unsupervised)
        if probs is not None:
            metrics['prediction_confidence_mean'] = float(probs.mean())
            metrics['prediction_confidence_std'] = float(probs.std())
            
            # High confidence predictions
            high_conf_mask = probs > 0.7
            metrics['high_confidence_predictions'] = int(high_conf_mask.sum())
            metrics['high_confidence_percentage'] = float(high_conf_mask.sum() / probs.size * 100)
            
            # Per-class confidence statistics
            class_confidences = []
            for i, cls in enumerate(classes):
                if i < probs.shape[1]:
                    class_conf = float(probs[:, i].mean())
                    class_confidences.append({
                        'class': cls,
                        'avg_confidence': class_conf,
                        'predictions': int(class_counts[i])
                    })
            
            # Sort by confidence and add to metrics
            class_confidences.sort(key=lambda x: x['avg_confidence'], reverse=True)
            metrics['class_confidence_ranking'] = class_confidences[:10]
        else:
            metrics['prediction_confidence_mean'] = 0.0
            metrics['prediction_confidence_std'] = 0.0
            metrics['high_confidence_predictions'] = 0
            metrics['high_confidence_percentage'] = 0.0
    
    return metrics

def print_evaluation_results(results: Dict[str, Any]):
    """Print detailed evaluation results"""
    
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    
    print(f"Test samples: {results['n_test_samples']}")
    print(f"Validation samples: {results['n_val_samples']}")
    print(f"Classes: {len(results['classes'])}")
    
    # Check if we have real metrics (non-null values)
    has_real_metrics = results.get('macro_f1') is not None
    
    if has_real_metrics:
        print(f"\nOVERALL METRICS (with true labels):")
        print(f"  Macro F1:       {results['macro_f1']:.4f}")
        print(f"  Micro F1:       {results['micro_f1']:.4f}")
        print(f"  Weighted F1:    {results['weighted_f1']:.4f}")
        print(f"  Subset Accuracy: {results['subset_accuracy']:.4f}")
        print(f"  Hamming Loss:   {results['hamming_loss']:.4f}")
        print(f"  Jaccard (macro): {results['jaccard_macro']:.4f}")
        print(f"  Jaccard (micro): {results['jaccard_micro']:.4f}")
    else:
        print(f"\nGENERATIVE MODEL METRICS (unsupervised):")
        print(f"  Prediction Confidence: {results['prediction_confidence_mean']:.4f}")
        print(f"  Confidence Std:        {results['prediction_confidence_std']:.4f}")
        print(f"  High Conf Predictions: {results['high_confidence_predictions']}")
        print(f"  High Conf Percentage:  {results['high_confidence_percentage']:.2f}%")
        print(f"  Note: Real F1/precision/recall metrics require true labels")
    
    print(f"\nPER-CLASS METRICS (Top 10 by F1):")
    
    if has_real_metrics:
        # Sort classes by F1 score
        class_f1_pairs = list(zip(results['classes'], results['per_class_f1'], 
                                 results['per_class_precision'], results['per_class_recall'],
                                 results['per_class_support']))
        class_f1_pairs.sort(key=lambda x: x[1], reverse=True)
        
        print(f"{'Class':<30} {'F1':<8} {'Prec':<8} {'Recall':<8} {'Support'}")
        print("-" * 70)
        
        for i, (cls, f1, prec, rec, support) in enumerate(class_f1_pairs[:10]):
            print(f"{cls:<30} {f1:<8.4f} {prec:<8.4f} {rec:<8.4f} {support}")
        
        if len(class_f1_pairs) > 10:
            print(f"... and {len(class_f1_pairs) - 10} more classes")
    else:
        # Show prediction frequency for unsupervised case
        print(f"{'Class':<30} {'Predictions':<12} {'Percentage'}")
        print("-" * 55)
        
        total_samples = results['n_test_samples']
        for cls_info in results.get('most_frequent_classes', [])[:10]:
            cls = cls_info['class']
            count = cls_info['count'] 
            percentage = cls_info['percentage']
            print(f"{cls:<30} {count:<12} {percentage:.2f}%")
    
    print("="*80)

def save_evaluation_results(results: Dict[str, Any], 
                          predictions: np.ndarray,
                          binary_predictions: np.ndarray, 
                          true_labels: np.ndarray,
                          output_path: Path):
    """Save detailed evaluation results"""
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save comprehensive results
    save_data = {
        'results': results,
        'predictions_proba': predictions.astype(np.float32),
        'predictions_binary': binary_predictions.astype(np.int8),
        'true_labels': true_labels.astype(np.int8),
        'evaluation_timestamp': str(Path().cwd()),
        'evaluation_type': 'labeled_test_evaluation'
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(save_data, f)
    
    # Generate sklearn classification report
    report_path = output_path.with_suffix('.txt')
    with open(report_path, 'w') as f:
        f.write("SKLEARN CLASSIFICATION REPORT\n")
        f.write("="*80 + "\n\n")
        
        report = classification_report(
            true_labels, binary_predictions,
            target_names=results['classes'],
            zero_division=0,
            digits=4
        )
        f.write(report)
        
        f.write("\n\nCONFUSION MATRICES (Top 10 classes):\n")
        f.write("-"*80 + "\n")
        
        mcm = multilabel_confusion_matrix(true_labels, binary_predictions)
        
        # Show top 10 classes by F1
        class_f1_pairs = list(zip(results['classes'], results['per_class_f1'], range(len(results['classes']))))
        class_f1_pairs.sort(key=lambda x: x[1], reverse=True)
        
        for cls, f1, idx in class_f1_pairs[:10]:
            tn, fp, fn, tp = mcm[idx].ravel()
            f.write(f"\n{cls} (F1: {f1:.4f}):\n")
            f.write(f"  TN: {tn:6d}  FP: {fp:6d}\n")
            f.write(f"  FN: {fn:6d}  TP: {tp:6d}\n")
    
    print(f"✅ Evaluation results saved to: {output_path}")
    print(f"✅ Classification report saved to: {report_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained transformer model with labeled data",
        epilog="""
Examples:
  python src/evaluate_model.py --log-type wp-error
  python src/evaluate_model.py --log-type vpn --test-split 0.3
  python src/evaluate_model.py --log-type dns --model-path models/transformer_dns.pth
  
Available log types (based on preprocessing.py):
  - vpn: VPN access logs (openvpn.log)
  - wp-access: Web server access logs 
  - wp-error: Web server error logs
  - intranet-error: Intranet server error logs
  - auth: Authentication logs (auth.log)
  - audit: System audit logs
  - dns: DNS server logs (dnsmasq.log)
  - share: Internal file sharing logs
  - monitor: System monitoring logs (CPU, etc.)
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--log-type", type=str, required=True, 
                       help="Log type to evaluate (e.g., wp-error, vpn, dns, auth, audit, etc.)")
    parser.add_argument("--model-path", type=str, help="Path to saved model (auto-detected if not provided)")
    parser.add_argument("--test-split", type=float, default=0.2, help="Fraction of data to use for testing (default: 0.2)")
    parser.add_argument("--val-split", type=float, default=0.3, help="Fraction of test data to use for validation (default: 0.3)")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory (default: results)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    
    args = parser.parse_args()
    
    # Detect system
    config = detect_system_resources()
    device = torch.device(config.device)
    
    print(f"Device: {device}")
    print(f"Evaluating log type: {args.log_type}")
    
    # Check if log type directory exists
    embeddings_dir = Path("embeddings")
    log_type_dir = embeddings_dir / args.log_type
    if not log_type_dir.exists():
        available_types = [d.name for d in embeddings_dir.iterdir() if d.is_dir()]
        print(f"❌ Log type '{args.log_type}' not found in {embeddings_dir}")
        print(f"   Available log types: {available_types}")
        print(f"   Make sure you've run preprocessing.py first to generate embeddings.")
        return 1
    
    # Auto-detect model path if not provided
    if args.model_path is None:
        models_dir = Path("models")
        pattern = f"transformer_{args.log_type}_*.pth"
        model_files = list(models_dir.glob(pattern))
        
        if not model_files:
            print(f"❌ No model files found for {args.log_type} in {models_dir}")
            print(f"   Expected pattern: {pattern}")
            print(f"   Make sure you've trained a model for this log type first.")
            return 1
        
        # Use the most recent model
        model_path = max(model_files, key=lambda x: x.stat().st_mtime)
        print(f"Auto-detected model: {model_path}")
    else:
        model_path = Path(args.model_path)
    
    try:
        # Load model
        print("Loading trained model...")
        model, saved_data = load_trained_model(model_path, device)
        
        # Load test data
        print("Loading test data...")
        embeddings_test, labels_test, classes = load_test_data(
            args.log_type, args.test_split, args.seed
        )
        
        # Evaluate
        print("Evaluating model...")
        results, predictions, binary_predictions, true_labels = evaluate_model_with_labels(
            model, embeddings_test, labels_test, classes, device, args.val_split
        )
        
        # Print results
        print_evaluation_results(results)
        
        # Save results (match training output structure)
        output_dir = Path(args.output_dir) / args.log_type  # Create log_type subdirectory
        output_path = output_dir / f"evaluation_{args.log_type}.pkl"
        save_evaluation_results(results, predictions, binary_predictions, true_labels, output_path)
        
        # Also save in training-compatible format
        training_format_path = output_dir / f"results_{args.log_type}_evaluation.pkl"
        with open(training_format_path, 'wb') as f:
            pickle.dump({
                'predictions': predictions.astype(np.float32),
                'binary_predictions': binary_predictions.astype(np.int8), 
                'classes': results['classes'],
                'metrics': results,
                'model_type': 'evaluation_results',
                'evaluation_mode': 'with_true_labels' if results['macro_f1'] is not None else 'unsupervised'
            }, f)
        
        # Save labels in evaluation format (match training label output)
        label_output_path = output_dir / f"label_{args.log_type}_evaluation.pkl"
        label_data = {
            'vectors': binary_predictions.astype(np.int8),
            'classes': results['classes'],
            'probabilities': predictions.astype(np.float32),
            'metadata': {
                'evaluation_type': 'real_evaluation_with_test_split',
                'test_split': args.test_split,
                'val_split': args.val_split,
                'optimal_thresholds': results['optimal_thresholds'],
                'n_test_samples': results['n_test_samples'],
                'n_val_samples': results['n_val_samples']
            }
        }
        
        with open(label_output_path, 'wb') as f:
            pickle.dump(label_data, f)
        
        print(f"\n✅ Evaluation completed successfully!")
        print(f"📁 Results saved to:")
        print(f"   • Detailed evaluation: {output_path}")
        print(f"   • Training-compatible: {training_format_path}")
        print(f"   • Label predictions: {label_output_path}")
        print(f"   • Classification report: {output_path.with_suffix('.txt')}")
        print(f"Key metrics:")
        if results['macro_f1'] is not None:
            print(f"  - Macro F1: {results['macro_f1']:.4f}")
            print(f"  - Micro F1: {results['micro_f1']:.4f}")
            print(f"  - Subset Accuracy: {results['subset_accuracy']:.4f}")
        else:
            print(f"  - Prediction Confidence: {results['prediction_confidence_mean']:.4f}")
            print(f"  - High Confidence Predictions: {results['high_confidence_percentage']:.2f}%")
            print(f"  - Avg Labels per Sample: {results['avg_labels_per_sample']:.2f}")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 