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
    
    saved_data = torch.load(model_path, map_location=device)
    
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
    
    # Calculate metrics
    print("Calculating metrics...")
    
    # Per-class metrics
    prec_c, rec_c, f1_c, support_c = precision_recall_fscore_support(
        labels_test, binary_predictions, average=None, zero_division=0
    )
    
    # Overall metrics
    macro_f1 = f1_score(labels_test, binary_predictions, average='macro', zero_division=0)
    micro_f1 = f1_score(labels_test, binary_predictions, average='micro', zero_division=0)
    weighted_f1 = f1_score(labels_test, binary_predictions, average='weighted', zero_division=0)
    
    subset_accuracy = accuracy_score(labels_test, binary_predictions)
    hamming = hamming_loss(labels_test, binary_predictions)
    jaccard_macro = jaccard_score(labels_test, binary_predictions, average='macro', zero_division=0)
    jaccard_micro = jaccard_score(labels_test, binary_predictions, average='micro', zero_division=0)
    
    results = {
        'per_class_precision': prec_c.tolist(),
        'per_class_recall': rec_c.tolist(),
        'per_class_f1': f1_c.tolist(),
        'per_class_support': support_c.tolist(),
        'macro_f1': float(macro_f1),
        'micro_f1': float(micro_f1),
        'weighted_f1': float(weighted_f1),
        'subset_accuracy': float(subset_accuracy),
        'hamming_loss': float(hamming),
        'jaccard_macro': float(jaccard_macro),
        'jaccard_micro': float(jaccard_micro),
        'optimal_thresholds': optimal_thresholds.tolist(),
        'classes': classes,
        'n_test_samples': len(labels_test),
        'n_val_samples': len(labels_val)
    }
    
    return results, predictions_test, binary_predictions, labels_test

def print_evaluation_results(results: Dict[str, Any]):
    """Print detailed evaluation results"""
    
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    
    print(f"Test samples: {results['n_test_samples']}")
    print(f"Validation samples: {results['n_val_samples']}")
    print(f"Classes: {len(results['classes'])}")
    
    print(f"\nOVERALL METRICS:")
    print(f"  Macro F1:       {results['macro_f1']:.4f}")
    print(f"  Micro F1:       {results['micro_f1']:.4f}")
    print(f"  Weighted F1:    {results['weighted_f1']:.4f}")
    print(f"  Subset Accuracy: {results['subset_accuracy']:.4f}")
    print(f"  Hamming Loss:   {results['hamming_loss']:.4f}")
    print(f"  Jaccard (macro): {results['jaccard_macro']:.4f}")
    print(f"  Jaccard (micro): {results['jaccard_micro']:.4f}")
    
    print(f"\nPER-CLASS METRICS (Top 10 by F1):")
    
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
    parser.add_argument("--output-dir", type=str, default="evaluation_results", help="Output directory (default: evaluation_results)")
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
        
        # Save results
        output_dir = Path(args.output_dir)
        output_path = output_dir / f"evaluation_{args.log_type}.pkl"
        save_evaluation_results(results, predictions, binary_predictions, true_labels, output_path)
        
        print(f"\n✅ Evaluation completed successfully!")
        print(f"Key metrics:")
        print(f"  - Macro F1: {results['macro_f1']:.4f}")
        print(f"  - Micro F1: {results['micro_f1']:.4f}")
        print(f"  - Subset Accuracy: {results['subset_accuracy']:.4f}")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 