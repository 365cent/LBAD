#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automatic Model Evaluation Script

This script automatically detects all trained transformer models and evaluates them
without requiring manual log-type specification. It handles both supervised and
unsupervised evaluation scenarios.

Usage:
    python evaluate_model.py                    # Evaluate all models
    python evaluate_model.py --model-path xxx   # Evaluate specific model
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
    optimize_per_class_thresholds
)

def discover_models() -> List[Dict[str, Any]]:
    """Discover all available trained models"""
    models_dir = Path("models")
    if not models_dir.exists():
        return []
    
    models = []
    for model_file in models_dir.glob("transformer_*.pth"):
        try:
            # Extract log type from filename
            parts = model_file.stem.split('_')
            if len(parts) >= 2:
                log_type = parts[1]
                
                # Check if corresponding embeddings exist
                embeddings_dir = Path("embeddings") / log_type
                if embeddings_dir.exists():
                    log_file = embeddings_dir / f"log_{log_type}.pkl"
                    label_file = embeddings_dir / f"label_{log_type}.pkl"
                    
                    if log_file.exists():
                        models.append({
                            'model_path': model_file,
                            'log_type': log_type,
                            'log_file': log_file,
                            'label_file': label_file if label_file.exists() else None,
                            'has_labels': label_file.exists()
                        })
        except Exception as e:
            print(f"⚠️  Error checking model {model_file}: {e}")
            continue
    
    return models

def load_model_smart(model_path: Path, device: torch.device) -> Tuple[UnsupervisedMultiLabelTransformer, Dict, List[str]]:
    """Smart model loading with automatic architecture detection"""
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load saved data
    saved_data = torch.load(model_path, map_location=device, weights_only=False)
    model_state = saved_data['model_state_dict']
    classes = saved_data.get('classes', [])
    
    # Extract architecture from saved state
    input_dim = model_state['input_proj.0.weight'].shape[1]
    latent_dim = model_state['input_proj.0.weight'].shape[0]
    n_labels = len(classes) if classes else 1
    
    # Smart cluster detection
    possible_clusters = [1, 4, 8, min(8, max(1, len(classes))), len(classes) if len(classes) > 0 else 1]
    possible_clusters = sorted(list(set([c for c in possible_clusters if c > 0])))
    
    print(f"🔧 Loading model: {input_dim}D → {latent_dim}D, {n_labels} labels")
    
    model = None
    n_clusters_used = None
    
    # Try different cluster values
    for n_clusters in possible_clusters:
        try:
            test_model = UnsupervisedMultiLabelTransformer(
                input_dim=input_dim,
                latent_dim=latent_dim,
                n_labels=n_labels,
                n_clusters=n_clusters
            )
            test_model.load_state_dict(model_state)
            model = test_model
            n_clusters_used = n_clusters
            break
        except RuntimeError:
            continue
    
    # Fallback with strict=False
    if model is None:
        n_clusters = possible_clusters[0]
        model = UnsupervisedMultiLabelTransformer(
            input_dim=input_dim,
            latent_dim=latent_dim,
            n_labels=n_labels,
            n_clusters=n_clusters
        )
        model.load_state_dict(model_state, strict=False)
        n_clusters_used = n_clusters
        print("⚠️  Loaded with strict=False")
    
    model.to(device).eval()
    print(f"✅ Model loaded: {n_labels} classes, {n_clusters_used} clusters")
    
    return model, saved_data, classes

def load_data_for_model(log_type: str, test_split: float = 0.2, seed: int = 42) -> Tuple[np.ndarray, Optional[np.ndarray], List[str]]:
    """Load embeddings and labels for a specific log type"""
    embeddings_dir = Path("embeddings") / log_type
    log_file = embeddings_dir / f"log_{log_type}.pkl"
    label_file = embeddings_dir / f"label_{log_type}.pkl"
    
    # Load embeddings
    with open(log_file, 'rb') as f:
        embeddings = pickle.load(f)
    
    # Load labels if available
    true_labels = None
    classes = []
    
    if label_file.exists():
        try:
            with open(label_file, 'rb') as f:
                label_data = pickle.load(f)
                if isinstance(label_data, dict):
                    classes = label_data.get('classes', [])
                    true_labels = label_data.get('vectors', None)
                else:
                    # Handle legacy format
                    true_labels = label_data if isinstance(label_data, np.ndarray) else None
        except Exception as e:
            print(f"⚠️  Error loading labels: {e}")
            true_labels = None
    
    # Split data for evaluation
    rng = np.random.default_rng(seed)
    n = len(embeddings)
    indices = np.arange(n)
    rng.shuffle(indices)
    
    n_test = int(n * test_split)
    test_indices = indices[:n_test]
    
    embeddings_test = embeddings[test_indices]
    labels_test = true_labels[test_indices] if true_labels is not None else None
    
    return embeddings_test, labels_test, classes

def evaluate_model(model: UnsupervisedMultiLabelTransformer, 
                  embeddings: np.ndarray, 
                  true_labels: Optional[np.ndarray],
                  classes: List[str],
                  device: torch.device,
                  val_split: float = 0.3) -> Dict[str, Any]:
    """Evaluate model with or without true labels"""
    
    n_samples = len(embeddings)
    n_val = int(n_samples * val_split) if true_labels is not None else 0
    
    # Generate predictions
    predictions = []
    batch_size = 64
    
    with torch.no_grad():
        for i in range(0, len(embeddings), batch_size):
            batch = torch.from_numpy(embeddings[i:i+batch_size]).float().to(device)
            outputs = model(batch)
            probs = torch.sigmoid(outputs['labels'])
            predictions.append(probs.cpu().numpy())
    
    predictions = np.vstack(predictions)
    
    # Optimize thresholds if we have true labels
    if true_labels is not None and n_val > 0:
        embeddings_val = embeddings[:n_val]
        embeddings_test = embeddings[n_val:]
        labels_val = true_labels[:n_val]
        labels_test = true_labels[n_val:]
        predictions_val = predictions[:n_val]
        predictions_test = predictions[n_val:]
        
        print("Optimizing thresholds on validation set...")
        optimal_thresholds = optimize_per_class_thresholds(
            labels_val, predictions_val, metric='f1', beta=1.0
        )
        binary_predictions = (predictions_test >= optimal_thresholds).astype(int)
        
        # Calculate real metrics
        results = {
            'per_class_precision': [],
            'per_class_recall': [],
            'per_class_f1': [],
            'per_class_support': [],
            'classes': classes,
            'n_test_samples': len(labels_test),
            'n_val_samples': len(labels_val),
            'optimal_thresholds': optimal_thresholds.tolist(),
            'evaluation_type': 'supervised'
        }
        
        try:
            prec_c, rec_c, f1_c, support_c = precision_recall_fscore_support(
                labels_test, binary_predictions, average=None, zero_division=0
            )
            
            results.update({
                'per_class_precision': prec_c.tolist(),
                'per_class_recall': rec_c.tolist(), 
                'per_class_f1': f1_c.tolist(),
                'per_class_support': support_c.tolist(),
                'macro_f1': float(f1_score(labels_test, binary_predictions, average='macro', zero_division=0)),
                'micro_f1': float(f1_score(labels_test, binary_predictions, average='micro', zero_division=0)),
                'weighted_f1': float(f1_score(labels_test, binary_predictions, average='weighted', zero_division=0)),
                'subset_accuracy': float(accuracy_score(labels_test, binary_predictions)),
                'hamming_loss': float(hamming_loss(labels_test, binary_predictions)),
                'jaccard_macro': float(jaccard_score(labels_test, binary_predictions, average='macro', zero_division=0)),
                'jaccard_micro': float(jaccard_score(labels_test, binary_predictions, average='micro', zero_division=0))
            })
        except Exception as e:
            print(f"⚠️  Error calculating supervised metrics: {e}")
            results['evaluation_type'] = 'unsupervised'
    else:
        # Unsupervised evaluation
        print("No true labels available - performing unsupervised evaluation...")
        
        # Use simple adaptive thresholding
        adaptive_thresholds = np.mean(predictions, axis=0) + 0.5 * np.std(predictions, axis=0)
        adaptive_thresholds = np.clip(adaptive_thresholds, 0.2, 0.8)
        binary_predictions = (predictions >= adaptive_thresholds).astype(int)
        
        # Calculate basic metrics
        labels_per_sample = binary_predictions.sum(axis=1)
        class_counts = binary_predictions.sum(axis=0)
        
        results = {
            'classes': classes,
            'n_test_samples': len(embeddings),
            'n_val_samples': 0,
            'evaluation_type': 'unsupervised',
            'optimal_thresholds': adaptive_thresholds.tolist(),
            'prediction_confidence_mean': float(predictions.mean()),
            'prediction_confidence_std': float(predictions.std()),
            'avg_labels_per_sample': float(labels_per_sample.mean()),
            'std_labels_per_sample': float(labels_per_sample.std()),
            'min_labels_per_sample': int(labels_per_sample.min()),
            'max_labels_per_sample': int(labels_per_sample.max()),
            'class_counts': class_counts.tolist(),
            'samples_with_no_labels': int((labels_per_sample == 0).sum()),
            'samples_with_one_label': int((labels_per_sample == 1).sum()),
            'samples_with_multiple_labels': int((labels_per_sample > 1).sum()),
            # Set supervised metrics to None
            'macro_f1': None,
            'micro_f1': None,
            'weighted_f1': None,
            'subset_accuracy': None,
            'hamming_loss': None,
            'jaccard_macro': None,
            'jaccard_micro': None,
            'per_class_precision': None,
            'per_class_recall': None,
            'per_class_f1': None,
            'per_class_support': None
        }
        
        # Add class frequency info
        if len(classes) > 0:
            class_freq_pairs = list(zip(classes, class_counts))
            class_freq_pairs.sort(key=lambda x: x[1], reverse=True)
            results['most_frequent_classes'] = [
                {'class': cls, 'count': int(count), 'percentage': float(count/len(binary_predictions)*100)}
                for cls, count in class_freq_pairs[:10]
            ]
        
        # High confidence predictions
        high_conf_mask = predictions > 0.7
        results['high_confidence_predictions'] = int(high_conf_mask.sum())
        results['high_confidence_percentage'] = float(high_conf_mask.sum() / predictions.size * 100)
    
    return results, predictions, binary_predictions

def print_results(results: Dict[str, Any], log_type: str):
    """Print evaluation results"""
    
    print(f"\n{'='*80}")
    print(f"EVALUATION RESULTS - {log_type.upper()}")
    print(f"{'='*80}")
    
    print(f"Test samples: {results['n_test_samples']}")
    if results['n_val_samples'] > 0:
        print(f"Validation samples: {results['n_val_samples']}")
    print(f"Classes: {len(results['classes'])}")
    print(f"Evaluation type: {results['evaluation_type']}")
    
    if results['evaluation_type'] == 'supervised' and results.get('macro_f1') is not None:
        print(f"\nSUPERVISED METRICS:")
        print(f"  Macro F1:       {results['macro_f1']:.4f}")
        print(f"  Micro F1:       {results['micro_f1']:.4f}")
        print(f"  Weighted F1:    {results['weighted_f1']:.4f}")
        print(f"  Subset Accuracy: {results['subset_accuracy']:.4f}")
        print(f"  Hamming Loss:   {results['hamming_loss']:.4f}")
        
        # Show top classes by F1
        if results['per_class_f1'] and len(results['classes']) > 0:
            print(f"\nTOP CLASSES BY F1:")
            class_f1_pairs = list(zip(results['classes'], results['per_class_f1']))
            class_f1_pairs.sort(key=lambda x: x[1], reverse=True)
            
            for i, (cls, f1) in enumerate(class_f1_pairs[:5]):
                print(f"  {i+1}. {cls:<25} F1: {f1:.4f}")
    else:
        print(f"\nUNSUPERVISED METRICS:")
        print(f"  Prediction Confidence: {results['prediction_confidence_mean']:.4f}")
        print(f"  Confidence Std:        {results['prediction_confidence_std']:.4f}")
        print(f"  Avg Labels/Sample:     {results['avg_labels_per_sample']:.2f}")
        print(f"  High Conf Predictions: {results['high_confidence_percentage']:.1f}%")
        
        # Show top predicted classes
        if 'most_frequent_classes' in results:
            print(f"\nTOP PREDICTED CLASSES:")
            for i, cls_info in enumerate(results['most_frequent_classes'][:5]):
                print(f"  {i+1}. {cls_info['class']:<25} {cls_info['percentage']:5.1f}%")
    
    print(f"{'='*80}")

def save_results(results: Dict[str, Any], predictions: np.ndarray, 
                binary_predictions: np.ndarray, log_type: str, 
                output_dir: Path):
    """Save evaluation results"""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save comprehensive results
    results_path = output_dir / f"evaluation_{log_type}.pkl"
    with open(results_path, 'wb') as f:
        pickle.dump({
            'results': results,
            'predictions': predictions.astype(np.float32),
            'binary_predictions': binary_predictions.astype(np.int8),
            'evaluation_type': results['evaluation_type'],
            'timestamp': str(Path().cwd())
        }, f)
    
    # Save text report
    report_path = output_dir / f"evaluation_{log_type}.txt"
    with open(report_path, 'w') as f:
        f.write(f"EVALUATION REPORT - {log_type.upper()}\n")
        f.write("="*50 + "\n\n")
        f.write(f"Evaluation type: {results['evaluation_type']}\n")
        f.write(f"Test samples: {results['n_test_samples']}\n")
        f.write(f"Classes: {len(results['classes'])}\n\n")
        
        if results['evaluation_type'] == 'supervised' and results.get('macro_f1') is not None:
            f.write("SUPERVISED METRICS:\n")
            f.write(f"  Macro F1: {results['macro_f1']:.4f}\n")
            f.write(f"  Micro F1: {results['micro_f1']:.4f}\n")
            f.write(f"  Weighted F1: {results['weighted_f1']:.4f}\n")
            f.write(f"  Subset Accuracy: {results['subset_accuracy']:.4f}\n")
        else:
            f.write("UNSUPERVISED METRICS:\n")
            f.write(f"  Prediction Confidence: {results['prediction_confidence_mean']:.4f}\n")
            f.write(f"  Avg Labels/Sample: {results['avg_labels_per_sample']:.2f}\n")
    
    print(f"✅ Results saved to: {results_path}")
    print(f"✅ Report saved to: {report_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Automatic Model Evaluation - Detects and evaluates all trained models",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model-path", type=str, help="Evaluate specific model path")
    parser.add_argument("--test-split", type=float, default=0.2, help="Test split fraction (default: 0.2)")
    parser.add_argument("--val-split", type=float, default=0.3, help="Validation split fraction (default: 0.3)")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory (default: results)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    
    args = parser.parse_args()
    
    # Detect system
    config = detect_system_resources()
    device = torch.device(config.device)
    
    print(f"🚀 Automatic Model Evaluation")
    print(f"Device: {device}")
    
    if args.model_path:
        # Evaluate specific model
        model_path = Path(args.model_path)
        if not model_path.exists():
            print(f"❌ Model not found: {model_path}")
            return 1
        
        # Extract log type from path
        log_type = model_path.stem.split('_')[1] if '_' in model_path.stem else 'unknown'
        models_to_evaluate = [{'model_path': model_path, 'log_type': log_type}]
    else:
        # Discover all models
        models_to_evaluate = discover_models()
        if not models_to_evaluate:
            print("❌ No trained models found in models/ directory")
            print("   Make sure you've trained models using transformer.py first")
            return 1
        
        print(f"🔍 Discovered {len(models_to_evaluate)} trained models:")
        for model_info in models_to_evaluate:
            status = "✅ Has labels" if model_info.get('has_labels', False) else "⚠️  No labels (unsupervised)"
            print(f"   • {model_info['log_type']}: {status}")
    
    # Evaluate each model
    for i, model_info in enumerate(models_to_evaluate, 1):
        log_type = model_info['log_type']
        model_path = model_info['model_path']
        
        print(f"\n{'='*60}")
        print(f"EVALUATING MODEL {i}/{len(models_to_evaluate)}: {log_type}")
        print(f"{'='*60}")
        
        try:
            # Load model
            print("Loading model...")
            model, saved_data, classes = load_model_smart(model_path, device)
            
            # Load data
            print("Loading test data...")
            embeddings, true_labels, data_classes = load_data_for_model(
                log_type, args.test_split, args.seed
            )
            
            # Use classes from model if data classes are empty
            if not data_classes and classes:
                data_classes = classes
            
            print(f"✅ Data loaded: {len(embeddings)} samples, {len(data_classes)} classes")
            
            # Evaluate
            print("Evaluating model...")
            results, predictions, binary_predictions = evaluate_model(
                model, embeddings, true_labels, data_classes, device, args.val_split
            )
            
            # Print results
            print_results(results, log_type)
            
            # Save results
            output_dir = Path(args.output_dir) / log_type
            save_results(results, predictions, binary_predictions, log_type, output_dir)
            
            print(f"✅ Completed evaluation for {log_type}")
            
        except Exception as e:
            print(f"❌ Evaluation failed for {log_type}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n🎉 Evaluation completed!")
    print(f"📁 Results saved to: {args.output_dir}/")
    
    return 0

if __name__ == "__main__":
    exit(main()) 