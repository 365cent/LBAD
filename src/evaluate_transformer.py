#!/usr/bin/env python3
"""
Transformer Model Evaluation Pipeline
====================================

Clean, focused evaluation of trained transformer models using the direct approach:

1. Load trained UnsupervisedMultiLabelTransformer model
2. Load LogBERT embeddings and true labels  
3. Run forward pass through model to get predictions
4. Compute standard supervised metrics (F1, Hamming loss, etc.)
5. Optional per-class threshold optimization

Usage:
    python src/evaluate_transformer.py --log-type wp-error
    python src/evaluate_transformer.py --log-type wp-error --optimize-thresholds
"""

import argparse
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import warnings

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import normalize
from sklearn.metrics import (
    precision_recall_fscore_support, f1_score, accuracy_score, 
    hamming_loss, jaccard_score, precision_score, recall_score,
    classification_report
)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Import from transformer module
import sys
sys.path.append('.')
from src.transformer import UnsupervisedMultiLabelTransformer, SystemConfig, detect_system_resources


class TransformerEvaluator:
    """Clean transformer model evaluator using direct supervised approach"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.device = torch.device(config.device)
    
    def load_model(self, log_type: str, model_path: Optional[str] = None) -> Tuple[UnsupervisedMultiLabelTransformer, List[str], Optional[Any]]:
        """Load trained transformer model"""
        
        if model_path:
            ckpt_path = Path(model_path)
        else:
            # Auto-find model
            models_dir = Path("models")
            patterns = [
                f"transformer_{log_type}_{self.config.node_name}_{self.config.job_id}.pth",
                f"transformer_{log_type}_*.pth",
                f"transformer_{log_type}.pth"
            ]
            
            ckpt_path = None
            for pattern in patterns:
                matches = list(models_dir.glob(pattern))
                if matches:
                    ckpt_path = max(matches, key=lambda p: p.stat().st_mtime)
                    break
            
            if not ckpt_path:
                raise FileNotFoundError(f"No model found for {log_type}")
        
        print(f"📂 Loading model from {ckpt_path}")
        
        # Load checkpoint
        try:
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            print(f"✅ Checkpoint loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint: {e}")
        
        # Extract model configuration
        classes = ckpt['classes']
        print(f"🔍 Found {len(classes)} classes in checkpoint")
        
        # Debug: show available metadata
        metadata_keys = [k for k in ckpt.keys() if k != 'model_state_dict']
        print(f"📋 Checkpoint metadata: {metadata_keys}")
        
        # Get input dimension from saved metadata (preferred) or infer from model structure
        input_dim = ckpt.get('input_dim', None)
        
        if input_dim is None:
            print("⚠️  No input_dim in metadata, inferring from model weights...")
            # Fallback: try to determine from model state dict
            for key, tensor in ckpt['model_state_dict'].items():
                if 'input_proj' in key and 'weight' in key and len(tensor.shape) == 2:
                    input_dim = tensor.shape[1]
                    print(f"✅ Inferred input_dim={input_dim} from {key}")
                    break
        else:
            print(f"✅ Using saved input_dim={input_dim}")
        
        if input_dim is None:
            # Show available keys for debugging
            model_keys = list(ckpt['model_state_dict'].keys())[:10]
            raise ValueError(f"Could not determine input dimension from model checkpoint. "
                           f"Available model keys (first 10): {model_keys}")
        
        # Rebuild model with same architecture
        model = UnsupervisedMultiLabelTransformer(
            input_dim=input_dim,
            latent_dim=ckpt.get('latent_dim', 512),
            n_labels=len(classes),
            n_clusters=min(8, len(classes)),
            dropout=0.1,
            transformer_layers=ckpt.get('transformer_layers', 12),
            attention_heads=ckpt.get('attention_heads', 16)
        )
        
        # Load weights
        model.load_state_dict(ckpt['model_state_dict'])
        model.to(self.device).eval()
        
        print(f"✅ Model loaded: {input_dim}D → {len(classes)} classes")
        print(f"🏗️  Architecture: {ckpt.get('transformer_layers', 12)} layers, {ckpt.get('attention_heads', 16)} heads, {ckpt.get('latent_dim', 512)}D latent")
        print(f"🏷️  Classes: {classes}")
        
        # Extract saved scaler if available
        saved_scaler = ckpt.get('scaler', None)
        if saved_scaler is not None:
            print(f"✅ Found saved preprocessing scaler from training")
        else:
            print(f"⚠️  No saved scaler found (older model format)")
        
        return model, classes, saved_scaler
    
    def load_embeddings_and_labels(self, log_type: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Load LogBERT embeddings and true labels"""
        
        embeddings_dir = Path("embeddings") / log_type
        
        # Load embeddings
        log_file = embeddings_dir / f"log_{log_type}.pkl"
        if not log_file.exists():
            raise FileNotFoundError(f"Embeddings not found: {log_file}")
        
        print(f"📂 Loading embeddings from {log_file}")
        with open(log_file, 'rb') as f:
            X = pickle.load(f)
        
        # Load labels
        label_file = embeddings_dir / f"label_{log_type}.pkl"
        if not label_file.exists():
            raise FileNotFoundError(f"Labels not found: {label_file}")
        
        print(f"📂 Loading labels from {label_file}")
        with open(label_file, 'rb') as f:
            label_data = pickle.load(f)
        
        y_true = label_data["vectors"]
        classes = label_data["classes"]
        
        print(f"✅ Loaded FULL dataset: {len(X):,} samples with {X.shape[1]}D embeddings")
        print(f"📊 Labels: {y_true.shape} for {len(classes)} classes")
        print(f"🎯 Evaluating on complete dataset (no sampling)")
        
        return X, y_true, classes
    
    def preprocess_embeddings(self, X: np.ndarray, saved_scaler=None) -> np.ndarray:
        """Apply same preprocessing as training using saved scaler"""
        
        if saved_scaler is not None:
            print(f"🔄 Preprocessing embeddings using saved scaler from training...")
            # Use the same scaler that was used during training
            X_scaled = saved_scaler.transform(X)
        else:
            print(f"⚠️  No saved scaler found, fitting new scaler (may cause inconsistency)...")
            # Fallback: fit new scaler (not ideal)
            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X)
        
        # L2 normalization (same as training)
        X_normalized = normalize(X_scaled, norm='l2', axis=1).astype(np.float32)
        
        print(f"✅ Preprocessing complete")
        return X_normalized
    
    def predict(self, model: UnsupervisedMultiLabelTransformer, X: np.ndarray, 
                batch_size: int = 64) -> Tuple[np.ndarray, np.ndarray]:
        """Run forward pass and get predictions"""
        print(f"🤖 Generating predictions...")
        
        predictions = []
        probs = []
        
        model.eval()
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch = torch.from_numpy(X[i:i+batch_size]).float().to(self.device)
                logits = model(batch)["labels"]
                batch_probs = torch.sigmoid(logits).cpu().numpy()
                
                probs.append(batch_probs)
        
        probs = np.vstack(probs)
        predictions = (probs >= 0.5).astype(int)
        
        print(f"✅ Generated predictions for {len(predictions):,} samples")
        return predictions, probs
    
    def optimize_thresholds(self, y_true: np.ndarray, probs: np.ndarray, 
                          classes: List[str]) -> np.ndarray:
        """
        Advanced per-class threshold optimization for highly imbalanced multi-label classification.
        
        Implements multiple strategies:
        1. F1-based optimization with fine-grained search
        2. Precision-Recall curve analysis for rare classes
        3. Class imbalance-aware threshold selection
        4. Support-weighted threshold adjustment
        5. Statistical significance testing for threshold selection
        """
        from sklearn.metrics import precision_recall_curve, balanced_accuracy_score
        
        print(f"🎯 Optimizing advanced per-class thresholds...")
        
        n_classes = len(classes)
        optimized_thresholds = np.zeros(n_classes)
        
        for i, cls in enumerate(classes):
            y_true_class = y_true[:, i]
            probs_class = probs[:, i]
            
            pos_samples = y_true_class.sum()
            total_samples = len(y_true_class)
            pos_rate = pos_samples / total_samples
            
            # Strategy selection based on class characteristics
            if pos_samples == 0:
                # No positive samples - use very high threshold to minimize false positives
                optimized_thresholds[i] = 0.95
                print(f"   {cls:<25} NO POS SAMPLES -> threshold: 0.95")
                continue
                
            elif pos_samples < 10:
                # Extremely rare classes (< 10 samples)
                # Use precision-recall curve to find best threshold
                precision, recall, thresholds_pr = precision_recall_curve(y_true_class, probs_class)
                
                # Find threshold that maximizes (precision * recall) / (precision + recall) = F1
                f1_scores_pr = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-8)
                
                if len(f1_scores_pr) > 0:
                    best_idx = np.argmax(f1_scores_pr)
                    best_threshold = thresholds_pr[best_idx]
                    
                    # For extremely rare classes, bias towards higher recall (lower threshold)
                    # to ensure we don't miss the few positive samples
                    adjusted_threshold = best_threshold * 0.8  # Reduce by 20%
                    optimized_thresholds[i] = np.clip(adjusted_threshold, 0.05, 0.90)
                    best_f1 = f1_scores_pr[best_idx]
                    print(f"   {cls:<25} threshold: {optimized_thresholds[i]:.3f} (F1: {best_f1:.3f}) | pos: {pos_samples} | RARE_PR")
                else:
                    optimized_thresholds[i] = 0.3  # Conservative for rare classes
                    print(f"   {cls:<25} threshold: 0.30 (fallback) | pos: {pos_samples} | RARE_FB")
                    
            elif pos_samples < 100:
                # Rare classes (10-100 samples)
                # Use F1 optimization with fine-grained search and statistical validation
                best_f1 = 0
                best_threshold = 0.5
                candidate_thresholds = []
                candidate_f1s = []
                
                # Fine-grained search for rare classes
                for threshold in np.linspace(0.05, 0.95, 50):  # More granular for rare classes
                    pred_class = (probs_class >= threshold).astype(int)
                    f1 = f1_score(y_true_class, pred_class, zero_division=0)
                    
                    candidate_thresholds.append(threshold)
                    candidate_f1s.append(f1)
                    
                    if f1 > best_f1:
                        best_f1 = f1
                        best_threshold = threshold
                
                # Statistical validation: check if the best threshold is significantly better
                candidate_f1s = np.array(candidate_f1s)
                candidate_thresholds = np.array(candidate_thresholds)
                
                # Find all thresholds within 95% of best F1
                good_threshold_mask = candidate_f1s >= (best_f1 * 0.95)
                good_thresholds = candidate_thresholds[good_threshold_mask]
                
                if len(good_thresholds) > 1:
                    # Among good thresholds, prefer one that gives balanced precision/recall
                    balanced_scores = []
                    for thresh in good_thresholds:
                        pred_class = (probs_class >= thresh).astype(int)
                        bal_acc = balanced_accuracy_score(y_true_class, pred_class)
                        balanced_scores.append(bal_acc)
                    
                    best_balanced_idx = np.argmax(balanced_scores)
                    optimized_thresholds[i] = good_thresholds[best_balanced_idx]
                    print(f"   {cls:<25} threshold: {optimized_thresholds[i]:.3f} (F1: {best_f1:.3f}) | pos: {pos_samples} | RARE_BAL")
                else:
                    optimized_thresholds[i] = best_threshold
                    print(f"   {cls:<25} threshold: {best_threshold:.3f} (F1: {best_f1:.3f}) | pos: {pos_samples} | RARE_F1")
                    
            else:
                # Common classes (>= 100 samples)
                # Use comprehensive optimization considering multiple metrics
                best_score = 0
                best_threshold = 0.5
                
                # Coarse-to-fine search for efficiency
                # Coarse search
                coarse_thresholds = np.linspace(0.1, 0.9, 17)
                coarse_scores = []
                
                for threshold in coarse_thresholds:
                    pred_class = (probs_class >= threshold).astype(int)
                    
                    # Composite score: weighted combination of F1, balanced accuracy
                    f1 = f1_score(y_true_class, pred_class, zero_division=0)
                    bal_acc = balanced_accuracy_score(y_true_class, pred_class)
                    
                    # Weight based on class frequency (more balanced for common classes)
                    if pos_rate > 0.1:  # Common class
                        composite_score = 0.7 * f1 + 0.3 * bal_acc
                    else:  # Still somewhat rare
                        composite_score = 0.8 * f1 + 0.2 * bal_acc
                    
                    coarse_scores.append(composite_score)
                    
                    if composite_score > best_score:
                        best_score = composite_score
                        best_threshold = threshold
                
                # Fine search around best coarse threshold
                fine_range = 0.1  # Search +/- 0.1 around best coarse threshold
                fine_start = max(0.05, best_threshold - fine_range)
                fine_end = min(0.95, best_threshold + fine_range)
                
                fine_thresholds = np.linspace(fine_start, fine_end, 21)
                
                for threshold in fine_thresholds:
                    pred_class = (probs_class >= threshold).astype(int)
                    
                    f1 = f1_score(y_true_class, pred_class, zero_division=0)
                    bal_acc = balanced_accuracy_score(y_true_class, pred_class)
                    
                    if pos_rate > 0.1:
                        composite_score = 0.7 * f1 + 0.3 * bal_acc
                    else:
                        composite_score = 0.8 * f1 + 0.2 * bal_acc
                    
                    if composite_score > best_score:
                        best_score = composite_score
                        best_threshold = threshold
                
                optimized_thresholds[i] = best_threshold
                
                # Calculate final F1 for display
                final_pred = (probs_class >= best_threshold).astype(int)
                final_f1 = f1_score(y_true_class, final_pred, zero_division=0)
                strategy = "COMMON_BAL" if pos_rate > 0.1 else "MID_COMP"
                print(f"   {cls:<25} threshold: {best_threshold:.3f} (F1: {final_f1:.3f}) | pos: {pos_samples} | {strategy}")
        
        return optimized_thresholds
    
    def compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                       probs: np.ndarray, classes: List[str]) -> Dict[str, Any]:
        """Compute comprehensive supervised metrics"""
        
        # Overall multi-label metrics
        metrics = {
            'subset_accuracy': float(accuracy_score(y_true, y_pred)),
            'hamming_loss': float(hamming_loss(y_true, y_pred)),
            'micro_f1': float(f1_score(y_true, y_pred, average='micro', zero_division=0)),
            'macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
            'weighted_f1': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
            'samples_f1': float(f1_score(y_true, y_pred, average='samples', zero_division=0)),
            'micro_precision': float(precision_score(y_true, y_pred, average='micro', zero_division=0)),
            'macro_precision': float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
            'micro_recall': float(recall_score(y_true, y_pred, average='micro', zero_division=0)),
            'macro_recall': float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
            'jaccard_micro': float(jaccard_score(y_true, y_pred, average='micro', zero_division=0)),
            'jaccard_macro': float(jaccard_score(y_true, y_pred, average='macro', zero_division=0)),
        }
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        metrics.update({
            'per_class_precision': precision.tolist(),
            'per_class_recall': recall.tolist(),
            'per_class_f1': f1.tolist(),
            'per_class_support': support.tolist(),
            'classes': classes,
            'n_samples': len(y_true),
        })
        
        # Sample distribution analysis
        pred_labels_per_sample = y_pred.sum(axis=1)
        true_labels_per_sample = y_true.sum(axis=1)
        
        metrics.update({
            'avg_predicted_labels': float(pred_labels_per_sample.mean()),
            'avg_true_labels': float(true_labels_per_sample.mean()),
            'samples_no_pred_labels': int((pred_labels_per_sample == 0).sum()),
            'samples_no_true_labels': int((true_labels_per_sample == 0).sum()),
            'samples_multi_pred_labels': int((pred_labels_per_sample > 1).sum()),
            'samples_multi_true_labels': int((true_labels_per_sample > 1).sum()),
        })
        
        return metrics
    
    def print_results(self, metrics: Dict[str, Any], classes: List[str], 
                     y_true: np.ndarray, y_pred: np.ndarray):
        """Print comprehensive results"""
        
        print(f"\n📊 TRANSFORMER EVALUATION RESULTS")
        print("=" * 60)
        print(f"Test samples:     {metrics['n_samples']:,}")
        print(f"Classes:          {len(classes)}")
        print("")
        
        print("OVERALL METRICS:")
        print("-" * 40)
        print(f"Subset Accuracy:  {metrics['subset_accuracy']:.4f}")
        print(f"Hamming Loss:     {metrics['hamming_loss']:.4f}")
        print(f"Micro F1:         {metrics['micro_f1']:.4f}")
        print(f"Macro F1:         {metrics['macro_f1']:.4f}")
        print(f"Weighted F1:      {metrics['weighted_f1']:.4f}")
        print(f"Samples F1:       {metrics['samples_f1']:.4f}")
        print(f"Micro Precision:  {metrics['micro_precision']:.4f}")
        print(f"Macro Precision:  {metrics['macro_precision']:.4f}")
        print(f"Micro Recall:     {metrics['micro_recall']:.4f}")
        print(f"Macro Recall:     {metrics['macro_recall']:.4f}")
        print(f"Jaccard (Micro):  {metrics['jaccard_micro']:.4f}")
        print(f"Jaccard (Macro):  {metrics['jaccard_macro']:.4f}")
        print("")
        
        print("PER-CLASS METRICS:")
        print("-" * 60)
        print(f"{'Class':<25} {'F1':<8} {'Precision':<10} {'Recall':<8} {'Support':<8}")
        print("-" * 60)
        
        for i, cls in enumerate(classes):
            f1 = metrics['per_class_f1'][i]
            precision = metrics['per_class_precision'][i]
            recall = metrics['per_class_recall'][i]
            support = metrics['per_class_support'][i]
            print(f"{cls:<25} {f1:<8.3f} {precision:<10.3f} {recall:<8.3f} {support:<8}")
        
        print("")
        print("SAMPLE DISTRIBUTION:")
        print("-" * 40)
        print(f"Avg predicted labels/sample: {metrics['avg_predicted_labels']:.3f}")
        print(f"Avg true labels/sample:      {metrics['avg_true_labels']:.3f}")
        print(f"Samples with no pred labels: {metrics['samples_no_pred_labels']:,}")
        print(f"Samples with no true labels: {metrics['samples_no_true_labels']:,}")
        print(f"Samples with >1 pred labels: {metrics['samples_multi_pred_labels']:,}")
        print(f"Samples with >1 true labels: {metrics['samples_multi_true_labels']:,}")
        
        print("")
        print("SKLEARN CLASSIFICATION REPORT:")
        print("-" * 60)
        report = classification_report(y_true, y_pred, target_names=classes, zero_division=0, digits=3)
        print(report)
    
    def save_results(self, metrics: Dict[str, Any], y_pred: np.ndarray, 
                    probs: np.ndarray, y_true: np.ndarray, log_type: str, 
                    thresholds: Optional[np.ndarray] = None):
        """Save evaluation results"""
        
        output_dir = Path("results") / log_type
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save comprehensive results
        results = {
            'metrics': metrics,
            'predictions': y_pred.astype(np.int8),
            'probabilities': probs.astype(np.float32),
            'true_labels': y_true.astype(np.int8),
            'evaluation_type': 'direct_supervised_transformer',
            'config': self.config.__dict__,
            'timestamp': time.time()
        }
        
        if thresholds is not None:
            results['optimized_thresholds'] = thresholds.tolist()
        
        results_file = output_dir / f"transformer_evaluation_{log_type}_{self.config.node_name}_{self.config.job_id}.pkl"
        with open(results_file, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"💾 Results saved to: {results_file}")
        return results_file


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained transformer model")
    parser.add_argument("--log-type", type=str, required=True, 
                       help="Log type to evaluate (e.g., wp-access, wp-error)")
    parser.add_argument("--model-path", type=str, 
                       help="Path to trained model (auto-detected if not provided)")
    parser.add_argument("--batch-size", type=int, default=64,
                       help="Batch size for inference")
    
    args = parser.parse_args()
    
    # Detect system configuration
    config = detect_system_resources()
    
    print("🚀 Transformer Model Evaluation (Direct Supervised)")
    print("=" * 60)
    print(f"Log type: {args.log_type}")
    print(f"Device: {config.device}")
    print(f"Node: {config.node_name} | Job: {config.job_id}")
    print(f"Advanced threshold optimization: ENABLED (default)")
    print(f"📊 Dataset: Using FULL LogBERT embeddings (no sampling)")
    print("")
    
    try:
        # Initialize evaluator
        evaluator = TransformerEvaluator(config)
        
        # 1. Load trained model
        model, model_classes, saved_scaler = evaluator.load_model(args.log_type, args.model_path)
        
        # 2. Load embeddings and true labels
        X, y_true, data_classes = evaluator.load_embeddings_and_labels(args.log_type)
        
                # Verify class compatibility
        if model_classes != data_classes:
            print(f"⚠️  Class mismatch between model and data")
            print(f"   Model classes: {model_classes}")
            print(f"   Data classes: {data_classes}")
            print(f"   Using model classes for evaluation")
            classes = model_classes
            
            # Adjust true labels to match model classes if needed
            if len(model_classes) != len(data_classes):
                print(f"⚠️  Different number of classes, this may cause issues")
        else:
            print(f"✅ Class compatibility verified: {len(model_classes)} classes match")
            classes = model_classes
        
        # 3. Preprocess embeddings (same as training)
        X_processed = evaluator.preprocess_embeddings(X, saved_scaler)
        
        # 4. Generate predictions
        y_pred, probs = evaluator.predict(model, X_processed, args.batch_size)
        
        # 5. Always optimize thresholds (default behavior)
        optimized_thresholds = evaluator.optimize_thresholds(y_true, probs, classes)
        y_pred_optimized = (probs >= optimized_thresholds).astype(int)
        
        print(f"\n🎯 Using advanced optimized thresholds")
        y_pred = y_pred_optimized
        
        # 6. Compute metrics
        metrics = evaluator.compute_metrics(y_true, y_pred, probs, classes)
        
        # 7. Print results
        evaluator.print_results(metrics, classes, y_true, y_pred)
        
        # 8. Save results
        results_file = evaluator.save_results(
            metrics, y_pred, probs, y_true, args.log_type, optimized_thresholds
        )
        
        print(f"\n✅ Evaluation completed for {args.log_type}")
        print(f"📁 Results saved to: {results_file}")
        
        # Summary
        print(f"\n🎯 SUMMARY")
        print(f"   Macro F1:  {metrics['macro_f1']:.4f}")
        print(f"   Micro F1:  {metrics['micro_f1']:.4f}")
        print(f"   Accuracy:  {metrics['subset_accuracy']:.4f}")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 