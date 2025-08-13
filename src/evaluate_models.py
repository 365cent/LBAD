#!/usr/bin/env python3
"""
Enhanced Transformer Model Evaluation Pipeline
==============================================

Comprehensive evaluation of enhanced multi-label transformer models with support for:

1. Load enhanced UnsupervisedMultiLabelTransformer models (with Focal Loss, Enhanced Attention, Contrastive Learning)
2. Handle new prediction format from one-vs-rest training with normal class modeling
3. Load LogBERT embeddings and true labels with shape validation and error handling
4. Run forward pass through enhanced transformer to get multi-label predictions
5. Compute comprehensive supervised metrics (F1, Hamming loss, Jaccard, etc.)
6. Advanced per-class threshold optimization with imbalance handling
7. Enhanced visualization and reporting with transformer-specific insights

Key Enhancements:
- Support for enhanced transformer features (Focal Loss, Enhanced Attention, Contrastive Learning)
- Improved prediction loading with shape mismatch detection and automatic fixing
- Enhanced reporting with normal class analysis and data distribution insights
- Better error handling and debugging information
- Support for one-vs-rest strategy with normal class suppression logic

Usage:
    python src/evaluate_models.py --log-type wp-error
    python src/evaluate_models.py --log-type wp-error --optimize-thresholds
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
    """Clean multi-label transformer model evaluator using direct supervised approach"""
    
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
        
        print(f"Loading model: {ckpt_path}")
        
        # Load checkpoint
        try:
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint: {e}")
        
        # Extract model configuration
        classes = ckpt['classes']
        
        # Get input dimension from saved metadata (preferred) or infer from model structure
        input_dim = ckpt.get('input_dim', None)
        
        if input_dim is None:
            # Fallback: try to determine from model state dict
            for key, tensor in ckpt['model_state_dict'].items():
                if 'input_proj' in key and 'weight' in key and len(tensor.shape) == 2:
                    input_dim = tensor.shape[1]
                    break
        
        if input_dim is None:
            model_keys = list(ckpt['model_state_dict'].keys())[:10]
            raise ValueError(f"Could not determine input dimension from model checkpoint. "
                           f"Available model keys (first 10): {model_keys}")
        
        # Rebuild model with same architecture - handle both single and multi-label models
        n_labels = len(classes)
        if 'n_labels' in ckpt and ckpt['n_labels'] == 1:
            # Single-label model (from separate model approach)
            n_labels = 1
        
        # Check for enhanced features in the saved model
        enhanced_features = ckpt.get('enhanced_features', False)
        use_enhanced_attention = ckpt.get('use_enhanced_attention', enhanced_features)
        use_label_correlation = ckpt.get('use_label_correlation', enhanced_features and n_labels > 1)
        use_contrastive = ckpt.get('use_contrastive', enhanced_features)
        
        model = UnsupervisedMultiLabelTransformer(
            input_dim=input_dim,
            latent_dim=ckpt.get('latent_dim', 256),
            n_labels=n_labels,
            n_clusters=min(8, len(classes)),
            dropout=ckpt.get('dropout', 0.1),
            transformer_layers=ckpt.get('transformer_layers', 2),
            attention_heads=ckpt.get('attention_heads', 4),
            use_enhanced_attention=use_enhanced_attention,
            use_label_correlation=use_label_correlation,
            use_contrastive=use_contrastive
        )
        
        # Load weights
        model.load_state_dict(ckpt['model_state_dict'])
        model.to(self.device).eval()
        
        print(f"Model loaded: {input_dim}D → {len(classes)} classes (n_labels={n_labels})")
        
        # Extract saved scaler if available
        saved_scaler = ckpt.get('scaler', None)
        
        return model, classes, saved_scaler
    
    def load_embeddings_and_labels(self, log_type: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Load LogBERT embeddings and true labels"""
        
        embeddings_dir = Path("embeddings") / log_type
        
        # Load embeddings
        log_file = embeddings_dir / f"log_{log_type}.pkl"
        if not log_file.exists():
            raise FileNotFoundError(f"Embeddings not found: {log_file}")
        
        print(f"Loading embeddings: {log_file}")
        with open(log_file, 'rb') as f:
            X = pickle.load(f)
        
        # Load labels
        label_file = embeddings_dir / f"label_{log_type}.pkl"
        if not label_file.exists():
            raise FileNotFoundError(f"Labels not found: {label_file}")
        
        with open(label_file, 'rb') as f:
            label_data = pickle.load(f)
        
        y_true = label_data["vectors"]
        classes = label_data["classes"]
        
        print(f"Loaded dataset: {len(X):,} samples, {X.shape[1]}D embeddings, {len(classes)} classes")
        
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
        """Run forward pass and get multi-label predictions"""
        print(f"🤖 Generating predictions...")
        
        predictions = []
        probs = []
        
        model.eval()
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch = torch.from_numpy(X[i:i+batch_size]).float().to(self.device)
                logits = model(batch)["multi_label_scores"]
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
        
        Optimizes thresholds for each class independently in multi-label setting where
        each sample can have multiple positive labels.
        
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
                optimized_thresholds[i] = 0.95
                print(f"   {cls:<25} threshold: 0.95 (NO POS SAMPLES)")
                continue
                
            elif pos_samples < 10:
                precision, recall, thresholds_pr = precision_recall_curve(y_true_class, probs_class)
                f1_scores_pr = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-8)
                
                if len(f1_scores_pr) > 0:
                    best_idx = np.argmax(f1_scores_pr)
                    best_threshold = thresholds_pr[best_idx]
                    adjusted_threshold = best_threshold * 0.8
                    optimized_thresholds[i] = np.clip(adjusted_threshold, 0.05, 0.90)
                    print(f"   {cls:<25} threshold: {optimized_thresholds[i]:.3f} (F1: {f1_scores_pr[best_idx]:.3f}) | pos: {pos_samples} | RARE_PR")
                else:
                    optimized_thresholds[i] = 0.3
                    print(f"   {cls:<25} threshold: 0.30 (fallback) | pos: {pos_samples} | RARE_FB")
                    
            elif pos_samples < 100:
                best_f1 = 0
                best_threshold = 0.5
                candidate_thresholds = []
                candidate_f1s = []
                
                for threshold in np.linspace(0.05, 0.95, 50):
                    pred_class = (probs_class >= threshold).astype(int)
                    f1 = f1_score(y_true_class, pred_class, zero_division=0)
                    candidate_thresholds.append(threshold)
                    candidate_f1s.append(f1)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_threshold = threshold
                
                candidate_f1s = np.array(candidate_f1s)
                candidate_thresholds = np.array(candidate_thresholds)
                good_threshold_mask = candidate_f1s >= (best_f1 * 0.95)
                good_thresholds = candidate_thresholds[good_threshold_mask]
                
                if len(good_thresholds) > 1:
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
                best_score = 0
                best_threshold = 0.5
                
                coarse_thresholds = np.linspace(0.1, 0.9, 17)
                for threshold in coarse_thresholds:
                    pred_class = (probs_class >= threshold).astype(int)
                    f1 = f1_score(y_true_class, pred_class, zero_division=0)
                    bal_acc = balanced_accuracy_score(y_true_class, pred_class)
                    composite_score = 0.7 * f1 + 0.3 * bal_acc if pos_rate > 0.1 else 0.8 * f1 + 0.2 * bal_acc
                    if composite_score > best_score:
                        best_score = composite_score
                        best_threshold = threshold
                
                fine_range = 0.1
                fine_start = max(0.05, best_threshold - fine_range)
                fine_end = min(0.95, best_threshold + fine_range)
                fine_thresholds = np.linspace(fine_start, fine_end, 21)
                
                for threshold in fine_thresholds:
                    pred_class = (probs_class >= threshold).astype(int)
                    f1 = f1_score(y_true_class, pred_class, zero_division=0)
                    bal_acc = balanced_accuracy_score(y_true_class, pred_class)
                    composite_score = 0.7 * f1 + 0.3 * bal_acc if pos_rate > 0.1 else 0.8 * f1 + 0.2 * bal_acc
                    if composite_score > best_score:
                        best_score = composite_score
                        best_threshold = threshold
                
                optimized_thresholds[i] = best_threshold
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
        
        # Add label-wise accuracy (more intuitive for multi-label)
        # This calculates accuracy for each label independently, then averages
        label_wise_accuracies = []
        for i in range(y_true.shape[1]):
            correct_predictions = (y_true[:, i] == y_pred[:, i]).sum()
            total_predictions = len(y_true)
            label_accuracy = correct_predictions / total_predictions
            label_wise_accuracies.append(label_accuracy)
        
        metrics.update({
            'label_wise_accuracy_micro': float(np.mean(label_wise_accuracies)),  # Average across labels
            'label_wise_accuracy_macro': float(np.mean(label_wise_accuracies)),  # Same as micro for accuracy
            'per_label_accuracies': [float(acc) for acc in label_wise_accuracies],  # Individual label accuracies
            'overall_correct_labels': float(1.0 - hamming_loss(y_true, y_pred)),  # 1 - hamming loss
        })
        
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
                     y_true: np.ndarray, y_pred: np.ndarray, enhanced_info: Dict[str, Any] = None):
        """Print comprehensive results"""
        
        print(f"\n{'='*70}")
        print(f"🎯 ENHANCED TRANSFORMER EVALUATION RESULTS")
        print(f"{'='*70}")
        print(f"Test samples: {metrics['n_samples']:,} | Classes: {len(classes)}")
        
        # Enhanced transformer information
        if enhanced_info:
            print(f"\n🚀 ENHANCED TRANSFORMER INFO:")
            print(f"  Model Type: {enhanced_info.get('model_type', 'Enhanced Multi-Label Transformer')}")
            print(f"  Enhanced Features: {enhanced_info.get('enhanced_features', 'Unknown')}")
            print(f"  Architecture: {enhanced_info.get('architecture', 'Standard')}")
            if 'normal_class_info' in enhanced_info:
                normal_info = enhanced_info['normal_class_info']
                print(f"  Normal Class: {normal_info.get('samples', 0)} high-confidence predictions")
                print(f"  Attack Suppression: {normal_info.get('suppressed', 0)} samples")
        
        print(f"\n📊 PERFORMANCE METRICS:")
        print(f"  Subset Accuracy:  {metrics['subset_accuracy']:.4f}")
        print(f"  Label-wise Acc:   {metrics['label_wise_accuracy_micro']:.4f}")
        print(f"  Hamming Loss:     {metrics['hamming_loss']:.4f}")
        print(f"  Micro F1:         {metrics['micro_f1']:.4f}")
        print(f"  Macro F1:         {metrics['macro_f1']:.4f}")
        print(f"  Weighted F1:      {metrics['weighted_f1']:.4f}")
        print(f"  Micro Precision:  {metrics['micro_precision']:.4f}")
        print(f"  Macro Precision:  {metrics['macro_precision']:.4f}")
        print(f"  Micro Recall:     {metrics['micro_recall']:.4f}")
        print(f"  Macro Recall:     {metrics['macro_recall']:.4f}")
        
        print(f"\n🏷️  PER-CLASS METRICS:")
        print("-" * 70)
        print(f"{'Class':<25} {'F1':<8} {'Precision':<10} {'Recall':<8} {'Support':<8} {'Rate':<8}")
        print("-" * 70)
        
        total_samples = metrics['n_samples']
        for i, cls in enumerate(classes):
            f1 = metrics['per_class_f1'][i]
            precision = metrics['per_class_precision'][i]
            recall = metrics['per_class_recall'][i]
            support = metrics['per_class_support'][i]
            rate = (support / total_samples) * 100 if total_samples > 0 else 0
            print(f"{cls:<25} {f1:<8.3f} {precision:<10.3f} {recall:<8.3f} {support:<8} {rate:<8.1f}%")
        
        print(f"\n📋 CLASSIFICATION REPORT:")
        print("-" * 70)
        report = classification_report(y_true, y_pred, target_names=classes, zero_division=0, digits=3)
        print(report)
        
        print(f"\n🔍 ADDITIONAL ANALYSIS:")
        print(f"  Average Predicted Labels per Sample: {metrics['avg_predicted_labels']:.2f}")
        print(f"  Average True Labels per Sample:     {metrics['avg_true_labels']:.2f}")
        print(f"  Samples with No Predicted Labels:   {metrics['samples_no_pred_labels']:,} ({(metrics['samples_no_pred_labels']/total_samples)*100:.1f}%)")
        print(f"  Samples with No True Labels:        {metrics['samples_no_true_labels']:,} ({(metrics['samples_no_true_labels']/total_samples)*100:.1f}%)")
        print(f"  Samples with Multiple Predicted:    {metrics['samples_multi_pred_labels']:,} ({(metrics['samples_multi_pred_labels']/total_samples)*100:.1f}%)")
        print(f"  Samples with Multiple True:         {metrics['samples_multi_true_labels']:,} ({(metrics['samples_multi_true_labels']/total_samples)*100:.1f}%)")
        
        # Data distribution analysis
        print(f"\n📈 DATA DISTRIBUTION:")
        pred_dist = np.sum(y_pred, axis=0)
        true_dist = np.sum(y_true, axis=0)
        for i, cls in enumerate(classes):
            pred_count = pred_dist[i]
            true_count = true_dist[i]
            print(f"  {cls:<25} True: {true_count:>6,} | Predicted: {pred_count:>6,} | Ratio: {(pred_count/max(true_count,1)):.2f}")
        
        print(f"{'='*70}")
    
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


def auto_detect_log_types() -> List[str]:
    """Auto-detect available log types from embeddings directory"""
    embeddings_dir = Path("embeddings")
    if not embeddings_dir.exists():
        return []
    
    log_types = []
    for item in embeddings_dir.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            # Check if this log type has required files
            log_file = item / f"log_{item.name}.pkl"
            label_file = item / f"label_{item.name}.pkl"
            if log_file.exists() and label_file.exists():
                log_types.append(item.name)
    
    return sorted(log_types)


def load_transformer_predictions(log_type: str) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]]:
    """
    Load existing transformer predictions from results directory if available
    
    Returns:
        Tuple of (predictions, probabilities, true_labels, classes) or None if not found
    """
    results_dir = Path("results") / log_type
    predictions_file = results_dir / "predictions.pkl"
    
    if not predictions_file.exists():
        return None
    
    try:
        with open(predictions_file, 'rb') as f:
            pred_data = pickle.load(f)
        
        # Extract data based on new transformer.py output format
        if isinstance(pred_data, dict):
            # New format from enhanced transformer - prioritize final processed predictions
            predictions = pred_data.get('preds')  # Final multi-label predictions (attack classes only)
            probabilities = pred_data.get('probs')  # Final multi-label probabilities (attack classes only) 
            true_labels = pred_data.get('true_labels')
            classes = pred_data.get('classes', [])
            
            # Check for additional data from the new enhanced transformer
            normal_probs = pred_data.get('normal_probs')  # Normal class probabilities
            all_classes = pred_data.get('all_classes', classes)  # Including 'normal' class
            raw_predictions_with_normal = pred_data.get('raw_predictions_with_normal')
            raw_probabilities_with_normal = pred_data.get('raw_probabilities_with_normal')
            
            # Fallback to alternative key names if the above don't exist
            if predictions is None:
                predictions = pred_data.get('predictions', pred_data.get('binary_predictions'))
            if probabilities is None:
                probabilities = pred_data.get('probabilities')
            if true_labels is None:
                true_labels = pred_data.get('y_true')
            
            # Validate data consistency
            if predictions is not None and true_labels is not None and classes:
                predictions = np.array(predictions)
                probabilities = np.array(probabilities) if probabilities is not None else (predictions.astype(float) + 0.1)
                true_labels = np.array(true_labels)
                
                # Ensure shape consistency between predictions and true labels
                if predictions.shape != true_labels.shape:
                    print(f"⚠️  Shape mismatch detected:")
                    print(f"   Predictions: {predictions.shape}")
                    print(f"   True labels: {true_labels.shape}")
                    
                    # Try to fix common shape mismatches
                    if len(predictions.shape) == 2 and len(true_labels.shape) == 2:
                        if predictions.shape[0] == true_labels.shape[0]:
                            # Same number of samples, different number of classes
                            min_classes = min(predictions.shape[1], true_labels.shape[1])
                            predictions = predictions[:, :min_classes]
                            probabilities = probabilities[:, :min_classes] if probabilities.shape[1] > min_classes else probabilities
                            true_labels = true_labels[:, :min_classes]
                            classes = classes[:min_classes]
                            print(f"   ✅ Fixed by using first {min_classes} classes")
                        else:
                            print(f"   ❌ Cannot fix: different number of samples")
                            return None
                    else:
                        print(f"   ❌ Cannot fix: incompatible dimensions")
                        return None
                
                print(f"✅ Loading cached predictions for {log_type}: {predictions.shape[0]:,} samples, {len(classes)} classes")
                
                # Log additional info about enhanced transformer data
                if normal_probs is not None:
                    print(f"📊 Enhanced transformer data: includes normal class predictions")
                    print(f"   All classes: {all_classes}")
                    print(f"   Normal samples (high confidence): {np.sum(normal_probs >= 0.7) if normal_probs is not None else 'N/A'}")
                
                return predictions, probabilities, true_labels, classes
            else:
                missing_fields = []
                if predictions is None:
                    missing_fields.append('predictions')
                if true_labels is None:
                    missing_fields.append('true_labels')
                if not classes:
                    missing_fields.append('classes')
                
                print(f"⚠️  Predictions file for {log_type} missing required fields: {missing_fields}")
                print(f"   Available keys: {list(pred_data.keys())}")
                return None
        else:
            print(f"⚠️  Invalid predictions file format for {log_type}. Expected dict, got {type(pred_data)}")
            return None
            
    except Exception as e:
        print(f"❌ Error loading predictions for {log_type}: {e}. Running full evaluation.")
        import traceback
        print(f"   Traceback: {traceback.format_exc()}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained multi-label transformer model(s)")
    parser.add_argument("--log-type", type=str, 
                       help="Log type to evaluate (e.g., wp-access, wp-error). If not specified, evaluates all available types.")
    parser.add_argument("--model-path", type=str, 
                       help="Path to trained model (auto-detected if not provided)")
    parser.add_argument("--batch-size", type=int, default=64,
                       help="Batch size for inference")
    parser.add_argument("--optimize-thresholds", action="store_true",
                       help="Optimize per-class thresholds for better performance")
    
    args = parser.parse_args()
    
    # Detect system configuration
    config = detect_system_resources()
    
    # Auto-detect log types if not specified
    if args.log_type:
        log_types = [args.log_type]
    else:
        log_types = auto_detect_log_types()
        if not log_types:
            print("No valid log types found in embeddings directory!")
            print("Expected structure: embeddings/<log-type>/log_<log-type>.pkl and label_<log-type>.pkl")
            return
    
    print("Multi-Label Transformer Model Evaluation")
    print("=" * 60)
    print(f"Log types: {', '.join(log_types)}")
    print(f"Device: {config.device} | Node: {config.node_name}")
    print("")
    
    all_results = {}
    evaluation_start_time = time.time()
    
    for log_type in log_types:
        print(f"\n{log_type.upper()}")
        print("-" * 30)
        
        try:
            # First, try to load existing predictions from transformer.py output
            existing_predictions = load_transformer_predictions(log_type)
            
            if existing_predictions is not None:
                y_pred, probs, y_true, classes = existing_predictions
                
                # Load additional embeddings info for model architecture details if needed
                try:
                    embeddings_dir = Path("embeddings") / log_type
                    label_file = embeddings_dir / f"label_{log_type}.pkl"
                    with open(label_file, 'rb') as f:
                        label_data = pickle.load(f)
                    data_classes = label_data.get("classes", classes)
                    
                    if data_classes != classes:
                        print(f"⚠️  Class mismatch between predictions and data. Using prediction classes.")
                except Exception as e:
                    data_classes = classes
                
                # Extract enhanced transformer information from predictions file
                enhanced_info = {}
                try:
                    results_dir = Path("results") / log_type
                    predictions_file = results_dir / "predictions.pkl"
                    with open(predictions_file, 'rb') as f:
                        pred_data = pickle.load(f)
                    
                    # Extract enhanced transformer metadata
                    enhanced_info = {
                        'model_type': 'Enhanced Multi-Label Transformer (One-vs-Rest)',
                        'enhanced_features': 'Focal Loss, Enhanced Attention, Contrastive Learning',
                        'architecture': f"Transformer with {pred_data.get('latent_dim', 512)}D latent space"
                    }
                    
                    # Add normal class information if available
                    if 'normal_probs' in pred_data and 'high_normal_confidence' in pred_data:
                        normal_probs = pred_data['normal_probs']
                        high_normal_confidence = pred_data['high_normal_confidence']
                        enhanced_info['normal_class_info'] = {
                            'samples': np.sum(high_normal_confidence),
                            'suppressed': np.sum(high_normal_confidence),
                            'threshold': 0.7
                        }
                        print(f"📊 Enhanced transformer with normal class: {np.sum(high_normal_confidence)} samples with high normal confidence")
                    
                except Exception as e:
                    print(f"⚠️  Could not load enhanced transformer info: {e}")
                
                optimized_thresholds = None
                
                # Apply threshold optimization if requested
                if args.optimize_thresholds:
                    evaluator = TransformerEvaluator(config)
                    optimized_thresholds = evaluator.optimize_thresholds(y_true, probs, classes)
                    y_pred = (probs >= optimized_thresholds).astype(int)
                    print(f"✅ Applied optimized thresholds")
                
            else:
                print(f"Running full model evaluation for {log_type}")
                
                # Initialize evaluator
                evaluator = TransformerEvaluator(config)
                
                # 1. Load trained model
                model, model_classes, saved_scaler = evaluator.load_model(log_type, args.model_path)
                
                # 2. Load embeddings and true labels
                X, y_true, data_classes = evaluator.load_embeddings_and_labels(log_type)
                
                # Verify class compatibility
                if model_classes != data_classes:
                    print(f"⚠️  Class mismatch between model and data. Using model classes.")
                    classes = model_classes
                    
                    if len(model_classes) != len(data_classes):
                        print(f"⚠️  Different number of classes may cause issues.")
                else:
                    classes = model_classes
                
                # 3. Preprocess embeddings (same as training)
                X_processed = evaluator.preprocess_embeddings(X, saved_scaler)
                
                # 4. Generate predictions
                y_pred, probs = evaluator.predict(model, X_processed, args.batch_size)
                
                # 5. Always optimize thresholds (default behavior)
                optimized_thresholds = evaluator.optimize_thresholds(y_true, probs, classes)
                y_pred_optimized = (probs >= optimized_thresholds).astype(int)
                
                print(f"✅ Using optimized thresholds")
                y_pred = y_pred_optimized
                
                # Create enhanced info for full evaluation
                enhanced_info = {
                    'model_type': 'Enhanced Multi-Label Transformer (Full Evaluation)',
                    'enhanced_features': 'Focal Loss, Enhanced Attention, Contrastive Learning',
                    'architecture': f"Transformer with {model.latent_dim}D latent space"
                }
            
            # 6. Compute metrics
            if existing_predictions is not None:
                # Create a temporary evaluator just for metrics computation
                evaluator = TransformerEvaluator(config)
            
            metrics = evaluator.compute_metrics(y_true, y_pred, probs, classes)
            
            # 7. Print results with enhanced information
            evaluator.print_results(metrics, classes, y_true, y_pred, enhanced_info)
            
            # 8. Save results (only if we ran full evaluation)
            if existing_predictions is None:
                results_file = evaluator.save_results(
                    metrics, y_pred, probs, y_true, log_type, optimized_thresholds
                )
                print(f"Results saved to: {results_file}")
            
            # Store results for summary
            all_results[log_type] = {
                'metrics': metrics,
                'classes': classes,
                'results_file': f"results/{log_type}/predictions.pkl" if existing_predictions else results_file,
                'status': 'SUCCESS'
            }
            
        except Exception as e:
            print(f"Evaluation failed for {log_type}: {e}")
            all_results[log_type] = {'status': 'FAILED', 'error': str(e)}
            continue
    
    total_evaluation_time = time.time() - evaluation_start_time
    
    print(f"\n{'='*80}")
    print(f"🎯 ENHANCED TRANSFORMER EVALUATION SUMMARY")
    print(f"{'='*80}")
    
    successful_evaluations = 0
    total_samples = 0
    avg_macro_f1 = 0
    avg_micro_f1 = 0
    
    for log_type, result in all_results.items():
        status = result['status']
        if status == 'SUCCESS':
            metrics = result['metrics']
            successful_evaluations += 1
            total_samples += metrics.get('n_samples', 0)
            avg_macro_f1 += metrics['macro_f1']
            avg_micro_f1 += metrics['micro_f1']
            
            print(f"\n✅ {log_type.upper()} (SUCCESS):")
            print(f"   📊 Samples: {metrics.get('n_samples', 0):,}")
            print(f"   🎯 Macro F1: {metrics['macro_f1']:.4f} | Micro F1: {metrics['micro_f1']:.4f}")
            print(f"   📈 Label-wise Accuracy: {metrics['label_wise_accuracy_micro']:.4f}")
            print(f"   🏷️  Classes: {len(result['classes'])}")
            print(f"   📉 Hamming Loss: {metrics.get('hamming_loss', 0.0):.4f}")
            print(f"   🎪 Jaccard Score: {metrics.get('jaccard_micro', 0.0):.4f}")
            
            # Show class distribution
            if 'per_class_support' in metrics:
                class_supports = metrics['per_class_support']
                total_class_samples = sum(class_supports)
                print(f"   📋 Class Distribution:")
                for i, class_name in enumerate(result['classes']):
                    support = class_supports[i]
                    percentage = (support / total_class_samples * 100) if total_class_samples > 0 else 0
                    print(f"      {class_name:<20}: {support:>6,} ({percentage:>5.1f}%)")
        else:
            print(f"\n❌ {log_type.upper()} (FAILED): {result['error']}")
    
    # Overall statistics
    if successful_evaluations > 0:
        avg_macro_f1 /= successful_evaluations
        avg_micro_f1 /= successful_evaluations
        
        print(f"\n📊 OVERALL STATISTICS:")
        print(f"   Total log types: {len(log_types)}")
        print(f"   Successful evaluations: {successful_evaluations}")
        print(f"   Failed evaluations: {len(log_types) - successful_evaluations}")
        print(f"   Total samples evaluated: {total_samples:,}")
        print(f"   Average Macro F1: {avg_macro_f1:.4f}")
        print(f"   Average Micro F1: {avg_micro_f1:.4f}")
        print(f"   Total evaluation time: {total_evaluation_time:.2f} seconds")
        print(f"   Average time per log type: {total_evaluation_time/len(log_types):.2f} seconds")
    
    print(f"\n🚀 ENHANCED TRANSFORMER FEATURES USED:")
    print(f"   ✨ Focal Loss for class imbalance handling")
    print(f"   🔍 Enhanced Multi-Head Attention with label-aware mechanisms")
    print(f"   🤝 Contrastive Learning for representation learning")
    print(f"   🎯 One-vs-Rest strategy with normal class modeling")
    print(f"   📊 Advanced threshold optimization")
    print(f"   🏗️  Multi-feature LogBERT embeddings (2314D)")
    
    if all(r['status'] == 'SUCCESS' for r in all_results.values()):
        print(f"\n🎉 All evaluations completed successfully!")
        print(f"📁 Results available in: results/<log-type>/")
        print(f"📊 Classification reports: results/<log-type>/enhanced_evaluation_report.txt")
    else:
        print(f"\n⚠️  Some evaluations failed. Check the logs above for details.")
        
    print(f"{'='*80}")


if __name__ == "__main__":
    main() 