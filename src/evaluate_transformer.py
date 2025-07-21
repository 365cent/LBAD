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
    
    def load_model(self, log_type: str, model_path: Optional[str] = None) -> Tuple[UnsupervisedMultiLabelTransformer, List[str]]:
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
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        
        # Extract model configuration
        classes = ckpt['classes']
        
        # Determine input dimension from checkpoint
        input_dim = None
        for key, tensor in ckpt['model_state_dict'].items():
            if 'input_projection' in key and 'weight' in key:
                input_dim = tensor.shape[1]
                break
        
        if input_dim is None:
            raise ValueError("Could not determine input dimension from model")
        
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
        
        return model, classes
    
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
        
        print(f"✅ Loaded {len(X):,} samples with {X.shape[1]}D embeddings")
        print(f"📊 Labels: {y_true.shape} for {len(classes)} classes")
        
        return X, y_true, classes
    
    def preprocess_embeddings(self, X: np.ndarray) -> np.ndarray:
        """Apply same preprocessing as training (RobustScaler + L2 normalization)"""
        print(f"🔄 Preprocessing embeddings...")
        
        # Apply RobustScaler (using fit_transform since we're evaluating)
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        
        # L2 normalization
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
        """Optimize per-class thresholds for F1 score"""
        print(f"🎯 Optimizing per-class thresholds...")
        
        best_thresh = np.zeros(len(classes))
        
        for i, cls in enumerate(classes):
            best_f1, best_t = 0, 0.5
            
            # Test thresholds from 0.1 to 0.9
            for t in np.linspace(0.1, 0.9, 17):
                pred = (probs[:, i] >= t).astype(int)
                f1 = f1_score(y_true[:, i], pred, zero_division=0)
                if f1 > best_f1:
                    best_f1, best_t = f1, t
            
            best_thresh[i] = best_t
            print(f"   {cls:<25} threshold: {best_t:.3f} (F1: {best_f1:.3f})")
        
        return best_thresh
    
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
    parser.add_argument("--optimize-thresholds", action="store_true",
                       help="Optimize per-class thresholds for F1 score")
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
    print(f"Threshold optimization: {'Enabled' if args.optimize_thresholds else 'Disabled (0.5 default)'}")
    print("")
    
    try:
        # Initialize evaluator
        evaluator = TransformerEvaluator(config)
        
        # 1. Load trained model
        model, model_classes = evaluator.load_model(args.log_type, args.model_path)
        
        # 2. Load embeddings and true labels
        X, y_true, data_classes = evaluator.load_embeddings_and_labels(args.log_type)
        
        # Verify class compatibility
        if model_classes != data_classes:
            print(f"⚠️  Class mismatch between model and data")
            print(f"   Model classes: {model_classes}")
            print(f"   Data classes: {data_classes}")
            print(f"   Using model classes for evaluation")
            classes = model_classes
        else:
            classes = model_classes
        
        # 3. Preprocess embeddings (same as training)
        X_processed = evaluator.preprocess_embeddings(X)
        
        # 4. Generate predictions
        y_pred, probs = evaluator.predict(model, X_processed, args.batch_size)
        
        # 5. Optimize thresholds if requested
        optimized_thresholds = None
        if args.optimize_thresholds:
            optimized_thresholds = evaluator.optimize_thresholds(y_true, probs, classes)
            y_pred_optimized = (probs >= optimized_thresholds).astype(int)
            
            print(f"\n🎯 Using optimized thresholds")
            y_pred = y_pred_optimized
        else:
            print(f"\n📊 Using default 0.5 threshold")
        
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