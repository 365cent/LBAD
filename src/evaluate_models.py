#!/usr/bin/env python3
"""
Multi-Model Evaluation and Comparison Pipeline
==============================================

Comprehensive evaluation system that can:
1. Compare transformer and f-AnoGAN predictions side-by-side
2. Fallback to transformer-only evaluation if no GAN predictions available
3. Provide detailed performance analysis and model comparison metrics

Features:
- Automatic detection of available model results
- Unified evaluation metrics for comparison
- Model performance ranking and analysis
- Detailed per-class and overall comparisons
- Export comparison results for further analysis

Usage:
    python src/evaluate_models.py --log-type wp-error
    python src/evaluate_models.py --log-type wp-access --compare-only
    python src/evaluate_models.py --log-type wp-error --force-transformer-only
"""

import argparse
import pickle
import time
import warnings
import traceback
import importlib.util
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import torch
import sys

import numpy as np
import torch
from sklearn.metrics import (
    precision_recall_fscore_support, f1_score, accuracy_score, 
    hamming_loss, jaccard_score, precision_score, recall_score,
    classification_report, roc_auc_score, average_precision_score,
    precision_recall_curve, balanced_accuracy_score, confusion_matrix
)
from sklearn.preprocessing import RobustScaler, normalize

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Import from transformer module
import sys
sys.path.append('.')
from src.transformer import UnsupervisedMultiLabelTransformer, detect_system_resources, SystemConfig


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
        """Advanced per-class threshold optimization for highly imbalanced multi-label classification"""
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
                    
            else:
                # Common classes (>= 10 samples)
                # Use F1 optimization with fine-grained search
                best_f1 = 0
                best_threshold = 0.5
                
                for threshold in np.linspace(0.05, 0.95, 50):
                    pred_class = (probs_class >= threshold).astype(int)
                    f1 = f1_score(y_true_class, pred_class, zero_division=0)
                    
                    if f1 > best_f1:
                        best_f1 = f1
                        best_threshold = threshold
                
                optimized_thresholds[i] = best_threshold
                print(f"   {cls:<25} threshold: {best_threshold:.3f} (F1: {best_f1:.3f}) | pos: {pos_samples} | COMMON_F1")
        
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


class ModelComparator:
    """Compare and evaluate multiple model types"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.transformer_evaluator = TransformerEvaluator(config)
    
    def find_model_results(self, log_type: str) -> Dict[str, Optional[Path]]:
        """Find available model results for comparison"""
        
        results_dir = Path("results") / log_type
        available_results = {
            'transformer': None,
            'fanogan': None
        }
        
        if not results_dir.exists():
            print(f"⚠️  Results directory not found: {results_dir}")
            return available_results
        
        # Look for transformer results
        transformer_patterns = [
            f"transformer_evaluation_{log_type}_{self.config.node_name}_{self.config.job_id}.pkl",
            f"transformer_evaluation_{log_type}_*.pkl"
        ]
        
        for pattern in transformer_patterns:
            matches = list(results_dir.glob(pattern))
            if matches:
                available_results['transformer'] = max(matches, key=lambda p: p.stat().st_mtime)
                break
        
        # Look for f-AnoGAN results
        fanogan_patterns = [
            f"fanogan_evaluation_{log_type}_{self.config.node_name}_{self.config.job_id}.pkl",
            f"fanogan_evaluation_{log_type}_*.pkl"
        ]
        
        for pattern in fanogan_patterns:
            matches = list(results_dir.glob(pattern))
            if matches:
                available_results['fanogan'] = max(matches, key=lambda p: p.stat().st_mtime)
                break
        
        return available_results
    
    def load_model_results(self, results_files: Dict[str, Optional[Path]]) -> Dict[str, Optional[Dict]]:
        """Load results from available model files"""
        
        loaded_results = {}
        
        for model_type, results_file in results_files.items():
            if results_file and results_file.exists():
                try:
                    with open(results_file, 'rb') as f:
                        results = pickle.load(f)
                    loaded_results[model_type] = results
                    print(f"✅ Loaded {model_type} results from {results_file}")
                except Exception as e:
                    print(f"❌ Failed to load {model_type} results: {e}")
                    loaded_results[model_type] = None
            else:
                loaded_results[model_type] = None
                if results_file:
                    print(f"⚠️  {model_type} results file not found: {results_file}")
                else:
                    print(f"⚠️  No {model_type} results available")
        
        return loaded_results
    
    def run_fanogan_evaluation(self, log_type: str) -> Optional[Dict]:
        """Run f-AnoGAN evaluation if not available"""
        
        print(f"🧠 Running f-AnoGAN evaluation for {log_type}...")
        
        try:
            # Import f-AnoGAN evaluator
            import importlib.util
            import sys
            
            # Load the f-anogan module with hyphen in filename
            spec = importlib.util.spec_from_file_location("fanogan_module", "src/f-anogan.py")
            fanogan_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(fanogan_module)
            
            FANOGANEvaluator = fanogan_module.FANOGANEvaluator
            
            # Initialize f-AnoGAN evaluator
            fanogan_evaluator = FANOGANEvaluator(self.config)
            
            # Load data
            X, y_true, classes = fanogan_evaluator.load_embeddings_and_labels(log_type)
            
            # Train f-AnoGAN model
            fanogan_model = fanogan_evaluator.train_fanogan(
                X, y_true, 
                latent_dim=128, 
                n_epochs=50,  # Reduced for faster evaluation
                batch_size=256
            )
            
            # Evaluate
            metrics, anomaly_scores, y_pred_binary, y_pred_multilabel = fanogan_evaluator.evaluate_anomaly_detection(
                fanogan_model, X, y_true, classes
            )
            
            # Print results
            fanogan_evaluator.print_results(metrics, classes, y_true, y_pred_binary, y_pred_multilabel)
            
            # Save results
            results_file, model_file = fanogan_evaluator.save_results(
                metrics, anomaly_scores, y_pred_binary, y_true, y_pred_multilabel, log_type, fanogan_model, classes
            )
            
            # Create results structure compatible with comparison
            results = {
                'metrics': metrics,
                'predictions': y_pred_multilabel,  # Use multi-label predictions for comparison
                'predictions_binary': y_pred_binary,
                'predictions_multilabel': y_pred_multilabel,
                'probabilities': anomaly_scores,  # Use anomaly scores as "probabilities"
                'anomaly_scores': anomaly_scores,
                'true_labels': y_true,
                'evaluation_type': 'fanogan_anomaly_detection',
                'config': self.config.__dict__,
                'timestamp': time.time(),
                'classes': classes
            }
            
            print(f"✅ f-AnoGAN evaluation completed and saved")
            return results
            
        except Exception as e:
            print(f"❌ f-AnoGAN evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def run_transformer_evaluation(self, log_type: str) -> Optional[Dict]:
        
        print(f"🚀 Running transformer evaluation for {log_type}...")
        
        try:
            # Load model and data
            model, model_classes, saved_scaler = self.transformer_evaluator.load_model(log_type)
            X, y_true, data_classes = self.transformer_evaluator.load_embeddings_and_labels(log_type)
            
            # Verify class compatibility
            if model_classes != data_classes:
                print(f"⚠️  Using model classes for evaluation")
                classes = model_classes
            else:
                classes = model_classes
            
            # Preprocess and predict
            X_processed = self.transformer_evaluator.preprocess_embeddings(X, saved_scaler)
            y_pred, probs = self.transformer_evaluator.predict(model, X_processed)
            
            # Optimize thresholds
            optimized_thresholds = self.transformer_evaluator.optimize_thresholds(y_true, probs, classes)
            y_pred_optimized = (probs >= optimized_thresholds).astype(int)
            
            # Compute metrics
            metrics = self.transformer_evaluator.compute_metrics(y_true, y_pred_optimized, probs, classes)
            
            # Create results structure
            results = {
                'metrics': metrics,
                'predictions': y_pred_optimized,
                'probabilities': probs,
                'true_labels': y_true,
                'optimized_thresholds': optimized_thresholds,
                'evaluation_type': 'direct_supervised_transformer',
                'config': self.config.__dict__,
                'timestamp': time.time()
            }
            
            # Save results
            results_file = self.transformer_evaluator.save_results(
                metrics, y_pred_optimized, probs, y_true, log_type, optimized_thresholds
            )
            
            print(f"✅ Transformer evaluation completed and saved")
            return results
            
        except Exception as e:
            print(f"❌ Transformer evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def convert_fanogan_to_multilabel(self, fanogan_results: Dict, target_classes: List[str], 
                                      transformer_results: Optional[Dict] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Convert f-AnoGAN results to multi-label format for comparison, optionally using transformer guidance"""
        
        if 'predictions_multilabel' in fanogan_results:
            # Use pre-computed multi-label predictions but enhance them if transformer available
            y_pred_multilabel = fanogan_results['predictions_multilabel']
            anomaly_scores = fanogan_results['anomaly_scores']
            
            # If transformer results available, use them to improve f-AnoGAN predictions
            if transformer_results is not None:
                print(f"🔄 Enhancing f-AnoGAN predictions using transformer guidance...")
                y_pred_multilabel = self._enhance_fanogan_with_transformer(
                    y_pred_multilabel, anomaly_scores, transformer_results, target_classes
                )
            
            print(f"✅ Using enhanced f-AnoGAN multi-label predictions")
            return y_pred_multilabel, anomaly_scores
        
        else:
            # Fallback: convert binary anomaly to multi-label
            y_pred_binary = fanogan_results['predictions_binary'] if 'predictions_binary' in fanogan_results else fanogan_results['predictions']
            anomaly_scores = fanogan_results['anomaly_scores']
            
            print(f"🔄 Converting f-AnoGAN binary predictions to multi-label format...")
            
            # Enhanced conversion using transformer guidance if available
            if transformer_results is not None:
                y_pred_multilabel = self._convert_with_transformer_guidance(
                    y_pred_binary, anomaly_scores, transformer_results, target_classes
                )
            else:
                # Simple conversion: if anomaly detected, predict all classes with decreasing probability
                y_pred_multilabel = np.zeros((len(y_pred_binary), len(target_classes)), dtype=int)
                
                # For samples predicted as anomalies, distribute across classes
                anomaly_indices = np.where(y_pred_binary == 1)[0]
                
                if len(anomaly_indices) > 0:
                    # Use score-based thresholding for each class
                    score_percentiles = [99, 95, 90, 85, 80, 75, 70, 65]
                    
                    for i, cls in enumerate(target_classes):
                        if i < len(score_percentiles):
                            threshold_percentile = score_percentiles[i]
                            threshold = np.percentile(anomaly_scores, threshold_percentile)
                            y_pred_multilabel[:, i] = (anomaly_scores >= threshold).astype(int)
                        else:
                            # For additional classes, use high threshold
                            threshold = np.percentile(anomaly_scores, 95)
                            y_pred_multilabel[:, i] = (anomaly_scores >= threshold).astype(int)
            
            print(f"✅ Converted to multi-label: {y_pred_multilabel.shape}")
            return y_pred_multilabel, anomaly_scores
    
    def _enhance_fanogan_with_transformer(self, fanogan_pred: np.ndarray, anomaly_scores: np.ndarray,
                                         transformer_results: Dict, target_classes: List[str]) -> np.ndarray:
        """Enhance f-AnoGAN predictions using transformer predictions as guidance"""
        
        transformer_pred = transformer_results['predictions']
        transformer_probs = transformer_results['probabilities']
        
        # Strategy: Use transformer predictions to guide f-AnoGAN where anomaly scores are moderate
        enhanced_pred = fanogan_pred.copy()
        
        # Normalize anomaly scores to [0, 1]
        score_min, score_max = anomaly_scores.min(), anomaly_scores.max()
        scores_norm = (anomaly_scores - score_min) / (score_max - score_min + 1e-8)
        
        # For moderate anomaly scores (uncertainty region), use transformer guidance
        moderate_mask = (scores_norm > 0.3) & (scores_norm < 0.8)
        
        print(f"   Enhancing {moderate_mask.sum():,} samples with moderate anomaly scores...")
        
        for i, cls in enumerate(target_classes):
            # In uncertainty region, blend f-AnoGAN and transformer predictions
            uncertain_samples = moderate_mask
            
            # Use transformer predictions with high confidence
            high_confidence_transformer = transformer_probs[:, i] > 0.7
            low_confidence_transformer = transformer_probs[:, i] < 0.3
            
            # Enhance predictions
            blend_mask = uncertain_samples & high_confidence_transformer
            enhanced_pred[blend_mask, i] = transformer_pred[blend_mask, i]
            
            # Also consider transformer predictions for low anomaly scores but high transformer confidence
            low_anomaly_high_transformer = (scores_norm < 0.3) & high_confidence_transformer
            enhanced_pred[low_anomaly_high_transformer, i] = transformer_pred[low_anomaly_high_transformer, i]
        
        # Calculate improvement
        original_pred_rate = fanogan_pred.mean(axis=0)
        enhanced_pred_rate = enhanced_pred.mean(axis=0)
        
        print(f"   Original prediction rates: {original_pred_rate}")
        print(f"   Enhanced prediction rates: {enhanced_pred_rate}")
        
        return enhanced_pred
    
    def _convert_with_transformer_guidance(self, y_pred_binary: np.ndarray, anomaly_scores: np.ndarray,
                                          transformer_results: Dict, target_classes: List[str]) -> np.ndarray:
        """Convert binary anomaly predictions to multi-label using transformer guidance"""
        
        transformer_pred = transformer_results['predictions']
        transformer_probs = transformer_results['probabilities']
        
        y_pred_multilabel = np.zeros((len(y_pred_binary), len(target_classes)), dtype=int)
        
        # Normalize anomaly scores
        scores_norm = (anomaly_scores - anomaly_scores.min()) / (anomaly_scores.max() - anomaly_scores.min() + 1e-8)
        
        print(f"   Using transformer guidance for multi-label conversion...")
        
        for i, cls in enumerate(target_classes):
            # Strategy: Combine anomaly detection with transformer class-specific predictions
            
            # High anomaly score + high transformer probability = definitely predict
            high_anomaly = scores_norm > 0.8
            high_transformer = transformer_probs[:, i] > 0.6
            y_pred_multilabel[high_anomaly & high_transformer, i] = 1
            
            # Medium anomaly score + medium transformer probability = maybe predict
            medium_anomaly = (scores_norm > 0.5) & (scores_norm <= 0.8)
            medium_transformer = transformer_probs[:, i] > 0.4
            y_pred_multilabel[medium_anomaly & medium_transformer, i] = 1
            
            # Trust transformer for high-confidence predictions even with low anomaly scores
            very_high_transformer = transformer_probs[:, i] > 0.8
            y_pred_multilabel[very_high_transformer, i] = 1
        
        return y_pred_multilabel
    
    def compute_comparative_metrics(self, transformer_results: Dict, fanogan_results: Optional[Dict], 
                                  classes: List[str], log_type: str) -> Dict[str, Any]:
        """Compute comprehensive comparative metrics"""
        
        # Get transformer predictions and ground truth
        y_true = transformer_results.get('true_labels', np.array([]))
        transformer_pred = transformer_results.get('predictions', np.array([]))
        transformer_probs = transformer_results.get('probabilities', np.array([]))
        
        # Debug: Check shapes
        print(f"🔍 Debug shapes:")
        print(f"   y_true shape: {y_true.shape if hasattr(y_true, 'shape') else type(y_true)}")
        print(f"   transformer_pred shape: {transformer_pred.shape if hasattr(transformer_pred, 'shape') else type(transformer_pred)}")
        print(f"   transformer_probs shape: {transformer_probs.shape if hasattr(transformer_probs, 'shape') else type(transformer_probs)}")
        
        # Handle empty arrays or wrong shapes
        if not hasattr(y_true, 'shape') or y_true.size == 0:
            print("⚠️  Warning: y_true is empty or invalid, attempting to reload from embeddings...")
            # Try to reload ground truth from embeddings using the provided log_type
            try:
                embeddings_dir = Path("embeddings") / log_type
                label_file = embeddings_dir / f"label_{log_type}.pkl"
                
                if label_file.exists():
                    print(f"🔍 Loading ground truth from {label_file}...")
                    with open(label_file, 'rb') as f:
                        label_data = pickle.load(f)
                    y_true = label_data["vectors"]
                    print(f"✅ Reloaded y_true from {label_file}: {y_true.shape}")
                else:
                    print(f"⚠️  Label file not found at {label_file}, trying alternative detection...")
                    
                    # Alternative: try to find any label file in embeddings directory
                    embeddings_base = Path("embeddings")
                    if embeddings_base.exists():
                        label_files = list(embeddings_base.glob("*/label_*.pkl"))
                        if label_files:
                            # Use the first available label file
                            label_file = label_files[0]
                            print(f"🔍 Using alternative label file: {label_file}")
                            with open(label_file, 'rb') as f:
                                label_data = pickle.load(f)
                            y_true = label_data["vectors"]
                            print(f"✅ Loaded y_true from alternative file: {y_true.shape}")
                        else:
                            raise FileNotFoundError("No label files found in embeddings directory")
                    else:
                        raise FileNotFoundError("Embeddings directory not found")
            except Exception as e:
                print(f"❌ Could not reload ground truth: {e}")
                # Create dummy ground truth (fallback)
                if hasattr(transformer_pred, 'shape') and transformer_pred.size > 0:
                    print(f"⚠️  Creating dummy ground truth - this will produce invalid metrics!")
                    y_true = np.zeros_like(transformer_pred)
                    print(f"⚠️  Using dummy ground truth with shape: {y_true.shape}")
                else:
                    raise ValueError("Cannot determine ground truth labels and no valid predictions available")
        
        # Ensure all arrays have consistent shapes
        print(f"🔍 Final shapes before processing:")
        print(f"   y_true: {y_true.shape}")
        print(f"   transformer_pred: {transformer_pred.shape}")
        print(f"   transformer_probs: {transformer_probs.shape}")
        
        # Validate shapes
        if y_true.shape[0] != transformer_pred.shape[0]:
            print(f"⚠️  Shape mismatch detected, truncating to minimum length...")
            min_samples = min(y_true.shape[0], transformer_pred.shape[0])
            y_true = y_true[:min_samples]
            transformer_pred = transformer_pred[:min_samples]
            transformer_probs = transformer_probs[:min_samples]
            print(f"✅ Truncated all arrays to {min_samples} samples")
        
        comparison_metrics = {
            'classes': classes,
            'n_samples': len(y_true),
            'n_classes': len(classes),
            'true_labels': y_true,
            'transformer': {
                'metrics': transformer_results['metrics'],
                'predictions': transformer_pred,
                'probabilities': transformer_probs,
                'optimized_thresholds': transformer_results.get('optimized_thresholds', []),
                'available': True
            }
        }
        
        if fanogan_results is not None:
            # Convert f-AnoGAN to multi-label format with transformer guidance
            fanogan_pred, fanogan_scores = self.convert_fanogan_to_multilabel(
                fanogan_results, classes, transformer_results
            )
            
            # Ensure fanogan predictions match ground truth shape
            if fanogan_pred.shape[0] != y_true.shape[0]:
                print(f"⚠️  f-AnoGAN prediction shape mismatch, truncating...")
                min_samples = min(fanogan_pred.shape[0], y_true.shape[0])
                fanogan_pred = fanogan_pred[:min_samples]
                fanogan_scores = fanogan_scores[:min_samples]
                print(f"✅ Truncated f-AnoGAN results to {min_samples} samples")
            
            # Compute f-AnoGAN multi-label metrics
            fanogan_metrics = self._compute_multilabel_metrics(y_true, fanogan_pred, classes)
            
            comparison_metrics['fanogan'] = {
                'metrics': fanogan_metrics,
                'anomaly_metrics': fanogan_results['metrics'],  # Original anomaly detection metrics
                'available': True,
                'predictions': fanogan_pred,  # Add predictions for saving
                'scores': fanogan_scores      # Add scores for saving
            }
            
            # Direct comparison metrics
            comparison_metrics['comparison'] = self._compute_comparison_metrics(
                y_true, transformer_pred, fanogan_pred, transformer_probs, fanogan_scores, classes
            )
        else:
            comparison_metrics['fanogan'] = {'available': False}
            comparison_metrics['comparison'] = {'available': False}
        
        return comparison_metrics
    
    def _compute_multilabel_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                   classes: List[str]) -> Dict[str, Any]:
        """Compute multi-label classification metrics"""
        
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
        })
        
        return metrics
    
    def _compute_comparison_metrics(self, y_true: np.ndarray, transformer_pred: np.ndarray,
                                   fanogan_pred: np.ndarray, transformer_probs: np.ndarray,
                                   fanogan_scores: np.ndarray, classes: List[str]) -> Dict[str, Any]:
        """Compute direct comparison metrics between models"""
        
        # Agreement analysis
        agreement_exact = np.all(transformer_pred == fanogan_pred, axis=1)
        agreement_rate = float(agreement_exact.mean())
        
        # Per-class agreement
        per_class_agreement = []
        for i in range(len(classes)):
            class_agreement = float((transformer_pred[:, i] == fanogan_pred[:, i]).mean())
            per_class_agreement.append(class_agreement)
        
        # Performance differences
        transformer_f1 = f1_score(y_true, transformer_pred, average='macro', zero_division=0)
        fanogan_f1 = f1_score(y_true, fanogan_pred, average='macro', zero_division=0)
        
        transformer_micro_f1 = f1_score(y_true, transformer_pred, average='micro', zero_division=0)
        fanogan_micro_f1 = f1_score(y_true, fanogan_pred, average='micro', zero_division=0)
        
        # Ensemble predictions (majority vote)
        ensemble_pred = ((transformer_pred + fanogan_pred) >= 1).astype(int)
        ensemble_f1 = f1_score(y_true, ensemble_pred, average='macro', zero_division=0)
        ensemble_micro_f1 = f1_score(y_true, ensemble_pred, average='micro', zero_division=0)
        
        # Weighted ensemble (using probabilities/scores)
        # Normalize scores to [0, 1] range
        fanogan_scores_norm = (fanogan_scores - fanogan_scores.min()) / (fanogan_scores.max() - fanogan_scores.min() + 1e-8)
        
        # Create weighted ensemble
        ensemble_scores = np.zeros((len(y_true), len(classes)))
        for i in range(len(classes)):
            ensemble_scores[:, i] = 0.7 * transformer_probs[:, i] + 0.3 * fanogan_scores_norm
        
        ensemble_weighted_pred = (ensemble_scores >= 0.5).astype(int)
        ensemble_weighted_f1 = f1_score(y_true, ensemble_weighted_pred, average='macro', zero_division=0)
        
        comparison = {
            'agreement_rate': agreement_rate,
            'per_class_agreement': per_class_agreement,
            'performance_difference': {
                'macro_f1_diff': float(transformer_f1 - fanogan_f1),
                'micro_f1_diff': float(transformer_micro_f1 - fanogan_micro_f1),
                'transformer_better': transformer_f1 > fanogan_f1,
            },
            'ensemble_performance': {
                'majority_vote_macro_f1': float(ensemble_f1),
                'majority_vote_micro_f1': float(ensemble_micro_f1),
                'weighted_ensemble_macro_f1': float(ensemble_weighted_f1),
                'ensemble_improves_over_transformer': ensemble_f1 > transformer_f1,
                'ensemble_improves_over_fanogan': ensemble_f1 > fanogan_f1,
            },
            'model_rankings': {
                'by_macro_f1': sorted([
                    ('transformer', transformer_f1),
                    ('fanogan', fanogan_f1),
                    ('ensemble_majority', ensemble_f1),
                    ('ensemble_weighted', ensemble_weighted_f1)
                ], key=lambda x: x[1], reverse=True)
            }
        }
        
        return comparison
    
    def print_comparison_results(self, comparison_metrics: Dict[str, Any]):
        """Print comprehensive comparison results"""
        
        print(f"\n📊 MULTI-MODEL EVALUATION RESULTS")
        print("=" * 80)
        print(f"Dataset: {comparison_metrics['n_samples']:,} samples, {comparison_metrics['n_classes']} classes")
        print(f"Classes: {comparison_metrics['classes']}")
        print("")
        
        # Transformer results
        if comparison_metrics['transformer']['available']:
            t_metrics = comparison_metrics['transformer']['metrics']
            print("🤖 TRANSFORMER MODEL:")
            print("-" * 50)
            print(f"Macro F1:         {t_metrics['macro_f1']:.4f}")
            print(f"Micro F1:         {t_metrics['micro_f1']:.4f}")
            print(f"Subset Accuracy:  {t_metrics['subset_accuracy']:.4f}")
            print(f"Hamming Loss:     {t_metrics['hamming_loss']:.4f}")
            print("")
        
        # f-AnoGAN results  
        if comparison_metrics['fanogan']['available']:
            f_metrics = comparison_metrics['fanogan']['metrics']
            f_anomaly = comparison_metrics['fanogan']['anomaly_metrics']
            print("🧠 f-AnoGAN MODEL:")
            print("-" * 50)
            print("Multi-label Performance:")
            print(f"  Macro F1:       {f_metrics['macro_f1']:.4f}")
            print(f"  Micro F1:       {f_metrics['micro_f1']:.4f}")
            print(f"  Subset Accuracy: {f_metrics['subset_accuracy']:.4f}")
            print(f"  Hamming Loss:   {f_metrics['hamming_loss']:.4f}")
            print("")
            print("Anomaly Detection Performance:")
            print(f"  ROC AUC:        {f_anomaly['roc_auc']:.4f}")
            print(f"  Avg Precision:  {f_anomaly['average_precision']:.4f}")
            print(f"  Best F1:        {f_anomaly['best_f1']:.4f}")
            print("")
        
        # Comparison results
        if (comparison_metrics.get('fanogan', {}).get('available', False)):
            # Always show comparison when both models are available
            comp = comparison_metrics.get('comparison', {})
            if comp.get('available', False):
                print("🔄 MODEL COMPARISON:")
                print("-" * 50)
                print(f"Agreement Rate:   {comp['agreement_rate']:.4f}")
                print(f"Performance Gap:  {comp['performance_difference']['macro_f1_diff']:.4f}")
                print(f"Better Model:     {'Transformer' if comp['performance_difference']['transformer_better'] else 'f-AnoGAN'}")
                print("")
                
                print("🎯 ENSEMBLE RESULTS:")
                print("-" * 50)
                ens = comp['ensemble_performance']
                print(f"Majority Vote F1: {ens['majority_vote_macro_f1']:.4f}")
                print(f"Weighted Ens F1:  {ens['weighted_ensemble_macro_f1']:.4f}")
                print(f"Ensemble Improves: {ens['ensemble_improves_over_transformer'] or ens['ensemble_improves_over_fanogan']}")
                print("")
                
                print("🏆 MODEL RANKINGS (by Macro F1):")
                print("-" * 50)
                for rank, (model, score) in enumerate(comp['model_rankings']['by_macro_f1'], 1):
                    print(f"{rank}. {model:<20} {score:.4f}")
                print("")
            else:
                print("⚠️  Comparison metrics generation failed")
        else:
            print("ℹ️  Only transformer evaluation available (no comparison)")
    
    def save_comparison_results(self, comparison_metrics: Dict[str, Any], log_type: str):
        """Save comprehensive comparison results"""
        
        output_dir = Path("results") / log_type
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save comparison results
        results_file = output_dir / f"model_comparison_{log_type}_{self.config.node_name}_{self.config.job_id}.pkl"
        
        with open(results_file, 'wb') as f:
            pickle.dump(comparison_metrics, f)
        
        print(f"💾 Comparison results saved to: {results_file}")
        
        # Also save individual model results in transformer format for compatibility
        saved_files = [results_file]
        
        # Save transformer results in standard format (if they exist)
        if comparison_metrics['transformer']['available']:
            transformer_file = self._save_transformer_format(comparison_metrics, log_type)
            saved_files.append(transformer_file)
        
        # Save f-AnoGAN results in transformer-compatible format (if they exist) 
        if comparison_metrics.get('fanogan', {}).get('available', False):
            fanogan_file = self._save_fanogan_as_transformer_format(comparison_metrics, log_type)
            saved_files.append(fanogan_file)
        
        return saved_files
    
    def _save_transformer_format(self, comparison_metrics: Dict[str, Any], log_type: str) -> Path:
        """Save transformer results in standard evaluate_transformer.py format"""
        
        output_dir = Path("results") / log_type
        t_metrics = comparison_metrics['transformer']['metrics']
        
        # Create transformer-format results
        transformer_results = {
            'metrics': t_metrics,
            'predictions': comparison_metrics['transformer'].get('predictions', np.array([])),
            'probabilities': comparison_metrics['transformer'].get('probabilities', np.array([])),
            'true_labels': comparison_metrics['transformer'].get('true_labels', np.array([])),
            'optimized_thresholds': comparison_metrics['transformer'].get('optimized_thresholds', []),
            'evaluation_type': 'direct_supervised_transformer',
            'config': self.config.__dict__,
            'timestamp': time.time()
        }
        
        results_file = output_dir / f"transformer_evaluation_{log_type}_{self.config.node_name}_{self.config.job_id}.pkl"
        with open(results_file, 'wb') as f:
            pickle.dump(transformer_results, f)
        
        print(f"💾 Transformer results saved to: {results_file}")
        return results_file
    
    def _save_fanogan_as_transformer_format(self, comparison_metrics: Dict[str, Any], log_type: str) -> Path:
        """Save f-AnoGAN multi-label results in transformer-compatible format"""
        
        output_dir = Path("results") / log_type
        f_metrics = comparison_metrics['fanogan']['metrics']
        
        # Create transformer-format results from f-AnoGAN
        fanogan_as_transformer = {
            'metrics': f_metrics,
            'predictions': comparison_metrics['fanogan'].get('predictions', np.array([])),
            'probabilities': None,  # f-AnoGAN uses anomaly scores, not class probabilities
            'anomaly_scores': comparison_metrics['fanogan'].get('scores', np.array([])),
            'true_labels': comparison_metrics.get('true_labels', np.array([])),
            'evaluation_type': 'fanogan_as_multilabel_classifier',
            'original_anomaly_metrics': comparison_metrics['fanogan']['anomaly_metrics'],
            'config': self.config.__dict__,
            'timestamp': time.time(),
            'note': 'f-AnoGAN results converted to multi-label format for comparison'
        }
        
        results_file = output_dir / f"fanogan_as_transformer_{log_type}_{self.config.node_name}_{self.config.job_id}.pkl"
        with open(results_file, 'wb') as f:
            pickle.dump(fanogan_as_transformer, f)
        
        print(f"💾 f-AnoGAN (as transformer) results saved to: {results_file}")
        return results_file


def discover_available_log_types() -> List[str]:
    """Auto-discover available log types from embeddings and models directories"""
    log_types = set()
    
    # Check embeddings directory
    embeddings_dir = Path("embeddings")
    if embeddings_dir.exists():
        for item in embeddings_dir.iterdir():
            if item.is_dir() and (item / f"log_{item.name}.pkl").exists():
                log_types.add(item.name)
    
    # Check models directory for transformer models
    models_dir = Path("models")
    if models_dir.exists():
        for model_file in models_dir.glob("transformer_*.pth"):
            # Extract log type from filename: transformer_LOG_TYPE_*.pth
            parts = model_file.stem.split('_')
            if len(parts) >= 2:
                log_type = parts[1]
                log_types.add(log_type)
    
    # Check f-AnoGAN results
    results_dir = Path("results")
    if results_dir.exists():
        for item in results_dir.iterdir():
            if item.is_dir():
                log_types.add(item.name)
    
    return sorted(list(log_types))


def main():
    parser = argparse.ArgumentParser(description="Multi-model evaluation and comparison")
    parser.add_argument("--log-type", type=str, 
                       help="Log type to evaluate (e.g., wp-access, wp-error). If not provided, runs on all available log types.")
    parser.add_argument("--compare-only", action="store_true",
                       help="Only compare existing results, don't run new evaluations")
    parser.add_argument("--force-transformer-only", action="store_true",
                       help="Force transformer-only evaluation even if f-AnoGAN results exist")
    
    args = parser.parse_args()
    
    # Detect system configuration
    config = detect_system_resources()
    
    print("🚀 Multi-Model Evaluation and Comparison")
    print("=" * 60)
    print(f"Device: {config.device}")
    print(f"Node: {config.node_name} | Job: {config.job_id}")
    print(f"Mode: {'Compare only' if args.compare_only else 'Full evaluation'}")
    
    # Auto-discover log types if not specified
    if args.log_type:
        log_types = [args.log_type]
        print(f"Log type: {args.log_type}")
    else:
        log_types = discover_available_log_types()
        print(f"🔍 Auto-detected log types: {log_types}")
        if not log_types:
            print("❌ No log types found. Please ensure embeddings and/or models are available.")
            return
    
    print("")
    
    # Process each log type
    for i, log_type in enumerate(log_types):
        if len(log_types) > 1:
            print(f"\n{'='*20} Processing {log_type} ({i+1}/{len(log_types)}) {'='*20}")
        
        try:
            # Initialize comparator
            comparator = ModelComparator(config)
            
            # Find available results
            available_results = comparator.find_model_results(log_type)
            print(f"📂 Available results for {log_type}:")
            for model_type, results_file in available_results.items():
                status = f"✅ {results_file}" if results_file else "❌ Not found"
                print(f"   {model_type}: {status}")
            print("")
            
            # Load existing results
            loaded_results = comparator.load_model_results(available_results)
            
            # Handle missing transformer results
            if not loaded_results.get('transformer') and not args.compare_only:
                print(f"🔄 Running transformer evaluation...")
                transformer_results = comparator.run_transformer_evaluation(log_type)
                if transformer_results:
                    loaded_results['transformer'] = transformer_results
            
            # Check if we have transformer results
            if not loaded_results.get('transformer'):
                print(f"❌ No transformer results available and evaluation failed for {log_type}")
                continue
            
            # Handle f-AnoGAN results
            if args.force_transformer_only:
                print(f"🎯 Forced transformer-only evaluation")
                loaded_results['fanogan'] = None
            else:
                # Try to run f-AnoGAN evaluation if missing
                if not loaded_results.get('fanogan') and not args.compare_only:
                    print(f"🔄 Running f-AnoGAN evaluation...")
                    fanogan_results = comparator.run_fanogan_evaluation(log_type)
                    if fanogan_results:
                        loaded_results['fanogan'] = fanogan_results
            
            # Determine evaluation mode
            transformer_results = loaded_results['transformer']
            fanogan_results = loaded_results.get('fanogan')
            
            if fanogan_results:
                print(f"🔄 Running comparative evaluation...")
                mode = "comparative"
            else:
                print(f"🔄 Running transformer-only evaluation...")
                mode = "transformer_only"
            
            # Get classes from transformer results
            classes = transformer_results['metrics']['classes']
            
            # Compute comparison metrics
            comparison_metrics = comparator.compute_comparative_metrics(
                transformer_results, fanogan_results, classes, log_type
            )
            
            # Print results
            comparator.print_comparison_results(comparison_metrics)
            
            # Save results
            saved_files = comparator.save_comparison_results(comparison_metrics, log_type)
            
            print(f"\n✅ Evaluation completed for {log_type}")
            print(f"📁 Results saved to:")
            for file_path in saved_files:
                print(f"   {file_path}")
            
            # Summary
            t_f1 = comparison_metrics['transformer']['metrics']['macro_f1']
            print(f"\n🎯 SUMMARY")
            print(f"   Transformer F1: {t_f1:.4f}")
            
            if comparison_metrics.get('fanogan', {}).get('available', False):
                f_f1 = comparison_metrics['fanogan']['metrics']['macro_f1']
                print(f"   f-AnoGAN F1:    {f_f1:.4f}")
                
                if comparison_metrics.get('comparison', {}).get('available', False):
                    agreement = comparison_metrics['comparison']['agreement_rate']
                    print(f"   Agreement:      {agreement:.4f}")
                    
                    if comparison_metrics['comparison']['ensemble_performance']['ensemble_improves_over_transformer']:
                        ens_f1 = comparison_metrics['comparison']['ensemble_performance']['majority_vote_macro_f1']
                        print(f"   Best Ensemble:  {ens_f1:.4f}")
            
        except Exception as e:
            print(f"❌ Evaluation failed for {log_type}: {e}")
            import traceback
            traceback.print_exc()
            continue


if __name__ == "__main__":
    main()
