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
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import warnings

import numpy as np
import torch
from sklearn.metrics import (
    precision_recall_fscore_support, f1_score, accuracy_score, 
    hamming_loss, jaccard_score, precision_score, recall_score,
    classification_report, roc_auc_score, average_precision_score
)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Import from transformer and f-anogan modules
import sys
sys.path.append('.')
from src.transformer import detect_system_resources, SystemConfig
from src.evaluate_transformer import TransformerEvaluator


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
    
    def run_transformer_evaluation(self, log_type: str) -> Optional[Dict]:
        """Run transformer evaluation if not available"""
        
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
    
    def convert_fanogan_to_multilabel(self, fanogan_results: Dict, target_classes: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Convert f-AnoGAN results to multi-label format for comparison"""
        
        if 'predictions_multilabel' in fanogan_results:
            # Use pre-computed multi-label predictions
            y_pred_multilabel = fanogan_results['predictions_multilabel']
            anomaly_scores = fanogan_results['anomaly_scores']
            
            print(f"✅ Using pre-computed f-AnoGAN multi-label predictions")
            return y_pred_multilabel, anomaly_scores
        
        else:
            # Fallback: convert binary anomaly to multi-label
            y_pred_binary = fanogan_results['predictions_binary'] if 'predictions_binary' in fanogan_results else fanogan_results['predictions']
            anomaly_scores = fanogan_results['anomaly_scores']
            
            print(f"🔄 Converting f-AnoGAN binary predictions to multi-label format...")
            
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
    
    def compute_comparative_metrics(self, transformer_results: Dict, fanogan_results: Optional[Dict], 
                                  classes: List[str]) -> Dict[str, Any]:
        """Compute comprehensive comparative metrics"""
        
        # Get transformer predictions and ground truth
        y_true = transformer_results['true_labels']
        transformer_pred = transformer_results['predictions']
        transformer_probs = transformer_results['probabilities']
        
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
            # Convert f-AnoGAN to multi-label format
            fanogan_pred, fanogan_scores = self.convert_fanogan_to_multilabel(fanogan_results, classes)
            
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
        if (comparison_metrics.get('comparison', {}).get('available', False) and 
            comparison_metrics.get('fanogan', {}).get('available', False)):
            comp = comparison_metrics['comparison']
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
        
        elif comparison_metrics.get('fanogan', {}).get('available', False):
            print("ℹ️  Both models available but comparison disabled")
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


def main():
    parser = argparse.ArgumentParser(description="Multi-model evaluation and comparison")
    parser.add_argument("--log-type", type=str, required=True,
                       help="Log type to evaluate (e.g., wp-access, wp-error)")
    parser.add_argument("--compare-only", action="store_true",
                       help="Only compare existing results, don't run new evaluations")
    parser.add_argument("--force-transformer-only", action="store_true",
                       help="Force transformer-only evaluation even if f-AnoGAN results exist")
    
    args = parser.parse_args()
    
    # Detect system configuration
    config = detect_system_resources()
    
    print("🚀 Multi-Model Evaluation and Comparison")
    print("=" * 60)
    print(f"Log type: {args.log_type}")
    print(f"Device: {config.device}")
    print(f"Node: {config.node_name} | Job: {config.job_id}")
    print(f"Mode: {'Compare only' if args.compare_only else 'Full evaluation'}")
    print("")
    
    try:
        # Initialize comparator
        comparator = ModelComparator(config)
        
        # Find available results
        available_results = comparator.find_model_results(args.log_type)
        print(f"📂 Available results:")
        for model_type, results_file in available_results.items():
            status = f"✅ {results_file}" if results_file else "❌ Not found"
            print(f"   {model_type}: {status}")
        print("")
        
        # Load existing results
        loaded_results = comparator.load_model_results(available_results)
        
        # Handle missing transformer results
        if not loaded_results.get('transformer') and not args.compare_only:
            print(f"🔄 Running transformer evaluation...")
            transformer_results = comparator.run_transformer_evaluation(args.log_type)
            if transformer_results:
                loaded_results['transformer'] = transformer_results
        
        # Check if we have transformer results
        if not loaded_results.get('transformer'):
            print(f"❌ No transformer results available and evaluation failed")
            return
        
        # Handle f-AnoGAN results
        if args.force_transformer_only:
            print(f"🎯 Forced transformer-only evaluation")
            loaded_results['fanogan'] = None
        
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
            transformer_results, fanogan_results, classes
        )
        
        # Print results
        comparator.print_comparison_results(comparison_metrics)
        
        # Save results
        saved_files = comparator.save_comparison_results(comparison_metrics, args.log_type)
        
        print(f"\n✅ Evaluation completed for {args.log_type}")
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
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
