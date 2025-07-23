#!/usr/bin/env python3
"""
Universal Model Evaluation Pipeline
==================================

This is a compatibility wrapper that redirects to the unified evaluate_models.py
Maintains backward compatibility with original evaluate_transformer.py usage.

Usage:
    python src/evaluate_transformer.py --log-type wp-error
    
All functionality has been moved to evaluate_models.py for better organization
and to avoid circular import issues.
"""

import sys
import subprocess
from pathlib import Path

def main():
    """Redirect to evaluate_models.py with transformer-only flag"""
    
    # Get command line arguments (excluding script name)
    args = sys.argv[1:]
    
    # Add transformer-only flag to maintain original behavior
    if '--compare-models' not in args and '--force-transformer-only' not in args:
        args.append('--force-transformer-only')
    
    # Build command to run evaluate_models.py
    script_path = Path(__file__).parent / 'evaluate_models.py'
    cmd = [sys.executable, str(script_path)] + args
    
    print("🔄 Redirecting to unified evaluation system...")
    print(f"Running: {' '.join(cmd)}")
    print("")
    
    # Execute the command
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"❌ Evaluation failed with exit code {e.returncode}")
        return e.returncode
    except KeyboardInterrupt:
        print("⚠️  Evaluation interrupted by user")
        return 130

if __name__ == "__main__":
    exit(main())

import argparse
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import warnings

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import normalize
from sklearn.metrics import (
    precision_recall_fscore_support, f1_score, accuracy_score, 
    hamming_loss, jaccard_score, precision_score, recall_score,
    classification_report, roc_auc_score, average_precision_score
)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Import from transformer module
import sys
sys.path.append('.')
from src.transformer import UnsupervisedMultiLabelTransformer, SystemConfig, detect_system_resources

# Import the new model comparator
from src.evaluate_models import ModelComparator


class UniversalEvaluator:
    """Universal evaluator that can handle transformer-only or multi-model evaluation"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.comparator = ModelComparator(config)
    
    def evaluate_transformer_only(self, log_type: str, model_path: Optional[str] = None, 
                                 batch_size: int = 64) -> Dict[str, Any]:
        """Evaluate transformer model only (original behavior)"""
        
        print("🤖 Transformer-Only Evaluation Mode")
        print("-" * 50)
        
        # Use the transformer evaluator from the comparator
        transformer_evaluator = self.comparator.transformer_evaluator
        
        # Load model and data
        model, model_classes, saved_scaler = transformer_evaluator.load_model(log_type, model_path)
        X, y_true, data_classes = transformer_evaluator.load_embeddings_and_labels(log_type)
        
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
        
        # Preprocess embeddings (same as training)
        X_processed = transformer_evaluator.preprocess_embeddings(X, saved_scaler)
        
        # Generate predictions
        y_pred, probs = transformer_evaluator.predict(model, X_processed, batch_size)
        
        # Always optimize thresholds (default behavior)
        optimized_thresholds = transformer_evaluator.optimize_thresholds(y_true, probs, classes)
        y_pred_optimized = (probs >= optimized_thresholds).astype(int)
        
        print(f"\n🎯 Using advanced optimized thresholds")
        y_pred = y_pred_optimized
        
        # Compute metrics
        metrics = transformer_evaluator.compute_metrics(y_true, y_pred, probs, classes)
        
        # Print results
        transformer_evaluator.print_results(metrics, classes, y_true, y_pred)
        
        # Save results
        results_file = transformer_evaluator.save_results(
            metrics, y_pred, probs, y_true, log_type, optimized_thresholds
        )
        
        return {
            'metrics': metrics,
            'predictions': y_pred,
            'probabilities': probs,
            'true_labels': y_true,
            'optimized_thresholds': optimized_thresholds,
            'results_file': results_file,
            'classes': classes
        }
    
    def evaluate_multi_model(self, log_type: str, compare_only: bool = False, 
                           force_transformer_only: bool = False) -> Dict[str, Any]:
        """Evaluate multiple models with comparison (new behavior)"""
        
        print("🔄 Multi-Model Evaluation Mode")
        print("-" * 50)
        
        # Find available results
        available_results = self.comparator.find_model_results(log_type)
        print(f"📂 Available results:")
        for model_type, results_file in available_results.items():
            status = f"✅ {results_file}" if results_file else "❌ Not found"
            print(f"   {model_type}: {status}")
        print("")
        
        # Load existing results
        loaded_results = self.comparator.load_model_results(available_results)
        
        # Handle missing transformer results
        if not loaded_results.get('transformer') and not compare_only:
            print(f"🔄 Running transformer evaluation...")
            transformer_results = self.comparator.run_transformer_evaluation(log_type)
            if transformer_results:
                loaded_results['transformer'] = transformer_results
        
        # Check if we have transformer results
        if not loaded_results.get('transformer'):
            print(f"❌ No transformer results available and evaluation failed")
            return {}
        
        # Handle f-AnoGAN results
        if force_transformer_only:
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
        comparison_metrics = self.comparator.compute_comparative_metrics(
            transformer_results, fanogan_results, classes
        )
        
        # Print results
        self.comparator.print_comparison_results(comparison_metrics)
        
        # Save results
        saved_files = self.comparator.save_comparison_results(comparison_metrics, log_type)
        
        return {
            'comparison_metrics': comparison_metrics,
            'saved_files': saved_files,
            'mode': mode
        }


def main():
    parser = argparse.ArgumentParser(description="Universal model evaluation pipeline")
    parser.add_argument("--log-type", type=str, required=True,
                       help="Log type to evaluate (e.g., wp-access, wp-error)")
    parser.add_argument("--model-path", type=str,
                       help="Path to trained transformer model (auto-detected if not provided)")
    parser.add_argument("--batch-size", type=int, default=64,
                       help="Batch size for inference")
    parser.add_argument("--compare-models", action="store_true",
                       help="Enable multi-model comparison (new behavior)")
    parser.add_argument("--transformer-only", action="store_true", 
                       help="Force transformer-only evaluation (original behavior)")
    parser.add_argument("--compare-only", action="store_true",
                       help="Only compare existing results, don't run new evaluations")
    
    args = parser.parse_args()
    
    # Detect system configuration
    config = detect_system_resources()
    
    # Determine evaluation mode
    if args.transformer_only or (not args.compare_models and not args.compare_only):
        mode = "transformer_only"
        description = "Transformer Model Evaluation (Original Mode)"
    else:
        mode = "multi_model"
        description = "Multi-Model Evaluation and Comparison"
    
    print(f"🚀 {description}")
    print("=" * 60)
    print(f"Log type: {args.log_type}")
    print(f"Device: {config.device}")
    print(f"Node: {config.node_name} | Job: {config.job_id}")
    print(f"Mode: {mode}")
    if mode == "transformer_only":
        print(f"Advanced threshold optimization: ENABLED (default)")
        print(f"📊 Dataset: Using FULL LogBERT embeddings (no sampling)")
    print("")
    
    try:
        # Initialize evaluator
        evaluator = UniversalEvaluator(config)
        
        if mode == "transformer_only":
            # Original transformer-only evaluation
            results = evaluator.evaluate_transformer_only(
                args.log_type, args.model_path, args.batch_size
            )
            
            print(f"\n✅ Evaluation completed for {args.log_type}")
            print(f"📁 Results saved to: {results['results_file']}")
            
            # Summary
            metrics = results['metrics']
            print(f"\n🎯 SUMMARY")
            print(f"   Macro F1:  {metrics['macro_f1']:.4f}")
            print(f"   Micro F1:  {metrics['micro_f1']:.4f}")
            print(f"   Accuracy:  {metrics['subset_accuracy']:.4f}")
            
        else:
            # Multi-model evaluation and comparison
            results = evaluator.evaluate_multi_model(
                args.log_type, args.compare_only, args.transformer_only
            )
            
            if results:
                print(f"\n✅ Evaluation completed for {args.log_type}")
                print(f"📁 Results saved to:")
                for file_path in results.get('saved_files', []):
                    print(f"   {file_path}")
                
                # Summary
                comparison_metrics = results.get('comparison_metrics', {})
                if comparison_metrics:
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