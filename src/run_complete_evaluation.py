#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete Evaluation Workflow

Runs the entire pipeline:
1. Preprocessing (if needed)
2. Transformer training
3. Prediction on all log types
4. Evaluation against original data
5. Comprehensive metrics and reports
"""

import os
import sys
import subprocess
import time
import logging
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CompleteEvaluationWorkflow:
    """Complete evaluation workflow for transformer predictions"""
    
    def __init__(self):
        self.base_dir = Path.cwd()
        self.log_types = ['wp-error', 'wp-access', 'vpn', 'auth', 'audit', 'dns', 'share', 'monitor']
        
    def run_command(self, command: str, description: str) -> bool:
        """Run a command and handle errors"""
        logger.info(f"Running: {description}")
        logger.info(f"Command: {command}")
        
        start_time = time.time()
        try:
            result = subprocess.run(
                command, 
                shell=True, 
                capture_output=True, 
                text=True, 
                cwd=self.base_dir
            )
            
            elapsed = time.time() - start_time
            
            if result.returncode == 0:
                logger.info(f"✓ {description} completed successfully in {elapsed:.2f}s")
                if result.stdout:
                    logger.info(f"Output: {result.stdout}")
                return True
            else:
                logger.error(f"✗ {description} failed after {elapsed:.2f}s")
                logger.error(f"Error: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"✗ {description} failed with exception: {e}")
            return False
    
    def check_prerequisites(self) -> bool:
        """Check if all required files and directories exist"""
        logger.info("Checking prerequisites...")
        
        required_dirs = ['logs', 'labels', 'embeddings']
        missing_dirs = []
        
        for dir_name in required_dirs:
            if not (self.base_dir / dir_name).exists():
                missing_dirs.append(dir_name)
        
        if missing_dirs:
            logger.error(f"Missing required directories: {missing_dirs}")
            logger.error("Please ensure logs/, labels/, and embeddings/ directories exist")
            return False
        
        # Check if embeddings exist
        embeddings_dir = self.base_dir / "embeddings"
        embedding_files = list(embeddings_dir.rglob("log_*.pkl"))
        
        if not embedding_files:
            logger.warning("No embedding files found. You may need to run logbert_embeddings.py first")
            return False
        
        logger.info(f"Found {len(embedding_files)} embedding files")
        return True
    
    def run_preprocessing(self) -> bool:
        """Run preprocessing if needed"""
        processed_dir = self.base_dir / "processed"
        
        if processed_dir.exists() and any(processed_dir.rglob("*.tfrecord")):
            logger.info("Processed data already exists, skipping preprocessing")
            return True
        
        logger.info("Running preprocessing...")
        return self.run_command(
            "python src/preprocessing.py",
            "Preprocessing log files"
        )
    
    def run_transformer_training(self) -> bool:
        """Run transformer training"""
        logger.info("Running transformer training...")
        return self.run_command(
            "python src/transformer.py",
            "Transformer training"
        )
    
    def run_predictions(self) -> bool:
        """Run predictions for all log types"""
        logger.info("Running predictions for all log types...")
        
        success_count = 0
        for log_type in self.log_types:
            logger.info(f"Predicting for log type: {log_type}")
            
            success = self.run_command(
                f"python src/transformer_predict.py -e {log_type}",
                f"Prediction for {log_type}"
            )
            
            if success:
                success_count += 1
            else:
                logger.warning(f"Prediction failed for {log_type}, continuing with others")
        
        logger.info(f"Completed predictions for {success_count}/{len(self.log_types)} log types")
        return success_count > 0
    
    def run_evaluation(self) -> bool:
        """Run comprehensive evaluation"""
        logger.info("Running comprehensive evaluation...")
        return self.run_command(
            "python src/evaluate_transformer.py",
            "Comprehensive evaluation"
        )
    
    def generate_summary_report(self):
        """Generate a summary report of all results"""
        logger.info("Generating summary report...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.base_dir / "predictions" / f"workflow_summary_{timestamp}.txt"
        
        with open(report_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("COMPLETE EVALUATION WORKFLOW SUMMARY\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n\n")
            
            # Check what files were created
            f.write("FILES CREATED:\n")
            f.write("-" * 40 + "\n")
            
            # Check processed files
            processed_dir = self.base_dir / "processed"
            if processed_dir.exists():
                tfrecord_files = list(processed_dir.rglob("*.tfrecord"))
                f.write(f"Processed TFRecord files: {len(tfrecord_files)}\n")
                for tfrecord in tfrecord_files:
                    f.write(f"  - {tfrecord.relative_to(self.base_dir)}\n")
            
            # Check model files
            models_dir = self.base_dir / "models"
            if models_dir.exists():
                model_files = list(models_dir.glob("*.pth"))
                f.write(f"\nModel files: {len(model_files)}\n")
                for model in model_files:
                    f.write(f"  - {model.relative_to(self.base_dir)}\n")
            
            # Check prediction files
            predictions_dir = self.base_dir / "predictions"
            if predictions_dir.exists():
                prediction_files = list(predictions_dir.glob("*.pkl"))
                evaluation_files = list(predictions_dir.glob("evaluation_results_*.json"))
                
                f.write(f"\nPrediction files: {len(prediction_files)}\n")
                for pred in prediction_files:
                    f.write(f"  - {pred.relative_to(self.base_dir)}\n")
                
                f.write(f"\nEvaluation files: {len(evaluation_files)}\n")
                for eval_file in evaluation_files:
                    f.write(f"  - {eval_file.relative_to(self.base_dir)}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("WORKFLOW COMPLETED SUCCESSFULLY\n")
            f.write("=" * 80 + "\n")
        
        logger.info(f"Summary report saved to: {report_file}")
        return report_file
    
    def run_complete_workflow(self):
        """Run the complete evaluation workflow"""
        logger.info("Starting complete evaluation workflow...")
        start_time = time.time()
        
        # Step 1: Check prerequisites
        if not self.check_prerequisites():
            logger.error("Prerequisites check failed. Exiting.")
            return False
        
        # Step 2: Run preprocessing (if needed)
        if not self.run_preprocessing():
            logger.error("Preprocessing failed. Exiting.")
            return False
        
        # Step 3: Run transformer training
        if not self.run_transformer_training():
            logger.error("Transformer training failed. Exiting.")
            return False
        
        # Step 4: Run predictions
        if not self.run_predictions():
            logger.error("Predictions failed. Exiting.")
            return False
        
        # Step 5: Run evaluation
        if not self.run_evaluation():
            logger.error("Evaluation failed. Exiting.")
            return False
        
        # Step 6: Generate summary report
        report_file = self.generate_summary_report()
        
        total_time = time.time() - start_time
        logger.info(f"Complete workflow finished in {total_time:.2f} seconds")
        logger.info(f"Summary report: {report_file}")
        
        return True

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run complete evaluation workflow")
    parser.add_argument("--skip-preprocessing", action="store_true",
                       help="Skip preprocessing step")
    parser.add_argument("--skip-training", action="store_true",
                       help="Skip transformer training step")
    parser.add_argument("--skip-predictions", action="store_true",
                       help="Skip prediction step")
    parser.add_argument("--skip-evaluation", action="store_true",
                       help="Skip evaluation step")
    
    args = parser.parse_args()
    
    workflow = CompleteEvaluationWorkflow()
    
    try:
        success = workflow.run_complete_workflow()
        
        if success:
            print("\n" + "=" * 80)
            print("🎉 COMPLETE EVALUATION WORKFLOW SUCCESSFUL! 🎉")
            print("=" * 80)
            print("All steps completed successfully!")
            print("Check the predictions/ directory for results and evaluation reports.")
            print("=" * 80)
        else:
            print("\n" + "=" * 80)
            print("❌ WORKFLOW FAILED ❌")
            print("=" * 80)
            print("One or more steps failed. Check the logs above for details.")
            print("=" * 80)
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\nWorkflow interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Workflow failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 