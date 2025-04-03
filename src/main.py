#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main Script for Log-Based Attack Detection Research
--------------------------------------------------
This script orchestrates the workflow for log preprocessing, embedding,
model training, and visualization.
"""

import os
import argparse
import subprocess
import time
import sys
from pathlib import Path
from datetime import datetime

# Base project directory (parent of the src directory)
BASE_DIR = Path(__file__).resolve().parent.parent

# Directory structure
PROCESSED_DIR = BASE_DIR / "processed"
EMBEDDING_DIR = BASE_DIR / "embeddings"
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
LOGS_DIR = BASE_DIR / "logs"
LABELS_DIR = BASE_DIR / "labels"

# Ensure directories exist
for directory in [PROCESSED_DIR, EMBEDDING_DIR, MODEL_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

def check_processed_files():
    """Check if processed TFRecord files exist."""
    if not PROCESSED_DIR.exists():
        return False
    
    tfrecord_files = list(PROCESSED_DIR.glob("**/*.tfrecord"))
    return len(tfrecord_files) > 0

def run_script(script_name, description, args=None):
    """Run a Python script and handle errors."""
    script_path = Path(__file__).parent / script_name
    
    print(f"\n{'='*60}")
    print(f"Running {description}: {script_name}")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    cmd = ['python3', str(script_path)]
    if args:
        cmd.extend(args)
    
    try:
        result = subprocess.run(cmd, check=True)
        elapsed_time = time.time() - start_time
        print(f"\n{description} completed in {elapsed_time:.2f} seconds")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nError running {description}: {e}")
        return False

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Log Preprocessing and Analysis Pipeline')
    
    parser.add_argument('--preprocess', action='store_true',
                        help='Run log preprocessing')
    
    # Embedding options
    parser.add_argument('--fasttext', action='store_true',
                        help='Generate FastText embeddings')
    
    parser.add_argument('--word2vec', action='store_true',
                        help='Generate Word2Vec embeddings')
    
    parser.add_argument('--tfidf', action='store_true',
                        help='Generate TF-IDF embeddings')
    
    parser.add_argument('--all-embeddings', action='store_true',
                        help='Generate all embedding types')
    
    # ML models
    parser.add_argument('--ml', action='store_true',
                        help='Run machine learning models')
    
    parser.add_argument('--model', choices=['rf', 'xgb', 'svm', 'knn', 'lr', 'all'],
                        default='all', help='ML model to use (default: all)')
    
    parser.add_argument('--test', action='store_true',
                        help='Run testing utilities')
    
    # Temporarily removed GAN-related arguments
    # parser.add_argument('--gan', action='store_true',
    #                     help='Run GAN-based data augmentation')
    
    # parser.add_argument('--evaluate', action='store_true',
    #                     help='Run GAN evaluation')
    
    # parser.add_argument('--eval-model', choices=['rf', 'xgb', 'all'],
    #                     default='all', help='Models to evaluate (default: all)')
    
    # parser.add_argument('--eval-data', choices=['original', 'augmented', 'both'],
    #                     default='both', help='Data to evaluate (default: both)')
    
    parser.add_argument('--all', action='store_true',
                        help='Run all steps')
    
    parser.add_argument('--skip-errors', action='store_true',
                        help='Continue execution even if a step fails')
    
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum number of samples to use (for ML with direct TFRecord)')
                        
    parser.add_argument('--direct', action='store_true',
                        help='Run ML directly on TFRecord data without embeddings')
    
    return parser.parse_args()

def main():
    """Main function to orchestrate the workflow."""
    args = parse_arguments()
    
    # If --all is specified or no arguments provided, run all steps
    if args.all or not any([args.preprocess, args.fasttext, args.word2vec, args.tfidf, 
                          args.all_embeddings, args.test, args.ml]):
        args.preprocess = True
        args.all_embeddings = True
        args.ml = True
        args.test = False
        # Removed GAN-related flags:
        # args.gan = False
        # args.evaluate = False
    
    # Set all embeddings if requested
    if args.all_embeddings:
        args.fasttext = True
        args.word2vec = True
        args.tfidf = True
    
    # Track successful steps
    successful = []
    
    # Check if we need preprocessing
    need_preprocessing = False
    if not check_processed_files():
        print("No processed files found. Preprocessing is required.")
        need_preprocessing = True
    
    # Step 1: Log Preprocessing (if needed or explicitly requested)
    if need_preprocessing or args.preprocess:
        if run_script('preprocessing.py', 'Log Preprocessing'):
            successful.append('preprocess')
        elif not args.skip_errors:
            print("Exiting due to preprocessing failure")
            return
    elif not check_processed_files():
        print("Error: No processed files found and preprocessing was not run.")
        print("Please run with --preprocess flag.")
        return
    
    # Step 2: Testing Utilities
    if args.test:
        if run_script('preprocess_testing.py', 'Testing Utilities'):
            successful.append('test')
        elif not args.skip_errors:
            print("Exiting due to testing failure")
            return
    
    # Embedding Generation
    # ===================
    
    # Step 3a: FastText Embedding
    if args.fasttext and ('preprocess' in successful or not need_preprocessing):
        if run_script('fasttext_embedding.py', 'FastText Embedding'):
            successful.append('fasttext')
        elif not args.skip_errors:
            print("Exiting due to FastText embedding failure")
            return
    
    # Step 3b: Word2Vec Embedding
    if args.word2vec and ('preprocess' in successful or not need_preprocessing):
        if run_script('word2vec_embedding.py', 'Word2Vec Embedding'):
            successful.append('word2vec')
        elif not args.skip_errors:
            print("Exiting due to Word2Vec embedding failure")
            return
    
    # Step 3c: TF-IDF Embedding
    if args.tfidf and ('preprocess' in successful or not need_preprocessing):
        if run_script('tfidf_embedding.py', 'TF-IDF Embedding'):
            successful.append('tfidf')
        elif not args.skip_errors:
            print("Exiting due to TF-IDF embedding failure")
            return
    
    # Machine Learning Models
    # ======================
    
    # Step 4: Run ML Models
    if args.ml:
        ml_args = []
        
        if args.model != 'all':
            ml_args.extend(['--model', args.model])
        
        if args.direct:
            ml_args.extend(['--data-source', 'tfrecord'])
            if args.max_samples:
                ml_args.extend(['--max-samples', str(args.max_samples)])
        else:
            # Run ML on each available embedding type
            for embedding_type in ['fasttext', 'word2vec', 'tfidf']:
                if embedding_type in successful:
                    print(f"\nRunning ML models with {embedding_type} embeddings...")
                    current_args = ml_args + ['--embedding-type', embedding_type]
                    
                    if run_script('ml_models.py', f'ML with {embedding_type.upper()} Embeddings', current_args):
                        successful.append(f'ml-{embedding_type}')
                    elif not args.skip_errors:
                        print(f"Exiting due to ML with {embedding_type} failure")
                        return
    
    # Temporarily removed GAN-related sections
    # GAN Augmentation
    # ===============
    
    # Step 5: GAN Augmentation with available embeddings
    # if args.gan:
    #     for embedding_type in ['fasttext', 'word2vec', 'tfidf']:
    #         if embedding_type in successful:
    #             gan_args = ['--embedding-type', embedding_type]
    #             if run_script('gan_augmentation.py', f'{embedding_type.upper()}-based GAN Augmentation', gan_args):
    #                 successful.append(f'gan-{embedding_type}')
    #             elif not args.skip_errors:
    #                 print(f"Exiting due to {embedding_type} GAN failure")
    #                 return
    
    # Evaluation
    # ==========
    
    # Step 6: Evaluate GAN augmentation
    # if args.evaluate and any('gan-' in step for step in successful):
    #     eval_args = []
    #     if args.eval_model != 'all':
    #         eval_args.extend(['--models', args.eval_model])
    #     if args.eval_data != 'both':
    #         eval_args.extend(['--data', args.eval_data])
    #         
    #     if run_script('gan_evaluation.py', 'GAN Augmentation Evaluation', eval_args):
    #         successful.append('evaluation')
    #     elif not args.skip_errors:
    #         print("Exiting due to evaluation failure")
    #         return
    
    # Summary
    print("\nWorkflow completed!")
    print("=" * 60)
    if successful:
        print(f"Successful steps: {', '.join(successful)}")
    else:
        print("No steps were executed successfully.")
    
    # Print embedding availability
    available_embeddings = []
    for embedding in ['fasttext', 'word2vec', 'tfidf']:
        if embedding in successful:
            available_embeddings.append(embedding)
    
    if available_embeddings:
        print(f"\nAvailable embeddings: {', '.join(available_embeddings)}")
        print(f"Embedding files available in: {EMBEDDING_DIR}")
    
    # Print model information
    ml_steps = [step for step in successful if step.startswith('ml-')]
    if ml_steps:
        print(f"\nCompleted ML models: {', '.join(ml_steps)}")
        print(f"Model files available in: {MODEL_DIR}")
        print(f"Results available in: {RESULTS_DIR}")
    
    # Removed GAN information output
    # # Print GAN information
    # gan_steps = [step for step in successful if step.startswith('gan-')]
    # if gan_steps:
    #     print(f"\nCompleted GAN augmentations: {', '.join(gan_steps)}")
    #     print(f"GAN augmentation results available in: {AUGMENTED_DIR}")
    # 
    # # Print evaluation information
    # if 'evaluation' in successful:
    #     print(f"\nEvaluation results available in: {EVAL_DIR}")
    
    print(f"\nAll results available in project directory: {BASE_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()
