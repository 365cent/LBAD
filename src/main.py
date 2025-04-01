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
AUGMENTED_DIR = BASE_DIR / "augmented"
EVAL_DIR = BASE_DIR / "evaluation"

# Ensure directories exist
for directory in [PROCESSED_DIR, EMBEDDING_DIR, MODEL_DIR, RESULTS_DIR, AUGMENTED_DIR, EVAL_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

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
    
    parser.add_argument('--fasttext', action='store_true',
                        help='Generate FastText embeddings')
    
    parser.add_argument('--word2vec', action='store_true',
                        help='Generate Word2Vec embeddings')
    
    parser.add_argument('--test', action='store_true',
                        help='Run testing utilities')
    
    parser.add_argument('--gan', action='store_true',
                        help='Run GAN-based data augmentation')
    
    parser.add_argument('--evaluate', action='store_true',
                        help='Run GAN evaluation')
    
    parser.add_argument('--eval-model', choices=['rf', 'xgb', 'all'],
                        default='all', help='Models to evaluate (default: all)')
    
    parser.add_argument('--eval-data', choices=['original', 'augmented', 'both'],
                        default='both', help='Data to evaluate (default: both)')
    
    parser.add_argument('--all', action='store_true',
                        help='Run all steps')
    
    parser.add_argument('--skip-errors', action='store_true',
                        help='Continue execution even if a step fails')
    
    return parser.parse_args()

def main():
    """Main function to orchestrate the workflow."""
    args = parse_arguments()
    
    # If --all is specified or no arguments provided, run all steps
    if args.all or not any([args.preprocess, args.fasttext, args.word2vec, args.test, args.gan, args.evaluate]):
        args.preprocess = True
        args.test = False
        args.fasttext = True
        args.word2vec = True
        args.gan = True
        args.evaluate = True
    
    # Track successful steps
    successful = []
    
    # Step 1: Log Preprocessing
    if args.preprocess:
        if run_script('preprocessing.py', 'Log Preprocessing'):
            successful.append('preprocess')
        elif not args.skip_errors:
            print("Exiting due to preprocessing failure")
            return
    
    # Step 2: Testing Utilities
    if args.test:
        if run_script('preprocess_testing.py', 'Testing Utilities'):
            successful.append('test')
        elif not args.skip_errors:
            print("Exiting due to testing failure")
            return
    
    # FastText Workflow
    # =================
    
    # Step 3a: FastText Embedding
    if args.fasttext and ('preprocess' in successful or not args.preprocess):
        if run_script('fasttext_embedding.py', 'FastText Embedding'):
            successful.append('fasttext')
        elif not args.skip_errors:
            print("Exiting due to FastText embedding failure")
            return
    
    # Step 4a: GAN Augmentation with FastText embeddings
    if args.gan and args.fasttext and ('fasttext' in successful or not args.fasttext):
        if run_script('gan_augmentation.py', 'FastText-based GAN Augmentation'):
            successful.append('gan-fasttext')
        elif not args.skip_errors:
            print("Exiting due to FastText GAN failure")
            return
    
    # Word2Vec Workflow
    # =================
    
    # Step 3b: Word2Vec Embedding
    if args.word2vec and ('preprocess' in successful or not args.preprocess):
        if run_script('word2vec_embedding.py', 'Word2Vec Embedding'):
            successful.append('word2vec')
        elif not args.skip_errors:
            print("Exiting due to Word2Vec embedding failure")
            return
    
    # Step 4b: GAN Augmentation with Word2Vec embeddings
    if args.gan and args.word2vec and ('word2vec' in successful or not args.word2vec):
        if run_script('gan_augmentation.py', 'Word2Vec-based GAN Augmentation'):
            successful.append('gan-word2vec')
        elif not args.skip_errors:
            print("Exiting due to Word2Vec GAN failure")
            return
    
    # Evaluation
    # ==========
    
    # Step 5: Evaluate GAN augmentation
    if args.evaluate and (any('gan-' in step for step in successful) or not args.gan):
        eval_args = []
        if args.eval_model != 'all':
            eval_args.extend(['--models', args.eval_model])
        if args.eval_data != 'both':
            eval_args.extend(['--data', args.eval_data])
            
        if run_script('gan_evaluation.py', 'GAN Augmentation Evaluation', eval_args):
            successful.append('evaluation')
        elif not args.skip_errors:
            print("Exiting due to evaluation failure")
            return
    
    # Summary
    print("\nWorkflow completed!")
    if successful:
        print(f"Successful steps: {', '.join(successful)}")
    else:
        print("No steps were executed successfully.")
    
    # Print final results location
    if 'evaluation' in successful:
        print(f"\nEvaluation results available in: {EVAL_DIR}")
    
    if any('gan-' in step for step in successful):
        print(f"GAN augmentation results available in: {AUGMENTED_DIR}")
    
    print(f"All results available in project directory: {BASE_DIR}")

if __name__ == "__main__":
    main()
