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
import tensorflow as tf
import platform

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

def optimize_macos_performance():
    """Apply performance optimizations for ML workloads on macOS."""
    print("Applying macOS performance optimizations...")
    
    # Configure TensorFlow for maximum performance
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            # Enable memory growth to avoid allocating all GPU memory at once
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
            print("✓ GPU memory growth enabled")
        except Exception as e:
            print(f"Error configuring GPU: {e}")
    
    # Set maximum thread priority
    os.system("sudo renice -20 $$")  # Requires password, might not work in all environments
    
    # Optimize NumPy thread settings
    NP_THREAD_COUNT = max(1, os.cpu_count() or 4)
    os.environ["OMP_NUM_THREADS"] = str(NP_THREAD_COUNT)
    os.environ["MKL_NUM_THREADS"] = str(NP_THREAD_COUNT)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(NP_THREAD_COUNT)
    os.environ["NUMEXPR_NUM_THREADS"] = str(NP_THREAD_COUNT)
    
    # Optimize TensorFlow for Apple Silicon
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Reduce TF logging
    os.environ["TF_MKL_ALLOC_MAX_BYTES"] = "10000000000"  # 10GB
    os.environ["TF_USE_LEGACY_TENSORFLOW_OPTIMIZATIONS"] = "0"
    
    # Enable aggressive compiler optimizations
    os.environ["CFLAGS"] = "-O3 -march=native -mtune=native"
    os.environ["CXXFLAGS"] = "-O3 -march=native -mtune=native"
    
    # Disable CPU throttling if running as root (uncomment if you have sudo without password)
    # subprocess.run("sudo pmset -a lessbright 0", shell=True)
    # subprocess.run("sudo pmset -a highperformance 1", shell=True)
    
    # Disable spotlight indexing during ML runs to free up I/O
    # subprocess.run("sudo mdutil -a -i off", shell=True)
    
    # Optimize disk I/O by increasing read-ahead cache
    if platform.machine() == 'arm64':  # Apple Silicon
        print("✓ Running on Apple Silicon - using optimized ML libraries")
        # Apple Silicon specific optimizations
        tf.config.optimizer.set_jit(True)  # Enable XLA JIT compilation
        print("✓ XLA JIT compilation enabled")
    
    print("✓ System optimized for high-performance ML workloads")
    return True

# Call optimization function before any ML operations
optimize_macos_performance()

def check_processed_files():
    """Check if processed TFRecord files exist."""
    if not PROCESSED_DIR.exists():
        return False
    
    tfrecord_files = list(PROCESSED_DIR.glob("**/*.tfrecord"))
    return len(tfrecord_files) > 0

def check_embedding_files(embedding_type):
    """Check if specific embedding files exist."""
    prefix = f"{embedding_type}_" if embedding_type != 'fasttext' else ""
    required_files = [
        EMBEDDING_DIR / f'{prefix}train_embeddings.pkl',
        EMBEDDING_DIR / f'{prefix}test_embeddings.pkl',
        EMBEDDING_DIR / f'{prefix}train_labels.pkl',
        EMBEDDING_DIR / f'{prefix}test_labels.pkl'
    ]
    return all(f.exists() for f in required_files)

def get_available_embeddings():
    """Get a list of all available embedding types by checking files."""
    available = []
    for embedding_type in ['fasttext', 'word2vec', 'tfidf']:
        if check_embedding_files(embedding_type):
            available.append(embedding_type)
    return available

def check_model_files(embedding_type, model_types):
    """Check if trained model files exist for specified embedding/model types."""
    if model_types == 'all':
        model_types = ['rf', 'xgb', 'knn', 'lr']
    elif isinstance(model_types, str):
        model_types = model_types.split(',')
    
    # Check for model files with timestamp patterns
    embedding_prefix = f"{embedding_type}_" if embedding_type != 'default' else ""
    
    # Look for result files in result directory
    result_dirs = list(RESULTS_DIR.glob("run_*"))
    if not result_dirs:
        return False
    
    # Check most recent result directory for model outputs
    latest_run = sorted(result_dirs)[-1]
    required_files = [latest_run / f"{model}_report.txt" for model in model_types]
    
    return all(f.exists() for f in required_files)

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
    
    parser.add_argument('--model', choices=['rf', 'xgb', 'knn', 'lr', 'all'],
                        default='all', help='ML model to use (default: all)')
    
    parser.add_argument('--test', action='store_true',
                        help='Run testing utilities')
    
    parser.add_argument('--force', action='store_true',
                        help='Force execution of all steps regardless of existing files')
    
    parser.add_argument('--all', action='store_true',
                        help='Run all steps')
    
    parser.add_argument('--skip-errors', action='store_true',
                        help='Continue execution even if a step fails')
    
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
    
    # Set all embeddings if requested
    if args.all_embeddings:
        args.fasttext = True
        args.word2vec = True
        args.tfidf = True
    
    # Track successful steps
    successful = []
    skipped = []
    
    # Check if we need preprocessing
    preprocessing_exists = check_processed_files()
    
    # Step 1: Log Preprocessing (if needed or explicitly requested)
    if args.preprocess:
        if preprocessing_exists and not args.force:
            print("Processed files already exist. Skipping preprocessing step.")
            skipped.append('preprocess')
            successful.append('preprocess')  # Count as successful for dependency checks
        else:
            if run_script('preprocessing.py', 'Log Preprocessing'):
                successful.append('preprocess')
            elif not args.skip_errors:
                print("Exiting due to preprocessing failure")
                return
    elif not preprocessing_exists:
        print("No processed files found. Preprocessing is required.")
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
    
    # Embedding Generation
    # ===================
    
    # Step 3a: FastText Embedding
    if args.fasttext:
        if check_embedding_files('fasttext') and not args.force:
            print("FastText embeddings already exist. Skipping generation.")
            skipped.append('fasttext')
            successful.append('fasttext')  # Count as successful for ML dependency
        elif 'preprocess' in successful or preprocessing_exists:
            if run_script('fasttext_embedding.py', 'FastText Embedding'):
                successful.append('fasttext')
            elif not args.skip_errors:
                print("Exiting due to FastText embedding failure")
                return
    
    # Step 3b: Word2Vec Embedding
    if args.word2vec:
        if check_embedding_files('word2vec') and not args.force:
            print("Word2Vec embeddings already exist. Skipping generation.")
            skipped.append('word2vec')
            successful.append('word2vec')  # Count as successful for ML dependency
        elif 'preprocess' in successful or preprocessing_exists:
            if run_script('word2vec_embedding.py', 'Word2Vec Embedding'):
                successful.append('word2vec')
            elif not args.skip_errors:
                print("Exiting due to Word2Vec embedding failure")
                return
    
    # Step 3c: TF-IDF Embedding
    if args.tfidf:
        if check_embedding_files('tfidf') and not args.force:
            print("TF-IDF embeddings already exist. Skipping generation.")
            skipped.append('tfidf')
            successful.append('tfidf')  # Count as successful for ML dependency
        elif 'preprocess' in successful or preprocessing_exists:
            if run_script('tfidf_embedding.py', 'TF-IDF Embedding'):
                successful.append('tfidf')
            elif not args.skip_errors:
                print("Exiting due to TF-IDF embedding failure")
                return
    
    # Machine Learning Models
    # ======================
    
    # Get previously available embeddings that might not have been generated in this run
    available_embeddings = get_available_embeddings()
    
    # Add newly created embeddings to available embeddings
    for emb_type in ['fasttext', 'word2vec', 'tfidf']:
        if emb_type in successful and emb_type not in available_embeddings:
            available_embeddings.append(emb_type)
    
    # Step 4: Run ML Models with available embeddings
    if args.ml:
        ml_args = []
        
        if args.model != 'all':
            ml_args.extend(['--model', args.model])
        
        if not available_embeddings:
            print("\nNo embeddings available. Skipping ML step.")
        else:
            print(f"\nAvailable embeddings for ML: {', '.join(available_embeddings)}")
        
        # If specific embedding types were requested, only use those for ML
        requested_embeddings = []
        if args.fasttext:
            requested_embeddings.append('fasttext')
        if args.word2vec:
            requested_embeddings.append('word2vec')
        if args.tfidf:
            requested_embeddings.append('tfidf')
        
        # If no specific embeddings were requested but ML was requested, use all available
        if not requested_embeddings and args.ml:
            requested_embeddings = available_embeddings
        
        # Filter available embeddings based on requested ones
        selected_embeddings = [emb for emb in requested_embeddings if emb in available_embeddings]
        
        # Run ML on selected embeddings
        for embedding_type in selected_embeddings:
            # Check if ML has already been run with these embeddings
            if check_model_files(embedding_type, args.model) and not args.force:
                print(f"ML models with {embedding_type} embeddings already exist. Skipping.")
                skipped.append(f'ml-{embedding_type}')
                successful.append(f'ml-{embedding_type}')
            else:
                print(f"\nRunning ML models with {embedding_type} embeddings...")
                current_args = ml_args + ['--embedding-type', embedding_type]
                
                if run_script('ml_models.py', f'ML with {embedding_type.upper()} Embeddings', current_args):
                    successful.append(f'ml-{embedding_type}')
                elif not args.skip_errors:
                    print(f"Exiting due to ML with {embedding_type} failure")
                    return
    
    # Summary
    print("\nWorkflow completed!")
    print("=" * 60)
    
    # Only show steps that were actually executed (not skipped)
    executed_steps = [step for step in successful if step not in skipped]
    
    if executed_steps:
        print(f"Executed steps: {', '.join(executed_steps)}")
    else:
        print("No steps were executed.")
        
    if skipped:
        print(f"Skipped steps (already completed): {', '.join(skipped)}")
    
    # Print embedding availability
    if available_embeddings:
        print(f"\nAvailable embeddings: {', '.join(available_embeddings)}")
        print(f"Embedding files available in: {EMBEDDING_DIR}")
    
    # Print model information
    ml_steps = [step for step in successful if step.startswith('ml-')]
    if ml_steps:
        print(f"\nAvailable ML models: {', '.join(ml_steps)}")
        print(f"Model files available in: {MODEL_DIR}")
        print(f"Results available in: {RESULTS_DIR}")
    
    print(f"\nAll results available in project directory: {BASE_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()
