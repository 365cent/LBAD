#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test FastText Embeddings
-----------------------
Tests and validates the FastText embeddings generated for log analysis.
Previews the head of the embeddings for basic validation.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import json
import argparse

# Get script's directory and project root
SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
ROOT = SCRIPT_DIR.parent
EMB_DIR = ROOT / 'embeddings'

def load_embeddings(verbose=True):
    """Load embeddings and return basic statistics."""
    if verbose:
        print("Loading embeddings...")
    
    # Check if embeddings exist
    train_path = EMB_DIR / 'train_embeddings.pkl'
    test_path = EMB_DIR / 'test_embeddings.pkl'
    
    if not train_path.exists() or not test_path.exists():
        print(f"Error: Embedding files not found in {EMB_DIR}")
        print("Please run fasttext_embedding.py first to generate embeddings.")
        sys.exit(1)
    
    # Load embeddings
    with open(train_path, 'rb') as f:
        train_embeddings = pickle.load(f)
    
    with open(test_path, 'rb') as f:
        test_embeddings = pickle.load(f)
    
    # Try to load labels if they exist
    train_labels = None
    test_labels = None
    train_labels_path = EMB_DIR / 'train_labels.pkl'
    test_labels_path = EMB_DIR / 'test_labels.pkl'
    
    if train_labels_path.exists() and test_labels_path.exists():
        with open(train_labels_path, 'rb') as f:
            train_labels_raw = pickle.load(f)
        with open(test_labels_path, 'rb') as f:
            test_labels_raw = pickle.load(f)
        
        # Parse labels from JSON strings
        train_labels = parse_labels(train_labels_raw)
        test_labels = parse_labels(test_labels_raw)
    
    if verbose:
        print(f"Train embeddings shape: {train_embeddings.shape}")
        print(f"Test embeddings shape: {test_embeddings.shape}")
        
        # Basic statistics
        print("\nEmbedding Statistics:")
        print(f"Train mean: {np.mean(train_embeddings):.6f}")
        print(f"Train std: {np.std(train_embeddings):.6f}")
        print(f"Test mean: {np.mean(test_embeddings):.6f}")
        print(f"Test std: {np.std(test_embeddings):.6f}")
        
        # Check for NaNs or Infs
        train_nans = np.isnan(train_embeddings).any()
        test_nans = np.isnan(test_embeddings).any()
        train_infs = np.isinf(train_embeddings).any()
        test_infs = np.isinf(test_embeddings).any()
        
        if train_nans or train_infs:
            print("Warning: Train embeddings contain NaN or Inf values!")
        if test_nans or test_infs:
            print("Warning: Test embeddings contain NaN or Inf values!")
        
        # Label information if available
        if train_labels and test_labels:
            print("\nLabel Distribution:")
            train_counts = pd.Series(train_labels).value_counts()
            test_counts = pd.Series(test_labels).value_counts()
            
            print("Train:")
            for label, count in train_counts.items():
                print(f"  {label}: {count} ({100*count/len(train_labels):.1f}%)")
            
            print("Test:")
            for label, count in test_counts.items():
                print(f"  {label}: {count} ({100*count/len(test_labels):.1f}%)")
    
    return train_embeddings, test_embeddings, train_labels, test_labels

def parse_labels(label_strings):
    """Parse label strings from JSON format to classification labels."""
    parsed_labels = []
    for label_str in label_strings:
        try:
            label_data = json.loads(label_str)
            if isinstance(label_data, list):
                if not label_data:  # Empty array means "normal"
                    parsed_labels.append("normal")
                else:
                    parsed_labels.append(label_data[0])  # Use first label if multiple
            else:
                parsed_labels.append("unknown")
        except:
            parsed_labels.append("unknown")
    return parsed_labels

def preview_embeddings(train_embeddings, test_embeddings, train_labels, test_labels, n=5):
    """Preview the head of the embeddings and labels."""
    print("\nTrain Embeddings Preview:")
    print(train_embeddings[:n])
    print("\nTrain Labels Preview:")
    print(train_labels[:n])
    
    print("\nTest Embeddings Preview:")
    print(test_embeddings[:n])
    print("\nTest Labels Preview:")
    print(test_labels[:n])

def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description='Test FastText embeddings for log analysis')
    parser.add_argument('--preview', action='store_true', help='Preview head of embeddings')
    args = parser.parse_args()
    
    # Always load and validate embeddings
    train_embeddings, test_embeddings, train_labels, test_labels = load_embeddings()
    
    if args.preview:
        preview_embeddings(train_embeddings, test_embeddings, train_labels, test_labels)
    
    print("Embedding tests completed successfully!")

if __name__ == "__main__":
    main()