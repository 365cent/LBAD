#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for the improved transformer implementation

This script tests the new transformer with a small subset of data to ensure
it works correctly before running on the full dataset.
"""

import os
import sys
import pickle
import numpy as np
from pathlib import Path
import argparse

# Add src to path
sys.path.append(str(Path(__file__).parent))

from transformer import (
    detect_system_resources, 
    load_embedding_data, 
    preprocess_data,
    train_model_for_log_type,
    find_available_embeddings
)

def test_single_log_type(log_type: str, sample_size: int = 1000):
    """Test the transformer with a single log type using a small sample"""
    
    print(f"Testing transformer with log type: {log_type}")
    print(f"Sample size: {sample_size}")
    
    # Detect system resources
    config = detect_system_resources()
    
    try:
        # Load embedding data
        print("Loading embedding data...")
        embeddings, labels, classes = load_embedding_data(log_type)
        
        # Sample data for testing
        if len(embeddings) > sample_size:
            indices = np.random.choice(len(embeddings), sample_size, replace=False)
            embeddings = embeddings[indices]
            labels = labels[indices]
            print(f"Sampled {sample_size} entries from {len(embeddings)} total")
        
        # Preprocess data
        print("Preprocessing data...")
        data, scaler = preprocess_data(embeddings, labels, config, test_size=0.3)
        
        print(f"Training data shape: {data['X_train'].shape}")
        print(f"Test data shape: {data['X_test'].shape}")
        print(f"Number of classes: {len(classes)}")
        
        # Train model with reduced epochs for testing
        print("Training model...")
        result = train_model_for_log_type(
            log_type, 
            config, 
            max_epochs=10,  # Reduced for testing
            patience=5
        )
        
        if result is not None:
            print("Training completed successfully!")
            print(f"Best validation loss: {result['best_val_loss']:.6f}")
            print(f"Micro F1: {result['metrics']['micro_f1']:.4f}")
            print(f"Macro F1: {result['metrics']['macro_f1']:.4f}")
            
            # Test model loading and prediction
            print("Testing model prediction...")
            model = result['model']
            X_test = data['X_test'][:10]  # Test with 10 samples
            
            model.eval()
            import torch
            with torch.no_grad():
                X_tensor = torch.from_numpy(X_test).to(model.device)
                logits = model(X_tensor)
                probabilities = torch.sigmoid(logits).cpu().numpy()
                predictions = (probabilities > result['thresholds']).astype(int)
            
            print(f"Prediction shape: {predictions.shape}")
            print(f"Sample predictions: {predictions[:3]}")
            
            return True
        else:
            print("Training failed!")
            return False
            
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description="Test the improved transformer implementation")
    parser.add_argument("--log-type", type=str, default=None, help="Test specific log type")
    parser.add_argument("--sample-size", type=int, default=1000, help="Number of samples to test with")
    args = parser.parse_args()
    
    # Find available embeddings
    available_types = find_available_embeddings()
    if not available_types:
        print("No embedding files found. Please run logbert_embeddings.py or fasttext_embedding.py first.")
        return
    
    print(f"Available log types: {available_types}")
    
    if args.log_type:
        if args.log_type not in available_types:
            print(f"Log type '{args.log_type}' not found. Available types: {available_types}")
            return
        types_to_test = [args.log_type]
    else:
        # Test with the first available type
        types_to_test = [available_types[0]]
    
    success_count = 0
    for log_type in types_to_test:
        print(f"\n{'='*60}")
        print(f"Testing log type: {log_type}")
        print(f"{'='*60}")
        
        if test_single_log_type(log_type, args.sample_size):
            success_count += 1
            print(f"✓ Test passed for {log_type}")
        else:
            print(f"✗ Test failed for {log_type}")
    
    print(f"\n{'='*60}")
    print(f"Test Summary: {success_count}/{len(types_to_test)} tests passed")
    if success_count == len(types_to_test):
        print("All tests passed! The transformer implementation is working correctly.")
        print("You can now run the full training with: python src/transformer.py")
    else:
        print("Some tests failed. Please check the error messages above.")
    print(f"{'='*60}")

if __name__ == "__main__":
    main() 