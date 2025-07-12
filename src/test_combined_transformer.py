#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for the combined transformer implementation

This script tests the new combined model that:
1. Identifies log type first (wp-access, wp-error, etc.)
2. Predicts attack classes for that specific log type
3. Combines results from all log types
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
    load_all_embeddings,
    train_combined_model
)

def test_combined_model(sample_size: int = 1000):
    """Test the combined transformer with a small sample"""
    
    print("Testing combined transformer implementation")
    print(f"Sample size per log type: {sample_size}")
    
    # Detect system resources
    config = detect_system_resources()
    
    try:
        # Load all embedding data
        print("Loading all embedding data...")
        all_data = load_all_embeddings()
        
        if not all_data:
            print("No embedding data found. Please run logbert_embeddings.py or fasttext_embedding.py first.")
            return False
        
        print(f"Loaded {len(all_data)} log types: {list(all_data.keys())}")
        
        # Sample data for testing
        sampled_data = {}
        for log_type, (embeddings, labels, classes) in all_data.items():
            if len(embeddings) > sample_size:
                indices = np.random.choice(len(embeddings), sample_size, replace=False)
                sampled_embeddings = embeddings[indices]
                sampled_labels = labels[indices]
                print(f"Sampled {sample_size} entries from {log_type} (total: {len(embeddings)})")
            else:
                sampled_embeddings = embeddings
                sampled_labels = labels
                print(f"Using all {len(embeddings)} entries from {log_type}")
            
            sampled_data[log_type] = (sampled_embeddings, sampled_labels, classes)
        
        # Train combined model with reduced epochs for testing
        print("\nTraining combined model...")
        result = train_combined_model(
            sampled_data, 
            config, 
            max_epochs=10,  # Reduced for testing
            patience=5
        )
        
        if result is not None:
            print("Combined model training completed successfully!")
            print(f"Log type accuracy: {result['metrics']['log_type_accuracy']:.4f}")
            print(f"Attack F1 (micro): {result['metrics']['attack_f1_micro']:.4f}")
            print(f"Attack F1 (macro): {result['metrics']['attack_f1_macro']:.4f}")
            
            # Test model prediction
            print("\nTesting model prediction...")
            model = result['model']
            scaler = result['scaler']
            
            # Test with a few samples from each log type
            for log_type, (embeddings, labels, classes) in sampled_data.items():
                test_samples = embeddings[:5]  # Test with 5 samples
                test_samples_scaled = scaler.transform(test_samples)
                
                model.eval()
                import torch
                with torch.no_grad():
                    X_tensor = torch.from_numpy(test_samples_scaled).to(model.device)
                    outputs = model(X_tensor)
                    
                    # Get predictions
                    predicted_log_types = outputs['predicted_log_types'].cpu().numpy()
                    attack_predictions = (outputs['attack_predictions'].cpu().numpy() > 0.5).astype(int)
                    
                    print(f"\nTest results for {log_type}:")
                    print(f"  Predicted log types: {predicted_log_types}")
                    print(f"  Attack predictions shape: {attack_predictions.shape}")
                    print(f"  Sample attack predictions: {attack_predictions[0]}")
            
            return True
        else:
            print("Combined model training failed!")
            return False
            
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description="Test the combined transformer implementation")
    parser.add_argument("--sample-size", type=int, default=1000, help="Number of samples per log type to test with")
    args = parser.parse_args()
    
    print(f"{'='*60}")
    print("COMBINED TRANSFORMER TEST")
    print(f"{'='*60}")
    
    success = test_combined_model(args.sample_size)
    
    print(f"\n{'='*60}")
    if success:
        print("✓ Combined transformer test passed!")
        print("The combined model is working correctly.")
        print("You can now run the full training with: python src/transformer.py")
    else:
        print("✗ Combined transformer test failed!")
        print("Please check the error messages above.")
    print(f"{'='*60}")

if __name__ == "__main__":
    main() 