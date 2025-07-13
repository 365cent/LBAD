#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Two-Stage Transformer Prediction System

Stage 1: Predict log type from embedding
Stage 2: Use corresponding model to predict labels

Optimized for Research Alliance of Canada Nibi node with M2 GPU support.
"""

import os
import sys
import time
import json
import pickle
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import pandas as pd
from sklearn.preprocessing import StandardScaler

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Set deterministic behavior for stable results
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# =============================================================================
# Model Classes (same as transformer.py)
# =============================================================================

class OptimizedTransformerBlock(nn.Module):
    """Memory and compute optimized transformer block for Nibi"""
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, d_model * 2)
        self.linear2 = nn.Linear(d_model * 2, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Pre-norm architecture for better training stability
        x2 = self.norm1(x)
        x2, _ = self.self_attn(x2, x2, x2)
        x = x + self.dropout(x2)
        
        x2 = self.norm2(x)
        x2 = self.linear2(F.gelu(self.linear1(x2)))
        x = x + self.dropout(x2)
        
        return x

class LogTypeClassifier(nn.Module):
    """Simple classifier to predict log type from embeddings"""
    
    def __init__(self, input_dim: int, n_log_types: int, dropout: float = 0.1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 4, n_log_types)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights for stable training"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        return self.classifier(x)

class UnsupervisedMultiLabelTransformer(nn.Module):
    """Optimized transformer for unsupervised multi-label learning on Nibi"""
    
    def __init__(self, input_dim: int, latent_dim: int, n_labels: int, 
                 n_clusters: int, dropout: float = 0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.n_labels = n_labels
        
        # Encoder
        self.input_proj = nn.Linear(input_dim, latent_dim)
        self.encoder_blocks = nn.ModuleList([
            OptimizedTransformerBlock(latent_dim, 8, dropout) for _ in range(3)
        ])
        
        # Decoder
        self.decoder_blocks = nn.ModuleList([
            OptimizedTransformerBlock(latent_dim, 8, dropout) for _ in range(2)
        ])
        self.output_proj = nn.Linear(latent_dim, input_dim)
        
        # Multi-label head
        self.label_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim // 2, n_labels)
            # No Sigmoid - using logits for mixed precision safety
        )
        
        # Cluster matcher
        self.cluster_head = nn.Linear(latent_dim, n_clusters)
        
        # Initialize weights for stable training
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights for stable training"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        
    def forward(self, x):
        # Encode
        z = self.input_proj(x).unsqueeze(1)  # Add sequence dimension
        for block in self.encoder_blocks:
            z = block(z)
        z_flat = z.squeeze(1)
        
        # Decode
        z_dec = z
        for block in self.decoder_blocks:
            z_dec = block(z_dec)
        x_recon = self.output_proj(z_dec.squeeze(1))
        
        # Predictions
        labels = self.label_head(z_flat)
        clusters = self.cluster_head(z_flat)
        
        return {
            'latent': z_flat,
            'reconstructed': x_recon,
            'labels': labels,
            'clusters': clusters
        }

# =============================================================================
# System Configuration
# =============================================================================

def detect_system_resources():
    """Detect system resources for optimal performance"""
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        device = "cuda"
        
        gpu_name = torch.cuda.get_device_properties(0).name
        print(f"Detected {n_gpus} GPU(s): {gpu_name}")
        print(f"GPU Memory: {gpu_memory_gb:.1f} GB per GPU")
        
    elif torch.backends.mps.is_available():
        n_gpus = 1
        gpu_memory_gb = 16.0  # Approximate for M2
        device = "mps"
        print("Detected MPS (Metal) device")
    else:
        n_gpus = 0
        gpu_memory_gb = 0
        device = "cpu"
        print("Using CPU device")

    return {
        'device': device,
        'n_gpus': n_gpus,
        'gpu_memory_gb': gpu_memory_gb
    }

# =============================================================================
# Model Loading and Prediction
# =============================================================================

class TwoStagePredictor:
    """Two-stage prediction system: log type -> labels"""
    
    def __init__(self, models_dir: Path, device: str):
        self.models_dir = models_dir
        self.device = torch.device(device)
        self.log_type_classifier = None
        self.label_models = {}
        self.log_type_mapping = {}
        self.scalers = {}
        
        self.load_models()
    
    def load_models(self):
        """Load all trained models"""
        print("Loading models...")
        
        # Load log type classifier
        classifier_files = list(self.models_dir.glob("log_type_classifier_*.pth"))
        if not classifier_files:
            raise FileNotFoundError("No log type classifier found. Please train models first.")
        
        classifier_path = classifier_files[0]  # Use the first one found
        print(f"Loading log type classifier from: {classifier_path}")
        
        # Fix for PyTorch 2.6 weights_only issue
        try:
            checkpoint = torch.load(classifier_path, map_location=self.device, weights_only=False)
        except Exception as e:
            print(f"Warning: Failed to load with weights_only=False, trying alternative method: {e}")
            # Try with safe globals
            from torch.serialization import add_safe_globals
            add_safe_globals([StandardScaler])
            checkpoint = torch.load(classifier_path, map_location=self.device)
        
        self.log_type_mapping = checkpoint['log_type_to_idx']
        self.idx_to_log_type = checkpoint['idx_to_log_type']
        
        # Create and load classifier
        input_dim = 768  # Standard embedding dimension
        n_log_types = len(self.idx_to_log_type)
        self.log_type_classifier = LogTypeClassifier(input_dim, n_log_types).to(self.device)
        self.log_type_classifier.load_state_dict(checkpoint['model_state_dict'])
        self.log_type_classifier.eval()
        
        # Load label prediction models for each log type
        for log_type in self.idx_to_log_type:
            model_files = list(self.models_dir.glob(f"transformer_{log_type}_*.pth"))
            if model_files:
                model_path = model_files[0]  # Use the first one found
                print(f"Loading label model for {log_type} from: {model_path}")
                
                # Fix for PyTorch 2.6 weights_only issue
                try:
                    checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                except Exception as e:
                    print(f"Warning: Failed to load with weights_only=False, trying alternative method: {e}")
                    # Try with safe globals
                    from torch.serialization import add_safe_globals
                    add_safe_globals([StandardScaler])
                    checkpoint = torch.load(model_path, map_location=self.device)
                
                classes = checkpoint.get('classes', [])
                
                if classes:
                    # Create model
                    input_dim = 768  # Standard embedding dimension
                    latent_dim = min(256, input_dim)
                    n_labels = len(classes)
                    n_clusters = min(8, n_labels)
                    
                    model = UnsupervisedMultiLabelTransformer(
                        input_dim=input_dim,
                        latent_dim=latent_dim,
                        n_labels=n_labels,
                        n_clusters=n_clusters
                    ).to(self.device)
                    
                    model.load_state_dict(checkpoint['model_state_dict'])
                    model.eval()
                    
                    self.label_models[log_type] = {
                        'model': model,
                        'classes': classes,
                        'scaler': checkpoint.get('scaler')
                    }
        
        print(f"Loaded {len(self.label_models)} label prediction models")
    
    def predict_log_type(self, embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Stage 1: Predict log type for each embedding"""
        if self.log_type_classifier is None:
            raise ValueError("Log type classifier not loaded")
        
        # Normalize embeddings using the classifier's scaler
        scaler = StandardScaler()
        embeddings_norm = scaler.fit_transform(embeddings).astype(np.float32)
        
        # Predict
        predictions = []
        batch_size = 256
        
        with torch.no_grad():
            for i in range(0, len(embeddings_norm), batch_size):
                batch = torch.from_numpy(embeddings_norm[i:i+batch_size]).to(self.device)
                outputs = self.log_type_classifier(batch)
                probs = F.softmax(outputs, dim=1)
                predictions.append(probs.cpu().numpy())
        
        predictions = np.vstack(predictions)
        predicted_indices = np.argmax(predictions, axis=1)
        predicted_log_types = [self.idx_to_log_type[idx] for idx in predicted_indices]
        
        return np.array(predicted_log_types), predictions
    
    def predict_labels(self, embeddings: np.ndarray, log_types: np.ndarray) -> Dict[str, Any]:
        """Stage 2: Predict labels using appropriate models for each log type"""
        
        results = {
            'predictions': {},
            'probabilities': {},
            'log_type_counts': {},
            'all_predictions': [],
            'all_probabilities': [],
            'all_classes': []
        }
        
        # Group embeddings by predicted log type
        unique_log_types = np.unique(log_types)
        
        for log_type in unique_log_types:
            if log_type not in self.label_models:
                print(f"Warning: No model found for log type '{log_type}'")
                continue
            
            # Get embeddings for this log type
            mask = log_types == log_type
            log_type_embeddings = embeddings[mask]
            
            if len(log_type_embeddings) == 0:
                continue
            
            print(f"Predicting labels for {len(log_type_embeddings)} samples of type '{log_type}'")
            
            # Get model and classes
            model_info = self.label_models[log_type]
            model = model_info['model']
            classes = model_info['classes']
            scaler = model_info['scaler']
            
            # Normalize embeddings
            if scaler is not None:
                embeddings_norm = scaler.transform(log_type_embeddings).astype(np.float32)
            else:
                embeddings_norm = log_type_embeddings.astype(np.float32)
            
            # Predict
            predictions = []
            batch_size = 256
            
            with torch.no_grad():
                for i in range(0, len(embeddings_norm), batch_size):
                    batch = torch.from_numpy(embeddings_norm[i:i+batch_size]).to(self.device)
                    outputs = model(batch)
                    logits = outputs['labels']
                    probs = torch.sigmoid(logits)
                    predictions.append(probs.cpu().numpy())
            
            predictions = np.vstack(predictions)
            binary_predictions = (predictions > 0.5).astype(int)
            
            # Store results
            results['predictions'][log_type] = binary_predictions
            results['probabilities'][log_type] = predictions
            results['log_type_counts'][log_type] = len(log_type_embeddings)
            
            # Add to combined results
            results['all_predictions'].extend(binary_predictions)
            results['all_probabilities'].extend(predictions)
            results['all_classes'].extend([classes] * len(binary_predictions))
        
        # Convert to arrays
        if results['all_predictions']:
            results['all_predictions'] = np.array(results['all_predictions'])
            results['all_probabilities'] = np.array(results['all_probabilities'])
        
        return results
    
    def predict(self, embeddings: np.ndarray) -> Dict[str, Any]:
        """Complete two-stage prediction pipeline"""
        print("Stage 1: Predicting log types...")
        log_types, log_type_probs = self.predict_log_type(embeddings)
        
        print("Stage 2: Predicting labels...")
        label_results = self.predict_labels(embeddings, log_types)
        
        # Combine results
        results = {
            'log_types': log_types,
            'log_type_probabilities': log_type_probs,
            'label_predictions': label_results['all_predictions'],
            'label_probabilities': label_results['all_probabilities'],
            'label_classes': label_results['all_classes'],
            'log_type_counts': label_results['log_type_counts'],
            'per_log_type_results': label_results['predictions']
        }
        
        return results

# =============================================================================
# Main Prediction Function
# =============================================================================

def find_embedding_file(log_type: str) -> str:
    """Find the embedding file for a given log type"""
    embeddings_dir = Path("embeddings")
    
    # Check if it's a direct file path
    if Path(log_type).exists():
        return log_type
    
    # Check if it's a log type name
    log_file = embeddings_dir / log_type / f"log_{log_type}.pkl"
    if log_file.exists():
        return str(log_file)
    
    # Check if it's in the root embeddings directory
    log_file = embeddings_dir / f"log_{log_type}.pkl"
    if log_file.exists():
        return str(log_file)
    
    # List available options
    available_files = []
    for file_path in embeddings_dir.rglob("log_*.pkl"):
        available_files.append(str(file_path))
    
    raise FileNotFoundError(
        f"Embedding file not found for '{log_type}'. "
        f"Available files:\n" + "\n".join(available_files)
    )

def predict_on_embeddings(embeddings_path: str, output_dir: str = "predictions"):
    """Main prediction function"""
    
    # Setup
    config = detect_system_resources()
    models_dir = Path("models")
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    if not models_dir.exists():
        raise FileNotFoundError("Models directory not found. Please train models first.")
    
    # Find the actual embedding file
    actual_embeddings_path = find_embedding_file(embeddings_path)
    print(f"Loading embeddings from: {actual_embeddings_path}")
    
    # Load embeddings
    with open(actual_embeddings_path, 'rb') as f:
        embeddings = pickle.load(f)
    
    print(f"Loaded {len(embeddings)} embeddings with shape {embeddings.shape}")
    
    # Initialize predictor
    predictor = TwoStagePredictor(models_dir, config['device'])
    
    # Run prediction
    start_time = time.time()
    results = predictor.predict(embeddings)
    prediction_time = time.time() - start_time
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save detailed results
    results_file = output_dir / f"prediction_results_{timestamp}.pkl"
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    
    # Save labels in evaluation format
    if len(results['label_predictions']) > 0:
        # Get the most common classes (assuming all models have similar class structure)
        all_classes = results['label_classes']
        if all_classes:
            # Use classes from the first model as reference
            reference_classes = all_classes[0]
            
            # Create label data in evaluation format
            label_data = {
                'vectors': results['label_predictions'].astype(np.int8),
                'classes': reference_classes,
                'probabilities': results['label_probabilities'].astype(np.float32),
                'log_types': results['log_types'],
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'model_type': 'two_stage_transformer',
                    'threshold': 0.5,
                    'prediction_time_seconds': prediction_time,
                    'n_samples': len(embeddings)
                }
            }
            
            label_file = output_dir / f"label_predictions_{timestamp}.pkl"
            with open(label_file, 'wb') as f:
                pickle.dump(label_data, f)
            
            print(f"Labels saved to: {label_file}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("PREDICTION SUMMARY")
    print(f"{'='*60}")
    print(f"Total samples: {len(embeddings)}")
    print(f"Prediction time: {prediction_time:.2f} seconds")
    print(f"Average time per sample: {prediction_time/len(embeddings)*1000:.2f} ms")
    
    print(f"\nLog type distribution:")
    unique_log_types, counts = np.unique(results['log_types'], return_counts=True)
    for log_type, count in zip(unique_log_types, counts):
        percentage = count / len(embeddings) * 100
        print(f"  {log_type}: {count} samples ({percentage:.1f}%)")
    
    if len(results['label_predictions']) > 0:
        avg_labels = np.mean(results['label_predictions'].sum(axis=1))
        print(f"\nLabel prediction summary:")
        print(f"  Average labels per sample: {avg_labels:.2f}")
        print(f"  Total label predictions: {len(results['label_predictions'])}")
    
    print(f"\nResults saved to: {results_file}")
    print(f"{'='*60}")
    
    return results

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Two-stage transformer prediction")
    parser.add_argument("--embeddings", "-e", required=True, 
                       help="Log type name (e.g., 'wp-error', 'wp-access') or path to embeddings file")
    parser.add_argument("--output", "-o", default="predictions",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    try:
        results = predict_on_embeddings(args.embeddings, args.output)
        print("Prediction completed successfully!")
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 