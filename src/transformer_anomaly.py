#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transformer-based Unsupervised Anomaly Detection for Log Analysis

This model handles extreme data imbalance and detects anomalies based on 
reconstruction error. The approach is inspired by:
- Autoencoder-based anomaly detection
- Transformer reconstruction capabilities

Key Innovation:
- Simplified and high-performance architecture
- Unsupervised training for better generalization
- Optimized for both GPU clusters and Apple Silicon (M2/MPS)
"""

import os
import sys
import gc
import json
import time
import pickle
import warnings
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import GradScaler, autocast

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for clusters
import matplotlib.pyplot as plt
import seaborn as sns
import re

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, accuracy_score, roc_auc_score, average_precision_score,
    confusion_matrix, precision_score, recall_score, classification_report
)

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
# Log Type Discovery and Grouping
# =============================================================================

@dataclass
class LogFamily:
    """Represents a family of related log types"""
    name: str
    log_types: List[str]
    description: str
    category: str

def discover_available_log_types() -> List[str]:
    """Discover all available log types from the embeddings directory"""
    embeddings_dir = Path("embeddings")
    if not embeddings_dir.exists():
        print("⚠️  Embeddings directory not found")
        return []
    
    log_types = [item.name for item in embeddings_dir.iterdir() if item.is_dir() and \
                 (item / f"log_{item.name}.pkl").exists() and \
                 (item / f"label_{item.name}.pkl").exists()]
    
    return sorted(log_types)

# =============================================================================
# Configuration and Resource Detection
# =============================================================================

@dataclass
class SystemConfig:
    """System configuration for optimal resource utilization"""
    device: str
    n_gpus: int
    supports_multiprocessing: bool

def detect_system_resources() -> SystemConfig:
    """Comprehensive system resource detection"""
    if torch.cuda.is_available():
        device = "cuda"
        n_gpus = torch.cuda.device_count()
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"🔥 CUDA detected: {n_gpus} GPU(s) with {gpu_memory_gb:.1f}GB memory")
    elif torch.backends.mps.is_available():
        device = "mps"
        n_gpus = 1
        print(f"🍎 Apple MPS detected")
    else:
        device = "cpu"
        n_gpus = 0
        print("💻 Using CPU for training")

    supports_multiprocessing = device != "mps"
    return SystemConfig(device=device, n_gpus=n_gpus, supports_multiprocessing=supports_multiprocessing)

# =============================================================================
# Anomaly Detection Transformer Architecture
# =============================================================================

class AnomalyDetectionTransformer(nn.Module):
    """
    Simplified Transformer-based autoencoder for anomaly detection.
    """
    def __init__(self, input_dim: int, latent_dim: int, nhead: int = 4, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.GELU(),
            nn.LayerNorm(latent_dim)
        )
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=latent_dim, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, input_dim),
        )
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def forward(self, x):
        # Unsqueeze to add sequence dimension for transformer
        x = x.unsqueeze(1)
        z = self.encoder(x)
        z = self.transformer_encoder(z)
        reconstructed = self.decoder(z)
        # Squeeze to remove sequence dimension
        return reconstructed.squeeze(1), z.squeeze(1)

# =============================================================================
# Training and Data Management
# =============================================================================

def load_and_preprocess_data(log_type: str, sample_size: int = None) -> Tuple[np.ndarray, np.ndarray, List[str], StandardScaler]:
    """Load and preprocess data for a single log type."""
    print(f"🔄 Loading data for: {log_type}...")
    log_file = Path("embeddings") / log_type / f"log_{log_type}.pkl"
    label_file = Path("embeddings") / log_type / f"label_{log_type}.pkl"

    if not log_file.exists():
        raise FileNotFoundError(f"Embedding file not found: {log_file}")

    with open(log_file, 'rb') as f:
        embeddings = pickle.load(f)
    
    with open(label_file, 'rb') as f:
        label_data = pickle.load(f)
        classes = label_data.get('classes', [])
        anomaly_flags = (label_data['vectors'].sum(axis=1) > 0).astype(np.int32)

    if sample_size and sample_size < len(embeddings):
        indices = np.random.choice(len(embeddings), size=sample_size, replace=False)
        embeddings, anomaly_flags = embeddings[indices], anomaly_flags[indices]

    scaler = StandardScaler()
    normal_embeddings = embeddings[anomaly_flags == 0]
    if len(normal_embeddings) > 0:
        scaler.fit(normal_embeddings)
        embeddings = scaler.transform(embeddings)
    
    return embeddings.astype(np.float32), anomaly_flags, classes, scaler

def train_anomaly_model(training_embeddings: np.ndarray, config: SystemConfig, log_type: str) -> AnomalyDetectionTransformer:
    """Train the anomaly detection model."""
    clear_gpu_memory()
    device = torch.device(config.device)
    embedding_dim = training_embeddings.shape[1]
    
    latent_dim = 128
    batch_size = 64 if config.device == "cuda" else 32
    transformer_layers = 2
    attention_heads = 4
    
    print(f"🏗️  Anomaly Detection Architecture:")
    print(f"   Input dim: {embedding_dim}, Latent dim: {latent_dim}")
    print(f"   Layers: {transformer_layers}, Heads: {attention_heads}, Batch size: {batch_size}")

    model = AnomalyDetectionTransformer(
        input_dim=embedding_dim,
        latent_dim=latent_dim,
        num_layers=transformer_layers,
        nhead=attention_heads
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    
    dataset = TensorDataset(torch.FloatTensor(training_embeddings).to(device))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    n_epochs = 50
    patience = 5
    best_loss = float('inf')
    epochs_no_improve = 0

    scaler = GradScaler() if config.device == "cuda" else None

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        for (x_batch,) in dataloader:
            optimizer.zero_grad()
            if scaler:
                with autocast():
                    reconstructed, _ = model(x_batch)
                    loss = F.mse_loss(reconstructed, x_batch)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                reconstructed, _ = model(x_batch)
                loss = F.mse_loss(reconstructed, x_batch)
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        scheduler.step()

        if avg_loss < best_loss:
            best_loss = avg_loss
            epochs_no_improve = 0
            checkpoint_path = Path("checkpoints") / "anomaly" / f"{log_type}_best.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_no_improve += 1

        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}/{n_epochs} | Loss: {avg_loss:.6f}")

        if epochs_no_improve >= patience:
            print(f"🛑 Early stopping at epoch {epoch}")
            break
            
    print("✅ Training complete!")
    # Load best model
    model.load_state_dict(torch.load(checkpoint_path))
    return model

def evaluate_and_save_results(model: AnomalyDetectionTransformer, embeddings: np.ndarray, 
                                anomaly_flags: np.ndarray, scaler: StandardScaler, 
                                log_type: str, config: SystemConfig):
    """Evaluate the model and save enhanced embeddings."""
    print(f"💾 Evaluating and saving results for {log_type}...")
    device = torch.device(config.device)
    model.eval()
    
    dataset = TensorDataset(torch.FloatTensor(embeddings).to(device))
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False)
    
    recon_errors = []
    with torch.no_grad():
        for (x_batch,) in dataloader:
            reconstructed, _ = model(x_batch)
            error = F.mse_loss(reconstructed, x_batch, reduction='none').mean(dim=1)
            recon_errors.append(error.cpu().numpy())
    
    recon_errors = np.concatenate(recon_errors)
    
    threshold = np.percentile(recon_errors[anomaly_flags == 0], 95) if (anomaly_flags == 0).any() else np.percentile(recon_errors, 95)
    predictions = (recon_errors > threshold).astype(int)
    
    # Create enhanced embeddings
    enhanced_embeddings = np.hstack([embeddings, predictions.reshape(-1, 1)])
    
    # Save results
    output_dir = Path("results") / "transformer_anomaly" / log_type
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "enhanced_embeddings.pkl", 'wb') as f:
        pickle.dump(enhanced_embeddings, f)
    
    # Save metrics
    metrics = {
        "f1": f1_score(anomaly_flags, predictions),
        "accuracy": accuracy_score(anomaly_flags, predictions),
        "precision": precision_score(anomaly_flags, predictions, zero_division=0),
        "recall": recall_score(anomaly_flags, predictions, zero_division=0),
        "roc_auc": roc_auc_score(anomaly_flags, recon_errors),
        "pr_auc": average_precision_score(anomaly_flags, recon_errors),
        "threshold": threshold
    }
    with open(output_dir / "metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
        
    print(f"   F1: {metrics['f1']:.4f}, ROC AUC: {metrics['roc_auc']:.4f}")
    
    # Generate and print detailed classification report
    report = generate_classification_report(anomaly_flags, predictions)
    print(report)

    # Save detailed classification report to a text file
    with open(output_dir / "classification_report.txt", 'w') as f:
        f.write(report)

def generate_classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    """Generates a detailed classification report similar to sklearn's,
    including data distribution and per-class accuracy.
    """
    total_samples = len(y_true)
    normal_samples = np.sum(y_true == 0)
    abnormal_samples = np.sum(y_true == 1)

    # Calculate confusion matrix components
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Per-class accuracy
    normal_accuracy = (tn / (tn + fp)) if (tn + fp) > 0 else 0
    abnormal_accuracy = (tp / (tp + fn)) if (tp + fn) > 0 else 0

    # Overall metrics
    overall_accuracy = accuracy_score(y_true, y_pred)
    overall_precision = precision_score(y_true, y_pred, zero_division=0)
    overall_recall = recall_score(y_true, y_pred, zero_division=0)
    overall_f1 = f1_score(y_true, y_pred, zero_division=0)

    report = f"""
Classification Report:
------------------------------------------------------
Total Samples: {total_samples}

Data Distribution:
  Normal Samples:   {normal_samples:<10} ({normal_samples / total_samples:.2%})
  Abnormal Samples: {abnormal_samples:<10} ({abnormal_samples / total_samples:.2%})

Accuracy per Class:
  Normal (Class 0): {normal_accuracy:.4f}
  Abnormal (Class 1): {abnormal_accuracy:.4f}

Overall Metrics:
  Accuracy:  {overall_accuracy:.4f}
  Precision: {overall_precision:.4f}
  Recall:    {overall_recall:.4f}
  F1-Score:  {overall_f1:.4f}

Confusion Matrix:
                  Predicted Normal   Predicted Abnormal
Actual Normal     {tn:<16}   {fp:<18}
Actual Abnormal   {fn:<16}   {tp:<18}
------------------------------------------------------
"""
    return report

def clear_gpu_memory():
    """Clear GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def main():
    """Main execution function."""
    config = detect_system_resources()
    log_types = discover_available_log_types()
    
    if not log_types:
        print("❌ No log types found. Exiting.")
        return

    for log_type in log_types:
        try:
            embeddings, anomaly_flags, classes, scaler = load_and_preprocess_data(log_type)
            
            normal_embeddings = embeddings[anomaly_flags == 0]
            if len(normal_embeddings) == 0:
                print(f"⚠️ No normal samples for {log_type}, skipping training.")
                continue

            model = train_anomaly_model(normal_embeddings, config, log_type)
            
            evaluate_and_save_results(model, embeddings, anomaly_flags, scaler, log_type, config)

        except Exception as e:
            print(f"❌ Error processing {log_type}: {e}")
        finally:
            clear_gpu_memory()

if __name__ == "__main__":
    main()