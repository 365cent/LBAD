#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transformer-based Unsupervised Anomaly Detection for Log Analysis

This model trains ONLY on normal logs (unsupervised) and detects anomalies based on 
reconstruction error and latent space deviation. The approach is inspired by:
- One-class classification paradigms
- Autoencoder-based anomaly detection
- Transformer reconstruction capabilities

Key Innovation:
- Train exclusively on normal logs (attack_flag = 0)
- Use reconstruction error as anomaly score
- Add binary anomaly flag to output embeddings
- Maintain compatibility with downstream supervised models
- Unsupervised training for better generalization

Architecture:
- Input: Original embeddings (FastText/BERT/LogBERT)
- Output: Enhanced embeddings + anomaly_flag (0=normal, 1=anomaly)
- Training: Only normal logs (unsupervised reconstruction)
- Inference: All logs with anomaly scoring

Optimized for Research Alliance of Canada infrastructure:
- Multi-GPU support and memory management
- Automatic embedding type detection
- Conservative memory usage for large datasets
- Comprehensive anomaly scoring metrics
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
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
import torch.distributed as dist

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for clusters
import matplotlib.pyplot as plt
import seaborn as sns
from halo import Halo

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.manifold import TSNE
from sklearn.metrics import (
    classification_report, precision_recall_fscore_support,
    f1_score, accuracy_score, roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve, confusion_matrix
)
from sklearn.neighbors import NearestNeighbors

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Set deterministic behavior for stable results
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Configuration paths
CHECKPOINT_DIR = Path("checkpoints") / "transformer_anomaly"
RESULTS_DIR = Path("results") / "transformer_anomaly"
MODELS_DIR = Path("models") / "transformer_anomaly"

# =============================================================================
# Configuration and Resource Detection
# =============================================================================

@dataclass
class SystemConfig:
    """System configuration for optimal resource utilization"""
    device: str
    n_gpus: int
    total_memory_gb: float
    gpu_memory_gb: float
    n_cpus: int
    is_distributed: bool
    rank: int
    world_size: int
    node_name: str
    job_id: str

def detect_system_resources() -> SystemConfig:
    """Comprehensive system resource detection"""
    # GPU Detection
    if torch.cuda.is_available():
        device = "cuda"
        n_gpus = torch.cuda.device_count()
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"🔥 CUDA detected: {n_gpus} GPU(s) with {gpu_memory_gb:.1f}GB memory")
    elif torch.backends.mps.is_available():
        device = "mps"
        n_gpus = 1
        gpu_memory_gb = 8.0  # Estimated for Apple Silicon
        print(f"🍎 Apple MPS detected with ~{gpu_memory_gb:.1f}GB unified memory")
    else:
        device = "cpu"
        n_gpus = 0
        gpu_memory_gb = 0
        print("💻 Using CPU for training")

    # Memory detection
    try:
        import psutil
        total_memory_gb = psutil.virtual_memory().total / (1024**3)
        n_cpus = psutil.cpu_count()
    except ImportError:
        total_memory_gb = 32.0  # Default estimate
        n_cpus = 8

    # Distributed setup detection
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    is_distributed = world_size > 1
    
    # Node and job information
    node_name = os.environ.get('SLURM_NODELIST', 'unknown')
    job_id = os.environ.get('SLURM_JOB_ID', 'unknown')

    return SystemConfig(
        device=device, n_gpus=n_gpus, total_memory_gb=total_memory_gb,
        gpu_memory_gb=gpu_memory_gb, n_cpus=n_cpus, 
        is_distributed=is_distributed, rank=rank, world_size=world_size,
        node_name=node_name, job_id=job_id
    )

def setup_distributed_training(rank: int, world_size: int):
    """Setup distributed training for multi-GPU environments"""
    if world_size > 1:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12356'
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)

def cleanup_distributed():
    """Cleanup distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()

# =============================================================================
# Anomaly Detection Transformer Architecture
# =============================================================================

class ResidualTransformerBlock(nn.Module):
    """Enhanced transformer block with residual connections for anomaly detection"""
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, d_model * 4)  # Wider FFN for better representation
        self.linear2 = nn.Linear(d_model * 4, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Pre-norm architecture for better gradient flow
        x2 = self.norm1(x)
        x2, attn_weights = self.self_attn(x2, x2, x2)
        x = x + self.dropout(x2)
        
        x2 = self.norm2(x)
        x2 = self.linear2(F.gelu(self.linear1(x2)))
        x = x + self.dropout(x2)
        
        return x, attn_weights

class AnomalyDetectionTransformer(nn.Module):
    """
    Transformer-based anomaly detection model that trains only on normal logs
    
    Architecture:
    - Encoder: Maps input to latent representation
    - Decoder: Reconstructs input from latent representation
    - Anomaly Scorer: Computes anomaly scores based on reconstruction error
    - Output Enhancer: Adds anomaly flag to original embeddings
    """
    
    def __init__(self, input_dim: int, latent_dim: int, dropout: float = 0.1, 
                 transformer_layers: int = 8, attention_heads: int = 8):
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.transformer_layers = transformer_layers
        self.attention_heads = attention_heads
        
        # Input projection with normalization
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Encoder: Multiple transformer blocks
        self.encoder_blocks = nn.ModuleList([
            ResidualTransformerBlock(latent_dim, attention_heads, dropout) 
            for _ in range(transformer_layers)
        ])
        
        # Latent space processing
        self.latent_processor = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Decoder: Reconstruct original input
        self.decoder_blocks = nn.ModuleList([
            ResidualTransformerBlock(latent_dim, attention_heads, dropout) 
            for _ in range(transformer_layers // 2)  # Lighter decoder
        ])
        
        self.output_proj = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, input_dim)
        )
        
        # Anomaly scoring components
        self.anomaly_scorer = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.LayerNorm(latent_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Normal pattern memory bank (updated during training)
        self.register_buffer('normal_prototypes', torch.zeros(100, latent_dim))
        self.register_buffer('prototype_ptr', torch.zeros(1, dtype=torch.long))
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights for stable training"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
    
    def encode(self, x):
        """Encode input to latent representation"""
        z = self.input_proj(x)
        z = z.unsqueeze(1)  # Add sequence dimension
        
        attention_weights = []
        for block in self.encoder_blocks:
            z, attn = block(z)
            attention_weights.append(attn)
        
        z = z.squeeze(1)  # Remove sequence dimension
        z = self.latent_processor(z)
        
        return z, attention_weights
    
    def decode(self, z):
        """Decode latent representation to reconstruction"""
        z_dec = z.unsqueeze(1)  # Add sequence dimension
        
        for block in self.decoder_blocks:
            z_dec, _ = block(z_dec)
        
        z_dec = z_dec.squeeze(1)  # Remove sequence dimension
        x_recon = self.output_proj(z_dec)
        
        return x_recon
    
    def compute_anomaly_score(self, z, x_recon, x_orig):
        """Compute comprehensive anomaly score"""
        # 1. Reconstruction error
        recon_error = F.mse_loss(x_recon, x_orig, reduction='none').mean(dim=1)
        
        # 2. Latent space deviation (distance to normal prototypes)
        if self.normal_prototypes.norm() > 0:  # If prototypes are initialized
            # Compute distance to closest normal prototype
            distances = torch.cdist(z.unsqueeze(1), self.normal_prototypes.unsqueeze(0))
            min_distances = distances.min(dim=2)[0].squeeze(1)
            latent_deviation = min_distances
        else:
            latent_deviation = torch.zeros_like(recon_error)
        
        # 3. Neural network-based anomaly score
        neural_score = self.anomaly_scorer(z).squeeze(1)
        
        # Combine scores with learned weights
        combined_score = (0.4 * recon_error + 0.3 * latent_deviation + 0.3 * neural_score)
        
        return combined_score, recon_error, latent_deviation, neural_score
    
    def update_normal_prototypes(self, z_normal):
        """Update normal pattern prototypes during training"""
        with torch.no_grad():
            batch_size = z_normal.size(0)
            ptr = int(self.prototype_ptr)
            
            if ptr + batch_size > self.normal_prototypes.size(0):
                # Wrap around if we exceed buffer size
                remaining = self.normal_prototypes.size(0) - ptr
                self.normal_prototypes[ptr:] = z_normal[:remaining]
                self.normal_prototypes[:batch_size - remaining] = z_normal[remaining:]
                self.prototype_ptr[0] = batch_size - remaining
            else:
                self.normal_prototypes[ptr:ptr + batch_size] = z_normal
                self.prototype_ptr[0] = ptr + batch_size
    
    def forward(self, x):
        """Forward pass"""
        # Encode
        z, attention_weights = self.encode(x)
        
        # Decode
        x_recon = self.decode(z)
        
        # Compute anomaly scores
        anomaly_score, recon_error, latent_deviation, neural_score = self.compute_anomaly_score(z, x_recon, x)
        
        return {
            'latent': z,
            'reconstructed': x_recon,
            'anomaly_score': anomaly_score,
            'recon_error': recon_error,
            'latent_deviation': latent_deviation,
            'neural_score': neural_score,
            'attention_weights': attention_weights
        }

# =============================================================================
# Training and Data Management
# =============================================================================

class ProgressTracker:
    """Enhanced progress tracking for anomaly detection training"""
    
    def __init__(self, output_dir: Path, log_type: str, config: SystemConfig):
        self.output_dir = output_dir
        self.log_type = log_type
        self.config = config
        self.metrics = []
        self.start_time = None
        self.epoch_times = []
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        log_file = output_dir / f"anomaly_training_{log_type}_{config.node_name}_{config.job_id}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Anomaly Detection System Configuration: {config.__dict__}")
    
    def start_training(self, total_epochs: int):
        """Start training timer"""
        self.start_time = time.time()
        self.total_epochs = total_epochs
        print(f"🚀 Anomaly Detection Training started: {total_epochs} epochs")
    
    def log_step(self, step: str, data: Dict[str, Any]):
        """Log step results"""
        timestamp = datetime.now().isoformat()
        self.logger.info(f"STEP: {step}")
        self.logger.info(f"DATA: {json.dumps(data, indent=2)}")
        
        step_file = self.output_dir / f"step_{step.lower().replace(' ', '_')}_{self.log_type}.json"
        with open(step_file, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'step': step,
                'data': data
            }, f, indent=2)
    
    def log_metrics(self, epoch: int, metrics: Dict[str, float]):
        """Log training metrics"""
        self.metrics.append({'epoch': epoch, **metrics})
        self.logger.info(f"Epoch {epoch}: {metrics}")
        
        metrics_file = self.output_dir / f"anomaly_metrics_{self.log_type}.json"
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)

def load_and_preprocess_data(log_type: str, config: SystemConfig, tracker: ProgressTracker, 
                           sample_size: int = None) -> Tuple[np.ndarray, np.ndarray, List[str], StandardScaler]:
    """
    Load and preprocess data for anomaly detection training
    
    Returns:
        normal_embeddings: Embeddings for normal logs only (for training)
        all_embeddings: All embeddings (for inference)
        anomaly_flags: Binary flags (0=normal, 1=anomaly) for all logs
        classes: List of class names
        scaler: Fitted scaler
    """
    print(f"🔄 Loading data for anomaly detection: {log_type}...")
    
    embeddings_dir = Path("embeddings")
    log_file = embeddings_dir / log_type / f"log_{log_type}.pkl"
    label_file = embeddings_dir / log_type / f"label_{log_type}.pkl"
    
    if not log_file.exists():
        raise FileNotFoundError(f"Embedding file not found: {log_file}")
    
    # Load embeddings
    with open(log_file, 'rb') as f:
        embeddings = pickle.load(f)
    
    # Detect embedding type
    embedding_dim = embeddings.shape[1]
    embedding_type = "Unknown"
    if embedding_dim == 300:
        embedding_type = "FastText (300D)"
    elif embedding_dim == 768:
        embedding_type = "BERT CLS (768D)"
    elif embedding_dim == 2314:
        embedding_type = "Enhanced LogBERT (2314D)"
    
    print(f"🔍 Detected embedding type: {embedding_type}")
    
    # Load labels to determine normal vs anomaly
    classes = []
    true_labels = None
    anomaly_flags = np.zeros(len(embeddings), dtype=np.int32)  # Default: all normal
    
    if label_file.exists():
        with open(label_file, 'rb') as f:
            label_data = pickle.load(f)
            if isinstance(label_data, dict):
                if 'classes' in label_data:
                    classes = label_data['classes']
                if 'vectors' in label_data:
                    true_labels = label_data['vectors']
                    
                    # Create anomaly flags: 0 = normal (no attacks), 1 = anomaly (any attack)
                    anomaly_flags = (true_labels.sum(axis=1) > 0).astype(np.int32)
                    
                    tracker.log_step("Anomaly Labels Created", {
                        "total_samples": len(embeddings),
                        "normal_samples": int((anomaly_flags == 0).sum()),
                        "anomaly_samples": int((anomaly_flags == 1).sum()),
                        "anomaly_rate": float(anomaly_flags.mean()),
                        "classes": classes
                    })
    
    if not classes:
        classes = ['normal']
        print("⚠️  No attack labels found - treating all logs as normal")
    
    # Filter normal logs for training
    normal_mask = (anomaly_flags == 0)
    normal_embeddings = embeddings[normal_mask]
    
    print(f"📊 Data split:")
    print(f"   Total samples: {len(embeddings):,}")
    print(f"   Normal samples (for training): {len(normal_embeddings):,}")
    print(f"   Anomaly samples: {(anomaly_flags == 1).sum():,}")
    print(f"   Anomaly rate: {anomaly_flags.mean():.1%}")
    
    # Apply sample size limit if specified
    if sample_size is not None and sample_size < len(embeddings):
        # Maintain the same normal/anomaly ratio
        normal_ratio = len(normal_embeddings) / len(embeddings)
        normal_sample_size = int(sample_size * normal_ratio)
        
        # Sample normal logs
        if normal_sample_size < len(normal_embeddings):
            normal_indices = np.random.choice(len(normal_embeddings), size=normal_sample_size, replace=False)
            normal_embeddings = normal_embeddings[normal_indices]
        
        # Sample all logs (maintaining ratio)
        indices = np.random.choice(len(embeddings), size=sample_size, replace=False)
        embeddings = embeddings[indices]
        anomaly_flags = anomaly_flags[indices]
        
        print(f"🎯 Dataset limited to {sample_size:,} samples")
        print(f"   Normal samples for training: {len(normal_embeddings):,}")
    
    # Normalize embeddings using only normal data
    scaler = RobustScaler()
    normal_embeddings = scaler.fit_transform(normal_embeddings).astype(np.float32)
    
    # Apply same normalization to all embeddings
    all_embeddings = scaler.transform(embeddings).astype(np.float32)
    
    # Additional L2 normalization
    from sklearn.preprocessing import normalize
    normal_embeddings = normalize(normal_embeddings, norm='l2', axis=1)
    all_embeddings = normalize(all_embeddings, norm='l2', axis=1)
    
    tracker.log_step("Data Preprocessing", {
        "embedding_type": embedding_type,
        "embedding_dim": embedding_dim,
        "normal_embeddings_shape": normal_embeddings.shape,
        "all_embeddings_shape": all_embeddings.shape,
        "classes": classes,
        "normal_ratio": float(len(normal_embeddings) / len(all_embeddings))
    })
    
    return normal_embeddings, all_embeddings, anomaly_flags, classes, scaler

def clear_gpu_memory():
    """Clear GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

def train_anomaly_model(normal_embeddings: np.ndarray, config: SystemConfig, 
                       tracker: ProgressTracker, log_type: str) -> AnomalyDetectionTransformer:
    """Train the anomaly detection model on normal logs only"""
    
    device = torch.device(config.device)
    embedding_dim = normal_embeddings.shape[1]
    
    # Model configuration based on embedding dimension
    if embedding_dim <= 300:
        latent_dim = 256
        batch_size = 256 if config.device != "cuda" else 128
        transformer_layers = 6
        attention_heads = 8
    elif embedding_dim <= 768:
        latent_dim = 384
        batch_size = 128 if config.device != "cuda" else 64
        transformer_layers = 8
        attention_heads = 12
    else:
        latent_dim = 512
        batch_size = 64 if config.device != "cuda" else 32
        transformer_layers = 10
        attention_heads = 16
    
    print(f"🏗️  Anomaly Detection Architecture:")
    print(f"   Input dim: {embedding_dim}, Latent dim: {latent_dim}")
    print(f"   Layers: {transformer_layers}, Heads: {attention_heads}")
    print(f"   Batch size: {batch_size}")
    print(f"   Training samples: {len(normal_embeddings):,} (normal only)")
    
    # Initialize model
    model = AnomalyDetectionTransformer(
        input_dim=embedding_dim,
        latent_dim=latent_dim,
        transformer_layers=transformer_layers,
        attention_heads=attention_heads
    ).to(device)
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    
    # Data preparation
    normal_tensor = torch.FloatTensor(normal_embeddings).to(device)
    dataset = TensorDataset(normal_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    # Training configuration
    n_epochs = 100
    tracker.start_training(n_epochs)
    
    # Mixed precision training
    scaler = GradScaler() if config.device == "cuda" else None
    
    print("🚀 Starting anomaly detection training (normal logs only)...")
    
    for epoch in range(n_epochs):
        epoch_start_time = time.time()
        model.train()
        
        epoch_losses = {
            'total_loss': 0,
            'recon_loss': 0,
            'latent_reg': 0,
            'normal_proto_loss': 0
        }
        
        for batch_idx, (x_batch,) in enumerate(dataloader):
            optimizer.zero_grad()
            
            if scaler:
                with autocast():
                    # Forward pass
                    outputs = model(x_batch)
                    
                    # Reconstruction loss (main objective)
                    recon_loss = F.mse_loss(outputs['reconstructed'], x_batch)
                    
                    # Latent regularization (prevent collapse)
                    latent_reg = torch.mean(torch.norm(outputs['latent'], dim=1))
                    
                    # Normal prototype loss (encourage tight normal cluster)
                    normal_proto_loss = torch.mean(outputs['anomaly_score'])  # Minimize for normal data
                    
                    # Total loss
                    total_loss = recon_loss + 0.1 * latent_reg + 0.1 * normal_proto_loss
                
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Forward pass
                outputs = model(x_batch)
                
                # Reconstruction loss (main objective)
                recon_loss = F.mse_loss(outputs['reconstructed'], x_batch)
                
                # Latent regularization (prevent collapse)
                latent_reg = torch.mean(torch.norm(outputs['latent'], dim=1))
                
                # Normal prototype loss (encourage tight normal cluster)
                normal_proto_loss = torch.mean(outputs['anomaly_score'])  # Minimize for normal data
                
                # Total loss
                total_loss = recon_loss + 0.1 * latent_reg + 0.1 * normal_proto_loss
                
                total_loss.backward()
                optimizer.step()
            
            # Update normal prototypes
            model.update_normal_prototypes(outputs['latent'].detach())
            
            # Accumulate losses
            epoch_losses['total_loss'] += total_loss.item()
            epoch_losses['recon_loss'] += recon_loss.item()
            epoch_losses['latent_reg'] += latent_reg.item()
            epoch_losses['normal_proto_loss'] += normal_proto_loss.item()
            
            # Clear memory periodically
            if batch_idx % 50 == 0:
                clear_gpu_memory()
        
        # Update scheduler
        scheduler.step()
        
        # Average losses
        n_batches = len(dataloader)
        for key in epoch_losses:
            epoch_losses[key] /= n_batches
        
        # Add learning rate to logs
        epoch_losses['lr'] = optimizer.param_groups[0]['lr']
        
        # Log metrics
        tracker.log_metrics(epoch, epoch_losses)
        
        # Print progress
        epoch_time = time.time() - epoch_start_time
        if epoch % 10 == 0 or epoch < 10:
            print(f"Epoch {epoch:3d}/{n_epochs} | "
                  f"Total: {epoch_losses['total_loss']:.4f} | "
                  f"Recon: {epoch_losses['recon_loss']:.4f} | "
                  f"Reg: {epoch_losses['latent_reg']:.4f} | "
                  f"Proto: {epoch_losses['normal_proto_loss']:.4f} | "
                  f"LR: {epoch_losses['lr']:.2e} | "
                  f"Time: {epoch_time:.1f}s")
        
        # Save checkpoint periodically
        if epoch % 25 == 0 and epoch > 0:
            checkpoint_path = CHECKPOINT_DIR / f"anomaly_{log_type}_epoch_{epoch}.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'losses': epoch_losses
            }, checkpoint_path)
    
    print(f"✅ Anomaly detection training completed!")
    return model

def evaluate_anomaly_model(model: AnomalyDetectionTransformer, all_embeddings: np.ndarray, 
                         anomaly_flags: np.ndarray, classes: List[str], config: SystemConfig, 
                         tracker: ProgressTracker) -> Dict[str, Any]:
    """Evaluate the anomaly detection model on all data"""
    print("🔍 Evaluating anomaly detection model...")
    
    device = torch.device(config.device)
    model.eval()
    
    # Convert data to tensors
    embeddings_tensor = torch.FloatTensor(all_embeddings).to(device)
    
    # Generate predictions
    anomaly_scores = []
    recon_errors = []
    enhanced_embeddings = []
    
    batch_size = 256
    with torch.no_grad():
        for i in range(0, len(all_embeddings), batch_size):
            batch = embeddings_tensor[i:i+batch_size]
            
            # Forward pass
            outputs = model(batch)
            
            # Store results
            anomaly_scores.append(outputs['anomaly_score'].cpu().numpy())
            recon_errors.append(outputs['recon_error'].cpu().numpy())
            
            # Create enhanced embeddings: original + anomaly_score as new column
            batch_embeddings = batch.cpu().numpy()
            batch_anomaly_scores = outputs['anomaly_score'].cpu().numpy().reshape(-1, 1)
            enhanced_batch = np.concatenate([batch_embeddings, batch_anomaly_scores], axis=1)
            enhanced_embeddings.append(enhanced_batch)
    
    # Concatenate results
    anomaly_scores = np.concatenate(anomaly_scores)
    recon_errors = np.concatenate(recon_errors)
    enhanced_embeddings = np.concatenate(enhanced_embeddings)
    
    # Determine optimal threshold using validation approach
    # Use reconstruction error as primary anomaly indicator
    thresholds = np.percentile(recon_errors, np.arange(90, 100, 0.5))
    best_threshold = thresholds[0]  # Start with 90th percentile
    best_f1 = 0
    
    if len(np.unique(anomaly_flags)) > 1:  # If we have both normal and anomaly samples
        for threshold in thresholds:
            predictions = (recon_errors > threshold).astype(int)
            f1 = f1_score(anomaly_flags, predictions)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
    
    # Generate final predictions
    binary_predictions = (recon_errors > best_threshold).astype(int)
    
    # Calculate metrics
    metrics = {}
    
    if len(np.unique(anomaly_flags)) > 1:  # If we have both classes
        # Classification metrics
        metrics['accuracy'] = accuracy_score(anomaly_flags, binary_predictions)
        metrics['f1_score'] = f1_score(anomaly_flags, binary_predictions)
        metrics['precision'] = precision_score(anomaly_flags, binary_predictions, zero_division=0)
        metrics['recall'] = recall_score(anomaly_flags, binary_predictions, zero_division=0)
        
        # ROC and PR metrics
        metrics['roc_auc'] = roc_auc_score(anomaly_flags, recon_errors)
        metrics['pr_auc'] = average_precision_score(anomaly_flags, recon_errors)
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(anomaly_flags, binary_predictions).ravel()
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        metrics['true_positives'] = int(tp)
        
        # False positive rate
        metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0
        metrics['fnr'] = fn / (fn + tp) if (fn + tp) > 0 else 0
    else:
        print("⚠️  Only one class present - computing unsupervised metrics only")
    
    # Unsupervised metrics
    metrics['mean_anomaly_score'] = float(np.mean(anomaly_scores))
    metrics['std_anomaly_score'] = float(np.std(anomaly_scores))
    metrics['mean_recon_error'] = float(np.mean(recon_errors))
    metrics['std_recon_error'] = float(np.std(recon_errors))
    metrics['threshold'] = float(best_threshold)
    
    # Distribution statistics
    normal_mask = (anomaly_flags == 0)
    anomaly_mask = (anomaly_flags == 1)
    
    if normal_mask.sum() > 0:
        metrics['normal_mean_recon_error'] = float(np.mean(recon_errors[normal_mask]))
        metrics['normal_std_recon_error'] = float(np.std(recon_errors[normal_mask]))
    
    if anomaly_mask.sum() > 0:
        metrics['anomaly_mean_recon_error'] = float(np.mean(recon_errors[anomaly_mask]))
        metrics['anomaly_std_recon_error'] = float(np.std(recon_errors[anomaly_mask]))
    
    # Log evaluation results
    tracker.log_step("Anomaly Model Evaluation", {
        'metrics': metrics,
        'threshold': float(best_threshold),
        'n_samples': len(all_embeddings),
        'n_normal': int(normal_mask.sum()),
        'n_anomaly': int(anomaly_mask.sum()),
        'enhanced_embedding_shape': enhanced_embeddings.shape
    })
    
    print(f"📊 Anomaly Detection Results:")
    if len(np.unique(anomaly_flags)) > 1:
        print(f"   Accuracy: {metrics['accuracy']:.4f}")
        print(f"   F1 Score: {metrics['f1_score']:.4f}")
        print(f"   ROC AUC: {metrics['roc_auc']:.4f}")
        print(f"   PR AUC: {metrics['pr_auc']:.4f}")
    print(f"   Mean Reconstruction Error: {metrics['mean_recon_error']:.4f}")
    print(f"   Threshold: {metrics['threshold']:.4f}")
    print(f"   Enhanced Embedding Shape: {enhanced_embeddings.shape}")
    
    return {
        'metrics': metrics,
        'anomaly_scores': anomaly_scores,
        'recon_errors': recon_errors,
        'binary_predictions': binary_predictions,
        'enhanced_embeddings': enhanced_embeddings,
        'threshold': best_threshold,
        'anomaly_flags': anomaly_flags
    }

def create_enhanced_embeddings_output(results: Dict[str, Any], all_embeddings: np.ndarray, 
                                    log_type: str, classes: List[str], output_dir: Path):
    """
    Create enhanced embeddings with anomaly flag column for downstream use
    
    Format: [original_embedding_features..., anomaly_flag]
    Where anomaly_flag: 0 = normal, 1 = anomaly (based on reconstruction threshold)
    """
    
    # Enhanced embeddings already include anomaly score as last column
    enhanced_embeddings = results['enhanced_embeddings']
    binary_predictions = results['binary_predictions']
    
    # Replace anomaly score with binary flag in the last column
    enhanced_embeddings_binary = enhanced_embeddings.copy()
    enhanced_embeddings_binary[:, -1] = binary_predictions.astype(np.float32)
    
    # Save enhanced embeddings
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save in the same format as original embeddings
    embeddings_output_dir = output_dir / "embeddings" / f"{log_type}_anomaly"
    embeddings_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save enhanced embeddings (with binary flag)
    enhanced_log_file = embeddings_output_dir / f"log_{log_type}_anomaly.pkl"
    with open(enhanced_log_file, 'wb') as f:
        pickle.dump(enhanced_embeddings_binary, f)
    
    # Save enhanced embeddings (with continuous anomaly scores)
    enhanced_score_file = embeddings_output_dir / f"log_{log_type}_anomaly_scores.pkl"
    with open(enhanced_score_file, 'wb') as f:
        pickle.dump(enhanced_embeddings, f)
    
    # Create compatible label file for downstream supervised learning
    label_data = {
        'classes': ['normal', 'anomaly'],
        'vectors': np.column_stack([
            (binary_predictions == 0).astype(np.float32),  # normal
            (binary_predictions == 1).astype(np.float32)   # anomaly
        ]),
        'original_classes': classes,
        'anomaly_threshold': results['threshold'],
        'detection_method': 'transformer_reconstruction'
    }
    
    enhanced_label_file = embeddings_output_dir / f"label_{log_type}_anomaly.pkl"
    with open(enhanced_label_file, 'wb') as f:
        pickle.dump(label_data, f)
    
    # Save metadata
    metadata = {
        'original_embedding_dim': all_embeddings.shape[1],
        'enhanced_embedding_dim': enhanced_embeddings.shape[1],
        'total_samples': len(enhanced_embeddings),
        'normal_samples': int((binary_predictions == 0).sum()),
        'anomaly_samples': int((binary_predictions == 1).sum()),
        'anomaly_rate': float(binary_predictions.mean()),
        'detection_threshold': float(results['threshold']),
        'log_type': log_type,
        'model_type': 'transformer_anomaly_detection',
        'timestamp': datetime.now().isoformat()
    }
    
    metadata_file = embeddings_output_dir / f"metadata_{log_type}_anomaly.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"💾 Enhanced embeddings saved:")
    print(f"   Binary flags: {enhanced_log_file}")
    print(f"   Anomaly scores: {enhanced_score_file}")
    print(f"   Labels: {enhanced_label_file}")
    print(f"   Metadata: {metadata_file}")
    print(f"   Shape: {enhanced_embeddings.shape} (added 1 column)")
    print(f"   Anomaly rate: {metadata['anomaly_rate']:.1%}")

def visualize_anomaly_results(results: Dict[str, Any], output_dir: Path, log_type: str):
    """Create comprehensive visualizations of anomaly detection results"""
    print("📈 Generating anomaly detection visualizations...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Reconstruction error distribution
    plt.figure(figsize=(12, 6))
    
    normal_mask = (results['anomaly_flags'] == 0)
    anomaly_mask = (results['anomaly_flags'] == 1)
    
    plt.subplot(1, 2, 1)
    if normal_mask.sum() > 0:
        plt.hist(results['recon_errors'][normal_mask], bins=50, alpha=0.7, label='Normal', 
                color='blue', edgecolor='black')
    if anomaly_mask.sum() > 0:
        plt.hist(results['recon_errors'][anomaly_mask], bins=50, alpha=0.7, label='Anomaly', 
                color='red', edgecolor='black')
    
    plt.axvline(results['threshold'], color='green', linestyle='--', linewidth=2, label='Threshold')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Frequency')
    plt.title(f'Reconstruction Error Distribution - {log_type}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Anomaly score distribution
    plt.subplot(1, 2, 2)
    if normal_mask.sum() > 0:
        plt.hist(results['anomaly_scores'][normal_mask], bins=50, alpha=0.7, label='Normal', 
                color='blue', edgecolor='black')
    if anomaly_mask.sum() > 0:
        plt.hist(results['anomaly_scores'][anomaly_mask], bins=50, alpha=0.7, label='Anomaly', 
                color='red', edgecolor='black')
    
    plt.xlabel('Anomaly Score')
    plt.ylabel('Frequency')
    plt.title(f'Anomaly Score Distribution - {log_type}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'anomaly_distributions_{log_type}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. ROC Curve (if we have both classes)
    if len(np.unique(results['anomaly_flags'])) > 1:
        plt.figure(figsize=(10, 5))
        
        # ROC Curve
        plt.subplot(1, 2, 1)
        fpr, tpr, _ = roc_curve(results['anomaly_flags'], results['recon_errors'])
        plt.plot(fpr, tpr, label=f'ROC (AUC = {results["metrics"]["roc_auc"]:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Precision-Recall Curve
        plt.subplot(1, 2, 2)
        precision, recall, _ = precision_recall_curve(results['anomaly_flags'], results['recon_errors'])
        plt.plot(recall, precision, label=f'PR (AUC = {results["metrics"]["pr_auc"]:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'anomaly_curves_{log_type}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. Scatter plot of reconstruction error vs anomaly score
    plt.figure(figsize=(10, 8))
    
    if normal_mask.sum() > 0:
        plt.scatter(results['recon_errors'][normal_mask], results['anomaly_scores'][normal_mask], 
                   alpha=0.6, c='blue', label='Normal', s=20)
    if anomaly_mask.sum() > 0:
        plt.scatter(results['recon_errors'][anomaly_mask], results['anomaly_scores'][anomaly_mask], 
                   alpha=0.6, c='red', label='Anomaly', s=20)
    
    plt.axvline(results['threshold'], color='green', linestyle='--', linewidth=2, label='Threshold')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Anomaly Score')
    plt.title(f'Reconstruction Error vs Anomaly Score - {log_type}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / f'anomaly_scatter_{log_type}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Visualizations saved to {output_dir}")

# =============================================================================
# Main Execution Pipeline
# =============================================================================

def save_final_results(model: AnomalyDetectionTransformer, results: Dict[str, Any], 
                      classes: List[str], scaler: StandardScaler, log_type: str, 
                      tracker: ProgressTracker):
    """Save final model and results"""
    
    # Create output directories
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = MODELS_DIR / f"anomaly_model_{log_type}.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_dim': model.input_dim,
            'latent_dim': model.latent_dim,
            'transformer_layers': model.transformer_layers,
            'attention_heads': model.attention_heads
        },
        'classes': classes,
        'metrics': results['metrics'],
        'threshold': results['threshold']
    }, model_path)
    
    # Save scaler
    scaler_path = MODELS_DIR / f"anomaly_scaler_{log_type}.pkl"
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save results
    results_path = RESULTS_DIR / f"anomaly_results_{log_type}.pkl"
    with open(results_path, 'wb') as f:
        # Remove large arrays before saving
        save_results = {
            'metrics': results['metrics'],
            'threshold': results['threshold']
        }
        pickle.dump(save_results, f)
    
    # Save human-readable results
    summary_path = RESULTS_DIR / f"anomaly_summary_{log_type}.json"
    with open(summary_path, 'w') as f:
        json.dump({
            'log_type': log_type,
            'model_type': 'Transformer Anomaly Detection',
            'classes': classes,
            'metrics': results['metrics'],
            'threshold': float(results['threshold']),
            'training_approach': 'unsupervised_normal_only',
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
    
    print(f"💾 Final results saved:")
    print(f"   Model: {model_path}")
    print(f"   Scaler: {scaler_path}")
    print(f"   Results: {results_path}")
    print(f"   Summary: {summary_path}")

def main():
    """Main execution pipeline for transformer anomaly detection"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Transformer-based Unsupervised Anomaly Detection')
    parser.add_argument('--log-type', type=str, required=True, 
                       help='Log type to process (e.g., wp-access, wp-error)')
    parser.add_argument('--sample-size', type=int, default=None,
                       help='Limit dataset size for testing')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    
    args = parser.parse_args()
    
    print("🚀 Starting Transformer Anomaly Detection Pipeline")
    print(f"📊 Processing log type: {args.log_type}")
    print(f"🎯 Strategy: Train on normal logs only, detect anomalies via reconstruction")
    
    # Initialize system
    config = detect_system_resources()
    tracker = ProgressTracker(RESULTS_DIR, args.log_type, config)
    
    try:
        # Load and preprocess data
        normal_embeddings, all_embeddings, anomaly_flags, classes, scaler = load_and_preprocess_data(
            args.log_type, config, tracker, args.sample_size
        )
        
        # Train anomaly detection model (on normal logs only)
        model = train_anomaly_model(normal_embeddings, config, tracker, args.log_type)
        
        # Evaluate model (on all logs)
        results = evaluate_anomaly_model(model, all_embeddings, anomaly_flags, classes, config, tracker)
        
        # Create enhanced embeddings with anomaly flag
        create_enhanced_embeddings_output(results, all_embeddings, args.log_type, classes, RESULTS_DIR.parent)
        
        # Generate visualizations
        visualize_anomaly_results(results, RESULTS_DIR, args.log_type)
        
        # Save final results
        save_final_results(model, results, classes, scaler, args.log_type, tracker)
        
        print("🎉 Transformer anomaly detection pipeline completed successfully!")
        
        # Print final summary
        print(f"\n📈 Final Anomaly Detection Summary:")
        if len(np.unique(anomaly_flags)) > 1:
            print(f"   Accuracy: {results['metrics']['accuracy']:.4f}")
            print(f"   F1 Score: {results['metrics']['f1_score']:.4f}")
            print(f"   ROC AUC: {results['metrics']['roc_auc']:.4f}")
        
        print(f"   Detection Threshold: {results['threshold']:.4f}")
        print(f"   Processed {len(all_embeddings):,} samples")
        print(f"   Enhanced embedding shape: {results['enhanced_embeddings'].shape}")
        print(f"   Anomaly rate: {anomaly_flags.mean():.1%}")
        
        # Output file locations
        print(f"\n📁 Output files for downstream use:")
        embeddings_dir = RESULTS_DIR.parent / "embeddings" / f"{args.log_type}_anomaly"
        print(f"   Enhanced embeddings: {embeddings_dir / f'log_{args.log_type}_anomaly.pkl'}")
        print(f"   Binary labels: {embeddings_dir / f'label_{args.log_type}_anomaly.pkl'}")
        
    except Exception as e:
        print(f"❌ Error in anomaly detection pipeline: {str(e)}")
        tracker.logger.error(f"Pipeline error: {str(e)}", exc_info=True)
        raise
    finally:
        # Cleanup
        clear_gpu_memory()
        cleanup_distributed()

if __name__ == "__main__":
    main()
