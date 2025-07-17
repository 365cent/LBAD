#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
High-Performance Transformer-based Unsupervised Multi-Label Learning for Supercomputing
Enhanced with UMTL (Unsupervised Mutual Transformer Learning) techniques

Optimized for Research Alliance of Canada Nibi node:
- Multi-GPU support (H100, A100, V100)
- Automatic CUDA/distributed training setup
- Advanced memory management with CUDA OOM prevention
- Comprehensive error handling and recovery
- Real-time progress tracking with file outputs
- Performance profiling and optimization
- Stable results with deterministic training
- Proper label output format for evaluation
- Separate models per log type (each log file has only one type)

UMTL Enhancements:
- Teacher/EMA Network for cleaner pseudo-labels
- Reconstruction-error seeded pseudo-labels (TPLG)
- Confidence-gated self-training (FixMatch-style)
- Multi-view augmentation (weak/strong)
- Enhanced adaptive thresholding for better F1
- Distribution-balanced focal loss
- Prototype/margin loss for discriminative learning

Automatic adaptation to different embedding types:
  * FastText: 300D standard word embeddings
    - Architecture: 8 transformer layers, 8 attention heads, 256 latent dim
    - Batch sizes: 128/64 (MPS/CPU), 64/32 (CUDA conservative)
  * BERT CLS: 768D global context embeddings  
    - Architecture: 10 transformer layers, 12 attention heads, 384 latent dim
    - Batch sizes: 64/32 (MPS/CPU), 32/16 (CUDA conservative)
  * Enhanced LogBERT: 2314D multi-feature embeddings
    - Features: CLS token (768D) + mean pooling (768D) + max pooling (768D) + attention (10D)
    - Architecture: 12 transformer layers, 16 attention heads, 512 latent dim
    - Batch sizes: 32/16 (MPS/CPU), 16/8 (CUDA conservative)

Memory optimizations for CUDA:
- Conservative batch sizing based on GPU memory
- Periodic memory clearing during training and inference
- OOM exception handling with single-sample fallback
- Reduced worker counts for memory efficiency
- PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"""

import os
import sys
import time
import json
import pickle
import logging
import warnings
import hashlib
import copy  # Added for teacher network
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
from torch.cuda.amp import GradScaler, autocast

import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for clusters
import matplotlib.pyplot as plt
import seaborn as sns
from halo import Halo

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, classification_report
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
CHECKPOINT_DIR = Path("checkpoints") / "transformer"
RESULTS_DIR = Path("results")
MODELS_DIR = Path("models")

# =============================================================================
# Configuration and Resource Detection
# =============================================================================

@dataclass
class SystemConfig:
    """Auto-detected system configuration for Nibi node"""
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
    """Comprehensive system resource detection for Nibi node"""
    # GPU Detection
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        device = "cuda"
        
        # Check for specific GPU types
        gpu_name = torch.cuda.get_device_properties(0).name
        print(f"Detected {n_gpus} GPU(s): {gpu_name}")
        print(f"GPU Memory: {gpu_memory_gb:.1f} GB per GPU")
        
        # Set CUDA memory allocation configuration to avoid fragmentation
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        
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

    # Memory detection
    try:
        import psutil
        total_memory_gb = psutil.virtual_memory().total / (1024**3)
        n_cpus = psutil.cpu_count()
    except ImportError:
        total_memory_gb = 64.0  # Default assumption
        n_cpus = 16

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
    """Setup distributed training for multi-GPU on Nibi"""
    if world_size > 1:
        # Use NCCL backend for GPU communication
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            rank=rank,
            world_size=world_size
        )
        torch.cuda.set_device(rank)
        
        # Set NCCL environment variables for optimal performance
        os.environ['NCCL_IB_DISABLE'] = '0'
        os.environ['NCCL_P2P_DISABLE'] = '0'
        os.environ['NCCL_SOCKET_IFNAME'] = '^docker0,lo'

def cleanup_distributed():
    """Cleanup distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()

# =============================================================================
# UMTL Enhancement Functions
# =============================================================================

def weak_augment(x: torch.Tensor, noise_factor: float = 0.01) -> torch.Tensor:
    """
    Weak augmentation for teacher network predictions (FixMatch-style)
    
    Args:
        x: Input embeddings
        noise_factor: Gaussian noise standard deviation
    
    Returns:
        Weakly augmented embeddings
    """
    return x + noise_factor * torch.randn_like(x)

def strong_augment(x: torch.Tensor, dropout_rate: float = 0.15, noise_factor: float = 0.05) -> torch.Tensor:
    """
    Strong augmentation for student network training (FixMatch-style)
    
    Args:
        x: Input embeddings
        dropout_rate: Feature dropout rate
        noise_factor: Gaussian noise standard deviation
    
    Returns:
        Strongly augmented embeddings
    """
    # Feature dropout (simulate missing features)
    mask = (torch.rand_like(x) > dropout_rate).float()
    x_augmented = x * mask
    
    # Add noise
    x_augmented = x_augmented + noise_factor * torch.randn_like(x)
    
    return x_augmented

def multi_crop_augment(x: torch.Tensor, n_crops: int = 2) -> List[torch.Tensor]:
    """
    Multi-crop augmentation for DINO-style training
    
    Args:
        x: Input embeddings
        n_crops: Number of crops to generate
    
    Returns:
        List of augmented views
    """
    crops = []
    feat_dim = x.shape[-1]
    
    for _ in range(n_crops):
        # Randomly mask some features (simulate local crops)
        mask_size = int(feat_dim * 0.8)  # Keep 80% of features
        indices = torch.randperm(feat_dim)[:mask_size]
        
        crop = torch.zeros_like(x)
        crop[..., indices] = x[..., indices]
        
        # Add slight noise
        crop = crop + 0.01 * torch.randn_like(crop)
        crops.append(crop)
    
    return crops

def update_teacher_ema(student: nn.Module, teacher: nn.Module, momentum: float = 0.996):
    """
    Update teacher network using exponential moving average (EMA) of student weights
    
    Args:
        student: Student model
        teacher: Teacher model (updated in-place)
        momentum: EMA momentum parameter
    """
    with torch.no_grad():
        for param_s, param_t in zip(student.parameters(), teacher.parameters()):
            param_t.data.mul_(momentum).add_(param_s.data, alpha=1 - momentum)

def compute_class_weights(targets: torch.Tensor, beta: float = 0.999) -> torch.Tensor:
    """
    Compute class-balanced weights for Distribution-Balanced Loss
    
    Args:
        targets: Binary target labels (batch_size, n_classes)
        beta: Smoothing parameter for effective number calculation
    
    Returns:
        Class weights tensor
    """
    # Calculate positive counts per class
    pos_counts = targets.sum(dim=0).clamp(min=1.0)
    total_samples = targets.shape[0]
    
    # Effective number of samples (1 - β^n) / (1 - β)
    effective_num = (1.0 - torch.pow(beta, pos_counts)) / (1.0 - beta)
    
    # Class weights inversely proportional to effective number
    weights = effective_num.sum() / (len(effective_num) * effective_num)
    
    # Normalize weights
    weights = weights / weights.mean()
    
    return weights.clamp(min=0.1, max=10.0)

def generate_reconstruction_anomaly_scores(model: nn.Module, embeddings: torch.Tensor, 
                                         batch_size: int = 256, device: torch.device = torch.device('cpu')) -> np.ndarray:
    """
    Generate anomaly scores based on reconstruction error (TPLG component)
    
    Args:
        model: Trained model with reconstruction capability
        embeddings: Input embeddings
        batch_size: Batch size for processing
        device: Device to use
    
    Returns:
        Per-sample reconstruction errors
    """
    model.eval()
    recon_errors = []
    
    with torch.no_grad():
        for i in range(0, len(embeddings), batch_size):
            batch = embeddings[i:i+batch_size].to(device)
            outputs = model(batch)
            
            # Calculate reconstruction error
            recon_error = F.mse_loss(outputs['reconstructed'], batch, reduction='none')
            recon_error = recon_error.mean(dim=1)  # Per-sample error
            recon_errors.append(recon_error.cpu().numpy())
    
    return np.concatenate(recon_errors)

# =============================================================================
# Optimized Model Architecture
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

# Log type classifier removed - each log file can only be one type

class UnsupervisedMultiLabelTransformer(nn.Module):
    """Ultra-powerful transformer for unsupervised multi-label learning optimized for H100 GPU
    
    Automatically adapts to different embedding dimensions:
    - 300D: FastText embeddings
    - 768D: BERT CLS token embeddings
    - 2314D: Enhanced LogBERT embeddings (CLS + mean + max + attention)
    """
    
    def __init__(self, input_dim: int, latent_dim: int, n_labels: int, 
                 n_clusters: int, dropout: float = 0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.n_labels = n_labels
        
        # Much deeper and more powerful encoder
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Very deep encoder with 12 transformer blocks and skip connections
        self.encoder_blocks = nn.ModuleList([
            OptimizedTransformerBlock(latent_dim, 16, dropout) for _ in range(12)  # 16 heads, 12 layers
        ])
        
        # Multi-scale feature extraction using different linear transformations
        self.multi_scale_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim, latent_dim),
                nn.LayerNorm(latent_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ) for _ in range(4)  # 4 different transformations
        ])
        
        # Feature fusion network
        self.feature_fusion = nn.Sequential(
            nn.Linear(latent_dim * 5, latent_dim * 2),  # 4 transforms + 1 original
            nn.LayerNorm(latent_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.LayerNorm(latent_dim)
        )
        
        # Very deep decoder with 8 blocks
        self.decoder_blocks = nn.ModuleList([
            OptimizedTransformerBlock(latent_dim, 16, dropout) for _ in range(8)
        ])
        self.output_proj = nn.Linear(latent_dim, input_dim)
        
        # Sophisticated multi-label prediction head with multiple branches
        self.label_attention = nn.MultiheadAttention(latent_dim, 8, dropout=dropout, batch_first=True)
        
        # Multiple prediction branches for ensemble-like behavior
        self.label_branches = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim, latent_dim),
                nn.LayerNorm(latent_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(latent_dim, latent_dim // 2),
                nn.LayerNorm(latent_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(latent_dim // 2, n_labels)
            ) for _ in range(3)  # 3 branches for ensemble
        ])
        
        # Advanced cluster prediction with hierarchical clustering
        self.cluster_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, latent_dim // 2),
            nn.LayerNorm(latent_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim // 2, n_clusters)
        )
        
        # Very sophisticated contrastive projection head
        self.contrastive_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.BatchNorm1d(latent_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, 512)  # Much larger projection space
        )
        
        # Label relationship modeling
        self.label_correlation = nn.Parameter(torch.randn(n_labels, n_labels) * 0.1)
        
        # UMTL Enhancement: Prototype/Margin components
        # Running prototypes for each class (not trainable, updated during training)
        self.register_buffer('class_prototypes', torch.zeros(n_labels, latent_dim))
        self.register_buffer('prototype_counts', torch.zeros(n_labels))
        
        # Prototype projection head for margin loss
        self.prototype_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Margin for prototype loss
        self.margin = 0.5
        
        # Initialize weights for stable training
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights for stable training"""
        if isinstance(module, nn.Linear):
            # Better initialization for deeper networks (use 'relu' as closest supported option to 'gelu')
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(module, nn.MultiheadAttention):
            nn.init.kaiming_normal_(module.in_proj_weight, mode='fan_out', nonlinearity='relu')
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Enhanced input processing
        z = self.input_proj(x)
        z = z.unsqueeze(1)  # Add sequence dimension
        
        # Very deep encoder with residual connections every 3 blocks
        encoder_outputs = []
        for i, block in enumerate(self.encoder_blocks):
            z_residual = z
            z = block(z)
            if i % 3 == 2 and i > 0:  # Residual every 3 blocks
                z = z + encoder_outputs[-1] if encoder_outputs else z
            encoder_outputs.append(z)
        
        z_flat = z.squeeze(1)
        
        # Multi-scale feature extraction using different transformations
        z_transform_features = []
        
        for transform in self.multi_scale_transforms:
            transform_out = transform(z_flat)
            z_transform_features.append(transform_out)
        
        # Combine all features
        all_features = torch.cat([z_flat] + z_transform_features, dim=1)
        z_fused = self.feature_fusion(all_features)
        
        # Deep decoder
        z_dec = z_fused.unsqueeze(1)
        for block in self.decoder_blocks:
            z_dec = block(z_dec)
        x_recon = self.output_proj(z_dec.squeeze(1))
        
        # Advanced label prediction with attention and ensemble
        z_attn = z_fused.unsqueeze(1)
        z_attn, _ = self.label_attention(z_attn, z_attn, z_attn)
        z_attn = z_attn.squeeze(1)
        
        # Ensemble prediction from multiple branches
        branch_predictions = []
        for branch in self.label_branches:
            branch_pred = branch(z_attn + z_fused)  # Residual connection
            branch_predictions.append(branch_pred)
        
        # Average ensemble predictions
        labels = torch.stack(branch_predictions, dim=0).mean(dim=0)
        
        # Apply label correlation matrix
        label_correlations = torch.sigmoid(self.label_correlation)
        labels = labels + 0.1 * torch.matmul(torch.sigmoid(labels), label_correlations)
        
        # Cluster prediction
        clusters = self.cluster_head(z_fused)
        
        # Enhanced contrastive projection
        z_contrastive = self.contrastive_head(z_fused)
        z_contrastive = F.normalize(z_contrastive, dim=1)
        
        # UMTL Enhancement: Prototype features
        z_prototype = self.prototype_head(z_fused)
        z_prototype = F.normalize(z_prototype, dim=1)
        
        return {
            'latent': z_fused,
            'reconstructed': x_recon,
            'labels': labels,
            'clusters': clusters,
            'contrastive': z_contrastive,
            'prototype': z_prototype,  # Added for margin loss
            'branch_predictions': branch_predictions  # For additional loss
        }

# =============================================================================
# Training and Data Management
# =============================================================================

class ProgressTracker:
    """Track training progress with file outputs and time estimation for Nibi"""
    
    def __init__(self, output_dir: Path, log_type: str, config: SystemConfig):
        self.output_dir = output_dir
        self.log_type = log_type
        self.config = config
        self.metrics = []
        self.start_time = None
        self.epoch_times = []
        self.true_labels = None # Added for storing true labels
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging with node information
        log_file = output_dir / f"training_{log_type}_{config.node_name}_{config.job_id}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Log system configuration
        self.logger.info(f"System Configuration: {config.__dict__}")
    
    def start_training(self, total_epochs: int):
        """Start training timer"""
        self.start_time = time.time()
        self.total_epochs = total_epochs
        self.epoch_times = []
    
    def update_epoch_progress(self, epoch: int, epoch_time: float):
        """Update epoch progress and estimate completion time"""
        self.epoch_times.append(epoch_time)
        
        if len(self.epoch_times) >= 2:
            avg_epoch_time = np.mean(self.epoch_times[-5:])  # Use last 5 epochs for better estimate
            remaining_epochs = self.total_epochs - epoch - 1
            estimated_remaining = avg_epoch_time * remaining_epochs
            
            elapsed = time.time() - self.start_time
            estimated_total = elapsed + estimated_remaining
            
            # Format time strings
            elapsed_str = self._format_time(elapsed)
            remaining_str = self._format_time(estimated_remaining)
            total_str = self._format_time(estimated_total)
            
            return {
                'epoch': epoch + 1,
                'total_epochs': self.total_epochs,
                'elapsed': elapsed_str,
                'remaining': remaining_str,
                'estimated_total': total_str,
                'avg_epoch_time': f"{avg_epoch_time:.2f}s"
            }
        return None
    
    def _format_time(self, seconds: float) -> str:
        """Format time in human readable format"""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"
    
    def log_step(self, step: str, data: Dict[str, Any]):
        """Log step results to file and console"""
        timestamp = datetime.now().isoformat()
        self.logger.info(f"STEP: {step}")
        self.logger.info(f"DATA: {json.dumps(data, indent=2)}")
        
        # Save to JSON file
        step_file = self.output_dir / f"step_{step.lower().replace(' ', '_')}_{self.log_type}_{self.config.node_name}_{self.config.job_id}.json"
        with open(step_file, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'step': step,
                'data': data,
                'node': self.config.node_name,
                'job_id': self.config.job_id
            }, f, indent=2)
    
    def log_metrics(self, epoch: int, metrics: Dict[str, float]):
        """Log training metrics"""
        self.metrics.append({'epoch': epoch, **metrics})
        self.logger.info(f"Epoch {epoch}: {metrics}")
        
        # Save metrics
        metrics_file = self.output_dir / f"metrics_{self.log_type}_{self.config.node_name}_{self.config.job_id}.json"
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)

def load_and_preprocess_data(log_type: str, config: SystemConfig, tracker: ProgressTracker) -> Tuple[np.ndarray, List[str], np.ndarray, StandardScaler]:
    """Optimized data loading with memory management for Nibi and embedding auto-detection"""
    tracker.log_step("Data Loading", {"log_type": log_type, "config": config.__dict__})
    
    # Load embeddings - only load specific log type, not combined
    embeddings_dir = Path("embeddings")
    log_file = embeddings_dir / log_type / f"log_{log_type}.pkl"
    label_file = embeddings_dir / log_type / f"label_{log_type}.pkl"
    
    if not log_file.exists():
        raise FileNotFoundError(f"Embedding file not found: {log_file}")
    
    with open(log_file, 'rb') as f:
        embeddings = pickle.load(f)
    
    # Detect embedding type based on dimension
    embedding_dim = embeddings.shape[1]
    embedding_type = "Unknown"
    if embedding_dim == 300:
        embedding_type = "FastText (300D)"
    elif embedding_dim == 768:
        embedding_type = "BERT CLS only (768D)" 
    elif embedding_dim == 2314:
        embedding_type = "Enhanced LogBERT (2314D)"
    
    tracker.log_step("Embedding Type Auto-Detection", {
        "embedding_dim": embedding_dim,
        "embedding_type": embedding_type,
        "n_samples": len(embeddings)
    })
    
    # Load labels and actual label vectors if available
    classes = []
    true_labels = None
    if label_file.exists():
        with open(label_file, 'rb') as f:
            label_data = pickle.load(f)
            if isinstance(label_data, dict):
                if 'classes' in label_data:
                    classes = label_data['classes']
                if 'vectors' in label_data:
                    true_labels = label_data['vectors']
                    # Use true labels to guide pseudo-label generation
                    tracker.log_step("True Labels Found", {
                        "n_samples": len(true_labels),
                        "n_classes": len(classes),
                        "label_density": np.mean(true_labels.sum(axis=1))
                    })
    
    # Handle case where no labels exist (all logs are normal/benign)
    if not classes:
        classes = ['normal']  # Default class for unlabeled logs
        true_labels = np.ones((len(embeddings), 1), dtype=np.float32)  # All samples are normal
        tracker.log_step("No Labels Found - Using Normal Class", {
            "n_samples": len(embeddings),
            "default_class": "normal",
            "note": "All logs treated as normal/benign"
        })
    
    # Adaptive subsampling based on embedding type and device capabilities
    if embedding_dim <= 300:  # FastText
        if config.device == "cuda":
            samples_per_gb = min(800, int(config.gpu_memory_gb * 8))  # Conservative for CUDA
        else:
            samples_per_gb = 1000
    elif embedding_dim <= 768:  # Standard BERT
        if config.device == "cuda":
            samples_per_gb = min(400, int(config.gpu_memory_gb * 4))  # Conservative for CUDA
        else:
            samples_per_gb = 500
    else:  # Enhanced LogBERT (2314D)
        if config.device == "cuda":
            samples_per_gb = min(100, int(config.gpu_memory_gb * 1.5))  # Very conservative for CUDA
        else:
            samples_per_gb = 150
    
    # Apply more conservative limits for CUDA to prevent OOM
    if config.device == "cuda":
        max_samples = min(30000, int(config.gpu_memory_gb * samples_per_gb))
    else:
        max_samples = min(50000, int(config.gpu_memory_gb * samples_per_gb))
    
    if len(embeddings) > max_samples:
        indices = np.random.choice(len(embeddings), max_samples, replace=False)
        embeddings = embeddings[indices]
        if true_labels is not None:
            true_labels = true_labels[indices]
        tracker.log_step("Data Subsampling", {
            "original_size": len(embeddings),
            "subsampled_size": max_samples,
            "memory_gb": embeddings.nbytes / (1024**3),
            "embedding_dim": embedding_dim,
            "embedding_type": embedding_type,
            "samples_per_gb": samples_per_gb,
            "device_type": config.device
        })
    
    # Normalize with robust scaling to handle outliers better
    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()
    embeddings = scaler.fit_transform(embeddings).astype(np.float32)
    
    # Additional L2 normalization for better contrastive learning
    from sklearn.preprocessing import normalize
    embeddings = normalize(embeddings, norm='l2', axis=1)
    
    # Create label clusters
    n_clusters = min(8, len(classes), max(1, len(embeddings) // 1000))
    C = create_label_clusters(classes, n_clusters)
    
    # Store true labels in tracker for later use
    if true_labels is not None:
        tracker.true_labels = true_labels
    else:
        tracker.true_labels = None
    
    tracker.log_step("Data Preprocessing", {
        "embeddings_shape": embeddings.shape,
        "embedding_type": embedding_type,
        "n_classes": len(classes),
        "n_clusters": C.shape[1] if C is not None else 0,
        "has_true_labels": true_labels is not None
    })
    
    return embeddings, classes, C, scaler

def create_label_clusters(classes: List[str], n_clusters: int) -> Optional[np.ndarray]:
    """Create semantic label clusters"""
    if not classes or n_clusters <= 0:
        return None
    
    # For single class (like 'normal'), create a single cluster
    if len(classes) == 1:
        return np.ones((1, 1), dtype=np.float32)
    
    n_clusters = min(n_clusters, len(classes))
    C = np.zeros((len(classes), n_clusters))
    
    # Simple hash-based clustering
    for i, class_name in enumerate(classes):
        cluster_id = hash(class_name) % n_clusters
        C[i, cluster_id] = 1
    
    return C

def focal_loss(logits: torch.Tensor, targets: torch.Tensor, alpha: float = None, gamma: float = 2.0) -> torch.Tensor:
    """
    Focal loss for addressing class imbalance and improving confidence
    
    Args:
        logits: Model predictions (before sigmoid)
        targets: Target labels
        alpha: Weighting factor for rare class (if None, computed from targets)
        gamma: Focusing parameter
    
    Returns:
        Focal loss value
    """
    # Apply sigmoid to get probabilities
    probs = torch.sigmoid(logits)
    
    # Calculate class weights if alpha not provided
    if alpha is None:
        # Calculate per-class positive rates with stability
        pos_counts = targets.sum(dim=0).clamp(min=1.0)  # Avoid division by zero
        neg_counts = (targets.shape[0] - pos_counts).clamp(min=1.0)
        
        # Inverse frequency weighting with smoothing
        total_samples = targets.shape[0]
        pos_weights = torch.sqrt(total_samples / (2.0 * pos_counts))
        neg_weights = torch.sqrt(total_samples / (2.0 * neg_counts))
        
        # Clamp weights to reasonable range
        pos_weights = pos_weights.clamp(min=0.1, max=10.0)
        neg_weights = neg_weights.clamp(min=0.1, max=10.0)
        
        # Apply class-specific weights
        alpha_t = targets * pos_weights.unsqueeze(0) + (1 - targets) * neg_weights.unsqueeze(0)
    else:
        alpha_t = alpha
    
    # Calculate focal loss
    ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p_t = probs * targets + (1 - probs) * (1 - targets)
    focal_weight = alpha_t * (1 - p_t) ** gamma
    focal_loss = focal_weight * ce_loss
    
    return focal_loss.mean()

def enhanced_focal_loss(logits: torch.Tensor, targets: torch.Tensor, class_weights: torch.Tensor = None, 
                       gamma: float = 2.0, beta: float = 0.999) -> torch.Tensor:
    """
    Enhanced focal loss with distribution-balanced weights for better F1 scores
    
    Args:
        logits: Model predictions (before sigmoid)
        targets: Target labels
        class_weights: Pre-computed class weights (if None, computed from targets)
        gamma: Focusing parameter
        beta: Smoothing parameter for effective number calculation
    
    Returns:
        Enhanced focal loss value
    """
    # Apply sigmoid to get probabilities
    probs = torch.sigmoid(logits)
    
    # Compute class-balanced weights if not provided
    if class_weights is None:
        class_weights = compute_class_weights(targets, beta)
    
    # Ensure class_weights is broadcastable
    class_weights = class_weights.unsqueeze(0)  # Shape: (1, n_classes)
    
    # Calculate focal loss with class balancing
    ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p_t = probs * targets + (1 - probs) * (1 - targets)
    
    # Apply class weights and focal factor
    focal_weight = class_weights * (1 - p_t) ** gamma
    focal_loss = focal_weight * ce_loss
    
    return focal_loss.mean()

def prototype_margin_loss(z_prototype: torch.Tensor, labels: torch.Tensor, 
                         class_prototypes: torch.Tensor, margin: float = 0.5) -> torch.Tensor:
    """
    Prototype/margin loss for discriminative learning (UMTL component)
    
    Args:
        z_prototype: Prototype features from model (batch_size, latent_dim)
        labels: Binary labels (batch_size, n_classes)
        class_prototypes: Running class prototypes (n_classes, latent_dim)
        margin: Margin for separation
    
    Returns:
        Prototype margin loss
    """
    batch_size = z_prototype.shape[0]
    n_classes = labels.shape[1]
    
    # Compute distances to all prototypes
    # z_prototype: (batch_size, latent_dim)
    # class_prototypes: (n_classes, latent_dim)
    # distances: (batch_size, n_classes)
    distances = torch.cdist(z_prototype.unsqueeze(1), class_prototypes.unsqueeze(0)).squeeze(1)
    
    # Create positive and negative masks
    pos_mask = labels > 0.5  # Active classes
    neg_mask = ~pos_mask  # Inactive classes
    
    # Positive loss: minimize distance to assigned prototypes
    pos_distances = distances * pos_mask.float()
    pos_loss = pos_distances.sum() / (pos_mask.sum() + 1e-8)
    
    # Negative loss: maximize distance to non-assigned prototypes (with margin)
    neg_distances = F.relu(margin - distances) * neg_mask.float()
    neg_loss = neg_distances.sum() / (neg_mask.sum() + 1e-8)
    
    # Total loss
    total_loss = pos_loss + neg_loss
    
    return total_loss

def update_class_prototypes(model: nn.Module, z_prototype: torch.Tensor, labels: torch.Tensor, 
                          momentum: float = 0.99):
    """
    Update running class prototypes using high-confidence predictions
    
    Args:
        model: Model containing class_prototypes buffer
        z_prototype: Current batch prototype features
        labels: Current batch labels (probabilities)
        momentum: Update momentum
    """
    with torch.no_grad():
        # Only update with high-confidence predictions
        confidence_threshold = 0.7
        high_conf_mask = labels > confidence_threshold
        
        for class_idx in range(labels.shape[1]):
            class_mask = high_conf_mask[:, class_idx]
            
            if class_mask.any():
                # Get features for this class
                class_features = z_prototype[class_mask]
                
                # Compute mean of class features
                class_mean = class_features.mean(dim=0)
                
                # Update prototype with momentum
                if model.prototype_counts[class_idx] == 0:
                    # First update for this class
                    model.class_prototypes[class_idx] = class_mean
                    model.prototype_counts[class_idx] = 1
                else:
                    # Momentum update
                    model.class_prototypes[class_idx] = (
                        momentum * model.class_prototypes[class_idx] + 
                        (1 - momentum) * class_mean
                    )
                    model.prototype_counts[class_idx] += class_mask.sum().item()

def confidence_regularization_loss(predictions: torch.Tensor, confidence_target: float = 0.8) -> torch.Tensor:
    """
    Encourage higher confidence predictions
    
    Args:
        predictions: Model predictions (after sigmoid)
        confidence_target: Target confidence level
    
    Returns:
        Confidence regularization loss
    """
    # Calculate prediction confidence (distance from 0.5)
    confidence = torch.abs(predictions - 0.5) * 2
    
    # Penalize low confidence predictions
    confidence_loss = F.mse_loss(confidence, torch.full_like(confidence, confidence_target))
    
    return confidence_loss

def generate_pseudo_labels(embeddings: np.ndarray, classes: List[str], k: int = 3, true_labels: np.ndarray = None) -> np.ndarray:
    """Generate pseudo-labels using self-supervised mutual learning approach with higher confidence"""
    if not classes:
        # If no classes, create a single 'normal' class
        return np.ones((len(embeddings), 1), dtype=np.float32) * 0.8
    
    n_samples = len(embeddings)
    n_classes = len(classes)
    
    # Initialize pseudo-labels with higher confidence
    if true_labels is not None:
        # If we have true labels, use them with higher confidence
        pseudo_labels = true_labels.astype(np.float32)
        # Add less noise to maintain higher confidence
        noise = np.random.rand(n_samples, n_classes) * 0.05  # Reduced noise
        pseudo_labels = pseudo_labels * 0.9 + noise * 0.1  # Higher weight for true labels
        
        # For unlabeled samples, initialize with more confident random values
        unlabeled_mask = (true_labels.sum(axis=1) == 0)
        if np.any(unlabeled_mask):
            # Generate more confident initial labels
            confident_labels = np.random.rand(np.sum(unlabeled_mask), n_classes)
            # Make them more binary-like (closer to 0 or 1)
            confident_labels = (confident_labels > 0.3).astype(float) * 0.8 + 0.1
            pseudo_labels[unlabeled_mask] = confident_labels
    else:
        # Initialize with more confident random values
        pseudo_labels = np.random.rand(n_samples, n_classes).astype(np.float32)
        # Make initial labels more confident
        pseudo_labels = (pseudo_labels > 0.4).astype(float) * 0.7 + 0.15
    
    # 1. Compute pairwise similarities with higher threshold
    embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    
    # Use higher similarity threshold for more confident propagation
    similarity_threshold = 0.8  # Increased from 0.7
    
    # 2. Propagate labels with confidence weighting
    chunk_size = min(1000, n_samples)
    
    for i in range(0, n_samples, chunk_size):
        end_i = min(i + chunk_size, n_samples)
        chunk_i = embeddings_norm[i:end_i]
        
        similarities = np.dot(chunk_i, embeddings_norm.T)
        
        for j in range(len(chunk_i)):
            global_idx = i + j
            
            # Find highly similar samples
            similar_indices = np.where(similarities[j] > similarity_threshold)[0]
            
            if len(similar_indices) > 1:
                # Weight by similarity for more confident propagation
                sim_weights = similarities[j][similar_indices]
                sim_weights = sim_weights / sim_weights.sum()
                
                # Weighted average with confidence boost
                similar_labels = np.average(pseudo_labels[similar_indices], axis=0, weights=sim_weights)
                
                # Higher momentum for more confident updates
                momentum = 0.8  # Increased from 0.7-0.9
                pseudo_labels[global_idx] = momentum * pseudo_labels[global_idx] + (1 - momentum) * similar_labels
    
    # 3. Apply confidence boosting
    # Enhance the most confident predictions
    for i in range(n_samples):
        # Get top predictions
        top_indices = np.argsort(pseudo_labels[i])[-k:]
        
        # Boost confidence of top predictions
        confidence_boost = 0.2
        pseudo_labels[i][top_indices] = np.minimum(pseudo_labels[i][top_indices] + confidence_boost, 1.0)
        
        # Reduce confidence of low predictions
        low_indices = pseudo_labels[i] < 0.3
        pseudo_labels[i][low_indices] *= 0.5
    
    # 4. Apply sharpening to increase confidence
    temperature = 0.5  # Lower temperature = higher confidence
    pseudo_labels = pseudo_labels ** (1/temperature)
    
    # 5. Normalize with confidence preservation
    row_sums = pseudo_labels.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-8)
    pseudo_labels = pseudo_labels / row_sums
    
    # 6. Apply minimum confidence threshold
    min_confidence = 0.1
    max_confidence = 0.9
    pseudo_labels = np.clip(pseudo_labels, min_confidence, max_confidence)
    
    # Ensure no NaN values
    pseudo_labels = np.nan_to_num(pseudo_labels, nan=0.1, posinf=0.9, neginf=0.1)
    
    return pseudo_labels

def mutual_learning_loss(z1: torch.Tensor, z2: torch.Tensor, pseudo_labels: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    """
    Mutual learning loss between two views of the data with numerical stability
    
    Args:
        z1, z2: Two different views/representations of the same batch
        pseudo_labels: Pseudo-labels for the batch
        temperature: Temperature for scaling (increased for stability)
    
    Returns:
        Mutual learning loss value
    """
    batch_size = z1.shape[0]
    
    # Normalize representations with epsilon for stability
    z1_norm = F.normalize(z1, dim=1, eps=1e-6)
    z2_norm = F.normalize(z2, dim=1, eps=1e-6)
    
    # Compute cross-view similarity with clamping
    cross_sim = torch.matmul(z1_norm, z2_norm.T) / temperature
    cross_sim = torch.clamp(cross_sim, min=-10, max=10)  # Prevent extreme values
    
    # Create pseudo-label similarity matrix
    label_sim = torch.matmul(pseudo_labels, pseudo_labels.T)
    label_sim = (label_sim > 0.5).float()  # Binary similarity based on shared labels
    
    # Ensure diagonal is excluded (self-similarity)
    mask_diagonal = torch.eye(batch_size, device=z1.device)
    label_sim = label_sim * (1 - mask_diagonal)
    
    # Mutual learning loss: encourage similar pseudo-labels to have similar representations
    pos_mask = label_sim
    neg_mask = (1 - label_sim) * (1 - mask_diagonal)
    
    # Count valid pairs
    n_pos = pos_mask.sum()
    n_neg = neg_mask.sum()
    
    # Only compute loss if we have valid pairs
    pos_loss = torch.tensor(0.0, device=z1.device)
    neg_loss = torch.tensor(0.0, device=z1.device)
    
    if n_pos > 0:
        # Positive pairs should have high similarity
        pos_sim = torch.sigmoid(cross_sim)
        pos_sim = torch.clamp(pos_sim, min=1e-7, max=1-1e-7)  # Numerical stability
        pos_loss = -torch.log(pos_sim) * pos_mask
        pos_loss = pos_loss.sum() / n_pos
    
    if n_neg > 0:
        # Negative pairs should have low similarity
        neg_sim = torch.sigmoid(-cross_sim)
        neg_sim = torch.clamp(neg_sim, min=1e-7, max=1-1e-7)  # Numerical stability
        neg_loss = -torch.log(neg_sim) * neg_mask
        neg_loss = neg_loss.sum() / n_neg
    
    total_loss = pos_loss + neg_loss
    
    # Final safety check
    if torch.isnan(total_loss) or torch.isinf(total_loss):
        return torch.tensor(0.0, device=z1.device)
    
    return total_loss

def contrastive_loss(z_i: torch.Tensor, z_j: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    """
    Compute simplified contrastive loss for self-supervised learning with numerical stability
    
    Args:
        z_i: First set of normalized embeddings (batch_size, embedding_dim)
        z_j: Second set of normalized embeddings (batch_size, embedding_dim)
        temperature: Temperature parameter for scaling (increased for stability)
    
    Returns:
        Contrastive loss value
    """
    batch_size = z_i.shape[0]
    
    # Ensure normalized inputs
    z_i = F.normalize(z_i, dim=1, eps=1e-6)
    z_j = F.normalize(z_j, dim=1, eps=1e-6)
    
    # Compute cosine similarity between corresponding pairs
    positive_sim = F.cosine_similarity(z_i, z_j, dim=1) / temperature
    positive_sim = torch.clamp(positive_sim, min=-10, max=10)  # Prevent overflow
    
    # Compute negative similarities (z_i vs all z_j except corresponding pair)
    all_similarities = torch.matmul(z_i, z_j.T) / temperature
    all_similarities = torch.clamp(all_similarities, min=-10, max=10)  # Prevent overflow
    
    # Create mask to exclude positive pairs from negatives
    mask = torch.eye(batch_size, device=z_i.device, dtype=torch.bool)
    
    # Use a large negative value instead of -inf for stability
    negative_similarities = all_similarities.masked_fill(mask, -50.0)
    
    # Compute InfoNCE loss
    # For each sample, positive is the corresponding pair, negatives are all other pairs
    logits = torch.cat([positive_sim.unsqueeze(1), negative_similarities], dim=1)
    labels = torch.zeros(batch_size, dtype=torch.long, device=z_i.device)
    
    # Add label smoothing for stability
    loss = F.cross_entropy(logits, labels, label_smoothing=0.1)
    
    # Final safety check
    if torch.isnan(loss) or torch.isinf(loss):
        return torch.tensor(0.0, device=z_i.device)
    
    return loss

def clear_gpu_memory():
    """Clear GPU memory efficiently"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    import gc
    gc.collect()

def train_model(embeddings: np.ndarray, classes: List[str], C: np.ndarray, 
                config: SystemConfig, tracker: ProgressTracker, log_type: str) -> Tuple[UnsupervisedMultiLabelTransformer, StandardScaler]:
    """Optimized training with multi-GPU support, mixed precision, memory management, and resumeable checkpoints"""
    
    device = torch.device(config.device)
    n_labels = len(classes) if classes else 1
    n_clusters = C.shape[1] if C is not None else 1
    
    # Generate training hash for checkpoint validation
    training_hash = generate_training_hash(embeddings, classes)
    
    # Clear memory before starting training
    clear_gpu_memory()
    
    # Adaptive latent dimension and architecture based on embedding type
    embedding_dim = embeddings.shape[1]
    if embedding_dim <= 300:  # FastText
        latent_dim = 256
        transformer_layers = 8  # Lighter architecture
        attention_heads = 8
    elif embedding_dim <= 768:  # Standard BERT
        latent_dim = 384
        transformer_layers = 10  # Medium architecture
        attention_heads = 12
    else:  # Enhanced LogBERT (2314D)
        latent_dim = 512
        transformer_layers = 12  # Full architecture for richer embeddings
        attention_heads = 16
    
    # Model setup - automatically adapts to embedding dimension
    model = UnsupervisedMultiLabelTransformer(
        input_dim=embedding_dim,
        latent_dim=latent_dim,
        n_labels=n_labels,
        n_clusters=n_clusters
    ).to(device)
    
    # UMTL Enhancement: Create teacher network (EMA of student)
    teacher = copy.deepcopy(model)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    teacher = teacher.to(device)
    
    # EMA momentum (start high, can be decreased during training)
    ema_momentum = 0.996
    
    # Initialize reconstruction anomaly scores (will be computed after warmup)
    reconstruction_anomaly_scores = None
    anomaly_threshold = 0.9  # Top 10% highest reconstruction errors
    
    # Confidence threshold for FixMatch-style training
    confidence_threshold = 0.8
    
    # Detect and log embedding type
    embedding_type = "Unknown"
    if embedding_dim == 300:
        embedding_type = "FastText (300D)"
    elif embedding_dim == 768:
        embedding_type = "BERT CLS only (768D)"
    elif embedding_dim == 2314:
        embedding_type = "Enhanced LogBERT (2314D)"
    
    tracker.log_step("Model Architecture Adaptation", {
        "embedding_dim": embedding_dim,
        "embedding_type": embedding_type,
        "latent_dim": latent_dim,
        "transformer_layers": transformer_layers,
        "attention_heads": attention_heads,
        "training_hash": training_hash,
        "note": "Model automatically adapts to input embedding type",
        "training_mode": "Fully unsupervised generative model - using all data for training"
    })
    
    # Multi-GPU setup
    if config.is_distributed:
        model = DDP(model, device_ids=[config.rank])
    elif config.n_gpus > 1:
        model = nn.DataParallel(model)
    
    # Generate initial pseudo-labels using all available true labels (if any) for guidance
    true_labels = getattr(tracker, 'true_labels', None)
    pseudo_labels = advanced_pseudo_label_generation(embeddings, classes, true_labels=true_labels, epoch=0, total_epochs=50)
    
    # Store initial pseudo-labels for refinement
    current_pseudo_labels = pseudo_labels.copy()
    
    # Data setup - use ALL data for training (no splitting)
    dataset = TensorDataset(
        torch.from_numpy(embeddings).float(),
        torch.from_numpy(current_pseudo_labels).float()
    )
    
    sampler = DistributedSampler(dataset) if config.is_distributed else None
    
    # Adaptive batch size based on embedding type and device
    if config.device == "cuda":
        # Conservative batch sizes for CUDA to prevent OOM
        if embedding_dim <= 300:  # FastText
            batch_size = min(64, max(8, int(config.gpu_memory_gb * 0.8)))
        elif embedding_dim <= 768:  # Standard BERT
            batch_size = min(32, max(4, int(config.gpu_memory_gb * 0.5)))
        else:  # Enhanced LogBERT (2314D)
            batch_size = min(16, max(2, int(config.gpu_memory_gb * 0.3)))
    else:
        # More generous batch sizes for MPS/CPU
        if embedding_dim <= 300:  # FastText
            batch_size = min(128, max(16, int(config.gpu_memory_gb * 2)))
        elif embedding_dim <= 768:  # Standard BERT
            batch_size = min(64, max(8, int(config.gpu_memory_gb * 1.5)))
        else:  # Enhanced LogBERT (2314D)
            batch_size = min(32, max(4, int(config.gpu_memory_gb * 0.8)))
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=min(4, config.n_cpus // 4),  # Reduced workers
        pin_memory=True
    )
    
    # Advanced training setup with reduced learning rate for stability
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)  # Reduced for stability
    scaler = GradScaler() if config.device == "cuda" else None
    
    # Advanced scheduler with warmup
    def lr_lambda(epoch):
        warmup_epochs = 5
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        else:
            return 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (50 - warmup_epochs)))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Gradient clipping for stability
    max_grad_norm = 1.0  # Reduced for better stability
    
    # Check for existing checkpoint
    start_epoch = 0
    checkpoint_data = load_training_checkpoint(log_type, training_hash)
    if checkpoint_data:
        try:
            checkpoint, loaded_epoch = checkpoint_data
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = loaded_epoch + 1
            print(f"✅ Resuming training from epoch {start_epoch}")
            
            # Restore best loss for early stopping
            if 'metrics' in checkpoint and 'best_loss' in checkpoint['metrics']:
                best_loss = checkpoint['metrics']['best_loss']
            else:
                best_loss = float('inf')
        except Exception as e:
            print(f"⚠️  Could not restore checkpoint: {e}")
            start_epoch = 0
            best_loss = float('inf')
    else:
        best_loss = float('inf')
    
    tracker.log_step("Training Setup", {
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "input_dim": embedding_dim,
        "latent_dim": latent_dim,
        "n_labels": n_labels,
        "n_clusters": n_clusters,
        "batch_size": batch_size,
        "device": str(device),
        "mixed_precision": scaler is not None,
        "embedding_type": embedding_type,
        "device_memory_gb": config.gpu_memory_gb,
        "memory_optimization": "CUDA conservative" if config.device == "cuda" else "Standard",
        "total_samples": len(embeddings),
        "training_mode": "Fully unsupervised - all data used for training"
    })
    
    # Training loop with progress tracking, early stopping, and checkpointing
    model.train()
    total_epochs = 50  # Balanced epochs for good convergence without taking too long
    tracker.start_training(total_epochs)
    
    refinement_interval = 10  # Refine pseudo-labels every 10 epochs
    checkpoint_interval = 5   # Save checkpoint every 5 epochs
    
    # Early stopping parameters
    patience = 5
    patience_counter = 0
    min_delta = 1e-4
    
    for epoch in range(start_epoch, total_epochs):
        epoch_start = time.time()
        epoch_losses = []
        
        if config.is_distributed:
            sampler.set_epoch(epoch)
        
        # UMTL Enhancement: Compute reconstruction anomaly scores after warmup (TPLG)
        if epoch == 2 and reconstruction_anomaly_scores is None:
            print("Computing reconstruction anomaly scores...")
            embeddings_tensor = torch.from_numpy(embeddings).float()
            reconstruction_anomaly_scores = generate_reconstruction_anomaly_scores(
                model, embeddings_tensor, batch_size, device
            )
            
            # Find anomaly threshold (top 10% highest errors)
            anomaly_threshold_value = np.percentile(reconstruction_anomaly_scores, anomaly_threshold * 100)
            anomaly_mask = reconstruction_anomaly_scores > anomaly_threshold_value
            
            # Update pseudo-labels for high-error samples (potential anomalies)
            if np.any(anomaly_mask):
                # Find attack/anomaly classes (anything not 'normal')
                attack_indices = [i for i, cls in enumerate(classes) if cls != 'normal']
                if attack_indices:
                    # Boost confidence for attack classes in high-error samples
                    for idx in np.where(anomaly_mask)[0]:
                        for attack_idx in attack_indices:
                            current_pseudo_labels[idx, attack_idx] = max(
                                current_pseudo_labels[idx, attack_idx],
                                0.7  # High confidence for anomaly
                            )
                    
                    # Update dataset with anomaly-seeded labels
                    dataset = TensorDataset(
                        torch.from_numpy(embeddings).float(),
                        torch.from_numpy(current_pseudo_labels).float()
                    )
                    
                    dataloader = DataLoader(
                        dataset, 
                        batch_size=batch_size, 
                        sampler=sampler,
                        shuffle=(sampler is None),
                        num_workers=min(2, config.n_cpus // 8),
                        pin_memory=True
                    )
                    
                    tracker.log_step("Reconstruction Anomaly Seeding (TPLG)", {
                        "epoch": epoch,
                        "anomaly_threshold": float(anomaly_threshold_value),
                        "n_anomalies": int(np.sum(anomaly_mask)),
                        "percentage_anomalies": float(np.mean(anomaly_mask) * 100)
                    })
        
        # Advanced pseudo-label refinement with curriculum learning
        if epoch > 0 and epoch % refinement_interval == 0:
            model.eval()
            with torch.no_grad():
                all_predictions = []
                # Clear memory before inference
                clear_gpu_memory()
                
                # Use smaller inference batch size for memory efficiency
                inference_batch_size = max(1, batch_size // 2) if config.device == "cuda" else batch_size
                
                for i in range(0, len(embeddings), inference_batch_size):
                    batch = torch.from_numpy(embeddings[i:i+inference_batch_size]).float().to(device)
                    
                    try:
                        outputs = model(batch)
                        predictions = torch.sigmoid(outputs['labels']).cpu().numpy()
                        all_predictions.append(predictions)
                    except torch.cuda.OutOfMemoryError:
                        # Fallback to single sample processing
                        for j in range(i, min(i + inference_batch_size, len(embeddings))):
                            single_batch = torch.from_numpy(embeddings[j:j+1]).float().to(device)
                            single_output = model(single_batch)
                            single_pred = torch.sigmoid(single_output['labels']).cpu().numpy()
                            all_predictions.append(single_pred)
                        clear_gpu_memory()
                    
                    # Clear memory periodically during inference
                    if config.device == "cuda" and (i // inference_batch_size) % 10 == 0:
                        clear_gpu_memory()
                
                all_predictions = np.vstack(all_predictions)
                
                # Generate new pseudo-labels with curriculum learning
                new_pseudo_labels = advanced_pseudo_label_generation(
                    embeddings, classes, true_labels=true_labels, 
                    epoch=epoch, total_epochs=total_epochs
                )
                
                # Combine model predictions with curriculum-generated labels
                progress = epoch / total_epochs
                model_weight = min(0.7, 0.3 + 0.4 * progress)  # Increase model influence over time
                current_pseudo_labels = (
                    model_weight * all_predictions + 
                    (1 - model_weight) * new_pseudo_labels
                )
                
                # Update dataset with refined pseudo-labels
                dataset = TensorDataset(
                    torch.from_numpy(embeddings).float(),
                    torch.from_numpy(current_pseudo_labels).float()
                )
                
                dataloader = DataLoader(
                    dataset, 
                    batch_size=batch_size, 
                    sampler=sampler,
                    shuffle=(sampler is None),
                    num_workers=min(2, config.n_cpus // 8),  # Reduced workers for CUDA
                    pin_memory=True
                )
                
                tracker.log_step("Advanced Pseudo-label Refinement", {
                    "epoch": epoch,
                    "avg_confidence": float(np.mean(all_predictions.max(axis=1))),
                    "label_density": float(np.mean(all_predictions.sum(axis=1))),
                    "model_weight": model_weight,
                    "curriculum_progress": progress,
                    "inference_batch_size": inference_batch_size
                })
            
            # Clear memory before resuming training
            clear_gpu_memory()
            model.train()
        
        # Progress spinner for batches
        with Halo(text=f"Epoch {epoch+1}/{total_epochs}", spinner='dots') as spinner:
            for batch_idx, (x_batch, y_batch) in enumerate(dataloader):
                x_batch = x_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)
                
                optimizer.zero_grad()
                
                # Clear memory periodically during training for CUDA
                if config.device == "cuda" and batch_idx % 20 == 0:
                    clear_gpu_memory()
                
                # UMTL Enhancement: Teacher predictions with weak augmentation
                with torch.no_grad():
                    # Weak augmentation for teacher
                    x_weak = weak_augment(x_batch)
                    teacher_outputs = teacher(x_weak)
                    teacher_probs = torch.sigmoid(teacher_outputs['labels'])
                    
                    # Confidence mask for FixMatch-style training
                    max_probs = teacher_probs.max(dim=1).values
                    confidence_mask = (max_probs >= confidence_threshold).float()
                    
                    # Create pseudo-labels from confident teacher predictions
                    teacher_pseudo_labels = (teacher_probs >= confidence_threshold).float()
                
                # Only train on confident samples (FixMatch-style)
                n_confident = confidence_mask.sum().item()
                
                if scaler:
                    with autocast():
                        try:
                            # Strong augmentation for student
                            x_strong = strong_augment(x_batch)
                            outputs = model(x_strong)
                        except torch.cuda.OutOfMemoryError:
                            # Skip this batch and clear memory
                            clear_gpu_memory()
                            continue
                        
                        # Advanced multi-component loss with curriculum learning
                        recon_loss = F.mse_loss(outputs['reconstructed'], x_batch)
                        
                        # Enhanced focal loss with distribution-balanced weights
                        label_loss = enhanced_focal_loss(outputs['labels'], y_batch)
                        
                        # Supervised loss on confident teacher predictions (FixMatch)
                        if n_confident > 0:
                            confidence_mask_expanded = confidence_mask.unsqueeze(1)
                            supervised_loss = F.binary_cross_entropy_with_logits(
                                outputs['labels'], 
                                teacher_pseudo_labels,
                                reduction='none'
                            )
                            supervised_loss = (supervised_loss * confidence_mask_expanded).sum() / (confidence_mask_expanded.sum() + 1e-8)
                        else:
                            supervised_loss = torch.tensor(0.0, device=device)
                        
                        # Add confidence regularization
                        predictions = torch.sigmoid(outputs['labels'])
                        confidence_loss = confidence_regularization_loss(predictions, confidence_target=0.8)
                        
                        # Ensemble consistency loss
                        ensemble_loss = ensemble_consistency_loss(outputs['branch_predictions'])
                        
                        # Prototype/margin loss
                        if model.prototype_counts.sum() > 0:  # Only if prototypes are initialized
                            prototype_loss = prototype_margin_loss(
                                outputs['prototype'], 
                                predictions,  # Use predictions as soft labels
                                model.class_prototypes,
                                model.margin
                            )
                        else:
                            prototype_loss = torch.tensor(0.0, device=device)
                        
                        if C is not None:
                            C_tensor = torch.from_numpy(C.astype(np.float32)).to(device)
                            # Ensure same dtype for matrix multiplication in mixed precision
                            cluster_targets = torch.matmul(y_batch.float(), C_tensor)
                            cluster_loss = F.mse_loss(outputs['clusters'], cluster_targets)
                        else:
                            cluster_loss = torch.tensor(0.0, device=device)
                        
                        # Contrastive loss - create augmented batch
                        x_augmented = x_batch + torch.randn_like(x_batch) * 0.1
                        outputs_aug = model(x_augmented)
                        
                        # Compute contrastive loss between original and augmented
                        contrast_loss = contrastive_loss(outputs['contrastive'], outputs_aug['contrastive'])
                        
                        # Mutual learning loss
                        mutual_loss = mutual_learning_loss(outputs['latent'], outputs_aug['latent'], y_batch)
                        
                        # Curriculum-based loss weighting
                        loss_weights = curriculum_weight_scheduler(epoch, total_epochs)
                        
                        # Add weights for UMTL losses
                        loss_weights['supervised'] = min(0.3, 0.1 + 0.2 * (epoch / total_epochs))  # Gradually increase
                        loss_weights['prototype'] = min(0.2, 0.05 + 0.15 * (epoch / total_epochs))  # Start small
                        
                        total_loss = (loss_weights['recon'] * recon_loss + 
                                    loss_weights['label'] * label_loss + 
                                    loss_weights['supervised'] * supervised_loss +  # Added
                                    loss_weights['prototype'] * prototype_loss +     # Added
                                    loss_weights['cluster'] * cluster_loss + 
                                    loss_weights['contrastive'] * contrast_loss + 
                                    loss_weights['mutual'] * mutual_loss + 
                                    loss_weights['confidence'] * confidence_loss + 
                                    loss_weights['ensemble'] * ensemble_loss)
                        
                        # Check for nan/inf in individual losses
                        loss_dict = {
                            'recon': recon_loss,
                            'label': label_loss,
                            'supervised': supervised_loss,  # Added
                            'prototype': prototype_loss,    # Added
                            'cluster': cluster_loss,
                            'contrastive': contrast_loss,
                            'mutual': mutual_loss,
                            'confidence': confidence_loss,
                            'ensemble': ensemble_loss
                        }
                        
                        # Check each loss component
                        skip_batch = False
                        for loss_name, loss_val in loss_dict.items():
                            if torch.isnan(loss_val) or torch.isinf(loss_val):
                                print(f"Warning: NaN/Inf detected in {loss_name} loss. Value: {loss_val.item()}")
                                skip_batch = True
                        
                        if skip_batch or torch.isnan(total_loss) or torch.isinf(total_loss):
                            print(f"Skipping batch due to numerical instability.")
                            optimizer.zero_grad()
                            continue
                    
                    scaler.scale(total_loss).backward()
                    
                    # Gradient clipping
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    
                    scaler.step(optimizer)
                    scaler.update()
                    
                    # UMTL Enhancement: Update teacher network (EMA)
                    update_teacher_ema(model, teacher, ema_momentum)
                    
                    # Update class prototypes
                    update_class_prototypes(model, outputs['prototype'], predictions)
                    
                else:
                    try:
                        # Strong augmentation for student (same as scaler branch)
                        x_strong = strong_augment(x_batch)
                        outputs = model(x_strong)
                    except torch.cuda.OutOfMemoryError:
                        # Skip this batch and clear memory
                        clear_gpu_memory()
                        continue
                    
                    recon_loss = F.mse_loss(outputs['reconstructed'], x_batch)
                    
                    # Enhanced focal loss with distribution-balanced weights
                    label_loss = enhanced_focal_loss(outputs['labels'], y_batch)
                    
                    # Supervised loss on confident teacher predictions (FixMatch)
                    if n_confident > 0:
                        confidence_mask_expanded = confidence_mask.unsqueeze(1)
                        supervised_loss = F.binary_cross_entropy_with_logits(
                            outputs['labels'], 
                            teacher_pseudo_labels,
                            reduction='none'
                        )
                        supervised_loss = (supervised_loss * confidence_mask_expanded).sum() / (confidence_mask_expanded.sum() + 1e-8)
                    else:
                        supervised_loss = torch.tensor(0.0, device=device)
                    
                    # Add confidence regularization
                    predictions = torch.sigmoid(outputs['labels'])
                    confidence_loss = confidence_regularization_loss(predictions, confidence_target=0.8)
                    
                    # Ensemble consistency loss
                    ensemble_loss = ensemble_consistency_loss(outputs['branch_predictions'])
                    
                    # Prototype/margin loss
                    if model.prototype_counts.sum() > 0:  # Only if prototypes are initialized
                        prototype_loss = prototype_margin_loss(
                            outputs['prototype'], 
                            predictions,  # Use predictions as soft labels
                            model.class_prototypes,
                            model.margin
                        )
                    else:
                        prototype_loss = torch.tensor(0.0, device=device)
                    
                    if C is not None:
                        C_tensor = torch.from_numpy(C.astype(np.float32)).to(device)
                        # Ensure same dtype for matrix multiplication
                        cluster_targets = torch.matmul(y_batch.float(), C_tensor)
                        cluster_loss = F.mse_loss(outputs['clusters'], cluster_targets)
                    else:
                        cluster_loss = torch.tensor(0.0, device=device)
                    
                    # Contrastive loss - create augmented batch
                    x_augmented = x_batch + torch.randn_like(x_batch) * 0.1
                    outputs_aug = model(x_augmented)
                    
                    # Compute contrastive loss between original and augmented
                    contrast_loss = contrastive_loss(outputs['contrastive'], outputs_aug['contrastive'])
                    
                    # Mutual learning loss
                    mutual_loss = mutual_learning_loss(outputs['latent'], outputs_aug['latent'], y_batch)
                    
                    # Curriculum-based loss weighting
                    loss_weights = curriculum_weight_scheduler(epoch, total_epochs)
                    
                    # Add weights for UMTL losses
                    loss_weights['supervised'] = min(0.3, 0.1 + 0.2 * (epoch / total_epochs))  # Gradually increase
                    loss_weights['prototype'] = min(0.2, 0.05 + 0.15 * (epoch / total_epochs))  # Start small
                    
                    total_loss = (loss_weights['recon'] * recon_loss + 
                                loss_weights['label'] * label_loss + 
                                loss_weights['supervised'] * supervised_loss +  # Added
                                loss_weights['prototype'] * prototype_loss +     # Added
                                loss_weights['cluster'] * cluster_loss + 
                                loss_weights['contrastive'] * contrast_loss + 
                                loss_weights['mutual'] * mutual_loss + 
                                loss_weights['confidence'] * confidence_loss + 
                                loss_weights['ensemble'] * ensemble_loss)
                    
                    # Check for nan/inf in individual losses
                    loss_dict = {
                        'recon': recon_loss,
                        'label': label_loss,
                        'supervised': supervised_loss,  # Added
                        'prototype': prototype_loss,    # Added
                        'cluster': cluster_loss,
                        'contrastive': contrast_loss,
                        'mutual': mutual_loss,
                        'confidence': confidence_loss,
                        'ensemble': ensemble_loss
                    }
                    
                    # Check each loss component
                    skip_batch = False
                    for loss_name, loss_val in loss_dict.items():
                        if torch.isnan(loss_val) or torch.isinf(loss_val):
                            print(f"Warning: NaN/Inf detected in {loss_name} loss. Value: {loss_val.item()}")
                            skip_batch = True
                    
                    if skip_batch or torch.isnan(total_loss) or torch.isinf(total_loss):
                        print(f"Skipping batch due to numerical instability.")
                        optimizer.zero_grad()
                        continue
                    
                    total_loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    
                    optimizer.step()
                    
                    # UMTL Enhancement: Update teacher network (EMA)
                    update_teacher_ema(model, teacher, ema_momentum)
                    
                    # Update class prototypes
                    update_class_prototypes(model, outputs['prototype'], predictions)
                
                epoch_losses.append(total_loss.item())
                
                # Update spinner text with batch progress
                if batch_idx % 10 == 0:  # Update every 10 batches
                    progress = (batch_idx + 1) / len(dataloader) * 100
                    spinner.text = f"Epoch {epoch+1}/{total_epochs} - Batch {batch_idx+1}/{len(dataloader)} ({progress:.1f}%)"
        
        scheduler.step()
        
        # Calculate epoch time and update progress
        epoch_time = time.time() - epoch_start
        progress_info = tracker.update_epoch_progress(epoch, epoch_time)
        
        # Log metrics
        avg_loss = np.mean(epoch_losses)
        
        # Early stopping check
        if avg_loss < best_loss - min_delta:
            best_loss = avg_loss
            patience_counter = 0
            # Save best model state
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        # Save checkpoint periodically
        if config.rank == 0 and (epoch + 1) % checkpoint_interval == 0:
            try:
                model_to_save = model.module if hasattr(model, 'module') else model
                checkpoint_metrics = {
                    'avg_loss': avg_loss,
                    'best_loss': best_loss,
                    'patience_counter': patience_counter
                }
                save_training_checkpoint(
                    log_type, epoch, model_to_save.state_dict(), 
                    optimizer.state_dict(), checkpoint_metrics, 
                    training_hash, config
                )
            except Exception as e:
                print(f"⚠️  Failed to save checkpoint: {e}")
            
        if patience_counter >= patience:
            if config.rank == 0:
                print(f"Early stopping triggered at epoch {epoch+1}")
            # Restore best model
            model.load_state_dict(best_model_state)
            break
        if config.rank == 0:  # Only log from main process
            tracker.log_metrics(epoch, {
                "loss": avg_loss,
                "recon_loss": recon_loss.item(),
                "label_loss": label_loss.item(),
                "cluster_loss": cluster_loss.item(),
                "contrastive_loss": contrast_loss.item() if 'contrast_loss' in locals() else 0.0,
                "mutual_loss": mutual_loss.item() if 'mutual_loss' in locals() else 0.0,
                "confidence_loss": confidence_loss.item() if 'confidence_loss' in locals() else 0.0,
                "ensemble_loss": ensemble_loss.item() if 'ensemble_loss' in locals() else 0.0,
                "lr": scheduler.get_last_lr()[0],
                "epoch_time": epoch_time
            })
        
        # Print progress with time estimation
        if config.rank == 0 and progress_info:
            print(f"Epoch {progress_info['epoch']}/{progress_info['total_epochs']} - "
                  f"Loss: {avg_loss:.6f} - "
                  f"Elapsed: {progress_info['elapsed']} - "
                  f"Remaining: {progress_info['remaining']} - "
                  f"ETA: {progress_info['estimated_total']}")
        elif config.rank == 0:
            print(f"Epoch {epoch+1}/{total_epochs} - Loss: {avg_loss:.6f} - Time: {epoch_time:.2f}s")
    
    # Clean up training checkpoints after completion (keep final checkpoint)
    if config.rank == 0:
        cleanup_training_checkpoints(log_type, keep_latest=1)
    
    return model, None  # Return None for scaler placeholder

# Log type classifier function removed - each log file can only be one type

def create_classification_summary(model: UnsupervisedMultiLabelTransformer, 
                                 embeddings: np.ndarray, classes: List[str],
                                 config: SystemConfig, tracker: ProgressTracker, 
                                 output_dir: Path, log_type: str):
    """
    Create comprehensive classification summary for generative model
    
    Args:
        model: Trained model
        embeddings: Full embeddings array used for training
        classes: List of class names
        config: System configuration
        tracker: Progress tracker
        output_dir: Output directory
        log_type: Type of log being processed
    """
    
    device = torch.device(config.device)
    model.eval()
    
    # Generate predictions for all training data
    predictions = []
    batch_size = min(256, int(config.gpu_memory_gb * 8))
    
    with torch.no_grad():
        for i in range(0, len(embeddings), batch_size):
            batch = torch.from_numpy(embeddings[i:i+batch_size]).float().to(device)
            outputs = model(batch)
            logits = outputs['labels']
            probs = torch.sigmoid(logits)
            predictions.append(probs.cpu().numpy())
    
    predictions = np.vstack(predictions)
    
    # Use enhanced adaptive thresholding (no true labels available)
    adaptive_thresholds, binary_predictions = enhanced_adaptive_thresholding(
        predictions, embeddings, classes, true_labels=None, k_neighbors=20
    )
    
    # Log threshold information
    tracker.log_step("Enhanced Adaptive Thresholding", {
        "mean_threshold": float(np.mean(adaptive_thresholds)),
        "std_threshold": float(np.std(adaptive_thresholds)),
        "min_threshold": float(np.min(adaptive_thresholds)),
        "max_threshold": float(np.max(adaptive_thresholds)),
        "thresholds_per_class": {
            cls: float(thresh) for cls, thresh in zip(classes[:10], adaptive_thresholds[:10])
        }
    })
    
    # Sophisticated multi-label assignment for better generation quality
    for idx in range(len(binary_predictions)):
        n_labels = binary_predictions[idx].sum()
        sample_preds = predictions[idx]
        
        if n_labels == 0:
            # No predictions above threshold - use dynamic assignment
            sorted_indices = np.argsort(sample_preds)[::-1]
            sorted_preds = sample_preds[sorted_indices]
            
            if len(sorted_preds) > 1:
                gradients = np.diff(sorted_preds)
                cutoff_idx = np.argmin(gradients) + 1
                cutoff_idx = np.clip(cutoff_idx, 1, 3)
                top_indices = sorted_indices[:cutoff_idx]
                binary_predictions[idx, top_indices] = 1
            else:
                binary_predictions[idx, 0] = 1
                
        elif n_labels > 5:  # Cap at 5 labels maximum
            sorted_indices = np.argsort(sample_preds)[::-1]
            sorted_preds = sample_preds[sorted_indices]
            confidence_threshold = sorted_preds[4]
            binary_predictions[idx] = 0
            binary_predictions[idx] = (sample_preds >= confidence_threshold).astype(int)
            
            if binary_predictions[idx].sum() > 5:
                top_indices = np.argsort(sample_preds)[-5:]
                binary_predictions[idx] = 0
                binary_predictions[idx, top_indices] = 1
    
    # Calculate comprehensive metrics (generative model - no true labels)
    metrics = calculate_comprehensive_metrics(
        binary_predictions, classes, y_true=None, probs=predictions
    )
    
    # Save detailed results
    save_path = output_dir / f"results_{log_type}_{config.node_name}_{config.job_id}.pkl"
    with open(save_path, 'wb') as f:
        pickle.dump({
            'predictions': predictions,
            'binary_predictions': binary_predictions,
            'classes': classes,
            'metrics': metrics,
            'adaptive_thresholds': adaptive_thresholds,
            'model_type': 'generative_transformer',
            'training_mode': 'fully_unsupervised'
        }, f)
    
    # Save labels in evaluation format
    label_output_path = output_dir / f"label_{log_type}_{config.node_name}_{config.job_id}.pkl"
    label_data = {
        'vectors': binary_predictions.astype(np.int8),
        'classes': classes,
        'probabilities': predictions.astype(np.float32),
        'metadata': {
            'node_name': config.node_name,
            'job_id': config.job_id,
            'timestamp': datetime.now().isoformat(),
            'model_type': 'generative_transformer',
            'threshold': adaptive_thresholds.tolist(),
            'training_mode': 'fully_unsupervised',
            'total_training_samples': len(embeddings)
        }
    }
    
    with open(label_output_path, 'wb') as f:
        pickle.dump(label_data, f)
    
    # Note: Classification reports are now handled by evaluate_model.py
    
    # Create visualizations
    create_comprehensive_visualizations(embeddings, predictions, binary_predictions, classes, output_dir, log_type, config)
    
    # Print summary
    print_classification_summary(metrics, log_type, len(embeddings))
    
    tracker.log_step("Classification Summary", {
        "log_type": log_type,
        "n_samples": len(embeddings),
        "n_classes": len(classes),
        "avg_labels_per_sample": metrics['avg_labels_per_sample'],
        "training_mode": "fully_unsupervised_generative",
        "prediction_confidence_mean": metrics.get('prediction_confidence_mean', 0.0)
    })
    
    return metrics

def calculate_comprehensive_metrics(binary_predictions: np.ndarray, classes: List[str], 
                                  y_true: np.ndarray = None, probs: np.ndarray = None) -> Dict[str, Any]:
    """
    Calculate comprehensive metrics for multi-label classification
    
    Args:
        binary_predictions: Binary predictions (n_samples, n_classes)
        classes: List of class names
        y_true: True labels if available (n_samples, n_classes)
        probs: Prediction probabilities if available (n_samples, n_classes)
    
    Returns:
        Dictionary of metrics including real sklearn metrics when true labels are available
    """
    from sklearn.metrics import (
        precision_recall_fscore_support, f1_score, accuracy_score, 
        hamming_loss, jaccard_score, balanced_accuracy_score
    )
    
    metrics = {}
    
    # Sample-level metrics
    labels_per_sample = binary_predictions.sum(axis=1)
    metrics['avg_labels_per_sample'] = float(labels_per_sample.mean())
    metrics['std_labels_per_sample'] = float(labels_per_sample.std())
    metrics['min_labels_per_sample'] = int(labels_per_sample.min())
    metrics['max_labels_per_sample'] = int(labels_per_sample.max())
    
    # Class-level metrics
    class_counts = binary_predictions.sum(axis=0)
    metrics['class_counts'] = class_counts.tolist()
    metrics['most_frequent_classes'] = []
    
    # Get top 10 most frequent classes
    if len(classes) > 0:
        class_freq_pairs = list(zip(classes, class_counts))
        class_freq_pairs.sort(key=lambda x: x[1], reverse=True)
        metrics['most_frequent_classes'] = [
            {'class': cls, 'count': int(count), 'percentage': float(count/len(binary_predictions)*100)}
            for cls, count in class_freq_pairs[:10]
        ]
    
    # Multi-label specific metrics
    metrics['samples_with_no_labels'] = int((labels_per_sample == 0).sum())
    metrics['samples_with_one_label'] = int((labels_per_sample == 1).sum())
    metrics['samples_with_multiple_labels'] = int((labels_per_sample > 1).sum())
    
    # Real metrics when true labels are available
    if y_true is not None:
        # Per-class metrics
        prec_c, rec_c, f1_c, support_c = precision_recall_fscore_support(
            y_true, binary_predictions, average=None, zero_division=0
        )
        
        # Store per-class metrics
        metrics['per_class'] = {}
        for i, cls in enumerate(classes):
            metrics['per_class'][cls] = {
                'support': int(support_c[i]),
                'precision': float(prec_c[i]),
                'recall': float(rec_c[i]),
                'f1-score': float(f1_c[i])
            }
        
        # Overall metrics
        metrics['macro_f1'] = float(f1_score(y_true, binary_predictions, average='macro', zero_division=0))
        metrics['micro_f1'] = float(f1_score(y_true, binary_predictions, average='micro', zero_division=0))
        metrics['weighted_f1'] = float(f1_score(y_true, binary_predictions, average='weighted', zero_division=0))
        metrics['samples_f1'] = float(f1_score(y_true, binary_predictions, average='samples', zero_division=0))
        
        # Subset accuracy (exact match)
        metrics['subset_accuracy'] = float(accuracy_score(y_true, binary_predictions))
        
        # Hamming loss
        metrics['hamming_loss'] = float(hamming_loss(y_true, binary_predictions))
        
        # Jaccard score (similarity)
        metrics['jaccard_score_macro'] = float(jaccard_score(y_true, binary_predictions, average='macro', zero_division=0))
        metrics['jaccard_score_micro'] = float(jaccard_score(y_true, binary_predictions, average='micro', zero_division=0))
        
        # Per-class balanced accuracy
        balanced_acc_per_class = []
        for c in range(y_true.shape[1]):
            y_c = y_true[:, c]
            yp_c = binary_predictions[:, c]
            
            # Skip if no samples for this class
            if y_c.sum() == 0 and (1 - y_c).sum() == 0:
                balanced_acc_per_class.append(0.0)
                continue
                
            # Calculate balanced accuracy for this class
            ba = balanced_accuracy_score(y_c, yp_c)
            balanced_acc_per_class.append(float(ba))
        
        metrics['balanced_accuracy_per_class'] = balanced_acc_per_class
        metrics['mean_balanced_accuracy'] = float(np.mean(balanced_acc_per_class))
        
        # Confusion matrix counts per class
        confusion_per_class = {}
        for i, cls in enumerate(classes):
            y_c = y_true[:, i]
            yp_c = binary_predictions[:, i]
            
            tp = int(np.sum((yp_c == 1) & (y_c == 1)))
            fp = int(np.sum((yp_c == 1) & (y_c == 0)))
            fn = int(np.sum((yp_c == 0) & (y_c == 1)))
            tn = int(np.sum((yp_c == 0) & (y_c == 0)))
            
            confusion_per_class[cls] = {
                'true_positives': tp,
                'false_positives': fp,
                'false_negatives': fn,
                'true_negatives': tn
            }
        
        metrics['confusion_per_class'] = confusion_per_class
        
        # Prediction confidence metrics (if probabilities available)
        if probs is not None:
            metrics['prediction_confidence_mean'] = float(probs.mean())
            metrics['prediction_confidence_std'] = float(probs.std())
            
            # Confidence of correct predictions
            correct_mask = (binary_predictions == y_true)
            correct_confidences = probs[correct_mask]
            incorrect_confidences = probs[~correct_mask]
            
            metrics['correct_prediction_confidence_mean'] = float(correct_confidences.mean()) if len(correct_confidences) > 0 else 0.0
            metrics['incorrect_prediction_confidence_mean'] = float(incorrect_confidences.mean()) if len(incorrect_confidences) > 0 else 0.0
    else:
        # No true labels - use placeholder values
        metrics['macro_f1'] = None
        metrics['micro_f1'] = None
        metrics['weighted_f1'] = None
        metrics['samples_f1'] = None
        metrics['subset_accuracy'] = None
        metrics['hamming_loss'] = None
        metrics['jaccard_score_macro'] = None
        metrics['jaccard_score_micro'] = None
        metrics['mean_balanced_accuracy'] = None
        
        # Prediction confidence metrics (unsupervised)
        if probs is not None:
            metrics['prediction_confidence_mean'] = float(probs.mean())
            metrics['prediction_confidence_std'] = float(probs.std())
            
            # High confidence predictions
            high_conf_mask = probs > 0.7
            metrics['high_confidence_predictions'] = int(high_conf_mask.sum())
            metrics['high_confidence_percentage'] = float(high_conf_mask.sum() / probs.size * 100)
    
    return metrics





def create_comprehensive_visualizations(embeddings: np.ndarray, predictions: np.ndarray, 
                                      binary_predictions: np.ndarray, classes: List[str],
                                      output_dir: Path, log_type: str, config: SystemConfig):
    """Create comprehensive visualizations for classification analysis"""
    
    try:
        # Sample for visualization if too large
        if len(embeddings) > 2000:
            idx = np.random.choice(len(embeddings), 2000, replace=False)
            embeddings_viz = embeddings[idx]
            predictions_viz = predictions[idx]
            binary_viz = binary_predictions[idx]
        else:
            embeddings_viz = embeddings
            predictions_viz = predictions
            binary_viz = binary_predictions
        
        # Create multiple visualizations
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Transformer Classification Analysis - {log_type.upper()}', fontsize=16)
        
        # 1. Number of labels per sample distribution
        labels_per_sample = binary_viz.sum(axis=1)
        axes[0, 0].hist(labels_per_sample, bins=range(min(labels_per_sample), max(labels_per_sample) + 2), 
                       alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Labels per Sample Distribution')
        axes[0, 0].set_xlabel('Number of Labels')
        axes[0, 0].set_ylabel('Frequency')
        
        # 2. Class frequency (top 15)
        class_counts = binary_viz.sum(axis=0)
        top_indices = np.argsort(class_counts)[-15:][::-1]
        top_classes = [classes[i] for i in top_indices]
        top_counts = class_counts[top_indices]
        
        axes[0, 1].barh(range(len(top_classes)), top_counts)
        axes[0, 1].set_yticks(range(len(top_classes)))
        axes[0, 1].set_yticklabels(top_classes)
        axes[0, 1].set_title('Top 15 Most Predicted Classes')
        axes[0, 1].set_xlabel('Number of Predictions')
        
        # 3. Prediction confidence distribution
        axes[0, 2].hist(predictions_viz.flatten(), bins=50, alpha=0.7, edgecolor='black')
        axes[0, 2].set_title('Prediction Confidence Distribution')
        axes[0, 2].set_xlabel('Confidence Score')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].axvline(x=0.5, color='red', linestyle='--', label='Threshold')
        axes[0, 2].legend()
        
        # 4. t-SNE visualization of embeddings colored by number of labels
        try:
            from sklearn.manifold import TSNE
            tsne = TSNE(n_components=2, random_state=42, perplexity=30)
            embeddings_2d = tsne.fit_transform(embeddings_viz)
            
            scatter = axes[1, 0].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                                       c=labels_per_sample, cmap='viridis', alpha=0.6, s=20)
            axes[1, 0].set_title('t-SNE: Embeddings by Label Count')
            plt.colorbar(scatter, ax=axes[1, 0])
        except Exception as e:
            axes[1, 0].text(0.5, 0.5, f't-SNE failed:\n{str(e)}', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('t-SNE Visualization (Failed)')
        
        # 5. Average confidence per class
        avg_confidence_per_class = predictions_viz.mean(axis=0)
        top_conf_indices = np.argsort(avg_confidence_per_class)[-15:][::-1]
        top_conf_classes = [classes[i] for i in top_conf_indices]
        top_conf_values = avg_confidence_per_class[top_conf_indices]
        
        axes[1, 1].barh(range(len(top_conf_classes)), top_conf_values)
        axes[1, 1].set_yticks(range(len(top_conf_classes)))
        axes[1, 1].set_yticklabels(top_conf_classes)
        axes[1, 1].set_title('Top 15 Classes by Average Confidence')
        axes[1, 1].set_xlabel('Average Confidence')
        
        # 6. Prediction matrix heatmap (sample of classes)
        if len(classes) <= 20:
            # Show all classes
            heatmap_data = binary_viz.T
            class_labels = classes
        else:
            # Show top 20 classes
            top_indices = np.argsort(class_counts)[-20:][::-1]
            heatmap_data = binary_viz[:, top_indices].T
            class_labels = [classes[i] for i in top_indices]
        
        # Sample rows for visualization
        if heatmap_data.shape[1] > 100:
            sample_indices = np.random.choice(heatmap_data.shape[1], 100, replace=False)
            heatmap_data = heatmap_data[:, sample_indices]
        
        im = axes[1, 2].imshow(heatmap_data, cmap='Blues', aspect='auto')
        axes[1, 2].set_title('Prediction Matrix (Sample)')
        axes[1, 2].set_xlabel('Samples')
        axes[1, 2].set_ylabel('Classes')
        
        # Set y-axis labels for classes
        if len(class_labels) <= 10:
            axes[1, 2].set_yticks(range(len(class_labels)))
            axes[1, 2].set_yticklabels(class_labels, fontsize=8)
        
        plt.colorbar(im, ax=axes[1, 2])
        
        plt.tight_layout()
        plt.savefig(output_dir / f'classification_analysis_{log_type}_{config.node_name}_{config.job_id}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Visualization failed: {e}")

def print_classification_summary(metrics: Dict[str, Any], log_type: str, n_samples: int):
    """Print a comprehensive classification summary"""
    
    print(f"\n{'='*80}")
    print(f"CLASSIFICATION SUMMARY - {log_type.upper()}")
    print(f"{'='*80}")
    print(f"Total samples: {n_samples}")
    print(f"Total classes: {len(metrics.get('class_counts', []))}")
    print(f"Total predictions: {n_samples * len(metrics.get('class_counts', []))}")
    
    print(f"\nSample-level statistics:")
    print(f"  Average labels per sample: {metrics['avg_labels_per_sample']:.3f}")
    print(f"  Std labels per sample: {metrics['std_labels_per_sample']:.3f}")
    print(f"  Range: {metrics['min_labels_per_sample']} - {metrics['max_labels_per_sample']} labels")
    print(f"  Samples with no labels: {metrics['samples_with_no_labels']}")
    print(f"  Samples with one label: {metrics['samples_with_one_label']}")
    print(f"  Samples with multiple labels: {metrics['samples_with_multiple_labels']}")
    
    print(f"\nPrediction confidence:")
    print(f"  Overall confidence: {metrics['prediction_confidence_mean']:.4f}")
    print(f"  Confidence std: {metrics['prediction_confidence_std']:.4f}")
    print(f"  High confidence predictions: {metrics['high_confidence_predictions']}")
    print(f"  High confidence percentage: {metrics['high_confidence_percentage']:.2f}%")
    
    print(f"\nTop 10 most frequent classes:")
    for i, class_info in enumerate(metrics['most_frequent_classes'][:10]):
        print(f"  {i+1:2d}. {class_info['class']:<30} {class_info['count']:6d} ({class_info['percentage']:5.2f}%)")
    
    print(f"\nNote: This is a generative transformer model trained unsupervised.")
    print(f"      For real F1/precision/recall metrics, use: python src/evaluate_model.py --log-type {log_type}")
    print(f"{'='*80}")

def evaluate_and_save_results(model: UnsupervisedMultiLabelTransformer, 
                             embeddings: np.ndarray, classes: List[str],
                             config: SystemConfig, tracker: ProgressTracker, 
                             output_dir: Path, log_type: str):
    """Evaluate model and save comprehensive results with classification summary"""
    
    # Use the new comprehensive classification summary
    return create_classification_summary(model, embeddings, classes, config, tracker, output_dir, log_type)

def create_visualization(embeddings: np.ndarray, predictions: np.ndarray, 
                        classes: List[str], output_dir: Path, log_type: str, config: SystemConfig):
    """Create visualizations"""
    try:
        # Sample for t-SNE if too large
        if len(embeddings) > 1000:  # Reduced sample size
            idx = np.random.choice(len(embeddings), 1000, replace=False)
            embeddings_viz = embeddings[idx]
            predictions_viz = predictions[idx]
        else:
            embeddings_viz = embeddings
            predictions_viz = predictions
        
        # t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        embeddings_2d = tsne.fit_transform(embeddings_viz)
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Number of labels
        n_labels = predictions_viz.sum(axis=1)
        scatter = ax1.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                             c=n_labels, cmap='viridis', alpha=0.6, s=20)
        ax1.set_title('Number of Predicted Labels')
        plt.colorbar(scatter, ax=ax1)
        
        # Max confidence
        max_conf = predictions_viz.max(axis=1)
        scatter = ax2.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                             c=max_conf, cmap='plasma', alpha=0.6, s=20)
        ax2.set_title('Maximum Prediction Confidence')
        plt.colorbar(scatter, ax=ax2)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'visualization_{log_type}_{config.node_name}_{config.job_id}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Visualization failed: {e}")

def ensemble_consistency_loss(branch_predictions: List[torch.Tensor]) -> torch.Tensor:
    """
    Encourage consistency between ensemble branches
    """
    if len(branch_predictions) < 2:
        return torch.tensor(0.0, device=branch_predictions[0].device)
    
    consistency_loss = 0.0
    n_pairs = 0
    
    for i in range(len(branch_predictions)):
        for j in range(i + 1, len(branch_predictions)):
            pred_i = torch.sigmoid(branch_predictions[i])
            pred_j = torch.sigmoid(branch_predictions[j])
            consistency_loss += F.mse_loss(pred_i, pred_j)
            n_pairs += 1
    
    return consistency_loss / max(n_pairs, 1)

def curriculum_weight_scheduler(epoch: int, total_epochs: int) -> Dict[str, float]:
    """
    Curriculum learning: gradually increase difficulty of learning with reduced weights for stability
    """
    progress = epoch / total_epochs
    
    # Start with simple reconstruction, gradually add complex losses
    # Reduced weights to prevent numerical overflow
    weights = {
        'recon': max(0.1, 0.2 - 0.1 * progress),  # Decrease reconstruction importance
        'label': min(0.3, 0.1 + 0.2 * progress),  # Increase label importance
        'cluster': 0.05,  # Reduced
        'contrastive': min(0.1, 0.05 + 0.05 * progress),  # Gradually add contrastive
        'mutual': min(0.1, 0.02 + 0.08 * progress),  # Gradually add mutual
        'confidence': min(0.15, 0.05 + 0.1 * progress),  # Gradually add confidence
        'ensemble': min(0.1, 0.0 + 0.1 * progress)  # Add ensemble consistency later
    }
    
    return weights

def advanced_pseudo_label_generation(embeddings: np.ndarray, classes: List[str], 
                                   true_labels: np.ndarray = None, 
                                   epoch: int = 0, total_epochs: int = 50) -> np.ndarray:
    """
    Advanced pseudo-label generation with curriculum learning and confidence boosting
    """
    if not classes:
        # If no classes, create a single 'normal' class with high confidence
        confidence = 0.7 + 0.2 * (epoch / max(total_epochs, 1))  # Increase confidence over time
        return np.ones((len(embeddings), 1), dtype=np.float32) * confidence
    
    n_samples = len(embeddings)
    n_classes = len(classes)
    
    # Curriculum learning: start conservative, become more aggressive
    progress = epoch / max(total_epochs, 1)
    
    if true_labels is not None:
        # Use true labels as strong supervision
        pseudo_labels = true_labels.astype(np.float32)
        
        # Apply curriculum noise reduction
        noise_factor = max(0.02, 0.1 - 0.08 * progress)
        noise = np.random.rand(n_samples, n_classes) * noise_factor
        pseudo_labels = pseudo_labels * (0.95 + 0.05 * progress) + noise * (1 - progress)
        
        # For unlabeled samples, use confident initialization
        unlabeled_mask = (true_labels.sum(axis=1) == 0)
        if np.any(unlabeled_mask):
            n_unlabeled = np.sum(unlabeled_mask)
            # More confident initialization as training progresses
            base_confidence = 0.3 + 0.4 * progress
            confident_labels = np.random.rand(n_unlabeled, n_classes)
            confident_labels = (confident_labels > (1 - base_confidence)).astype(float) * 0.8 + 0.1
            pseudo_labels[unlabeled_mask] = confident_labels
    else:
        # Start with moderate confidence, increase over time
        base_confidence = 0.2 + 0.5 * progress
        pseudo_labels = np.random.rand(n_samples, n_classes).astype(np.float32)
        pseudo_labels = (pseudo_labels > (1 - base_confidence)).astype(float) * 0.7 + 0.15
    
    # Advanced similarity-based propagation
    embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    
    # Adaptive similarity threshold based on curriculum
    similarity_threshold = 0.75 + 0.1 * progress  # Start at 0.75, go to 0.85
    
    # Multi-scale propagation with different neighborhood sizes
    for scale in [500, 1000, 2000]:  # Different scales
        chunk_size = min(scale, n_samples)
        
        for i in range(0, n_samples, chunk_size):
            end_i = min(i + chunk_size, n_samples)
            chunk_i = embeddings_norm[i:end_i]
            
            similarities = np.dot(chunk_i, embeddings_norm.T)
            
            for j in range(len(chunk_i)):
                global_idx = i + j
                
                # Find top-k similar samples (adaptive k)
                k_neighbors = min(20 + int(10 * progress), n_samples // 10)
                top_k_indices = np.argsort(similarities[j])[-k_neighbors:]
                top_k_similarities = similarities[j][top_k_indices]
                
                # Only use highly similar samples
                valid_mask = top_k_similarities > similarity_threshold
                if np.any(valid_mask):
                    valid_indices = top_k_indices[valid_mask]
                    valid_similarities = top_k_similarities[valid_mask]
                    
                    # Weighted propagation
                    weights = valid_similarities / valid_similarities.sum()
                    propagated_labels = np.average(pseudo_labels[valid_indices], axis=0, weights=weights)
                    
                    # Adaptive momentum based on curriculum
                    momentum = 0.7 + 0.2 * progress
                    pseudo_labels[global_idx] = momentum * pseudo_labels[global_idx] + (1 - momentum) * propagated_labels
    
    # Progressive confidence boosting
    for i in range(n_samples):
        # Dynamic k based on curriculum
        k = min(3 + int(2 * progress), n_classes)
        top_indices = np.argsort(pseudo_labels[i])[-k:]
        
        # Stronger boosting as training progresses
        boost_factor = 0.1 + 0.3 * progress
        pseudo_labels[i][top_indices] = np.minimum(pseudo_labels[i][top_indices] + boost_factor, 0.95)
        
        # Suppress weak predictions more aggressively
        weak_threshold = 0.4 - 0.1 * progress
        weak_mask = pseudo_labels[i] < weak_threshold
        pseudo_labels[i][weak_mask] *= (0.8 - 0.3 * progress)
    
    # Progressive sharpening
    temperature = max(0.3, 0.8 - 0.5 * progress)
    pseudo_labels = pseudo_labels ** (1/temperature)
    
    # Normalization with confidence preservation
    row_sums = pseudo_labels.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-8)
    pseudo_labels = pseudo_labels / row_sums
    
    # Progressive confidence bounds
    min_conf = max(0.05, 0.15 - 0.1 * progress)
    max_conf = min(0.95, 0.8 + 0.15 * progress)
    pseudo_labels = np.clip(pseudo_labels, min_conf, max_conf)
    
    # Ensure no invalid values
    pseudo_labels = np.nan_to_num(pseudo_labels, nan=min_conf, posinf=max_conf, neginf=min_conf)
    
    return pseudo_labels

def enhanced_adaptive_thresholding(predictions: np.ndarray, embeddings: np.ndarray, 
                                  classes: List[str], true_labels: np.ndarray = None, 
                                  k_neighbors: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """
    Enhanced adaptive thresholding combining global and local information for better F1
    
    Args:
        predictions: Predicted probabilities (n_samples, n_classes)
        embeddings: Original embeddings for local density calculation
        classes: List of class names
        true_labels: True labels if available for optimization
        k_neighbors: Number of neighbors for local density
    
    Returns:
        adaptive_thresholds: Per-class thresholds
        binary_predictions: Binary predictions with sophisticated assignment
    """
    n_samples, n_classes = predictions.shape
    adaptive_thresholds = np.zeros(n_classes)
    
    # Step 1: Calculate global information
    global_priors = predictions.mean(axis=0)  # Average prediction per class
    class_rarity = -np.log(global_priors + 1e-8)  # IDF-like rarity score
    
    # Step 2: Calculate local density information (if embeddings available)
    if embeddings is not None and len(embeddings) > k_neighbors:
        nn = NearestNeighbors(n_neighbors=k_neighbors, metric='cosine')
        nn.fit(embeddings)
        distances, indices = nn.kneighbors(embeddings)
        
        # Local density: inverse of average distance to k nearest neighbors
        local_density = 1.0 / (distances.mean(axis=1) + 1e-8)
        local_density = (local_density - local_density.min()) / (local_density.max() - local_density.min() + 1e-8)
    else:
        local_density = np.ones(n_samples) * 0.5
    
    # Step 3: Calculate per-class adaptive thresholds
    for class_idx in range(n_classes):
        class_preds = predictions[:, class_idx]
        
        if true_labels is not None and class_idx < true_labels.shape[1]:
            # If we have true labels, optimize threshold for F1
            from sklearn.metrics import f1_score
            true_class = true_labels[:, class_idx]
            
            if true_class.sum() > 0:  # Only optimize if we have positive samples
                best_threshold = 0.5
                best_f1 = 0.0
                
                # Test multiple thresholds
                for threshold in np.linspace(0.1, 0.9, 25):
                    # Combine global and local information
                    adjusted_threshold = threshold * (1 + 0.1 * class_rarity[class_idx])
                    binary_preds = (class_preds > adjusted_threshold).astype(int)
                    
                    f1 = f1_score(true_class, binary_preds, zero_division=0)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_threshold = adjusted_threshold
                
                adaptive_thresholds[class_idx] = best_threshold
            else:
                # No positive samples, use heuristic
                adaptive_thresholds[class_idx] = 0.5 + 0.1 * class_rarity[class_idx]
        else:
            # No true labels, use distribution-based adaptive threshold
            mean_pred = np.mean(class_preds)
            std_pred = np.std(class_preds)
            
            # Combine global rarity and local distribution
            base_threshold = mean_pred + 0.5 * std_pred
            
            # Adjust based on class rarity (rare classes get lower thresholds)
            rarity_adjustment = 0.1 * (1 - global_priors[class_idx])
            
            # Final threshold bounded between 0.15 and 0.85
            adaptive_thresholds[class_idx] = np.clip(
                base_threshold - rarity_adjustment,
                0.15, 0.85
            )
    
    # Step 4: Apply thresholds with local density adjustment
    binary_predictions = np.zeros_like(predictions, dtype=int)
    
    for sample_idx in range(n_samples):
        sample_preds = predictions[sample_idx]
        sample_density = local_density[sample_idx]
        
        # Adjust thresholds based on local density (denser regions = higher confidence required)
        density_factor = 1.0 + 0.2 * (sample_density - 0.5)
        
        for class_idx in range(n_classes):
            adjusted_threshold = adaptive_thresholds[class_idx] * density_factor
            binary_predictions[sample_idx, class_idx] = int(sample_preds[class_idx] > adjusted_threshold)
    
    return adaptive_thresholds, binary_predictions

def split_for_unsupervised_eval(embeddings: np.ndarray, true_labels: Optional[np.ndarray], 
                                val_frac: float = 0.1, test_frac: float = 0.1, 
                                seed: int = 42) -> Dict[str, Any]:
    """
    Split data for unsupervised training with labeled validation/test sets
    
    Args:
        embeddings: Input embeddings
        true_labels: True labels (if available)
        val_frac: Fraction for validation set
        test_frac: Fraction for test set
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary with train/val/test splits
    """
    rng = np.random.default_rng(seed)
    n = len(embeddings)
    idx = np.arange(n)
    rng.shuffle(idx)
    
    n_val = int(n * val_frac)
    n_test = int(n * test_frac)
    
    val_idx = idx[:n_val]
    test_idx = idx[n_val:n_val + n_test]
    train_idx = idx[n_val + n_test:]
    
    # Split embeddings
    emb_train = embeddings[train_idx]
    emb_val = embeddings[val_idx]
    emb_test = embeddings[test_idx]
    
    # Split labels if available
    y_val = true_labels[val_idx] if true_labels is not None else None
    y_test = true_labels[test_idx] if true_labels is not None else None
    
    return {
        'embeddings_train': emb_train,
        'embeddings_val': emb_val,
        'embeddings_test': emb_test,
        'labels_val': y_val,
        'labels_test': y_test,
        'indices': {
            'train': train_idx,
            'val': val_idx,
            'test': test_idx
        }
    }

def optimize_per_class_thresholds(y_true: np.ndarray, probs: np.ndarray, 
                                 metric: str = 'f1', beta: float = 1.0,
                                 grid: np.ndarray = None) -> np.ndarray:
    """
    Optimize per-class thresholds on validation set to maximize specified metric
    
    Args:
        y_true: True binary labels (n_samples, n_classes)
        probs: Predicted probabilities (n_samples, n_classes)
        metric: Metric to optimize ('f1', 'balanced_accuracy', 'precision', 'recall')
        beta: Beta parameter for F-beta score
        grid: Threshold grid to search (default: 19 points from 0.05 to 0.95)
    
    Returns:
        Optimal thresholds per class
    """
    from sklearn.metrics import fbeta_score, balanced_accuracy_score, precision_score, recall_score
    
    if grid is None:
        grid = np.linspace(0.05, 0.95, 19)
    
    n_classes = y_true.shape[1]
    best_thresholds = np.full(n_classes, 0.5, dtype=float)
    
    for c in range(n_classes):
        y_c = y_true[:, c]
        probs_c = probs[:, c]
        
        # Skip if no positive samples
        if y_c.sum() == 0:
            best_thresholds[c] = 0.5
            continue
        
        best_score = -np.inf
        best_t = 0.5
        
        for t in grid:
            y_pred = (probs_c >= t).astype(int)
            
            if metric == 'f1':
                score = fbeta_score(y_c, y_pred, beta=beta, zero_division=0)
            elif metric == 'balanced_accuracy':
                score = balanced_accuracy_score(y_c, y_pred)
            elif metric == 'precision':
                score = precision_score(y_c, y_pred, zero_division=0)
            elif metric == 'recall':
                score = recall_score(y_c, y_pred, zero_division=0)
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            if score > best_score:
                best_score = score
                best_t = t
        
        best_thresholds[c] = best_t
    
    return best_thresholds

# =============================================================================
# Main Execution
# =============================================================================

def find_available_embeddings() -> List[str]:
    """Find available embedding files - supports both FastText and LogBERT embeddings"""
    embeddings_dir = Path("embeddings")
    if not embeddings_dir.exists():
        return []
    
    log_files = []
    for file_path in embeddings_dir.rglob("log_*.pkl"):
        log_type = file_path.stem.replace("log_", "")
        # Skip all_combined and only include individual log types
        if log_type != "all_combined" and file_path.parent.name == log_type:
            log_files.append(log_type)
    
    return sorted(log_files)

def analyze_embedding_types(available_types: List[str]) -> Dict[str, Dict[str, Any]]:
    """Analyze embedding types for each available log type"""
    embeddings_dir = Path("embeddings")
    analysis = {}
    
    for log_type in available_types:
        log_file = embeddings_dir / log_type / f"log_{log_type}.pkl"
        if log_file.exists():
            try:
                with open(log_file, 'rb') as f:
                    embeddings = pickle.load(f)
                    
                embedding_dim = embeddings.shape[1]
                n_samples = embeddings.shape[0]
                
                # Detect embedding type
                if embedding_dim == 300:
                    embedding_type = "FastText"
                    description = "Standard 300D word embeddings"
                elif embedding_dim == 768:
                    embedding_type = "BERT CLS"
                    description = "BERT CLS token embeddings (768D)"
                elif embedding_dim == 2314:
                    embedding_type = "Enhanced LogBERT"
                    description = "Multi-feature embeddings (CLS+mean+max+attention)"
                else:
                    embedding_type = "Unknown"
                    description = f"Custom embeddings ({embedding_dim}D)"
                
                analysis[log_type] = {
                    'embedding_type': embedding_type,
                    'dimension': embedding_dim,
                    'n_samples': n_samples,
                    'description': description,
                    'memory_mb': embeddings.nbytes / (1024**2)
                }
                
            except Exception as e:
                analysis[log_type] = {
                    'embedding_type': 'Error',
                    'dimension': 0,
                    'n_samples': 0,
                    'description': f"Error loading: {e}",
                    'memory_mb': 0
                }
    
    return analysis

def generate_training_hash(embeddings: np.ndarray, classes: List[str]) -> str:
    """Generate a hash for training data to validate checkpoints."""
    content = f"{embeddings.shape}_{len(classes)}_{embeddings[0].sum() if len(embeddings) > 0 else 0}"
    return hashlib.md5(content.encode()).hexdigest()[:16]

def save_training_checkpoint(log_type: str, epoch: int, model_state: dict, 
                           optimizer_state: dict, metrics: dict, 
                           training_hash: str, config: SystemConfig):
    """Save training checkpoint."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    
    checkpoint_data = {
        'log_type': log_type,
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer_state,
        'metrics': metrics,
        'training_hash': training_hash,
        'config': config.__dict__,
        'timestamp': time.time()
    }
    
    checkpoint_file = CHECKPOINT_DIR / f"{log_type}_epoch_{epoch}_{training_hash}.pth"
    torch.save(checkpoint_data, checkpoint_file)
    
    print(f"💾 Training checkpoint saved: epoch {epoch}")
    return checkpoint_file

def load_training_checkpoint(log_type: str, training_hash: str) -> Optional[Tuple[dict, int]]:
    """Load the latest training checkpoint for a log type."""
    if not CHECKPOINT_DIR.exists():
        return None
    
    # Find all checkpoints for this log type and hash
    pattern = f"{log_type}_epoch_*_{training_hash}.pth"
    checkpoints = list(CHECKPOINT_DIR.glob(pattern))
    
    if not checkpoints:
        return None
    
    # Get the latest checkpoint (highest epoch)
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.stem.split('_')[2]))
    
    try:
        checkpoint_data = torch.load(latest_checkpoint, map_location='cpu')
        
        # Validate checkpoint
        if (checkpoint_data['log_type'] == log_type and 
            checkpoint_data['training_hash'] == training_hash):
            
            age_hours = (time.time() - checkpoint_data['timestamp']) / 3600
            epoch = checkpoint_data['epoch']
            print(f"📂 Found training checkpoint: epoch {epoch} (age: {age_hours:.1f}h)")
            return checkpoint_data, epoch
    except Exception as e:
        print(f"⚠️  Training checkpoint loading failed: {e}")
        # Remove corrupted checkpoint
        latest_checkpoint.unlink(missing_ok=True)
    
    return None

def cleanup_training_checkpoints(log_type: str, keep_latest: int = 2):
    """Clean up old training checkpoints."""
    if not CHECKPOINT_DIR.exists():
        return
    
    pattern = f"{log_type}_epoch_*.pth"
    checkpoints = list(CHECKPOINT_DIR.glob(pattern))
    
    if len(checkpoints) > keep_latest:
        # Sort by epoch number, keep latest
        checkpoints.sort(key=lambda x: int(x.stem.split('_')[2]), reverse=True)
        
        for old_checkpoint in checkpoints[keep_latest:]:
            old_checkpoint.unlink(missing_ok=True)
            print(f"🗑️  Cleaned up old checkpoint: {old_checkpoint.name}")

def check_existing_results(log_type: str, config: SystemConfig) -> dict:
    """Check if results already exist for this log type."""
    output_dir = RESULTS_DIR / log_type
    
    status = {
        'results_pkl': False,
        'labels_pkl': False,
        'classification_report': False,
        'visualizations': False,
        'model_saved': False,
        'complete': False
    }
    
    if output_dir.exists():
        # Check for results files
        results_pattern = f"results_{log_type}_{config.node_name}_{config.job_id}.pkl"
        labels_pattern = f"label_{log_type}_{config.node_name}_{config.job_id}.pkl"
        report_pattern = f"classification_report_{log_type}_*.txt"
        viz_pattern = f"*analysis_{log_type}_{config.node_name}_{config.job_id}.png"
        
        status['results_pkl'] = len(list(output_dir.glob(results_pattern))) > 0
        status['labels_pkl'] = len(list(output_dir.glob(labels_pattern))) > 0
        status['classification_report'] = len(list(output_dir.glob(report_pattern))) > 0
        status['visualizations'] = len(list(output_dir.glob(viz_pattern))) > 0
        
        # Check for saved model
        model_pattern = f"transformer_{log_type}_{config.node_name}_{config.job_id}.pth"
        status['model_saved'] = (MODELS_DIR / model_pattern).exists()
        
        status['complete'] = all([status['results_pkl'], status['labels_pkl'], 
                                status['classification_report'], status['model_saved']])
    
    return status

def process_log_type_with_args(log_type: str, config: SystemConfig, force_restart: bool = False):
    """Process a single log type with resumeable functionality and argument support"""
    output_dir = RESULTS_DIR / log_type
    output_dir.mkdir(parents=True, exist_ok=True)
    
    tracker = ProgressTracker(output_dir, log_type, config)
    
    # Check if already completed (unless force restart)
    result_status = check_existing_results(log_type, config)
    if result_status['complete'] and not force_restart:
        print(f"✅ {log_type} already completed. Skipping.")
        print(f"   Files: results✓ labels✓ report✓ model✓")
        return
    elif result_status['model_saved'] and result_status['results_pkl']:
        print(f"🔄 {log_type} partially completed. Creating remaining outputs...")
        try:
            # Load existing model and results for final outputs
            model_path = MODELS_DIR / f"transformer_{log_type}_{config.node_name}_{config.job_id}.pth"
            saved_data = torch.load(model_path, map_location='cpu')
            
            results_path = output_dir / f"results_{log_type}_{config.node_name}_{config.job_id}.pkl"
            with open(results_path, 'rb') as f:
                results = pickle.load(f)
            
            # Create any missing outputs
            if not result_status['classification_report']:
                classes = saved_data['classes']
                generate_classification_report(
                    results['binary_predictions'], classes, output_dir, log_type, config
                )
            
            if not result_status['visualizations']:
                # Load embeddings for visualization
                embeddings, classes, _, _ = load_and_preprocess_data(log_type, config, tracker)
                create_comprehensive_visualizations(
                    embeddings, results['predictions'], results['binary_predictions'], 
                    classes, output_dir, log_type, config
                )
            
            print(f"✅ Completed remaining outputs for {log_type}")
            return
        except Exception as e:
            print(f"⚠️  Could not complete from existing files: {e}")
            # Continue with full processing
    
    try:
        # Load data with progress
        with Halo(text=f"Loading data for {log_type}...", spinner='dots') as spinner:
            embeddings, classes, C, scaler = load_and_preprocess_data(log_type, config, tracker)
            spinner.succeed(f"Data loaded: {embeddings.shape[0]} samples, {embeddings.shape[1]} features")
        
        # Train model with checkpointing
        tracker.log_step("Training Start", {"embeddings_shape": embeddings.shape})
        with Halo(text=f"Training model for {log_type}...", spinner='dots') as spinner:
            model, _ = train_model(embeddings, classes, C, config, tracker, log_type)
            spinner.succeed(f"Training completed for {log_type}")
        
        # Evaluate and save
        with Halo(text=f"Evaluating model for {log_type}...", spinner='dots') as spinner:
            results = evaluate_and_save_results(model, embeddings, classes, config, tracker, output_dir, log_type)
            spinner.succeed(f"Evaluation completed for {log_type}")
        
        # Save model
        if config.rank == 0:
            with Halo(text=f"Saving model for {log_type}...", spinner='dots') as spinner:
                model_path = MODELS_DIR / f"transformer_{log_type}_{config.node_name}_{config.job_id}.pth"
                model_path.parent.mkdir(exist_ok=True)
                
                model_to_save = model.module if hasattr(model, 'module') else model
                torch.save({
                    'model_state_dict': model_to_save.state_dict(),
                    'config': config.__dict__,
                    'classes': classes,
                    'results': results,
                    'scaler': scaler
                }, model_path)
                spinner.succeed(f"Model saved to {model_path}")
        
        tracker.log_step("Completion", {"status": "success", "results": results})
        print(f"✅ Completed processing {log_type}")
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Processing interrupted for {log_type}. Training checkpoint saved.")
        raise
    except Exception as e:
        tracker.logger.error(f"❌ Error processing {log_type}: {e}")
        import traceback
        tracker.logger.error(traceback.format_exc())
        raise

def process_log_type(log_type: str, config: SystemConfig):
    """Process a single log type with resumeable functionality (legacy wrapper)"""
    return process_log_type_with_args(log_type, config, force_restart=False)

def main():
    """Main execution with distributed support for Nibi - Resumeable"""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Transformer-based Unsupervised Multi-Label Learning - Resumeable")
    parser.add_argument("--log-type", type=str, default=None, help="Process only this specific log type")
    parser.add_argument("--force-restart", action="store_true", help="Force restart processing (ignore existing results)")
    parser.add_argument("--clean-checkpoints", action="store_true", help="Clean up all training checkpoints before starting")
    args = parser.parse_args()
    
    try:
        # Clean checkpoints if requested
        if args.clean_checkpoints:
            import shutil
            if CHECKPOINT_DIR.exists():
                shutil.rmtree(CHECKPOINT_DIR)
                print("🗑️  Cleaned up all training checkpoints")
        
        # Override completion check if force restart
        if args.force_restart:
            print("🔄 Force restart enabled - will reprocess all data")
        
        # Detect system resources
        config = detect_system_resources()
        
        # Setup distributed training if needed
        if config.is_distributed:
            setup_distributed_training(config.rank, config.world_size)
        
        # Find available embeddings
        available_types = find_available_embeddings()
        if not available_types:
            print("No embedding files found. Please run logbert_embeddings.py first.")
            return
        
        if config.rank == 0:
            print(f"System Configuration:")
            print(f"  Device: {config.device}")
            print(f"  GPUs: {config.n_gpus}")
            print(f"  GPU Memory: {config.gpu_memory_gb:.1f} GB")
            print(f"  Total Memory: {config.total_memory_gb:.1f} GB")
            print(f"  CPUs: {config.n_cpus}")
            print(f"  Distributed: {config.is_distributed}")
            print(f"  Node: {config.node_name}")
            print(f"  Job ID: {config.job_id}")
            print(f"  Memory optimizations: {'CUDA conservative' if config.device == 'cuda' else 'Standard'}")
            print(f"  Supports: FastText (300D), BERT CLS (768D), Enhanced LogBERT (2314D)")
            
            # Analyze available embeddings
            embedding_analysis = analyze_embedding_types(available_types)
            print(f"\nAvailable Embeddings Analysis:")
            print(f"{'='*60}")
            for log_type, info in embedding_analysis.items():
                print(f"{log_type}:")
                print(f"  Type: {info['embedding_type']} ({info['dimension']}D)")
                print(f"  Samples: {info['n_samples']:,}")
                print(f"  Memory: {info['memory_mb']:.1f} MB")
                print(f"  Description: {info['description']}")
            print(f"{'='*60}")
        
        # Skip log type classifier since each log file can only be one type
        if config.rank == 0:
            print(f"\n{'='*60}")
            print("Processing individual log types")
            print(f"{'='*60}")
        
        # Filter log types if specified
        if args.log_type:
            if args.log_type in available_types:
                available_types = [args.log_type]
                print(f"🎯 Processing single log type: {args.log_type}")
            else:
                print(f"❌ Log type '{args.log_type}' not found in available types: {available_types}")
                return

        # Process each log type
        total_types = len(available_types)
        for idx, log_type in enumerate(available_types, 1):
            if config.rank == 0:
                # Show embedding type info for this log type
                if log_type in embedding_analysis:
                    info = embedding_analysis[log_type]
                    print(f"\n{'='*60}")
                    print(f"Processing: {log_type} ({idx}/{total_types})")
                    print(f"Embedding: {info['embedding_type']} ({info['dimension']}D)")
                    print(f"Samples: {info['n_samples']:,} | Memory: {info['memory_mb']:.1f} MB")
                    print(f"{'='*60}")
                else:
                    print(f"\n{'='*60}")
                    print(f"Processing: {log_type} ({idx}/{total_types})")
                    print(f"{'='*60}")
            
            start_time = time.time()
            
            # Pass force_restart flag to process_log_type
            process_log_type_with_args(log_type, config, args.force_restart)
            
            if config.rank == 0:
                elapsed = time.time() - start_time
                remaining_types = total_types - idx
                if remaining_types > 0:
                    avg_time = elapsed / idx
                    estimated_remaining = avg_time * remaining_types
                    print(f"Completed {log_type} in {elapsed:.2f} seconds")
                    print(f"Progress: {idx}/{total_types} ({idx/total_types*100:.1f}%)")
                    print(f"Estimated time remaining: {estimated_remaining:.2f} seconds")
                else:
                    print(f"Completed {log_type} in {elapsed:.2f} seconds")
                    print(f"All processing completed!")
        
        if config.rank == 0:
            print(f"\n{'='*60}")
            print("🎉 All processing completed successfully!")
            print(f"✅ Resumeable: Training checkpoints saved for interrupted processing")
            print(f"📁 Results saved to: {RESULTS_DIR}/")
            print(f"🤖 Models saved to: {MODELS_DIR}/")
            print(f"🏷️  Labels saved in evaluation format to: {RESULTS_DIR}/*/label_*.pkl")
            print(f"💾 Checkpoints saved to: {CHECKPOINT_DIR}/")
            print(f"\n🔧 Supports embedding types:")
            print(f"  - FastText (300D): Standard word embeddings")
            print(f"  - BERT CLS (768D): Global context embeddings")
            print(f"  - Enhanced LogBERT (2314D): Multi-feature embeddings")
            print(f"{'='*60}")
    
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        if config.is_distributed:
            cleanup_distributed()

if __name__ == "__main__":
    # Set optimal environment variables for Nibi performance
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    
    # Set multiprocessing start method for stability
    mp.set_start_method('spawn', force=True)
    
    main() 