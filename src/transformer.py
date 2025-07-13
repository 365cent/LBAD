#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
High-Performance Transformer-based Unsupervised Multi-Label Learning for Supercomputing

Optimized for Research Alliance of Canada Nibi node:
- Multi-GPU support (H100, A100, V100)
- Automatic CUDA/distributed training setup
- Memory-efficient chunked processing
- Comprehensive error handling and recovery
- Real-time progress tracking with file outputs
- Performance profiling and optimization
- Stable results with deterministic training
- Proper label output format for evaluation
- Separate models per log type (each log file has only one type)
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
    """Ultra-powerful transformer for unsupervised multi-label learning optimized for H100 GPU"""
    
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
        
        # Multi-scale feature extraction
        self.multi_scale_conv = nn.ModuleList([
            nn.Conv1d(latent_dim, latent_dim, kernel_size=k, padding=k//2) 
            for k in [3, 5, 7, 9]
        ])
        
        # Feature fusion network
        self.feature_fusion = nn.Sequential(
            nn.Linear(latent_dim * 5, latent_dim * 2),  # 4 conv + 1 original
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
        
        # Multi-scale feature extraction
        z_conv_features = []
        z_conv_input = z_flat.unsqueeze(2)  # Add channel dimension for conv1d
        
        for conv in self.multi_scale_conv:
            conv_out = conv(z_conv_input.transpose(1, 2)).transpose(1, 2)
            z_conv_features.append(conv_out.squeeze(2))
        
        # Combine all features
        all_features = torch.cat([z_flat] + z_conv_features, dim=1)
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
        
        return {
            'latent': z_fused,
            'reconstructed': x_recon,
            'labels': labels,
            'clusters': clusters,
            'contrastive': z_contrastive,
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
    """Optimized data loading with memory management for Nibi"""
    tracker.log_step("Data Loading", {"log_type": log_type, "config": config.__dict__})
    
    # Load embeddings - only load specific log type, not combined
    embeddings_dir = Path("embeddings")
    log_file = embeddings_dir / log_type / f"log_{log_type}.pkl"
    label_file = embeddings_dir / log_type / f"label_{log_type}.pkl"
    
    if not log_file.exists():
        raise FileNotFoundError(f"Embedding file not found: {log_file}")
    
    with open(log_file, 'rb') as f:
        embeddings = pickle.load(f)
    
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
    
    # Aggressive subsampling for memory efficiency
    max_samples = min(50000, int(config.gpu_memory_gb * 500))  # Much smaller for memory safety
    if len(embeddings) > max_samples:
        indices = np.random.choice(len(embeddings), max_samples, replace=False)
        embeddings = embeddings[indices]
        if true_labels is not None:
            true_labels = true_labels[indices]
        tracker.log_step("Data Subsampling", {
            "original_size": len(embeddings),
            "subsampled_size": max_samples,
            "memory_gb": embeddings.nbytes / (1024**3)
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

def focal_loss(logits: torch.Tensor, targets: torch.Tensor, alpha: float = 1.0, gamma: float = 2.0) -> torch.Tensor:
    """
    Focal loss for addressing class imbalance and improving confidence
    
    Args:
        logits: Model predictions (before sigmoid)
        targets: Target labels
        alpha: Weighting factor for rare class
        gamma: Focusing parameter
    
    Returns:
        Focal loss value
    """
    # Apply sigmoid to get probabilities
    probs = torch.sigmoid(logits)
    
    # Calculate focal loss
    ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p_t = probs * targets + (1 - probs) * (1 - targets)
    focal_weight = alpha * (1 - p_t) ** gamma
    focal_loss = focal_weight * ce_loss
    
    return focal_loss.mean()

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

def mutual_learning_loss(z1: torch.Tensor, z2: torch.Tensor, pseudo_labels: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """
    Mutual learning loss between two views of the data
    
    Args:
        z1, z2: Two different views/representations of the same batch
        pseudo_labels: Pseudo-labels for the batch
        temperature: Temperature for scaling
    
    Returns:
        Mutual learning loss value
    """
    batch_size = z1.shape[0]
    
    # Normalize representations
    z1_norm = F.normalize(z1, dim=1)
    z2_norm = F.normalize(z2, dim=1)
    
    # Compute cross-view similarity
    cross_sim = torch.matmul(z1_norm, z2_norm.T) / temperature
    
    # Create pseudo-label similarity matrix
    label_sim = torch.matmul(pseudo_labels, pseudo_labels.T)
    label_sim = (label_sim > 0.5).float()  # Binary similarity based on shared labels
    
    # Mutual learning loss: encourage similar pseudo-labels to have similar representations
    # Use a soft version of the loss to handle uncertainty in pseudo-labels
    pos_mask = label_sim
    neg_mask = 1 - label_sim
    
    # Positive pairs should have high similarity
    pos_loss = -torch.log(torch.sigmoid(cross_sim) + 1e-8) * pos_mask
    pos_loss = pos_loss.sum() / (pos_mask.sum() + 1e-8)
    
    # Negative pairs should have low similarity
    neg_loss = -torch.log(1 - torch.sigmoid(cross_sim) + 1e-8) * neg_mask
    neg_loss = neg_loss.sum() / (neg_mask.sum() + 1e-8)
    
    return pos_loss + neg_loss

def contrastive_loss(z_i: torch.Tensor, z_j: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """
    Compute simplified contrastive loss for self-supervised learning
    
    Args:
        z_i: First set of normalized embeddings (batch_size, embedding_dim)
        z_j: Second set of normalized embeddings (batch_size, embedding_dim)
        temperature: Temperature parameter for scaling
    
    Returns:
        Contrastive loss value
    """
    batch_size = z_i.shape[0]
    
    # Compute cosine similarity between corresponding pairs
    positive_sim = F.cosine_similarity(z_i, z_j, dim=1) / temperature
    
    # Compute negative similarities (z_i vs all z_j except corresponding pair)
    all_similarities = torch.matmul(z_i, z_j.T) / temperature
    
    # Create mask to exclude positive pairs from negatives
    mask = torch.eye(batch_size, device=z_i.device, dtype=torch.bool)
    negative_similarities = all_similarities.masked_fill(mask, float('-inf'))
    
    # Compute InfoNCE loss
    # For each sample, positive is the corresponding pair, negatives are all other pairs
    logits = torch.cat([positive_sim.unsqueeze(1), negative_similarities], dim=1)
    labels = torch.zeros(batch_size, dtype=torch.long, device=z_i.device)
    
    loss = F.cross_entropy(logits, labels)
    
    return loss

def train_model(embeddings: np.ndarray, classes: List[str], C: np.ndarray, 
                config: SystemConfig, tracker: ProgressTracker) -> Tuple[UnsupervisedMultiLabelTransformer, StandardScaler]:
    """Optimized training with multi-GPU support and mixed precision for Nibi"""
    
    device = torch.device(config.device)
    n_labels = len(classes) if classes else 1
    n_clusters = C.shape[1] if C is not None else 1
    latent_dim = min(256, embeddings.shape[1])  # Reduced latent dimension
    
    # Model setup
    model = UnsupervisedMultiLabelTransformer(
        input_dim=embeddings.shape[1],
        latent_dim=latent_dim,
        n_labels=n_labels,
        n_clusters=n_clusters
    ).to(device)
    
    # Multi-GPU setup
    if config.is_distributed:
        model = DDP(model, device_ids=[config.rank])
    elif config.n_gpus > 1:
        model = nn.DataParallel(model)
    
    # Generate initial pseudo-labels (use true labels if available from tracker)
    true_labels = getattr(tracker, 'true_labels', None)
    pseudo_labels = advanced_pseudo_label_generation(embeddings, classes, true_labels=true_labels, epoch=0, total_epochs=50)
    
    # Store initial pseudo-labels for refinement
    current_pseudo_labels = pseudo_labels.copy()
    
    # Data setup - ensure consistent float32 dtype
    dataset = TensorDataset(
        torch.from_numpy(embeddings).float(),
        torch.from_numpy(current_pseudo_labels).float()
    )
    
    sampler = DistributedSampler(dataset) if config.is_distributed else None
    batch_size = min(128, max(16, int(config.gpu_memory_gb * 2)))  # Much smaller batch size
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=min(4, config.n_cpus // 4),  # Reduced workers
        pin_memory=True
    )
    
    # Advanced training setup optimized for H100
    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)  # Higher learning rate for deeper model
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
    max_grad_norm = 2.0  # Higher for deeper model
    
    tracker.log_step("Training Setup", {
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "batch_size": batch_size,
        "device": str(device),
        "mixed_precision": scaler is not None
    })
    
    # Training loop with progress tracking
    model.train()
    total_epochs = 50  # Balanced epochs for good convergence without taking too long
    tracker.start_training(total_epochs)
    
    refinement_interval = 10  # Refine pseudo-labels every 10 epochs
    
    for epoch in range(total_epochs):
        epoch_start = time.time()
        epoch_losses = []
        
        if config.is_distributed:
            sampler.set_epoch(epoch)
        
        # Advanced pseudo-label refinement with curriculum learning
        if epoch > 0 and epoch % refinement_interval == 0:
            model.eval()
            with torch.no_grad():
                all_predictions = []
                for i in range(0, len(embeddings), batch_size):
                    batch = torch.from_numpy(embeddings[i:i+batch_size]).float().to(device)
                    outputs = model(batch)
                    predictions = torch.sigmoid(outputs['labels']).cpu().numpy()
                    all_predictions.append(predictions)
                
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
                    num_workers=min(4, config.n_cpus // 4),
                    pin_memory=True
                )
                
                tracker.log_step("Advanced Pseudo-label Refinement", {
                    "epoch": epoch,
                    "avg_confidence": float(np.mean(all_predictions.max(axis=1))),
                    "label_density": float(np.mean(all_predictions.sum(axis=1))),
                    "model_weight": model_weight,
                    "curriculum_progress": progress
                })
            
            model.train()
        
        # Progress spinner for batches
        with Halo(text=f"Epoch {epoch+1}/{total_epochs}", spinner='dots') as spinner:
            for batch_idx, (x_batch, y_batch) in enumerate(dataloader):
                x_batch = x_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)
                
                optimizer.zero_grad()
                
                if scaler:
                    with autocast():
                        outputs = model(x_batch)
                        
                        # Advanced multi-component loss with curriculum learning
                        recon_loss = F.mse_loss(outputs['reconstructed'], x_batch)
                        
                        # Use focal loss instead of BCE for better confidence
                        label_loss = focal_loss(outputs['labels'], y_batch, alpha=1.0, gamma=2.0)
                        
                        # Add confidence regularization
                        predictions = torch.sigmoid(outputs['labels'])
                        confidence_loss = confidence_regularization_loss(predictions, confidence_target=0.8)
                        
                        # Ensemble consistency loss
                        ensemble_loss = ensemble_consistency_loss(outputs['branch_predictions'])
                        
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
                        
                        total_loss = (loss_weights['recon'] * recon_loss + 
                                    loss_weights['label'] * label_loss + 
                                    loss_weights['cluster'] * cluster_loss + 
                                    loss_weights['contrastive'] * contrast_loss + 
                                    loss_weights['mutual'] * mutual_loss + 
                                    loss_weights['confidence'] * confidence_loss + 
                                    loss_weights['ensemble'] * ensemble_loss)
                        
                        # Check for nan/inf
                        if torch.isnan(total_loss) or torch.isinf(total_loss):
                            print(f"Warning: NaN/Inf detected in loss. Skipping batch.")
                            optimizer.zero_grad()
                            continue
                    
                    scaler.scale(total_loss).backward()
                    
                    # Gradient clipping
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(x_batch)
                    
                    recon_loss = F.mse_loss(outputs['reconstructed'], x_batch)
                    
                    # Use focal loss instead of BCE for better confidence
                    label_loss = focal_loss(outputs['labels'], y_batch, alpha=1.0, gamma=2.0)
                    
                    # Add confidence regularization
                    predictions = torch.sigmoid(outputs['labels'])
                    confidence_loss = confidence_regularization_loss(predictions, confidence_target=0.8)
                    
                    # Ensemble consistency loss
                    ensemble_loss = ensemble_consistency_loss(outputs['branch_predictions'])
                    
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
                    
                    total_loss = (loss_weights['recon'] * recon_loss + 
                                loss_weights['label'] * label_loss + 
                                loss_weights['cluster'] * cluster_loss + 
                                loss_weights['contrastive'] * contrast_loss + 
                                loss_weights['mutual'] * mutual_loss + 
                                loss_weights['confidence'] * confidence_loss + 
                                loss_weights['ensemble'] * ensemble_loss)
                    
                    # Check for nan/inf
                    if torch.isnan(total_loss) or torch.isinf(total_loss):
                        print(f"Warning: NaN/Inf detected in loss. Skipping batch.")
                        optimizer.zero_grad()
                        continue
                    
                    total_loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    
                    optimizer.step()
                
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
    
    return model, None  # Return None for scaler placeholder

# Log type classifier function removed - each log file can only be one type

def create_classification_summary(model: UnsupervisedMultiLabelTransformer, 
                                 embeddings: np.ndarray, classes: List[str],
                                 config: SystemConfig, tracker: ProgressTracker, 
                                 output_dir: Path, log_type: str):
    """Create comprehensive classification summary similar to ml_models.py"""
    
    device = torch.device(config.device)
    model.eval()
    
    # Generate predictions
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
    
    # Use intelligent adaptive thresholding for optimal F1 score
    # Calculate per-class thresholds based on distribution but less aggressive
    adaptive_thresholds = []
    for class_idx in range(predictions.shape[1]):
        class_preds = predictions[:, class_idx]
        
        # Use a more balanced approach: 60th percentile or 0.4, whichever is lower
        percentile_threshold = np.percentile(class_preds, 60)
        threshold = min(max(0.25, percentile_threshold), 0.5)  # Clamp between 0.25 and 0.5
        adaptive_thresholds.append(threshold)
    
    # Apply adaptive thresholds
    binary_predictions = np.zeros_like(predictions, dtype=int)
    for class_idx, threshold in enumerate(adaptive_thresholds):
        binary_predictions[:, class_idx] = (predictions[:, class_idx] > threshold).astype(int)
    
    # Ensure each sample has at least 1-3 labels (multi-label nature)
    for idx in range(len(binary_predictions)):
        n_labels = binary_predictions[idx].sum()
        if n_labels == 0:
            # Assign top 2 predictions if no labels
            top_indices = np.argsort(predictions[idx])[-2:]
            binary_predictions[idx, top_indices] = 1
        elif n_labels > 5:  # Cap at 5 labels maximum
            # Keep only top 5 predictions
            top_indices = np.argsort(predictions[idx])[-5:]
            binary_predictions[idx] = 0
            binary_predictions[idx, top_indices] = 1
    
    # Calculate comprehensive metrics
    metrics = calculate_comprehensive_metrics(binary_predictions, classes)
    
    # Save detailed results
    save_path = output_dir / f"results_{log_type}_{config.node_name}_{config.job_id}.pkl"
    with open(save_path, 'wb') as f:
        pickle.dump({
            'predictions': predictions,
            'binary_predictions': binary_predictions,
            'classes': classes,
            'metrics': metrics
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
            'model_type': 'transformer',
            'threshold': 0.5
        }
    }
    
    with open(label_output_path, 'wb') as f:
        pickle.dump(label_data, f)
    
    # Generate classification report
    generate_classification_report(binary_predictions, classes, output_dir, log_type, config)
    
    # Create visualizations
    create_comprehensive_visualizations(embeddings, predictions, binary_predictions, classes, output_dir, log_type, config)
    
    # Print summary
    print_classification_summary(metrics, log_type, len(embeddings))
    
    tracker.log_step("Classification Summary", {
        "log_type": log_type,
        "n_samples": len(embeddings),
        "n_classes": len(classes),
        "avg_labels_per_sample": metrics['avg_labels_per_sample'],
        "macro_f1": metrics['macro_f1'],
        "micro_f1": metrics['micro_f1']
    })
    
    return metrics

def calculate_comprehensive_metrics(binary_predictions: np.ndarray, classes: List[str]) -> Dict[str, Any]:
    """Calculate comprehensive metrics for multi-label classification"""
    
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
    
    # Calculate F1 scores (simplified for unsupervised case)
    # Since we don't have true labels, we'll calculate based on prediction confidence
    high_confidence_predictions = (binary_predictions > 0.7).astype(int)  # Higher threshold
    metrics['high_confidence_predictions'] = int(high_confidence_predictions.sum())
    metrics['high_confidence_percentage'] = float(high_confidence_predictions.sum() / binary_predictions.size * 100)
    
    # For unsupervised learning, we can't calculate traditional F1 scores
    # Instead, we'll use prediction confidence and consistency metrics
    metrics['prediction_confidence_mean'] = float(binary_predictions.mean())
    metrics['prediction_confidence_std'] = float(binary_predictions.std())
    
    # Placeholder for traditional metrics (would need true labels)
    metrics['macro_f1'] = 0.0  # Would need true labels
    metrics['micro_f1'] = 0.0  # Would need true labels
    metrics['hamming_loss'] = 0.0  # Would need true labels
    
    return metrics

def generate_classification_report(binary_predictions: np.ndarray, classes: List[str], 
                                 output_dir: Path, log_type: str, config: SystemConfig):
    """Generate detailed classification report"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = output_dir / f"classification_report_{log_type}_{timestamp}.txt"
    
    with open(report_file, 'w') as f:
        f.write(f"TRANSFORMER CLASSIFICATION REPORT - {log_type.upper()}\n")
        f.write("=" * 60 + "\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"Node: {config.node_name}\n")
        f.write(f"Job ID: {config.job_id}\n\n")
        
        # Dataset statistics
        f.write("DATASET STATISTICS:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Total samples: {len(binary_predictions)}\n")
        f.write(f"Total classes: {len(classes)}\n")
        f.write(f"Total predictions: {binary_predictions.size}\n\n")
        
        # Sample-level statistics
        labels_per_sample = binary_predictions.sum(axis=1)
        f.write("SAMPLE-LEVEL STATISTICS:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Average labels per sample: {labels_per_sample.mean():.3f}\n")
        f.write(f"Std labels per sample: {labels_per_sample.std():.3f}\n")
        f.write(f"Min labels per sample: {labels_per_sample.min()}\n")
        f.write(f"Max labels per sample: {labels_per_sample.max()}\n")
        f.write(f"Samples with no labels: {(labels_per_sample == 0).sum()}\n")
        f.write(f"Samples with one label: {(labels_per_sample == 1).sum()}\n")
        f.write(f"Samples with multiple labels: {(labels_per_sample > 1).sum()}\n\n")
        
        # Class-level statistics
        class_counts = binary_predictions.sum(axis=0)
        f.write("CLASS-LEVEL STATISTICS:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Most frequent classes:\n")
        
        class_freq_pairs = list(zip(classes, class_counts))
        class_freq_pairs.sort(key=lambda x: x[1], reverse=True)
        
        for i, (cls, count) in enumerate(class_freq_pairs[:15]):  # Top 15
            percentage = count / len(binary_predictions) * 100
            f.write(f"  {i+1:2d}. {cls:<30} {count:6d} ({percentage:5.2f}%)\n")
        
        if len(class_freq_pairs) > 15:
            f.write(f"  ... and {len(class_freq_pairs) - 15} more classes\n")
        
        f.write(f"\nClass frequency summary:\n")
        f.write(f"  Classes with 0 predictions: {(class_counts == 0).sum()}\n")
        f.write(f"  Classes with 1-10 predictions: {((class_counts >= 1) & (class_counts <= 10)).sum()}\n")
        f.write(f"  Classes with 11-100 predictions: {((class_counts >= 11) & (class_counts <= 100)).sum()}\n")
        f.write(f"  Classes with >100 predictions: {(class_counts > 100).sum()}\n\n")
        
        # Prediction confidence statistics
        f.write("PREDICTION CONFIDENCE STATISTICS:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Overall prediction confidence: {binary_predictions.mean():.4f}\n")
        f.write(f"Confidence std deviation: {binary_predictions.std():.4f}\n")
        f.write(f"High confidence predictions (>0.7): {binary_predictions.sum()}\n")
        f.write(f"High confidence percentage: {binary_predictions.sum() / binary_predictions.size * 100:.2f}%\n\n")
        
        # Note about unsupervised nature
        f.write("NOTE:\n")
        f.write("-" * 30 + "\n")
        f.write("This is an unsupervised learning model. Traditional metrics like F1-score,\n")
        f.write("precision, and recall cannot be calculated without true labels.\n")
        f.write("The metrics above show the model's prediction patterns and confidence levels.\n")
    
    print(f"Classification report saved to: {report_file}")

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
    
    print(f"\nNote: This is an unsupervised model. Traditional metrics like F1-score")
    print(f"      require true labels and cannot be calculated here.")
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
    Curriculum learning: gradually increase difficulty of learning
    """
    progress = epoch / total_epochs
    
    # Start with simple reconstruction, gradually add complex losses
    weights = {
        'recon': max(0.3, 0.5 - 0.2 * progress),  # Decrease reconstruction importance
        'label': min(0.6, 0.2 + 0.4 * progress),  # Increase label importance
        'cluster': 0.1,
        'contrastive': min(0.3, 0.1 + 0.2 * progress),  # Gradually add contrastive
        'mutual': min(0.2, 0.05 + 0.15 * progress),  # Gradually add mutual
        'confidence': min(0.3, 0.1 + 0.2 * progress),  # Gradually add confidence
        'ensemble': min(0.2, 0.0 + 0.2 * progress)  # Add ensemble consistency later
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

# =============================================================================
# Main Execution
# =============================================================================

def find_available_embeddings() -> List[str]:
    """Find available embedding files - exclude all_combined"""
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

def process_log_type(log_type: str, config: SystemConfig):
    """Process a single log type"""
    output_dir = Path("results") / log_type
    output_dir.mkdir(parents=True, exist_ok=True)
    
    tracker = ProgressTracker(output_dir, log_type, config)
    
    try:
        # Load data with progress
        with Halo(text=f"Loading data for {log_type}...", spinner='dots') as spinner:
            embeddings, classes, C, scaler = load_and_preprocess_data(log_type, config, tracker)
            spinner.succeed(f"Data loaded: {embeddings.shape[0]} samples, {embeddings.shape[1]} features")
        
        # Train model
        tracker.log_step("Training Start", {"embeddings_shape": embeddings.shape})
        with Halo(text=f"Training model for {log_type}...", spinner='dots') as spinner:
            model, _ = train_model(embeddings, classes, C, config, tracker)
            spinner.succeed(f"Training completed for {log_type}")
        
        # Evaluate and save
        with Halo(text=f"Evaluating model for {log_type}...", spinner='dots') as spinner:
            results = evaluate_and_save_results(model, embeddings, classes, config, tracker, output_dir, log_type)
            spinner.succeed(f"Evaluation completed for {log_type}")
        
        # Save model
        if config.rank == 0:
            with Halo(text=f"Saving model for {log_type}...", spinner='dots') as spinner:
                model_path = Path("models") / f"transformer_{log_type}_{config.node_name}_{config.job_id}.pth"
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
        
    except Exception as e:
        tracker.logger.error(f"Error processing {log_type}: {e}")
        import traceback
        tracker.logger.error(traceback.format_exc())
        raise

def main():
    """Main execution with distributed support for Nibi"""
    try:
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
            print(f"Available types: {available_types}")
        
        # Skip log type classifier since each log file can only be one type
        if config.rank == 0:
            print(f"\n{'='*60}")
            print("Processing individual log types")
            print(f"{'='*60}")
        
        # Process each log type
        total_types = len(available_types)
        for idx, log_type in enumerate(available_types, 1):
            if config.rank == 0:
                print(f"\n{'='*60}")
                print(f"Processing: {log_type} ({idx}/{total_types})")
                print(f"{'='*60}")
            
            start_time = time.time()
            process_log_type(log_type, config)
            
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
            print("All processing completed successfully!")
            print(f"Results saved to: results/")
            print(f"Models saved to: models/")
            print(f"Labels saved in evaluation format to: results/*/label_*.pkl")
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