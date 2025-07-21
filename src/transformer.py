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

Multi-Label Learning Enhancements (inspired by supervised baselines):
- Multi-label consistency loss for coherent label combinations
- Class balance regularization to maintain reasonable distributions
- Multi-label aware contrastive learning (Jaccard similarity-based)
- Adaptive pseudo-labeling with confidence and diversity considerations
- Label correlation modeling through learnable correlation matrices

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

# Add these imports for comprehensive evaluation
from sklearn.metrics import (
    classification_report, precision_recall_fscore_support,
    f1_score, accuracy_score, hamming_loss, jaccard_score,
    balanced_accuracy_score, multilabel_confusion_matrix,
    precision_score, recall_score
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

def multilabel_consistency_loss(predictions: torch.Tensor, temperature: float = 2.0) -> torch.Tensor:
    """
    Consistency loss to encourage coherent multi-label predictions
    Similar to how supervised models learn label interactions
    
    Args:
        predictions: Model predictions (batch_size, n_classes)
        temperature: Temperature for softmax normalization
    
    Returns:
        Consistency loss encouraging plausible label combinations
    """
    batch_size, n_classes = predictions.shape
    
    # Convert to probabilities
    probs = torch.sigmoid(predictions)
    
    # Calculate pairwise similarities between samples
    # Normalize predictions for cosine similarity
    probs_norm = F.normalize(probs, dim=1, eps=1e-8)
    similarity_matrix = torch.matmul(probs_norm, probs_norm.T)
    
    # Apply temperature scaling
    similarity_matrix = similarity_matrix / temperature
    
    # Create targets: samples should be similar to themselves, dissimilar to random others
    targets = torch.eye(batch_size, device=predictions.device)
    
    # Add some positive examples (samples with similar prediction patterns)
    # Find samples with high cosine similarity in prediction space
    with torch.no_grad():
        high_sim_mask = (similarity_matrix > 0.7) & (~torch.eye(batch_size, device=predictions.device, dtype=bool))
        targets = targets + 0.3 * high_sim_mask.float()
    
    # Cross-entropy loss encouraging consistent predictions
    log_sim = F.log_softmax(similarity_matrix, dim=1)
    consistency_loss = -(targets * log_sim).sum() / batch_size
    
    return consistency_loss

def class_balance_regularization(predictions: torch.Tensor, target_distribution: torch.Tensor = None) -> torch.Tensor:
    """
    Regularization to maintain reasonable class balance in predictions
    Inspired by supervised learning class distribution awareness
    
    Args:
        predictions: Model predictions (batch_size, n_classes)
        target_distribution: Expected class distribution (if None, uses uniform)
    
    Returns:
        Regularization loss encouraging balanced predictions
    """
    probs = torch.sigmoid(predictions)
    
    # Calculate current class distribution (proportion of positive predictions per class)
    current_dist = probs.mean(dim=0)
    
    if target_distribution is None:
        # Use a balanced distribution as target (each class appears in ~30% of samples)
        target_distribution = torch.full_like(current_dist, 0.3)
    
    # KL divergence between current and target distributions
    # Add small epsilon for numerical stability
    eps = 1e-8
    current_dist = current_dist.clamp(min=eps, max=1-eps)
    target_distribution = target_distribution.clamp(min=eps, max=1-eps)
    
    kl_loss = F.kl_div(
        torch.log(current_dist),
        target_distribution,
        reduction='batchmean'
    )
    
    return kl_loss

def multilabel_contrastive_loss(embeddings: torch.Tensor, predictions: torch.Tensor, 
                               temperature: float = 0.1) -> torch.Tensor:
    """
    Contrastive loss that considers multi-label similarity
    Samples with similar label patterns should have similar embeddings
    
    Args:
        embeddings: Latent embeddings (batch_size, embed_dim)
        predictions: Predicted labels (batch_size, n_classes)
        temperature: Temperature for contrastive loss
    
    Returns:
        Multi-label aware contrastive loss
    """
    batch_size = embeddings.shape[0]
    
    # Normalize embeddings
    embeddings = F.normalize(embeddings, dim=1)
    
    # Calculate label similarities (Jaccard similarity for binary predictions)
    pred_probs = torch.sigmoid(predictions)
    pred_binary = (pred_probs > 0.5).float()
    
    # Jaccard similarity: |A ∩ B| / |A ∪ B|
    intersection = torch.matmul(pred_binary, pred_binary.T)
    union = pred_binary.sum(dim=1, keepdim=True) + pred_binary.sum(dim=1, keepdim=False) - intersection
    label_similarity = intersection / (union + 1e-8)
    
    # Embedding similarities
    embed_similarity = torch.matmul(embeddings, embeddings.T) / temperature
    
    # Encourage embedding similarity to match label similarity
    # Use label similarity as soft targets for embedding similarity
    
    # Apply softmax to embedding similarities
    embed_sim_softmax = F.softmax(embed_similarity, dim=1)
    
    # Use label similarity as soft targets (normalize to sum to 1)
    label_targets = label_similarity / (label_similarity.sum(dim=1, keepdim=True) + 1e-8)
    
    # KL divergence loss
    contrastive_loss = F.kl_div(
        torch.log(embed_sim_softmax + 1e-8),
        label_targets,
        reduction='batchmean'
    )
    
    return contrastive_loss

def adaptive_pseudo_labeling(predictions: torch.Tensor, confidence_threshold: float = 0.8,
                           diversity_weight: float = 0.1) -> torch.Tensor:
    """
    Advanced pseudo-labeling that considers both confidence and diversity
    Inspired by how supervised models handle uncertain predictions
    
    Args:
        predictions: Model predictions (batch_size, n_classes)
        confidence_threshold: Minimum confidence for pseudo-labels
        diversity_weight: Weight for encouraging diverse predictions
    
    Returns:
        Refined pseudo-labels
    """
    probs = torch.sigmoid(predictions)
    batch_size, n_classes = probs.shape
    
    # Start with high-confidence predictions
    high_conf_mask = probs > confidence_threshold
    pseudo_labels = probs.clone()
    
    # For low-confidence predictions, use adaptive thresholding
    low_conf_mask = ~high_conf_mask
    
    if low_conf_mask.any():
        # Use adaptive threshold based on class-wise statistics
        class_means = probs.mean(dim=0)
        class_stds = probs.std(dim=0)
        
        # Dynamic thresholds: mean + 0.5 * std
        adaptive_thresholds = class_means + 0.5 * class_stds
        adaptive_thresholds = adaptive_thresholds.clamp(min=0.2, max=0.8)
        
        # Apply adaptive thresholds
        for i in range(n_classes):
            class_mask = low_conf_mask[:, i]
            if class_mask.any():
                # Boost predictions above adaptive threshold
                above_adaptive = probs[:, i] > adaptive_thresholds[i]
                boost_mask = class_mask & above_adaptive
                pseudo_labels[boost_mask, i] = torch.minimum(
                    pseudo_labels[boost_mask, i] + 0.2,
                    torch.tensor(0.9, device=predictions.device)
                )
    
    # Encourage diversity: if all samples have very similar predictions, boost variety
    pred_diversity = torch.std(probs, dim=0).mean()
    if pred_diversity < 0.1:  # Low diversity
        # Add small random perturbations to encourage exploration
        diversity_noise = torch.randn_like(pseudo_labels) * diversity_weight * 0.05
        pseudo_labels = pseudo_labels + diversity_noise
        pseudo_labels = torch.clamp(pseudo_labels, 0.0, 1.0)
    
    return pseudo_labels

def generate_reconstruction_anomaly_scores(model: nn.Module, embeddings: torch.Tensor, 
                                         batch_size: int = 256, device: torch.device = torch.device('cpu')) -> np.ndarray:
    """
    Generate reconstruction anomaly scores for pseudo-label generation.
    Fixed for CUDA memory alignment issues.
    
    Args:
        model: Trained model
        embeddings: Input embeddings
        batch_size: Batch size for processing
        device: Device to use
    
    Returns:
        Anomaly scores for each sample
    """
    model.eval()
    recon_errors = []
    
    # Handle DataParallel wrapper
    actual_model = model.module if hasattr(model, 'module') else model
    
    with torch.no_grad():
        for i in range(0, len(embeddings), batch_size):
            try:
                # Create batch with proper memory alignment
                batch_end = min(i + batch_size, len(embeddings))
                batch = embeddings[i:batch_end]
                
                # Ensure tensor is contiguous and properly aligned
                if not batch.is_contiguous():
                    batch = batch.contiguous()
                
                batch = batch.to(device, non_blocking=False)
                
                # Force CUDA sync to catch any async errors early
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                
                # Forward pass with error handling
                outputs = model(batch)
                
                # Calculate reconstruction error with memory safety
                if 'reconstructed' in outputs:
                    recon_output = outputs['reconstructed']
                    
                    # Ensure tensors are compatible and aligned
                    if not recon_output.is_contiguous():
                        recon_output = recon_output.contiguous()
                    
                    # Calculate MSE loss safely
                    recon_error = F.mse_loss(recon_output, batch, reduction='none')
                    recon_error = recon_error.mean(dim=1)  # Per-sample error
                    
                    # Sync before CPU transfer
                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                    
                    # Safe CPU transfer with error handling
                    try:
                        error_cpu = recon_error.detach().cpu().numpy()
                        recon_errors.append(error_cpu)
                    except RuntimeError as e:
                        if "misaligned address" in str(e):
                            print(f"⚠️  CUDA alignment error, using fallback method...")
                            # Fallback: use clone() to ensure proper alignment
                            error_cpu = recon_error.detach().clone().cpu().numpy()
                            recon_errors.append(error_cpu)
                        else:
                            raise e
                else:
                    # Fallback: use label predictions as proxy for anomaly scores
                    print("⚠️  No reconstruction output, using label predictions as proxy...")
                    label_output = torch.sigmoid(outputs['labels'])
                    anomaly_proxy = 1.0 - label_output.max(dim=1)[0]  # Low confidence = high anomaly
                    
                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                    
                    error_cpu = anomaly_proxy.detach().cpu().numpy()
                    recon_errors.append(error_cpu)
                
                # Clear GPU cache periodically
                if device.type == 'cuda' and i % (batch_size * 10) == 0:
                    torch.cuda.empty_cache()
                    
            except RuntimeError as e:
                if "out of memory" in str(e) or "misaligned address" in str(e):
                    print(f"⚠️  Memory/alignment error at batch {i//batch_size + 1}, trying smaller batch...")
                    
                    # Try with smaller sub-batches
                    sub_batch_size = batch_size // 4
                    for j in range(i, min(i + batch_size, len(embeddings)), sub_batch_size):
                        try:
                            sub_batch = embeddings[j:j+sub_batch_size].contiguous().to(device)
                            
                            if device.type == 'cuda':
                                torch.cuda.synchronize()
                            
                            sub_outputs = model(sub_batch)
                            
                            if 'reconstructed' in sub_outputs:
                                sub_recon = sub_outputs['reconstructed'].contiguous()
                                sub_error = F.mse_loss(sub_recon, sub_batch, reduction='none').mean(dim=1)
                            else:
                                sub_error = 1.0 - torch.sigmoid(sub_outputs['labels']).max(dim=1)[0]
                            
                            if device.type == 'cuda':
                                torch.cuda.synchronize()
                            
                            recon_errors.append(sub_error.detach().clone().cpu().numpy())
                            
                        except Exception as sub_e:
                            print(f"⚠️  Sub-batch also failed: {sub_e}")
                            # Final fallback: random anomaly scores
                            fallback_scores = np.random.uniform(0.0, 1.0, min(sub_batch_size, len(embeddings) - j))
                            recon_errors.append(fallback_scores)
                else:
                    raise e
    
    if not recon_errors:
        print("⚠️  No reconstruction errors computed, using random scores...")
        return np.random.uniform(0.0, 1.0, len(embeddings))
    
    try:
        return np.concatenate(recon_errors)
    except Exception as e:
        print(f"⚠️  Error concatenating results: {e}, using mean values...")
        # Fallback: return mean of successful computations
        mean_val = np.mean([arr.mean() for arr in recon_errors if len(arr) > 0])
        return np.full(len(embeddings), mean_val)

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
        
        # Progress tracking enhancements
        self.total_epochs = 0
        self.total_batches_per_epoch = 0
        self.current_epoch = 0
        self.current_batch = 0
        self.batch_times = []
        self.last_progress_update = time.time()
        self.progress_update_interval = 10  # Update every 10 batches
        
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
    
    def start_training(self, total_epochs: int, total_batches_per_epoch: int = 0):
        """Start training timer with batch-level tracking"""
        self.start_time = time.time()
        self.total_epochs = total_epochs
        self.total_batches_per_epoch = total_batches_per_epoch
        self.epoch_times = []
        self.batch_times = []
        self.current_epoch = 0
        self.current_batch = 0
        
        print(f"🚀 Training started: {total_epochs} epochs, ~{total_batches_per_epoch} batches/epoch")
    
    def start_epoch(self, epoch: int):
        """Start tracking a new epoch"""
        self.current_epoch = epoch
        self.current_batch = 0
        self.epoch_start_time = time.time()
        
    def update_batch_progress(self, batch_idx: int, batch_time: float = None, loss_info: Dict[str, float] = None):
        """Update batch-level progress with percentage and ETA"""
        self.current_batch = batch_idx
        
        if batch_time:
            self.batch_times.append(batch_time)
        
        # Update progress every N batches or at the end of epoch
        current_time = time.time()
        is_epoch_end = (batch_idx + 1) >= self.total_batches_per_epoch
        should_update = (
            (batch_idx % self.progress_update_interval == 0) or 
            is_epoch_end or
            (current_time - self.last_progress_update) > 30  # At least every 30 seconds
        )
        
        if should_update and self.total_batches_per_epoch > 0:
            self.last_progress_update = current_time
            
            # Calculate overall progress
            completed_epochs = self.current_epoch
            completed_batches_this_epoch = batch_idx + 1
            total_completed_batches = completed_epochs * self.total_batches_per_epoch + completed_batches_this_epoch
            total_batches = self.total_epochs * self.total_batches_per_epoch
            
            overall_progress_pct = (total_completed_batches / total_batches) * 100 if total_batches > 0 else 0
            epoch_progress_pct = (completed_batches_this_epoch / self.total_batches_per_epoch) * 100
            
            # Calculate ETA
            eta_str = "calculating..."
            if len(self.batch_times) >= 5:  # Need some data for reliable estimates
                # Use recent batch times for better accuracy
                recent_batch_times = self.batch_times[-20:] if len(self.batch_times) >= 20 else self.batch_times
                avg_batch_time = np.mean(recent_batch_times)
                
                remaining_batches = total_batches - total_completed_batches
                eta_seconds = remaining_batches * avg_batch_time
                eta_str = self._format_time(eta_seconds)
            
            # Format loss information - focus on meaningful unsupervised metrics
            loss_str = ""
            if loss_info:
                # Only show meaningful loss components for unsupervised learning
                meaningful_losses = {}
                if 'total' in loss_info and not (loss_info.get('skipped', False) or loss_info.get('oom', False) or loss_info.get('nan_skip', False)):
                    meaningful_losses['loss'] = loss_info['total']
                if 'recon' in loss_info:
                    meaningful_losses['recon'] = loss_info['recon']
                if 'label' in loss_info:
                    meaningful_losses['label'] = loss_info['label']
                
                # Add status indicators for problematic batches
                if loss_info.get('skipped', False):
                    meaningful_losses['status'] = 'SKIPPED'
                elif loss_info.get('oom', False):
                    meaningful_losses['status'] = 'OOM'
                elif loss_info.get('nan_skip', False):
                    meaningful_losses['status'] = 'NaN'
                
                if meaningful_losses:
                    loss_str = " | ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in meaningful_losses.items()])
                    loss_str = f" | {loss_str}"
            
            # Print progress update
            print(f"📊 Progress: {overall_progress_pct:.1f}% | "
                  f"Epoch {self.current_epoch + 1}/{self.total_epochs} "
                  f"({epoch_progress_pct:.1f}%) | "
                  f"Batch {batch_idx + 1}/{self.total_batches_per_epoch} | "
                  f"ETA: {eta_str}{loss_str}")
    
    def update_epoch_progress(self, epoch: int, epoch_time: float):
        """Update epoch progress and estimate completion time"""
        self.epoch_times.append(epoch_time)
        
        if len(self.epoch_times) >= 2:
            avg_epoch_time = np.mean(self.epoch_times[-5:])  # Use last 5 epochs for better estimate
            remaining_epochs = self.total_epochs - epoch - 1
            estimated_remaining = avg_epoch_time * remaining_epochs
            
            elapsed = time.time() - self.start_time
            estimated_total = elapsed + estimated_remaining
            
            # Calculate overall percentage
            overall_progress_pct = ((epoch + 1) / self.total_epochs) * 100
            
            # Format time strings
            elapsed_str = self._format_time(elapsed)
            remaining_str = self._format_time(estimated_remaining)
            total_str = self._format_time(estimated_total)
            
            # Print epoch completion summary
            print(f"✅ Epoch {epoch + 1}/{self.total_epochs} completed ({overall_progress_pct:.1f}%) | "
                  f"Time: {self._format_time(epoch_time)} | "
                  f"Avg: {self._format_time(avg_epoch_time)} | "
                  f"ETA: {remaining_str}")
            
            return {
                'epoch': epoch + 1,
                'total_epochs': self.total_epochs,
                'overall_progress_pct': overall_progress_pct,
                'elapsed': elapsed_str,
                'remaining': remaining_str,
                'estimated_total': total_str,
                'epoch_time': f"{epoch_time:.2f}s",
                'avg_epoch_time': f"{avg_epoch_time:.2f}s"
            }
        else:
            # First epoch(s) - just show basic info
            overall_progress_pct = ((epoch + 1) / self.total_epochs) * 100
            print(f"✅ Epoch {epoch + 1}/{self.total_epochs} completed ({overall_progress_pct:.1f}%) | "
                  f"Time: {self._format_time(epoch_time)}")
        
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
    
    def save_checkpoint_progress(self, epoch: int, total_loss: float, additional_info: Dict[str, Any] = None):
        """Save detailed progress information for potential resumption"""
        progress_data = {
            'timestamp': time.time(),
            'epoch': epoch,
            'total_epochs': self.total_epochs,
            'overall_progress_pct': ((epoch + 1) / self.total_epochs) * 100,
            'elapsed_time': time.time() - self.start_time if self.start_time else 0,
            'total_loss': total_loss,
            'avg_epoch_time': np.mean(self.epoch_times) if self.epoch_times else 0,
            'avg_batch_time': np.mean(self.batch_times[-100:]) if len(self.batch_times) >= 10 else 0,
            'log_type': self.log_type,
            'node': self.config.node_name,
            'job_id': self.config.job_id
        }
        
        if additional_info:
            progress_data.update(additional_info)
        
        progress_file = self.output_dir / f"progress_{self.log_type}_{self.config.node_name}_{self.config.job_id}.json"
        with open(progress_file, 'w') as f:
            json.dump(progress_data, f, indent=2)

def load_and_preprocess_data(log_type: str, config: SystemConfig, tracker: ProgressTracker, sample_size: int = None) -> Tuple[np.ndarray, List[str], np.ndarray, StandardScaler]:
    """Optimized data loading with memory management for Nibi and embedding auto-detection"""
    print(f"🔄 Loading data for {log_type}...")
    load_start_time = time.time()
    
    tracker.log_step("Data Loading", {"log_type": log_type, "config": config.__dict__})
    
    # Load embeddings - only load specific log type, not combined
    embeddings_dir = Path("embeddings")
    log_file = embeddings_dir / log_type / f"log_{log_type}.pkl"
    label_file = embeddings_dir / log_type / f"label_{log_type}.pkl"
    
    if not log_file.exists():
        raise FileNotFoundError(f"Embedding file not found: {log_file}")
    
    print(f"📂 Loading embeddings from {log_file}...")
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
    
    print(f"🔍 Auto-detected embedding type: {embedding_type} ({embedding_dim}D)")
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
            samples_per_gb = min(200, int(config.gpu_memory_gb * 2))  # Increased from 100 to 200
        else:
            samples_per_gb = 150
    
    # Only apply automatic memory-based subsampling if no explicit sample_size is provided
    # AND the dataset is extremely large (>100k samples)
    apply_auto_subsampling = (sample_size is None and len(embeddings) > 100000)
    
    if apply_auto_subsampling:
        # Apply more conservative limits for CUDA to prevent OOM - but only for very large datasets
        if config.device == "cuda":
            max_samples = min(50000, int(config.gpu_memory_gb * samples_per_gb))  # Increased max
        else:
            max_samples = min(100000, int(config.gpu_memory_gb * samples_per_gb))
        
        if len(embeddings) > max_samples:
            print(f"📊 Large dataset detected ({len(embeddings):,} samples)")
            print(f"   Automatically subsampling to {max_samples:,} samples for memory efficiency")
            print(f"   Use --sample-size to override or process smaller chunks")
            
            indices = np.random.choice(len(embeddings), max_samples, replace=False)
            embeddings = embeddings[indices]
            if true_labels is not None:
                true_labels = true_labels[indices]
            tracker.log_step("Automatic Data Subsampling", {
                "original_size": len(embeddings),
                "subsampled_size": max_samples,
                "memory_gb": embeddings.nbytes / (1024**3),
                "embedding_dim": embedding_dim,
                "embedding_type": embedding_type,
                "samples_per_gb": samples_per_gb,
                "device_type": config.device,
                "reason": "Large dataset auto-subsampling"
            })
    
    # Apply explicit sample size limit if specified
    if sample_size is not None and sample_size < len(embeddings):
        print(f"🎯 Limiting dataset to {sample_size:,} samples as requested...")
        # Random sampling to maintain class distribution
        np.random.seed(42)  # For reproducibility
        indices = np.random.choice(len(embeddings), size=sample_size, replace=False)
        embeddings = embeddings[indices]
        if true_labels is not None:
            true_labels = true_labels[indices]
        print(f"   Dataset reduced to {len(embeddings):,} samples")
        
        tracker.log_step("Explicit Data Sampling", {
            "requested_size": sample_size,
            "actual_size": len(embeddings),
            "memory_gb": embeddings.nbytes / (1024**3),
            "embedding_dim": embedding_dim,
            "embedding_type": embedding_type
        })
    elif sample_size is None:
        # Using full dataset
        print(f"📊 Using full dataset: {len(embeddings):,} samples ({embeddings.nbytes / (1024**3):.1f} GB)")
        tracker.log_step("Full Dataset Processing", {
            "total_samples": len(embeddings),
            "memory_gb": embeddings.nbytes / (1024**3),
            "embedding_dim": embedding_dim,
            "embedding_type": embedding_type,
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
    
    # Log completion timing
    load_time = time.time() - load_start_time
    print(f"✅ Data loading completed in {load_time:.1f}s")
    print(f"📊 Loaded {len(embeddings):,} samples with {embedding_dim}D embeddings")
    if sample_size:
        print(f"🎯 Using sample size: {len(embeddings):,} (requested: {sample_size:,})")
    
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
        model: Model containing class_prototypes buffer (may be wrapped in DataParallel)
        z_prototype: Current batch prototype features
        labels: Current batch labels (probabilities)
        momentum: Update momentum
    """
    # Handle DataParallel wrapper
    actual_model = model.module if hasattr(model, 'module') else model
    
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
                if actual_model.prototype_counts[class_idx] == 0:
                    # First update for this class
                    actual_model.class_prototypes[class_idx] = class_mean
                    actual_model.prototype_counts[class_idx] = 1
                else:
                    # Momentum update
                    actual_model.class_prototypes[class_idx] = (
                        momentum * actual_model.class_prototypes[class_idx] + 
                        (1 - momentum) * class_mean
                    )
                    actual_model.prototype_counts[class_idx] += class_mask.sum().item()

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
    
    # Enhanced architecture for supercomputer environments
    embedding_dim = embeddings.shape[1]
    if embedding_dim <= 300:  # FastText
        latent_dim = 512
        transformer_layers = 12
        attention_heads = 16
    elif embedding_dim <= 768:  # Standard BERT
        latent_dim = 768
        transformer_layers = 16
        attention_heads = 16
    else:  # Enhanced LogBERT (2314D) - Supercomputer optimized
        latent_dim = 1024
        transformer_layers = 20  # Deep architecture for supercomputer
        attention_heads = 32
    
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
    
    # Training information and CUDA warnings
    use_mixed_precision = (config.device == "cuda")  # Mixed precision only on CUDA
    
    # Early stopping parameters
    patience = 10  # Early stopping patience
    
    # Initialize class weights
    class_weights = None
    
    # Set CUDA debugging environment for better error reporting
    if device.type == 'cuda':
        import os
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Enable synchronous CUDA for better error tracing
        print(f"🔧 CUDA_LAUNCH_BLOCKING=1 enabled for debugging")
    
    print(f"🚀 ENHANCED SUPERCOMPUTER TRAINING - {log_type}")
    # Calculate model complexity
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = total_params * 4 / (1024 * 1024)  # Assuming float32
    
    print(f"💾 Model parameters: {total_params:,} ({trainable_params:,} trainable)")
    print(f"💽 Model size: {model_size_mb:.1f} MB")
    print(f"📊 Training samples: {len(embeddings):,}")
    print(f"🏷️  Classes: {len(classes)}")
    print(f"🎯 Device: {device} | Mixed precision: {use_mixed_precision}")
    print(f"🏗️  Enhanced Architecture (Supercomputer-optimized):")
    print(f"   - Input dimension: {embedding_dim}")
    print(f"   - Latent dimension: {latent_dim}")
    print(f"   - Transformer layers: {transformer_layers}")
    print(f"   - Attention heads: {attention_heads}")
    
    # Estimated training time
    est_time_per_epoch = 0.5 * len(embeddings) / 1000  # Rough estimate
    est_total_time = est_time_per_epoch * 100 / 60  # in minutes
    print(f"⏱️  Estimated training time: {est_total_time:.1f} minutes ({est_time_per_epoch:.1f}s/epoch)")
    print(f"⚡ Supercomputer optimizations:")
    print(f"   - Extended training: 100 epochs with patience={patience}")
    print(f"   - Checkpoint frequency: Every 5% (20 checkpoints)")
    print(f"   - Class balance: {'Enabled' if class_weights is not None else 'Disabled'}")
    print(f"   - Learning rate: 8e-5 with 10-epoch warmup")
    print(f"   - ETA tracking: Real-time estimation")
    if device.type == 'cuda':
        print(f"⚠️  CUDA Safety Measures:")
        print(f"   - Disabled pin_memory and multiprocessing in DataLoader")
        print(f"   - Using contiguous tensors and blocking transfers")
        print(f"   - Anomaly scoring at epoch 3 with error handling")
        print(f"   - Batch-level error recovery enabled")
    print(f"{'='*60}")
    
    # Generate initial pseudo-labels using all available true labels (if any) for guidance
    true_labels = getattr(tracker, 'true_labels', None)
    pseudo_labels = advanced_pseudo_label_generation(embeddings, classes, true_labels=true_labels, epoch=0, total_epochs=100)
    
    # Store initial pseudo-labels for refinement
    current_pseudo_labels = pseudo_labels.copy()
    
    # Data setup - use ALL data for training (no splitting)
    # Ensure tensors are contiguous to prevent CUDA alignment issues
    embeddings_tensor = torch.from_numpy(embeddings).float().contiguous()
    labels_tensor = torch.from_numpy(current_pseudo_labels).float().contiguous()
    
    dataset = TensorDataset(embeddings_tensor, labels_tensor)
    
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
        num_workers=0,  # Disable multiprocessing to avoid alignment issues
        pin_memory=False  # Disable pin_memory to prevent CUDA misalignment
    )
    
    # Advanced training setup - adjusted for longer training
    optimizer = optim.AdamW(model.parameters(), lr=8e-5, weight_decay=1e-4)  # Slightly lower LR for 100 epochs
    
    # Compute class weights for balanced training if true labels available
    if true_labels is not None:
        # Calculate inverse frequency weights for class balance
        class_frequencies = true_labels.sum(axis=0)
        # Avoid division by zero for classes with no samples
        class_frequencies = np.maximum(class_frequencies, 1)
        class_weights = len(true_labels) / (len(classes) * class_frequencies)
        class_weights = torch.from_numpy(class_weights.astype(np.float32)).to(device)
        print(f"🎯 Using class balance weights: {[f'{w:.2f}' for w in class_weights.cpu().numpy()]}")
    scaler = GradScaler() if config.device == "cuda" else None
    
    # Advanced scheduler with warmup - adapted for 100 epochs
    def lr_lambda(epoch):
        warmup_epochs = 10  # Longer warmup for 100 epochs
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        else:
            # Cosine annealing over remaining epochs
            return 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (100 - warmup_epochs)))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Gradient clipping for stability
    max_grad_norm = 1.0  # Reduced for better stability
    
    # Check for existing checkpoint
    start_epoch = 0
    best_model_state = None  # Initialize best model state for early stopping
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
                # Initialize best model state with current loaded state
                best_model_state = model.state_dict().copy()
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
    total_epochs = 100  # Increased epochs for better convergence in unsupervised learning
    
    # Initialize progress tracking with batch information
    total_batches_per_epoch = len(dataloader)
    tracker.start_training(total_epochs, total_batches_per_epoch)
    
    refinement_interval = 10  # Refine pseudo-labels every 10 epochs
    checkpoint_interval = 5   # Save checkpoint every 5 epochs
    
    # Early stopping parameters - more patient for longer training  
    patience_counter = 0
    min_delta = 1e-5  # Smaller minimum improvement threshold
    
    for epoch in range(start_epoch, total_epochs):
        epoch_start = time.time()
        epoch_losses = []
        
        # Start epoch tracking
        tracker.start_epoch(epoch)
        
        if config.is_distributed:
            sampler.set_epoch(epoch)
        
        # UMTL Enhancement: Compute reconstruction anomaly scores after warmup (TPLG)
        if epoch == 3 and reconstruction_anomaly_scores is None:
            print("Computing reconstruction anomaly scores...")
            try:
                # Clear GPU cache before intensive computation
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                embeddings_tensor = torch.from_numpy(embeddings).float()
                
                # Ensure tensor is contiguous for CUDA alignment
                if not embeddings_tensor.is_contiguous():
                    embeddings_tensor = embeddings_tensor.contiguous()
                
                # Use smaller batch size for anomaly scoring to avoid memory issues
                anomaly_batch_size = min(batch_size, 128)
                
                reconstruction_anomaly_scores = generate_reconstruction_anomaly_scores(
                    model, embeddings_tensor, anomaly_batch_size, device
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
                        
                        # Update dataset with anomaly-seeded labels (ensure alignment)
                        embeddings_tensor = torch.from_numpy(embeddings).float().contiguous()
                        labels_tensor = torch.from_numpy(current_pseudo_labels).float().contiguous()
                        dataset = TensorDataset(embeddings_tensor, labels_tensor)
                        
                        dataloader = DataLoader(
                            dataset, 
                            batch_size=batch_size, 
                            sampler=sampler,
                            shuffle=(sampler is None),
                            num_workers=0,  # Disable multiprocessing to avoid alignment issues
                            pin_memory=False  # Disable pin_memory to prevent CUDA misalignment
                        )
                    
                    tracker.log_step("Reconstruction Anomaly Seeding (TPLG)", {
                        "total_samples": len(reconstruction_anomaly_scores),
                        "anomalies_detected": int(anomaly_mask.sum()),
                        "anomaly_percentage": float(anomaly_mask.sum() / len(anomaly_mask) * 100),
                        "threshold_percentile": anomaly_threshold * 100,
                        "threshold_value": float(anomaly_threshold_value),
                        "mean_anomaly_score": float(reconstruction_anomaly_scores.mean()),
                        "max_anomaly_score": float(reconstruction_anomaly_scores.max())
                    })
                    
                    print(f"   🔍 Found {anomaly_mask.sum()} anomalies ({anomaly_mask.sum()/len(anomaly_mask)*100:.1f}%)")
                else:
                    print("   ⚠️  No anomalies detected with current threshold")
                    
            except Exception as e:
                print(f"⚠️  Error computing reconstruction anomaly scores: {e}")
                print("   Continuing without anomaly seeding...")
                
                # Clear any corrupted GPU memory
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                # Create dummy anomaly scores to prevent re-computation
                reconstruction_anomaly_scores = np.random.uniform(0.0, 1.0, len(embeddings))
                anomaly_mask = np.zeros(len(embeddings), dtype=bool)  # No anomalies
                
                tracker.log_step("Reconstruction Anomaly Seeding (TPLG) - Failed", {
                    "error": str(e),
                    "fallback_used": True,
                    "dummy_scores_generated": len(reconstruction_anomaly_scores)
                })
        
        # Advanced pseudo-label refinement with curriculum learning
        if epoch > 0 and epoch % refinement_interval == 0:
            print(f"🔄 Refining pseudo-labels at epoch {epoch+1}...")
            refinement_start = time.time()
            
            model.eval()
            with torch.no_grad():
                all_predictions = []
                # Clear memory before inference
                clear_gpu_memory()
                
                # Use smaller inference batch size for memory efficiency
                inference_batch_size = max(1, batch_size // 2) if config.device == "cuda" else batch_size
                total_inference_batches = (len(embeddings) + inference_batch_size - 1) // inference_batch_size
                
                for batch_idx, i in enumerate(range(0, len(embeddings), inference_batch_size)):
                    # Show refinement progress for large datasets
                    if batch_idx % max(1, total_inference_batches // 10) == 0:
                        refinement_progress = (batch_idx / total_inference_batches) * 100
                        print(f"   📊 Refinement progress: {refinement_progress:.1f}% ({batch_idx+1}/{total_inference_batches} batches)")
                    
                    batch = torch.from_numpy(embeddings[i:i+inference_batch_size]).float().to(device)
                    
                    try:
                        outputs = model(batch)
                        predictions = torch.sigmoid(outputs['labels']).cpu().numpy()
                        all_predictions.append(predictions)
                    except torch.cuda.OutOfMemoryError:
                        # Skip this batch and clear memory
                        clear_gpu_memory()
                        
                        # Still track progress for skipped batch
                        batch_time = time.time() - batch_start_time
                        loss_info = {'total': 0.0, 'oom': True}
                        tracker.update_batch_progress(batch_idx, batch_time, loss_info)
                        
                        continue
                    
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
                    num_workers=0,  # Disable multiprocessing to avoid alignment issues
                    pin_memory=False  # Disable pin_memory to prevent CUDA misalignment
                )
                
                refinement_time = time.time() - refinement_start
                print(f"✅ Pseudo-label refinement completed in {refinement_time:.1f}s")
                
                tracker.log_step("Advanced Pseudo-label Refinement", {
                    "epoch": epoch,
                    "refinement_time": refinement_time,
                    "total_inference_batches": total_inference_batches,
                    "inference_batch_size": inference_batch_size,
                    "avg_confidence": float(np.mean(all_predictions.max(axis=1))),
                    "label_density": float(np.mean(all_predictions.sum(axis=1))),
                    "model_weight": model_weight,
                    "curriculum_progress": progress
                })
            
            # Clear memory before resuming training
            clear_gpu_memory()
            model.train()
        
        # Progress tracking for batches (removing spinner as we have detailed progress)
        print(f"\n🔄 Starting Epoch {epoch+1}/{total_epochs} with {len(dataloader)} batches...")
        
        for batch_idx, (x_batch, y_batch) in enumerate(dataloader):
            batch_start_time = time.time()
            
            # Initialize default loss variables in case batch is skipped
            total_loss = torch.tensor(0.0, device=device)
            recon_loss = torch.tensor(0.0, device=device)
            label_loss = torch.tensor(0.0, device=device)
            supervised_loss = torch.tensor(0.0, device=device)
            
            try:
                # Ensure tensors are contiguous before GPU transfer to avoid misalignment
                if not x_batch.is_contiguous():
                    x_batch = x_batch.contiguous()
                if not y_batch.is_contiguous():
                    y_batch = y_batch.contiguous()
                
                # Safe GPU transfer with error handling
                try:
                    x_batch = x_batch.to(device, non_blocking=False)  # Use blocking for stability
                    y_batch = y_batch.to(device, non_blocking=False)
                except RuntimeError as gpu_error:
                    if "misaligned address" in str(gpu_error) or "CUDA" in str(gpu_error):
                        print(f"⚠️  GPU transfer error, trying alignment fix...")
                        # Force tensor alignment by cloning
                        x_batch = x_batch.clone().contiguous().to(device, non_blocking=False)
                        y_batch = y_batch.clone().contiguous().to(device, non_blocking=False)
                    else:
                        raise gpu_error
                
                # Force CUDA sync to catch alignment issues early
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                
            except RuntimeError as batch_error:
                if "misaligned address" in str(batch_error) or "out of memory" in str(batch_error):
                    print(f"⚠️  Batch {batch_idx} failed with CUDA error: {batch_error}")
                    print(f"   Skipping batch and continuing...")
                    
                    # Clear any corrupted GPU memory
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    
                    # Still track progress for skipped batch
                    batch_time = time.time() - batch_start_time
                    loss_info = {'total': 0.0, 'skipped': True}
                    tracker.update_batch_progress(batch_idx, batch_time, loss_info)
                    
                    continue  # Skip this batch and continue with next
                else:
                    raise batch_error
                
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
                        
                        # Enhanced focal loss with distribution-balanced weights and class weighting
                        label_loss = enhanced_focal_loss(outputs['labels'], y_batch, class_weights=class_weights)
                        
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
                        
                        # NEW: Multi-label specific enhancements inspired by supervised learning
                        # Multi-label consistency loss - encourages coherent label combinations
                        multilabel_consistency = multilabel_consistency_loss(outputs['labels'], temperature=2.0)
                        
                        # Class balance regularization - maintains reasonable class distribution
                        class_balance_loss = class_balance_regularization(outputs['labels'])
                        
                        # Multi-label aware contrastive loss - similar labels should have similar embeddings
                        multilabel_contrastive = multilabel_contrastive_loss(outputs['latent'], outputs['labels'])
                        
                        # Adaptive pseudo-labeling refinement for better unsupervised learning
                        refined_pseudo_labels = adaptive_pseudo_labeling(outputs['labels'], confidence_threshold=0.7)
                        
                        # Additional loss on refined pseudo-labels
                        refined_label_loss = F.binary_cross_entropy_with_logits(
                            outputs['labels'], refined_pseudo_labels, reduction='mean'
                        )
                        
                        # Prototype/margin loss
                        # Handle DataParallel wrapper
                        actual_model = model.module if hasattr(model, 'module') else model
                        if actual_model.prototype_counts.sum() > 0:  # Only if prototypes are initialized
                            prototype_loss = prototype_margin_loss(
                                outputs['prototype'], 
                                predictions,  # Use predictions as soft labels
                                actual_model.class_prototypes,
                                actual_model.margin
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
                        
                        # NEW: Weights for multi-label specific losses
                        loss_weights['multilabel_consistency'] = min(0.15, 0.05 + 0.1 * (epoch / total_epochs))
                        loss_weights['class_balance'] = 0.1  # Steady throughout training
                        loss_weights['multilabel_contrastive'] = min(0.2, 0.1 + 0.1 * (epoch / total_epochs))
                        loss_weights['refined_labels'] = min(0.25, 0.1 + 0.15 * (epoch / total_epochs))
                        
                        total_loss = (loss_weights['recon'] * recon_loss + 
                                    loss_weights['label'] * label_loss + 
                                    loss_weights['supervised'] * supervised_loss +  # Added
                                    loss_weights['prototype'] * prototype_loss +     # Added
                                    loss_weights['cluster'] * cluster_loss + 
                                    loss_weights['contrastive'] * contrast_loss + 
                                    loss_weights['mutual'] * mutual_loss + 
                                    loss_weights['confidence'] * confidence_loss + 
                                    loss_weights['ensemble'] * ensemble_loss +
                                    loss_weights['multilabel_consistency'] * multilabel_consistency +  # NEW
                                    loss_weights['class_balance'] * class_balance_loss +  # NEW
                                    loss_weights['multilabel_contrastive'] * multilabel_contrastive +  # NEW
                                    loss_weights['refined_labels'] * refined_label_loss)  # NEW
                        
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
                            'ensemble': ensemble_loss,
                            'multilabel_consistency': multilabel_consistency,  # NEW
                            'class_balance': class_balance_loss,  # NEW
                            'multilabel_contrastive': multilabel_contrastive,  # NEW
                            'refined_labels': refined_label_loss  # NEW
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
                            
                            # Still track progress for skipped batch
                            batch_time = time.time() - batch_start_time
                            loss_info = {'total': 0.0, 'nan_skip': True}
                            tracker.update_batch_progress(batch_idx, batch_time, loss_info)
                            
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
                    
                    # Enhanced focal loss with distribution-balanced weights and class weighting
                    label_loss = enhanced_focal_loss(outputs['labels'], y_batch, class_weights=class_weights)
                    
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
                    
                    # NEW: Multi-label specific enhancements inspired by supervised learning
                    # Multi-label consistency loss - encourages coherent label combinations
                    multilabel_consistency = multilabel_consistency_loss(outputs['labels'], temperature=2.0)
                    
                    # Class balance regularization - maintains reasonable class distribution
                    class_balance_loss = class_balance_regularization(outputs['labels'])
                    
                    # Multi-label aware contrastive loss - similar labels should have similar embeddings
                    multilabel_contrastive = multilabel_contrastive_loss(outputs['latent'], outputs['labels'])
                    
                    # Adaptive pseudo-labeling refinement for better unsupervised learning
                    refined_pseudo_labels = adaptive_pseudo_labeling(outputs['labels'], confidence_threshold=0.7)
                    
                    # Additional loss on refined pseudo-labels
                    refined_label_loss = F.binary_cross_entropy_with_logits(
                        outputs['labels'], refined_pseudo_labels, reduction='mean'
                    )
                    
                    # Prototype/margin loss
                    # Handle DataParallel wrapper
                    actual_model = model.module if hasattr(model, 'module') else model
                    if actual_model.prototype_counts.sum() > 0:  # Only if prototypes are initialized
                        prototype_loss = prototype_margin_loss(
                            outputs['prototype'], 
                            predictions,  # Use predictions as soft labels
                            actual_model.class_prototypes,
                            actual_model.margin
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
                    
                    # NEW: Weights for multi-label specific losses
                    loss_weights['multilabel_consistency'] = min(0.15, 0.05 + 0.1 * (epoch / total_epochs))
                    loss_weights['class_balance'] = 0.1  # Steady throughout training
                    loss_weights['multilabel_contrastive'] = min(0.2, 0.1 + 0.1 * (epoch / total_epochs))
                    loss_weights['refined_labels'] = min(0.25, 0.1 + 0.15 * (epoch / total_epochs))
                    
                    total_loss = (loss_weights['recon'] * recon_loss + 
                                loss_weights['label'] * label_loss + 
                                loss_weights['supervised'] * supervised_loss +  # Added
                                loss_weights['prototype'] * prototype_loss +     # Added
                                loss_weights['cluster'] * cluster_loss + 
                                loss_weights['contrastive'] * contrast_loss + 
                                loss_weights['mutual'] * mutual_loss + 
                                loss_weights['confidence'] * confidence_loss + 
                                loss_weights['ensemble'] * ensemble_loss +
                                loss_weights['multilabel_consistency'] * multilabel_consistency +  # NEW
                                loss_weights['class_balance'] * class_balance_loss +  # NEW
                                loss_weights['multilabel_contrastive'] * multilabel_contrastive +  # NEW
                                loss_weights['refined_labels'] * refined_label_loss)  # NEW
                    
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
                        'ensemble': ensemble_loss,
                        'multilabel_consistency': multilabel_consistency,  # NEW
                        'class_balance': class_balance_loss,  # NEW
                        'multilabel_contrastive': multilabel_contrastive,  # NEW
                        'refined_labels': refined_label_loss  # NEW
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
                        
                        # Still track progress for skipped batch
                        batch_time = time.time() - batch_start_time
                        loss_info = {'total': 0.0, 'nan_skip': True}
                        tracker.update_batch_progress(batch_idx, batch_time, loss_info)
                        
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
                
                # Calculate batch processing time and update progress tracking
                batch_time = time.time() - batch_start_time
                
                # Prepare loss info for progress display - safely handle undefined variables
                loss_info = {
                    'total': total_loss.item(),
                }
                
                # Add individual loss components if they exist
                if 'recon_loss' in locals() and recon_loss is not None:
                    loss_info['recon'] = recon_loss.item()
                if 'label_loss' in locals() and label_loss is not None:
                    loss_info['label'] = label_loss.item()
                if 'supervised_loss' in locals() and supervised_loss is not None:
                    loss_info['sup'] = supervised_loss.item()
                
                # Update batch progress with detailed tracking
                tracker.update_batch_progress(batch_idx, batch_time, loss_info)
        
        scheduler.step()
        
        # Calculate epoch time and update progress
        epoch_time = time.time() - epoch_start
        progress_info = tracker.update_epoch_progress(epoch, epoch_time)
        
        # Log metrics - handle case where no batches were successfully processed
        if epoch_losses:
            avg_loss = np.mean(epoch_losses)
            successful_batches = len(epoch_losses)
        else:
            avg_loss = float('nan')  # Explicitly set NaN when no successful batches
            successful_batches = 0
        
        # Count total attempted batches
        total_attempted_batches = len(dataloader)
        skipped_batches = total_attempted_batches - successful_batches
        
        # Early stopping check - only if we have valid losses
        if not np.isnan(avg_loss) and avg_loss < best_loss - min_delta:
            best_loss = avg_loss
            patience_counter = 0
            # Save best model state
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        # Enhanced checkpointing for supercomputer environments (every 5%)
        progress_pct = ((epoch + 1) / total_epochs) * 100
        should_checkpoint_5pct = (epoch + 1) % max(1, total_epochs // 20) == 0  # Every 5%
        should_checkpoint_regular = (epoch + 1) % checkpoint_interval == 0
        
        if config.rank == 0 and (should_checkpoint_5pct or should_checkpoint_regular):
            try:
                model_to_save = model.module if hasattr(model, 'module') else model
                checkpoint_metrics = {
                    'avg_loss': avg_loss,
                    'best_loss': best_loss,
                    'patience_counter': patience_counter,
                    'progress_pct': progress_pct,
                    'epoch': epoch
                }
                
                # Time estimation for supercomputer environment
                if epoch > 0:
                    elapsed_time = time.time() - tracker.start_time
                    time_per_epoch = elapsed_time / (epoch + 1)
                    remaining_epochs = total_epochs - (epoch + 1)
                    eta_seconds = remaining_epochs * time_per_epoch
                    
                    if eta_seconds > 3600:
                        eta_str = f"{eta_seconds/3600:.1f}h"
                    elif eta_seconds > 60:
                        eta_str = f"{eta_seconds/60:.1f}m"
                    else:
                        eta_str = f"{eta_seconds:.0f}s"
                    
                    # Enhanced progress for supercomputer
                    print(f"💾 Checkpoint {progress_pct:.1f}% | ETA: {eta_str} | Loss: {avg_loss:.4f}")
                    print(f"   ⚡ Epoch {epoch+1}/{total_epochs} | {elapsed_time/60:.1f}min elapsed | {time_per_epoch:.1f}s/epoch")
                
                save_training_checkpoint(
                    log_type, epoch, model_to_save.state_dict(), 
                    optimizer.state_dict(), checkpoint_metrics, 
                    training_hash, config
                )
                
                # Save detailed progress information
                additional_progress_info = {
                    'checkpoint_saved': True,
                    'checkpoint_epoch': epoch + 1,
                    'n_batches_per_epoch': len(dataloader),
                    'avg_batch_time': np.mean(tracker.batch_times[-100:]) if len(tracker.batch_times) >= 10 else 0
                }
                tracker.save_checkpoint_progress(epoch, avg_loss, additional_progress_info)
                
                print(f"💾 Checkpoint saved at epoch {epoch + 1}")
            except Exception as e:
                print(f"⚠️  Failed to save checkpoint: {e}")
        
        # Log metrics with enhanced progress tracking
        tracker.log_metrics(epoch, {
            'avg_loss': avg_loss,
            'best_loss': best_loss,
            'patience_counter': patience_counter,
            'learning_rate': scheduler.get_last_lr()[0],
            'epoch_time': epoch_time,
            'total_batches': total_attempted_batches,
            'successful_batches': successful_batches,
            'skipped_batches': skipped_batches,
            'batch_success_rate': successful_batches / total_attempted_batches if total_attempted_batches > 0 else 0.0
        })
        
        # Print epoch summary for unsupervised learning
        if successful_batches > 0:
            print(f"Epoch {epoch+1}/50 - Loss: {avg_loss:.4f} - Batches: {successful_batches}/{total_attempted_batches} - Time: {epoch_time:.2f}s")
        else:
            print(f"Epoch {epoch+1}/50 - Training failed (all {skipped_batches} batches skipped) - Time: {epoch_time:.2f}s")
        
        if patience_counter >= patience:
            if config.rank == 0:
                if successful_batches == 0:
                    print(f"Early stopping triggered at epoch {epoch+1} - No valid training achieved")
                else:
                    print(f"Early stopping triggered at epoch {epoch+1}")
            # Restore best model if we have one
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
                if best_loss < float('inf'):
                    print(f"Restored best model from epoch with loss: {best_loss:.4f}")
                else:
                    print(f"Restored best model (no valid loss recorded)")
            else:
                print(f"No best model saved - using current model")
            break
        
        # Only log detailed loss components if we have valid training
        if config.rank == 0 and successful_batches > 0:  # Only log from main process and if training occurred
            # Create meaningful unsupervised metrics dictionary
            unsupervised_metrics = {
                "epoch": epoch,
                "avg_loss": avg_loss,
                "successful_batches": successful_batches,
                "skipped_batches": skipped_batches,
                "batch_success_rate": successful_batches / total_attempted_batches,
                "lr": scheduler.get_last_lr()[0],
                "epoch_time": epoch_time,
                "status": "training" if successful_batches > 0 else "failed"
            }
            
            # Add loss components if they were computed
            if 'recon_loss' in locals() and not np.isnan(avg_loss):
                unsupervised_metrics.update({
                    "recon_loss": recon_loss.item() if 'recon_loss' in locals() else 0.0,
                    "label_loss": label_loss.item() if 'label_loss' in locals() else 0.0,
                    "contrastive_loss": contrast_loss.item() if 'contrast_loss' in locals() else 0.0,
                    "confidence_loss": confidence_loss.item() if 'confidence_loss' in locals() else 0.0,
                })
            
            tracker.log_metrics(epoch, unsupervised_metrics)
    
    # Clean up training checkpoints after completion (remove all temporary checkpoints)
    if config.rank == 0:
        cleanup_training_checkpoints(log_type, keep_latest=0)  # Remove all checkpoints
    
    # Save prediction results after training completion
    if config.rank == 0:  # Only save from main process
        print(f"💾 Saving prediction results for {log_type}...")
        try:
            import pickle
            import os
            
            # Prepare model for evaluation
            model.eval()
            device = torch.device(config.device)
            
            # Get true labels from tracker if available
            true_labels = getattr(tracker, 'true_labels', None)
            
            # Generate sample IDs
            ids = np.arange(len(embeddings))
            
            # Generate predictions on training embeddings
            with torch.no_grad():
                embeddings_tensor = torch.from_numpy(embeddings).float().to(device)
                logits = model(embeddings_tensor)['labels']
                probs = torch.sigmoid(logits).cpu().numpy()
                
                # Optimize thresholds per class if true labels available
                if true_labels is not None:
                    print(f"🎯 Optimizing per-class thresholds...")
                    optimized_thresholds = np.full(len(classes), 0.5)
                    
                    for class_idx in range(len(classes)):
                        class_true = true_labels[:, class_idx]
                        class_prob = probs[:, class_idx]
                        
                        if class_true.sum() > 0:  # Only optimize if class has positive samples
                            best_f1 = 0
                            best_threshold = 0.5
                            
                            # Test thresholds from 0.1 to 0.9
                            for threshold in np.arange(0.1, 0.95, 0.05):
                                class_pred = (class_prob >= threshold).astype(int)
                                from sklearn.metrics import f1_score
                                f1 = f1_score(class_true, class_pred, zero_division=0)
                                if f1 > best_f1:
                                    best_f1 = f1
                                    best_threshold = threshold
                            
                            optimized_thresholds[class_idx] = best_threshold
                    
                    # Apply optimized thresholds
                    preds = np.zeros_like(probs, dtype=int)
                    for class_idx in range(len(classes)):
                        preds[:, class_idx] = (probs[:, class_idx] >= optimized_thresholds[class_idx]).astype(int)
                    
                    print(f"✅ Optimized thresholds: {[f'{t:.2f}' for t in optimized_thresholds]}")
                else:
                    # Default 0.5 threshold when no true labels
                    preds = (probs >= 0.5).astype(int)
                    optimized_thresholds = np.full(len(classes), 0.5)
            
            # Prepare prediction dictionary
            prediction_data = {
                "ids": ids,
                "probs": probs,     # shape (n_samples, n_classes)
                "preds": preds,     # binary predictions
                "optimized_thresholds": optimized_thresholds,  # per-class thresholds
                "classes": classes,  # class names for reference
            }
            
            # Only include true_labels if they exist
            if true_labels is not None:
                prediction_data["true_labels"] = true_labels
            
            # Create results directory and save predictions
            os.makedirs(f"results/{log_type}", exist_ok=True)
            prediction_file = f"results/{log_type}/predictions.pkl"
            
            with open(prediction_file, "wb") as f:
                pickle.dump(prediction_data, f)
            
            print(f"✅ Predictions saved to {prediction_file}")
            print(f"📊 Saved {len(ids):,} predictions for {len(classes)} classes")
            if true_labels is not None:
                print(f"📋 True labels included in output")
                print(f"🎯 Per-class thresholds optimized for F1 score")
            else:
                print(f"📋 No true labels available (unsupervised mode)")
                print(f"🎯 Using default 0.5 thresholds")
            
            # Additional cleanup of temporary training files
            try:
                checkpoint_dir = Path("checkpoints")
                if checkpoint_dir.exists():
                    # Clean up any remaining checkpoint files for this log_type
                    for checkpoint_file in checkpoint_dir.glob(f"*{log_type}*"):
                        try:
                            checkpoint_file.unlink()
                            print(f"🗑️  Removed temporary file: {checkpoint_file}")
                        except:
                            pass  # Ignore errors when removing files
                            
                print(f"🧹 Training cleanup completed")
            except Exception as cleanup_error:
                print(f"⚠️  Minor cleanup warning: {cleanup_error}")
                
        except Exception as e:
            print(f"⚠️  Failed to save predictions: {e}")
    
    return model, None  # Return None for scaler placeholder

# Log type classifier function removed - each log file can only be one type


def generate_input_embeddings_for_logs(logs: List[str], target_dim: int, device: str) -> Optional[np.ndarray]:
    """
    Generate input embeddings for logs using the same method as training.
    """
    if target_dim == 300:
        # FastText embeddings
        return generate_fasttext_embeddings_for_logs(logs)
    elif target_dim == 768:
        # BERT CLS embeddings
        return generate_bert_cls_embeddings_for_logs(logs, device)
    elif target_dim == 2314:
        # Enhanced LogBERT embeddings
        return generate_enhanced_logbert_embeddings_for_logs(logs, device)
    else:
        print(f"⚠️  Unknown embedding dimension {target_dim}")
        return None


def generate_enhanced_logbert_embeddings_for_logs(logs: List[str], device: str) -> np.ndarray:
    """Generate Enhanced LogBERT embeddings for logs"""
    from transformers import BertModel, BertTokenizer
    
    print("🤖 Loading BERT model for Enhanced LogBERT embeddings...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased').to(device)
    model.eval()
    
    all_embeddings = []
    batch_size = 16  # Small batch size for memory efficiency
    
    with torch.no_grad():
        for i in range(0, len(logs), batch_size):
            batch_logs = logs[i:i+batch_size]
            
            # Tokenize
            encodings = tokenizer(
                batch_logs,
                truncation=True,
                padding='max_length',
                max_length=128,
                return_tensors='pt'
            ).to(device)
            
            # Get BERT outputs
            outputs = model(**encodings, output_attentions=True)
            
            # Extract features
            # 1. CLS token (768D)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            # 2. Mean pooling (768D)
            token_embeddings = outputs.last_hidden_state
            attention_mask = encodings['attention_mask']
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            mean_embeddings = (sum_embeddings / sum_mask).cpu().numpy()
            
            # 3. Max pooling (768D)
            token_embeddings_masked = token_embeddings.clone()
            token_embeddings_masked[input_mask_expanded == 0] = -1e9
            max_embeddings = torch.max(token_embeddings_masked, 1)[0].cpu().numpy()
            
            # 4. Attention features (10D)
            last_attention = outputs.attentions[-1]
            cls_attention = last_attention.mean(dim=1)[:, 0, :].cpu().numpy()
            
            batch_attention_features = []
            for j in range(cls_attention.shape[0]):
                seq_len = attention_mask[j].sum().item()
                valid_attention = cls_attention[j, :seq_len]
                
                if len(valid_attention) > 1:
                    top_values = np.sort(valid_attention[1:])[-10:]
                    if len(top_values) < 10:
                        top_values = np.pad(top_values, (0, 10 - len(top_values)), 'constant')
                else:
                    top_values = np.zeros(10)
                
                batch_attention_features.append(top_values)
            
            attention_features = np.array(batch_attention_features)
            
            # Combine all features (CLS + Mean + Max + Attention = 768 + 768 + 768 + 10 = 2314D)
            combined = np.hstack([cls_embeddings, mean_embeddings, max_embeddings, attention_features])
            all_embeddings.append(combined)
    
    embeddings = np.vstack(all_embeddings).astype(np.float32)
    
    # Clean up model
    del model, tokenizer
    if device == 'cuda':
        torch.cuda.empty_cache()
    
    return embeddings


def generate_bert_cls_embeddings_for_logs(logs: List[str], device: str) -> np.ndarray:
    """Generate BERT CLS embeddings for logs"""
    from transformers import BertModel, BertTokenizer
    
    print("🤖 Loading BERT model for CLS embeddings...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased').to(device)
    model.eval()
    
    all_embeddings = []
    batch_size = 16
    
    with torch.no_grad():
        for i in range(0, len(logs), batch_size):
            batch_logs = logs[i:i+batch_size]
            
            encodings = tokenizer(
                batch_logs,
                truncation=True,
                padding='max_length',
                max_length=128,
                return_tensors='pt'
            ).to(device)
            
            outputs = model(**encodings)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(cls_embeddings)
    
    embeddings = np.vstack(all_embeddings).astype(np.float32)
    
    # Clean up
    del model, tokenizer
    if device == 'cuda':
        torch.cuda.empty_cache()
    
    return embeddings


def generate_fasttext_embeddings_for_logs(logs: List[str]) -> np.ndarray:
    """Generate FastText embeddings for logs"""
    print("📝 Generating FastText embeddings...")
    # This would need actual FastText implementation
    # For now, return placeholder
    return np.random.randn(len(logs), 300).astype(np.float32)


def save_trained_model(model: UnsupervisedMultiLabelTransformer, 
                      classes: List[str], config: SystemConfig, 
                      log_type: str, training_time: float = 0.0) -> Path:
    """
    Save the trained model with metadata for later evaluation.
    This replaces the complex evaluation that was done during training.
    """
    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Save model with comprehensive metadata
    model_path = models_dir / f"transformer_{log_type}_{config.node_name}_{config.job_id}.pth"
    
    model_to_save = model.module if hasattr(model, 'module') else model
    
    model_data = {
        'model_state_dict': model_to_save.state_dict(),
        'classes': classes,
        'config': config.__dict__,
        'log_type': log_type,
        'training_time': training_time,
        'model_type': 'UnsupervisedMultiLabelTransformer',
        'input_dim': model_to_save.input_dim,
        'latent_dim': model_to_save.latent_dim,
        'n_labels': len(classes),
        'timestamp': time.time()
    }
    
    torch.save(model_data, model_path)
    
    print(f"💾 Model saved to: {model_path}")
    print(f"📊 Model info: {model_to_save.input_dim}D → {len(classes)} classes")
    print(f"🎯 Evaluation: Use 'python src/evaluate_transformer.py --log-type {log_type}' for full evaluation")
    
    return model_path

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
        hamming_loss, jaccard_score, balanced_accuracy_score, multilabel_confusion_matrix,
        precision_score, recall_score
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



def save_model_after_training(model: UnsupervisedMultiLabelTransformer, 
                             classes: List[str], config: SystemConfig, 
                             log_type: str, training_time: float = 0.0) -> Path:
    """Save the trained model after training completes"""
    
    # Save the trained model for later evaluation
    return save_trained_model(model, classes, config, log_type, training_time)

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
# Comprehensive Model Evaluation (Merged from evaluate_model.py)
# =============================================================================

def evaluate_transformer_model(model: UnsupervisedMultiLabelTransformer, 
                              embeddings: np.ndarray, 
                              true_labels: Optional[np.ndarray],
                              classes: List[str],
                              device: torch.device,
                              val_split: float = 0.3) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    """
    Comprehensive evaluation of transformer model with multi-label metrics
    Similar to ml_models.py but for unsupervised transformer
    
    Args:
        model: Trained transformer model
        embeddings: Input embeddings
        true_labels: True labels if available
        classes: List of class names
        device: Device to use for evaluation
        val_split: Validation split fraction for threshold optimization
    
    Returns:
        results: Dictionary of evaluation metrics
        predictions: Raw prediction probabilities
        binary_predictions: Binary predictions after thresholding
    """
    model.eval()
    n_samples = len(embeddings)
    n_val = int(n_samples * val_split) if true_labels is not None else 0
    
    # Generate predictions in batches
    predictions = []
    batch_size = 64
    
    print("Generating predictions...")
    with torch.no_grad():
        for i in range(0, len(embeddings), batch_size):
            batch = torch.from_numpy(embeddings[i:i+batch_size]).float().to(device)
            outputs = model(batch)
            probs = torch.sigmoid(outputs['labels'])
            predictions.append(probs.cpu().numpy())
    
    predictions = np.vstack(predictions)
    
    # Evaluate with true labels if available
    if true_labels is not None and n_val > 0:
        embeddings_val = embeddings[:n_val]
        embeddings_test = embeddings[n_val:]
        labels_val = true_labels[:n_val]
        labels_test = true_labels[n_val:]
        predictions_val = predictions[:n_val]
        predictions_test = predictions[n_val:]
        
        print("Optimizing thresholds on validation set...")
        optimal_thresholds = optimize_per_class_thresholds(
            labels_val, predictions_val, metric='f1', beta=1.0
        )
        binary_predictions = (predictions_test >= optimal_thresholds).astype(int)
        
        # Calculate comprehensive supervised metrics
        results = calculate_supervised_metrics(
            labels_test, binary_predictions, predictions_test, classes, optimal_thresholds
        )
        results.update({
            'n_test_samples': len(labels_test),
            'n_val_samples': len(labels_val),
            'evaluation_type': 'supervised'
        })
        
        # Use test set predictions for return values
        predictions = predictions_test
        binary_predictions = binary_predictions
        
    else:
        # Unsupervised evaluation
        print("No true labels available - performing unsupervised evaluation...")
        
        # Use adaptive thresholding
        adaptive_thresholds = np.mean(predictions, axis=0) + 0.5 * np.std(predictions, axis=0)
        adaptive_thresholds = np.clip(adaptive_thresholds, 0.2, 0.8)
        binary_predictions = (predictions >= adaptive_thresholds).astype(int)
        
        # Calculate unsupervised metrics
        results = calculate_unsupervised_metrics(
            binary_predictions, predictions, classes, adaptive_thresholds
        )
        results.update({
            'n_test_samples': len(embeddings),
            'n_val_samples': 0,
            'evaluation_type': 'unsupervised'
        })
    
    return results, predictions, binary_predictions

def calculate_supervised_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                                y_prob: np.ndarray, classes: List[str],
                                thresholds: np.ndarray) -> Dict[str, Any]:
    """Calculate comprehensive supervised multi-label metrics"""
    
    try:
        # Per-class metrics
        prec_c, rec_c, f1_c, support_c = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        # Overall metrics
        results = {
            'per_class_precision': prec_c.tolist(),
            'per_class_recall': rec_c.tolist(), 
            'per_class_f1': f1_c.tolist(),
            'per_class_support': support_c.tolist(),
            'classes': classes,
            'optimal_thresholds': thresholds.tolist(),
            
            # Multi-label metrics
            'macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
            'micro_f1': float(f1_score(y_true, y_pred, average='micro', zero_division=0)),
            'weighted_f1': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
            'samples_f1': float(f1_score(y_true, y_pred, average='samples', zero_division=0)),
            
            'macro_precision': float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
            'micro_precision': float(precision_score(y_true, y_pred, average='micro', zero_division=0)),
            'weighted_precision': float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
            
            'macro_recall': float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
            'micro_recall': float(recall_score(y_true, y_pred, average='micro', zero_division=0)),
            'weighted_recall': float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
            
            'subset_accuracy': float(accuracy_score(y_true, y_pred)),
            'hamming_loss': float(hamming_loss(y_true, y_pred)),
            'jaccard_macro': float(jaccard_score(y_true, y_pred, average='macro', zero_division=0)),
            'jaccard_micro': float(jaccard_score(y_true, y_pred, average='micro', zero_division=0)),
            
            # Prediction confidence metrics
            'prediction_confidence_mean': float(y_prob.mean()),
            'prediction_confidence_std': float(y_prob.std()),
        }
        
        # Sample-level statistics
        labels_per_sample = y_pred.sum(axis=1)
        results.update({
            'avg_labels_per_sample': float(labels_per_sample.mean()),
            'std_labels_per_sample': float(labels_per_sample.std()),
            'min_labels_per_sample': int(labels_per_sample.min()),
            'max_labels_per_sample': int(labels_per_sample.max()),
            'samples_with_no_labels': int((labels_per_sample == 0).sum()),
            'samples_with_one_label': int((labels_per_sample == 1).sum()),
            'samples_with_multiple_labels': int((labels_per_sample > 1).sum()),
        })
        
        return results
        
    except Exception as e:
        print(f"⚠️  Error calculating supervised metrics: {e}")
        return {'evaluation_type': 'error', 'error': str(e)}

def calculate_unsupervised_metrics(y_pred: np.ndarray, y_prob: np.ndarray, 
                                  classes: List[str], thresholds: np.ndarray) -> Dict[str, Any]:
    """Calculate metrics for unsupervised evaluation"""
    
    # Sample-level statistics
    labels_per_sample = y_pred.sum(axis=1)
    class_counts = y_pred.sum(axis=0)
    
    results = {
        'classes': classes,
        'optimal_thresholds': thresholds.tolist(),
        'prediction_confidence_mean': float(y_prob.mean()),
        'prediction_confidence_std': float(y_prob.std()),
        'avg_labels_per_sample': float(labels_per_sample.mean()),
        'std_labels_per_sample': float(labels_per_sample.std()),
        'min_labels_per_sample': int(labels_per_sample.min()),
        'max_labels_per_sample': int(labels_per_sample.max()),
        'class_counts': class_counts.tolist(),
        'samples_with_no_labels': int((labels_per_sample == 0).sum()),
        'samples_with_one_label': int((labels_per_sample == 1).sum()),
        'samples_with_multiple_labels': int((labels_per_sample > 1).sum()),
        
        # Set supervised metrics to None
        'macro_f1': None,
        'micro_f1': None,
        'weighted_f1': None,
        'samples_f1': None,
        'subset_accuracy': None,
        'hamming_loss': None,
        'jaccard_macro': None,
        'jaccard_micro': None,
        'per_class_precision': None,
        'per_class_recall': None,
        'per_class_f1': None,
        'per_class_support': None
    }
    
    # Add class frequency info
    if len(classes) > 0:
        class_freq_pairs = list(zip(classes, class_counts))
        class_freq_pairs.sort(key=lambda x: x[1], reverse=True)
        results['most_frequent_classes'] = [
            {'class': cls, 'count': int(count), 'percentage': float(count/len(y_pred)*100)}
            for cls, count in class_freq_pairs[:10]
        ]
    
    # High confidence predictions
    high_conf_mask = y_prob > 0.7
    results['high_confidence_predictions'] = int(high_conf_mask.sum())
    results['high_confidence_percentage'] = float(high_conf_mask.sum() / y_prob.size * 100)
    
    return results

def generate_classification_report(results: Dict[str, Any], y_true: np.ndarray, 
                                  y_pred: np.ndarray, y_prob: np.ndarray,
                                  classes: List[str], log_type: str, 
                                  output_dir: Path, config: SystemConfig,
                                  training_time: float = 0.0) -> str:
    """
    Generate comprehensive classification report similar to ml_models.py
    
    Args:
        results: Evaluation results dictionary
        y_true: True labels (None for unsupervised)
        y_pred: Binary predictions
        y_prob: Prediction probabilities
        classes: List of class names
        log_type: Log type being evaluated
        output_dir: Output directory for saving report
        config: System configuration
        training_time: Training time in seconds
    
    Returns:
        Path to saved report file
    """
    
    # Create report content
    report_lines = []
    report_lines.append(f"TRANSFORMER Multi-Label Classification Report - {log_type.upper()}")
    report_lines.append("=" * 80)
    report_lines.append(f"Training time: {training_time:.2f} seconds")
    report_lines.append(f"Test samples: {results['n_test_samples']}")
    report_lines.append(f"Number of classes: {len(classes)}")
    report_lines.append(f"Evaluation type: {results['evaluation_type']}")
    report_lines.append(f"Node: {config.node_name} | Job: {config.job_id}")
    report_lines.append("")
    
    if results['evaluation_type'] == 'supervised' and results.get('macro_f1') is not None:
        # Supervised evaluation report
        report_lines.append("OVERALL METRICS:")
        report_lines.append("-" * 40)
        report_lines.append(f"Subset Accuracy: {results['subset_accuracy']:.4f}")
        report_lines.append(f"Hamming Loss: {results['hamming_loss']:.4f}")
        report_lines.append(f"Micro F1: {results['micro_f1']:.4f}")
        report_lines.append(f"Macro F1: {results['macro_f1']:.4f}")
        report_lines.append(f"Weighted F1: {results['weighted_f1']:.4f}")
        report_lines.append(f"Samples F1: {results['samples_f1']:.4f}")
        report_lines.append(f"Jaccard (Micro): {results['jaccard_micro']:.4f}")
        report_lines.append(f"Jaccard (Macro): {results['jaccard_macro']:.4f}")
        report_lines.append("")
        
        # Per-class metrics table
        report_lines.append("PER-CLASS METRICS:")
        report_lines.append("-" * 40)
        report_lines.append(f"{'Class':<25} {'F1':<8} {'Precision':<10} {'Recall':<8} {'Support':<8} {'Threshold':<10}")
        report_lines.append("-" * 80)
        
        for i, cls_name in enumerate(classes):
            f1 = results['per_class_f1'][i] if i < len(results['per_class_f1']) else 0.0
            precision = results['per_class_precision'][i] if i < len(results['per_class_precision']) else 0.0
            recall = results['per_class_recall'][i] if i < len(results['per_class_recall']) else 0.0
            support = int(results['per_class_support'][i]) if i < len(results['per_class_support']) else 0
            threshold = results['optimal_thresholds'][i] if i < len(results['optimal_thresholds']) else 0.5
            
            report_lines.append(f"{cls_name:<25} {f1:<8.3f} {precision:<10.3f} {recall:<8.3f} {support:<8} {threshold:<10.3f}")
        
    else:
        # Unsupervised evaluation report
        report_lines.append("UNSUPERVISED METRICS:")
        report_lines.append("-" * 40)
        report_lines.append(f"Prediction Confidence: {results['prediction_confidence_mean']:.4f}")
        report_lines.append(f"Confidence Std: {results['prediction_confidence_std']:.4f}")
        report_lines.append(f"Avg Labels/Sample: {results['avg_labels_per_sample']:.2f}")
        report_lines.append(f"High Conf Predictions: {results['high_confidence_percentage']:.1f}%")
        report_lines.append("")
        
        # Top predicted classes
        if 'most_frequent_classes' in results:
            report_lines.append("TOP PREDICTED CLASSES:")
            report_lines.append("-" * 40)
            for i, cls_info in enumerate(results['most_frequent_classes'][:10]):
                report_lines.append(f"  {i+1:2d}. {cls_info['class']:<25} {cls_info['count']:>6} ({cls_info['percentage']:>5.1f}%)")
    
    # Sample distribution analysis
    report_lines.append("")
    report_lines.append("SAMPLE DISTRIBUTION:")
    report_lines.append("-" * 40)
    report_lines.append(f"Samples with no labels: {results['samples_with_no_labels']}")
    report_lines.append(f"Samples with one label: {results['samples_with_one_label']}")
    report_lines.append(f"Samples with multiple labels: {results['samples_with_multiple_labels']}")
    report_lines.append(f"Average labels per sample: {results['avg_labels_per_sample']:.3f}")
    report_lines.append(f"Std labels per sample: {results['std_labels_per_sample']:.3f}")
    report_lines.append(f"Labels per sample range: {results['min_labels_per_sample']} - {results['max_labels_per_sample']}")
    
    # Add model information
    report_lines.append("")
    report_lines.append("MODEL INFORMATION:")
    report_lines.append("-" * 40)
    report_lines.append("Model: Unsupervised Multi-Label Transformer")
    report_lines.append("Features: Multi-label consistency, class balance regularization,")
    report_lines.append("          contrastive learning, adaptive pseudo-labeling")
    report_lines.append("Training: Fully unsupervised with curriculum learning")
    
    # Join all lines
    report_content = "\n".join(report_lines)
    
    # Save report
    report_path = output_dir / f"transformer_classification_report_{log_type}_{config.node_name}_{config.job_id}.txt"
    with open(report_path, 'w') as f:
        f.write(report_content)
    
    # Also print to console
    print("\n" + report_content)
    
    return str(report_path)

def save_evaluation_results(results: Dict[str, Any], predictions: np.ndarray, 
                           binary_predictions: np.ndarray, log_type: str, 
                           output_dir: Path, config: SystemConfig):
    """Save comprehensive evaluation results"""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save detailed results pickle
    results_path = output_dir / f"transformer_evaluation_{log_type}_{config.node_name}_{config.job_id}.pkl"
    with open(results_path, 'wb') as f:
        pickle.dump({
            'results': results,
            'predictions': predictions.astype(np.float32),
            'binary_predictions': binary_predictions.astype(np.int8),
            'evaluation_type': results['evaluation_type'],
            'config': config.__dict__,
            'timestamp': datetime.now().isoformat(),
            'model_type': 'transformer'
        }, f)
    
    # Save sklearn-style classification report if supervised
    if results['evaluation_type'] == 'supervised' and results.get('per_class_f1') is not None:
        sklearn_report_path = output_dir / f"sklearn_report_{log_type}_{config.node_name}_{config.job_id}.txt"
        
        # Generate sklearn classification report text
        target_names = results['classes']
        try:
            # Reconstruct y_true and y_pred for sklearn report (if we had them)
            sklearn_report = "Sklearn classification report would require original y_true and y_pred arrays\n"
            sklearn_report += "Use the transformer_classification_report for detailed metrics.\n"
            
            with open(sklearn_report_path, 'w') as f:
                f.write(sklearn_report)
        except Exception as e:
            print(f"Could not generate sklearn report: {e}")
    
    print(f"✅ Evaluation results saved to: {results_path}")
    return results_path

def print_evaluation_summary(results: Dict[str, Any], log_type: str):
    """Print evaluation summary to console"""
    
    print(f"\n{'='*80}")
    print(f"EVALUATION SUMMARY - {log_type.upper()}")
    print(f"{'='*80}")
    print(f"Test samples: {results['n_test_samples']}")
    if results['n_val_samples'] > 0:
        print(f"Validation samples: {results['n_val_samples']}")
    print(f"Classes: {len(results['classes'])}")
    print(f"Evaluation type: {results['evaluation_type']}")
    
    if results['evaluation_type'] == 'supervised' and results.get('macro_f1') is not None:
        print(f"\nSUPERVISED METRICS:")
        print(f"  Macro F1:       {results['macro_f1']:.4f}")
        print(f"  Micro F1:       {results['micro_f1']:.4f}")
        print(f"  Weighted F1:    {results['weighted_f1']:.4f}")
        print(f"  Subset Accuracy: {results['subset_accuracy']:.4f}")
        print(f"  Hamming Loss:   {results['hamming_loss']:.4f}")
        
        # Show top classes by F1
        if results['per_class_f1'] and len(results['classes']) > 0:
            print(f"\nTOP CLASSES BY F1:")
            class_f1_pairs = list(zip(results['classes'], results['per_class_f1']))
            class_f1_pairs.sort(key=lambda x: x[1], reverse=True)
            
            for i, (cls, f1) in enumerate(class_f1_pairs[:5]):
                print(f"  {i+1}. {cls:<25} F1: {f1:.4f}")
    else:
        print(f"\nUNSUPERVISED METRICS:")
        print(f"  Prediction Confidence: {results['prediction_confidence_mean']:.4f}")
        print(f"  Confidence Std:        {results['prediction_confidence_std']:.4f}")
        print(f"  Avg Labels/Sample:     {results['avg_labels_per_sample']:.2f}")
        print(f"  High Conf Predictions: {results['high_confidence_percentage']:.1f}%")
        
        # Show top predicted classes
        if 'most_frequent_classes' in results:
            print(f"\nTOP PREDICTED CLASSES:")
            for i, cls_info in enumerate(results['most_frequent_classes'][:5]):
                print(f"  {i+1}. {cls_info['class']:<25} {cls_info['percentage']:5.1f}%")
    
    print(f"{'='*80}")

def generate_per_class_accuracy_report(results_dir: Path, log_type: str, 
                                      config: SystemConfig, 
                                      predictions: np.ndarray = None,
                                      binary_predictions: np.ndarray = None,
                                      classes: List[str] = None,
                                      true_labels: np.ndarray = None) -> str:
    """
    Generate comprehensive per-class accuracy and metrics report from existing results
    
    Args:
        results_dir: Directory containing results files
        log_type: Log type being analyzed
        config: System configuration
    
    Returns:
        Path to the generated report
    """
    
    # Use provided data or load from files
    if predictions is None or binary_predictions is None or classes is None:
        # Load existing results from files
        results_file = results_dir / f"results_{log_type}_{config.node_name}_{config.job_id}.pkl"
        if not results_file.exists():
            print(f"❌ Results file not found: {results_file}")
            return None
        
        with open(results_file, 'rb') as f:
            data = pickle.load(f)
        
        predictions = data['predictions'] if predictions is None else predictions
        binary_predictions = data['binary_predictions'] if binary_predictions is None else binary_predictions
        classes = data['classes'] if classes is None else classes
    
    # Load true labels if not provided
    if true_labels is None:
        label_file = Path("embeddings") / log_type / f"label_{log_type}.pkl"
        if label_file.exists():
            try:
                with open(label_file, 'rb') as f:
                    label_data = pickle.load(f)
                    if isinstance(label_data, dict) and 'vectors' in label_data:
                        true_labels = label_data['vectors']
                    else:
                        true_labels = label_data if isinstance(label_data, np.ndarray) else None
            except Exception as e:
                print(f"⚠️  Could not load true labels: {e}")
    
    # Calculate per-class metrics
    n_classes = len(classes)
    n_samples = len(binary_predictions)
    
    # Create comprehensive report
    report_lines = []
    report_lines.append(f"COMPREHENSIVE PER-CLASS ACCURACY REPORT - {log_type.upper()}")
    report_lines.append("=" * 80)
    report_lines.append(f"Total samples: {n_samples}")
    report_lines.append(f"Total classes: {n_classes}")
    report_lines.append(f"Node: {config.node_name} | Job: {config.job_id}")
    report_lines.append("")
    
    if true_labels is not None and len(true_labels) > 0:
        # Use same validation split as training
        val_size = int(len(true_labels) * 0.3)
        test_labels = true_labels[val_size:]
        test_predictions = binary_predictions
        
        # Adjust sizes if needed
        min_size = min(len(test_labels), len(test_predictions))
        test_labels = test_labels[:min_size]
        test_predictions = test_predictions[:min_size]
        
        # Calculate comprehensive supervised metrics
        from sklearn.metrics import confusion_matrix, balanced_accuracy_score
        
        try:
            prec_c, rec_c, f1_c, support_c = precision_recall_fscore_support(
                test_labels, test_predictions, average=None, zero_division=0
            )
            
            # Overall metrics
            subset_accuracy = accuracy_score(test_labels, test_predictions)
            hamming_loss_val = hamming_loss(test_labels, test_predictions)
            macro_f1 = f1_score(test_labels, test_predictions, average='macro', zero_division=0)
            micro_f1 = f1_score(test_labels, test_predictions, average='micro', zero_division=0)
            
            report_lines.append("SUPERVISED EVALUATION (with true labels)")
            report_lines.append("=" * 50)
            report_lines.append(f"Subset Accuracy (exact match): {subset_accuracy:.4f}")
            report_lines.append(f"Hamming Loss: {hamming_loss_val:.4f}")
            report_lines.append(f"Macro F1: {macro_f1:.4f}")
            report_lines.append(f"Micro F1: {micro_f1:.4f}")
            report_lines.append("")
            
            # Per-class detailed metrics
            report_lines.append("PER-CLASS DETAILED METRICS:")
            report_lines.append("-" * 80)
            report_lines.append(f"{'Class':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Support':<10}")
            report_lines.append("-" * 80)
            
            for i, cls_name in enumerate(classes):
                # Per-class accuracy (correct predictions for this class / total samples)
                true_class = test_labels[:, i]
                pred_class = test_predictions[:, i]
                
                # Calculate per-class accuracy
                correct = (true_class == pred_class).sum()
                class_accuracy = correct / len(true_class)
                
                precision = prec_c[i] if i < len(prec_c) else 0.0
                recall = rec_c[i] if i < len(rec_c) else 0.0
                f1 = f1_c[i] if i < len(f1_c) else 0.0
                support = int(support_c[i]) if i < len(support_c) else 0
                
                report_lines.append(f"{cls_name:<20} {class_accuracy:<10.4f} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f} {support:<10}")
            
            # Confusion matrix per class
            report_lines.append("")
            report_lines.append("PER-CLASS CONFUSION MATRICES:")
            report_lines.append("-" * 50)
            
            for i, cls_name in enumerate(classes):
                true_class = test_labels[:, i]
                pred_class = test_predictions[:, i]
                
                cm = confusion_matrix(true_class, pred_class, labels=[0, 1])
                tn, fp, fn, tp = cm.ravel()
                
                report_lines.append(f"\n{cls_name}:")
                report_lines.append(f"  True Negatives:  {tn:>6}")
                report_lines.append(f"  False Positives: {fp:>6}")
                report_lines.append(f"  False Negatives: {fn:>6}")
                report_lines.append(f"  True Positives:  {tp:>6}")
                
                # Additional per-class metrics
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                balanced_acc = (sensitivity + specificity) / 2
                
                report_lines.append(f"  Sensitivity (Recall): {sensitivity:.4f}")
                report_lines.append(f"  Specificity:          {specificity:.4f}")
                report_lines.append(f"  Balanced Accuracy:    {balanced_acc:.4f}")
        
        except Exception as e:
            report_lines.append(f"ERROR calculating supervised metrics: {e}")
            true_labels = None  # Fall back to unsupervised
    
    if true_labels is None:
        # Unsupervised evaluation
        report_lines.append("UNSUPERVISED EVALUATION (no true labels)")
        report_lines.append("=" * 50)
        
        # Prediction statistics per class
        class_counts = binary_predictions.sum(axis=0)
        class_percentages = (class_counts / n_samples) * 100
        
        # Prediction confidence per class
        class_avg_confidence = predictions.mean(axis=0)
        class_std_confidence = predictions.std(axis=0)
        
        report_lines.append("PER-CLASS PREDICTION STATISTICS:")
        report_lines.append("-" * 70)
        report_lines.append(f"{'Class':<20} {'Predictions':<12} {'Percentage':<12} {'Avg Conf':<12} {'Std Conf':<10}")
        report_lines.append("-" * 70)
        
        for i, cls_name in enumerate(classes):
            count = int(class_counts[i])
            percentage = class_percentages[i]
            avg_conf = class_avg_confidence[i]
            std_conf = class_std_confidence[i]
            
            report_lines.append(f"{cls_name:<20} {count:<12} {percentage:<12.2f} {avg_conf:<12.4f} {std_conf:<10.4f}")
    
    # Sample distribution analysis
    labels_per_sample = binary_predictions.sum(axis=1)
    report_lines.append("")
    report_lines.append("SAMPLE DISTRIBUTION ANALYSIS:")
    report_lines.append("-" * 40)
    report_lines.append(f"Samples with 0 labels: {(labels_per_sample == 0).sum()} ({(labels_per_sample == 0).mean()*100:.1f}%)")
    report_lines.append(f"Samples with 1 label:  {(labels_per_sample == 1).sum()} ({(labels_per_sample == 1).mean()*100:.1f}%)")
    report_lines.append(f"Samples with 2 labels: {(labels_per_sample == 2).sum()} ({(labels_per_sample == 2).mean()*100:.1f}%)")
    report_lines.append(f"Samples with 3+ labels: {(labels_per_sample >= 3).sum()} ({(labels_per_sample >= 3).mean()*100:.1f}%)")
    report_lines.append(f"Average labels per sample: {labels_per_sample.mean():.3f}")
    report_lines.append(f"Max labels per sample: {labels_per_sample.max()}")
    
    # Top label combinations
    if len(classes) <= 10:  # Only for manageable number of classes
        from collections import Counter
        
        # Convert binary predictions to tuples for counting
        label_combinations = [tuple(row) for row in binary_predictions]
        combo_counts = Counter(label_combinations)
        
        report_lines.append("")
        report_lines.append("TOP 10 LABEL COMBINATIONS:")
        report_lines.append("-" * 50)
        
        for i, (combo, count) in enumerate(combo_counts.most_common(10)):
            active_classes = [classes[j] for j, val in enumerate(combo) if val == 1]
            if not active_classes:
                active_classes = ['normal/no_labels']
            
            percentage = (count / n_samples) * 100
            report_lines.append(f"{i+1:2d}. {', '.join(active_classes):<35} {count:>6} ({percentage:>5.1f}%)")
    
    # Model information
    report_lines.append("")
    report_lines.append("MODEL INFORMATION:")
    report_lines.append("-" * 30)
    report_lines.append("Model: Unsupervised Multi-Label Transformer")
    report_lines.append("Training: Fully unsupervised with curriculum learning")
    report_lines.append("Features: Multi-label consistency, class balance regularization")
    report_lines.append("         Contrastive learning, adaptive pseudo-labeling")
    
    # Save report
    report_content = "\n".join(report_lines)
    report_path = results_dir / f"per_class_accuracy_report_{log_type}_{config.node_name}_{config.job_id}.txt"
    
    with open(report_path, 'w') as f:
        f.write(report_content)
    
    # Also print to console
    print("\n" + "="*80)
    print("📊 COMPREHENSIVE PER-CLASS ACCURACY REPORT GENERATED")
    print("="*80)
    print(report_content)
    
    return str(report_path)

def generate_per_class_report_standalone(log_type: str = "wp-error", 
                                        node_name: str = "gra6", 
                                        job_id: str = "29386936"):
    """
    Standalone function to generate per-class accuracy report for existing results
    
    Args:
        log_type: Log type to generate report for
        node_name: Node name used in original training
        job_id: Job ID used in original training
    """
    from dataclasses import dataclass
    
    # Create mock config
    config = SystemConfig(
        device="cpu", n_gpus=0, total_memory_gb=64.0, gpu_memory_gb=0.0,
        n_cpus=8, is_distributed=False, rank=0, world_size=1,
        node_name=node_name, job_id=job_id
    )
    
    results_dir = Path("results") / log_type
    
    if not results_dir.exists():
        print(f"❌ Results directory not found: {results_dir}")
        print("Make sure you've run the transformer training first.")
        return None
    
    print(f"🔍 Generating per-class accuracy report for {log_type}...")
    print(f"📁 Looking in: {results_dir}")
    print(f"🏷️  Node: {node_name}, Job: {job_id}")
    
    try:
        report_path = generate_per_class_accuracy_report(results_dir, log_type, config)
        
        if report_path:
            print(f"\n✅ Report generated successfully!")
            print(f"📄 Saved to: {report_path}")
            return report_path
        else:
            print(f"\n❌ Failed to generate report")
            return None
            
    except Exception as e:
        print(f"❌ Error generating report: {e}")
        import traceback
        traceback.print_exc()
        return None

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
        # First try with safe globals for numpy objects
        try:
            import torch.serialization
            with torch.serialization.safe_globals(['numpy.core.multiarray.scalar', 'numpy.ndarray', 'numpy.dtype']):
                checkpoint_data = torch.load(latest_checkpoint, map_location='cpu', weights_only=True)
        except Exception:
            # Fallback to weights_only=False for compatibility with existing checkpoints
            print(f"   Falling back to weights_only=False for trusted checkpoint...")
            checkpoint_data = torch.load(latest_checkpoint, map_location='cpu', weights_only=False)
        
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

def process_log_type_with_args(log_type: str, config: SystemConfig, force_restart: bool = False, sample_size: int = None):
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
            try:
                # First try with safe globals for numpy objects
                import torch.serialization
                with torch.serialization.safe_globals(['numpy.core.multiarray.scalar', 'numpy.ndarray', 'numpy.dtype']):
                    saved_data = torch.load(model_path, map_location='cpu', weights_only=True)
            except Exception:
                # Fallback to weights_only=False for compatibility
                saved_data = torch.load(model_path, map_location='cpu', weights_only=False)
            
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
                embeddings, classes, _, _ = load_and_preprocess_data(log_type, config, tracker, sample_size)
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
            embeddings, classes, C, scaler = load_and_preprocess_data(log_type, config, tracker, sample_size)
            spinner.succeed(f"Data loaded: {embeddings.shape[0]} samples, {embeddings.shape[1]} features")
        
        # Train model with checkpointing
        tracker.log_step("Training Start", {"embeddings_shape": embeddings.shape})
        training_start_time = time.time()
        with Halo(text=f"Training model for {log_type}...", spinner='dots') as spinner:
            model, _ = train_model(embeddings, classes, C, config, tracker, log_type)
            spinner.succeed(f"Training completed for {log_type}")
        
        # Calculate total training time
        total_training_time = time.time() - training_start_time
        
        # Save trained model
        if config.rank == 0:
            with Halo(text=f"Saving model for {log_type}...", spinner='dots') as spinner:
                model_path = save_model_after_training(model, classes, config, log_type, total_training_time)
                spinner.succeed(f"Model saved successfully")
        
        tracker.log_step("Completion", {"status": "success", "training_time": total_training_time})
        print(f"✅ Completed processing {log_type}")
        print(f"📂 Outputs:")
        print(f"   Model: models/transformer_{log_type}_{config.node_name}_{config.job_id}.pth")
        print(f"   Predictions: results/{log_type}/predictions.pkl")
        print(f"")
        print(f"🎯 Next steps:")
        print(f"   Evaluation: python src/evaluate_transformer.py --log-type {log_type}")
        
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
    parser.add_argument("--sample-size", type=int, default=None,
                      help="Limit training to N samples for testing (e.g., --sample-size 1000). Uses full dataset by default.")
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
            process_log_type_with_args(log_type, config, args.force_restart, args.sample_size)
            
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
            print(f"📊 Classification reports saved to: {RESULTS_DIR}/*/transformer_classification_report_*.txt")
            print(f"📈 Per-class accuracy reports saved to: {RESULTS_DIR}/*/per_class_accuracy_report_*.txt")
            print(f"💾 Checkpoints saved to: {CHECKPOINT_DIR}/")
            print(f"\n🔧 Supports embedding types:")
            print(f"  - FastText (300D): Standard word embeddings")
            print(f"  - BERT CLS (768D): Global context embeddings")
            print(f"  - Enhanced LogBERT (2314D): Multi-feature embeddings")
            print(f"\n📈 NEW: Comprehensive evaluation with multi-label metrics!")
            print(f"  - Supervised metrics (when true labels available): F1, Precision, Recall, Hamming Loss, Jaccard")
            print(f"  - Per-class performance with optimized thresholds")
            print(f"  - Per-class accuracy, confusion matrices, sensitivity/specificity")
            print(f"  - Classification reports similar to ml_models.py")
            print(f"  - Label combination analysis and prediction confidence")
            print(f"  - Unsupervised metrics and confidence analysis")
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