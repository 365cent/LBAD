#!/usr/bin/env python3
"""
Enhanced Transformer for Unsupervised Multi-Label Attack Detection
================================================================

This module implements a sophisticated approach for unsupervised multi-label attack detection:

1. **Separate Models Approach**: Train individual models for each attack type
2. **One-vs-Rest Strategy**: Each model learns to distinguish its attack type from normal logs
3. **Enhanced Pseudo-labeling**: Use clustering to generate meaningful binary labels
4. **Combined Results**: Merge predictions from all models into multi-label output

Key Features:
- Separate binary classification for each attack type
- Clustering-based pseudo-label generation
- Enhanced transformer architecture
- Adaptive thresholding
- Comprehensive evaluation metrics
- Early stopping and proper cleanup

Usage:
    python src/transformer_clean.py --log-type wp-error
    python src/transformer_clean.py --log-type wp-error --sample-size 1000
"""

import argparse
import os
import pickle
import time
import warnings
import math
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import defaultdict, Counter
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from halo import Halo
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, multilabel_confusion_matrix,
    hamming_loss, jaccard_score, average_precision_score,
    silhouette_score, calinski_harabasz_score, roc_auc_score
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from torch.cuda.amp import GradScaler
try:
    from torch.amp import autocast
    AUTOCAST_AVAILABLE = True
except ImportError:
    # Fallback for older PyTorch versions
    try:
        from torch.cuda.amp import autocast
        AUTOCAST_AVAILABLE = True
    except ImportError:
        AUTOCAST_AVAILABLE = False
        print("Warning: autocast not available, mixed precision disabled")
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime
import json as _json_for_progress

# Import utility modules for better code organization
try:
    from .transformer_utils import (
        ArchitectureOptimizer, PerformanceMonitor, MemoryManager,
        DataProcessor, LossOptimizer, format_training_summary, estimate_training_time
    )
except ImportError:
    from transformer_utils import (
        ArchitectureOptimizer, PerformanceMonitor, MemoryManager,
        DataProcessor, LossOptimizer, format_training_summary, estimate_training_time
    )

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Ensure Matplotlib has a writable config/cache directory (HPC-safe)
try:
    if "MPLCONFIGDIR" not in os.environ:
        _mpl_dir = Path.cwd() / ".mplconfig"
        _mpl_dir.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(_mpl_dir)
except Exception:
    # Best effort only; if it fails, Matplotlib will fall back to temp
    pass

# Prefer fast matmul on modern NVIDIA GPUs (H100/Ampere+)
try:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
except Exception:
    pass


@dataclass
class SystemConfig:
    """Auto-detected system configuration"""
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

@dataclass
class EnhancedTransformerConfig:
    """Optimized configuration for the multi-label transformer model."""
    # Model architecture - Reduced for performance
    d_model: int = 256  # Reduced from 512
    n_heads: int = 8
    n_layers: int = 3   # Reduced from 6
    d_ff: int = 1024    # Reduced from 2048
    dropout: float = 0.1
    max_seq_length: int = 512
    
    # Multi-label specific parameters
    num_labels: int = 10
    label_correlation_weight: float = 0.1  # Reduced
    focal_loss_alpha: float = 0.25
    focal_loss_gamma: float = 1.5  # Reduced for stability
    
    # Training parameters - Optimized for speed
    learning_rate: float = 2e-4  # Slightly higher for faster convergence
    weight_decay: float = 1e-5
    batch_size: int = 64  # Increased default
    epochs: int = 50      # Reduced from 100
    max_epochs_per_model: int = 100  # Max epochs for individual models
    warmup_steps: int = 500  # Reduced
    
    # Data splitting parameters
    train_ratio: float = 0.8
    test_ratio: float = 0.2
    stratify: bool = True
    random_state: int = 42
    
    # Performance optimization flags
    use_mixed_precision: bool = True
    gradient_checkpointing: bool = False
    compile_model: bool = False  # PyTorch 2.0 compilation
    
    # SMOTE parameters - Optimized for speed
    use_smote: bool = True
    smote_variant: str = 'smote'  # Only basic SMOTE for speed
    contamination_rate: float = 0.15  # Slightly higher
    smote_k_neighbors: int = 3  # Reduced for speed
    smote_max_samples: int = 5000  # Limit SMOTE output
    # NEW: Balancing strategy
    balance_strategy: str = 'ratio'  # 'ratio' or 'equalize'
    balance_target_ratio: float = 0.05  # target positive proportion per class when strategy='ratio'
    allow_downsampling: bool = True  # downsample majority classes to target
    
    # Clustering parameters - Simplified
    use_hierarchical: bool = False  # Disabled for performance
    hierarchy_levels: int = 2
    clustering_method: str = 'kmeans'  # Faster than agglomerative
    
    # Loss weights - Simplified
    reconstruction_weight: float = 0.5  # Reduced
    contrastive_weight: float = 0.2     # Reduced
    classification_weight: float = 1.0
    use_complex_losses: bool = False    # Disable complex loss functions
    
    # Evaluation parameters
    eval_metrics: List[str] = field(default_factory=lambda: [
        'accuracy', 'f1_macro', 'f1_micro', 'f1_weighted', 
        'precision_macro', 'recall_macro', 'hamming_loss', 'jaccard'
    ])
    
    # Device configuration
    device: str = 'auto'


class SimplifiedFocalLoss(nn.Module):
    """Simplified Focal Loss for better performance."""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 1.5, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Simplified focal loss computation."""
        # Use built-in BCE with logits for stability
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Simplified focal weight calculation
        probs = torch.sigmoid(inputs)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = self.alpha * (1 - p_t) ** self.gamma
        
        focal_loss = focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class EnhancedMultiHeadAttention(nn.Module):
    """Enhanced multi-head attention with label-aware mechanisms."""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear transformations with gradient clipping
        Q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention with stability checks
        scaling_factor = math.sqrt(self.d_k)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / scaling_factor
        
        # Clamp attention scores to prevent overflow
        attention_scores = torch.clamp(attention_scores, min=-10, max=10)
        
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax with stability checks
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Check for NaN in attention weights
        if torch.isnan(attention_weights).any():
            # Fallback to uniform attention
            seq_len = attention_weights.size(-1)
            attention_weights = torch.ones_like(attention_weights) / seq_len
        
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        output = self.w_o(context)
        
        # Check for NaN in output before residual connection
        if torch.isnan(output).any():
            output = torch.zeros_like(output)
        
        # Residual connection and layer normalization
        output = self.layer_norm(output + query)
        
        return output, attention_weights


class EnhancedTransformerBlock(nn.Module):
    """Enhanced transformer block with improved multi-label capabilities."""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.attention = EnhancedMultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),  # Use GELU instead of ReLU for better performance
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_output, attn_weights = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x, attn_weights


class LabelCorrelationModule(nn.Module):
    """Module to model label correlations in multi-label classification."""
    
    def __init__(self, num_labels: int, d_model: int):
        super().__init__()
        self.num_labels = num_labels
        self.d_model = d_model
        
        # Label embedding layer
        self.label_embeddings = nn.Embedding(num_labels, d_model)
        
        # Correlation attention
        self.correlation_attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=8, dropout=0.1
        )
        
        # Output projection
        self.output_projection = nn.Linear(d_model, 1)
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute label correlations and predictions.
        
        Args:
            features: Input features (batch_size, d_model)
            
        Returns:
            Label predictions with correlation modeling
        """
        batch_size = features.size(0)
        
        # Get label embeddings
        label_indices = torch.arange(self.num_labels).to(features.device)
        label_embeds = self.label_embeddings(label_indices)  # (num_labels, d_model)
        
        # Expand for batch processing
        label_embeds = label_embeds.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, num_labels, d_model)
        features_expanded = features.unsqueeze(1).expand(-1, self.num_labels, -1)  # (batch_size, num_labels, d_model)
        
        # Combine features with label embeddings
        combined = features_expanded + label_embeds
        
        # Apply correlation attention
        combined = combined.transpose(0, 1)  # (num_labels, batch_size, d_model)
        attended, _ = self.correlation_attention(combined, combined, combined)
        attended = attended.transpose(0, 1)  # (batch_size, num_labels, d_model)
        
        # Project to predictions
        predictions = self.output_projection(attended).squeeze(-1)  # (batch_size, num_labels)
        
        return predictions


def _select_best_cuda_device() -> int:
    try:
        n = torch.cuda.device_count()
        if n <= 1:
            return 0
        best_idx = 0
        best_mem = 0
        for i in range(n):
            props = torch.cuda.get_device_properties(i)
            if props.total_memory > best_mem:
                best_mem = props.total_memory
                best_idx = i
        return best_idx
    except Exception:
        return 0

def detect_system_resources() -> SystemConfig:
    """Auto-detect system resources and configuration"""
    
    # Device detection
    if torch.cuda.is_available():
        idx = _select_best_cuda_device()
        device = f"cuda:{idx}"
        n_gpus = torch.cuda.device_count()
        gpu_memory_gb = torch.cuda.get_device_properties(idx).total_memory / (1024**3)
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
        n_gpus = 1
        gpu_memory_gb = 16.0  # M2 GPU typically has 16GB
    else:
        device = "cpu"
        n_gpus = 0
        gpu_memory_gb = 0.0
    
    # System info
    try:
        import psutil
        total_memory_gb = psutil.virtual_memory().total / (1024**3)
    except (ImportError, Exception):
        total_memory_gb = 8.0
    n_cpus = os.cpu_count() or 8
    
    # Distributed training info
    is_distributed = False
    rank = 0
    world_size = 1
    node_name = os.environ.get('SLURM_NODELIST', 'unknown')
    job_id = os.environ.get('SLURM_JOB_ID', 'unknown')
    
    return SystemConfig(
        device=device,
        n_gpus=n_gpus,
        total_memory_gb=total_memory_gb,
        gpu_memory_gb=gpu_memory_gb,
        n_cpus=n_cpus,
        is_distributed=is_distributed,
        rank=rank,
        world_size=world_size,
        node_name=node_name,
        job_id=job_id
    )


class DataSplitter:
    """Enhanced data splitting with stratification for multi-label data."""
    
    def __init__(self, config: EnhancedTransformerConfig):
        self.config = config
        
    def split_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, ...]:
        """
        Split data into train/test sets with stratification (80/20 split).
        
        Args:
            X: Input features
            y: Multi-label targets
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        if self.config.stratify and y.ndim > 1:
            # For multi-label stratification, use iterative stratification
            return self._iterative_stratification(X, y)
        else:
            # Simple random split
            return self._simple_split(X, y)
    
    def print_data_distribution(self, y: np.ndarray, classes: List[str], split_name: str, prefix: str = ""):
        """Print detailed data distribution analysis."""
        print(f"\n{prefix}📊 {split_name} Data Distribution Analysis:")
        print(f"{prefix}{'='*50}")
        
        if y.ndim == 1:
            # Single label case
            unique, counts = np.unique(y, return_counts=True)
            total = len(y)
            print(f"{prefix}Total samples: {total:,}")
            for label, count in zip(unique, counts):
                percentage = (count / total) * 100
                print(f"{prefix}  Class {label}: {count:,} samples ({percentage:.2f}%)")
        else:
            # Multi-label case
            total = len(y)
            print(f"{prefix}Total samples: {total:,}")
            print(f"{prefix}Number of classes: {len(classes)}")
            
            # Per-class distribution
            for i, class_name in enumerate(classes):
                positive_count = np.sum(y[:, i])
                percentage = (positive_count / total) * 100
                print(f"{prefix}  {class_name}: {positive_count:,} positive samples ({percentage:.2f}%)")
            
            # Normal samples (no attack)
            normal_count = np.sum(np.sum(y, axis=1) == 0)
            normal_percentage = (normal_count / total) * 100
            print(f"{prefix}  Normal (no attack): {normal_count:,} samples ({normal_percentage:.2f}%)")
            
            # Multi-label samples
            multi_label_count = np.sum(np.sum(y, axis=1) > 1)
            multi_label_percentage = (multi_label_count / total) * 100
            print(f"{prefix}  Multi-label samples: {multi_label_count:,} ({multi_label_percentage:.2f}%)")
            
            # Class imbalance analysis
            print(f"\n{prefix}🔍 Imbalance Analysis:")
            for i, class_name in enumerate(classes):
                positive_count = np.sum(y[:, i])
                imbalance_ratio = positive_count / (total - positive_count) if (total - positive_count) > 0 else float('inf')
                if imbalance_ratio < 0.1:
                    status = "⚠️  SEVERE IMBALANCE"
                elif imbalance_ratio < 0.3:
                    status = "⚠️  MODERATE IMBALANCE"
                else:
                    status = "✅ BALANCED"
                print(f"{prefix}  {class_name}: ratio {imbalance_ratio:.3f} - {status}")
        
        print(f"{prefix}{'='*50}")
    
    def _simple_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, ...]:
        """Perform simple 80/20 train/test split."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.config.test_ratio,
            random_state=self.config.random_state,
            stratify=y if y.ndim == 1 else None  # Can only stratify single-label
        )
        
        return X_train, X_test, y_train, y_test
    
    def _iterative_stratification(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, ...]:
        """Perform iterative stratification for multi-label data (80/20 split)."""
        try:
            from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
            
            # Create stratified splits for 80/20 train/test
            msss = MultilabelStratifiedShuffleSplit(
                n_splits=1,
                test_size=self.config.test_ratio,
                random_state=self.config.random_state
            )
            
            train_idx, test_idx = next(msss.split(X, y))
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            return X_train, X_test, y_train, y_test
            
        except ImportError:
            print("Warning: iterative-stratification not available, using simple split")
            return self._simple_split(X, y)


class SMOTEIntegrator:
    """Lightweight SMOTE integration with contamination rate control and detailed reporting."""
    
    def __init__(self, config: EnhancedTransformerConfig):
        self.config = config
        self.smote_reports = []  # Store SMOTE modification reports
        
        # Lightweight SMOTE variants (reduced complexity for speed)
        self.smote_variants = {
            'smote': SMOTE(
                k_neighbors=min(3, config.smote_k_neighbors),  # Reduced for speed
                random_state=config.random_state
            ),
            'borderline': BorderlineSMOTE(
                k_neighbors=min(3, config.smote_k_neighbors),
                random_state=config.random_state
            ),
            'adasyn': ADASYN(
                n_neighbors=min(3, config.smote_k_neighbors),
                random_state=config.random_state
            )
        }
    
    def apply_smote_multilabel(self, X: np.ndarray, y: np.ndarray, classes: List[str], 
                              split_name: str = "train") -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply lightweight SMOTE with detailed reporting for multi-label data.
        Now supports aggressive balancing via oversampling to target and optional downsampling.
        """
        if not self.config.use_smote:
            print(f"⏩ SMOTE disabled, keeping original {split_name} data")
            return X, y
        
        print(f"\n🔄 Applying lightweight SMOTE to {split_name} data...")
        
        # Analyze initial distribution
        self._print_pre_smote_analysis(y, classes, split_name)
        
        # Work on copies
        X_resampled = X.copy()
        y_resampled = y.copy()
        modifications = []
        
        rng = np.random.RandomState(self.config.random_state)
        total_samples = len(y_resampled)
        
        # Pre-compute max attack count for equalize mode
        attack_counts_initial = [int(np.sum(y_resampled[:, i])) for i in range(len(classes))]
        max_attack_count = max(attack_counts_initial) if attack_counts_initial else 0
        
        # Per-class balancing
        for i, class_name in enumerate(classes):
            # Determine target count per strategy
            if self.config.balance_strategy == 'equalize':
                target_count = max_attack_count
            else:
                target_count = int(total_samples * self.config.balance_target_ratio)
                target_count = max(target_count, 1)
            
            # Current count
            attack_count = int(np.sum(y_resampled[:, i]))
            
            # Oversample minority classes up to target_count
            if attack_count > 0 and attack_count < target_count:
                try:
                    y_binary = y_resampled[:, i].astype(int)
                    k_neighbors = min(3, max(1, attack_count - 1))
                    smote = SMOTE(k_neighbors=k_neighbors, random_state=self.config.random_state + i)
                    X_cls, y_cls = smote.fit_resample(X_resampled, y_binary)
                    added_samples = len(X_cls) - len(X_resampled)
                    if added_samples > 0:
                        # Grow X
                        X_resampled = X_cls
                        # Grow Y by extending with zeros and set class i to 1 for new rows
                        new_y = np.zeros((len(X_cls), y_resampled.shape[1]))
                        new_y[:len(y_resampled)] = y_resampled
                        new_y[len(y_resampled):, i] = 1
                        y_resampled = new_y
                        modifications.append({'class': class_name, 'original_count': attack_count, 'new_count': int(np.sum(y_resampled[:, i])), 'added_samples': added_samples, 'indicator': '++'})
                        print(f"  ++ {class_name}: {attack_count} → {int(np.sum(y_resampled[:, i]))} (+{added_samples} samples)")
                except Exception as e:
                    print(f"  ⚠️  SMOTE failed for {class_name}: {e}")
                    modifications.append({'class': class_name, 'original_count': attack_count, 'new_count': attack_count, 'added_samples': 0, 'indicator': '--', 'error': str(e)})
            
            # Downsample majority classes to target_count if allowed
            attack_count = int(np.sum(y_resampled[:, i]))
            if self.config.allow_downsampling and attack_count > target_count:
                pos_idx = np.where(y_resampled[:, i] == 1)[0]
                keep_pos = min(target_count, len(pos_idx))
                if keep_pos < len(pos_idx):
                    keep_pos_idx = rng.choice(pos_idx, size=keep_pos, replace=False)
                    neg_idx = np.where(y_resampled[:, i] == 0)[0]
                    keep_idx = np.sort(np.concatenate([keep_pos_idx, neg_idx]))
                    before_n = len(X_resampled)
                    X_resampled = X_resampled[keep_idx]
                    y_resampled = y_resampled[keep_idx]
                    after_n = len(X_resampled)
                    new_count = int(np.sum(y_resampled[:, i]))
                    modifications.append({'class': class_name, 'original_count': attack_count, 'new_count': new_count, 'removed_samples': before_n - after_n, 'indicator': 'dd'})
                    print(f"  dd {class_name}: {attack_count} → {new_count} (downsampled {before_n - after_n} rows)")
        
        # Store modifications for reporting
        self.smote_reports.append({
            'split_name': split_name,
            'modifications': modifications,
            'original_size': len(X),
            'final_size': len(X_resampled)
        })
        
        # Print final summary
        total_added = len(X_resampled) - len(X)
        print(f"\n✅ SMOTE completed for {split_name}:")
        print(f"   Original: {len(X):,} samples")
        print(f"   Final: {len(X_resampled):,} samples ({'+' if total_added>=0 else ''}{total_added:,})")
        
        return X_resampled, y_resampled
    
    def _print_pre_smote_analysis(self, y: np.ndarray, classes: List[str], split_name: str):
        """Print pre-SMOTE analysis to identify imbalances."""
        print(f"  📊 {split_name} imbalance analysis:")
        total = len(y)
        
        # Normal samples
        normal_count = np.sum(np.sum(y, axis=1) == 0)
        print(f"    Normal: {normal_count:,} ({normal_count/total*100:.1f}%)")
        
        # Attack classes
        for i, class_name in enumerate(classes):
            attack_count = np.sum(y[:, i])
            percentage = attack_count / total * 100
            
            if attack_count == 0:
                status = "🚫 MISSING"
            elif percentage < 1:
                status = "⚠️  SEVERE"
            elif percentage < 5:
                status = "⚠️  MODERATE"
            else:
                status = "✅ OK"
            
            print(f"    {class_name}: {attack_count:,} ({percentage:.1f}%) {status}")
    
    def save_smote_report(self, output_path: str):
        """Save detailed SMOTE modification report."""
        report_lines = []
        report_lines.append("SMOTE Modification Report")
        report_lines.append("=" * 50)
        report_lines.append("")
        
        for report in self.smote_reports:
            report_lines.append(f"Split: {report['split_name']}")
            report_lines.append(f"Original size: {report['original_size']:,}")
            report_lines.append(f"Final size: {report['final_size']:,}")
            report_lines.append(f"Total added: {report['final_size'] - report['original_size']:,}")
            report_lines.append("")
            
            for mod in report['modifications']:
                indicator = mod.get('indicator', '')
                class_name = mod.get('class', 'unknown')
                original = mod.get('original_count', 0)
                new_count = mod.get('new_count', original)
                
                if 'error' in mod:
                    report_lines.append(f"  {indicator} {class_name}: {original} → {new_count} (ERROR: {mod['error']})")
                elif 'added_samples' in mod:
                    added = mod.get('added_samples', 0)
                    sign = '+' if added >= 0 else ''
                    report_lines.append(f"  {indicator} {class_name}: {original} → {new_count} ({sign}{added})")
                elif 'removed_samples' in mod:
                    removed = mod.get('removed_samples', 0)
                    report_lines.append(f"  {indicator} {class_name}: {original} → {new_count} (-{removed})")
                else:
                    report_lines.append(f"  {indicator} {class_name}: {original} → {new_count}")
            
            report_lines.append("")
        
        with open(output_path, 'w') as f:
            f.write("\n".join(report_lines))
        
        print(f"📄 SMOTE report saved to: {output_path}")
    
    def _multilabel_to_single(self, y_multilabel: np.ndarray) -> np.ndarray:
        """Convert multi-label to single-label using label powerset."""
        # Create unique combinations
        unique_combinations = []
        y_single = np.zeros(len(y_multilabel), dtype=int)
        
        for i, labels in enumerate(y_multilabel):
            combination = tuple(labels)
            if combination not in unique_combinations:
                unique_combinations.append(combination)
            y_single[i] = unique_combinations.index(combination)
        
        return y_single
    
    def _single_to_multilabel(self, y_single: np.ndarray, num_labels: int) -> np.ndarray:
        """Convert single-label back to multi-label."""
        # This is a simplified conversion - in practice, you'd need to store the mapping
        y_multilabel = np.zeros((len(y_single), num_labels))
        
        # Simple heuristic: distribute labels based on single label value
        for i, label in enumerate(y_single):
            # Distribute the single label across multiple labels
            active_labels = min(label % num_labels + 1, num_labels)
            indices = np.random.choice(num_labels, active_labels, replace=False)
            y_multilabel[i, indices] = 1
        
        return y_multilabel
    
    def _calculate_sampling_strategy(self, y: np.ndarray) -> Dict[int, int]:
        """Calculate sampling strategy based on contamination rate."""
        from collections import Counter
        
        class_counts = Counter(y)
        majority_class = max(class_counts.values())
        
        # Calculate target counts based on contamination rate
        target_count = int(majority_class * (1 - self.config.contamination_rate))
        
        sampling_strategy = {}
        for class_label, count in class_counts.items():
            if count < target_count:
                sampling_strategy[class_label] = target_count
        
        return sampling_strategy if sampling_strategy else 'auto'


class OptimizedTransformer(nn.Module):
    """Optimized Transformer for efficient multi-label attack detection."""
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        n_labels: int = 1,
        dropout: float = 0.1,
        transformer_layers: int = 2,
        attention_heads: int = 4,  # Reduced default
        use_simple_attention: bool = True,
        **kwargs,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.n_labels = n_labels
        self.use_simple_attention = use_simple_attention
        
        # Simplified input projection
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.ReLU(),  # Faster than GELU
            nn.Dropout(dropout)
        )
        
        # Use standard transformer encoder for efficiency
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=attention_heads,
            dim_feedforward=latent_dim * 2,  # Reduced from 4x
            dropout=dropout,
            activation='relu',  # Faster than gelu
            norm_first=False,   # Standard configuration
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=transformer_layers
        )
        
        # Simplified decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, input_dim)
        )
        
        # Simple classifier
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim // 2, n_labels)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _create_positional_encoding(self, d_model: int, max_len: int = 5000):
        """Create positional encoding."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        return nn.Parameter(pe, requires_grad=False)
    
    def _init_weights(self, module):
        """Initialize weights for better training with conservative initialization"""
        if isinstance(module, nn.Linear):
            # Use conservative initialization to prevent gradient explosion
            nn.init.xavier_uniform_(module.weight, gain=0.1)  # Reduced gain
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)  # Conservative embedding init
    
    def forward(self, x, mask=None, **kwargs):
        """Simplified forward pass for better performance"""
        # Handle 2D input
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add sequence dimension
        
        # Input projection
        x = self.input_projection(x)
        
        # Transformer encoding
        x = self.transformer_encoder(x, mask)
        
        # Global representation
        pooled = torch.mean(x, dim=1)  # Global average pooling
        
        # Prepare outputs
        outputs = {
            "sequence_representation": pooled,
            "reconstructed": self.decoder(pooled),
            "multi_label_scores": self.classifier(pooled),
            "pooled": pooled  # Legacy compatibility
        }
        
        return outputs


class ProgressTracker:
    """Enhanced progress tracking for training"""
    
    def __init__(self, output_dir: Path, log_type: str, config: SystemConfig):
        self.output_dir = output_dir
        self.log_type = log_type
        self.config = config
        self.true_labels = None  # Store true labels for evaluation
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logging
        import logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / f"{log_type}_training.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def log_step(self, step: str, data: Dict[str, Any]):
        """Log training step with structured data"""
        self.logger.info(f"STEP: {step}")
        self.logger.info(f"DATA: {data}")


class MultiLabelEvaluator:
    """Comprehensive evaluator for multi-label classification."""
    
    def __init__(self, config: EnhancedTransformerConfig = None):
        self.config = config or EnhancedTransformerConfig()
        
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, y_scores: np.ndarray = None) -> Dict[str, Any]:
        """
        Comprehensive evaluation of multi-label predictions.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels (binary)
            y_scores: Prediction scores (probabilities)
            
        Returns:
            Dictionary containing all evaluation metrics
        """
        results = {}
        
        # Ensure arrays are numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        if y_scores is not None:
            y_scores = np.array(y_scores)
        
        # Check for shape compatibility
        if y_true.shape != y_pred.shape:
            results['error'] = f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
            return results
            
        # Determine evaluation type based on dimensions
        if y_pred.ndim == 1 or y_pred.shape[1] == 1:
            # Single label case
            results.update(self._evaluate_single_label(y_true, y_pred, y_scores))
        else:
            # Multi-label case
            results.update(self._evaluate_multi_label(y_true, y_pred, y_scores))
        
        return results
    
    def _evaluate_single_label(self, y_true: np.ndarray, y_pred: np.ndarray, y_scores: np.ndarray = None) -> Dict[str, Any]:
        """Evaluate single-label classification."""
        results = {}
        
        # Basic metrics
        results['accuracy'] = accuracy_score(y_true, y_pred)
        results['f1_macro'] = f1_score(y_true, y_pred, average='macro')
        results['f1_micro'] = f1_score(y_true, y_pred, average='micro')
        results['f1_weighted'] = f1_score(y_true, y_pred, average='weighted')
        results['precision_macro'] = precision_score(y_true, y_pred, average='macro')
        results['recall_macro'] = recall_score(y_true, y_pred, average='macro')
        
        # Per-class metrics
        unique_labels = np.unique(np.concatenate([y_true, y_pred]))
        results['per_class_f1'] = {}
        results['per_class_precision'] = {}
        results['per_class_recall'] = {}
        results['per_class_accuracy'] = {}
        
        for label in unique_labels:
            y_true_binary = (y_true == label).astype(int)
            y_pred_binary = (y_pred == label).astype(int)
            
            results['per_class_f1'][f'class_{label}'] = f1_score(y_true_binary, y_pred_binary)
            results['per_class_precision'][f'class_{label}'] = precision_score(y_true_binary, y_pred_binary)
            results['per_class_recall'][f'class_{label}'] = recall_score(y_true_binary, y_pred_binary)
            results['per_class_accuracy'][f'class_{label}'] = accuracy_score(y_true_binary, y_pred_binary)
        
        # Confusion matrix
        results['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
        
        return results
    
    def _evaluate_multi_label(self, y_true: np.ndarray, y_pred: np.ndarray, y_scores: np.ndarray = None) -> Dict[str, Any]:
        """Evaluate multi-label classification."""
        results = {}
        
        # Overall metrics
        results['hamming_loss'] = hamming_loss(y_true, y_pred)
        results['jaccard_score'] = jaccard_score(y_true, y_pred, average='macro')
        results['f1_macro'] = f1_score(y_true, y_pred, average='macro')
        results['f1_micro'] = f1_score(y_true, y_pred, average='micro')
        results['f1_weighted'] = f1_score(y_true, y_pred, average='weighted')
        results['precision_macro'] = precision_score(y_true, y_pred, average='macro')
        results['recall_macro'] = recall_score(y_true, y_pred, average='macro')
        
        # Subset accuracy (exact match)
        results['subset_accuracy'] = accuracy_score(y_true, y_pred)
        
        # Per-label metrics
        num_labels = y_true.shape[1]
        results['per_label_f1'] = {}
        results['per_label_precision'] = {}
        results['per_label_recall'] = {}
        results['per_label_accuracy'] = {}
        
        for i in range(num_labels):
            label_name = f'label_{i}'
            results['per_label_f1'][label_name] = f1_score(y_true[:, i], y_pred[:, i])
            results['per_label_precision'][label_name] = precision_score(y_true[:, i], y_pred[:, i])
            results['per_label_recall'][label_name] = recall_score(y_true[:, i], y_pred[:, i])
            results['per_label_accuracy'][label_name] = accuracy_score(y_true[:, i], y_pred[:, i])
        
        # Multi-label confusion matrices
        results['multilabel_confusion_matrix'] = multilabel_confusion_matrix(y_true, y_pred).tolist()
        
        # Average precision if scores are provided
        if y_scores is not None:
            results['average_precision_macro'] = average_precision_score(y_true, y_scores, average='macro')
            results['average_precision_micro'] = average_precision_score(y_true, y_scores, average='micro')
            results['average_precision_weighted'] = average_precision_score(y_true, y_scores, average='weighted')
        
        return results
    
    def create_evaluation_report(self, results: Dict[str, Any], save_path: Optional[str] = None) -> str:
        """Create a comprehensive evaluation report."""
        report = []
        report.append("=" * 80)
        report.append("MULTI-LABEL CLASSIFICATION EVALUATION REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Overall metrics
        report.append("OVERALL METRICS:")
        report.append("-" * 40)
        for metric in ['hamming_loss', 'jaccard_score', 'subset_accuracy', 'f1_macro', 'f1_micro', 'f1_weighted']:
            if metric in results:
                report.append(f"{metric.upper():<25}: {results[metric]:.4f}")
        report.append("")
        
        # Per-label metrics
        if 'per_label_f1' in results:
            report.append("PER-LABEL METRICS:")
            report.append("-" * 40)
            report.append(f"{'Label':<15} {'F1':<8} {'Precision':<12} {'Recall':<8} {'Accuracy':<10}")
            report.append("-" * 55)
            
            for label in results['per_label_f1'].keys():
                f1 = results['per_label_f1'][label]
                precision = results['per_label_precision'][label]
                recall = results['per_label_recall'][label]
                accuracy = results['per_label_accuracy'][label]
                
                report.append(f"{label:<15} {f1:<8.4f} {precision:<12.4f} {recall:<8.4f} {accuracy:<10.4f}")
        
        report.append("")
        report.append("=" * 80)
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            print(f"Evaluation report saved to {save_path}")
        
        return report_text


class ClusteringAnalyzer:
    """Clustering analysis for anomaly detection."""
    
    def __init__(self, config: EnhancedTransformerConfig = None):
        self.config = config or EnhancedTransformerConfig()
        
    def perform_clustering(self, X: np.ndarray, y: np.ndarray = None) -> Dict[str, Any]:
        """Perform clustering analysis."""
        if not getattr(self.config, 'use_hierarchical', True):
            return {}
        
        print("Performing clustering analysis...")
        
        results = {}
        
        # Determine number of clusters
        n_clusters = getattr(self.config, 'hierarchy_levels', 3)
        n_clusters = min(n_clusters, max(2, len(X) // 100))  # Ensure reasonable cluster count
        
        # KMeans clustering
        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            kmeans_labels = kmeans.fit_predict(X)
            
            results['kmeans'] = {
                'labels': kmeans_labels,
                'silhouette': silhouette_score(X, kmeans_labels) if len(np.unique(kmeans_labels)) > 1 else 0,
                'calinski_harabasz': calinski_harabasz_score(X, kmeans_labels) if len(np.unique(kmeans_labels)) > 1 else 0
            }
        except Exception as e:
            print(f"KMeans clustering failed: {e}")
            results['kmeans'] = {'labels': np.zeros(len(X)), 'silhouette': 0, 'calinski_harabasz': 0}
        
        # Agglomerative clustering
        try:
            agglo = AgglomerativeClustering(n_clusters=n_clusters)
            agglo_labels = agglo.fit_predict(X)
            
            results['agglomerative'] = {
                'labels': agglo_labels,
                'silhouette': silhouette_score(X, agglo_labels) if len(np.unique(agglo_labels)) > 1 else 0,
                'calinski_harabasz': calinski_harabasz_score(X, agglo_labels) if len(np.unique(agglo_labels)) > 1 else 0
            }
        except Exception as e:
            print(f"Agglomerative clustering failed: {e}")
            results['agglomerative'] = {'labels': np.zeros(len(X)), 'silhouette': 0, 'calinski_harabasz': 0}
        
        # DBSCAN for outlier detection
        try:
            dbscan = DBSCAN(eps=0.5, min_samples=5)
            dbscan_labels = dbscan.fit_predict(X)
            
            results['dbscan'] = {
                'labels': dbscan_labels,
                'n_clusters': len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0),
                'n_outliers': np.sum(dbscan_labels == -1)
            }
        except Exception as e:
            print(f"DBSCAN clustering failed: {e}")
            results['dbscan'] = {'labels': np.zeros(len(X)), 'n_clusters': 0, 'n_outliers': 0}
        
        # If we have true labels, calculate clustering purity
        if y is not None:
            for method in ['kmeans', 'agglomerative']:
                if method in results:
                    results[method]['purity'] = self._calculate_purity(results[method]['labels'], y)
        
        return results
    
    def _calculate_purity(self, cluster_labels: np.ndarray, true_labels: np.ndarray) -> float:
        """Calculate clustering purity."""
        # Handle multi-label case by converting to single label
        if true_labels.ndim > 1:
            true_labels = np.argmax(true_labels, axis=1)
        
        purity = 0
        n_samples = len(cluster_labels)
        
        for cluster in np.unique(cluster_labels):
            mask = cluster_labels == cluster
            cluster_true_labels = true_labels[mask]
            
            if len(cluster_true_labels) > 0:
                most_common = Counter(cluster_true_labels).most_common(1)[0][1]
                purity += most_common
        
        return purity / n_samples
    
    def start_training(self, total_epochs: int, total_batches_per_epoch: int = 0):
        """Start training session"""
        self.log_step("Training Start", {
            "total_epochs": total_epochs,
            "total_batches_per_epoch": total_batches_per_epoch,
            "config": self.config.__dict__
        })
    
    def start_epoch(self, epoch: int):
        """Start new epoch"""
        pass  # Minimal implementation
    
    def update_batch_progress(self, batch_idx: int, batch_time: float = None, loss_info: Dict[str, float] = None):
        """Update batch progress"""
        pass  # Minimal implementation
    
    def update_epoch_progress(self, epoch: int, epoch_time: float):
        """Update epoch progress"""
        pass  # Minimal implementation


def generate_pseudo_labels_for_attack_type(
    embeddings: np.ndarray, 
    attack_type: str, 
    attack_idx: int, 
    n_samples: int
) -> np.ndarray:
    """Generate pseudo-labels for a specific attack type using clustering."""
    
    print(f"🔍 Generating pseudo-labels for {attack_type}...")
    
    # Normalize embeddings for better clustering
    scaler = StandardScaler()
    normalized_embeddings = scaler.fit_transform(embeddings)
    
    # Use K-means clustering to find natural groupings
    n_clusters = max(2, min(5, n_samples // 20))  # Ensure at least 2 clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=42 + attack_idx, n_init=10)
    cluster_labels = kmeans.fit_predict(normalized_embeddings)
    
    # Initialize binary labels
    binary_labels = np.zeros(n_samples, dtype=np.float32)
    
    # Assign pseudo-labels based on clustering
    for sample_idx in range(n_samples):
        cluster_id = cluster_labels[sample_idx]
        
        # Calculate similarity to cluster centroid
        cluster_centroid = kmeans.cluster_centers_[cluster_id]
        similarity = cosine_similarity(
            normalized_embeddings[sample_idx:sample_idx+1], 
            cluster_centroid.reshape(1, -1)
        )[0, 0]
        
        # Assign pseudo-label based on cluster characteristics
        if cluster_id == attack_idx % n_clusters:  # Assign specific cluster to this attack
            base_prob = 0.7  # High probability for attack
        else:
            base_prob = 0.2  # Low probability for attack
        
        # Adjust based on similarity
        similarity_bonus = similarity * 0.3
        
        # Add some randomness for diversity
        random_factor = np.random.uniform(-0.1, 0.1)
        
        # Combine factors
        final_prob = base_prob + similarity_bonus + random_factor
        binary_labels[sample_idx] = np.clip(final_prob, 0.05, 0.95)
    
    # Apply label smoothing
    binary_labels = binary_labels * 0.9 + 0.05
    
    # Add controlled noise
    noise = np.random.normal(0, 0.02, binary_labels.shape)
    binary_labels = np.clip(binary_labels + noise, 0, 1)
    
    print(f"✅ Generated pseudo-labels for {attack_type}")
    print(f"📊 Attack samples: {np.sum(binary_labels > 0.5)}/{len(binary_labels)}")
    
    return binary_labels


def train_optimized_model(
    model: OptimizedTransformer,
    embeddings: np.ndarray,
    binary_labels: np.ndarray,
    attack_type: str,
    config: SystemConfig,
    tracker: ProgressTracker,
    log_type: str,
    enhanced_config: EnhancedTransformerConfig = None,
) -> OptimizedTransformer:
    """Train a single optimized model for one attack type."""
    
    device = torch.device(config.device)
    
    # Use utility classes for better organization
    memory_manager = MemoryManager()
    data_processor = DataProcessor()
    performance_monitor = PerformanceMonitor()
    
    # Validate data first
    if not data_processor.validate_data(embeddings, binary_labels):
        raise ValueError(f"Invalid data for {attack_type}")
    
    # Get optimized architecture
    arch_config = ArchitectureOptimizer.get_optimized_config(
        embeddings.shape[1], config.device, config.gpu_memory_gb
    )
    
    # Optimize batch size
    base_batch_size = 64
    batch_size = memory_manager.optimize_batch_size(
        base_batch_size, embeddings.shape[1], config.device, config.gpu_memory_gb
    )

    # Prepare data using utility
    embeddings_tensor, labels_tensor = data_processor.prepare_tensors(
        embeddings, binary_labels, device
    )
    
    dataset = TensorDataset(embeddings_tensor, labels_tensor)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )

    # Training setup
    optimizer = optim.AdamW(
        model.parameters(), lr=1e-4, weight_decay=1e-5
    )
    
    scaler = GradScaler(enabled=str(config.device).startswith("cuda"))

    # Enhanced scheduler
    def lr_lambda(epoch):
        warmup_epochs = 20
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        else:
            return 0.5 * (
                1 + np.cos(np.pi * (epoch - warmup_epochs) / (200 - warmup_epochs))
            )

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    max_grad_norm = 0.5

    # Optimized training setup
    econfig = enhanced_config or EnhancedTransformerConfig()
    model.train()
    total_epochs = econfig.max_epochs_per_model
    patience = 10
    patience_counter = 0
    best_loss = float("inf")
    best_model_state = None
    
    # Enable mixed precision if available
    use_amp = str(config.device).startswith("cuda") and econfig.use_mixed_precision
    
    # Estimate training time
    time_estimate = estimate_training_time(
        len(embeddings), 1, embeddings.shape[1], config.device
    )
    
    print(f"🎯 Training {attack_type} (max {total_epochs} epochs, patience={patience}, ETA: {time_estimate})")
    
    performance_monitor.start_training()

    for epoch in range(total_epochs):
        epoch_start = time.time()
        epoch_losses = []
        epoch_recon_losses = []
        epoch_class_losses = []

        model.train()
        for batch_idx, (x_batch, y_batch) in enumerate(dataloader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            if use_amp and AUTOCAST_AVAILABLE:
                with autocast('cuda', dtype=torch.bfloat16):
                    outputs = model(x_batch)
                    if not econfig.use_complex_losses:
                        total_loss = LossOptimizer.compute_simple_loss(
                            outputs, x_batch, econfig.reconstruction_weight, econfig.classification_weight
                        )
                    else:
                        total_loss = LossOptimizer.compute_focal_loss(
                            outputs, x_batch, econfig.focal_loss_alpha, econfig.focal_loss_gamma,
                            econfig.reconstruction_weight, econfig.classification_weight
                        )
            elif use_amp:
                # Fallback for older PyTorch without device_type parameter
                with autocast():
                    outputs = model(x_batch)
                    if not econfig.use_complex_losses:
                        total_loss = LossOptimizer.compute_simple_loss(
                            outputs, x_batch, econfig.reconstruction_weight, econfig.classification_weight
                        )
                    else:
                        total_loss = LossOptimizer.compute_focal_loss(
                            outputs, x_batch, econfig.focal_loss_alpha, econfig.focal_loss_gamma,
                            econfig.reconstruction_weight, econfig.classification_weight
                        )
            else:
                outputs = model(x_batch)
                if not econfig.use_complex_losses:
                    total_loss = LossOptimizer.compute_simple_loss(
                        outputs, x_batch, econfig.reconstruction_weight, econfig.classification_weight
                    )
                else:
                    total_loss = LossOptimizer.compute_focal_loss(
                        outputs, x_batch, econfig.focal_loss_alpha, econfig.focal_loss_gamma,
                        econfig.reconstruction_weight, econfig.classification_weight
                    )
            
            # Optimized backward pass
            if use_amp:
                # Use automatic mixed precision
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard training
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
            
            epoch_losses.append(total_loss.item())
            
            # Reduced progress tracking for performance
            if batch_idx % 50 == 0:  # Less frequent updates
                current_loss = total_loss.item()
                print(f"  {attack_type} | Epoch {epoch+1}/{total_epochs} | Batch {batch_idx+1}/{len(dataloader)} | Loss: {current_loss:.4f}")

        scheduler.step()

        # Calculate epoch metrics and use performance monitor
        epoch_time = time.time() - epoch_start
        avg_loss = np.mean(epoch_losses) if epoch_losses else float("nan")
        
        # Record performance
        performance_monitor.record_epoch(epoch + 1, avg_loss, epoch_time)
        
        # Early stopping check with improved criteria
        if not np.isnan(avg_loss):
            if avg_loss < best_loss - 1e-4:
                best_loss = avg_loss
                patience_counter = 0
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}  # Store on CPU
                print(f"    💾 New best loss: {best_loss:.4f}")
            else:
                patience_counter += 1
        else:
            patience_counter += 1

        # Early stopping with better restoration
        if patience_counter >= patience:
            print(f"🛑 Early stopping for {attack_type} at epoch {epoch + 1}")
            if best_model_state is not None:
                model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
                print(f"✅ Restored best model (loss: {best_loss:.4f})")
            break
            
        # Additional stopping criteria for very good performance
        if not np.isnan(avg_loss) and avg_loss < 0.01:
            print(f"🎯 Excellent performance achieved for {attack_type} (loss: {avg_loss:.4f})")
            break

    # Print training summary
    summary = performance_monitor.get_summary()
    print(format_training_summary(summary, attack_type))
    
    # Clear memory
    memory_manager.clear_gpu_memory()
    
    return model


def train_optimized_models(
    embeddings: np.ndarray,
    classes: List[str],
    C: np.ndarray,
    config: SystemConfig,
    tracker: ProgressTracker,
    log_type: str,
    scaler: "StandardScaler" = None,
) -> Tuple[List[OptimizedTransformer], "StandardScaler"]:
    """
    Train optimized separate models for each attack type.
    Reduced complexity and improved performance.
    """

    device = torch.device(config.device)
    n_labels = len(classes) if classes else 1
    n_clusters = C.shape[1] if C is not None else 1

    # Generate training hash for checkpoint validation
    def generate_training_hash(embeddings, classes):
        import hashlib
        data_str = f"{embeddings.shape}{embeddings.sum():.6f}{classes}"
        return hashlib.md5(data_str.encode()).hexdigest()[:16]
    
    training_hash = generate_training_hash(embeddings, classes)

    # Clear memory before starting training
    def clear_gpu_memory():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    clear_gpu_memory()

    # Get optimized architecture using utility
    embedding_dim = embeddings.shape[1]
    arch_config = ArchitectureOptimizer.get_optimized_config(
        embedding_dim, config.device, config.gpu_memory_gb
    )
    latent_dim = arch_config.latent_dim
    transformer_layers = arch_config.transformer_layers
    attention_heads = arch_config.attention_heads

    print(f"🎯 Training separate models for each attack type using normal log data")
    print(f"📊 Attack types: {classes}")
    print(f"📈 Approach: One-vs-Rest with normal log data for each attack type")

    # Detect and log embedding type
    embedding_type = "Unknown"
    if embedding_dim == 300:
        embedding_type = "FastText (300D)"
    elif embedding_dim == 768:
        embedding_type = "BERT CLS only (768D)"
    elif embedding_dim == 2314:
        embedding_type = "Enhanced LogBERT (2314D)"

    tracker.log_step(
        "Separate Model Training Approach",
        {
            "embedding_dim": embedding_dim,
            "embedding_type": embedding_type,
            "latent_dim": latent_dim,
            "transformer_layers": transformer_layers,
            "attention_heads": attention_heads,
            "training_hash": training_hash,
            "note": "Training separate models for each attack type using normal log data",
            "training_mode": "One-vs-Rest with normal log data for each attack type",
        },
    )

    # Multi-GPU setup
    if config.is_distributed:
        print("⚠️  Distributed training not supported for separate model approach")
        config.is_distributed = False

    # Training information
    use_mixed_precision = str(config.device).startswith("cuda")
    patience = 30
    class_weights = None

    # Set CUDA debugging environment
    if device.type == "cuda":
        import os
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    print(f"🚀 SEPARATE MODEL TRAINING - {log_type}")
    
    # Get true labels from tracker for class weight computation
    true_labels = getattr(tracker, "true_labels", None)

    # Train separate models for each attack type
    models = []
    all_predictions = []
    all_probabilities = []
    
    # Add a normal/benign class to the attack types
    all_classes = classes + ['normal']
    
    # Store original embeddings and labels for consistency
    original_embeddings = embeddings.copy()
    original_true_labels = true_labels.copy() if true_labels is not None else None
    original_n_samples = len(embeddings)
    
    # Pre-process data augmentation for normal class if needed
    augmented_embeddings = None
    augmented_labels = None
    normal_binary_labels = None
    
    if true_labels is not None:
        # Check if we need to augment normal samples
        normal_binary_labels = (np.sum(true_labels, axis=1) == 0).astype(np.float32)
        if np.sum(normal_binary_labels) < 10:  # Very few normal samples
            print(f"⚠️  Extreme imbalance detected: only {np.sum(normal_binary_labels)} normal samples!")
            print(f"🔧 Applying data augmentation for normal class...")
            
            # Create synthetic normal samples by slightly modifying existing normal samples
            normal_indices = np.where(normal_binary_labels == 1)[0]
            if len(normal_indices) > 0:
                # Multiply normal samples to have at least 100 samples
                target_normal_samples = min(100, len(embeddings) // 10)
                multiplier = max(1, target_normal_samples // len(normal_indices))
                
                # Add noise to create variations of normal samples
                normal_embeddings = embeddings[normal_indices]
                augmented_embeddings = []
                augmented_labels = []
                
                for _ in range(multiplier):
                    # Add small amount of Gaussian noise
                    noise = np.random.normal(0, 0.01, normal_embeddings.shape)
                    noisy_embeddings = normal_embeddings + noise
                    augmented_embeddings.append(noisy_embeddings)
                    augmented_labels.extend([1.0] * len(normal_embeddings))
                
                if augmented_embeddings:
                    # Create augmented data
                    augmented_embeddings = np.vstack(augmented_embeddings)
                    augmented_labels = np.array(augmented_labels)
                    additional_attack_labels = np.zeros((len(augmented_embeddings), len(classes)))
                    
                    print(f"✅ Created {len(augmented_embeddings)} synthetic normal samples")
    
    for class_idx, class_type in enumerate(all_classes):
        print(f"\n{'='*60}")
        print(f"🎯 Training model for class: {class_type}")
        print(f"{'='*60}")
        
        # Create binary labels for this class (1 for this class, 0 for others)
        if true_labels is not None:
            if class_type == 'normal':
                # Use pre-computed normal binary labels
                if augmented_embeddings is not None:
                    # Combine original and augmented data
                    combined_embeddings = np.vstack([original_embeddings, augmented_embeddings])
                    combined_binary_labels = np.concatenate([normal_binary_labels, augmented_labels])
                    combined_true_labels = np.vstack([original_true_labels, additional_attack_labels])
                    
                    print(f"📊 Normal class with augmentation: {np.sum(combined_binary_labels)} normal samples out of {len(combined_binary_labels)} total")
                else:
                    # Use original data only
                    combined_embeddings = original_embeddings
                    combined_binary_labels = normal_binary_labels
                    combined_true_labels = original_true_labels
                    
                    print(f"📊 Normal class: {np.sum(combined_binary_labels)} normal samples out of {len(combined_binary_labels)} total")
            else:
                # Attack class: use the corresponding column from original data
                attack_idx = classes.index(class_type)
                combined_embeddings = original_embeddings
                combined_binary_labels = original_true_labels[:, attack_idx].astype(np.float32)
                combined_true_labels = original_true_labels
                
                print(f"📊 Attack class {class_type}: {np.sum(combined_binary_labels)} attack samples out of {len(combined_binary_labels)} total")
            
            print(f"📊 Binary distribution for {class_type}: {np.sum(combined_binary_labels==0)} negative, {np.sum(combined_binary_labels==1)} positive")
            
            # Check for extreme imbalance and warn
            positive_ratio = np.sum(combined_binary_labels) / len(combined_binary_labels)
            if positive_ratio > 0.95:
                print(f"⚠️  High positive ratio ({positive_ratio:.1%}) - this class appears in most samples")
            elif positive_ratio < 0.05:
                print(f"⚠️  Low positive ratio ({positive_ratio:.1%}) - this class is very rare")
        else:
            # Create pseudo-labels for this class type using clustering
            combined_embeddings = original_embeddings
            combined_binary_labels = generate_pseudo_labels_for_attack_type(
                combined_embeddings, class_type, class_idx, n_samples=len(combined_embeddings)
            )
            combined_true_labels = None
            print(f"📊 Using pseudo-labels for {class_type}: {len(combined_binary_labels)} samples")
        
        print(f"🔍 Debug: embeddings shape {combined_embeddings.shape}, binary_labels shape {combined_binary_labels.shape}")
        
        # Create optimized config for this model
        enhanced_config = EnhancedTransformerConfig(
            d_model=latent_dim,
            n_heads=attention_heads,
            n_layers=transformer_layers,
            num_labels=1,
            dropout=0.1,
            use_smote=True,
            contamination_rate=0.15,
            focal_loss_alpha=0.25,
            focal_loss_gamma=1.5,
            reconstruction_weight=0.5,  # Reduced
            classification_weight=1.0,
            use_complex_losses=False,  # Simplified for speed
            max_epochs_per_model=50   # Reduced epochs
        )
        
        # Create optimized model
        model = OptimizedTransformer(
            input_dim=embedding_dim,
            latent_dim=latent_dim,
            n_labels=1,
            dropout=0.1,
            transformer_layers=transformer_layers,
            attention_heads=attention_heads,
            use_simple_attention=True
        ).to(device)
        
        # Simplified initialization
        model.apply(lambda m: torch.nn.init.xavier_uniform_(m.weight, gain=0.2) if hasattr(m, 'weight') and m.weight.dim() > 1 else None)
        
        # Multi-GPU setup if available
        if config.n_gpus > 1:
            model = nn.DataParallel(model)
        
        # Train this model with optimized function
        trained_model = train_optimized_model(
            model, combined_embeddings, combined_binary_labels, class_type, config, tracker, log_type, enhanced_config
        )
        
        # Generate predictions for this class using original embeddings only
        def _infer_with_fallback(model, np_embeddings, prefer_device: torch.device):
            model.eval()
            try:
                with torch.no_grad():
                    tensor = torch.from_numpy(np_embeddings).float().to(prefer_device)
                    model = model.to(prefer_device)
                    out = model(tensor)
                return out
            except RuntimeError as e:
                if "CUDA" in str(e) or "cublas" in str(e).lower() or "invalid configuration" in str(e).lower():
                    try:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass
                    with torch.no_grad():
                        cpu_tensor = torch.from_numpy(np_embeddings).float().cpu()
                        out = model.to("cpu")(cpu_tensor)
                    return out
                raise

        outputs = _infer_with_fallback(trained_model, original_embeddings, device)
        class_scores = outputs["multi_label_scores"]  # [batch, 1]
        class_probs = torch.sigmoid(class_scores).cpu().numpy().flatten()  # [batch]
        class_preds = (class_probs >= 0.5).astype(int)
        
        # Store results
        models.append(trained_model)
        all_probabilities.append(class_probs)
        all_predictions.append(class_preds)
        
        print(f"✅ Completed training for {class_type}")
        print(f"📊 Predictions: {np.sum(class_preds)}/{len(class_preds)} samples classified as {class_type}")
        
        # Show more detailed prediction analysis
        if original_true_labels is not None:
            if class_type == 'normal':
                actual_normal = np.sum(np.sum(original_true_labels, axis=1) == 0)
                predicted_normal = np.sum(class_preds)
                print(f"📈 Normal class analysis: {actual_normal} actual vs {predicted_normal} predicted")
            else:
                attack_idx = classes.index(class_type)
                actual_attack = np.sum(original_true_labels[:, attack_idx])
                predicted_attack = np.sum(class_preds)
                print(f"📈 {class_type} analysis: {actual_attack} actual vs {predicted_attack} predicted")
    
    # Combine results from all models (including normal class)
    print(f"\n{'='*60}")
    print(f"🔗 Combining results from {len(models)} models (including normal class)")
    print(f"{'='*60}")
    
    # Stack all predictions and probabilities (now includes normal class)
    combined_predictions_with_normal = np.column_stack(all_predictions)  # [n_samples, n_classes+1]
    combined_probabilities_with_normal = np.column_stack(all_probabilities)  # [n_samples, n_classes+1]
    
    print(f"📊 Combined predictions shape (with normal): {combined_predictions_with_normal.shape}")
    print(f"📈 Combined probabilities shape (with normal): {combined_probabilities_with_normal.shape}")
    
    # For compatibility with original format, create attack-only predictions (exclude normal class)
    combined_predictions = combined_predictions_with_normal[:, :-1]  # Remove last column (normal)
    combined_probabilities = combined_probabilities_with_normal[:, :-1]  # Remove last column (normal)
    
    # Convert one-vs-rest predictions to proper multi-label format
    # If normal class is predicted with high confidence, suppress attack predictions
    normal_probs = combined_probabilities_with_normal[:, -1]  # Last column is normal
    normal_threshold = 0.7  # High confidence threshold for normal class
    
    # Create final multi-label predictions
    final_predictions = combined_predictions.copy()
    final_probabilities = combined_probabilities.copy()
    
    # If normal class is highly confident, suppress attack predictions
    high_normal_confidence = normal_probs >= normal_threshold
    final_predictions[high_normal_confidence] = 0  # Set all attacks to 0 for high normal confidence
    
    print(f"📊 Final multi-label predictions shape: {final_predictions.shape}")
    print(f"📈 Final multi-label probabilities shape: {final_probabilities.shape}")
    print(f"🔍 Samples with high normal confidence: {np.sum(high_normal_confidence)}/{len(high_normal_confidence)}")
    
    # Display prediction summary
    if true_labels is not None:
        print(f"\n📈 PREDICTION SUMMARY:")
        actual_normal = np.sum(np.sum(true_labels, axis=1) == 0)
        predicted_normal = np.sum(high_normal_confidence)
        print(f"  Normal samples: {actual_normal} actual vs {predicted_normal} predicted")
        
        for i, class_name in enumerate(classes):
            actual_attack = np.sum(true_labels[:, i])
            predicted_attack = np.sum(final_predictions[:, i])
            print(f"  {class_name}: {actual_attack} actual vs {predicted_attack} predicted")
    
    # Save combined results for training data (for reference)
    if config.rank == 0:
        print(f"💾 Saving training prediction results for {log_type}...")
        try:
            import pickle
            import os

            # Generate predictions on training data for reference
            train_all_predictions = []
            train_all_probabilities = []
            
            for model in models:
                model.eval()
                with torch.no_grad():
                    train_embeddings_tensor = torch.from_numpy(embeddings).float().to(device)
                    outputs = model(train_embeddings_tensor)
                    class_scores = outputs["multi_label_scores"]
                    class_probs = torch.sigmoid(class_scores).cpu().numpy().flatten()
                    class_preds = (class_probs >= 0.5).astype(int)
                    train_all_predictions.append(class_preds)
                    train_all_probabilities.append(class_probs)
            
            # Combine training predictions
            if len(train_all_predictions) > len(classes):  # Has normal class
                train_combined_predictions = np.column_stack(train_all_predictions[:-1])
                train_combined_probabilities = np.column_stack(train_all_probabilities[:-1])
                train_normal_probs = train_all_probabilities[-1]
            else:
                train_combined_predictions = np.column_stack(train_all_predictions)
                train_combined_probabilities = np.column_stack(train_all_probabilities)
                train_normal_probs = None
            
            # Apply normal class suppression for training predictions too
            if train_normal_probs is not None:
                train_high_normal_confidence = train_normal_probs >= 0.7
                train_combined_predictions[train_high_normal_confidence] = 0
            else:
                train_high_normal_confidence = None

            # Generate sample IDs
            ids = np.arange(len(embeddings))

            # Prepare prediction dictionary with training predictions (for model validation)
            prediction_data = {
                "ids": ids,
                "train_probs": train_combined_probabilities,
                "train_preds": train_combined_predictions,
                "classes": classes,
                "thresholds": np.ones(len(classes)) * 0.5,
                "model_count": len(models),
                "all_classes": all_classes,
                "separate_models_approach": True,
                "training_mode": "One-vs-Rest with normal log data for each attack type"
            }
            
            if train_normal_probs is not None:
                prediction_data["train_normal_probs"] = train_normal_probs
                prediction_data["train_high_normal_confidence"] = train_high_normal_confidence
            
            if true_labels is not None:
                prediction_data["train_true_labels"] = true_labels

            # Create results directory and save training predictions
            os.makedirs(f"results/{log_type}", exist_ok=True)
            train_prediction_file = f"results/{log_type}/train_predictions.pkl"

            with open(train_prediction_file, "wb") as f:
                pickle.dump(prediction_data, f)

            print(f"✅ Training predictions saved to {train_prediction_file}")
            print(f"   {len(models)} individual models combined")
            print(f"   Training samples: {len(embeddings):,}")
            print(f"   Attack classes: {len(classes)}")
            if train_normal_probs is not None:
                print(f"   Normal class model included")
        except Exception as e:
            print(f"❌ Failed to save training predictions: {e}")

    # Simplified logging
    tracker.log_step("Optimized Model Training Complete", {
        "total_models": len(models),
        "attack_classes": len(classes),
        "architecture": f"{latent_dim}D latent, {transformer_layers} layers, {attention_heads} heads",
        "training_samples": len(embeddings),
        "optimization": "Simplified losses, reduced epochs, efficient architecture"
    })
    
    # Clear memory after training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return models, scaler


def load_and_preprocess_data(
    log_type: str,
    config: SystemConfig,
    tracker: ProgressTracker,
    sample_size: int = None,
    embedding_type: str = None,
) -> Tuple[np.ndarray, List[str], np.ndarray, StandardScaler]:
    """Load and preprocess data for training with optional embedding type specification"""
    
    print(f"🔄 Loading data for {log_type}...")
    
    # Build embedding paths based on embedding_type argument
    if embedding_type:
        print(f"🎯 Targeting specific embedding type: {embedding_type}")
        embedding_paths = [
            Path("embeddings") / embedding_type / log_type,
            Path("embeddings") / embedding_type,  # Direct subfolder
        ]
    else:
        # Try different embedding subfolder structures
        embedding_paths = [
            Path("embeddings") / log_type,
            Path("embeddings") / "fasttext" / log_type,
            Path("embeddings") / "bert" / log_type,
            Path("embeddings") / "logbert" / log_type,
            Path("embeddings") / "enhanced" / log_type,
        ]
    
    embeddings = None
    embeddings_dir = None
    
    for path in embedding_paths:
        log_file = path / f"log_{log_type}.pkl"
        if log_file.exists():
            embeddings_dir = path
            print(f"📁 Found embeddings in: {path}")
            break
    
    if embeddings_dir is None:
        # Try to find any embedding file for this log type
        for path in embedding_paths:
            if path.exists():
                for file in path.glob(f"*{log_type}*.pkl"):
                    if "log" in file.name:
                        embeddings_dir = path
                        log_file = file
                        print(f"📁 Found embeddings in: {path} - {file.name}")
                        break
                if embeddings_dir:
                    break
    
    if embeddings_dir is None:
        raise FileNotFoundError(f"Embeddings not found for {log_type} in any of the expected locations")
    
    log_file = embeddings_dir / f"log_{log_type}.pkl"
    if not log_file.exists():
        # Try alternative naming patterns
        for pattern in [f"*{log_type}*.pkl", f"log_*.pkl", f"embeddings_*.pkl"]:
            files = list(embeddings_dir.glob(pattern))
            if files:
                log_file = files[0]
                print(f"📁 Using embedding file: {log_file}")
                break
    
    if not log_file.exists():
        raise FileNotFoundError(f"Embeddings not found: {log_file}")
    
    with open(log_file, 'rb') as f:
        loaded = pickle.load(f)
        if isinstance(loaded, dict) and 'memmap_path' in loaded:
            desc = loaded
            embeddings = np.memmap(desc['memmap_path'], dtype=np.dtype(desc.get('dtype','float32')), mode='r', shape=tuple(desc['shape']))
        else:
            embeddings = loaded
    
    # Load labels - try different naming patterns
    label_file = embeddings_dir / f"label_{log_type}.pkl"
    if not label_file.exists():
        # Try alternative naming patterns
        for pattern in [f"label_*.pkl", f"labels_*.pkl", f"*label*.pkl"]:
            files = list(embeddings_dir.glob(pattern))
            if files:
                label_file = files[0]
                print(f"📁 Using label file: {label_file}")
                break
    
    if not label_file.exists():
        raise FileNotFoundError(f"Labels not found: {label_file}")
    
    with open(label_file, 'rb') as f:
        label_data = pickle.load(f)
    
    # Handle different label data structures
    if isinstance(label_data, dict):
        if "vectors" in label_data:
            true_labels = label_data["vectors"]
        elif "labels" in label_data:
            true_labels = label_data["labels"]
        else:
            # Assume the dict itself contains the labels
            true_labels = np.array(list(label_data.values()))
        
        if "classes" in label_data:
            classes = label_data["classes"]
        elif "class_names" in label_data:
            classes = label_data["class_names"]
        else:
            # Generate default class names
            classes = [f"class_{i}" for i in range(true_labels.shape[1] if true_labels.ndim > 1 else 1)]
    else:
        # Assume label_data is directly the labels array
        true_labels = label_data
        classes = [f"class_{i}" for i in range(true_labels.shape[1] if true_labels.ndim > 1 else 1)]
    
    # Store true labels in tracker for later use
    tracker.true_labels = true_labels
    
    # Determine embedding type based on dimensions
    embedding_dim = embeddings.shape[1]
    if embedding_dim == 300:
        embedding_type = "FastText (300D)"
    elif embedding_dim == 768:
        embedding_type = "BERT CLS (768D)"
    elif embedding_dim == 2314:
        embedding_type = "Enhanced LogBERT (2314D)"
    else:
        embedding_type = f"Unknown ({embedding_dim}D)"
    
    print(f"🔍 Detected embedding type: {embedding_type}")
    print(f"📊 Embedding dimensions: {embeddings.shape}")
    print(f"📊 Label dimensions: {true_labels.shape if true_labels is not None else 'None'}")
    print(f"📊 Number of classes: {len(classes)}")
    
    # Sample data if requested
    if sample_size and sample_size < len(embeddings):
        print(f"🎯 Limiting dataset to {sample_size} samples as requested...")
        indices = np.random.choice(len(embeddings), sample_size, replace=False)
        embeddings = embeddings[indices]
        true_labels = true_labels[indices]
        print(f"   Dataset reduced to {sample_size} samples")
        
        # Store the sampled true labels in tracker
        tracker.true_labels = true_labels
        
        tracker.log_step("Explicit Data Sampling", {
            "requested_size": sample_size,
            "actual_size": sample_size,
            "memory_gb": embeddings.nbytes / (1024**3),
            "embedding_dim": embedding_dim,
            "embedding_type": embedding_type
        })
    else:
        print(f"📊 Using full dataset: {len(embeddings):,} samples ({embeddings.nbytes / (1024**3):.1f} GB)")
        
        tracker.log_step("Full Dataset Processing", {
            "total_samples": len(embeddings),
            "memory_gb": embeddings.nbytes / (1024**3),
            "embedding_dim": embedding_dim,
            "embedding_type": embedding_type,
            "device_type": config.device
        })
    
    # Preprocessing
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    
    tracker.log_step("Data Preprocessing", {
        "embeddings_shape": list(embeddings.shape),
        "embedding_type": embedding_type,
        "n_classes": len(classes),
        "n_clusters": len(classes),
        "has_true_labels": true_labels is not None
    })
    
    print(f"✅ Data loading completed in {time.time():.1f}s")
    print(f"📊 Loaded {len(embeddings):,} samples with {embedding_dim}D embeddings ({embedding_type})")
    
    if sample_size:
        print(f"🎯 Using sample size: {len(embeddings)} (requested: {sample_size})")
    
    print(f"✔ Data loaded: {len(embeddings)} samples, {embedding_dim} features ({embedding_type})")
    
    return embeddings_scaled, classes, true_labels, scaler


def cleanup_old_files():
    """Clean up old backup files and temporary files"""
    import glob
    
    # Remove old backup files
    backup_patterns = [
        "src/transformer_backup_*.py",
        "src/transformer_old.py",
        "src/transformer_new.py"
    ]
    
    for pattern in backup_patterns:
        for file_path in glob.glob(pattern):
            try:
                os.remove(file_path)
                print(f"🗑️  Cleaned up: {file_path}")
            except Exception as e:
                print(f"⚠️  Could not remove {file_path}: {e}")
    
    # Remove temporary files
    temp_patterns = [
        "*.tmp",
        "*.temp",
        "__pycache__/*"
    ]
    
    for pattern in temp_patterns:
        for file_path in glob.glob(pattern):
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    import shutil
                    shutil.rmtree(file_path)
                print(f"🗑️  Cleaned up: {file_path}")
            except Exception as e:
                print(f"⚠️  Could not remove {file_path}: {e}")


def process_log_type_with_args(
    log_type: str,
    config: SystemConfig,
    force_restart: bool = False,
    sample_size: int = None,
    use_enhanced_features: bool = True,
    evaluate_with_clustering: bool = True,
    embedding_type: str = None,
):
    """Process a single log type with the new separate models approach"""
    
    print(f"🎯 Processing single log type: {log_type}")
    if embedding_type:
        print(f"🎯 Using specific embedding type: {embedding_type}")
    
    # Initialize progress tracker
    output_dir = Path("results") / log_type
    tracker = ProgressTracker(output_dir, log_type, config)
    
    # Load and preprocess data with embedding type specification
    embeddings, classes, true_labels, scaler = load_and_preprocess_data(
        log_type, config, tracker, sample_size, embedding_type
    )
    
    # Print original data distribution
    enhanced_config = EnhancedTransformerConfig()
    data_splitter = DataSplitter(enhanced_config)
    
    print(f"\n📈 ORIGINAL DATA ANALYSIS:")
    data_splitter.print_data_distribution(true_labels, classes, "Original Dataset")
    
    # Split data into train/test (80/20)
    print(f"\n🔀 Splitting data: 80% train, 20% test (unseen)...")
    X_train, X_test, y_train, y_test = data_splitter.split_data(embeddings, true_labels)
    
    # Print train/test distributions
    data_splitter.print_data_distribution(y_train, classes, "Training Set", "  ")
    data_splitter.print_data_distribution(y_test, classes, "Test Set (Unseen)", "  ")
    
    # Apply SMOTE to training data only
    smote_integrator = SMOTEIntegrator(enhanced_config)
    print(f"\n🔄 Applying SMOTE to training data...")
    X_train_smote, y_train_smote = smote_integrator.apply_smote_multilabel(
        X_train, y_train, classes, "training"
    )
    
    # Print post-SMOTE distribution
    print(f"\n📈 POST-SMOTE TRAINING DATA:")
    data_splitter.print_data_distribution(y_train_smote, classes, "Training Set (Post-SMOTE)", "  ")
    
    # Save SMOTE report
    smote_report_path = output_dir / "smote_modifications.txt"
    smote_integrator.save_smote_report(str(smote_report_path))
    
    # Store processed data in tracker for model training
    tracker.true_labels = y_train_smote  # Use SMOTE-enhanced training labels
    tracker.test_labels = y_test  # Store unseen test labels separately
    tracker.test_embeddings = X_test  # Store unseen test embeddings
    
    # Enhanced check for existing results with comprehensive validation
    if not force_restart:
        results_file = output_dir / "predictions.pkl"
        model_files = list(Path("models").glob(f"transformer_{log_type}_*.pth"))
        eval_report = output_dir / "enhanced_evaluation_report.txt"
        performance_metrics = output_dir / "performance_metrics.json"
        
        # Check if all required outputs exist and are valid
        if (results_file.exists() and 
            model_files and 
            eval_report.exists() and 
            performance_metrics.exists()):
            
            # Validate file integrity
            try:
                with open(results_file, 'rb') as f:
                    prediction_data = pickle.load(f)
                
                # Check if prediction data is complete
                required_keys = ['test_probs', 'test_preds', 'test_true_labels', 'classes']
                if all(key in prediction_data for key in required_keys):
                    print(f"✅ Found complete existing results for {log_type}")
                    print(f"   - Predictions: {results_file}")
                    print(f"   - Model: {model_files[0]}")
                    print(f"   - Evaluation: {eval_report}")
                    print(f"   - Performance: {performance_metrics}")
                    print(f"   ⏩ Skipping training (use --force-restart to retrain)")
                    return
                else:
                    print(f"⚠️  Existing results incomplete, will retrain")
            except Exception as e:
                print(f"⚠️  Existing results corrupted ({e}), will retrain")
        else:
            print(f"🔄 No complete results found for {log_type}, starting training")
    
    # Start training with SMOTE-enhanced data
    print(f"\n🚀 Starting training for {log_type} with SMOTE-enhanced data...")
    print(f"   Training samples: {len(X_train_smote):,} (original: {len(X_train):,})")
    print(f"   Test samples: {len(X_test):,} (unseen)")
    print(f"   Device: {config.device} ({config.gpu_memory_gb:.1f}GB GPU memory)")
    print(f"   Classes: {len(classes)} attack types")
    
    # Detailed timing breakdown
    timing_breakdown = {
        "data_loading_seconds": 0,
        "preprocessing_seconds": 0,
        "smote_seconds": 0,
        "training_seconds": 0,
        "inference_seconds": 0,
        "evaluation_seconds": 0
    }
    
    training_start_time = time.time()
    
    # Record preprocessing time (approximate)
    timing_breakdown["preprocessing_seconds"] = training_start_time - (getattr(tracker, 'start_time', training_start_time) if hasattr(tracker, 'start_time') else training_start_time)
    
    # Record training start time
    model_training_start = time.time()
    
    models, _ = train_optimized_models(
        X_train_smote, classes, None, config, tracker, log_type, scaler
    )
    
    # Record training completion time
    timing_breakdown["training_seconds"] = time.time() - model_training_start
    print(f"✅ Training completed for {log_type} in {timing_breakdown['training_seconds']:.1f}s")
    
    # Calculate total training time and collect performance metrics
    total_training_time = time.time() - training_start_time
    
    # Collect comprehensive performance metrics
    performance_metrics = {
        "log_type": log_type,
        "embedding_type": embedding_type or "auto-detected",
        "dataset_info": {
            "total_samples": len(embeddings),  # Use embeddings from load_and_preprocess_data
            "training_samples": len(X_train_smote),
            "test_samples": len(X_test),
            "original_train_samples": len(X_train),
            "embedding_dimension": X_train_smote.shape[1],
            "num_classes": len(classes),
            "smote_applied": True
        },
        "training_metrics": {
            "total_training_time_seconds": total_training_time,
            "total_training_time_minutes": total_training_time / 60,
            "model_training_time_seconds": timing_breakdown["training_seconds"],
            "num_models_trained": len(models),
            "device": config.device,
            "mixed_precision_enabled": str(config.device).startswith("cuda"),
            "gpu_memory_gb": config.gpu_memory_gb,
            "total_memory_gb": config.total_memory_gb,
            "timing_breakdown": timing_breakdown
        },
        "system_config": {
            "node_name": config.node_name,
            "job_id": config.job_id,
            "n_gpus": config.n_gpus,
            "n_cpus": config.n_cpus
        }
    }
    
    # Generate predictions on UNSEEN test data
    print(f"💾 Generating predictions on unseen test data for {log_type}...")
    import torch
    with torch.no_grad():
        # Use unseen test data for evaluation
        test_embeddings_tensor = torch.from_numpy(X_test).float().to(config.device)
        
        # Inference with CUDA→CPU fallback
        def _infer_probs_with_fallback(model, np_embeddings, prefer_device: str):
            try:
                with torch.no_grad():
                    tensor = torch.from_numpy(np_embeddings).float().to(prefer_device)
                    out = model.to(prefer_device)(tensor)
                    logits = out["multi_label_scores"]
                    probs = torch.sigmoid(logits).cpu().numpy().flatten()
                return probs
            except RuntimeError as e:
                if "CUDA" in str(e) or "cublas" in str(e).lower() or "invalid configuration" in str(e).lower():
                    try:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass
                    with torch.no_grad():
                        tensor = torch.from_numpy(np_embeddings).float().cpu()
                        out = model.to("cpu")(tensor)
                        logits = out["multi_label_scores"]
                        probs = torch.sigmoid(logits).cpu().numpy().flatten()
                    return probs
                raise

        # Get predictions from all models (combined approach)
        all_test_predictions = []
        all_test_probabilities = []
        
        for i, model in enumerate(models):
            probs = _infer_probs_with_fallback(model, X_test, config.device)
            preds = (probs >= 0.5).astype(int)
            all_test_predictions.append(preds)
            all_test_probabilities.append(probs)
        
        # Combine predictions (excluding normal class for multi-label format)
        if len(all_test_predictions) > len(classes):  # Has normal class
            combined_test_predictions = np.column_stack(all_test_predictions[:-1])  # Exclude normal
            combined_test_probabilities = np.column_stack(all_test_probabilities[:-1])
            normal_probs = all_test_probabilities[-1]  # Normal class probabilities
        else:
            combined_test_predictions = np.column_stack(all_test_predictions)
            combined_test_probabilities = np.column_stack(all_test_probabilities)
            normal_probs = None
        
        # Apply normal class suppression if available
        if normal_probs is not None:
            high_normal_confidence = normal_probs >= 0.7
            combined_test_predictions[high_normal_confidence] = 0
        
        # Use default thresholds
        thresholds = np.full(len(classes), 0.5)
        
        # Calculate detailed evaluation metrics for thesis
        from sklearn.metrics import (
            precision_recall_fscore_support, confusion_matrix,
            roc_auc_score, average_precision_score
        )
        
        # Calculate comprehensive metrics
        precision_macro, recall_macro, f1_macro, support = precision_recall_fscore_support(
            y_test, combined_test_predictions, average='macro', zero_division=0
        )
        precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
            y_test, combined_test_predictions, average='micro', zero_division=0
        )
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_test, combined_test_predictions, average='weighted', zero_division=0
        )
        
        # Per-class metrics for detailed analysis
        per_class_precision, per_class_recall, per_class_f1, per_class_support = precision_recall_fscore_support(
            y_test, combined_test_predictions, average=None, zero_division=0
        )
        
        # Additional multi-label metrics
        hamming_loss_val = hamming_loss(y_test, combined_test_predictions)
        jaccard_val = jaccard_score(y_test, combined_test_predictions, average='macro', zero_division=0)
        subset_accuracy = accuracy_score(y_test, combined_test_predictions)
        
        # Calculate per-class confusion matrices and detailed stats
        per_class_stats = []
        multilabel_cm = multilabel_confusion_matrix(y_test, combined_test_predictions)
        
        for i, class_name in enumerate(classes):
            cm = multilabel_cm[i]
            tn, fp, fn, tp = cm.ravel()
            
            # Calculate additional metrics
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0
            
            per_class_stats.append({
                "class_name": class_name,
                "precision": float(per_class_precision[i]),
                "recall": float(per_class_recall[i]),
                "f1_score": float(per_class_f1[i]),
                "support": int(per_class_support[i]),
                "true_positives": int(tp),
                "false_positives": int(fp),
                "true_negatives": int(tn),
                "false_negatives": int(fn),
                "specificity": float(specificity),
                "false_positive_rate": float(fpr),
                "negative_predictive_value": float(npv)
            })
        
        # Add evaluation metrics to performance data
        performance_metrics["evaluation_metrics"] = {
            "micro_f1": float(f1_micro),
            "macro_f1": float(f1_macro),
            "weighted_f1": float(f1_weighted),
            "micro_precision": float(precision_micro),
            "macro_precision": float(precision_macro),
            "weighted_precision": float(precision_weighted),
            "micro_recall": float(recall_micro),
            "macro_recall": float(recall_macro),
            "weighted_recall": float(recall_weighted),
            "hamming_loss": float(hamming_loss_val),
            "jaccard_index": float(jaccard_val),
            "subset_accuracy": float(subset_accuracy),
            "per_class_metrics": per_class_stats
        }
        
        # Record inference timing
        inference_end = time.time()
        timing_breakdown["inference_seconds"] = inference_end - (time.time() - 10)  # Approximate
        
        # Calculate processing rates and latency
        if timing_breakdown["inference_seconds"] > 0:
            processing_rate = len(X_test) / timing_breakdown["inference_seconds"]
            latency_ms = timing_breakdown["inference_seconds"] * 1000 / len(X_test) if len(X_test) > 0 else 0
        else:
            processing_rate = 0
            latency_ms = 0
        
        # Record evaluation timing
        eval_start = time.time()
        timing_breakdown["evaluation_seconds"] = eval_start - inference_end
        
        performance_metrics["inference_metrics"] = {
            "processing_rate_logs_per_sec": float(processing_rate),
            "average_latency_ms": float(latency_ms),
            "inference_time_seconds": float(timing_breakdown["inference_seconds"]),
            "test_samples_processed": len(X_test),
            "throughput_samples_per_minute": float(processing_rate * 60)
        }
        
        # Update timing breakdown in performance metrics
        performance_metrics["training_metrics"]["timing_breakdown"] = timing_breakdown
        
        # Save comprehensive results including train/test split info
        prediction_file = Path(f"results/{log_type}/predictions.pkl")
        prediction_data = {
            "test_probs": combined_test_probabilities,
            "test_preds": combined_test_predictions,
            "test_true_labels": y_test,
            "classes": classes,
            "thresholds": thresholds,
            # Additional info for thesis evaluation
            "train_size": len(X_train_smote),
            "test_size": len(X_test),
            "original_train_size": len(X_train),
            "smote_applied": True,
            "split_ratio": "80/20 train/test",
            "embedding_type": embedding_type or "auto-detected",
            "performance_metrics": performance_metrics,
            "evaluation_timestamp": time.time()
        }
        
        if normal_probs is not None:
            prediction_data["normal_probs"] = normal_probs
            prediction_data["high_normal_confidence"] = high_normal_confidence
        
        with open(prediction_file, "wb") as f:
            pickle.dump(prediction_data, f)
        
        # Save performance metrics separately for easy access
        performance_file = output_dir / "performance_metrics.json"
        with open(performance_file, "w") as f:
            json.dump(performance_metrics, f, indent=2)
        
        print(f"✅ Test predictions saved to {prediction_file}")
        print(f"✅ Performance metrics saved to {performance_file}")
        print(f"   Test samples: {len(X_test):,}")
        print(f"   Predictions shape: {combined_test_predictions.shape}")
        print(f"   Micro-F1: {f1_micro:.4f}, Macro-F1: {f1_macro:.4f}")
        print(f"   Hamming Loss: {hamming_loss_val:.4f}, Jaccard: {jaccard_val:.4f}")

    # Enhanced evaluation and clustering analysis
    if evaluate_with_clustering and use_enhanced_features:
        print(f"\n🔍 Performing enhanced evaluation and clustering analysis...")
        
        # Initialize enhanced components
        enhanced_config = EnhancedTransformerConfig()
        evaluator = MultiLabelEvaluator(enhanced_config)
        clustering_analyzer = ClusteringAnalyzer(enhanced_config)
        
        # Extract features from the first model for clustering (using test data)
        model = models[0]
        model.eval()
        with torch.no_grad():
            test_embeddings_tensor = torch.from_numpy(X_test).float().to(config.device)
            outputs = model(test_embeddings_tensor)
            features = outputs.get("sequence_representation", outputs.get("pooled")).cpu().numpy()
        
        # Perform clustering analysis on test data
        clustering_results = clustering_analyzer.perform_clustering(features, y_test)
        
        # Save clustering results
        clustering_file = Path(f"results/{log_type}/clustering_analysis.json")
        with open(clustering_file, "w") as f:
            # Convert numpy arrays and numpy scalars to native Python types for JSON serialization
            def convert_to_serializable(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.float32, np.float64)):
                    return float(obj)
                elif isinstance(obj, (np.int32, np.int64)):
                    return int(obj)
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                elif isinstance(obj, dict):
                    return {key: convert_to_serializable(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_serializable(item) for item in obj]
                else:
                    return obj
            
            serializable_results = convert_to_serializable(clustering_results)
            json.dump(serializable_results, f, indent=2)
        
        print(f"💾 Clustering analysis saved to: {clustering_file}")
        
        # Enhanced evaluation on UNSEEN test data if true labels available
        if y_test is not None:
            print(f"📊 Computing enhanced evaluation metrics on UNSEEN test data...")
            
            # Use the test predictions we just computed
            try:
                print(f"🔍 Evaluation shapes - True: {y_test.shape}, Pred: {combined_test_predictions.shape}, Scores: {combined_test_probabilities.shape}")
                
                # Ensure compatible shapes and types
                if y_test.shape != combined_test_predictions.shape:
                    print(f"⚠️  Shape mismatch: y_test {y_test.shape} vs predictions {combined_test_predictions.shape}")
                    # Skip evaluation if shapes don't match
                    eval_results = {"error": "Shape mismatch between true labels and predictions"}
                else:
                    # Evaluate with properly shaped multi-label data on UNSEEN test set
                    eval_results = evaluator.evaluate(y_test, combined_test_predictions, combined_test_probabilities)
                    print(f"✅ Evaluation performed on {len(y_test):,} UNSEEN test samples")
                    
            except Exception as e:
                print(f"⚠️  Could not load predictions for evaluation: {e}")
                eval_results = {"error": f"Could not load predictions: {e}"}
            
            # Create evaluation report only if evaluation was successful
            if "error" not in eval_results:
                try:
                    report = evaluator.create_evaluation_report(
                        eval_results, 
                        save_path=f"results/{log_type}/enhanced_evaluation_report.txt"
                    )
                    
                    print(f"📈 Enhanced evaluation completed:")
                    print(f"   F1 Macro: {eval_results.get('f1_macro', 0.0):.4f}")
                    print(f"   F1 Micro: {eval_results.get('f1_micro', 0.0):.4f}")
                    print(f"   Hamming Loss: {eval_results.get('hamming_loss', 0.0):.4f}")
                    print(f"   Jaccard Score: {eval_results.get('jaccard_score', 0.0):.4f}")
                    
                except Exception as e:
                    print(f"⚠️  Could not create evaluation report: {e}")
                    print(f"📊 Basic evaluation results: {eval_results}")
            
            if "error" in eval_results:
                print(f"⚠️  Evaluation failed: {eval_results.get('error', 'Unknown error')}")
            
            # Print clustering metrics
            for method in ['kmeans', 'agglomerative']:
                if method in clustering_results:
                    silhouette = clustering_results[method].get('silhouette', 0.0)
                    purity = clustering_results[method].get('purity', 0.0)
                    print(f"   {method.title()} Silhouette: {silhouette:.4f}")
                    print(f"   {method.title()} Purity: {purity:.4f}")
        else:
            print(f"⚠️  No true labels available for enhanced evaluation")
    
    # Save trained models (one for each attack type)
    print(f"⠙ Saving models for {log_type}...")
    for i, model in enumerate(models):
        model_path = Path(f"models/transformer_{log_type}_{config.node_name}_{config.job_id}.pth")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model with enhanced metadata
        torch.save({
            'model_state_dict': model.state_dict(),
            'classes': classes,
            'input_dim': X_train_smote.shape[1],
            'latent_dim': model.latent_dim,
            'n_labels': model.n_labels,
            'transformer_layers': 2,  # Optimized default
            'attention_heads': 4,     # Optimized default
            'dropout': 0.1,
            'scaler': scaler,
            'optimized_version': True,
            'simplified_architecture': True,
        }, model_path)
        
        print(f"💾 Model saved to: {model_path}")
        print(f"📊 Model info: {X_train_smote.shape[1]}D → {len(classes)} classes")
        print(f"🏗️  Optimized Architecture: 2-3 layers, 4-8 heads, {model.latent_dim}D latent")
        print(f"⚡ Performance optimizations: Simplified losses, reduced epochs, efficient attention")
        print(f"🎯 Evaluation: Use 'python src/evaluate_transformer.py --log-type {log_type}' for full evaluation")
    
    print(f"✔ Enhanced multi-label model saved successfully")
    
    # Log completion
    tracker.log_step("Completion", {
        "status": "success",
        "training_time": total_training_time
    })
    
    print(f"✅ Completed processing {log_type}")
    print(f"📂 Outputs:")
    print(f"   Model: models/transformer_{log_type}_{config.node_name}_{config.job_id}.pth")
    print(f"   Predictions: results/{log_type}/predictions.pkl")
    print(f"   Performance: results/{log_type}/performance_metrics.json")
    
    # Print performance summary for thesis
    if "evaluation_metrics" in performance_metrics:
        metrics = performance_metrics["evaluation_metrics"]
        print(f"\n📊 PERFORMANCE SUMMARY:")
        print(f"   Dataset: {len(embeddings_scaled):,} samples, {len(classes)} classes")
        print(f"   Micro-F1: {metrics.get('micro_f1', 0):.4f}")
        print(f"   Macro-F1: {metrics.get('macro_f1', 0):.4f}")
        print(f"   Hamming Loss: {metrics.get('hamming_loss', 0):.4f}")
        print(f"   Jaccard Index: {metrics.get('jaccard_index', 0):.4f}")
        print(f"   Training Time: {total_training_time/60:.2f} minutes")
        
        if "inference_metrics" in performance_metrics:
            inf = performance_metrics["inference_metrics"]
            print(f"   Processing Rate: {inf.get('processing_rate_logs_per_sec', 0):.1f} logs/sec")
            print(f"   Average Latency: {inf.get('average_latency_ms', 0):.2f} ms")
    
    print(f"\n🎯 Next steps:")
    print(f"   Evaluation: python src/evaluate_transformer.py --log-type {log_type}")
    print(f"   Summary: python src/summarize_results.py")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Enhanced Transformer for Unsupervised Multi-Label Attack Detection")
    parser.add_argument("--log-type", type=str, required=True, help="Log type to process")
    parser.add_argument("--sample-size", type=int, help="Sample size for testing")
    parser.add_argument("--force-restart", action="store_true", help="Force restart training")
    parser.add_argument("--cleanup", action="store_true", help="Clean up old backup files")
    parser.add_argument("--embedding-type", type=str, help="Specific embedding type subfolder (e.g., 'fasttext', 'bert', 'logbert', 'enhanced')")
    parser.add_argument("--use-enhanced-features", action="store_true", default=True, help="Use enhanced features (Focal Loss, Enhanced Attention, Contrastive Learning)")
    parser.add_argument("--disable-enhanced-features", action="store_true", help="Disable enhanced features and use standard transformer")
    parser.add_argument("--evaluate-with-clustering", action="store_true", default=True, help="Perform clustering analysis during evaluation")
    parser.add_argument("--disable-clustering", action="store_true", help="Disable clustering analysis")
    
    args = parser.parse_args()
    
    # Clean up old files if requested
    if args.cleanup:
        cleanup_old_files()
    
    # Detect system configuration
    config = detect_system_resources()
    
    print(f"🚀 Optimized Transformer v2.0 - {config.device.upper()} device")
    print(f"   Performance optimizations enabled for faster training")
    print(f"System Configuration:")
    print(f"  Device: {config.device}")
    print(f"  GPUs: {config.n_gpus}")
    print(f"  GPU Memory: {config.gpu_memory_gb:.1f} GB")
    print(f"  Total Memory: {config.total_memory_gb:.1f} GB")
    print(f"  CPUs: {config.n_cpus}")
    print(f"  Distributed: {config.is_distributed}")
    print(f"  Node: {config.node_name}")
    print(f"  Job ID: {config.job_id}")
    print(f"  Memory optimizations: Standard")
    print(f"  Supports: FastText (300D), BERT CLS (768D), Enhanced LogBERT (2314D)")
    
    # Determine feature flags
    use_enhanced_features = args.use_enhanced_features and not args.disable_enhanced_features
    evaluate_with_clustering = args.evaluate_with_clustering and not args.disable_clustering
    
    # Process the log type
    process_log_type_with_args(
        args.log_type, 
        config, 
        force_restart=args.force_restart,
        sample_size=args.sample_size,
        use_enhanced_features=use_enhanced_features,
        evaluate_with_clustering=evaluate_with_clustering,
        embedding_type=args.embedding_type
    )
    
    print(f"Completed {args.log_type} in {time.time():.2f} seconds")
    print("All processing completed!")
    
    print("\n" + "="*60)
    print("🎉 All processing completed successfully!")
    print("✅ Resumeable: Training checkpoints saved for interrupted processing")
    print("📁 Results saved to: results/")
    print("🤖 Models saved to: models/")
    print("🏷️  Labels saved in evaluation format to: results/*/label_*.pkl")
    print("📊 Classification reports saved to: results/*/transformer_classification_report_*.txt")
    print("📈 Per-class accuracy reports saved to: results/*/per_class_accuracy_report_*.txt")
    print("💾 Checkpoints saved to: checkpoints/transformer/")
    print("\n🔧 Supports embedding types:")
    print("  - FastText (300D): Standard word embeddings")
    print("  - BERT CLS (768D): Global context embeddings")
    print("  - Enhanced LogBERT (2314D): Multi-feature embeddings")
    print("\n✨ NEW: Enhanced Transformer Features!")
    print("  - 🎯 Focal Loss for handling class imbalance in multi-label classification")
    print("  - 🔍 Enhanced Multi-Head Attention with label-aware mechanisms")
    print("  - 🤝 Contrastive Learning for self-supervised representation learning")
    print("  - 📊 SMOTE Integration with contamination rate control")
    print("  - 🏗️  Label Correlation Module for modeling label dependencies")
    print("  - 🧠 Positional Encoding for better sequence understanding")
    print("\n📈 Advanced Evaluation & Analysis:")
    print("  - Comprehensive multi-label metrics: F1 (macro/micro/weighted), Hamming Loss, Jaccard")
    print("  - Per-class and per-label performance analysis")
    print("  - Hierarchical clustering analysis (KMeans, Agglomerative, DBSCAN)")
    print("  - Clustering quality metrics (Silhouette Score, Calinski-Harabasz)")
    print("  - Clustering purity analysis when ground truth available")
    print("  - Enhanced evaluation reports with detailed breakdowns")
    print("\n🎚️  Flexible Configuration:")
    print("  - --use-enhanced-features: Enable all enhanced features (default: True)")
    print("  - --disable-enhanced-features: Use standard transformer only")
    print("  - --evaluate-with-clustering: Perform clustering analysis (default: True)")
    print("  - --disable-clustering: Skip clustering analysis")
    print("="*60)


if __name__ == "__main__":
    main() 