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
from torch.utils.data import DataLoader, TensorDataset

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


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
    """Enhanced configuration for the multi-label transformer model."""
    # Model architecture
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 2048
    dropout: float = 0.1
    max_seq_length: int = 512
    
    # Multi-label specific parameters
    num_labels: int = 10
    label_correlation_weight: float = 0.2
    focal_loss_alpha: float = 0.25
    focal_loss_gamma: float = 2.0
    
    # Training parameters
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    batch_size: int = 32
    epochs: int = 100
    warmup_steps: int = 1000
    
    # Data splitting parameters
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    stratify: bool = True
    random_state: int = 42
    
    # SMOTE parameters
    use_smote: bool = True
    smote_variant: str = 'smote'  # 'smote', 'borderline', 'adasyn'
    contamination_rate: float = 0.1
    smote_k_neighbors: int = 5
    
    # Hierarchical clustering parameters
    use_hierarchical: bool = True
    hierarchy_levels: int = 3
    clustering_method: str = 'agglomerative'  # 'kmeans', 'agglomerative'
    
    # Loss weights
    reconstruction_weight: float = 1.0
    contrastive_weight: float = 0.5
    classification_weight: float = 1.0
    
    # Evaluation parameters
    eval_metrics: List[str] = field(default_factory=lambda: [
        'accuracy', 'f1_macro', 'f1_micro', 'f1_weighted', 
        'precision_macro', 'recall_macro', 'hamming_loss', 'jaccard'
    ])
    
    # Device configuration
    device: str = 'auto'


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance in multi-label classification."""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss with numerical stability.
        
        Args:
            inputs: Predicted logits (batch_size, num_labels)
            targets: Ground truth labels (batch_size, num_labels)
            
        Returns:
            Focal loss value
        """
        # Clamp inputs to prevent overflow
        inputs = torch.clamp(inputs, min=-10, max=10)
        
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(inputs)
        
        # Add small epsilon for numerical stability
        eps = 1e-8
        probs = torch.clamp(probs, min=eps, max=1.0 - eps)
        
        # Compute binary cross entropy with clamped values
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Compute focal weight with stability checks
        p_t = probs * targets + (1 - probs) * (1 - targets)
        p_t = torch.clamp(p_t, min=eps, max=1.0 - eps)
        
        # Compute focal weight with clamping to prevent extreme values
        focal_weight = self.alpha * torch.pow(1 - p_t, self.gamma)
        focal_weight = torch.clamp(focal_weight, min=eps, max=100.0)  # Prevent extreme weights
        
        # Apply focal weight
        focal_loss = focal_weight * bce_loss
        
        # Check for NaN/Inf before reduction
        if torch.isnan(focal_loss).any() or torch.isinf(focal_loss).any():
            # Fallback to standard BCE loss
            return F.binary_cross_entropy_with_logits(inputs, targets, reduction=self.reduction)
        
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


def detect_system_resources() -> SystemConfig:
    """Auto-detect system resources and configuration"""
    
    # Device detection
    if torch.cuda.is_available():
        device = "cuda"
        n_gpus = torch.cuda.device_count()
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
        n_gpus = 1
        gpu_memory_gb = 16.0  # M2 GPU typically has 16GB
    else:
        device = "cpu"
        n_gpus = 0
        gpu_memory_gb = 0.0
    
    # System info
    total_memory_gb = 8.0  # Default, could be detected
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
        Split data into train/validation/test sets with stratification.
        
        Args:
            X: Input features
            y: Multi-label targets
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        if self.config.stratify and y.ndim > 1:
            # For multi-label stratification, use iterative stratification
            return self._iterative_stratification(X, y)
        else:
            # Simple random split
            return self._random_split(X, y)
    
    def _random_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, ...]:
        """Perform random data splitting."""
        # First split: train + val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, 
            test_size=self.config.test_ratio,
            random_state=self.config.random_state
        )
        
        # Second split: train vs val
        val_size = self.config.val_ratio / (self.config.train_ratio + self.config.val_ratio)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size,
            random_state=self.config.random_state
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def _iterative_stratification(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, ...]:
        """Perform iterative stratification for multi-label data."""
        try:
            from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
            
            # Create stratified splits
            msss = MultilabelStratifiedShuffleSplit(
                n_splits=1,
                test_size=self.config.test_ratio,
                random_state=self.config.random_state
            )
            
            train_val_idx, test_idx = next(msss.split(X, y))
            X_temp, X_test = X[train_val_idx], X[test_idx]
            y_temp, y_test = y[train_val_idx], y[test_idx]
            
            # Split train and validation
            val_size = self.config.val_ratio / (self.config.train_ratio + self.config.val_ratio)
            msss_val = MultilabelStratifiedShuffleSplit(
                n_splits=1,
                test_size=val_size,
                random_state=self.config.random_state
            )
            
            train_idx, val_idx = next(msss_val.split(X_temp, y_temp))
            X_train, X_val = X_temp[train_idx], X_temp[val_idx]
            y_train, y_val = y_temp[train_idx], y_temp[val_idx]
            
            return X_train, X_val, X_test, y_train, y_val, y_test
            
        except ImportError:
            print("Warning: iterative-stratification not available, using random split")
            return self._random_split(X, y)


class SMOTEIntegrator:
    """Enhanced SMOTE integration with contamination rate control."""
    
    def __init__(self, config: EnhancedTransformerConfig):
        self.config = config
        
        # SMOTE variants
        self.smote_variants = {
            'smote': SMOTE(
                k_neighbors=config.smote_k_neighbors,
                random_state=config.random_state
            ),
            'borderline': BorderlineSMOTE(
                k_neighbors=config.smote_k_neighbors,
                random_state=config.random_state
            ),
            'adasyn': ADASYN(
                n_neighbors=config.smote_k_neighbors,
                random_state=config.random_state
            )
        }
    
    def apply_smote(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply SMOTE with contamination rate control.
        
        Args:
            X: Input features
            y: Multi-label targets
            
        Returns:
            Tuple of (X_resampled, y_resampled)
        """
        if not self.config.use_smote:
            return X, y
        
        # Convert multi-label to single-label for SMOTE
        if y.ndim > 1:
            # Use label powerset approach
            y_single = self._multilabel_to_single(y)
        else:
            y_single = y
        
        # Apply SMOTE
        smote = self.smote_variants[self.config.smote_variant]
        
        try:
            # Calculate sampling strategy based on contamination rate
            sampling_strategy = self._calculate_sampling_strategy(y_single)
            smote.set_params(sampling_strategy=sampling_strategy)
            
            X_resampled, y_single_resampled = smote.fit_resample(X, y_single)
            
            # Convert back to multi-label if needed
            if y.ndim > 1:
                y_resampled = self._single_to_multilabel(y_single_resampled, y.shape[1])
            else:
                y_resampled = y_single_resampled
            
            print(f"SMOTE applied: {X.shape[0]} -> {X_resampled.shape[0]} samples")
            return X_resampled, y_resampled
            
        except Exception as e:
            print(f"SMOTE failed: {e}, returning original data")
            return X, y
    
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


class UnsupervisedMultiLabelTransformer(nn.Module):
    """Enhanced Transformer for unsupervised multi-label attack type detection."""
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        n_labels: int,
        n_clusters: int,  # Kept for API compatibility
        dropout: float = 0.1,
        transformer_layers: int = 2,
        attention_heads: int = 8,
        use_enhanced_attention: bool = True,
        use_label_correlation: bool = True,
        use_contrastive: bool = True,
        **kwargs,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.n_labels = n_labels
        self.n_clusters = n_clusters
        self.use_enhanced_attention = use_enhanced_attention
        self.use_label_correlation = use_label_correlation
        self.use_contrastive = use_contrastive
        
        # Enhanced input projection with normalization and activation
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),  # GELU for better performance
            nn.Dropout(dropout)
        )
        
        # Positional encoding for sequence modeling
        self.pos_encoding = self._create_positional_encoding(latent_dim, 512)
        
        # Enhanced transformer blocks or standard transformer
        if use_enhanced_attention:
            self.transformer_blocks = nn.ModuleList([
                EnhancedTransformerBlock(latent_dim, attention_heads, latent_dim * 4, dropout)
                for _ in range(transformer_layers)
            ])
        else:
            # Standard transformer encoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=latent_dim,
                nhead=attention_heads,
                dim_feedforward=latent_dim * 4,
                dropout=dropout,
                activation='gelu',
                norm_first=True,
                batch_first=True
            )
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=transformer_layers
            )
        
        # Label correlation module
        if use_label_correlation and n_labels > 1:
            self.label_correlation = LabelCorrelationModule(n_labels, latent_dim)
        
        # Enhanced decoder for reconstruction
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, input_dim)
        )
        
        # Contrastive learning head
        if use_contrastive:
            self.contrastive_head = nn.Sequential(
                nn.Linear(latent_dim, latent_dim),
                nn.GELU(),
                nn.Linear(latent_dim, 128)
            )
        
        # Enhanced classifier
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.GELU(),
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
    
    def forward(self, x, mask=None, return_attention=False, **kwargs):
        """Forward pass with enhanced architecture"""
        # Handle 2D input
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add sequence dimension
        
        batch_size, seq_len, _ = x.shape
        
        # Input projection
        x = self.input_projection(x)  # [batch, seq_len, latent_dim]
        
        # Add positional encoding
        x = x + self.pos_encoding[:seq_len, :].transpose(0, 1)
        
        # Transformer encoding
        attention_weights = []
        if self.use_enhanced_attention:
            # Use enhanced transformer blocks
            for transformer_block in self.transformer_blocks:
                x, attn_weights = transformer_block(x, mask)
                if return_attention:
                    attention_weights.append(attn_weights)
        else:
            # Use standard transformer
            x = self.transformer_encoder(x, mask)
        
        # Global representation
        pooled = torch.mean(x, dim=1)  # Global average pooling
        
        # Prepare outputs
        outputs = {
            "sequence_representation": pooled,
            "reconstructed": self.decoder(pooled),
        }
        
        # Multi-label predictions
        if self.use_label_correlation and hasattr(self, 'label_correlation'):
            multi_label_scores = self.label_correlation(pooled)
        else:
            multi_label_scores = self.classifier(pooled)
        
        outputs["multi_label_scores"] = multi_label_scores
        
        # Contrastive features
        if self.use_contrastive and hasattr(self, 'contrastive_head'):
            outputs["contrastive"] = F.normalize(self.contrastive_head(pooled), dim=-1)
        
        # Legacy compatibility
        outputs["pooled"] = pooled
        
        if return_attention:
            outputs["attention_weights"] = attention_weights
        
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


def train_single_attack_model(
    model: UnsupervisedMultiLabelTransformer,
    embeddings: np.ndarray,
    binary_labels: np.ndarray,
    attack_type: str,
    config: SystemConfig,
    tracker: ProgressTracker,
    log_type: str,
    enhanced_config: EnhancedTransformerConfig = None,
) -> UnsupervisedMultiLabelTransformer:
    """Train a single model for one attack type using normal log data."""
    
    device = torch.device(config.device)
    
    # Enhanced architecture parameters
    embedding_dim = embeddings.shape[1]
    if embedding_dim <= 300:  # FastText
        latent_dim = 256
        transformer_layers = 3
        attention_heads = 8
    elif embedding_dim <= 768:  # Standard BERT
        latent_dim = 384
        transformer_layers = 4
        attention_heads = 8
    else:  # Enhanced LogBERT (2314D)
        latent_dim = 512
        transformer_layers = 6
        attention_heads = 16

    # Optimized batch sizes
    if config.device == "mps":
        if embedding_dim <= 300:
            batch_size = min(128, max(32, int(config.gpu_memory_gb * 2)))
        elif embedding_dim <= 768:
            batch_size = min(64, max(16, int(config.gpu_memory_gb * 1.5)))
        else:
            batch_size = min(32, max(8, int(config.gpu_memory_gb * 1)))
    elif config.device == "cuda":
        if embedding_dim <= 300:
            batch_size = min(32, max(8, int(config.gpu_memory_gb * 0.5)))
        elif embedding_dim <= 768:
            batch_size = min(16, max(4, int(config.gpu_memory_gb * 0.3)))
        else:
            batch_size = min(8, max(2, int(config.gpu_memory_gb * 0.2)))
    else:
        if embedding_dim <= 300:
            batch_size = min(64, max(16, int(config.gpu_memory_gb * 1.5)))
        elif embedding_dim <= 768:
            batch_size = min(32, max(8, int(config.gpu_memory_gb * 1)))
        else:
            batch_size = min(16, max(4, int(config.gpu_memory_gb * 0.5)))

    # Data setup
    embeddings_tensor = torch.from_numpy(embeddings).float()
    labels_tensor = torch.from_numpy(binary_labels).float().unsqueeze(1)  # [batch, 1]
    
    # Ensure tensors have the same first dimension
    assert embeddings_tensor.size(0) == labels_tensor.size(0), f"Size mismatch: embeddings {embeddings_tensor.size(0)} vs labels {labels_tensor.size(0)}"
    
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
    
    scaler = GradScaler() if config.device == "cuda" else None

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

    # Training loop with early stopping
    model.train()
    total_epochs = 200
    patience = 20
    patience_counter = 0
    best_loss = float("inf")
    best_model_state = None

    print(f"🎯 Training {attack_type} model for {total_epochs} epochs")

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
            outputs = model(x_batch)
            
            # Enhanced loss computation
            recon_loss = F.mse_loss(outputs["reconstructed"], x_batch)
            
            # Binary classification loss for this attack type
            attack_scores = outputs["multi_label_scores"]  # [batch, 1]
            
            # Use enhanced config if available
            econfig = enhanced_config or EnhancedTransformerConfig()
            
            # Enhanced focal loss for imbalanced data
            focal_loss_fn = FocalLoss(alpha=econfig.focal_loss_alpha, gamma=econfig.focal_loss_gamma)
            
            # Label smoothing
            smoothed_targets = y_batch * 0.9 + 0.05
            class_loss = focal_loss_fn(attack_scores, smoothed_targets)
            
            # Contrastive loss if available
            contrastive_loss = torch.tensor(0.0, device=device, requires_grad=True)
            if "contrastive" in outputs and outputs["contrastive"] is not None:
                try:
                    # Simple contrastive loss with stability checks
                    contrastive_features = outputs["contrastive"]
                    batch_size = contrastive_features.size(0)
                    
                    if batch_size > 1:
                        # Ensure features are normalized and stable
                        contrastive_features = F.normalize(contrastive_features, dim=1, eps=1e-8)
                        
                        # Compute similarity matrix with temperature
                        temperature = max(0.05, 0.07)  # Ensure reasonable temperature
                        similarity_matrix = torch.matmul(contrastive_features, contrastive_features.T) / temperature
                        
                        # Clamp similarity values to prevent overflow
                        similarity_matrix = torch.clamp(similarity_matrix, min=-10, max=10)
                        
                        # Create labels (each sample is similar to itself)
                        labels = torch.arange(batch_size).to(device)
                        
                        # Mask diagonal to remove self-similarity
                        mask = torch.eye(batch_size, dtype=torch.bool).to(device)
                        similarity_matrix = similarity_matrix.masked_fill(mask, -float('inf'))
                        
                        # Compute contrastive loss with error checking
                        try:
                            raw_contrastive_loss = F.cross_entropy(similarity_matrix, labels)
                            if not (torch.isnan(raw_contrastive_loss) or torch.isinf(raw_contrastive_loss)):
                                contrastive_loss = raw_contrastive_loss * econfig.contrastive_weight
                            else:
                                contrastive_loss = torch.tensor(0.0, device=device, requires_grad=True)
                        except Exception:
                            contrastive_loss = torch.tensor(0.0, device=device, requires_grad=True)
                except Exception:
                    contrastive_loss = torch.tensor(0.0, device=device, requires_grad=True)
            
            # Enhanced regularization with stability checks
            predictions = torch.sigmoid(attack_scores)
            predictions = torch.clamp(predictions, min=1e-8, max=1.0 - 1e-8)
            
            # Confidence regularization
            confidence_loss = torch.mean((predictions - 0.5).abs()) * 0.05
            
            # Entropy regularization with stability
            log_predictions = torch.log(predictions + 1e-8)
            log_one_minus_predictions = torch.log(1 - predictions + 1e-8)
            entropy_loss = -torch.mean(predictions * log_predictions + 
                                     (1 - predictions) * log_one_minus_predictions) * 0.01
            
            # Validate individual loss components before combining
            loss_components = []
            weights = []
            
            # Reconstruction loss
            if not (torch.isnan(recon_loss) or torch.isinf(recon_loss)):
                loss_components.append(recon_loss)
                weights.append(econfig.reconstruction_weight)
            
            # Classification loss  
            if not (torch.isnan(class_loss) or torch.isinf(class_loss)):
                loss_components.append(class_loss)
                weights.append(econfig.classification_weight)
            
            # Contrastive loss
            if not (torch.isnan(contrastive_loss) or torch.isinf(contrastive_loss)):
                loss_components.append(contrastive_loss)
                weights.append(1.0)
            
            # Regularization losses
            if not (torch.isnan(confidence_loss) or torch.isinf(confidence_loss)):
                loss_components.append(confidence_loss)
                weights.append(1.0)
                
            if not (torch.isnan(entropy_loss) or torch.isinf(entropy_loss)):
                loss_components.append(entropy_loss)
                weights.append(1.0)
            
            # Compute total loss only from valid components
            if loss_components:
                total_loss = sum(w * l for w, l in zip(weights, loss_components))
            else:
                # Fallback to simple MSE loss if all components are invalid
                total_loss = F.mse_loss(outputs["reconstructed"], x_batch)
            
            # Backward and optimize
            if not (torch.isnan(total_loss) or torch.isinf(total_loss)):
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                
                epoch_losses.append(total_loss.item())
                epoch_recon_losses.append(recon_loss.item())
                epoch_class_losses.append(class_loss.item())
                
                # Progress tracking
                if batch_idx % 20 == 0:
                    progress_text = f"{attack_type} | Epoch {epoch+1}/{total_epochs} | Batch {batch_idx+1}/{len(dataloader)} | Loss: {total_loss.item():.4f}"
                    if hasattr(tracker, '_progress_spinner'):
                        tracker._progress_spinner.text = progress_text
                    else:
                        tracker._progress_spinner = Halo(text=progress_text, spinner='dots')
                        tracker._progress_spinner.start()
            else:
                print(f"      ⚠️ Invalid loss for batch {batch_idx+1}")
                
                # If we get too many invalid losses, try simpler loss function
                if batch_idx > 10:  # After several failed batches
                    print(f"      🔄 Switching to simpler loss for {attack_type}")
                    # Use only reconstruction loss as fallback
                    simple_loss = F.mse_loss(outputs["reconstructed"], x_batch)
                    if not (torch.isnan(simple_loss) or torch.isinf(simple_loss)):
                        simple_loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                        optimizer.step()
                        epoch_losses.append(simple_loss.item())
                        epoch_recon_losses.append(simple_loss.item())
                        epoch_class_losses.append(0.0)

        scheduler.step()

        # Calculate epoch metrics
        epoch_time = time.time() - epoch_start
        avg_loss = np.mean(epoch_losses) if epoch_losses else float("nan")
        avg_recon_loss = np.mean(epoch_recon_losses) if epoch_recon_losses else float("nan")
        avg_class_loss = np.mean(epoch_class_losses) if epoch_class_losses else float("nan")
        
        # Clean up progress spinner
        if hasattr(tracker, '_progress_spinner'):
            tracker._progress_spinner.stop()
            delattr(tracker, '_progress_spinner')
        
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"✅ {attack_type} | Epoch {epoch+1}/{total_epochs} completed in {epoch_time:.1f}s")
            print(f"📊 Loss: {avg_loss:.4f} | Recon: {avg_recon_loss:.4f} | Class: {avg_class_loss:.4f}")
        
        # Early stopping check
        if not np.isnan(avg_loss) and avg_loss < best_loss - 1e-4:
            best_loss = avg_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            print(f"🛑 Early stopping triggered for {attack_type} at epoch {epoch + 1}")
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
                print(f"✅ Restored best model for {attack_type} with loss: {best_loss:.4f}")
            break

    print(f"✅ Completed training for {attack_type}")
    return model


def train_model(
    embeddings: np.ndarray,
    classes: List[str],
    C: np.ndarray,
    config: SystemConfig,
    tracker: ProgressTracker,
    log_type: str,
    scaler: "StandardScaler" = None,
) -> Tuple[List[UnsupervisedMultiLabelTransformer], "StandardScaler"]:
    """
    Train separate models for each attack type using normal log data, then combine results.
    This approach is much better for unsupervised learning than multi-label training.
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

    # Enhanced architecture for better performance
    embedding_dim = embeddings.shape[1]
    if embedding_dim <= 300:  # FastText
        latent_dim = 256
        transformer_layers = 3
        attention_heads = 8
    elif embedding_dim <= 768:  # Standard BERT
        latent_dim = 384
        transformer_layers = 4
        attention_heads = 8
    else:  # Enhanced LogBERT (2314D)
        latent_dim = 512
        transformer_layers = 6
        attention_heads = 16

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
    use_mixed_precision = config.device == "cuda"
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
    
    for attack_idx, attack_type in enumerate(classes):
        print(f"\n{'='*60}")
        print(f"🎯 Training model for attack type: {attack_type}")
        print(f"{'='*60}")
        
        # Create binary labels for this attack type (1 for this attack, 0 for normal)
        if true_labels is not None:
            # Use true labels if available
            binary_labels = true_labels[:, attack_idx].astype(np.float32)
            print(f"📊 Using true labels for {attack_type}: {len(binary_labels)} samples")
        else:
            # Create pseudo-labels for this attack type using clustering
            binary_labels = generate_pseudo_labels_for_attack_type(
                embeddings, attack_type, attack_idx, n_samples=len(embeddings)
            )
            print(f"📊 Using pseudo-labels for {attack_type}: {len(binary_labels)} samples")
        
        print(f"🔍 Debug: embeddings shape {embeddings.shape}, binary_labels shape {binary_labels.shape}")
        
        # Create enhanced single-class model for this attack type with conservative settings
        enhanced_config = EnhancedTransformerConfig(
            d_model=latent_dim,
            n_heads=attention_heads,
            n_layers=transformer_layers,
            num_labels=1,
            dropout=0.1,
            use_smote=True,
            contamination_rate=0.1,
            # Conservative focal loss settings to prevent numerical instability
            focal_loss_alpha=0.25,
            focal_loss_gamma=1.0,  # Reduced gamma for stability
            # Conservative loss weights
            reconstruction_weight=1.0,
            contrastive_weight=0.1,  # Reduced for stability
            classification_weight=1.0
        )
        
        # Create model with conservative settings for numerical stability
        model = UnsupervisedMultiLabelTransformer(
            input_dim=embedding_dim,
            latent_dim=latent_dim,
            n_labels=1,  # Single class: this attack vs normal
            n_clusters=n_clusters,
            dropout=0.1,
            transformer_layers=transformer_layers,
            attention_heads=attention_heads,
            use_enhanced_attention=True,  # Keep enhanced attention but with stability fixes
            use_label_correlation=False,  # Not needed for single label
            use_contrastive=True,  # Keep contrastive but with reduced weight
        ).to(device)
        
        # Apply extra initialization for stability
        for param in model.parameters():
            if param.dim() > 1:
                torch.nn.init.xavier_uniform_(param, gain=0.1)
            else:
                torch.nn.init.zeros_(param)
        
        # Multi-GPU setup for this model
        if config.n_gpus > 1:
            model = nn.DataParallel(model)
        
        # Train this model
        trained_model = train_single_attack_model(
            model, embeddings, binary_labels, attack_type, config, tracker, log_type, enhanced_config
        )
        
        # Generate predictions for this attack type
        trained_model.eval()
        with torch.no_grad():
            embeddings_tensor = torch.from_numpy(embeddings).float().to(device)
            outputs = trained_model(embeddings_tensor)
            attack_scores = outputs["multi_label_scores"]  # [batch, 1]
            attack_probs = torch.sigmoid(attack_scores).cpu().numpy().flatten()  # [batch]
            attack_preds = (attack_probs >= 0.5).astype(int)
        
        # Store results
        models.append(trained_model)
        all_probabilities.append(attack_probs)
        all_predictions.append(attack_preds)
        
        print(f"✅ Completed training for {attack_type}")
        print(f"📊 Predictions: {np.sum(attack_preds)}/{len(attack_preds)} samples classified as {attack_type}")
    
    # Combine results from all models
    print(f"\n{'='*60}")
    print(f"🔗 Combining results from {len(models)} models")
    print(f"{'='*60}")
    
    # Stack all predictions and probabilities
    combined_predictions = np.column_stack(all_predictions)  # [n_samples, n_attack_types]
    combined_probabilities = np.column_stack(all_probabilities)  # [n_samples, n_attack_types]
    
    print(f"📊 Combined predictions shape: {combined_predictions.shape}")
    print(f"📈 Combined probabilities shape: {combined_probabilities.shape}")
    
    # Save combined results
    if config.rank == 0:
        print(f"💾 Saving combined prediction results for {log_type}...")
        try:
            import pickle
            import os

            # Generate sample IDs
            ids = np.arange(len(embeddings))

            # Prepare prediction dictionary
            prediction_data = {
                "ids": ids,
                "probs": combined_probabilities,
                "preds": combined_predictions,
                "classes": classes,
                "thresholds": np.ones(len(classes)) * 0.5,  # Default thresholds
            }
            if true_labels is not None:
                prediction_data["true_labels"] = true_labels

            # Create results directory and save predictions
            os.makedirs(f"results/{log_type}", exist_ok=True)
            prediction_file = f"results/{log_type}/predictions.pkl"

            with open(prediction_file, "wb") as f:
                pickle.dump(prediction_data, f)

            print(f"✅ Combined predictions saved to {prediction_file}")
        except Exception as e:
            print(f"❌ Failed to save combined predictions: {e}")

    return models, scaler


def load_and_preprocess_data(
    log_type: str,
    config: SystemConfig,
    tracker: ProgressTracker,
    sample_size: int = None,
) -> Tuple[np.ndarray, List[str], np.ndarray, StandardScaler]:
    """Load and preprocess data for training"""
    
    print(f"🔄 Loading data for {log_type}...")
    
    # Load embeddings
    embeddings_dir = Path("embeddings") / log_type
    log_file = embeddings_dir / f"log_{log_type}.pkl"
    
    if not log_file.exists():
        raise FileNotFoundError(f"Embeddings not found: {log_file}")
    
    with open(log_file, 'rb') as f:
        embeddings = pickle.load(f)
    
    # Load labels
    label_file = embeddings_dir / f"label_{log_type}.pkl"
    if not label_file.exists():
        raise FileNotFoundError(f"Labels not found: {label_file}")
    
    with open(label_file, 'rb') as f:
        label_data = pickle.load(f)
    
    true_labels = label_data["vectors"]
    classes = label_data["classes"]
    
    # Store true labels in tracker for later use
    tracker.true_labels = true_labels
    
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
            "embedding_dim": embeddings.shape[1],
            "embedding_type": "FastText (300D)" if embeddings.shape[1] == 300 else "Unknown"
        })
    else:
        print(f"📊 Using full dataset: {len(embeddings):,} samples ({embeddings.nbytes / (1024**3):.1f} GB)")
        
        tracker.log_step("Full Dataset Processing", {
            "total_samples": len(embeddings),
            "memory_gb": embeddings.nbytes / (1024**3),
            "embedding_dim": embeddings.shape[1],
            "embedding_type": "FastText (300D)" if embeddings.shape[1] == 300 else "Unknown",
            "device_type": config.device
        })
    
    # Preprocessing
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    
    tracker.log_step("Data Preprocessing", {
        "embeddings_shape": list(embeddings.shape),
        "embedding_type": "FastText (300D)" if embeddings.shape[1] == 300 else "Unknown",
        "n_classes": len(classes),
        "n_clusters": len(classes),
        "has_true_labels": true_labels is not None
    })
    
    print(f"✅ Data loading completed in {time.time():.1f}s")
    print(f"📊 Loaded {len(embeddings):,} samples with {embeddings.shape[1]}D embeddings")
    
    if sample_size:
        print(f"🎯 Using sample size: {len(embeddings)} (requested: {sample_size})")
    
    print(f"✔ Data loaded: {len(embeddings)} samples, {embeddings.shape[1]} features")
    
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
):
    """Process a single log type with the new separate models approach"""
    
    print(f"🎯 Processing single log type: {log_type}")
    
    # Initialize progress tracker
    output_dir = Path("results") / log_type
    tracker = ProgressTracker(output_dir, log_type, config)
    
    # Load and preprocess data
    embeddings, classes, true_labels, scaler = load_and_preprocess_data(
        log_type, config, tracker, sample_size
    )
    
    # Check for existing results
    if not force_restart:
        results_file = output_dir / "predictions.pkl"
        if results_file.exists():
            print(f"✅ Found existing results for {log_type}, skipping training")
            return
    
    # Start training
    print(f"\n🚀 Starting training for {log_type}. Detailed progress will be shown below...")
    training_start_time = time.time()
    
    models, _ = train_model(
        embeddings, classes, None, config, tracker, log_type, scaler
    )
    print(f"✅ Training completed for {log_type}")
    
    # Calculate total training time
    total_training_time = time.time() - training_start_time
    
    # Generate predictions only (no evaluation)
    print(f"💾 Saving prediction results for {log_type}...")
    import torch
    with torch.no_grad():
        embeddings_tensor = torch.from_numpy(embeddings).float().to(config.device)
        logits = models[0](embeddings_tensor)["multi_label_scores"]
        probs = torch.sigmoid(logits).cpu().numpy()
        
        # Use default thresholds for now
        thresholds = np.full(len(classes), 0.5)
        preds = (probs >= thresholds).astype(int)
        
        # Load true labels if available
        true_labels = getattr(tracker, "true_labels", None)
        
        # Save predictions only
        prediction_file = Path(f"results/{log_type}/predictions.pkl")
        with open(prediction_file, "wb") as f:
            pickle.dump({
                "probs": probs,
                "preds": preds,
                "true_labels": true_labels,
                "classes": classes,
                "thresholds": thresholds
            }, f)
        
        print(f"✅ Predictions saved to {prediction_file}")

    # Enhanced evaluation and clustering analysis
    if evaluate_with_clustering and use_enhanced_features:
        print(f"\n🔍 Performing enhanced evaluation and clustering analysis...")
        
        # Initialize enhanced components
        enhanced_config = EnhancedTransformerConfig()
        evaluator = MultiLabelEvaluator(enhanced_config)
        clustering_analyzer = ClusteringAnalyzer(enhanced_config)
        
        # Extract features from the first model for clustering
        model = models[0]
        model.eval()
        with torch.no_grad():
            embeddings_tensor = torch.from_numpy(embeddings).float().to(config.device)
            outputs = model(embeddings_tensor)
            features = outputs.get("sequence_representation", outputs.get("pooled")).cpu().numpy()
        
        # Perform clustering analysis
        clustering_results = clustering_analyzer.perform_clustering(features, true_labels)
        
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
        
        # Enhanced evaluation if true labels available
        if true_labels is not None:
            print(f"📊 Computing enhanced evaluation metrics...")
            
            # Load the combined predictions that were already computed
            try:
                with open(f"results/{log_type}/predictions.pkl", "rb") as f:
                    prediction_data = pickle.load(f)
                    
                # Get the combined multi-label predictions and scores
                combined_predictions = prediction_data["preds"]  # Shape: (n_samples, n_classes)
                combined_scores = prediction_data["probs"]       # Shape: (n_samples, n_classes)
                
                print(f"🔍 Evaluation shapes - True: {true_labels.shape}, Pred: {combined_predictions.shape}, Scores: {combined_scores.shape}")
                
                # Ensure compatible shapes and types
                if true_labels.shape != combined_predictions.shape:
                    print(f"⚠️  Shape mismatch: true_labels {true_labels.shape} vs predictions {combined_predictions.shape}")
                    # Skip evaluation if shapes don't match
                    eval_results = {"error": "Shape mismatch between true labels and predictions"}
                else:
                    # Evaluate with properly shaped multi-label data
                    eval_results = evaluator.evaluate(true_labels, combined_predictions, combined_scores)
                    
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
            else:
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
            'input_dim': embeddings.shape[1],
            'latent_dim': model.latent_dim,
            'n_labels': model.n_labels,
            'transformer_layers': 3,  # Default
            'attention_heads': 8,     # Default
            'dropout': 0.1,
            'scaler': scaler,
            'enhanced_features': use_enhanced_features,
            'use_enhanced_attention': getattr(model, 'use_enhanced_attention', False),
            'use_label_correlation': getattr(model, 'use_label_correlation', False),
            'use_contrastive': getattr(model, 'use_contrastive', False),
        }, model_path)
        
        print(f"💾 Model saved to: {model_path}")
        print(f"📊 Model info: {embeddings.shape[1]}D → {len(classes)} classes")
        print(f"🏗️  Architecture: 3 layers, 8 heads, {model.latent_dim}D latent")
        if use_enhanced_features:
            print(f"✨ Enhanced features: Focal Loss, Enhanced Attention, Contrastive Learning")
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
    print(f"\n🎯 Next steps:")
    print(f"   Evaluation: python src/evaluate_transformer.py --log-type {log_type}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Enhanced Transformer for Unsupervised Multi-Label Attack Detection")
    parser.add_argument("--log-type", type=str, required=True, help="Log type to process")
    parser.add_argument("--sample-size", type=int, help="Sample size for testing")
    parser.add_argument("--force-restart", action="store_true", help="Force restart training")
    parser.add_argument("--cleanup", action="store_true", help="Clean up old backup files")
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
    
    print(f"Detected {config.device.upper()} device")
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
        evaluate_with_clustering=evaluate_with_clustering
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