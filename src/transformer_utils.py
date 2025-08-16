#!/usr/bin/env python3
"""
Transformer Utilities Module
============================

This module contains utility functions for the optimized transformer implementation,
making the main code more manageable and reducing duplication.
"""

import time
import torch
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass


@dataclass
class ModelArchitectureConfig:
    """Configuration for model architecture based on embedding dimensions."""
    latent_dim: int
    transformer_layers: int
    attention_heads: int
    batch_size_multiplier: float = 1.0


class ArchitectureOptimizer:
    """Optimizes model architecture parameters based on input dimensions and device."""
    
    @staticmethod
    def get_optimized_config(embedding_dim: int, device_type: str, gpu_memory_gb: float) -> ModelArchitectureConfig:
        """
        Get optimized architecture configuration based on embedding dimensions and device.
        
        Args:
            embedding_dim: Dimension of input embeddings
            device_type: Type of device ('cuda', 'mps', 'cpu')
            gpu_memory_gb: Available GPU memory in GB
            
        Returns:
            ModelArchitectureConfig with optimized parameters
        """
        # Base configurations optimized for performance
        if embedding_dim <= 300:  # FastText
            base_config = ModelArchitectureConfig(
                latent_dim=128,
                transformer_layers=2,
                attention_heads=4
            )
        elif embedding_dim <= 768:  # Standard BERT
            base_config = ModelArchitectureConfig(
                latent_dim=256,
                transformer_layers=2,
                attention_heads=4
            )
        else:  # Enhanced LogBERT (2314D)
            base_config = ModelArchitectureConfig(
                latent_dim=384,
                transformer_layers=3,
                attention_heads=8
            )
        
        # Adjust batch size multiplier based on device and memory
        if device_type == "mps":
            base_config.batch_size_multiplier = min(2.0, gpu_memory_gb / 8.0)
        elif device_type == "cuda":
            base_config.batch_size_multiplier = min(1.5, gpu_memory_gb / 16.0)
        else:  # CPU
            base_config.batch_size_multiplier = 0.5
        
        return base_config


class PerformanceMonitor:
    """Monitors and reports training performance metrics."""
    
    def __init__(self):
        self.start_time = None
        self.epoch_times = []
        self.loss_history = []
        
    def start_training(self):
        """Start training timer."""
        self.start_time = time.time()
        
    def record_epoch(self, epoch: int, loss: float, epoch_time: float):
        """Record epoch metrics."""
        self.epoch_times.append(epoch_time)
        self.loss_history.append(loss)
        
        # Print progress less frequently for performance
        if epoch % 10 == 0 or loss < min(self.loss_history):
            avg_epoch_time = np.mean(self.epoch_times[-10:])  # Last 10 epochs
            print(f"  Epoch {epoch}: Loss {loss:.4f}, Time {epoch_time:.1f}s (avg: {avg_epoch_time:.1f}s)")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get training summary."""
        total_time = time.time() - self.start_time if self.start_time else 0
        return {
            "total_time": total_time,
            "total_epochs": len(self.epoch_times),
            "avg_epoch_time": np.mean(self.epoch_times) if self.epoch_times else 0,
            "best_loss": min(self.loss_history) if self.loss_history else float('inf'),
            "final_loss": self.loss_history[-1] if self.loss_history else float('inf')
        }


class MemoryManager:
    """Manages memory usage and cleanup during training."""
    
    @staticmethod
    def clear_gpu_memory():
        """Clear GPU memory cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    @staticmethod
    def get_memory_usage() -> Dict[str, float]:
        """Get current memory usage."""
        memory_info = {}
        
        if torch.cuda.is_available():
            memory_info['gpu_allocated'] = torch.cuda.memory_allocated() / 1024**3
            memory_info['gpu_cached'] = torch.cuda.memory_reserved() / 1024**3
        
        # Could add CPU memory monitoring here
        return memory_info
    
    @staticmethod
    def optimize_batch_size(base_batch_size: int, embedding_dim: int, device_type: str, memory_gb: float) -> int:
        """Optimize batch size based on available memory."""
        # Memory usage estimates (empirical)
        memory_per_sample = embedding_dim * 4 / 1024**3  # rough estimate in GB
        
        if device_type == "mps":
            # MPS has different memory characteristics
            max_batch_size = int(memory_gb * 0.7 / memory_per_sample)
        elif device_type == "cuda":
            max_batch_size = int(memory_gb * 0.8 / memory_per_sample)
        else:
            max_batch_size = int(memory_gb * 0.5 / memory_per_sample)
        
        return min(base_batch_size, max(8, max_batch_size))


class DataProcessor:
    """Processes data for training with optimizations."""
    
    @staticmethod
    def prepare_tensors(embeddings: np.ndarray, labels: np.ndarray, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare tensors for training with memory optimization."""
        # Convert to tensors efficiently
        embeddings_tensor = torch.from_numpy(embeddings).float()
        labels_tensor = torch.from_numpy(labels).float()
        
        # Ensure correct dimensions
        if labels_tensor.dim() == 1:
            labels_tensor = labels_tensor.unsqueeze(1)
        
        # Move to device
        embeddings_tensor = embeddings_tensor.to(device, non_blocking=True)
        labels_tensor = labels_tensor.to(device, non_blocking=True)
        
        return embeddings_tensor, labels_tensor
    
    @staticmethod
    def validate_data(embeddings: np.ndarray, labels: np.ndarray) -> bool:
        """Validate data dimensions and content."""
        if embeddings.shape[0] != labels.shape[0]:
            print(f"❌ Data size mismatch: {embeddings.shape[0]} vs {labels.shape[0]}")
            return False
        
        if np.isnan(embeddings).any() or np.isnan(labels).any():
            print("❌ Data contains NaN values")
            return False
        
        if np.isinf(embeddings).any() or np.isinf(labels).any():
            print("❌ Data contains infinite values")
            return False
        
        return True


class LossOptimizer:
    """Optimizes loss computation for better performance."""
    
    @staticmethod
    def compute_simple_loss(outputs: Dict[str, torch.Tensor], targets: torch.Tensor, 
                          recon_weight: float = 0.5, class_weight: float = 1.0) -> torch.Tensor:
        """Compute simplified loss for better performance."""
        # Reconstruction loss
        recon_loss = torch.nn.functional.mse_loss(outputs["reconstructed"], targets)
        
        # Classification loss
        class_scores = outputs["multi_label_scores"]
        class_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            class_scores, targets[:, :class_scores.size(1)] if targets.size(1) > class_scores.size(1) else targets
        )
        
        # Combine losses
        total_loss = recon_weight * recon_loss + class_weight * class_loss
        
        return total_loss
    
    @staticmethod
    def compute_focal_loss(outputs: Dict[str, torch.Tensor], targets: torch.Tensor,
                          alpha: float = 0.25, gamma: float = 1.5,
                          recon_weight: float = 0.5, class_weight: float = 1.0) -> torch.Tensor:
        """Compute focal loss with simplified implementation."""
        # Reconstruction loss
        recon_loss = torch.nn.functional.mse_loss(outputs["reconstructed"], targets)
        
        # Simplified focal loss
        class_scores = outputs["multi_label_scores"]
        class_targets = targets[:, :class_scores.size(1)] if targets.size(1) > class_scores.size(1) else targets
        
        # Binary cross entropy
        bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            class_scores, class_targets, reduction='none'
        )
        
        # Focal weight
        probs = torch.sigmoid(class_scores)
        p_t = probs * class_targets + (1 - probs) * (1 - class_targets)
        focal_weight = alpha * (1 - p_t) ** gamma
        
        focal_loss = (focal_weight * bce_loss).mean()
        
        # Combine losses
        total_loss = recon_weight * recon_loss + class_weight * focal_loss
        
        return total_loss


def format_training_summary(summary: Dict[str, Any], attack_type: str) -> str:
    """Format training summary for display."""
    lines = [
        f"✅ {attack_type} Training Complete",
        f"   Total time: {summary['total_time']:.1f}s",
        f"   Epochs: {summary['total_epochs']}",
        f"   Avg epoch time: {summary['avg_epoch_time']:.1f}s",
        f"   Best loss: {summary['best_loss']:.4f}",
        f"   Final loss: {summary['final_loss']:.4f}"
    ]
    return "\n".join(lines)


def estimate_training_time(n_samples: int, n_classes: int, embedding_dim: int, device_type: str) -> str:
    """Estimate training time based on dataset characteristics."""
    # Empirical estimates (in seconds per epoch per model)
    base_time_per_epoch = {
        'cuda': 0.1,
        'mps': 0.15,
        'cpu': 0.5
    }.get(device_type, 0.5)
    
    # Scale by data size and complexity
    time_per_epoch = base_time_per_epoch * (n_samples / 1000) * (embedding_dim / 300)
    estimated_epochs = 30  # Average epochs with early stopping
    
    total_time = time_per_epoch * estimated_epochs * (n_classes + 1)  # +1 for normal class
    
    if total_time < 60:
        return f"{total_time:.0f} seconds"
    elif total_time < 3600:
        return f"{total_time/60:.1f} minutes"
    else:
        return f"{total_time/3600:.1f} hours"
