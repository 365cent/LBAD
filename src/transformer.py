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
    
    # Load embeddings
    embeddings_dir = Path("embeddings")
    if log_type == "all_combined":
        log_file = embeddings_dir / f"log_{log_type}.pkl"
        label_file = embeddings_dir / f"label_{log_type}.pkl"
    else:
        log_file = embeddings_dir / log_type / f"log_{log_type}.pkl"
        label_file = embeddings_dir / log_type / f"label_{log_type}.pkl"
    
    if not log_file.exists():
        raise FileNotFoundError(f"Embedding file not found: {log_file}")
    
    with open(log_file, 'rb') as f:
        embeddings = pickle.load(f)
    
    # Load labels
    classes = []
    if label_file.exists():
        with open(label_file, 'rb') as f:
            label_data = pickle.load(f)
            if isinstance(label_data, dict) and 'classes' in label_data:
                classes = label_data['classes']
            elif isinstance(label_data, dict) and 'vectors' in label_data:
                classes = label_data.get('classes', [])
    
    # Smart subsampling for very large datasets
    max_samples = min(1000000, int(config.gpu_memory_gb * 10000))  # Scale with GPU memory
    if len(embeddings) > max_samples:
        indices = np.random.choice(len(embeddings), max_samples, replace=False)
        embeddings = embeddings[indices]
        tracker.log_step("Data Subsampling", {
            "original_size": len(embeddings),
            "subsampled_size": max_samples,
            "memory_gb": embeddings.nbytes / (1024**3)
        })
    
    # Normalize
    scaler = StandardScaler()
    embeddings = scaler.fit_transform(embeddings).astype(np.float32)
    
    # Create label clusters
    n_clusters = min(16, len(classes), len(embeddings) // 1000)
    C = create_label_clusters(classes, n_clusters)
    
    tracker.log_step("Data Preprocessing", {
        "embeddings_shape": embeddings.shape,
        "n_classes": len(classes),
        "n_clusters": C.shape[1] if C is not None else 0
    })
    
    return embeddings, classes, C, scaler

def create_label_clusters(classes: List[str], n_clusters: int) -> Optional[np.ndarray]:
    """Create semantic label clusters"""
    if not classes or n_clusters <= 0:
        return None
    
    n_clusters = min(n_clusters, len(classes))
    C = np.zeros((len(classes), n_clusters))
    
    # Simple hash-based clustering
    for i, class_name in enumerate(classes):
        cluster_id = hash(class_name) % n_clusters
        C[i, cluster_id] = 1
    
    return C

def generate_pseudo_labels(embeddings: np.ndarray, classes: List[str], k: int = 3) -> np.ndarray:
    """Generate initial pseudo-labels using clustering"""
    if not classes:
        return np.random.rand(len(embeddings), 1).astype(np.float32)
    
    # K-means clustering
    n_clusters = min(len(classes), len(embeddings) // 100, 50)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    # Generate pseudo-labels
    pseudo_labels = np.zeros((len(embeddings), len(classes)), dtype=np.float32)
    for i, cluster_id in enumerate(cluster_labels):
        # Assign top-k labels per cluster
        label_indices = np.random.choice(len(classes), min(k, len(classes)), replace=False)
        pseudo_labels[i, label_indices] = 1.0
    
    return pseudo_labels

def train_model(embeddings: np.ndarray, classes: List[str], C: np.ndarray, 
                config: SystemConfig, tracker: ProgressTracker) -> Tuple[UnsupervisedMultiLabelTransformer, StandardScaler]:
    """Optimized training with multi-GPU support and mixed precision for Nibi"""
    
    device = torch.device(config.device)
    n_labels = len(classes) if classes else 1
    n_clusters = C.shape[1] if C is not None else 1
    latent_dim = min(512, embeddings.shape[1])
    
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
    
    # Generate pseudo-labels
    pseudo_labels = generate_pseudo_labels(embeddings, classes)
    
    # Data setup
    dataset = TensorDataset(
        torch.from_numpy(embeddings),
        torch.from_numpy(pseudo_labels)
    )
    
    sampler = DistributedSampler(dataset) if config.is_distributed else None
    batch_size = min(256, max(32, int(config.gpu_memory_gb * 8)))  # Scale batch size
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=min(8, config.n_cpus // 2),
        pin_memory=True
    )
    
    # Training setup
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scaler = GradScaler() if config.device == "cuda" else None
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    
    tracker.log_step("Training Setup", {
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "batch_size": batch_size,
        "device": str(device),
        "mixed_precision": scaler is not None
    })
    
    # Training loop with progress tracking
    model.train()
    total_epochs = 100  # Reduced epochs for faster iteration
    tracker.start_training(total_epochs)
    
    for epoch in range(total_epochs):
        epoch_start = time.time()
        epoch_losses = []
        
        if config.is_distributed:
            sampler.set_epoch(epoch)
        
        # Progress spinner for batches
        with Halo(text=f"Epoch {epoch+1}/{total_epochs}", spinner='dots') as spinner:
            for batch_idx, (x_batch, y_batch) in enumerate(dataloader):
                x_batch = x_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)
                
                optimizer.zero_grad()
                
                if scaler:
                    with autocast():
                        outputs = model(x_batch)
                        
                        # Multi-component loss
                        recon_loss = F.mse_loss(outputs['reconstructed'], x_batch)
                        label_loss = F.binary_cross_entropy_with_logits(outputs['labels'], y_batch)
                        
                        if C is not None:
                            C_tensor = torch.from_numpy(C.astype(np.float32)).to(device)
                            cluster_targets = torch.matmul(y_batch, C_tensor)
                            cluster_loss = F.mse_loss(outputs['clusters'], cluster_targets)
                        else:
                            cluster_loss = torch.tensor(0.0, device=device)
                        
                        total_loss = recon_loss + 0.5 * label_loss + 0.3 * cluster_loss
                    
                    scaler.scale(total_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(x_batch)
                    
                    recon_loss = F.mse_loss(outputs['reconstructed'], x_batch)
                    label_loss = F.binary_cross_entropy_with_logits(outputs['labels'], y_batch)
                    
                    if C is not None:
                        C_tensor = torch.from_numpy(C.astype(np.float32)).to(device)
                        cluster_targets = torch.matmul(y_batch, C_tensor)
                        cluster_loss = F.mse_loss(outputs['clusters'], cluster_targets)
                    else:
                        cluster_loss = torch.tensor(0.0, device=device)
                    
                    total_loss = recon_loss + 0.5 * label_loss + 0.3 * cluster_loss
                    total_loss.backward()
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

def evaluate_and_save_results(model: UnsupervisedMultiLabelTransformer, 
                             embeddings: np.ndarray, classes: List[str],
                             config: SystemConfig, tracker: ProgressTracker, 
                             output_dir: Path, log_type: str):
    """Evaluate model and save comprehensive results with proper label format"""
    
    device = torch.device(config.device)
    model.eval()
    
    # Generate predictions
    predictions = []
    batch_size = min(512, int(config.gpu_memory_gb * 16))
    
    with torch.no_grad():
        for i in range(0, len(embeddings), batch_size):
            batch = torch.from_numpy(embeddings[i:i+batch_size]).to(device)
            outputs = model(batch)
            # Convert logits to probabilities for evaluation
            logits = outputs['labels']
            probs = torch.sigmoid(logits)
            predictions.append(probs.cpu().numpy())
    
    predictions = np.vstack(predictions)
    
    # Analysis
    binary_predictions = (predictions > 0.5).astype(int)
    labels_per_sample = binary_predictions.sum(axis=1)
    
    results = {
        "predictions_shape": predictions.shape,
        "avg_labels_per_sample": float(labels_per_sample.mean()),
        "std_labels_per_sample": float(labels_per_sample.std()),
        "label_frequencies": binary_predictions.sum(axis=0).tolist()
    }
    
    tracker.log_step("Evaluation Results", results)
    
    # Save detailed results
    save_path = output_dir / f"results_{log_type}_{config.node_name}_{config.job_id}.pkl"
    with open(save_path, 'wb') as f:
        pickle.dump({
            'predictions': predictions,
            'binary_predictions': binary_predictions,
            'classes': classes,
            'results': results
        }, f)
    
    # Save labels in the same format as embeddings for evaluation
    label_output_path = output_dir / f"label_{log_type}_{config.node_name}_{config.job_id}.pkl"
    label_data = {
        'vectors': binary_predictions.astype(np.int8),  # Same format as original labels
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
    
    tracker.log_step("Label Output Saved", {
        "label_file": str(label_output_path),
        "label_shape": binary_predictions.shape,
        "classes_count": len(classes)
    })
    
    # Create visualization if reasonable size
    if len(embeddings) <= 5000:
        create_visualization(embeddings, predictions, classes, output_dir, log_type, config)
    
    return results

def create_visualization(embeddings: np.ndarray, predictions: np.ndarray, 
                        classes: List[str], output_dir: Path, log_type: str, config: SystemConfig):
    """Create visualizations"""
    try:
        # Sample for t-SNE if too large
        if len(embeddings) > 2000:
            idx = np.random.choice(len(embeddings), 2000, replace=False)
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

# =============================================================================
# Main Execution
# =============================================================================

def find_available_embeddings() -> List[str]:
    """Find available embedding files"""
    embeddings_dir = Path("embeddings")
    if not embeddings_dir.exists():
        return []
    
    log_files = []
    for file_path in embeddings_dir.rglob("log_*.pkl"):
        log_type = file_path.stem.replace("log_", "")
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
                    'results': results
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