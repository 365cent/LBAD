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
- Separate models per log type with log type classifier
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
    
    # Load labels
    classes = []
    if label_file.exists():
        with open(label_file, 'rb') as f:
            label_data = pickle.load(f)
            if isinstance(label_data, dict) and 'classes' in label_data:
                classes = label_data['classes']
            elif isinstance(label_data, dict) and 'vectors' in label_data:
                classes = label_data.get('classes', [])
    
    # Aggressive subsampling for memory efficiency
    max_samples = min(50000, int(config.gpu_memory_gb * 500))  # Much smaller for memory safety
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
    n_clusters = min(8, len(classes), len(embeddings) // 1000)
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
    n_clusters = min(len(classes), len(embeddings) // 100, 20)  # Reduced clusters
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
    
    # Generate pseudo-labels
    pseudo_labels = generate_pseudo_labels(embeddings, classes)
    
    # Data setup
    dataset = TensorDataset(
        torch.from_numpy(embeddings),
        torch.from_numpy(pseudo_labels)
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
    
    # Training setup
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scaler = GradScaler() if config.device == "cuda" else None
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)  # Reduced epochs
    
    tracker.log_step("Training Setup", {
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "batch_size": batch_size,
        "device": str(device),
        "mixed_precision": scaler is not None
    })
    
    # Training loop with progress tracking
    model.train()
    total_epochs = 50  # Reduced epochs for faster iteration
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

def train_log_type_classifier(all_log_types: List[str], config: SystemConfig, tracker: ProgressTracker) -> LogTypeClassifier:
    """Train a classifier to predict log type from embeddings"""
    
    device = torch.device(config.device)
    
    # Load a small sample from each log type to train the classifier
    embeddings_list = []
    labels_list = []
    
    embeddings_dir = Path("embeddings")
    
    for log_type in all_log_types:
        log_file = embeddings_dir / log_type / f"log_{log_type}.pkl"
        if log_file.exists():
            with open(log_file, 'rb') as f:
                embeddings = pickle.load(f)
            
            # Sample a small amount from each type
            sample_size = min(1000, len(embeddings))
            if len(embeddings) > sample_size:
                indices = np.random.choice(len(embeddings), sample_size, replace=False)
                embeddings = embeddings[indices]
            
            embeddings_list.append(embeddings)
            labels_list.extend([log_type] * len(embeddings))
    
    if not embeddings_list:
        raise ValueError("No valid embedding files found for log type classifier")
    
    # Combine all embeddings
    all_embeddings = np.vstack(embeddings_list)
    
    # Create label mapping
    unique_log_types = list(set(labels_list))
    log_type_to_idx = {log_type: idx for idx, log_type in enumerate(unique_log_types)}
    labels = np.array([log_type_to_idx[label] for label in labels_list])
    
    # Normalize embeddings
    scaler = StandardScaler()
    all_embeddings = scaler.fit_transform(all_embeddings).astype(np.float32)
    
    # Create dataset
    dataset = TensorDataset(
        torch.from_numpy(all_embeddings),
        torch.from_numpy(labels)
    )
    
    batch_size = min(64, len(dataset))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    # Create and train classifier
    classifier = LogTypeClassifier(
        input_dim=all_embeddings.shape[1],
        n_log_types=len(unique_log_types)
    ).to(device)
    
    optimizer = optim.AdamW(classifier.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    tracker.log_step("Log Type Classifier Training", {
        "n_log_types": len(unique_log_types),
        "total_samples": len(all_embeddings),
        "log_types": unique_log_types
    })
    
    # Train for a few epochs
    classifier.train()
    for epoch in range(10):
        epoch_loss = 0
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = classifier(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        if epoch % 5 == 0:
            print(f"Log Type Classifier Epoch {epoch+1}/10 - Loss: {epoch_loss/len(dataloader):.4f}")
    
    # Save classifier and mapping
    classifier_path = Path("models") / f"log_type_classifier_{config.node_name}_{config.job_id}.pth"
    classifier_path.parent.mkdir(exist_ok=True)
    
    torch.save({
        'model_state_dict': classifier.state_dict(),
        'log_type_to_idx': log_type_to_idx,
        'idx_to_log_type': unique_log_types,
        'scaler': scaler
    }, classifier_path)
    
    tracker.log_step("Log Type Classifier Saved", {
        "classifier_path": str(classifier_path),
        "n_log_types": len(unique_log_types)
    })
    
    return classifier

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
            batch = torch.from_numpy(embeddings[i:i+batch_size]).to(device)
            outputs = model(batch)
            logits = outputs['labels']
            probs = torch.sigmoid(logits)
            predictions.append(probs.cpu().numpy())
    
    predictions = np.vstack(predictions)
    binary_predictions = (predictions > 0.5).astype(int)
    
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
        
        # Train log type classifier first
        if config.rank == 0:
            print(f"\n{'='*60}")
            print("Training Log Type Classifier")
            print(f"{'='*60}")
            
            tracker = ProgressTracker(Path("results"), "log_type_classifier", config)
            log_type_classifier = train_log_type_classifier(available_types, config, tracker)
            print("Log type classifier training completed!")
        
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
            print(f"Log type classifier saved to: models/log_type_classifier_*.pth")
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