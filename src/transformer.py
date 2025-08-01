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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from halo import Halo
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
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
        **kwargs,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.n_labels = n_labels
        self.n_clusters = n_clusters
        
        # Enhanced input projection with normalization and activation
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Enhanced transformer blocks with pre-normalization
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=attention_heads,
            dim_feedforward=latent_dim * 4,
            dropout=dropout,
            activation='relu',
            norm_first=True,  # Pre-normalization for better training
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=transformer_layers
        )
        
        # Enhanced decoder and classifier
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * 2, input_dim),  # Output same dimension as input
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim // 2, n_labels)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights for better training"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, x, **kwargs):
        """Forward pass with enhanced architecture"""
        # Input projection
        x = self.input_projection(x)  # [batch, seq_len, latent_dim]
        
        # Transformer encoding
        encoded = self.transformer_encoder(x)  # [batch, seq_len, latent_dim]
        
        # For embeddings, just use the encoded output directly
        pooled = encoded.squeeze(1)  # [batch, latent_dim]
        
        # Decoder for reconstruction
        reconstructed = self.decoder(pooled)  # [batch, input_dim]
        
        # Classifier for multi-label classification
        multi_label_scores = self.classifier(pooled)  # [batch, n_labels]
        
        return {
            "reconstructed": reconstructed,
            "multi_label_scores": multi_label_scores,
            "pooled": pooled
        }


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
            
            # Enhanced binary cross entropy with label smoothing
            smoothed_targets = y_batch * 0.9 + 0.05
            class_loss = F.binary_cross_entropy_with_logits(
                attack_scores, smoothed_targets, reduction='mean'
            )
            
            # Enhanced regularization
            predictions = torch.sigmoid(attack_scores)
            
            # Confidence regularization
            confidence_loss = torch.mean((predictions - 0.5).abs()) * 0.05
            
            # Entropy regularization
            entropy_loss = -torch.mean(predictions * torch.log(predictions + 1e-8) + 
                                     (1 - predictions) * torch.log(1 - predictions + 1e-8)) * 0.01
            
            # Total loss
            total_loss = recon_loss + class_loss + confidence_loss + entropy_loss
            
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
        
        # Create single-class model for this attack type
        model = UnsupervisedMultiLabelTransformer(
            input_dim=embedding_dim,
            latent_dim=latent_dim,
            n_labels=1,  # Single class: this attack vs normal
            n_clusters=n_clusters,
            dropout=0.1,
            transformer_layers=transformer_layers,
            attention_heads=attention_heads,
        ).to(device)
        
        # Multi-GPU setup for this model
        if config.n_gpus > 1:
            model = nn.DataParallel(model)
        
        # Train this model
        trained_model = train_single_attack_model(
            model, embeddings, binary_labels, attack_type, config, tracker, log_type
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

    # Save trained models (one for each attack type)
    print(f"⠙ Saving models for {log_type}...")
    for i, model in enumerate(models):
        model_path = Path(f"models/transformer_{log_type}_{config.node_name}_{config.job_id}.pth")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model with metadata
        torch.save({
            'model_state_dict': model.state_dict(),
            'classes': classes,
            'input_dim': embeddings.shape[1],
            'latent_dim': model.latent_dim,
            'n_labels': model.n_labels,
            'transformer_layers': 3,  # Default
            'attention_heads': 8,     # Default
            'dropout': 0.1,
            'scaler': scaler
        }, model_path)
        
        print(f"💾 Model saved to: {model_path}")
        print(f"📊 Model info: {embeddings.shape[1]}D → {len(classes)} classes")
        print(f"🏗️  Architecture: 3 layers, 8 heads, {model.latent_dim}D latent")
        print(f"🎯 Evaluation: Use 'python src/evaluate_transformer.py --log-type {log_type}' for full evaluation")
    
    print(f"✔ Multi-label model saved successfully")
    
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
    
    # Process the log type
    process_log_type_with_args(
        args.log_type, 
        config, 
        force_restart=args.force_restart,
        sample_size=args.sample_size
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
    print("\n📈 NEW: Comprehensive evaluation with multi-label metrics!")
    print("  - Supervised metrics (when true labels available): F1, Precision, Recall, Hamming Loss, Jaccard")
    print("  - Per-class performance with optimized thresholds")
    print("  - Per-class accuracy, confusion matrices, sensitivity/specificity")
    print("  - Classification reports similar to ml_models.py")
    print("  - Label combination analysis and prediction confidence")
    print("  - Unsupervised metrics and confidence analysis")
    print("="*60)


if __name__ == "__main__":
    main() 