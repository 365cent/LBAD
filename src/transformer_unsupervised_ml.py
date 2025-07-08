#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transformer-based Unsupervised Multi-Label Learning for Log Analysis

This script implements a sophisticated unsupervised multi-label learning approach
combining X-Transformer and UMTL strategies. It uses LogBERT embeddings as input
to learn multi-label predictions through pseudo-labels and iterative refinement.

Key Features:
- Semantic label clustering for manageable label space
- Transformer projector/inverse for reconstruction-based learning
- Multi-label matching with pseudo-label generation
- Iterative refinement and label cleaning
- No supervision during training (pseudo-labels only)
- Optimized for M2 GPU (MPS device) when available

Architecture Components:
1. Label Embedding & Clustering: Reduce label space using semantic grouping
2. Transformer Projector: Maps embeddings to latent space
3. Transformer Inverse: Reconstructs embeddings from latent space
4. Neural Matcher: Matches instances to label clusters
5. Multi-label Prediction Head: Generates final multi-label predictions
6. Label Cleaner: Refines pseudo-labels iteratively

Loss Components:
- L_recon: Reconstruction loss ||x - x̂||₁
- L_rank: Multi-label ranking loss BCE(pseudo_y, score_l)
- L_cluster: Cluster matching loss hinge(g(x,k))

Reference: Combined strategy from X-Transformer and UMTL papers
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pickle
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import argparse
from typing import Dict, List, Tuple, Optional, Any
from halo import Halo
import json
import warnings
# Memory management for large datasets - auto-detect and use up to 80% of resources
import psutil
warnings.filterwarnings('ignore')

# Configuration
EMBEDDINGS_DIR = Path("embeddings")
RESULTS_DIR = Path("results")
MODELS_DIR = Path("models")
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 200
REFINEMENT_ITERATIONS = 5
EARLY_STOPPING_PATIENCE = 25
RECONSTRUCTION_WEIGHT = 1.0
RANKING_WEIGHT = 0.5
CLUSTER_WEIGHT = 0.3
RANDOM_STATE = 42
TOP_K_LABELS = 5
N_LABEL_CLUSTERS = 8

def get_memory_limits():
    """Auto-detect system resources and set limits to use up to 80% of available memory."""
    total_memory_gb = psutil.virtual_memory().total / (1024**3)
    available_memory_gb = psutil.virtual_memory().available / (1024**3)
    max_memory_gb = min(total_memory_gb, available_memory_gb) * 0.8
    
    # Adjust other parameters based on available memory
    if max_memory_gb > 16:
        gradient_accumulation_steps = 2
        chunk_size = 50000
        max_samples = 1000000
    elif max_memory_gb > 8:
        gradient_accumulation_steps = 4
        chunk_size = 25000
        max_samples = 750000
    else:
        gradient_accumulation_steps = 8
        chunk_size = 10000
        max_samples = 500000
    
    return max_memory_gb, gradient_accumulation_steps, chunk_size, max_samples

MAX_MEMORY_GB, GRADIENT_ACCUMULATION_STEPS, CHUNK_SIZE, MAX_SAMPLES_FOR_FULL_TRAINING = get_memory_limits()
SUBSAMPLE_LARGE_DATASETS = True  # Whether to subsample very large datasets

# Set random seeds for reproducibility
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """Return best available computation device, optimized for M2 GPU."""
    if torch.backends.mps.is_available():
        print("Using MPS (Metal Performance Shaders) device - M2 GPU")
        return torch.device("mps")
    if torch.cuda.is_available():
        print("Using CUDA device")
        return torch.device("cuda")
    print("Using CPU device")
    return torch.device("cpu")


def clear_memory(device):
    """Clear memory based on device type."""
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        pass  # MPS doesn't need explicit cleanup
    import gc
    gc.collect()


def estimate_memory_usage(embeddings_shape, dtype=np.float32):
    """Estimate memory usage for embeddings in GB."""
    bytes_per_element = np.dtype(dtype).itemsize
    total_bytes = embeddings_shape[0] * embeddings_shape[1] * bytes_per_element
    return total_bytes / (1024**3)  # Convert to GB


def smart_subsample(embeddings: np.ndarray, max_samples: int, random_state: int = RANDOM_STATE) -> np.ndarray:
    """Intelligently subsample large datasets while preserving diversity."""
    if len(embeddings) <= max_samples:
        return np.arange(len(embeddings))
    
    # Use stratified sampling to preserve data diversity
    np.random.seed(random_state)
    
    # Simple random sampling for now - could be improved with clustering-based sampling
    selected_indices = np.random.choice(len(embeddings), max_samples, replace=False)
    selected_indices.sort()  # Keep chronological order
    
    return selected_indices


class ChunkedDataset:
    """Memory-efficient dataset that loads data in chunks."""
    
    def __init__(self, embeddings: np.ndarray, pseudo_labels: np.ndarray, chunk_size: int = CHUNK_SIZE):
        self.embeddings = embeddings
        self.pseudo_labels = pseudo_labels
        self.chunk_size = chunk_size
        self.n_samples = len(embeddings)
        self.n_chunks = (self.n_samples + chunk_size - 1) // chunk_size
        
    def __len__(self):
        return self.n_chunks
    
    def __getitem__(self, chunk_idx):
        start_idx = chunk_idx * self.chunk_size
        end_idx = min(start_idx + self.chunk_size, self.n_samples)
        
        chunk_embeddings = self.embeddings[start_idx:end_idx]
        chunk_labels = self.pseudo_labels[start_idx:end_idx]
        
        return torch.from_numpy(chunk_embeddings.astype(np.float32)), torch.from_numpy(chunk_labels.astype(np.float32))


def create_memory_efficient_dataloader(embeddings: np.ndarray, pseudo_labels: np.ndarray, 
                                      batch_size: int = BATCH_SIZE, shuffle: bool = True):
    """Create memory-efficient dataloader for large datasets."""
    memory_gb = estimate_memory_usage(embeddings.shape)
    
    if memory_gb > MAX_MEMORY_GB:
        print(f"Large dataset detected ({memory_gb:.2f} GB). Using chunked processing.")
        # Use chunked processing
        chunked_dataset = ChunkedDataset(embeddings, pseudo_labels)
        return chunked_dataset, True  # Return dataset and chunked flag
    else:
        # Standard tensor processing
        embeddings_tensor = torch.from_numpy(embeddings.astype(np.float32))
        labels_tensor = torch.from_numpy(pseudo_labels.astype(np.float32))
        dataset = TensorDataset(embeddings_tensor, labels_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return dataloader, False  # Return dataloader and chunked flag


def find_available_embeddings():
    """Find available embedding files."""
    if not EMBEDDINGS_DIR.exists():
        return []
    
    # Look for log embedding files
    log_files = []
    for file_path in EMBEDDINGS_DIR.rglob("log_*.pkl"):
        log_type = file_path.stem.replace("log_", "")
        log_files.append(log_type)
    
    return sorted(log_files)


def load_embeddings_and_labels(log_type: str) -> Tuple[np.ndarray, Optional[Dict], Optional[List[str]]]:
    """Load embeddings and corresponding labels for a log type."""
    spinner = Halo(text=f"Loading embeddings for {log_type}", spinner='dots')
    spinner.start()
    
    # Determine file paths
    if log_type == "all_combined":
        log_file = EMBEDDINGS_DIR / f"log_{log_type}.pkl"
        label_file = EMBEDDINGS_DIR / f"label_{log_type}.pkl"
    else:
        log_file = EMBEDDINGS_DIR / log_type / f"log_{log_type}.pkl"
        label_file = EMBEDDINGS_DIR / log_type / f"label_{log_type}.pkl"
    
    # Load embeddings
    if not log_file.exists():
        spinner.fail(f"Embedding file not found: {log_file}")
        return None, None, None
    
    with open(log_file, 'rb') as f:
        embeddings = pickle.load(f)
    
    spinner.text = f"Loaded {embeddings.shape[0]} embeddings with {embeddings.shape[1]} dimensions"
    
    # Load labels if available
    label_data = None
    classes = None
    if label_file.exists():
        with open(label_file, 'rb') as f:
            label_data = pickle.load(f)
        classes = label_data.get('classes', [])
        spinner.succeed(f"Loaded embeddings and labels for {log_type}")
    else:
        spinner.succeed(f"Loaded embeddings for {log_type} (no labels found)")
    
    return embeddings, label_data, classes


# ---------------------------------------------------------------------------
# Step 1: Label Embedding and Clustering
# ---------------------------------------------------------------------------

def create_label_clusters(classes: List[str], n_clusters: int = N_LABEL_CLUSTERS) -> Tuple[np.ndarray, Dict]:
    """Create semantic label clusters from class names."""
    spinner = Halo(text="Creating semantic label clusters", spinner='dots')
    spinner.start()
    
    if not classes:
        spinner.warn("No classes provided, using default clustering")
        return np.eye(1), {'cluster_0': ['unknown']}
    
    # Simple clustering based on string similarity for now
    # In a real implementation, you'd use actual label embeddings
    n_clusters = min(n_clusters, len(classes))
    
    # Create label-to-cluster mapping using K-means on label names
    # For demonstration, we'll use simple hashing-based clustering
    label_to_cluster = {}
    cluster_assignments = []
    
    for i, class_name in enumerate(classes):
        # Simple hash-based clustering for demonstration
        cluster_id = hash(class_name) % n_clusters
        label_to_cluster[class_name] = cluster_id
        cluster_assignments.append(cluster_id)
    
    # Create cluster-to-labels mapping
    cluster_to_labels = {}
    for i in range(n_clusters):
        cluster_to_labels[f'cluster_{i}'] = []
    
    for class_name, cluster_id in label_to_cluster.items():
        cluster_to_labels[f'cluster_{cluster_id}'].append(class_name)
    
    # Create label-to-cluster matrix C ∈ {0, 1}^{L×K}
    C = np.zeros((len(classes), n_clusters))
    for i, class_name in enumerate(classes):
        cluster_id = label_to_cluster[class_name]
        C[i, cluster_id] = 1
    
    spinner.succeed(f"Created {n_clusters} label clusters from {len(classes)} classes")
    return C, cluster_to_labels


# ---------------------------------------------------------------------------
# Step 2: Transformer Architecture Components
# ---------------------------------------------------------------------------

class TransformerProjector(nn.Module):
    """Transformer projector: maps embeddings to latent space."""
    
    def __init__(self, input_dim: int, latent_dim: int, num_heads: int = 8, 
                 num_layers: int = 3, dropout: float = 0.1):
        super(TransformerProjector, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, latent_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=latent_dim * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_proj = nn.Linear(latent_dim, latent_dim)
        
        # Positional encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, 1, latent_dim))
        
    def forward(self, x):
        # x shape: (batch_size, input_dim)
        batch_size = x.size(0)
        
        # Project and add positional encoding
        x = self.input_proj(x).unsqueeze(1)  # (batch_size, 1, latent_dim)
        x = x + self.pos_embedding.expand(batch_size, -1, -1)
        
        # Transform
        z = self.transformer(x)  # (batch_size, 1, latent_dim)
        z = self.output_proj(z.squeeze(1))  # (batch_size, latent_dim)
        
        return z


class TransformerInverse(nn.Module):
    """Transformer inverse: reconstructs embeddings from latent space."""
    
    def __init__(self, latent_dim: int, output_dim: int, num_heads: int = 8, 
                 num_layers: int = 3, dropout: float = 0.1):
        super(TransformerInverse, self).__init__()
        
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        
        # Input projection
        self.input_proj = nn.Linear(latent_dim, latent_dim)
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=latent_dim * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_proj = nn.Linear(latent_dim, output_dim)
        
        # Positional encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, 1, latent_dim))
        
    def forward(self, z):
        # z shape: (batch_size, latent_dim)
        batch_size = z.size(0)
        
        # Project and add positional encoding
        z = self.input_proj(z).unsqueeze(1)  # (batch_size, 1, latent_dim)
        z_pos = z + self.pos_embedding.expand(batch_size, -1, -1)
        
        # Transform
        x_hat = self.transformer(z_pos, z_pos)  # (batch_size, 1, latent_dim)
        x_hat = self.output_proj(x_hat.squeeze(1))  # (batch_size, output_dim)
        
        return x_hat


class NeuralMatcher(nn.Module):
    """Neural matcher: matches instances to label clusters."""
    
    def __init__(self, latent_dim: int, n_clusters: int, dropout: float = 0.1):
        super(NeuralMatcher, self).__init__()
        
        self.latent_dim = latent_dim
        self.n_clusters = n_clusters
        
        # Cluster matching layers
        self.cluster_weights = nn.Parameter(torch.randn(n_clusters, latent_dim))
        self.matching_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim // 2, n_clusters)
        )
        
    def forward(self, z):
        # z shape: (batch_size, latent_dim)
        # Compute matching scores g(x, k) = w_k^T · φ_T(x)
        cluster_scores = torch.matmul(z, self.cluster_weights.t())  # (batch_size, n_clusters)
        
        # Additional matching head
        match_scores = self.matching_head(z)  # (batch_size, n_clusters)
        
        # Combine scores
        final_scores = cluster_scores + match_scores
        
        return final_scores


class MultiLabelHead(nn.Module):
    """Multi-label prediction head."""
    
    def __init__(self, latent_dim: int, n_labels: int, dropout: float = 0.1):
        super(MultiLabelHead, self).__init__()
        
        self.prediction_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim // 2, latent_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim // 4, n_labels)
        )
        
    def forward(self, z):
        # z shape: (batch_size, latent_dim)
        scores = self.prediction_head(z)  # (batch_size, n_labels)
        return torch.sigmoid(scores)


class LabelCleaner(nn.Module):
    """Label cleaner for pseudo-label refinement."""
    
    def __init__(self, n_labels: int, hidden_dim: int = 128, dropout: float = 0.1):
        super(LabelCleaner, self).__init__()
        
        self.cleaner = nn.Sequential(
            nn.Linear(n_labels, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_labels)
        )
        
    def forward(self, pseudo_labels):
        # pseudo_labels shape: (batch_size, n_labels)
        cleaned_labels = torch.sigmoid(self.cleaner(pseudo_labels))
        return cleaned_labels


# ---------------------------------------------------------------------------
# Step 3: Combined Model Architecture
# ---------------------------------------------------------------------------

class UnsupervisedMultiLabelTransformer(nn.Module):
    """Combined unsupervised multi-label transformer model."""
    
    def __init__(self, input_dim: int, latent_dim: int, n_labels: int, 
                 n_clusters: int, dropout: float = 0.1):
        super(UnsupervisedMultiLabelTransformer, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.n_labels = n_labels
        self.n_clusters = n_clusters
        
        # Core components
        self.projector = TransformerProjector(input_dim, latent_dim, dropout=dropout)
        self.inverse = TransformerInverse(latent_dim, input_dim, dropout=dropout)
        self.matcher = NeuralMatcher(latent_dim, n_clusters, dropout=dropout)
        self.multilabel_head = MultiLabelHead(latent_dim, n_labels, dropout=dropout)
        self.label_cleaner = LabelCleaner(n_labels, dropout=dropout)
        
    def forward(self, x):
        # Step 1: Project to latent space
        z = self.projector(x)
        
        # Step 2: Reconstruct
        x_hat = self.inverse(z)
        
        # Step 3: Match to clusters
        cluster_scores = self.matcher(z)
        
        # Step 4: Predict labels
        label_scores = self.multilabel_head(z)
        
        return {
            'latent': z,
            'reconstructed': x_hat,
            'cluster_scores': cluster_scores,
            'label_scores': label_scores
        }
    
    def clean_labels(self, pseudo_labels):
        """Clean pseudo-labels using label cleaner."""
        return self.label_cleaner(pseudo_labels)


# ---------------------------------------------------------------------------
# Step 4: Training and Pseudo-label Generation
# ---------------------------------------------------------------------------

def generate_initial_pseudo_labels(embeddings: np.ndarray, classes: List[str], 
                                  C: np.ndarray, top_k: int = TOP_K_LABELS) -> np.ndarray:
    """Generate initial pseudo-labels using clustering."""
    spinner = Halo(text="Generating initial pseudo-labels", spinner='dots')
    spinner.start()
    
    n_samples = embeddings.shape[0]
    n_labels = len(classes)
    
    # Initialize pseudo-labels
    pseudo_labels = np.zeros((n_samples, n_labels), dtype=np.float32)
    
    # Use K-means to create initial clusters
    n_clusters = min(C.shape[1], n_samples // 10)
    if n_clusters < 2:
        n_clusters = 2
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE)
    cluster_assignments = kmeans.fit_predict(embeddings)
    
    # Assign labels based on cluster centroids
    for i, cluster_id in enumerate(cluster_assignments):
        # For each cluster, assign top-k most relevant labels
        # This is a simplified approach - in practice, you'd use more sophisticated methods
        if cluster_id < C.shape[1]:
            cluster_labels = C[:, cluster_id]
            top_label_indices = np.argsort(cluster_labels)[-top_k:]
            pseudo_labels[i, top_label_indices] = 1.0
        else:
            # Random assignment for overflow clusters
            random_indices = np.random.choice(n_labels, min(top_k, n_labels), replace=False)
            pseudo_labels[i, random_indices] = 1.0
    
    spinner.succeed(f"Generated pseudo-labels with avg {pseudo_labels.sum(axis=1).mean():.2f} labels per sample")
    return pseudo_labels


def compute_reconstruction_confidence(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    """Compute reconstruction confidence scores."""
    # L1 reconstruction error
    recon_error = torch.mean(torch.abs(x - x_hat), dim=1)
    
    # Convert to confidence (lower error = higher confidence)
    confidence = torch.exp(-recon_error)
    
    return confidence


def refine_pseudo_labels(model: UnsupervisedMultiLabelTransformer, 
                        embeddings: torch.Tensor, 
                        pseudo_labels: torch.Tensor,
                        confidence_threshold: float = 0.5) -> torch.Tensor:
    """Refine pseudo-labels using reconstruction confidence."""
    model.eval()
    
    with torch.no_grad():
        outputs = model(embeddings)
        
        # Compute reconstruction confidence
        confidence = compute_reconstruction_confidence(embeddings, outputs['reconstructed'])
        
        # Refine labels based on confidence
        refined_labels = pseudo_labels.clone()
        
        # High confidence: keep current labels
        # Low confidence: reduce number of labels
        low_confidence_mask = confidence < confidence_threshold
        refined_labels[low_confidence_mask] *= 0.5  # Reduce label strength
        
        # Use label cleaner
        cleaned_labels = model.clean_labels(refined_labels)
        
    return cleaned_labels


def refine_pseudo_labels_chunked(model: UnsupervisedMultiLabelTransformer, 
                                embeddings: np.ndarray, 
                                pseudo_labels: np.ndarray,
                                device: torch.device,
                                confidence_threshold: float = 0.5) -> np.ndarray:
    """Refine pseudo-labels using chunked processing for memory efficiency."""
    model.eval()
    
    refined_labels = np.zeros_like(pseudo_labels)
    chunk_size = CHUNK_SIZE
    
    with torch.no_grad():
        for start_idx in range(0, len(embeddings), chunk_size):
            end_idx = min(start_idx + chunk_size, len(embeddings))
            
            # Process chunk
            chunk_embeddings = torch.from_numpy(embeddings[start_idx:end_idx].astype(np.float32)).to(device)
            chunk_pseudo_labels = torch.from_numpy(pseudo_labels[start_idx:end_idx].astype(np.float32)).to(device)
            
            # Get model outputs
            outputs = model(chunk_embeddings)
            
            # Compute reconstruction confidence
            confidence = compute_reconstruction_confidence(chunk_embeddings, outputs['reconstructed'])
            
            # Refine labels based on confidence
            chunk_refined = chunk_pseudo_labels.clone()
            low_confidence_mask = confidence < confidence_threshold
            chunk_refined[low_confidence_mask] *= 0.5
            
            # Use label cleaner
            cleaned_chunk = model.clean_labels(chunk_refined)
            
            # Store refined labels
            refined_labels[start_idx:end_idx] = cleaned_chunk.cpu().numpy()
            
            # Clear memory
            del chunk_embeddings, chunk_pseudo_labels, outputs, cleaned_chunk
            clear_memory(device)
    
    return refined_labels


def train_unsupervised_multilabel_model(embeddings: np.ndarray, classes: List[str], 
                                       C: np.ndarray, device: torch.device) -> UnsupervisedMultiLabelTransformer:
    """Train the unsupervised multi-label transformer model with memory-efficient processing."""
    spinner = Halo(text="Analyzing dataset and memory requirements", spinner='dots')
    spinner.start()
    
    # Setup
    n_samples, input_dim = embeddings.shape
    n_labels = len(classes)
    n_clusters = C.shape[1]
    latent_dim = min(256, input_dim // 2)
    
    # Memory analysis
    memory_gb = estimate_memory_usage(embeddings.shape)
    spinner.text = f"Dataset: {n_samples:,} samples, {memory_gb:.2f} GB"
    
    # Smart subsampling for very large datasets
    if SUBSAMPLE_LARGE_DATASETS and n_samples > MAX_SAMPLES_FOR_FULL_TRAINING:
        spinner.text = "Large dataset detected. Applying intelligent subsampling..."
        selected_indices = smart_subsample(embeddings, MAX_SAMPLES_FOR_FULL_TRAINING)
        embeddings = embeddings[selected_indices]
        n_samples = len(embeddings)
        memory_gb = estimate_memory_usage(embeddings.shape)
        spinner.text = f"Subsampled to {n_samples:,} samples ({memory_gb:.2f} GB)"
    
    # Normalize embeddings
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings).astype(np.float32)
    
    # Keep C and initial pseudo-labels in CPU memory until needed
    C_tensor = torch.from_numpy(C.astype(np.float32)).to(device)
    
    # Generate initial pseudo-labels
    initial_pseudo_labels = generate_initial_pseudo_labels(embeddings_scaled, classes, C)
    
    # Initialize model
    print("Initializing unsupervised multi-label transformer model...")
    model = UnsupervisedMultiLabelTransformer(
        input_dim=input_dim,
        latent_dim=latent_dim,
        n_labels=n_labels,
        n_clusters=n_clusters,
        dropout=0.1
    ).to(device)
    
    # Optimizers with gradient accumulation support
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    cleaner_optimizer = optim.Adam(model.label_cleaner.parameters(), lr=LEARNING_RATE * 0.5)
    
    # Create memory-efficient data processing
    current_pseudo_labels = initial_pseudo_labels
    
    print("Starting unsupervised multi-label training with memory-efficient processing...")
    
    # Training loop with iterative refinement and memory-efficient processing
    for refinement_iter in range(REFINEMENT_ITERATIONS):
        print(f"\n--- Refinement Iteration {refinement_iter + 1}/{REFINEMENT_ITERATIONS} ---")
        
        model.train()
        epoch_losses = []
        
        # Create memory-efficient dataloader for current iteration
        dataloader_or_dataset, is_chunked = create_memory_efficient_dataloader(
            embeddings_scaled, current_pseudo_labels, BATCH_SIZE, shuffle=True
        )
        
        for epoch in range(EPOCHS // REFINEMENT_ITERATIONS):
            epoch_loss = 0.0
            num_batches = 0
            
            # Handle chunked vs standard processing
            if is_chunked:
                # Chunked processing with gradient accumulation
                for chunk_idx in range(len(dataloader_or_dataset)):
                    chunk_embeddings, chunk_labels = dataloader_or_dataset[chunk_idx]
                    
                    # Process chunk in mini-batches
                    chunk_size = chunk_embeddings.size(0)
                    num_mini_batches = (chunk_size + BATCH_SIZE - 1) // BATCH_SIZE
                    
                    optimizer.zero_grad()
                    accumulated_loss = 0.0
                    
                    for mini_batch_idx in range(num_mini_batches):
                        start_idx = mini_batch_idx * BATCH_SIZE
                        end_idx = min(start_idx + BATCH_SIZE, chunk_size)
                        
                        batch_embeddings = chunk_embeddings[start_idx:end_idx].to(device)
                        batch_pseudo_labels = chunk_labels[start_idx:end_idx].to(device)
                        
                        # Forward pass
                        outputs = model(batch_embeddings)
                        
                        # Compute losses
                        recon_loss = F.l1_loss(outputs['reconstructed'], batch_embeddings)
                        rank_loss = F.binary_cross_entropy(outputs['label_scores'], batch_pseudo_labels)
                        
                        # Cluster matching loss
                        cluster_targets = torch.matmul(batch_pseudo_labels, C_tensor)
                        cluster_loss = F.mse_loss(outputs['cluster_scores'], cluster_targets)
                        
                        # Combined loss (normalized by accumulation steps)
                        total_loss = (RECONSTRUCTION_WEIGHT * recon_loss + 
                                     RANKING_WEIGHT * rank_loss + 
                                     CLUSTER_WEIGHT * cluster_loss) / GRADIENT_ACCUMULATION_STEPS
                        
                        total_loss.backward()
                        accumulated_loss += total_loss.item()
                        
                        # Train label cleaner on high-confidence samples
                        if mini_batch_idx % 2 == 0:  # Train cleaner less frequently
                            with torch.no_grad():
                                confidence = compute_reconstruction_confidence(batch_embeddings, outputs['reconstructed'])
                                high_confidence_mask = confidence > confidence.median()
                            
                            if high_confidence_mask.sum() > 0:
                                cleaner_optimizer.zero_grad()
                                cleaned_labels = model.clean_labels(batch_pseudo_labels[high_confidence_mask])
                                cleaner_targets = outputs['label_scores'][high_confidence_mask].detach()
                                cleaner_loss = F.binary_cross_entropy(cleaned_labels, cleaner_targets)
                                cleaner_loss.backward()
                                cleaner_optimizer.step()
                        
                        # Clear batch from GPU memory
                        del batch_embeddings, batch_pseudo_labels, outputs
                        clear_memory(device)
                    
                    # Apply accumulated gradients
                    if (chunk_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                        optimizer.step()
                        optimizer.zero_grad()
                    
                    epoch_loss += accumulated_loss * GRADIENT_ACCUMULATION_STEPS
                    num_batches += num_mini_batches
                    
                    # Clear chunk from memory
                    del chunk_embeddings, chunk_labels
                    clear_memory(device)
                
                # Apply any remaining accumulated gradients
                optimizer.step()
                
            else:
                # Standard processing for smaller datasets
                for batch_idx, (batch_embeddings, batch_pseudo_labels) in enumerate(dataloader_or_dataset):
                    batch_embeddings = batch_embeddings.to(device)
                    batch_pseudo_labels = batch_pseudo_labels.to(device)
                    
                    optimizer.zero_grad()
                    
                    # Forward pass
                    outputs = model(batch_embeddings)
                    
                    # Compute losses
                    recon_loss = F.l1_loss(outputs['reconstructed'], batch_embeddings)
                    rank_loss = F.binary_cross_entropy(outputs['label_scores'], batch_pseudo_labels)
                    
                    cluster_targets = torch.matmul(batch_pseudo_labels, C_tensor)
                    cluster_loss = F.mse_loss(outputs['cluster_scores'], cluster_targets)
                    
                    total_loss = (RECONSTRUCTION_WEIGHT * recon_loss + 
                                 RANKING_WEIGHT * rank_loss + 
                                 CLUSTER_WEIGHT * cluster_loss)
                    
                    total_loss.backward()
                    optimizer.step()
                    
                    # Train label cleaner
                    if batch_idx % 2 == 0:
                        with torch.no_grad():
                            confidence = compute_reconstruction_confidence(batch_embeddings, outputs['reconstructed'])
                            high_confidence_mask = confidence > confidence.median()
                        
                        if high_confidence_mask.sum() > 0:
                            cleaner_optimizer.zero_grad()
                            cleaned_labels = model.clean_labels(batch_pseudo_labels[high_confidence_mask])
                            cleaner_targets = outputs['label_scores'][high_confidence_mask].detach()
                            cleaner_loss = F.binary_cross_entropy(cleaned_labels, cleaner_targets)
                            cleaner_loss.backward()
                            cleaner_optimizer.step()
                    
                    epoch_loss += total_loss.item()
                    num_batches += 1
                    
                    # Clear memory periodically
                    if batch_idx % 10 == 0:
                        clear_memory(device)
            
            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
            epoch_losses.append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}, Loss: {avg_loss:.6f}, Processing: {'Chunked' if is_chunked else 'Standard'}")
        
        # Refine pseudo-labels for next iteration
        if refinement_iter < REFINEMENT_ITERATIONS - 1:
            print("  Refining pseudo-labels using memory-efficient processing...")
            current_pseudo_labels = refine_pseudo_labels_chunked(
                model, embeddings_scaled, current_pseudo_labels, device
            )
    
    model.eval()
    return model, scaler


# ---------------------------------------------------------------------------
# Step 5: Evaluation and Prediction
# ---------------------------------------------------------------------------

def evaluate_unsupervised_model(model: UnsupervisedMultiLabelTransformer, 
                               embeddings: np.ndarray, 
                               label_data: Dict, 
                               classes: List[str], 
                               scaler: StandardScaler, 
                               device: torch.device) -> Dict[str, Any]:
    """Evaluate the unsupervised multi-label model with memory-efficient processing."""
    spinner = Halo(text="Evaluating unsupervised multi-label model", spinner='dots')
    spinner.start()
    
    # Normalize embeddings
    embeddings_scaled = scaler.transform(embeddings).astype(np.float32)
    
    # Memory-efficient prediction generation
    memory_gb = estimate_memory_usage(embeddings_scaled.shape)
    predictions = np.zeros((len(embeddings_scaled), len(classes)), dtype=np.float32)
    
    model.eval()
    with torch.no_grad():
        if memory_gb > MAX_MEMORY_GB:
            # Chunked processing
            spinner.text = f"Large dataset ({memory_gb:.2f} GB). Using chunked evaluation..."
            chunk_size = CHUNK_SIZE
            
            for start_idx in range(0, len(embeddings_scaled), chunk_size):
                end_idx = min(start_idx + chunk_size, len(embeddings_scaled))
                
                chunk_embeddings = torch.from_numpy(embeddings_scaled[start_idx:end_idx]).to(device)
                outputs = model(chunk_embeddings)
                predictions[start_idx:end_idx] = outputs['label_scores'].cpu().numpy()
                
                # Clear memory
                del chunk_embeddings, outputs
                clear_memory(device)
                
                if start_idx % (chunk_size * 10) == 0:
                    spinner.text = f"Evaluation progress: {start_idx:,}/{len(embeddings_scaled):,}"
        else:
            # Standard processing
            embeddings_tensor = torch.from_numpy(embeddings_scaled).to(device)
            outputs = model(embeddings_tensor)
            predictions = outputs['label_scores'].cpu().numpy()
    
    # Create ground truth labels if available
    if label_data and 'vectors' in label_data:
        ground_truth = label_data['vectors']
        
        # Compute metrics
        # Binary predictions using threshold
        binary_predictions = (predictions > 0.5).astype(int)
        
        # Compute per-label metrics
        per_label_metrics = {}
        for i, class_name in enumerate(classes):
            if i < ground_truth.shape[1]:
                true_labels = ground_truth[:, i]
                pred_labels = binary_predictions[:, i]
                
                tp = np.sum((true_labels == 1) & (pred_labels == 1))
                fp = np.sum((true_labels == 0) & (pred_labels == 1))
                fn = np.sum((true_labels == 1) & (pred_labels == 0))
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                per_label_metrics[class_name] = {
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1
                }
        
        # Overall metrics
        overall_metrics = {
            'micro_precision': np.mean([m['precision'] for m in per_label_metrics.values()]),
            'micro_recall': np.mean([m['recall'] for m in per_label_metrics.values()]),
            'micro_f1': np.mean([m['f1_score'] for m in per_label_metrics.values()])
        }
        
        spinner.succeed("Model evaluation completed")
        
        return {
            'predictions': predictions,
            'binary_predictions': binary_predictions,
            'per_label_metrics': per_label_metrics,
            'overall_metrics': overall_metrics
        }
    else:
        spinner.succeed("Model evaluation completed (no ground truth)")
        return {
            'predictions': predictions,
            'binary_predictions': (predictions > 0.5).astype(int)
        }


def generate_comprehensive_report(log_type: str, embeddings: np.ndarray, 
                                 classes: List[str], evaluation_results: Dict[str, Any],
                                 C: np.ndarray, cluster_to_labels: Dict) -> str:
    """Generate comprehensive summary report."""
    report = []
    report.append(f"{'='*80}")
    report.append(f"UNSUPERVISED MULTI-LABEL TRANSFORMER REPORT")
    report.append(f"Log Type: {log_type}")
    report.append(f"{'='*80}")
    report.append("")
    
    # Dataset information
    report.append(f"Dataset Information:")
    report.append(f"  - Total samples: {len(embeddings):,}")
    report.append(f"  - Embedding dimension: {embeddings.shape[1]}")
    report.append(f"  - Number of labels: {len(classes)}")
    report.append(f"  - Label clusters: {C.shape[1]}")
    report.append("")
    
    # Model architecture
    report.append(f"Model Architecture:")
    report.append(f"  - Transformer Projector: {embeddings.shape[1]} → 256 → 256")
    report.append(f"  - Transformer Inverse: 256 → {embeddings.shape[1]}")
    report.append(f"  - Neural Matcher: 256 → {C.shape[1]} clusters")
    report.append(f"  - Multi-label Head: 256 → {len(classes)} labels")
    report.append(f"  - Label Cleaner: {len(classes)} → 128 → {len(classes)}")
    report.append("")
    
    # Cluster information
    report.append(f"Label Clusters:")
    for cluster_name, cluster_labels in cluster_to_labels.items():
        if cluster_labels:
            report.append(f"  - {cluster_name}: {', '.join(cluster_labels[:3])}{'...' if len(cluster_labels) > 3 else ''}")
    report.append("")
    
    # Predictions summary
    predictions = evaluation_results['predictions']
    binary_predictions = evaluation_results['binary_predictions']
    
    report.append(f"Prediction Summary:")
    report.append(f"  - Avg predictions per sample: {predictions.mean(axis=0).sum():.2f}")
    report.append(f"  - Avg binary predictions per sample: {binary_predictions.mean(axis=0).sum():.2f}")
    report.append(f"  - Most frequent predicted labels:")
    
    label_frequencies = binary_predictions.sum(axis=0)
    top_labels = np.argsort(label_frequencies)[-5:][::-1]
    for i, label_idx in enumerate(top_labels):
        if label_idx < len(classes):
            frequency = label_frequencies[label_idx]
            percentage = (frequency / len(embeddings)) * 100
            report.append(f"    {i+1}. {classes[label_idx]}: {frequency} ({percentage:.1f}%)")
    report.append("")
    
    # Performance metrics (if available)
    if 'per_label_metrics' in evaluation_results:
        report.append(f"Performance Metrics:")
        report.append(f"  - Overall Micro-Precision: {evaluation_results['overall_metrics']['micro_precision']:.4f}")
        report.append(f"  - Overall Micro-Recall: {evaluation_results['overall_metrics']['micro_recall']:.4f}")
        report.append(f"  - Overall Micro-F1: {evaluation_results['overall_metrics']['micro_f1']:.4f}")
        report.append("")
        
        report.append(f"Top Performing Labels:")
        per_label_f1 = [(name, metrics['f1_score']) for name, metrics in evaluation_results['per_label_metrics'].items()]
        per_label_f1.sort(key=lambda x: x[1], reverse=True)
        
        for i, (label_name, f1_score) in enumerate(per_label_f1[:5]):
            report.append(f"  {i+1}. {label_name}: F1={f1_score:.4f}")
    
    report.append("")
    report.append(f"Generated by Unsupervised Multi-Label Transformer")
    report.append(f"Architecture: X-Transformer + UMTL Combined Strategy")
    report.append(f"{'='*80}")
    
    return "\n".join(report)


def create_multilabel_visualization(embeddings: np.ndarray, predictions: np.ndarray, 
                                   classes: List[str], log_type: str, output_dir: Path):
    """Create visualization of multi-label predictions."""
    spinner = Halo(text="Creating multi-label visualizations", spinner='dots')
    spinner.start()
    
    # Sample for visualization if too large
    if embeddings.shape[0] > 3000:
        idx = np.random.choice(len(embeddings), 3000, replace=False)
        embeddings_viz = embeddings[idx]
        predictions_viz = predictions[idx]
    else:
        embeddings_viz = embeddings
        predictions_viz = predictions
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=RANDOM_STATE, perplexity=30)
    embeddings_2d = tsne.fit_transform(embeddings_viz)
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Number of predicted labels per sample
    n_labels_per_sample = predictions_viz.sum(axis=1)
    scatter = axes[0, 0].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                                c=n_labels_per_sample, cmap='viridis', alpha=0.6, s=20)
    axes[0, 0].set_title('Number of Predicted Labels per Sample')
    axes[0, 0].set_xlabel('t-SNE Component 1')
    axes[0, 0].set_ylabel('t-SNE Component 2')
    plt.colorbar(scatter, ax=axes[0, 0])
    
    # 2. Most frequent label
    most_frequent_label = np.argmax(predictions_viz, axis=1)
    scatter = axes[0, 1].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                                c=most_frequent_label, cmap='tab20', alpha=0.6, s=20)
    axes[0, 1].set_title('Most Confident Predicted Label')
    axes[0, 1].set_xlabel('t-SNE Component 1')
    axes[0, 1].set_ylabel('t-SNE Component 2')
    
    # 3. Prediction confidence (max probability)
    max_confidence = np.max(predictions_viz, axis=1)
    scatter = axes[1, 0].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                                c=max_confidence, cmap='plasma', alpha=0.6, s=20)
    axes[1, 0].set_title('Maximum Prediction Confidence')
    axes[1, 0].set_xlabel('t-SNE Component 1')
    axes[1, 0].set_ylabel('t-SNE Component 2')
    plt.colorbar(scatter, ax=axes[1, 0])
    
    # 4. Label distribution histogram
    label_counts = predictions_viz.sum(axis=0)
    top_labels = np.argsort(label_counts)[-10:][::-1]
    axes[1, 1].bar(range(len(top_labels)), label_counts[top_labels])
    axes[1, 1].set_title('Top 10 Most Predicted Labels')
    axes[1, 1].set_xlabel('Label Index')
    axes[1, 1].set_ylabel('Frequency')
    
    # Set x-axis labels if classes are available
    if len(classes) > 0:
        label_names = [classes[i] if i < len(classes) else f'Label_{i}' for i in top_labels]
        axes[1, 1].set_xticks(range(len(top_labels)))
        axes[1, 1].set_xticklabels(label_names, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'multilabel_predictions_{log_type}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    spinner.succeed(f"Multi-label visualization saved to {output_dir}")


def main():
    # Update global configuration first
    global EPOCHS, REFINEMENT_ITERATIONS, MAX_MEMORY_GB, MAX_SAMPLES_FOR_FULL_TRAINING, SUBSAMPLE_LARGE_DATASETS, CHUNK_SIZE
    
    parser = argparse.ArgumentParser(description="Memory-Efficient Unsupervised Multi-Label Transformer for Log Analysis")
    parser.add_argument("--log-type", type=str, default=None, 
                       help="Process specific log type (default: process all)")
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                       help="Number of training epochs")
    parser.add_argument("--refinement-iterations", type=int, default=REFINEMENT_ITERATIONS,
                       help="Number of pseudo-label refinement iterations")
    parser.add_argument("--max-memory-gb", type=float, default=MAX_MEMORY_GB,
                       help="Maximum memory usage in GB for embeddings")
    parser.add_argument("--max-samples", type=int, default=MAX_SAMPLES_FOR_FULL_TRAINING,
                       help="Maximum samples for full training (larger datasets will be subsampled)")
    parser.add_argument("--no-subsample", action="store_true",
                       help="Disable subsampling for large datasets (use chunked processing instead)")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE,
                       help="Chunk size for memory-efficient processing")
    args = parser.parse_args()
    
    # Setup
    device = get_device()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Find available embeddings
    available_types = find_available_embeddings()
    if not available_types:
        print("No embedding files found. Please run logbert_embeddings.py first.")
        return
    
    print(f"Available embedding types: {', '.join(available_types)}")
    
    # Determine types to process
    if args.log_type:
        if args.log_type not in available_types:
            print(f"Log type '{args.log_type}' not found.")
            return
        types_to_process = [args.log_type]
    else:
        types_to_process = available_types
    
    # Update global configuration
    EPOCHS = args.epochs
    REFINEMENT_ITERATIONS = args.refinement_iterations
    MAX_MEMORY_GB = args.max_memory_gb
    MAX_SAMPLES_FOR_FULL_TRAINING = args.max_samples
    SUBSAMPLE_LARGE_DATASETS = not args.no_subsample
    CHUNK_SIZE = args.chunk_size
    
    print(f"Memory configuration: Max {MAX_MEMORY_GB:.1f} GB, Chunk size: {CHUNK_SIZE:,}, Max samples: {MAX_SAMPLES_FOR_FULL_TRAINING:,}")
    print(f"Subsampling {'enabled' if SUBSAMPLE_LARGE_DATASETS else 'disabled'} for large datasets")
    
    # Process each log type
    for log_type in types_to_process:
        print(f"\n{'='*80}")
        print(f"Processing log type: {log_type}")
        print(f"{'='*80}")
        
        try:
            # Step 1: Load embeddings and labels
            embeddings, label_data, classes = load_embeddings_and_labels(log_type)
            if embeddings is None:
                print(f"Failed to load embeddings for {log_type}")
                continue
            
            print(f"Loaded {len(embeddings)} embeddings with {embeddings.shape[1]} dimensions")
            
            # Step 2: Create label clusters
            C, cluster_to_labels = create_label_clusters(classes, n_clusters=min(N_LABEL_CLUSTERS, len(classes)))
            
            # Step 3: Train unsupervised multi-label model
            print(f"\nTraining unsupervised multi-label transformer...")
            print(f"Architecture: X-Transformer + UMTL Combined Strategy")
            print(f"Training epochs: {EPOCHS}, Refinement iterations: {REFINEMENT_ITERATIONS}")
            
            model, scaler = train_unsupervised_multilabel_model(embeddings, classes, C, device)
            
            # Step 4: Evaluate model
            print("\nEvaluating unsupervised multi-label model...")
            evaluation_results = evaluate_unsupervised_model(model, embeddings, label_data, classes, scaler, device)
            
            # Step 5: Generate comprehensive report
            summary_report = generate_comprehensive_report(log_type, embeddings, classes, evaluation_results, C, cluster_to_labels)
            
            # Save results
            output_dir = RESULTS_DIR / log_type
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save summary report
            with open(output_dir / "unsupervised_multilabel_summary.txt", "w") as f:
                f.write(summary_report)
            
            # Save model
            model_path = MODELS_DIR / f"unsupervised_multilabel_{log_type}.pth"
            torch.save({
                'model_state_dict': model.state_dict(),
                'scaler': scaler,
                'classes': classes,
                'C': C,
                'cluster_to_labels': cluster_to_labels,
                'config': {
                    'input_dim': embeddings.shape[1],
                    'latent_dim': 256,
                    'n_labels': len(classes),
                    'n_clusters': C.shape[1]
                }
            }, model_path)
            
            # Save detailed results
            with open(output_dir / "detailed_multilabel_results.pkl", "wb") as f:
                pickle.dump({
                    'evaluation_results': evaluation_results,
                    'embeddings_shape': embeddings.shape,
                    'classes': classes,
                    'C': C,
                    'cluster_to_labels': cluster_to_labels
                }, f)
            
            # Create visualization
            create_multilabel_visualization(embeddings, evaluation_results['predictions'], classes, log_type, output_dir)
            
            # Display summary
            print("\n" + summary_report)
            
            # Clear memory
            del model, embeddings, evaluation_results
            clear_memory(device)
            
        except Exception as e:
            print(f"Error processing {log_type}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*80}")
    print("Unsupervised Multi-Label Transformer analysis completed!")
    print(f"Results saved to: {RESULTS_DIR}")
    print(f"Models saved to: {MODELS_DIR}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main() 