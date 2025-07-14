#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LogBERT Embeddings for Log Analysis - Using BERT CLS tokens

This script extracts BERT CLS token embeddings for log analysis, following the same
input/output format as fasttext_embedding.py for compatibility with downstream tasks.

Key Features:
- Uses pre-trained BERT to extract CLS token embeddings (768D vectors)
- Creates binary multi-label vectors with clear column mapping
- Maintains compatibility with FastText output format
- Optimized for M2 GPU (MPS device) when available
- Enhanced progress tracking with dots spinner
- Visualization shows ALL classes without sampling/reduction

Output files per log type (3 files for clarity):
- log_{type}.pkl: Raw log text embeddings (2314D enhanced BERT vectors, float32)
  * Combines CLS token (768D) + mean pooling (768D) + max pooling (768D) + attention features (10D)
  * Captures global context, average meaning, key features, and attention patterns
- label_{type}.pkl: Binary label vectors with metadata (same format as FastText)
  * 'vectors': Binary arrays where [0 1 0] means only second class is present
  * 'classes': List of attack types corresponding to each column
  * 'description': Explanation of the binary vector format
- attack_types_{type}.txt: Human-readable attack type mapping and examples

Performance optimizations:
- Batch processing (8 samples per batch for BERT)
- Memory-efficient data types (int8 for labels, float32 for embeddings)
- GPU acceleration when available (MPS for M2, CUDA for NVIDIA)
- Optimized pickle protocol for faster I/O
"""

import os
import pandas as pd
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
from pathlib import Path
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import json
import multiprocessing as mp
import argparse
from transformers import BertModel, BertTokenizer
from torch.utils.data import DataLoader, Dataset
from halo import Halo
from typing import List, Dict, Tuple, Optional
import time
import psutil
import hashlib

# Configuration
OUTPUT_DIR = Path("embeddings")
PROCESSED_DIR = Path("processed")
CHECKPOINT_DIR = Path("checkpoints") / "logbert"
VECTOR_SIZE = 2314  # Enhanced BERT: CLS(768) + Mean(768) + Max(768) + Attention(10)
MAX_SEQ_LENGTH = 128
BATCH_SIZE = 8
NUM_WORKERS = 2

# Performance thresholds for auto-optimization
SMALL_DATASET_THRESHOLD = 10000    # < 10K entries
MEDIUM_DATASET_THRESHOLD = 100000  # < 100K entries
LARGE_DATASET_THRESHOLD = 500000   # < 500K entries
# > 500K entries = Very Large Dataset

# Performance configurations based on dataset size
PERF_CONFIG = {
    'small': {'batch_size': 16, 'workers': 4, 'clear_freq': 100},
    'medium': {'batch_size': 12, 'workers': 3, 'clear_freq': 50},
    'large': {'batch_size': 8, 'workers': 2, 'clear_freq': 25},
    'very_large': {'batch_size': 4, 'workers': 1, 'clear_freq': 10}
}

# Enable TensorFlow optimizations but limit threads
tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.threading.set_intra_op_parallelism_threads(2)

# Set CUDA memory allocation configuration to avoid fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

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
        torch.cuda.synchronize()  # Wait for all operations to complete
    elif device.type == "mps":
        pass  # MPS doesn't need explicit cleanup
    import gc
    gc.collect()


def get_available_gpu_memory(device):
    """Get available GPU memory in MB."""
    if device.type == "cuda":
        return torch.cuda.get_device_properties(device).total_memory / (1024**2)
    elif device.type == "mps":
        return 8192  # Assume 8GB for M2 GPU
    return 0


def adjust_batch_size_for_memory(initial_batch_size, device):
    """Adjust batch size based on available GPU memory."""
    if device.type == "cuda":
        try:
            # Get memory info
            total_memory = torch.cuda.get_device_properties(device).total_memory / (1024**3)  # GB
            allocated_memory = torch.cuda.memory_allocated(device) / (1024**3)  # GB
            free_memory = total_memory - allocated_memory
            
            # If less than 2GB free, use very small batch size
            if free_memory < 2.0:
                return max(initial_batch_size // 4, 1)
            elif free_memory < 4.0:
                return max(initial_batch_size // 2, 1)
            else:
                return initial_batch_size
        except:
            return max(initial_batch_size // 2, 1)
    return initial_batch_size


def parse_tfrecord(example: tf.Tensor) -> Dict[str, tf.Tensor]:
    """Parse a serialized TFRecord example."""
    feature_description = {
        "l": tf.io.FixedLenFeature([], tf.string),
        "y": tf.io.FixedLenFeature([], tf.string),
    }
    return tf.io.parse_single_example(example, feature_description)





def process_single_tfrecord(path: Path) -> Tuple[List[str], List[str]]:
    """Process a single TFRecord file and return logs and labels."""
    logs = []
    labels = []
    
    dataset = tf.data.TFRecordDataset(str(path), compression_type="GZIP")
    dataset = dataset.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    for parsed in dataset:
        logs.append(parsed["l"].numpy().decode("utf-8"))
        labels.append(parsed["y"].numpy().decode("utf-8"))
    
    return logs, labels


def load_tfrecord_files(directory=PROCESSED_DIR, log_type_filter=None):
    """Load TFRecord files from directory into a DataFrame with optimized processing."""
    # Get list of all tfrecord files
    if log_type_filter:
        log_type_dir_path = directory / log_type_filter
        if not log_type_dir_path.exists():
            raise FileNotFoundError(f"No directory found for '{log_type_filter}'")
        tfrecord_files = list(log_type_dir_path.glob("*.tfrecord"))
        if not tfrecord_files:
            raise FileNotFoundError(f"No TFRecord files found for log type '{log_type_filter}'")
    else:
        tfrecord_files = []
        for log_dir_path in directory.iterdir():
            if log_dir_path.is_dir():
                tfrecord_files.extend(log_dir_path.glob("*.tfrecord"))
        if not tfrecord_files:
            raise FileNotFoundError(f"No TFRecord files found in {directory}")
    
    print(f"Loading {len(tfrecord_files)} TFRecord files...")
    
    # Process files in batches
    all_logs = []
    all_labels_json = []
    all_log_types = []
    
    spinner = Halo(text='Loading files', spinner='dots')
    spinner.start()
    
    for file_idx, file_path in enumerate(tfrecord_files):
        try:
            spinner.text = f"Loading file {file_idx+1}/{len(tfrecord_files)}: {file_path.name}"
            log_type = file_path.parent.name
            
            logs, labels = process_single_tfrecord(file_path)
            
            all_logs.extend(logs)
            all_labels_json.extend(labels)
            all_log_types.extend([log_type] * len(logs))
                
        except Exception as e:
            spinner.text = f"Error processing file {file_path}: {e}"
            spinner.fail()
            spinner = Halo(text='Loading files', spinner='dots')
            spinner.start()
    
    spinner.succeed(f"Loaded {len(all_logs)} log entries")
    
    return pd.DataFrame({
        'log': all_logs, 
        'label_json': all_labels_json,
        'log_type': all_log_types
    })


def normalize_label(label):
    """Normalize attack labels to ensure consistency."""
    if not label:
        return label
    return label.replace('-', '_').lower().strip()


def get_labels_from_json(label_json_str):
    """Extract labels from JSON string."""
    try:
        labels = json.loads(label_json_str)
        if not isinstance(labels, list):
            labels = [labels]
        return {normalize_label(label) for label in labels if label}
    except json.JSONDecodeError:
        return set()


def collect_unique_labels_from_data(df):
    """Extract all unique attack labels from the dataset efficiently."""
    all_unique_labels = set()
    
    spinner = Halo(text='Collecting unique labels', spinner='dots')
    spinner.start()
    
    for label_json_str in df['label_json']:
        all_unique_labels.update(get_labels_from_json(label_json_str))
    
    # Remove empty labels
    all_unique_labels.discard('')
    all_unique_labels.discard(None)
    
    spinner.succeed(f"Found {len(all_unique_labels)} unique attack types")
    return sorted(list(all_unique_labels))


def create_binary_label_vector(label_json_str, all_attack_types):
    """Create binary vector representation for multi-label classification."""
    labels = get_labels_from_json(label_json_str)
    
    # Binary vector: [0 1 0] means only second class is present
    binary_vector = np.zeros(len(all_attack_types), dtype=np.int8)  # Use int8 for memory efficiency
    
    # Vectorized approach for better performance
    if labels:
        attack_indices = [i for i, attack in enumerate(all_attack_types) if attack in labels]
        binary_vector[attack_indices] = 1
    
    return binary_vector


def display_data_distribution(df, log_type_name="all combined"):
    """Calculate and display data distribution statistics."""
    print(f"\n{'='*20} Data Distribution for '{log_type_name}' {'='*20}")
    
    total_logs = len(df)
    print(f"Total log entries: {total_logs}")
    
    # Extract all unique labels
    all_labels_count = {}
    normal_count = 0
    attack_count = 0
    
    spinner = Halo(text='Analyzing data distribution', spinner='dots')
    spinner.start()
    
    for label_json_str in df['label_json']:
        labels = get_labels_from_json(label_json_str)
        if labels:
            for label in labels:
                all_labels_count[label] = all_labels_count.get(label, 0) + 1
            attack_count += 1
        else:
            normal_count += 1
    
    spinner.succeed("Data distribution analysis complete")
    
    # Display attack distribution
    if all_labels_count:
        print("\nAttack type distribution:")
        for attack, count in sorted(all_labels_count.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_logs) * 100
            print(f"  {attack}: {count} occurrences ({percentage:.2f}%)")
    
    # Display normal vs attack statistics
    attack_percentage = (attack_count / total_logs) * 100 if total_logs > 0 else 0
    normal_percentage = (normal_count / total_logs) * 100 if total_logs > 0 else 0
    
    print(f"\nLogs with attacks: {attack_count} ({attack_percentage:.2f}%)")
    print(f"Normal logs: {normal_count} ({normal_percentage:.2f}%)")
    
    # Display log type distribution if processing combined dataset
    if log_type_name == "all combined":
        print("\nLog type distribution:")
        type_counts = df['log_type'].value_counts()
        for log_type, count in type_counts.items():
            percentage = (count / total_logs) * 100
            print(f"  {log_type}: {count} entries ({percentage:.2f}%)")
    
    print(f"{'='*70}\n")
    return attack_count, normal_count


def estimate_dataset_size(directory=PROCESSED_DIR, log_type_filter=None):
    """Estimate dataset size by examining TFRecord file sizes."""
    # Get list of all tfrecord files
    if log_type_filter:
        log_type_dir_path = directory / log_type_filter
        if not log_type_dir_path.exists():
            return 0, "unknown"
        tfrecord_files = list(log_type_dir_path.glob("*.tfrecord"))
    else:
        tfrecord_files = []
        for log_dir_path in directory.iterdir():
            if log_dir_path.is_dir():
                tfrecord_files.extend(log_dir_path.glob("*.tfrecord"))
    
    if not tfrecord_files:
        return 0, "unknown"
    
    # Estimate based on file sizes (rough approximation: 100 bytes per log entry on average)
    total_size = sum(f.stat().st_size for f in tfrecord_files)
    estimated_entries = total_size // 100  # rough estimate
    
    # Categorize dataset size
    if estimated_entries < SMALL_DATASET_THRESHOLD:
        category = "small"
    elif estimated_entries < MEDIUM_DATASET_THRESHOLD:
        category = "medium"
    elif estimated_entries < LARGE_DATASET_THRESHOLD:
        category = "large"
    else:
        category = "very_large"
    
    return estimated_entries, category


def get_performance_config(dataset_size_category, device_type="cpu"):
    """Get optimized performance configuration based on dataset size and device."""
    config = PERF_CONFIG[dataset_size_category].copy()
    
    # Adjust for device capabilities
    if device_type == "mps":  # M2 GPU
        config['batch_size'] = min(config['batch_size'] * 2, 32)  # Double batch size for GPU
        config['workers'] = min(config['workers'], 4)  # MPS works best with fewer workers
    elif device_type == "cuda":
        # Be more conservative with CUDA batch sizes due to memory constraints
        config['batch_size'] = min(config['batch_size'], 4)  # Keep small batch sizes for CUDA
        config['workers'] = min(config['workers'], 2)  # Reduce workers to save memory
        config['clear_freq'] = max(config['clear_freq'] // 2, 1)  # Clear memory more frequently
    
    # Adjust for system memory
    memory_gb = psutil.virtual_memory().total / (1024**3)
    if memory_gb < 8:  # Less than 8GB RAM
        config['batch_size'] = max(config['batch_size'] // 2, 1)
        config['workers'] = max(config['workers'] // 2, 1)
    elif memory_gb > 32:  # More than 32GB RAM and not CUDA
        if device_type != "cuda":
            config['batch_size'] = min(config['batch_size'] * 2, 64)
            config['workers'] = min(config['workers'] * 2, 8)
    
    return config


def estimate_processing_time(num_entries, batch_size, device_type="cpu"):
    """Estimate processing time based on dataset size and hardware."""
    # Base processing rates (entries per second) - empirically determined
    base_rates = {
        "cpu": 15,      # entries per second on CPU
        "mps": 45,      # entries per second on M2 GPU
        "cuda": 60      # entries per second on CUDA GPU
    }
    
    rate = base_rates.get(device_type, base_rates["cpu"])
    
    # Adjust for batch size efficiency
    efficiency_factor = min(batch_size / 8.0, 2.0)  # Optimal around batch size 8-16
    adjusted_rate = rate * efficiency_factor
    
    estimated_seconds = num_entries / adjusted_rate
    
    # Format time estimate
    if estimated_seconds < 60:
        return f"{estimated_seconds:.0f} seconds"
    elif estimated_seconds < 3600:
        return f"{estimated_seconds/60:.1f} minutes"
    else:
        return f"{estimated_seconds/3600:.1f} hours"


def find_available_log_types():
    """Find available log types in the processed directory."""
    if not PROCESSED_DIR.exists():
        return []
    return sorted([path.name for path in PROCESSED_DIR.iterdir() 
                  if path.is_dir() and list(path.glob("*.tfrecord"))])

def generate_data_hash(df):
    """Generate a hash of the dataset for checkpoint validation."""
    # Create a hash based on log content and labels
    content = f"{len(df)}_{df['log'].iloc[0] if len(df) > 0 else ''}_{df['log'].iloc[-1] if len(df) > 0 else ''}"
    return hashlib.md5(content.encode()).hexdigest()[:16]

def save_checkpoint(log_type: str, stage: str, data: dict, data_hash: str):
    """Save checkpoint for resumeable processing."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_file = CHECKPOINT_DIR / f"{log_type}_{stage}_{data_hash}.pkl"
    
    checkpoint_data = {
        'log_type': log_type,
        'stage': stage,
        'data_hash': data_hash,
        'timestamp': time.time(),
        'data': data
    }
    
    with open(checkpoint_file, 'wb') as f:
        pickle.dump(checkpoint_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"💾 Checkpoint saved: {checkpoint_file.name}")
    return checkpoint_file

def load_checkpoint(log_type: str, stage: str, data_hash: str) -> Optional[dict]:
    """Load checkpoint if it exists and matches the data hash."""
    if not CHECKPOINT_DIR.exists():
        return None
    
    checkpoint_file = CHECKPOINT_DIR / f"{log_type}_{stage}_{data_hash}.pkl"
    
    if checkpoint_file.exists():
        try:
            with open(checkpoint_file, 'rb') as f:
                checkpoint_data = pickle.load(f)
            
            # Validate checkpoint
            if (checkpoint_data['log_type'] == log_type and 
                checkpoint_data['stage'] == stage and 
                checkpoint_data['data_hash'] == data_hash):
                
                age_hours = (time.time() - checkpoint_data['timestamp']) / 3600
                print(f"📂 Found checkpoint: {checkpoint_file.name} (age: {age_hours:.1f}h)")
                return checkpoint_data['data']
        except Exception as e:
            print(f"⚠️  Checkpoint loading failed: {e}")
            # Remove corrupted checkpoint
            checkpoint_file.unlink(missing_ok=True)
    
    return None

def cleanup_old_checkpoints(log_type: str, keep_latest: int = 3):
    """Clean up old checkpoints, keeping only the latest ones."""
    if not CHECKPOINT_DIR.exists():
        return
    
    # Find all checkpoints for this log type
    pattern = f"{log_type}_*.pkl"
    checkpoints = list(CHECKPOINT_DIR.glob(pattern))
    
    if len(checkpoints) > keep_latest:
        # Sort by modification time, keep latest
        checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        for old_checkpoint in checkpoints[keep_latest:]:
            old_checkpoint.unlink(missing_ok=True)
            print(f"🗑️  Cleaned up old checkpoint: {old_checkpoint.name}")

def check_existing_outputs(log_type: str) -> dict:
    """Check if outputs already exist for this log type."""
    output_dir = OUTPUT_DIR / log_type
    
    status = {
        'log_embeddings': False,
        'label_vectors': False,
        'attack_types': False,
        'visualization': False,
        'complete': False
    }
    
    if output_dir.exists():
        status['log_embeddings'] = (output_dir / f"log_{log_type}.pkl").exists()
        status['label_vectors'] = (output_dir / f"label_{log_type}.pkl").exists()
        status['attack_types'] = (output_dir / f"attack_types_{log_type}.txt").exists()
        status['visualization'] = (output_dir / "visualization.png").exists()
        status['complete'] = all([status['log_embeddings'], status['label_vectors'], 
                                status['attack_types'], status['visualization']])
    
    return status


# ---------------------------------------------------------------------------
# Dataset Class for BERT
# ---------------------------------------------------------------------------

class LogBERTEmbeddingDataset(Dataset):
    """Dataset class for LogBERT embedding extraction from preprocessed TFRecord data."""
    
    def __init__(self, texts: List[str], tokenizer: BertTokenizer, max_length: int = MAX_SEQ_LENGTH):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # The text is already preprocessed from TFRecord files
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'idx': idx
        }


def extract_bert_embeddings(df, device, performance_config=None):
    """Extract enhanced BERT embeddings capturing multiple features for better anomaly detection."""
    if performance_config is None:
        performance_config = {'batch_size': BATCH_SIZE, 'workers': NUM_WORKERS, 'clear_freq': 50}
    
    num_entries = len(df)
    batch_size = performance_config['batch_size']
    num_workers = performance_config['workers']
    clear_freq = performance_config['clear_freq']
    
    # Estimate processing time
    time_estimate = estimate_processing_time(num_entries, batch_size, device.type)
    
    spinner = Halo(text=f'Initializing BERT model (ETA: {time_estimate})', spinner='dots')
    spinner.start()
    
    # Clear memory before loading model
    clear_memory(device)
    
    # Load pre-trained BERT model and tokenizer with attention implementation fix
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased', attn_implementation="eager").to(device)
    model.eval()
    
    # Clear memory after loading model
    clear_memory(device)
    
    # Adjust batch size based on available GPU memory
    adjusted_batch_size = adjust_batch_size_for_memory(batch_size, device)
    if adjusted_batch_size != batch_size:
        print(f"Adjusted batch size from {batch_size} to {adjusted_batch_size} due to memory constraints")
        batch_size = adjusted_batch_size
    
    spinner.succeed(f"BERT model loaded - Processing {num_entries:,} entries (batch_size={batch_size}, workers={num_workers})")
    
    # Create dataset and dataloader with optimized settings
    dataset = LogBERTEmbeddingDataset(df['log'].tolist(), tokenizer)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type in ["cuda", "mps"],
        persistent_workers=num_workers > 0
    )
    
    # Extract multiple types of embeddings
    all_cls_embeddings = []
    all_mean_embeddings = []
    all_max_embeddings = []
    all_attention_features = []
    start_time = time.time()
    
    spinner = Halo(text=f'Extracting enhanced BERT embeddings (ETA: {time_estimate})', spinner='dots')
    spinner.start()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Update progress with time estimation
            if batch_idx % 10 == 0:
                elapsed = time.time() - start_time
                if batch_idx > 0:
                    rate = (batch_idx * batch_size) / elapsed
                    remaining_entries = num_entries - (batch_idx * batch_size)
                    eta_seconds = remaining_entries / rate if rate > 0 else 0
                    
                    if eta_seconds < 60:
                        eta_str = f"{eta_seconds:.0f}s"
                    elif eta_seconds < 3600:
                        eta_str = f"{eta_seconds/60:.1f}m"
                    else:
                        eta_str = f"{eta_seconds/3600:.1f}h"
                    
                    progress_pct = (batch_idx * batch_size) / num_entries * 100
                    spinner.text = f'Extracting enhanced embeddings: {progress_pct:.1f}% (ETA: {eta_str})'
                else:
                    spinner.text = f'Extracting embeddings: batch {batch_idx+1}/{len(dataloader)}'
            
            # Clear memory before processing each batch for CUDA
            if device.type == "cuda" and batch_idx % 5 == 0:
                clear_memory(device)
            
            try:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                # Get outputs with attention weights
                outputs = model(
                    input_ids=input_ids, 
                    attention_mask=attention_mask,
                    output_attentions=True
                )
            except torch.cuda.OutOfMemoryError:
                # If we run out of memory, clear everything and process samples one by one
                spinner.text = f'Memory error at batch {batch_idx+1}, switching to single-sample processing'
                clear_memory(device)
                
                # Process each sample in the batch individually
                batch_outputs = []
                for i in range(len(batch['input_ids'])):
                    single_input = batch['input_ids'][i:i+1].to(device)
                    single_mask = batch['attention_mask'][i:i+1].to(device)
                    
                    single_output = model(
                        input_ids=single_input,
                        attention_mask=single_mask,
                        output_attentions=True
                    )
                    batch_outputs.append(single_output)
                    
                    # Clear after each sample
                    single_input = single_input.cpu()
                    single_mask = single_mask.cpu()
                    clear_memory(device)
                
                # Combine outputs
                outputs = type(batch_outputs[0])(
                    last_hidden_state=torch.cat([out.last_hidden_state for out in batch_outputs], dim=0),
                    attentions=tuple(torch.cat([out.attentions[i] for out in batch_outputs], dim=0) 
                                   for i in range(len(batch_outputs[0].attentions)))
                )
            
            # 1. CLS token embeddings (global context)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_cls_embeddings.append(cls_embeddings)
            
            # 2. Mean pooling (average representation)
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            mean_embeddings = (sum_embeddings / sum_mask).cpu().numpy()
            all_mean_embeddings.append(mean_embeddings)
            
            # 3. Max pooling (capture key features)
            # Set padding tokens to large negative value before max pooling
            token_embeddings_masked = token_embeddings.clone()
            token_embeddings_masked[input_mask_expanded == 0] = -1e9
            max_embeddings = torch.max(token_embeddings_masked, 1)[0].cpu().numpy()
            all_max_embeddings.append(max_embeddings)
            
            # 4. Attention-based features (which tokens are important)
            # Average attention from last layer, focusing on CLS token's attention
            last_attention = outputs.attentions[-1]  # [batch, heads, seq, seq]
            # Average over heads and take CLS token's attention to other tokens
            cls_attention = last_attention.mean(dim=1)[:, 0, :].cpu().numpy()
            
            # Get top-k attention scores as features
            top_k = 10
            batch_attention_features = []
            for i in range(cls_attention.shape[0]):
                # Get actual sequence length for this sample
                seq_len = attention_mask[i].sum().item()
                # Only consider attention to actual tokens (not padding)
                valid_attention = cls_attention[i, :seq_len]
                
                if len(valid_attention) > 1:  # Ensure we have more than just CLS token
                    # Sort and get top-k values (excluding CLS token itself)
                    top_values = np.sort(valid_attention[1:])[-top_k:]
                    # Pad if necessary
                    if len(top_values) < top_k:
                        top_values = np.pad(top_values, (0, top_k - len(top_values)), 'constant')
                else:
                    top_values = np.zeros(top_k)
                
                batch_attention_features.append(top_values)
            
            all_attention_features.append(np.array(batch_attention_features))
            
            # Clear memory periodically based on performance config
            if batch_idx % clear_freq == 0:
                clear_memory(device)
            
            # Move tensors to CPU immediately after processing to free GPU memory
            if device.type == "cuda":
                input_ids = input_ids.cpu()
                attention_mask = attention_mask.cpu()
    
    total_time = time.time() - start_time
    rate = num_entries / total_time
    spinner.succeed(f"Enhanced BERT embedding extraction complete ({total_time:.1f}s, {rate:.1f} entries/sec)")
    
    # Concatenate all features
    cls_features = np.vstack(all_cls_embeddings).astype(np.float32)
    mean_features = np.vstack(all_mean_embeddings).astype(np.float32)
    max_features = np.vstack(all_max_embeddings).astype(np.float32)
    attention_features = np.vstack(all_attention_features).astype(np.float32)
    
    # Combine all features into a single embedding
    # This creates a richer representation while maintaining a single vector per log
    combined_embeddings = np.hstack([
        cls_features,      # 768D - global context
        mean_features,     # 768D - average meaning
        max_features,      # 768D - key features
        attention_features # 10D - attention patterns
    ])  # Total: 2314D
    
    spinner.text = f"Combined embedding shape: {combined_embeddings.shape} (2314D per log)"
    
    # Clear model from memory
    del model, tokenizer
    clear_memory(device)
    
    return combined_embeddings


def process_embeddings(df, device, use_global_attack_list=False, performance_config=None, log_type=None):
    """Process logs and create BERT embeddings with binary label vectors - resumeable."""
    data_hash = generate_data_hash(df)
    
    # Check for checkpoint
    if log_type:
        checkpoint_data = load_checkpoint(log_type, "embeddings", data_hash)
        if checkpoint_data:
            print("✅ Resuming from embeddings checkpoint")
            # Reconstruct dataframe from checkpoint
            df['log_embedding'] = checkpoint_data['log_embeddings']
            df['binary_labels'] = checkpoint_data['binary_labels']
            if 'attack_types' in checkpoint_data:
                df.attrs['attack_types'] = checkpoint_data['attack_types']
            if 'log_type_to_attacks' in checkpoint_data:
                df.attrs['log_type_to_attacks'] = checkpoint_data['log_type_to_attacks']
            return df
    
    # Extract BERT embeddings with performance optimization
    log_embeddings = extract_bert_embeddings(df, device, performance_config)
    df['log_embedding'] = list(log_embeddings)
    
    # Process labels
    spinner = Halo(text="Processing binary label vectors", spinner='dots')
    spinner.start()
    
    if use_global_attack_list:
        # Use all labels across all log types
        attack_types = collect_unique_labels_from_data(df)
        
        binary_labels = []
        for label_json in df['label_json']:
            binary_vector = create_binary_label_vector(label_json, attack_types)
            binary_labels.append(binary_vector)
        
        df['binary_labels'] = binary_labels
        df.attrs['attack_types'] = attack_types
        
        # Save checkpoint
        if log_type:
            checkpoint_data = {
                'log_embeddings': list(log_embeddings),
                'binary_labels': binary_labels,
                'attack_types': attack_types
            }
            save_checkpoint(log_type, "embeddings", checkpoint_data, data_hash)
    else:
        # Process by log type
        log_type_to_attacks = {}
        for lt, group_df in df.groupby('log_type'):
            spinner.text = f"Processing log type: {lt}"
            log_type_to_attacks[lt] = collect_unique_labels_from_data(group_df)
        
        binary_labels = []
        for idx, row in df.iterrows():
            lt = row['log_type']
            if lt in log_type_to_attacks:
                binary_vector = create_binary_label_vector(row['label_json'], log_type_to_attacks[lt])
            else:
                binary_vector = np.array([], dtype=np.int8)
            binary_labels.append(binary_vector)
        
        df['binary_labels'] = binary_labels
        df.attrs['log_type_to_attacks'] = log_type_to_attacks
        
        # Save checkpoint
        if log_type:
            checkpoint_data = {
                'log_embeddings': list(log_embeddings),
                'binary_labels': binary_labels,
                'log_type_to_attacks': log_type_to_attacks
            }
            save_checkpoint(log_type, "embeddings", checkpoint_data, data_hash)
    
    spinner.succeed("Label processing complete")
    return df


def visualize_embeddings(df, output_file=None):
    """Create t-SNE visualization with balanced class sampling for performance and minority visibility."""
    # Parameters for sampling
    MAX_TOTAL_POINTS = 50000
    MAX_POINTS_PER_CLASS = 1500

    spinner = Halo(text="Preparing visualization data", spinner='dots')
    spinner.start()

    # Build visualization labels
    spinner.text = "Generating labels for visualization"
    viz_labels = []
    for label_json_str in df['label_json']:
        labels = get_labels_from_json(label_json_str)
        if not labels:
            viz_labels.append("normal")
        else:
            viz_labels.append(", ".join(sorted(labels)))

    # Attach labels temporarily to the dataframe
    df = df.copy()
    df['viz_label'] = viz_labels

    # Balanced sampling
    spinner.text = "Applying balanced sampling to limit dataset size"
    np.random.seed(42)

    selected_indices = []
    label_to_indices = {}
    for idx, lbl in enumerate(viz_labels):
        label_to_indices.setdefault(lbl, []).append(idx)

    for lbl, indices in label_to_indices.items():
        if len(indices) > MAX_POINTS_PER_CLASS:
            sampled = np.random.choice(indices, MAX_POINTS_PER_CLASS, replace=False)
            selected_indices.extend(sampled)
        else:
            selected_indices.extend(indices)

    if len(selected_indices) > MAX_TOTAL_POINTS:
        selected_indices = list(np.random.choice(selected_indices, MAX_TOTAL_POINTS, replace=False))

    # Gather embeddings and labels for the sampled indices
    embeddings = np.vstack([df.at[i, 'log_embedding'] for i in selected_indices]).astype(np.float32)
    sampled_labels = [viz_labels[i] for i in selected_indices]
    sampled_log_types = [df.at[i, 'log_type'] for i in selected_indices]

    spinner.text = f"Running t-SNE on {len(embeddings)} sampled points"

    # Choose perplexity based on size
    perplexity = min(50, max(5, len(embeddings)//1000))

    # Execute t-SNE
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        n_iter=500,
        learning_rate='auto',
        init='pca',
        method='barnes_hut',
        random_state=42
    )

    reduced = tsne.fit_transform(embeddings)

    # Prepare DataFrame for plotting
    df_plot = pd.DataFrame({
        'x': reduced[:, 0],
        'y': reduced[:, 1],
        'label': sampled_labels,
        'log_type': sampled_log_types
    })

    spinner.succeed("t-SNE dimensionality reduction complete")

    # Count visualization labels
    label_counts = df_plot['label'].value_counts()
    print(f"\nVisualization showing ALL {len(label_counts)} unique label combinations:")
    for label, count in label_counts.head(10).items():
        percentage = (count / len(df_plot)) * 100
        print(f"  {label}: {count} ({percentage:.2f}%)")
    
    if len(label_counts) > 10:
        print(f"  ... and {len(label_counts) - 10} more label combinations")
    
    # Create color palette
    unique_labels = sorted(df_plot['label'].unique())
    palette = sns.color_palette("husl", len(unique_labels))
    color_map = {label: palette[i] for i, label in enumerate(unique_labels)}
    
    if "normal" in color_map:
        color_map["normal"] = "green"
    
    # Create scatter plot
    spinner = Halo(text="Creating visualization plot", spinner='dots')
    spinner.start()
    
    plt.figure(figsize=(16, 10))
    
    for label in unique_labels:
        mask = df_plot['label'] == label
        subset = df_plot[mask]
        
        plt.scatter(
            subset['x'], 
            subset['y'], 
            c=[color_map[label]], 
            label=label,
            alpha=0.6,
            s=20,
            edgecolors='none'
        )
    
    plt.title(f't-SNE Visualization: LogBERT CLS Embeddings (All {len(unique_labels)} Classes)', fontsize=14)
    plt.xlabel('t-SNE Component 1', fontsize=12)
    plt.ylabel('t-SNE Component 2', fontsize=12)
    
    # Optimize legend for many classes
    if len(unique_labels) <= 20:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        plt.tight_layout(rect=[0,0,0.85,1])
    else:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=6, ncol=2)
        plt.tight_layout(rect=[0,0,0.8,1])
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        spinner.succeed(f"Saved visualization to {output_file}")
    else:
        plt.show()
        spinner.succeed("Displayed visualization")
    
    plt.close()
    
    # Clear memory
    del embeddings, reduced, df_plot


def save_embeddings_and_labels(df, output_dir, log_type_name):
    """Save only log embeddings and label vectors as requested - matching FastText format."""
    spinner = Halo(text=f"Saving embeddings for {log_type_name}", spinner='dots')
    spinner.start()
    
    # Extract and save log embeddings as log_<type>.pkl
    log_embeddings = np.vstack(df['log_embedding'].tolist()).astype(np.float32)
    log_filename = f"log_{log_type_name}.pkl"
    
    spinner.text = f"Saving log embeddings to {log_filename}"
    with open(output_dir / log_filename, 'wb') as f:
        pickle.dump(log_embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Extract and save binary label vectors as label_<type>.pkl
    if 'binary_labels' in df.columns and len(df['binary_labels']) > 0:
        # Filter out empty arrays and stack efficiently
        valid_vectors = [vec for vec in df['binary_labels'] if len(vec) > 0]
        
        if valid_vectors:
            binary_vectors = np.vstack(valid_vectors).astype(np.int8)
            label_filename = f"label_{log_type_name}.pkl"
            
            # Get class mapping information
            classes = []
            if 'attack_types' in df.attrs:
                classes = df.attrs['attack_types']
            elif 'log_type_to_attacks' in df.attrs and log_type_name in df.attrs['log_type_to_attacks']:
                classes = df.attrs['log_type_to_attacks'][log_type_name]
            
            # Create simplified label data (removed example and column_explanation)
            label_data = {
                'vectors': binary_vectors,
                'classes': classes,
                'description': 'Binary multi-label vectors where [0 1 0] means only the second class is present'
            }
            
            spinner.text = f"Saving label vectors to {label_filename}"
            with open(output_dir / label_filename, 'wb') as f:
                pickle.dump(label_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Save attack types and details in separate text file
            attack_info_filename = f"attack_types_{log_type_name}.txt"
            spinner.text = f"Saving attack type details to {attack_info_filename}"
            
            with open(output_dir / attack_info_filename, 'w', encoding='utf-8') as f:
                f.write(f"Attack Types and Column Mapping for {log_type_name}\n")
                f.write("=" * 60 + "\n\n")
                f.write("Binary Vector Format:\n")
                f.write("- Each row is a binary vector representing one log entry\n")
                f.write("- Vector format: [0 1 0] means only the second attack type is present\n")
                f.write("- Multiple attacks can be present: [1 0 1] means first and third attacks\n\n")
                f.write(f"Total attack types: {len(classes)}\n")
                f.write(f"Vector dimension: {len(classes)}\n\n")
                
                if classes:
                    f.write("Column Mapping (Index -> Attack Type):\n")
                    f.write("-" * 40 + "\n")
                    for i, attack_type in enumerate(classes):
                        f.write(f"Column {i:2d}: {attack_type}\n")
                    
                    f.write("\nExample Interpretations:\n")
                    f.write("-" * 25 + "\n")
                    if len(classes) >= 1:
                        example1 = np.zeros(len(classes), dtype=np.int8)
                        example1[0] = 1
                        f.write(f"{list(example1)} -> Only '{classes[0]}' attack present\n")
                    
                    if len(classes) >= 2:
                        example2 = np.zeros(len(classes), dtype=np.int8)
                        example2[1] = 1
                        f.write(f"{list(example2)} -> Only '{classes[1]}' attack present\n")
                        
                        example3 = np.zeros(len(classes), dtype=np.int8)
                        example3[0] = 1
                        example3[1] = 1
                        f.write(f"{list(example3)} -> Both '{classes[0]}' and '{classes[1]}' attacks present\n")
                    
                    all_zeros = np.zeros(len(classes), dtype=np.int8)
                    f.write(f"{list(all_zeros)} -> Normal log (no attacks)\n")
                else:
                    f.write("No attack types found for this log type.\n")
                
                f.write(f"\nGenerated by Enhanced LogBERT embeddings extraction\n")
                f.write(f"Compatible with FastText embedding format\n")
                f.write(f"Embedding dimension: 2314D (Enhanced BERT vectors)\n")
                f.write(f"  - CLS token: 768D (global context)\n")
                f.write(f"  - Mean pooling: 768D (average representation)\n")
                f.write(f"  - Max pooling: 768D (key features)\n")
                f.write(f"  - Attention features: 10D (important token patterns)\n")
                
            spinner.succeed(f"Saved {log_filename}, {label_filename}, and {attack_info_filename}")
            
            # Print summary
            print(f"\nSaved files for {log_type_name}:")
            print(f"  - {log_filename}: Log embeddings {log_embeddings.shape} (2314D enhanced BERT vectors)")
            print(f"  - {label_filename}: Binary label vectors {binary_vectors.shape}")
            print(f"  - {attack_info_filename}: Attack types and column mapping details")
            print(f"  - Classes: {classes}")
        else:
            spinner.warn(f"No valid binary labels found for {log_type_name}")
    else:
        spinner.warn(f"No binary labels found for {log_type_name}, saved only log embeddings")


def main():
    parser = argparse.ArgumentParser(description="Generate LogBERT CLS embeddings for log data - Resumeable")
    parser.add_argument("--log-type", type=str, default=None, help="Process only this specific log type")
    parser.add_argument("--sample-size", type=int, default=None, help="Process only this many log entries (for testing)")
    parser.add_argument("--force-restart", action="store_true", help="Force restart processing (ignore existing outputs)")
    parser.add_argument("--clean-checkpoints", action="store_true", help="Clean up all checkpoints before starting")
    args = parser.parse_args()
    
    # Clean checkpoints if requested
    if args.clean_checkpoints:
        import shutil
        if CHECKPOINT_DIR.exists():
            shutil.rmtree(CHECKPOINT_DIR)
            print("🗑️  Cleaned up all checkpoints")
    
    # Override completion check if force restart
    if args.force_restart:
        print("🔄 Force restart enabled - will reprocess all data")

    # Ensure output directories exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Get device
    device = get_device()
    
    # Set multiprocessing start method for compatibility
    mp.set_start_method('spawn', force=True)
    
    # Find available log types and estimate sizes
    spinner = Halo(text="Analyzing available log types and estimating sizes", spinner='dots')
    spinner.start()
    available_types = find_available_log_types()
    if not available_types:
        spinner.fail("No log types with data found.")
        return
    
    # Estimate sizes for each log type
    log_type_info = {}
    total_estimated = 0
    for log_type in available_types:
        estimated_size, category = estimate_dataset_size(log_type_filter=log_type)
        log_type_info[log_type] = {'size': estimated_size, 'category': category}
        total_estimated += estimated_size
    
    spinner.succeed(f"Found {len(available_types)} log types (total ~{total_estimated:,} entries)")
    
    # Display size analysis
    print("\n" + "="*60)
    print("DATASET SIZE ANALYSIS")
    print("="*60)
    for log_type in sorted(available_types, key=lambda x: log_type_info[x]['size']):
        info = log_type_info[log_type]
        print(f"{log_type:15} | ~{info['size']:,} entries ({info['category']} dataset)")
    print("="*60)
    
    if args.log_type:
        if args.log_type not in available_types:
            print(f"Log type '{args.log_type}' not found. Available types: {', '.join(available_types)}")
            return
        types_to_process = [args.log_type]
        run_combined = False
        print(f"\nProcessing single log type: {args.log_type}")
    else:
        # Sort by size (smallest first) for efficient processing
        types_to_process = sorted(available_types, key=lambda x: log_type_info[x]['size'])
        run_combined = True
        print(f"\nProcessing all log types (starting with smallest for efficiency)")
    
    # Estimate total processing time if processing all
    if run_combined and not args.sample_size:
        combined_estimate, combined_category = estimate_dataset_size()
        combined_config = get_performance_config(combined_category, device.type)
        combined_time = estimate_processing_time(combined_estimate, combined_config['batch_size'], device.type)
        print(f"Estimated total processing time: {combined_time}")
        print("Tip: Use --sample-size N to test with smaller dataset first")

    # Process individual log types
    for log_type in types_to_process:
        print(f"\n{'='*50}\nProcessing log type: {log_type}\n{'='*50}")
        
        # Check if already completed (unless force restart)
        output_status = check_existing_outputs(log_type)
        if output_status['complete'] and not args.sample_size and not args.force_restart:
            print(f"✅ {log_type} already completed. Skipping.")
            print(f"   Files: log✓ labels✓ attack_types✓ visualization✓")
            continue
        elif output_status['log_embeddings'] and output_status['label_vectors']:
            print(f"🔄 {log_type} partially completed. Only creating visualization...")
            try:
                # Load existing data for visualization only
                output_dir = OUTPUT_DIR / log_type
                with open(output_dir / f"log_{log_type}.pkl", 'rb') as f:
                    embeddings = pickle.load(f)
                with open(output_dir / f"label_{log_type}.pkl", 'rb') as f:
                    label_data = pickle.load(f)
                
                # Create a minimal df for visualization
                df_viz = pd.DataFrame({
                    'log_embedding': list(embeddings),
                    'label_json': ['[]'] * len(embeddings),  # Dummy labels for viz
                    'log_type': [log_type] * len(embeddings)
                })
                
                visualize_embeddings(df_viz, output_file=output_dir / "visualization.png")
                print(f"✅ Visualization completed for {log_type}")
                continue
            except Exception as e:
                print(f"⚠️  Could not create visualization from existing data: {e}")
                # Continue with full processing
        
        try:
            # Load data for this log type
            df = load_tfrecord_files(log_type_filter=log_type)
            if df.empty:
                print(f"No data for log type '{log_type}'. Skipping.")
                continue

            # Sample data if requested
            original_size = len(df)
            if args.sample_size and len(df) > args.sample_size:
                print(f"Sampling {args.sample_size} entries from {len(df)} total entries")
                df = df.sample(n=args.sample_size, random_state=42).reset_index(drop=True)

            # Get performance configuration based on actual dataset size
            dataset_size = len(df)
            if dataset_size < SMALL_DATASET_THRESHOLD:
                size_category = "small"
            elif dataset_size < MEDIUM_DATASET_THRESHOLD:
                size_category = "medium"
            elif dataset_size < LARGE_DATASET_THRESHOLD:
                size_category = "large"
            else:
                size_category = "very_large"
            
            perf_config = get_performance_config(size_category, device.type)
            time_estimate = estimate_processing_time(dataset_size, perf_config['batch_size'], device.type)
            
            print(f"Dataset size: {dataset_size:,} entries ({size_category}) - ETA: {time_estimate}")
            print(f"Performance config: batch_size={perf_config['batch_size']}, workers={perf_config['workers']}")

            # Display data distribution
            display_data_distribution(df, log_type)

            # Process embeddings with performance optimization and checkpointing
            df = process_embeddings(df, device, use_global_attack_list=False, 
                                 performance_config=perf_config, log_type=log_type)
            
            # Save outputs
            output_dir = OUTPUT_DIR / log_type
            output_dir.mkdir(parents=True, exist_ok=True)
            
            save_embeddings_and_labels(df, output_dir, log_type)
            
            # Create visualization
            visualize_embeddings(
                df, 
                output_file=output_dir / "visualization.png"
            )
            
            # Clean up old checkpoints after successful completion
            cleanup_old_checkpoints(log_type)
            
            # Clear memory
            del df
            clear_memory(device)
            
            print(f"✅ Completed processing {log_type}")
            
        except KeyboardInterrupt:
            print(f"\n⚠️  Processing interrupted for {log_type}. Checkpoint saved.")
            break
        except Exception as e:
            print(f"❌ Error processing log type {log_type}: {e}")
            import traceback
            traceback.print_exc()

    # Process combined model if requested
    if run_combined:
        print(f"\n{'='*50}\nProcessing all log types combined\n{'='*50}")
        
        # Check if combined already exists (unless force restart)
        combined_status = check_existing_outputs("all_combined")
        if combined_status['complete'] and not args.sample_size and not args.force_restart:
            print(f"✅ Combined model already completed. Skipping.")
            print(f"   Files: log✓ labels✓ attack_types✓ visualization✓")
        else:
            try:
                # Load all data
                df_all = load_tfrecord_files()
                if df_all.empty:
                    print("No data found for combined log types")
                else:
                    # Sample data if requested
                    if args.sample_size and len(df_all) > args.sample_size:
                        print(f"Sampling {args.sample_size} entries from {len(df_all)} total entries")
                        df_all = df_all.sample(n=args.sample_size, random_state=42).reset_index(drop=True)
                    
                    # Get performance configuration
                    dataset_size = len(df_all)
                    if dataset_size < SMALL_DATASET_THRESHOLD:
                        size_category = "small"
                    elif dataset_size < MEDIUM_DATASET_THRESHOLD:
                        size_category = "medium"
                    elif dataset_size < LARGE_DATASET_THRESHOLD:
                        size_category = "large"
                    else:
                        size_category = "very_large"
                    
                    perf_config = get_performance_config(size_category, device.type)
                    time_estimate = estimate_processing_time(dataset_size, perf_config['batch_size'], device.type)
                    
                    print(f"Combined dataset size: {dataset_size:,} entries ({size_category}) - ETA: {time_estimate}")
                    print(f"Performance config: batch_size={perf_config['batch_size']}, workers={perf_config['workers']}")
                    
                    display_data_distribution(df_all, "all combined")
                    df_all = process_embeddings(df_all, device, use_global_attack_list=True, 
                                             performance_config=perf_config, log_type="all_combined")
                    
                    # Save combined outputs
                    save_embeddings_and_labels(df_all, OUTPUT_DIR, "all_combined")
                    
                    # Create visualization
                    visualize_embeddings(
                        df_all,
                        output_file=OUTPUT_DIR / "visualization_all_combined.png"
                    )
                    
                    # Clean up old checkpoints
                    cleanup_old_checkpoints("all_combined")
                    
                    # Clear memory
                    del df_all
                    clear_memory(device)
                    
                    print(f"✅ Completed processing combined model")
            
            except KeyboardInterrupt:
                print(f"\n⚠️  Combined processing interrupted. Checkpoint saved.")
            except Exception as e:
                print(f"❌ Error processing combined log types: {e}")
                import traceback
                traceback.print_exc()
    
    spinner = Halo(spinner='dots', text='Completing processing')
    spinner.start()
    spinner.succeed("Enhanced LogBERT embedding processing complete!")
    
    print("\n🎉 Processing Summary:")
    print("=" * 60)
    print("✅ Resumeable: Checkpoints saved for interrupted processing")
    print("✅ Output format: Compatible with FastText embeddings for downstream tasks")
    print("✅ Embedding dimension: 2314D (Enhanced BERT) vs 300D (FastText)")
    print("\n📊 Enhanced embeddings capture:")
    print("  1. Global context (CLS token) - what the log means overall")
    print("  2. Average representation (mean pooling) - typical patterns")
    print("  3. Key features (max pooling) - most important elements")
    print("  4. Attention patterns - which parts of the log are most significant")
    print("\n💡 This richer representation should improve anomaly detection performance.")
    print("💾 Checkpoints saved in: checkpoints/logbert/")
    print("=" * 60)


if __name__ == "__main__":
    main() 