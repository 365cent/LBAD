#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LogBERT: Log Anomaly Detection via BERT
Following the original LogBERT approach with modular commands.
Usage:
    python logbert.py vocab
    python logbert.py train  
    python logbert.py predict
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import BertForSequenceClassification, BertTokenizer
from torch.optim import AdamW
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROCESSED_DIR = Path("processed")
OUTPUT_DIR = Path("embeddings")
MODEL_DIR = OUTPUT_DIR / "logbert_model"
VOCAB_DIR = OUTPUT_DIR / "vocab"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
VOCAB_DIR.mkdir(parents=True, exist_ok=True)

MAX_SEQ_LENGTH = 128
BATCH_SIZE = 8
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5
NUM_WORKERS = 2

# Enable TensorFlow optimizations but limit threads
tf.config.threading.set_inter_op_parallelism_threads(2)
tf.config.threading.set_intra_op_parallelism_threads(2)

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """Return best available computation device."""
    if torch.backends.mps.is_available():
        print("Using MPS (Metal Performance Shaders) device")
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


def load_tfrecords_parallel(directory: Path = PROCESSED_DIR) -> pd.DataFrame:
    """Load all TFRecord files using parallel processing."""
    tfrecord_files = list(directory.glob("**/*.tfrecord"))
    if not tfrecord_files:
        raise FileNotFoundError(f"No TFRecord files found in {directory}")
    
    print(f"Found {len(tfrecord_files)} TFRecord files. Loading with {NUM_WORKERS} workers...")
    
    all_logs = []
    all_labels = []
    
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(process_single_tfrecord, path) for path in tfrecord_files]
        
        for future in tqdm(futures, desc="Loading TFRecords"):
            logs, labels = future.result()
            all_logs.extend(logs)
            all_labels.extend(labels)
    
    return pd.DataFrame({"log": all_logs, "label_json": all_labels})


def collect_attack_types(df: pd.DataFrame) -> List[str]:
    """Return sorted list of all attack labels present in the data."""
    attack_set = set()
    for label_str in df["label_json"]:
        try:
            labels = json.loads(label_str)
            if not isinstance(labels, list):
                labels = [labels]
            attack_set.update([lbl for lbl in labels if lbl])
        except json.JSONDecodeError:
            continue
    return sorted(attack_set)


def save_vocab(attack_types: List[str]) -> None:
    """Save vocabulary information."""
    vocab_info = {
        "attack_types": attack_types,
        "vocab_size": len(attack_types),
        "label_to_idx": {lbl: i for i, lbl in enumerate(attack_types)},
        "idx_to_label": {i: lbl for i, lbl in enumerate(attack_types)}
    }
    
    with open(VOCAB_DIR / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab_info, f, indent=2)
    
    print(f"Vocabulary saved: {len(attack_types)} unique attack types")
    print(f"Vocab file: {VOCAB_DIR / 'vocab.json'}")


def load_vocab() -> Dict:
    """Load vocabulary information."""
    vocab_path = VOCAB_DIR / "vocab.json"
    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}. Run 'python logbert.py vocab' first.")
    
    with open(vocab_path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_data_distribution(df: pd.DataFrame, attack_types: List[str]) -> None:
    """Create and save distribution charts for the data."""
    label_counts = Counter()
    log_lengths = []
    multi_label_count = 0
    
    for idx, row in df.iterrows():
        log_lengths.append(len(row["log"]))
        try:
            labels = json.loads(row["label_json"])
            if not isinstance(labels, list):
                labels = [labels]
            if len(labels) > 1:
                multi_label_count += 1
            for lbl in labels:
                if lbl:
                    label_counts[lbl] += 1
        except json.JSONDecodeError:
            pass
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('LogBERT Data Distribution Analysis', fontsize=16)
    
    # 1. Attack type distribution
    ax1 = axes[0, 0]
    labels_df = pd.DataFrame(list(label_counts.items()), columns=['Attack Type', 'Count'])
    labels_df = labels_df.sort_values('Count', ascending=True)
    ax1.barh(labels_df['Attack Type'], labels_df['Count'])
    ax1.set_xlabel('Count')
    ax1.set_title('Distribution of Attack Types')
    ax1.grid(axis='x', alpha=0.3)
    
    # 2. Log length distribution
    ax2 = axes[0, 1]
    ax2.hist(log_lengths, bins=50, edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Log Length (characters)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Log Lengths')
    ax2.axvline(np.mean(log_lengths), color='red', linestyle='--', 
                label=f'Mean: {np.mean(log_lengths):.0f}')
    ax2.legend()
    
    # 3. Top 10 attack types pie chart
    ax3 = axes[1, 0]
    top_labels = labels_df.nlargest(10, 'Count')
    ax3.pie(top_labels['Count'], labels=top_labels['Attack Type'], autopct='%1.1f%%')
    ax3.set_title('Top 10 Attack Types (Percentage)')
    
    # 4. Summary statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    stats_text = f"""Dataset Summary Statistics:
    
    Total samples: {len(df):,}
    Unique attack types: {len(attack_types)}
    Multi-label samples: {multi_label_count:,} ({multi_label_count/len(df)*100:.1f}%)
    
    Log length statistics:
    - Mean: {np.mean(log_lengths):.0f} chars
    - Median: {np.median(log_lengths):.0f} chars
    - Min: {np.min(log_lengths)} chars
    - Max: {np.max(log_lengths):,} chars
    
    Most common attacks:
    """
    for attack, count in label_counts.most_common(5):
        stats_text += f"\n    - {attack}: {count:,} ({count/len(df)*100:.1f}%)"
    
    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, 
             fontsize=12, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'data_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved distribution charts to {OUTPUT_DIR / 'data_distribution.png'}")


# ---------------------------------------------------------------------------
# Dataset Class
# ---------------------------------------------------------------------------

class LogBERTDataset(Dataset):
    """Dataset class for LogBERT following original approach."""
    
    def __init__(self, df: pd.DataFrame, tokenizer: BertTokenizer, 
                 label_to_idx: Dict[str, int], device: torch.device):
        self.texts = df["log"].tolist()
        self.label_to_idx = label_to_idx
        self.tokenizer = tokenizer
        self.device = device
        
        # Pre-encode all labels
        self.labels = [self._encode_labels(j) for j in df["label_json"]]
        
        # Pre-tokenize dataset
        print("Tokenizing dataset...")
        self.encodings = self._batch_tokenize()
    
    def _batch_tokenize(self) -> List[Dict[str, torch.Tensor]]:
        """Tokenize all texts in batches for efficiency."""
        batch_size = 100
        all_encodings = []
        
        for i in tqdm(range(0, len(self.texts), batch_size), desc="Tokenizing"):
            batch_texts = self.texts[i:i + batch_size]
            batch_enc = self.tokenizer(
                batch_texts,
                truncation=True,
                padding="max_length",
                max_length=MAX_SEQ_LENGTH,
                return_tensors="pt",
            )
            
            for j in range(len(batch_texts)):
                enc = {k: v[j] for k, v in batch_enc.items()}
                all_encodings.append(enc)
            
            del batch_enc
            clear_memory(self.device)
        
        return all_encodings
    
    def _encode_labels(self, label_json: str) -> np.ndarray:
        """Encode labels as multi-hot vector."""
        vec = np.zeros(len(self.label_to_idx), dtype=np.float32)
        try:
            labels = json.loads(label_json)
            if not isinstance(labels, list):
                labels = [labels]
            for lbl in labels:
                if lbl in self.label_to_idx:
                    vec[self.label_to_idx[lbl]] = 1.0
        except json.JSONDecodeError:
            pass
        return vec
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.encodings[idx].copy()
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item


# ---------------------------------------------------------------------------
# Commands following original LogBERT approach
# ---------------------------------------------------------------------------

def vocab_command():
    """Build vocabulary from the dataset."""
    print("=" * 50)
    print("LogBERT: Building Vocabulary")
    print("=" * 50)
    
    # Load data
    df = load_tfrecords_parallel()
    attack_types = collect_attack_types(df)
    
    # Save vocabulary
    save_vocab(attack_types)
    
    # Create distribution plots
    plot_data_distribution(df, attack_types)
    
    print(f"\nVocabulary building complete!")
    print(f"Found {len(attack_types)} unique attack types")
    print(f"Total samples: {len(df):,}")


def train_command():
    """Train the LogBERT model."""
    print("=" * 50)
    print("LogBERT: Training Model")
    print("=" * 50)
    
    device = get_device()
    
    # Load vocabulary
    vocab_info = load_vocab()
    attack_types = vocab_info["attack_types"]
    label_to_idx = vocab_info["label_to_idx"]
    
    # Load data
    df = load_tfrecords_parallel()
    print(f"Loaded {len(df)} samples with {len(attack_types)} attack types")
    
    # Initialize model and tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=len(attack_types),
        problem_type="multi_label_classification",
    ).to(device)
    
    # Split data
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    del df
    clear_memory(device)
    
    # Create datasets
    train_ds = LogBERTDataset(train_df, tokenizer, label_to_idx, device)
    del train_df
    clear_memory(device)
    
    val_ds = LogBERTDataset(val_df, tokenizer, label_to_idx, device)
    del val_df
    clear_memory(device)
    
    # Create data loaders
    train_loader = DataLoader(
        train_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=device.type in ["cuda", "mps"]
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=device.type in ["cuda", "mps"]
    )
    
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
        print("-" * 20)
        
        # Training
        model.train()
        train_loss = 0.0
        train_pbar = tqdm(train_loader, desc="Training")
        
        for batch in train_pbar:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            del outputs, batch
            clear_memory(device)
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                outputs = model(**batch)
                val_loss += outputs.loss.item()
                
                del outputs, batch
                clear_memory(device)
        
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(MODEL_DIR)
            tokenizer.save_pretrained(MODEL_DIR)
            print(f"Best model saved to {MODEL_DIR}")
    
    print(f"\nTraining complete! Best validation loss: {best_val_loss:.4f}")


def predict_command():
    """Make predictions using the trained model and extract embeddings."""
    print("=" * 50)
    print("LogBERT: Making Predictions & Extracting Embeddings")
    print("=" * 50)
    
    device = get_device()
    
    # Load vocabulary
    vocab_info = load_vocab()
    attack_types = vocab_info["attack_types"]
    label_to_idx = vocab_info["label_to_idx"]
    idx_to_label = vocab_info["idx_to_label"]
    
    # Check if model exists
    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_DIR}. Run 'python logbert.py train' first.")
    
    # Load model and tokenizer with hidden states output enabled
    tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
    model = BertForSequenceClassification.from_pretrained(
        MODEL_DIR,
        output_hidden_states=True
    ).to(device)
    model.eval()
    
    # Load test data
    df = load_tfrecords_parallel()
    test_ds = LogBERTDataset(df, tokenizer, label_to_idx, device)
    test_loader = DataLoader(
        test_ds, 
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=device.type in ["cuda", "mps"]
    )
    
    # Make predictions and extract embeddings
    all_predictions = []
    all_true_labels = []
    all_cls_embeddings = []
    
    print("Making predictions and extracting CLS embeddings...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting & Extracting"):
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            true_labels = batch["labels"].numpy()
            
            outputs = model(**inputs)
            predictions = torch.sigmoid(outputs.logits).cpu().numpy()
            
            # Extract [CLS] embeddings from the last hidden state
            cls_embeddings = outputs.hidden_states[-1][:, 0, :].cpu().numpy()  # shape: (batch_size, hidden_dim)
            
            all_predictions.append(predictions)
            all_true_labels.append(true_labels)
            all_cls_embeddings.append(cls_embeddings)
            
            del outputs, batch
            clear_memory(device)
    
    # Combine all predictions and embeddings
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_true_labels = np.concatenate(all_true_labels, axis=0)
    all_cls_embeddings = np.concatenate(all_cls_embeddings, axis=0)
    
    # Convert to binary predictions (threshold = 0.5)
    binary_predictions = (all_predictions > 0.5).astype(int)
    
    # Calculate metrics
    f1_micro = f1_score(all_true_labels, binary_predictions, average='micro')
    f1_macro = f1_score(all_true_labels, binary_predictions, average='macro')
    
    print(f"\nPrediction Results:")
    print(f"F1 Score (Micro): {f1_micro:.4f}")
    print(f"F1 Score (Macro): {f1_macro:.4f}")
    print(f"CLS Embeddings Shape: {all_cls_embeddings.shape}")
    
    # Save predictions
    predictions_data = {
        'predictions': all_predictions.tolist(),
        'true_labels': all_true_labels.tolist(),
        'binary_predictions': binary_predictions.tolist(),
        'attack_types': attack_types,
        'f1_micro': f1_micro,
        'f1_macro': f1_macro
    }
    
    with open(OUTPUT_DIR / 'predictions.json', 'w') as f:
        json.dump(predictions_data, f, indent=2)
    
    # Save CLS embeddings for unsupervised learning
    embeddings_path = OUTPUT_DIR / 'logbert_cls_embeddings.npy'
    np.save(embeddings_path, all_cls_embeddings)
    
    print(f"Predictions saved to {OUTPUT_DIR / 'predictions.json'}")
    print(f"CLS embeddings saved to {embeddings_path}")
    print(f"Embeddings can be used for unsupervised learning (clustering, anomaly detection, etc.)")


def main():
    """Main function following original LogBERT command structure."""
    parser = argparse.ArgumentParser(description='LogBERT: Log Anomaly Detection via BERT')
    parser.add_argument('command', choices=['vocab', 'train', 'predict'], 
                        help='Command to run: vocab, train, or predict')
    
    args = parser.parse_args()
    
    # Set multiprocessing start method for compatibility
    mp.set_start_method('spawn', force=True)
    
    if args.command == 'vocab':
        vocab_command()
    elif args.command == 'train':
        train_command()
    elif args.command == 'predict':
        predict_command()


if __name__ == "__main__":
    main()