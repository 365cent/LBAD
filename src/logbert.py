#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LogBERT: Log Anomaly Detection via BERT
Following the original LogBERT approach with modular commands.
Usage:
    python logbert.py vocab [--log-type TYPE]
    python logbert.py train [--log-type TYPE]  
    python logbert.py predict [--log-type TYPE]
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


def get_paths_for_log_type(log_type: Optional[str]):
    """Get appropriate paths based on log type."""
    if log_type:
        # Specific log type processing
        processed_dir = PROCESSED_DIR / log_type
        output_dir = OUTPUT_DIR / log_type
        model_dir = output_dir / "logbert_model"
        vocab_dir = output_dir / "vocab"
    else:
        # All log types combined
        processed_dir = PROCESSED_DIR
        output_dir = OUTPUT_DIR
        model_dir = output_dir / "logbert_model"
        vocab_dir = output_dir / "vocab"
    
    return processed_dir, output_dir, model_dir, vocab_dir


def find_available_log_types():
    """Find available log types in the processed directory."""
    if not PROCESSED_DIR.exists():
        return []
    return sorted([path.name for path in PROCESSED_DIR.iterdir() 
                   if path.is_dir() and list(path.glob("*.tfrecord"))])


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


def load_tfrecords_parallel(directory: Path) -> pd.DataFrame:
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


def save_vocab(attack_types: List[str], vocab_dir: Path) -> None:
    """Save vocabulary information."""
    vocab_dir.mkdir(parents=True, exist_ok=True)
    vocab_info = {
        "attack_types": attack_types,
        "vocab_size": len(attack_types),
        "label_to_idx": {lbl: i for i, lbl in enumerate(attack_types)},
        "idx_to_label": {i: lbl for i, lbl in enumerate(attack_types)}
    }
    
    with open(vocab_dir / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab_info, f, indent=2)
    
    print(f"Vocabulary saved: {len(attack_types)} unique attack types")
    print(f"Vocab file: {vocab_dir / 'vocab.json'}")


def load_vocab(vocab_dir: Path) -> Dict:
    """Load vocabulary information."""
    vocab_path = vocab_dir / "vocab.json"
    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}. Run vocab command first.")
    
    with open(vocab_path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_data_distribution(df: pd.DataFrame, attack_types: List[str], output_dir: Path) -> None:
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
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / 'data_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved distribution charts to {output_dir / 'data_distribution.png'}")


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

def vocab_command(log_type: Optional[str] = None):
    """Build vocabulary from the dataset."""
    print("=" * 50)
    print(f"LogBERT: Building Vocabulary{f' for {log_type}' if log_type else ''}")
    print("=" * 50)
    
    # Get paths based on log type
    processed_dir, output_dir, model_dir, vocab_dir = get_paths_for_log_type(log_type)
    
    if log_type and not processed_dir.exists():
        available_types = find_available_log_types()
        raise FileNotFoundError(f"Log type '{log_type}' not found. Available types: {', '.join(available_types)}")
    
    # Load data
    df = load_tfrecords_parallel(processed_dir)
    attack_types = collect_attack_types(df)
    
    # Save vocabulary
    save_vocab(attack_types, vocab_dir)
    
    # Create distribution plots
    plot_data_distribution(df, attack_types, output_dir)
    
    print(f"\nVocabulary building complete!")
    print(f"Found {len(attack_types)} unique attack types")
    print(f"Total samples: {len(df):,}")


def train_command(log_type: Optional[str] = None):
    """Train the LogBERT model."""
    print("=" * 50)
    print(f"LogBERT: Training Model{f' for {log_type}' if log_type else ''}")
    print("=" * 50)
    
    device = get_device()
    
    # Get paths based on log type
    processed_dir, output_dir, model_dir, vocab_dir = get_paths_for_log_type(log_type)
    
    # Load vocabulary
    vocab_info = load_vocab(vocab_dir)
    attack_types = vocab_info["attack_types"]
    label_to_idx = vocab_info["label_to_idx"]
    
    # Load data
    df = load_tfrecords_parallel(processed_dir)
    print(f"Loaded {len(df)} samples with {len(attack_types)} attack types")
    
    # Initialize model and tokenizer
    # Use local cache directory if defined to avoid $HOME/.cache quota issues
    cache_dir = str(Path("hf_cache").resolve())
    try:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", cache_dir=cache_dir)
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=len(attack_types),
        problem_type="multi_label_classification",
        cache_dir=cache_dir,
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
            model_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(model_dir)
            tokenizer.save_pretrained(model_dir)
            print(f"Best model saved to {model_dir}")
    
    print(f"\nTraining complete! Best validation loss: {best_val_loss:.4f}")


def predict_command(log_type: Optional[str] = None):
    """Make predictions using the trained model and extract embeddings."""
    print("=" * 50)
    print(f"LogBERT: Making Predictions & Extracting Embeddings{f' for {log_type}' if log_type else ''}")
    print("=" * 50)
    
    device = get_device()
    
    # Get paths based on log type
    processed_dir, output_dir, model_dir, vocab_dir = get_paths_for_log_type(log_type)
    
    # Load vocabulary
    vocab_info = load_vocab(vocab_dir)
    attack_types = vocab_info["attack_types"]
    label_to_idx = vocab_info["label_to_idx"]
    idx_to_label = vocab_info["idx_to_label"]
    
    # Check if model exists
    if not model_dir.exists():
        raise FileNotFoundError(f"Model not found: {model_dir}. Run train command first.")
    
    # Load model and tokenizer with hidden states output enabled
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    model = BertForSequenceClassification.from_pretrained(
        model_dir,
        output_hidden_states=True
    ).to(device)
    model.eval()
    
    # Load test data
    df = load_tfrecords_parallel(processed_dir)
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
    
    # Calculate additional metrics for summary
    total_samples = len(all_true_labels)
    total_positive_predictions = np.sum(binary_predictions)
    total_true_positives = np.sum(all_true_labels)
    accuracy_per_label = []
    
    for i in range(len(attack_types)):
        true_labels_col = all_true_labels[:, i]
        pred_labels_col = binary_predictions[:, i]
        if np.sum(true_labels_col) > 0:  # Only calculate if there are positive samples
            accuracy = np.mean(true_labels_col == pred_labels_col)
            accuracy_per_label.append((attack_types[i], accuracy, np.sum(true_labels_col), np.sum(pred_labels_col)))
    
    # Sort by accuracy for summary
    accuracy_per_label.sort(key=lambda x: x[1], reverse=True)
    
    # Print results
    print(f"\nPrediction Results:")
    print(f"F1 Score (Micro): {f1_micro:.4f}")
    print(f"F1 Score (Macro): {f1_macro:.4f}")
    print(f"CLS Embeddings Shape: {all_cls_embeddings.shape}")
    
    # Create summary text
    summary_lines = []
    summary_lines.append("=" * 60)
    summary_lines.append(f"LogBERT Classification Summary{f' - {log_type}' if log_type else ' - All Log Types'}")
    summary_lines.append("=" * 60)
    summary_lines.append(f"Total samples: {total_samples:,}")
    summary_lines.append(f"Total attack types: {len(attack_types)}")
    summary_lines.append(f"CLS embeddings shape: {all_cls_embeddings.shape}")
    summary_lines.append("")
    summary_lines.append("Overall Performance:")
    summary_lines.append(f"  F1 Score (Micro): {f1_micro:.4f}")
    summary_lines.append(f"  F1 Score (Macro): {f1_macro:.4f}")
    summary_lines.append(f"  Total true positives: {total_true_positives:,}")
    summary_lines.append(f"  Total predicted positives: {total_positive_predictions:,}")
    summary_lines.append("")
    
    if accuracy_per_label:
        summary_lines.append("Per-Attack-Type Performance (Top 10 by accuracy):")
        summary_lines.append(f"{'Attack Type':<25} {'Accuracy':<10} {'True Count':<12} {'Pred Count':<12}")
        summary_lines.append("-" * 65)
        
        for attack_type, accuracy, true_count, pred_count in accuracy_per_label[:10]:
            summary_lines.append(f"{attack_type:<25} {accuracy:.4f}     {true_count:<12} {pred_count:<12}")
        
        if len(accuracy_per_label) > 10:
            summary_lines.append(f"... and {len(accuracy_per_label) - 10} more attack types")
        summary_lines.append("")
    
    summary_lines.append("Output Files:")
    summary_lines.append(f"  Predictions: {output_dir / 'predictions.json'}")
    summary_lines.append(f"  CLS Embeddings: {output_dir / 'logbert_cls_embeddings.npy'}")
    summary_lines.append(f"  Summary: {output_dir / 'classification_summary.txt'}")
    summary_lines.append("")
    summary_lines.append("Note: CLS embeddings can be used for unsupervised learning")
    summary_lines.append("(clustering, anomaly detection, etc.)")
    
    # Save predictions
    predictions_data = {
        'predictions': all_predictions.tolist(),
        'true_labels': all_true_labels.tolist(),
        'binary_predictions': binary_predictions.tolist(),
        'attack_types': attack_types,
        'f1_micro': f1_micro,
        'f1_macro': f1_macro
    }
    
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'predictions.json', 'w') as f:
        json.dump(predictions_data, f, indent=2)
    
    # Save CLS embeddings for unsupervised learning
    embeddings_path = output_dir / 'logbert_cls_embeddings.npy'
    np.save(embeddings_path, all_cls_embeddings)
    
    # Save summary to text file
    summary_path = output_dir / 'classification_summary.txt'
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary_lines))
    
    print(f"Predictions saved to {output_dir / 'predictions.json'}")
    print(f"CLS embeddings saved to {embeddings_path}")
    print(f"Classification summary saved to {summary_path}")
    print(f"Embeddings can be used for unsupervised learning (clustering, anomaly detection, etc.)")


def main():
    """Main function following original LogBERT command structure."""
    parser = argparse.ArgumentParser(description='LogBERT: Log Anomaly Detection via BERT')
    parser.add_argument('command', choices=['vocab', 'train', 'predict'], 
                        help='Command to run: vocab, train, or predict')
    parser.add_argument('--log-type', type=str, default=None,
                        help='Process specific log type (e.g., wp-error, dns, auth). If not specified, processes all log types combined.')
    
    args = parser.parse_args()
    
    # Validate log type if specified
    if args.log_type:
        available_types = find_available_log_types()
        if not available_types:
            print("No processed log types found. Run preprocessing first.")
            return
        if args.log_type not in available_types:
            print(f"Log type '{args.log_type}' not found. Available types: {', '.join(available_types)}")
            return
    
    # Set multiprocessing start method for compatibility
    mp.set_start_method('spawn', force=True)
    
    if args.command == 'vocab':
        vocab_command(args.log_type)
    elif args.command == 'train':
        train_command(args.log_type)
    elif args.command == 'predict':
        predict_command(args.log_type)


if __name__ == "__main__":
    main()