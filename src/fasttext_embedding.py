#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FastText Embedding for Log Analysis - Using Pre-trained Models

Key Improvements:
- Uses pre-trained FastText models instead of training from scratch
- Embeds logs as FastText vectors for better semantic representation
- Creates binary multi-label vectors with clear column mapping
- Removed randomness for consistent results
- Enhanced progress tracking with dots spinner
- Visualization shows ALL classes without sampling/reduction

Output files per log type (3 files for clarity):
- log_{type}.pkl: Raw log text embeddings (300D FastText vectors, float32)
- label_{type}.pkl: Binary label vectors with metadata
  * 'vectors': Binary arrays where [0 1 0] means only second class is present
  * 'classes': List of attack types corresponding to each column
  * 'description': Explanation of the binary vector format
- attack_types_{type}.txt: Human-readable attack type mapping and examples

Performance optimizations:
- Batch processing (500 samples per batch)
- Memory-efficient data types (int8 for labels, float32 for embeddings)
- Vectorized operations where possible
- Optimized pickle protocol for faster I/O
- Barnes-Hut t-SNE for faster visualization
- Reduced progress update frequency

Example label structure:
{
  'vectors': array([[0, 1, 0], [1, 0, 1], ...], dtype=int8),  # Binary vectors
  'classes': ['attack_type_1', 'attack_type_2', 'attack_type_3'],
  'description': 'Binary multi-label vectors where [0 1 0] means only the second class is present'
}
"""

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from gensim.models import KeyedVectors
from gensim.utils import simple_preprocess
from pathlib import Path
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import json
import multiprocessing
import argparse
from functools import partial
from halo import Halo
import gensim.downloader as api

# Configuration
OUTPUT_DIR = Path("embeddings")
PROCESSED_DIR = Path("processed")
VECTOR_SIZE = 300  # Standard FastText vector size

def parse_example(example):
    """Parse a TensorFlow Example protocol buffer."""
    feature_description = {
        'l': tf.io.FixedLenFeature([], tf.string),  # log
        'y': tf.io.FixedLenFeature([], tf.string),  # label
    }
    return tf.io.parse_single_example(example, feature_description)

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
            
            # Use TensorFlow's optimized API for batch processing
            dataset = tf.data.TFRecordDataset(str(file_path), compression_type="GZIP", num_parallel_reads=4)
            dataset = dataset.batch(1000)  # Process in batches
            
            for batch in dataset:
                parsed_batch = tf.io.parse_example(batch, {
                    'l': tf.io.FixedLenFeature([], tf.string),
                    'y': tf.io.FixedLenFeature([], tf.string)
                })
                
                logs = [log.decode('utf-8') for log in parsed_batch['l'].numpy()]
                labels = [label.decode('utf-8') for label in parsed_batch['y'].numpy()]
                
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

def load_pretrained_fasttext():
    """Load pre-trained FastText model."""
    spinner = Halo(text='Loading pre-trained FastText model', spinner='dots')
    spinner.start()
    
    try:
        # Try to load from local cache first
        model = api.load("fasttext-wiki-news-subwords-300")
        spinner.succeed("Loaded pre-trained FastText model (fasttext-wiki-news-subwords-300)")
        return model
    except Exception as e:
        spinner.text = "Downloading pre-trained FastText model (this may take a while)"
        try:
            model = api.load("fasttext-wiki-news-subwords-300")
            spinner.succeed("Downloaded and loaded pre-trained FastText model")
            return model
        except Exception as e2:
            # Try alternative smaller model if main one fails
            spinner.text = "Trying alternative word2vec model"
            try:
                model = api.load("word2vec-google-news-300")
                spinner.succeed("Loaded alternative word2vec model (word2vec-google-news-300)")
                return model
            except Exception as e3:
                spinner.fail(f"Failed to load any pre-trained model: {e3}")
                print("Consider installing fasttext manually or check internet connection")
                return None

def preprocess_text(text):
    """Preprocess text for embedding."""
    return simple_preprocess(text)

def embed_text(model, tokens):
    """Generate embedding for tokenized text using pre-trained FastText."""
    if not tokens:
        return np.zeros(model.vector_size, dtype=np.float32)
    
    # Vectorized approach for better performance
    valid_tokens = [token for token in tokens if token in model]
    
    if valid_tokens:
        # Get all embeddings at once and compute mean
        embeddings_matrix = np.array([model[token] for token in valid_tokens], dtype=np.float32)
        return np.mean(embeddings_matrix, axis=0)
    else:
        return np.zeros(model.vector_size, dtype=np.float32)

def embed_labels(model, labels):
    """Generate embeddings for labels."""
    if not labels:
        return np.zeros(model.vector_size)
    
    label_embeddings = []
    for label in labels:
        if label:
            tokens = preprocess_text(label)
            embedding = embed_text(model, tokens)
            label_embeddings.append(embedding)
    
    if label_embeddings:
        return np.mean(label_embeddings, axis=0)
    else:
        return np.zeros(model.vector_size)

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

def process_embeddings_batch(model, tokens_batch):
    """Process a batch of tokens for embedding generation."""
    return [embed_text(model, tokens) for tokens in tokens_batch]

def process_labels_batch(labels_batch, attack_types):
    """Process a batch of labels for binary vector generation."""
    return [create_binary_label_vector(label_json, attack_types) for label_json in labels_batch]

def process_embeddings(df, model, use_global_attack_list=False):
    """Process logs and create embeddings with binary label vectors - optimized version."""
    spinner = Halo(text='Processing log embeddings', spinner='dots')
    spinner.start()
    
    total_count = len(df)
    batch_size = 500  # Optimized batch size for better performance
    
    # Tokenize logs in batches
    spinner.text = "Tokenizing logs in batches"
    tokenized_logs = []
    
    # Use vectorized string operations where possible
    for i in range(0, total_count, batch_size):
        end_idx = min(i + batch_size, total_count)
        batch_logs = df['log'].iloc[i:end_idx]
        
        if i % 2000 == 0:  # Update progress less frequently
            spinner.text = f"Tokenizing logs: {end_idx}/{total_count} ({end_idx/total_count*100:.1f}%)"
        
        # Process batch
        batch_tokens = [preprocess_text(log) for log in batch_logs]
        tokenized_logs.extend(batch_tokens)
    
    df['tokens'] = tokenized_logs
    
    # Create log embeddings in batches
    spinner.text = "Creating log embeddings in batches"
    log_embeddings = []
    
    for i in range(0, total_count, batch_size):
        end_idx = min(i + batch_size, total_count)
        tokens_batch = tokenized_logs[i:end_idx]
        
        if i % 2000 == 0:
            spinner.text = f"Creating log embeddings: {end_idx}/{total_count} ({end_idx/total_count*100:.1f}%)"
        
        # Process batch with multiprocessing could be added here if needed
        batch_embeddings = process_embeddings_batch(model, tokens_batch)
        log_embeddings.extend(batch_embeddings)
    
    # Convert to numpy array for better memory efficiency
    log_embeddings_array = np.array(log_embeddings, dtype=np.float32)
    df['log_embedding'] = list(log_embeddings_array)  # Convert back to list for pandas
    
    # Process labels in batches
    spinner.text = "Processing binary label vectors in batches"
    
    if use_global_attack_list:
        # Use all labels across all log types
        attack_types = collect_unique_labels_from_data(df)
        
        binary_labels = []
        for i in range(0, total_count, batch_size):
            end_idx = min(i + batch_size, total_count)
            labels_batch = df['label_json'].iloc[i:end_idx]
            
            if i % 2000 == 0:
                spinner.text = f"Processing labels: {end_idx}/{total_count} ({end_idx/total_count*100:.1f}%)"
            
            batch_binary = process_labels_batch(labels_batch, attack_types)
            binary_labels.extend(batch_binary)
        
        df['binary_labels'] = binary_labels
        df.attrs['attack_types'] = attack_types
    else:
        # Process by log type
        log_type_to_attacks = {}
        for log_type, group_df in df.groupby('log_type'):
            spinner.text = f"Processing log type: {log_type}"
            log_type_to_attacks[log_type] = collect_unique_labels_from_data(group_df)
        
        # Process all labels in batches
        binary_labels = []
        for i in range(0, total_count, batch_size):
            end_idx = min(i + batch_size, total_count)
            
            if i % 2000 == 0:
                spinner.text = f"Processing binary labels: {end_idx}/{total_count} ({end_idx/total_count*100:.1f}%)"
            
            batch_binary = []
            for j in range(i, end_idx):
                row = df.iloc[j]
                log_type = row['log_type']
                if log_type in log_type_to_attacks:
                    binary_vector = create_binary_label_vector(row['label_json'], log_type_to_attacks[log_type])
                else:
                    binary_vector = np.array([], dtype=np.int8)
                batch_binary.append(binary_vector)
            
            binary_labels.extend(batch_binary)
        
        df['binary_labels'] = binary_labels
        df.attrs['log_type_to_attacks'] = log_type_to_attacks
    
    spinner.succeed("Embedding processing complete")
    return df

def visualize_embeddings(df, output_file=None):
    """Create t-SNE visualization with balanced class sampling for performance and minority visibility."""
    # ------------------------- NEW IMPLEMENTATION START -------------------------
    # Parameters for sampling – tweak here if necessary
    MAX_TOTAL_POINTS = 50000   # Hard cap on total points sent to t-SNE
    MAX_POINTS_PER_CLASS = 1500  # Limit for any single class to avoid domination

    spinner = Halo(text="Preparing visualization data", spinner='dots')
    spinner.start()

    # -------------------------------------------------------------------------
    # 1. Build visualization labels for every row (normal vs attacks etc.)
    # -------------------------------------------------------------------------
    spinner.text = "Generating labels for visualization"
    viz_labels = []
    for label_json_str in df['label_json']:
        labels = get_labels_from_json(label_json_str)
        if not labels:
            viz_labels.append("normal")
        else:
            viz_labels.append(", ".join(sorted(labels)))

    # Attach labels temporarily to the dataframe for easy indexing
    df = df.copy()
    df['viz_label'] = viz_labels

    # -------------------------------------------------------------------------
    # 2. Balanced sampling – keep all minority classes, down-sample major ones
    # -------------------------------------------------------------------------
    spinner.text = "Applying balanced sampling to limit dataset size"
    np.random.seed(42)  # Reproducibility

    selected_indices = []
    label_to_indices = {}
    for idx, lbl in enumerate(viz_labels):
        label_to_indices.setdefault(lbl, []).append(idx)

    for lbl, indices in label_to_indices.items():
        if len(indices) > MAX_POINTS_PER_CLASS:
            sampled = np.random.choice(indices, MAX_POINTS_PER_CLASS, replace=False)
            selected_indices.extend(sampled)
        else:
            selected_indices.extend(indices)  # Keep all if already small

    # If we still exceed the global cap, randomly subsample the union (keeps balance reasonably well)
    if len(selected_indices) > MAX_TOTAL_POINTS:
        selected_indices = list(np.random.choice(selected_indices, MAX_TOTAL_POINTS, replace=False))

    # -------------------------------------------------------------------------
    # 3. Gather embeddings and labels for the sampled indices
    # -------------------------------------------------------------------------
    embeddings = np.vstack([df.at[i, 'log_embedding'] for i in selected_indices]).astype(np.float32)
    sampled_labels = [viz_labels[i] for i in selected_indices]
    sampled_log_types = [df.at[i, 'log_type'] for i in selected_indices]

    spinner.text = f"Running t-SNE on {len(embeddings)} sampled points"

    # Choose perplexity based on size (rule of thumb: 5–50 & < samples/3)
    perplexity = min(50, max(5, len(embeddings)//1000))

    # -------------------------------------------------------------------------
    # 4. Execute t-SNE (Barnes-Hut) – use safer sklearn parameters
    # -------------------------------------------------------------------------
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        n_iter=500,          # Reasonable iterations
        learning_rate='auto',
        init='pca',
        method='barnes_hut',
        random_state=42
    )

    reduced = tsne.fit_transform(embeddings)

    # -------------------------------------------------------------------------
    # 5. Prepare DataFrame for plotting
    # -------------------------------------------------------------------------
    df_plot = pd.DataFrame({
        'x': reduced[:, 0],
        'y': reduced[:, 1],
        'label': sampled_labels,
        'log_type': sampled_log_types
    })

    spinner.succeed("t-SNE dimensionality reduction complete")

    # ------------------------- NEW IMPLEMENTATION END -------------------------

    # Count visualization labels
    label_counts = df_plot['label'].value_counts()
    print(f"\nVisualization showing ALL {len(label_counts)} unique label combinations:")
    for label, count in label_counts.head(10).items():  # Show top 10 for readability
        percentage = (count / len(df_plot)) * 100
        print(f"  {label}: {count} ({percentage:.2f}%)")
    
    if len(label_counts) > 10:
        print(f"  ... and {len(label_counts) - 10} more label combinations")
    
    # Create optimized color palette
    unique_labels = sorted(df_plot['label'].unique())
    palette = sns.color_palette("husl", len(unique_labels))
    color_map = {label: palette[i] for i, label in enumerate(unique_labels)}
    
    if "normal" in color_map:
        color_map["normal"] = "green"
    
    # Create optimized scatter plot
    spinner = Halo(text="Creating visualization plot", spinner='dots')
    spinner.start()
    
    plt.figure(figsize=(16, 10))  # Larger figure for better readability
    
    # Use matplotlib directly for better performance with large datasets
    for label in unique_labels:
        mask = df_plot['label'] == label
        subset = df_plot[mask]
        
        plt.scatter(
            subset['x'], 
            subset['y'], 
            c=[color_map[label]], 
            label=label,
            alpha=0.6,
            s=20,  # Smaller points for large datasets
            edgecolors='none'  # Remove edges for better performance
        )
    
    plt.title(f't-SNE Visualization: FastText Log Embeddings (All {len(unique_labels)} Classes)', fontsize=14)
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
        plt.savefig(output_file, dpi=150, bbox_inches='tight')  # Slightly lower DPI for faster saving
        spinner.succeed(f"Saved visualization to {output_file}")
    else:
        plt.show()
        spinner.succeed("Displayed visualization")
    
    plt.close()
    
    # Clear memory
    del embeddings, reduced, df_plot

def find_available_log_types():
    """Find available log types in the processed directory."""
    if not PROCESSED_DIR.exists():
        return []
    return sorted([path.name for path in PROCESSED_DIR.iterdir() 
                  if path.is_dir() and list(path.glob("*.tfrecord"))])

def save_embeddings_and_labels(df, output_dir, log_type_name):
    """Save only log embeddings and label vectors as requested - optimized version."""
    spinner = Halo(text=f"Saving embeddings for {log_type_name}", spinner='dots')
    spinner.start()
    
    # Extract and save log embeddings as log_<type>.pkl (optimized)
    log_embeddings = np.vstack(df['log_embedding'].tolist()).astype(np.float32)
    log_filename = f"log_{log_type_name}.pkl"
    
    spinner.text = f"Saving log embeddings to {log_filename}"
    with open(output_dir / log_filename, 'wb') as f:
        pickle.dump(log_embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)  # Use highest protocol for speed
    
    # Extract and save binary label vectors as label_<type>.pkl
    if 'binary_labels' in df.columns and len(df['binary_labels']) > 0:
        # Filter out empty arrays and stack efficiently
        valid_vectors = [vec for vec in df['binary_labels'] if len(vec) > 0]
        
        if valid_vectors:
            binary_vectors = np.vstack(valid_vectors).astype(np.int8)  # Memory efficient
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
                
                f.write(f"\nGenerated by FastText embeddings extraction\n")
                f.write(f"Embedding dimension: 300D (FastText vectors)\n")
                
            spinner.succeed(f"Saved {log_filename}, {label_filename}, and {attack_info_filename}")
            
            # Print summary with proper vector format
            print(f"\nSaved files for {log_type_name}:")
            print(f"  - {log_filename}: Log embeddings {log_embeddings.shape} (300D FastText vectors)")
            print(f"  - {label_filename}: Binary label vectors {binary_vectors.shape}")
            print(f"  - {attack_info_filename}: Attack types and column mapping details")
            print(f"  - Classes: {classes}")
        else:
            spinner.warn(f"No valid binary labels found for {log_type_name}")
    else:
        spinner.warn(f"No binary labels found for {log_type_name}, saved only log embeddings")

def main():
    parser = argparse.ArgumentParser(description="Generate FastText embeddings for log data using pre-trained models")
    parser.add_argument("--log-type", type=str, default=None, help="Process only this specific log type")
    parser.add_argument("--output-subdir", type=str, default=None, help="Optional subdirectory under embeddings/ (e.g., 'fasttext')")
    args = parser.parse_args()

    # Optionally route outputs to embeddings/<subdir>
    global OUTPUT_DIR
    if args.output_subdir:
        OUTPUT_DIR = Path("embeddings") / args.output_subdir

    # Ensure output directories exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load pre-trained FastText model
    model = load_pretrained_fasttext()
    if model is None:
        print("Failed to load pre-trained FastText model. Exiting.")
        return
    
    # Find available log types
    spinner = Halo(text="Finding available log types", spinner='dots')
    spinner.start()
    available_types = find_available_log_types()
    if not available_types:
        spinner.fail("No log types with data found.")
        return
    
    spinner.succeed(f"Found {len(available_types)} log types: {', '.join(available_types)}")
    
    # Determine log types to process
    if args.log_type:
        if args.log_type not in available_types:
            print(f"Log type '{args.log_type}' not found. Available types: {', '.join(available_types)}")
            return
        types_to_process = [args.log_type]
        run_combined = False
    else:
        types_to_process = available_types
        run_combined = True

    # Process individual log types
    for log_type in types_to_process:
        print(f"\n{'='*50}\nProcessing log type: {log_type}\n{'='*50}")
        try:
            # Load data for this log type
            df = load_tfrecord_files(log_type_filter=log_type)
            if df.empty:
                print(f"No data for log type '{log_type}'. Skipping.")
                continue

            # Display data distribution
            display_data_distribution(df, log_type)

            # Process embeddings
            df = process_embeddings(df, model, use_global_attack_list=False)
            
            # Save outputs
            output_dir = OUTPUT_DIR / log_type
            output_dir.mkdir(parents=True, exist_ok=True)
            
            save_embeddings_and_labels(df, output_dir, log_type)
            
            # Create visualization
            visualize_embeddings(
                df, 
                output_file=output_dir / "visualization.png"
            )
            
        except Exception as e:
            print(f"Error processing log type {log_type}: {e}")
            import traceback
            traceback.print_exc()

    # Process combined model if requested
    if run_combined:
        print(f"\n{'='*50}\nProcessing all log types combined\n{'='*50}")
        try:
            # Load all data
            df_all = load_tfrecord_files()
            if df_all.empty:
                print("No data found for combined log types")
            else:
                display_data_distribution(df_all, "all combined")
                df_all = process_embeddings(df_all, model, use_global_attack_list=True)
                
                # Save combined outputs
                save_embeddings_and_labels(df_all, OUTPUT_DIR, "all_combined")
                
                # Create visualization
                visualize_embeddings(
                    df_all,
                    output_file=OUTPUT_DIR / "visualization_all_combined.png"
                )
        
        except Exception as e:
            print(f"Error processing combined log types: {e}")
            import traceback
            traceback.print_exc()
    
    spinner = Halo(spinner='dots', text='Completing processing')
    spinner.start()
    spinner.succeed("FastText embedding processing complete!")

if __name__ == "__main__":
    main()
