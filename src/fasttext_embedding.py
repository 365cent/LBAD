#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FastText Embedding for Log Analysis - Optimized
"""

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from gensim.models import FastText
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

# Configuration
OUTPUT_DIR = Path("embeddings")
MODEL_DIR = Path("models")
PROCESSED_DIR = Path("processed")
VECTOR_SIZE = 128
RANDOM_SEED = 42

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
    
    for file_path in tqdm(tfrecord_files, ascii=True, desc="Loading files"):
        try:
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
            print(f"Error processing file {file_path}: {e}")
    
    print(f"Loaded {len(all_logs)} log entries")
    return pd.DataFrame({
        'log': all_logs, 
        'label_json': all_labels_json,
        'log_type': all_log_types
    })

def normalize_label(label):
    """Normalize attack labels to ensure consistency."""
    if not label:
        return label
    return label.replace('-', '_')

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
    for label_json_str in df['label_json']:
        all_unique_labels.update(get_labels_from_json(label_json_str))
    return sorted(list(all_unique_labels))

def create_attack_binary_vector(label_json_str, target_attack_types):
    """Convert labels from JSON string into a binary vector."""
    normalized_labels = get_labels_from_json(label_json_str)
    
    # Create binary vector
    binary_vector = np.zeros(len(target_attack_types), dtype=np.int8)  # Use int8 instead of int
    for i, attack in enumerate(target_attack_types):
        if attack in normalized_labels:
            binary_vector[i] = 1
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
    
    for label_json_str in df['label_json']:
        labels = get_labels_from_json(label_json_str)
        if labels:
            for label in labels:
                all_labels_count[label] = all_labels_count.get(label, 0) + 1
            attack_count += 1
        else:
            normal_count += 1
    
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

def preprocess_log(log):
    """Preprocess a single log entry."""
    return simple_preprocess(log)

def preprocess_logs_and_labels(df, use_global_attack_list=False):
    """Tokenize log entries and create binary label vectors with parallel processing."""
    # Parallelize tokenization
    num_cores = max(1, multiprocessing.cpu_count() - 1)
    with multiprocessing.Pool(num_cores) as pool:
        df['tokens'] = list(tqdm(pool.imap(preprocess_log, df['log'], chunksize=1000),
                          total=len(df), desc="Tokenizing logs", ascii=True))
    
    if use_global_attack_list:
        # Use all labels across all log types
        attack_types = collect_unique_labels_from_data(df)
        
        # Vectorize binary label creation
        vector_func = partial(create_attack_binary_vector, target_attack_types=attack_types)
        df['binary_labels'] = list(map(vector_func, df['label_json']))
        df.attrs['attack_types'] = attack_types
    else:
        # Process by log type
        log_type_to_attacks = {}
        for log_type, group_df in df.groupby('log_type'):
            log_type_to_attacks[log_type] = collect_unique_labels_from_data(group_df)
        
        # Apply to each log entry grouped by log type
        def get_binary_vector(row):
            log_type = row['log_type']
            if log_type not in log_type_to_attacks:
                return np.array([])
            return create_attack_binary_vector(row['label_json'], log_type_to_attacks[log_type])
        
        df['binary_labels'] = df.apply(get_binary_vector, axis=1)
        df.attrs['log_type_to_attacks'] = log_type_to_attacks
    
    return df

def train_fasttext_model(corpus, vector_size=VECTOR_SIZE, window=5, min_count=5, epochs=5, model_name_suffix=""):
    """Train an optimized FastText model on the corpus."""
    model_name = f"fasttext_model{model_name_suffix}.bin"
    model_path = MODEL_DIR / model_name
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Use existing model if available
    if model_path.exists():
        print(f"Loading existing model from {model_path}")
        return FastText.load(str(model_path))

    # Train new model with optimized parameters
    print(f"Training FastText model on {len(corpus)} documents...")
    model = FastText(
        vector_size=vector_size,
        window=window,
        min_count=min_count,  # Increased from 1 to 5 to reduce vocab size
        workers=multiprocessing.cpu_count() - 1,
        seed=RANDOM_SEED,
        sg=1,  # Use skip-gram for better quality
        negative=10  # More negative samples for better quality
    )
    
    model.build_vocab(corpus_iterable=corpus)
    model.train(corpus_iterable=corpus, total_examples=len(corpus), epochs=epochs)
    
    # Trim memory
    model.trim_rule = None
    model.callbacks = None
    model.save(str(model_path))
    
    return model

def generate_embeddings(model, corpus):
    """Generate document embeddings with optimized batching."""
    print("Generating document embeddings...")
    
    # Generate embeddings using NumPy operations for speed
    model_vocab = set(model.wv.key_to_index.keys())
    batch_size = 1000
    num_batches = (len(corpus) + batch_size - 1) // batch_size
    all_embeddings = np.zeros((len(corpus), model.vector_size), dtype=np.float32)
    
    for i in tqdm(range(num_batches), desc="Generating embeddings", ascii=True):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(corpus))
        batch = corpus[start_idx:end_idx]
        
        for j, doc in enumerate(batch):
            valid_words = [word for word in doc if word in model_vocab]
            idx = start_idx + j
            
            if valid_words:
                # Use numpy operations directly for faster computation
                word_vectors = np.array([model.wv[word] for word in valid_words])
                all_embeddings[idx] = np.mean(word_vectors, axis=0)
    
    return all_embeddings

def visualize_embeddings(embeddings, label_json_list, log_types, output_file=None):
    """Create optimized t-SNE visualization of embeddings."""
    print("Creating visualization...")
    
    # Apply t-SNE to get the 2D coordinates
    perplexity = min(30, max(5, len(embeddings)//100))
    tsne = TSNE(
        n_components=2, 
        random_state=RANDOM_SEED, 
        perplexity=perplexity,
        n_iter=1000,  # Reduced from default
        n_jobs=-1  # Use all available cores
    )
    reduced = tsne.fit_transform(embeddings)
    
    # Process labels more efficiently
    processed_labels = []
    for label_json_str in label_json_list:
        labels = get_labels_from_json(label_json_str)
        if not labels:
            processed_labels.append("normal")
        else:
            processed_labels.append(next(iter(labels)))
    
    # Create plot dataframe
    df_plot = pd.DataFrame({
        'x': reduced[:, 0],
        'y': reduced[:, 1],
        'label': processed_labels,
        'log_type': log_types
    })
    
    # Count visualization labels
    label_counts = df_plot['label'].value_counts()
    print("\nVisualization label distribution:")
    for label, count in label_counts.items():
        percentage = (count / len(df_plot)) * 100
        print(f"  {label}: {count} ({percentage:.2f}%)")
    
    # Color palette with consistent colors
    unique_labels = sorted(df_plot['label'].unique())
    palette = sns.color_palette("husl", len(unique_labels))
    color_map = {label: palette[i] for i, label in enumerate(unique_labels)}
    
    if "normal" in color_map:
        color_map["normal"] = "green"
    if "unknown" in color_map:
        color_map["unknown"] = "gray"
    
    # Create scatter plot
    plt.figure(figsize=(12, 10))
    sns.scatterplot(
        x='x', 
        y='y', 
        hue='label', 
        style='log_type',
        data=df_plot, 
        palette=color_map,
        alpha=0.7,
        s=40  # Reduced from 50
    )
    
    plt.title('t-SNE Visualization of FastText Log Embeddings')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout(rect=[0,0,0.85,1])
    
    if output_file:
        plt.savefig(output_file, dpi=200, bbox_inches='tight')  # Reduced from 300 dpi
        print(f"Saved visualization to {output_file}")
    else:
        plt.show()
    
    plt.close()
    
    # Create individual attack visualizations for ALL attack types
    if output_file and len([l for l in unique_labels if l != "normal" and l != "unknown"]) > 1:
        output_dir = os.path.dirname(output_file)
        base_name = os.path.splitext(os.path.basename(output_file))[0]
        
        # Create visualizations for all attack types (not just top 3)
        attack_labels = [l for l in unique_labels if l != "normal" and l != "unknown"]
        
        print(f"\nCreating individual attack visualizations for all {len(attack_labels)} attack types...")
        
        for i, attack in enumerate(attack_labels):
            print(f"  Processing visualization {i+1}/{len(attack_labels)}: {attack}")
            df_attack = df_plot.copy()
            df_attack['label'] = df_attack['label'].apply(
                lambda x: x if x == attack else "normal" if x != "unknown" else "unknown"
            )
            
            plt.figure(figsize=(10, 8))
            sns.scatterplot(
                x='x', y='y', hue='label', style='log_type',
                data=df_attack, 
                palette={
                    attack: color_map[attack],
                    "normal": "green",
                    "unknown": "gray"
                },
                alpha=0.7,
                s=40
            )
            
            plt.title(f't-SNE Visualization: {attack}')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout(rect=[0,0,0.85,1])
            
            attack_output_file = os.path.join(output_dir, f"{base_name}_{attack}.png")
            plt.savefig(attack_output_file, dpi=200, bbox_inches='tight')
            plt.close()

def find_available_log_types():
    """Find available log types in the processed directory."""
    if not PROCESSED_DIR.exists():
        return []
    return sorted([path.name for path in PROCESSED_DIR.iterdir() 
                  if path.is_dir() and list(path.glob("*.tfrecord"))])

def save_key_file(attack_types, output_dir, filename="key.txt"):
    """Save a file mapping attack type indices to their names."""
    output_path = Path(output_dir) / filename
    with open(output_path, 'w') as f:
        f.write("# Attack type to index mapping\n")
        f.write("# index,attack_type\n")
        for i, attack_type in enumerate(attack_types):
            f.write(f"{i},{attack_type}\n")
    print(f"Saved attack type mapping to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate FastText embeddings for log data")
    parser.add_argument("--log-type", type=str, default=None, help="Process only this specific log type")
    parser.add_argument("--vector-size", type=int, default=VECTOR_SIZE, help="Size of embedding vectors")
    parser.add_argument("--sample-size", type=int, default=5000, help="Maximum samples for visualization")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs")
    args = parser.parse_args()

    # Ensure output directories exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # Find available log types
    available_types = find_available_log_types()
    if not available_types:
        print("No log types with data found.")
        return
    
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

            # Display data distribution before preprocessing
            display_data_distribution(df, log_type)

            # Preprocess and tokenize
            df = preprocess_logs_and_labels(df, use_global_attack_list=False)
            
            # Train FastText model
            model = train_fasttext_model(
                df['tokens'].tolist(),
                vector_size=args.vector_size,
                epochs=args.epochs,
                model_name_suffix=f"_{log_type}"
            )
            
            # Generate embeddings
            embeddings = generate_embeddings(model, df['tokens'].tolist())
            
            # Save outputs
            output_dir = OUTPUT_DIR / log_type
            output_dir.mkdir(parents=True, exist_ok=True)
            
            with open(output_dir / "embeddings.pkl", 'wb') as f:
                pickle.dump(embeddings, f)
            
            if 'binary_labels' in df and len(df['binary_labels']) > 0:
                binary_labels = np.vstack(df['binary_labels'].tolist())
                with open(output_dir / "labels.pkl", 'wb') as f:
                    pickle.dump(binary_labels, f)
                
                if 'log_type_to_attacks' in df.attrs:
                    attack_types = df.attrs['log_type_to_attacks'][log_type]
                    save_key_file(attack_types, output_dir)
            
            # Create visualization with sampling
            if len(embeddings) > args.sample_size:
                sample_idx = np.random.choice(len(embeddings), args.sample_size, replace=False)
                sample_embeddings = embeddings[sample_idx]
                sample_labels = df['label_json'].iloc[sample_idx].tolist()
                sample_log_types = df['log_type'].iloc[sample_idx].tolist()
            else:
                sample_embeddings = embeddings
                sample_labels = df['label_json'].tolist()
                sample_log_types = df['log_type'].tolist()
                
            visualize_embeddings(
                sample_embeddings, 
                sample_labels,
                sample_log_types,
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
            df_all = load_tfrecord_files()
            if not df_all.empty:
                display_data_distribution(df_all, "all combined")
                df_all = preprocess_logs_and_labels(df_all, use_global_attack_list=True)
                
                model_all = train_fasttext_model(
                    df_all['tokens'].tolist(),
                    vector_size=args.vector_size,
                    epochs=args.epochs,
                    model_name_suffix="_all_combined"
                )
                
                embeddings_all = generate_embeddings(model_all, df_all['tokens'].tolist())
                
                # Save outputs
                with open(OUTPUT_DIR / "embeddings_all_combined.pkl", 'wb') as f:
                    pickle.dump(embeddings_all, f)
                
                if 'binary_labels' in df_all and len(df_all['binary_labels']) > 0 and 'attack_types' in df_all.attrs:
                    binary_labels = np.vstack(df_all['binary_labels'].tolist())
                    with open(OUTPUT_DIR / "labels_all_combined.pkl", 'wb') as f:
                        pickle.dump(binary_labels, f)
                    
                    save_key_file(df_all.attrs['attack_types'], OUTPUT_DIR, "key_all_combined.txt")
                
                # Create visualization with sampling
                if len(embeddings_all) > args.sample_size:
                    sample_idx = np.random.choice(len(embeddings_all), args.sample_size, replace=False)
                    sample_embeddings = embeddings_all[sample_idx]
                    sample_labels = df_all['label_json'].iloc[sample_idx].tolist()
                    sample_log_types = df_all['log_type'].iloc[sample_idx].tolist()
                else:
                    sample_embeddings = embeddings_all
                    sample_labels = df_all['label_json'].tolist()
                    sample_log_types = df_all['log_type'].tolist()
                
                visualize_embeddings(
                    sample_embeddings,
                    sample_labels,
                    sample_log_types,
                    output_file=OUTPUT_DIR / "visualization_all_combined.png"
                )
        
        except Exception as e:
            print(f"Error processing combined log types: {e}")
            import traceback
            traceback.print_exc()
    
    print("\nFastText embedding processing complete!")

if __name__ == "__main__":
    main()
