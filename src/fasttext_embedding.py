#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FastText Embedding for Log Analysis - Using Pre-trained Models

Key Improvements:
- Uses the pre-trained FastText wiki-news vectors for immediate embeddings
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

import argparse
import json
import os
import pickle
from functools import lru_cache
from itertools import repeat
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

_BASE_DIR = Path(__file__).resolve().parent.parent
try:
    mpl_dir = _BASE_DIR / ".mplconfig"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_dir)
    gensim_dir = _BASE_DIR / "gensim_data"
    gensim_dir.mkdir(parents=True, exist_ok=True)
    os.environ["GENSIM_DATA_DIR"] = str(gensim_dir)
except Exception:  # pragma: no cover - best effort only
    pass

import numpy as np
import pandas as pd
import tensorflow as tf
try:
    from gensim.models import KeyedVectors
    from gensim.utils import simple_preprocess
    import gensim.downloader as api
    HAS_GENSIM = True
except ImportError:  # pragma: no cover - environment fallback
    KeyedVectors = None  # type: ignore
    simple_preprocess = None  # type: ignore
    api = None  # type: ignore
    HAS_GENSIM = False
    print("Warning: gensim library not available, FastText embeddings disabled")
import matplotlib.pyplot as plt
import seaborn as sns
from halo import Halo
from sklearn.manifold import TSNE

# Configuration
OUTPUT_DIR = Path("embeddings") / "fasttext"
PROCESSED_DIR = Path("processed")
VECTOR_SIZE = 300  # Standard FastText vector size

def parse_example(example):
    """Parse a TensorFlow Example protocol buffer."""
    feature_description = {
        'l': tf.io.FixedLenFeature([], tf.string),  # log
        'y': tf.io.FixedLenFeature([], tf.string),  # label
    }
    return tf.io.parse_single_example(example, feature_description)

def load_tfrecord_files(
    directory: Path = PROCESSED_DIR,
    log_type_filter: Optional[str] = None,
) -> pd.DataFrame:
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

    dataset_options = tf.data.Options()
    dataset_options.experimental_deterministic = False
    batch_size = 1024
    autotune = getattr(tf.data, 'AUTOTUNE', tf.data.experimental.AUTOTUNE)

    for file_idx, file_path in enumerate(tfrecord_files):
        try:
            spinner.text = f"Loading file {file_idx+1}/{len(tfrecord_files)}: {file_path.name}"
            log_type = file_path.parent.name
            
            # Use TensorFlow's optimized API for batch processing
            dataset = tf.data.TFRecordDataset(
                str(file_path),
                compression_type="GZIP",
                num_parallel_reads=autotune,
            )
            dataset = dataset.with_options(dataset_options)
            dataset = dataset.batch(batch_size).prefetch(autotune)
            
            for batch in dataset:
                parsed_batch = tf.io.parse_example(batch, {
                    'l': tf.io.FixedLenFeature([], tf.string),
                    'y': tf.io.FixedLenFeature([], tf.string)
                })
                
                logs = [log.decode('utf-8') for log in parsed_batch['l'].numpy()]
                labels = [label.decode('utf-8') for label in parsed_batch['y'].numpy()]
                
                all_logs.extend(logs)
                all_labels_json.extend(labels)
                all_log_types.extend(repeat(log_type, len(logs)))
                
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

def normalize_label(label: Optional[str]) -> Optional[str]:
    """Normalize attack labels to ensure consistency."""
    if not label:
        return label
    return label.replace('-', '_').lower().strip()


def get_labels_from_json(label_json_str: str) -> Set[str]:
    """Extract labels from JSON string."""
    try:
        labels = json.loads(label_json_str)
        if not isinstance(labels, list):
            labels = [labels]
        return {normalize_label(label) for label in labels if label}
    except json.JSONDecodeError:
        return set()

def collect_unique_labels_from_data(
    df: pd.DataFrame,
    *,
    show_progress: bool = True,
) -> List[str]:
    """Extract all unique attack labels from the dataset efficiently."""
    all_unique_labels = set()

    spinner = Halo(text='Collecting unique labels', spinner='dots') if show_progress else None
    if spinner:
        spinner.start()

    for label_json_str in df['label_json']:
        all_unique_labels.update(get_labels_from_json(label_json_str))

    all_unique_labels.discard('')
    all_unique_labels.discard(None)

    if spinner:
        spinner.succeed(f"Found {len(all_unique_labels)} unique attack types")

    return sorted(all_unique_labels)

@lru_cache(maxsize=50000)
def _cached_preprocess(text: str) -> Tuple[str, ...]:
    """Tokenize text with caching to avoid repeated preprocessing."""
    return tuple(simple_preprocess(text))


def preprocess_text(text: str) -> Tuple[str, ...]:
    """Preprocess text for embedding."""
    if not text:
        return tuple()
    return _cached_preprocess(text)

def load_pretrained_fasttext() -> Optional[KeyedVectors]:
    """Load the pre-trained FastText model via gensim."""

    if not HAS_GENSIM or api is None:
        return None

    spinner = Halo(text='Loading pre-trained FastText model', spinner='dots')
    spinner.start()

    try:
        model = api.load("fasttext-wiki-news-subwords-300")
        spinner.succeed("Loaded pre-trained FastText model (fasttext-wiki-news-subwords-300)")
        return model
    except Exception as exc:
        spinner.fail(f"Failed to load fasttext-wiki-news-subwords-300: {exc}")
        return None

def embed_text(
    model: KeyedVectors,
    tokens: Sequence[str],
    vector_cache: Optional[MutableMapping[str, np.ndarray]] = None,
) -> np.ndarray:
    """Generate embedding for tokenized text using FastText vectors."""
    if not tokens:
        return np.zeros(model.vector_size, dtype=np.float32)

    cache: MutableMapping[str, np.ndarray]
    cache = vector_cache if vector_cache is not None else {}

    accumulator = np.zeros(model.vector_size, dtype=np.float32)
    count = 0

    for token in tokens:
        vector = cache.get(token)
        if vector is None:
            try:
                vector = np.asarray(model.get_vector(token), dtype=np.float32)
                cache[token] = vector
            except KeyError:
                continue
        accumulator += vector
        count += 1

    if count == 0:
        return np.zeros(model.vector_size, dtype=np.float32)
    return accumulator / count

def embed_labels(model: KeyedVectors, labels: Sequence[str]) -> np.ndarray:
    """Generate embeddings for labels."""
    if not labels:
        return np.zeros(model.vector_size, dtype=np.float32)
    
    label_embeddings = []
    for label in labels:
        if label:
            tokens = preprocess_text(label)
            embedding = embed_text(model, tokens)
            label_embeddings.append(embedding)
    
    if label_embeddings:
        return np.mean(label_embeddings, axis=0)
    else:
        return np.zeros(model.vector_size, dtype=np.float32)

def create_binary_label_vector(
    label_json_str: str,
    all_attack_types: Sequence[str],
    *,
    attack_index: Optional[Mapping[str, int]] = None,
) -> np.ndarray:
    """Create binary vector representation for multi-label classification."""
    labels = get_labels_from_json(label_json_str)

    if not all_attack_types:
        return np.zeros(0, dtype=np.int8)

    lookup = attack_index or {attack: idx for idx, attack in enumerate(all_attack_types)}

    binary_vector = np.zeros(len(all_attack_types), dtype=np.int8)
    for label in labels:
        index = lookup.get(label)
        if index is not None:
            binary_vector[index] = 1

    return binary_vector

def display_data_distribution(
    df: pd.DataFrame,
    log_type_name: str = "all combined",
) -> Tuple[int, int]:
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

def process_embeddings(
    df: pd.DataFrame,
    model: KeyedVectors,
    use_global_attack_list: bool = False,
) -> pd.DataFrame:
    """Process logs and create embeddings with binary label vectors."""
    spinner = Halo(text='Processing log embeddings', spinner='dots')
    spinner.start()

    df = df.copy()
    total_count = len(df)
    if total_count == 0:
        spinner.succeed('No records found for embedding')
        df['log_embedding'] = []
        df['binary_labels'] = []
        return df

    batch_size = 500
    embedding_cache: Dict[str, np.ndarray] = {}
    log_embeddings = np.empty((total_count, model.vector_size), dtype=np.float32)
    binary_labels: List[np.ndarray] = [np.zeros(0, dtype=np.int8) for _ in range(total_count)]

    if use_global_attack_list:
        attack_types = collect_unique_labels_from_data(df, show_progress=False)
        attack_lookup = {attack: idx for idx, attack in enumerate(attack_types)}
        df.attrs['attack_types'] = attack_types
    else:
        log_type_to_attacks: Dict[str, List[str]] = {}
        log_type_to_index: Dict[str, Dict[str, int]] = {}
        for log_type, group_df in df.groupby('log_type'):
            spinner.text = f"Building label vocabulary for {log_type}"
            attacks = collect_unique_labels_from_data(group_df, show_progress=False)
            log_type_to_attacks[log_type] = attacks
            log_type_to_index[log_type] = {attack: idx for idx, attack in enumerate(attacks)}
        df.attrs['log_type_to_attacks'] = log_type_to_attacks

    spinner.text = 'Generating embeddings and label vectors'
    for start in range(0, total_count, batch_size):
        end = min(start + batch_size, total_count)
        spinner.text = f"Processing records {end}/{total_count} ({end/total_count*100:.1f}%)"

        batch = df.iloc[start:end]
        for offset, row in enumerate(batch.itertuples(index=False)):
            idx = start + offset
            tokens = preprocess_text(row.log)
            log_embeddings[idx] = embed_text(model, tokens, vector_cache=embedding_cache)

            if use_global_attack_list:
                binary_labels[idx] = create_binary_label_vector(
                    row.label_json,
                    attack_types,
                    attack_index=attack_lookup,
                )
            else:
                attacks = log_type_to_attacks.get(row.log_type, [])
                if attacks:
                    binary_labels[idx] = create_binary_label_vector(
                        row.label_json,
                        attacks,
                        attack_index=log_type_to_index[row.log_type],
                    )
                else:
                    binary_labels[idx] = np.zeros(0, dtype=np.int8)

    df['log_embedding'] = list(log_embeddings)
    df['binary_labels'] = binary_labels

    spinner.succeed('Embedding processing complete')
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
    rng = np.random.default_rng(42)

    selected_indices = []
    label_to_indices = {}
    for idx, lbl in enumerate(viz_labels):
        label_to_indices.setdefault(lbl, []).append(idx)

    for lbl, indices in label_to_indices.items():
        if len(indices) > MAX_POINTS_PER_CLASS:
            sampled = rng.choice(indices, MAX_POINTS_PER_CLASS, replace=False)
            selected_indices.extend(sampled)
        else:
            selected_indices.extend(indices)  # Keep all if already small

    # If we still exceed the global cap, randomly subsample the union (keeps balance reasonably well)
    if len(selected_indices) > MAX_TOTAL_POINTS:
        selected_indices = list(rng.choice(selected_indices, MAX_TOTAL_POINTS, replace=False))

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
    parser = argparse.ArgumentParser(description="Generate FastText embeddings for log data using pre-trained vectors")
    parser.add_argument("--log-type", type=str, default=None, help="Process only this specific log type")
    parser.add_argument("--output-subdir", type=str, default=None, help="Optional subdirectory under embeddings/ (e.g., 'fasttext')")
    args = parser.parse_args()

    # Check if gensim library is available
    if not HAS_GENSIM:
        print("❌ gensim library not available. Please install it:")
        print("   pip install gensim")
        return

    # Optionally route outputs to embeddings/<subdir>
    global OUTPUT_DIR
    if args.output_subdir:
        OUTPUT_DIR = Path("embeddings") / args.output_subdir

    # Ensure output directories exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
        run_combined = False  # Disable combined processing by default

    # Process individual log types
    for log_type in types_to_process:
        print(f"\n{'='*50}\nProcessing log type: {log_type}\n{'='*50}")
        try:
            # Pre-check: skip if expected outputs already exist
            output_dir = OUTPUT_DIR / log_type
            log_pkl = output_dir / f"log_{log_type}.pkl"
            label_pkl = output_dir / f"label_{log_type}.pkl"
            attack_txt = output_dir / f"attack_types_{log_type}.txt"
            viz_png = output_dir / "visualization.png"
            if log_pkl.exists() and log_pkl.stat().st_size > 0 and \
               label_pkl.exists() and label_pkl.stat().st_size > 0 and \
               attack_txt.exists() and attack_txt.stat().st_size > 0 and \
               viz_png.exists() and viz_png.stat().st_size > 0:
                print(f"Outputs already exist for '{log_type}', skipping.")
                continue
            # Load data for this log type
            df = load_tfrecord_files(log_type_filter=log_type)
            if df.empty:
                print(f"No data for log type '{log_type}'. Skipping.")
                continue

            # Display data distribution
            display_data_distribution(df, log_type)

            vector_model = model
            if vector_model is None:
                print(f"Skipping '{log_type}' due to missing FastText vectors")
                continue

            # Process embeddings
            df = process_embeddings(df, vector_model, use_global_attack_list=False)
            
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
            # Pre-check for combined outputs
            combined_log_pkl = OUTPUT_DIR / "log_all_combined.pkl"
            combined_label_pkl = OUTPUT_DIR / "label_all_combined.pkl"
            combined_viz_png = OUTPUT_DIR / "visualization_all_combined.png"
            if combined_log_pkl.exists() and combined_log_pkl.stat().st_size > 0 and \
               combined_label_pkl.exists() and combined_label_pkl.stat().st_size > 0 and \
               combined_viz_png.exists() and combined_viz_png.stat().st_size > 0:
                print("Combined outputs already exist, skipping combined processing.")
                raise SystemExit
            # Load all data
            df_all = load_tfrecord_files()
            if df_all.empty:
                print("No data found for combined log types")
            else:
                display_data_distribution(df_all, "all combined")
                combined_vector_model = model
                if combined_vector_model is None:
                    print("Skipping combined processing due to missing FastText vectors")
                else:
                    df_all = process_embeddings(df_all, combined_vector_model, use_global_attack_list=True)

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
