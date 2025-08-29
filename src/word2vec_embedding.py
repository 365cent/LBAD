#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Word2Vec Embedding for Log Analysis
----------------------------------
Converts processed TFRecord log files into Word2Vec embeddings for 
analysis and visualization.
"""

import os
import pandas as pd
import numpy as np
import tensorflow as tf
try:
    from gensim.models import Word2Vec
    from gensim.utils import simple_preprocess
    HAS_GENSIM = True
except ImportError:
    HAS_GENSIM = False
    print("Warning: gensim library not available, Word2Vec embeddings disabled")
from pathlib import Path
import pickle
from tqdm import tqdm

# Ensure Matplotlib can write cache/config on HPC
try:
    if "MPLCONFIGDIR" not in os.environ:
        _mpl_dir = Path.cwd() / ".mplconfig"
        _mpl_dir.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(_mpl_dir)
except Exception:
    pass

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import json
import multiprocessing
import argparse

# Configuration
OUTPUT_DIR = Path("embeddings")
MODEL_DIR = Path("models")
PROCESSED_DIR = Path("processed")
VECTOR_SIZE = 100
RANDOM_SEED = 42

# Ensure directories exist
OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

def parse_example(example):
    """Parse a TensorFlow Example protocol buffer."""
    feature_description = {
        'l': tf.io.FixedLenFeature([], tf.string),  # log
        'y': tf.io.FixedLenFeature([], tf.string),  # label
    }
    return tf.io.parse_single_example(example, feature_description)

def load_tfrecord_files(directory=PROCESSED_DIR):
    """Load all TFRecord files from directory into a DataFrame."""
    print(f"Loading TFRecord files from {directory}...")
    tfrecord_files = list(Path(directory).glob("**/*.tfrecord"))
    
    if not tfrecord_files:
        raise FileNotFoundError(f"No TFRecord files found in {directory}")
    
    all_logs = []
    all_labels = []
    
    for file_path in tfrecord_files:
        try:
            # Load dataset with GZIP compression
            dataset = tf.data.TFRecordDataset(str(file_path), compression_type="GZIP")
            
            for raw_record in dataset:
                parsed = parse_example(raw_record)
                log = parsed['l'].numpy().decode('utf-8')
                label = parsed['y'].numpy().decode('utf-8')
                
                all_logs.append(log)
                all_labels.append(label)
                
        except Exception as e:
            print(f"Error with {file_path}: {e}")
    
    print(f"Loaded {len(all_logs)} log entries")
    return pd.DataFrame({'log': all_logs, 'label': all_labels})

def preprocess_logs(df):
    """Tokenize log entries for Word2Vec training."""
    print("Tokenizing log entries...")
    df['tokens'] = df['log'].apply(lambda x: simple_preprocess(str(x)))
    return df

def train_word2vec_model(corpus, vector_size=VECTOR_SIZE, window=5, min_count=1, epochs=10):
    """Train a Word2Vec model on the corpus."""
    print(f"Training Word2Vec model on {len(corpus)} documents...")
    model = Word2Vec(
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=max(1, multiprocessing.cpu_count() - 1),  # Use all cores except one
        seed=RANDOM_SEED,
        sg=1  # Use skip-gram algorithm (1) instead of CBOW (0)
    )
    
    # Build vocabulary and train
    model.build_vocab(corpus_iterable=corpus)
    model.train(
        corpus_iterable=corpus,
        total_examples=len(corpus),
        epochs=epochs
    )
    
    return model

def generate_embeddings(model, corpus):
    """Generate document embeddings by averaging word vectors."""
    print("Generating document embeddings...")
    embeddings = []
    
    for doc in tqdm(corpus):
        # Get word vectors for each word in document
        word_vectors = [model.wv[word] for word in doc if word in model.wv]
        
        # Average the vectors (or use zeros if no words have embeddings)
        if word_vectors:
            doc_vector = np.mean(word_vectors, axis=0)
        else:
            doc_vector = np.zeros(model.vector_size)
            
        embeddings.append(doc_vector)
    
    return np.array(embeddings)

def visualize_embeddings(embeddings, labels, output_file=None):
    """Create t-SNE visualization of embeddings with better label handling."""
    print("Creating t-SNE visualization...")
    
    # Parse the JSON strings to get actual labels
    parsed_labels = []
    for label_str in labels:
        try:
            label_data = json.loads(label_str)
            if isinstance(label_data, list):
                if not label_data:  # Empty array means "normal"
                    parsed_labels.append("normal")
                else:
                    parsed_labels.append(label_data[0])  # Use first label if multiple
            else:
                parsed_labels.append("unknown")
        except:
            parsed_labels.append("unknown")
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=RANDOM_SEED, perplexity=30)
    reduced = tsne.fit_transform(embeddings)
    
    # Create plot with improved visualization
    plt.figure(figsize=(12, 10))
    
    # Create DataFrame with results
    df = pd.DataFrame({
        'x': reduced[:, 0],
        'y': reduced[:, 1],
        'label': parsed_labels
    })
    
    # Get unique labels and count occurrences
    label_counts = df['label'].value_counts()
    print(f"Label distribution: {dict(label_counts)}")
    
    # Use a better color palette with distinct colors for different labels
    colors = sns.color_palette("husl", len(df['label'].unique()))
    sns.scatterplot(x='x', y='y', hue='label', data=df, palette=colors)
    
    plt.title('t-SNE Visualization of Word2Vec Log Embeddings')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {output_file}")
    else:
        plt.show()
    
    plt.close()  # Properly close figure to free memory

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
    
    for label_json_str in df['label']:
        all_unique_labels.update(get_labels_from_json(label_json_str))
    
    # Remove empty labels
    all_unique_labels.discard('')
    all_unique_labels.discard(None)
    
    return sorted(list(all_unique_labels))

def create_binary_label_vector(label_json_str, all_attack_types):
    """Create binary vector representation for multi-label classification."""
    labels = get_labels_from_json(label_json_str)
    
    # Binary vector: [0 1 0] means only second class is present
    binary_vector = np.zeros(len(all_attack_types), dtype=np.int8)
    
    if labels:
        attack_indices = [i for i, attack in enumerate(all_attack_types) if attack in labels]
        binary_vector[attack_indices] = 1
    
    return binary_vector

def find_available_log_types():
    """Find available log types in the processed directory."""
    if not PROCESSED_DIR.exists():
        return []
    return sorted([path.name for path in PROCESSED_DIR.iterdir() 
                  if path.is_dir() and list(path.glob("*.tfrecord"))])

def load_tfrecord_files_by_type(log_type_filter=None):
    """Load TFRecord files by log type."""
    if log_type_filter:
        log_type_dir_path = PROCESSED_DIR / log_type_filter
        if not log_type_dir_path.exists():
            raise FileNotFoundError(f"No directory found for '{log_type_filter}'")
        tfrecord_files = list(log_type_dir_path.glob("*.tfrecord"))
        if not tfrecord_files:
            raise FileNotFoundError(f"No TFRecord files found for log type '{log_type_filter}'")
    else:
        tfrecord_files = list(PROCESSED_DIR.glob("**/*.tfrecord"))
        if not tfrecord_files:
            raise FileNotFoundError(f"No TFRecord files found in {PROCESSED_DIR}")
    
    all_logs = []
    all_labels = []
    all_log_types = []
    
    for file_path in tfrecord_files:
        try:
            log_type = file_path.parent.name
            dataset = tf.data.TFRecordDataset(str(file_path), compression_type="GZIP")
            
            for raw_record in dataset:
                parsed = parse_example(raw_record)
                log = parsed['l'].numpy().decode('utf-8')
                label = parsed['y'].numpy().decode('utf-8')
                
                all_logs.append(log)
                all_labels.append(label)
                all_log_types.append(log_type)
                
        except Exception as e:
            print(f"Error with {file_path}: {e}")
    
    return pd.DataFrame({
        'log': all_logs, 
        'label': all_labels,
        'log_type': all_log_types
    })

def save_embeddings_and_labels_compatible(embeddings, labels, log_type, output_dir):
    """Save embeddings and labels in format compatible with other embedding methods."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save embeddings in compatible format
    log_filename = f"log_{log_type}.pkl"
    with open(output_dir / log_filename, 'wb') as f:
        pickle.dump(embeddings.astype(np.float32), f, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Process labels into binary vectors
    attack_types = collect_unique_labels_from_data(pd.DataFrame({'label': labels}))
    
    if attack_types:
        binary_vectors = []
        for label_json in labels:
            binary_vector = create_binary_label_vector(label_json, attack_types)
            binary_vectors.append(binary_vector)
        
        binary_vectors = np.vstack(binary_vectors).astype(np.int8)
        
        # Create label data structure compatible with other methods
        label_data = {
            'vectors': binary_vectors,
            'classes': attack_types,
            'description': 'Binary multi-label vectors where [0 1 0] means only the second class is present'
        }
        
        label_filename = f"label_{log_type}.pkl"
        with open(output_dir / label_filename, 'wb') as f:
            pickle.dump(label_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Save attack types info
        attack_info_filename = f"attack_types_{log_type}.txt"
        with open(output_dir / attack_info_filename, 'w', encoding='utf-8') as f:
            f.write(f"Attack Types and Column Mapping for {log_type}\n")
            f.write("=" * 60 + "\n\n")
            f.write("Binary Vector Format:\n")
            f.write("- Each row is a binary vector representing one log entry\n")
            f.write("- Vector format: [0 1 0] means only the second attack type is present\n")
            f.write("- Multiple attacks can be present: [1 0 1] means first and third attacks\n\n")
            f.write(f"Total attack types: {len(attack_types)}\n")
            f.write(f"Vector dimension: {len(attack_types)}\n\n")
            
            if attack_types:
                f.write("Column Mapping (Index -> Attack Type):\n")
                f.write("-" * 40 + "\n")
                for i, attack_type in enumerate(attack_types):
                    f.write(f"Column {i:2d}: {attack_type}\n")
            
            f.write(f"\nGenerated by Word2Vec embeddings extraction\n")
            f.write(f"Embedding dimension: {VECTOR_SIZE}D (Word2Vec vectors)\n")
        
        print(f"\nSaved files for {log_type}:")
        print(f"  - {log_filename}: Log embeddings {embeddings.shape} ({VECTOR_SIZE}D Word2Vec vectors)")
        print(f"  - {label_filename}: Binary label vectors {binary_vectors.shape}")
        print(f"  - {attack_info_filename}: Attack types and column mapping details")
        print(f"  - Classes: {attack_types}")
    else:
        print(f"No attack types found for {log_type}, saved only embeddings")

def main():
	parser = argparse.ArgumentParser(description="Generate Word2Vec embeddings from TFRecords")
	parser.add_argument("--output-subdir", type=str, default=None, help="Optional subdirectory under embeddings/ (e.g., 'word2vec')")
	parser.add_argument("--log-type", type=str, default=None, help="Process only this specific log type")
	args = parser.parse_args()
	
	# Check if gensim library is available
	if not HAS_GENSIM:
		print("❌ gensim library not available. Please install it:")
		print("   pip install gensim")
		return
	
	global OUTPUT_DIR
	if args.output_subdir:
		OUTPUT_DIR = Path("embeddings") / args.output_subdir
	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
	
	# Find available log types
	available_types = find_available_log_types()
	if not available_types:
		print("No log types with data found.")
		return
	
	print(f"Found {len(available_types)} log types: {', '.join(available_types)}")
	
	# Determine log types to process
	if args.log_type:
		if args.log_type not in available_types:
			print(f"Log type '{args.log_type}' not found. Available types: {', '.join(available_types)}")
			return
		types_to_process = [args.log_type]
	else:
		types_to_process = available_types
	
	# Process each log type separately
	for log_type in types_to_process:
		print(f"\n{'='*50}\nProcessing log type: {log_type}\n{'='*50}")
		
		try:
			# Load data for this log type
			df = load_tfrecord_files_by_type(log_type_filter=log_type)
			if df.empty:
				print(f"No data for log type '{log_type}'. Skipping.")
				continue
			
			# Preprocess logs
			df = preprocess_logs(df)
			
			# Train Word2Vec model for this log type
			model_path = MODEL_DIR / f"word2vec_{log_type}_model.bin"
			
			if model_path.exists():
				print(f"Loading existing model from {model_path}")
				model = Word2Vec.load(str(model_path))
			else:
				model = train_word2vec_model(df['tokens'].tolist())
				model.save(str(model_path))
				print(f"Saved model to {model_path}")
			
			# Generate embeddings for all data
			embeddings = generate_embeddings(model, df['tokens'].tolist())
			labels = df['label'].tolist()
			
			# Save in compatible format
			output_dir = OUTPUT_DIR / log_type
			save_embeddings_and_labels_compatible(embeddings, labels, log_type, output_dir)
			
			# Create visualization
			sample_size = min(5000, len(embeddings))
			sample_idx = np.random.choice(len(embeddings), sample_size, replace=False)
			
			visualize_embeddings(
				embeddings[sample_idx], 
				df['label'].iloc[sample_idx].tolist(),
				output_file=output_dir / "visualization.png"
			)
			
			print(f"✅ Completed processing {log_type}")
			
		except Exception as e:
			print(f"Error processing log type {log_type}: {e}")
			import traceback
			traceback.print_exc()
	
	print("Word2Vec embedding processing complete!")

if __name__ == "__main__":
    main()