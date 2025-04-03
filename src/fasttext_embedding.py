#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FastText Embedding for Log Analysis
----------------------------------
Converts processed TFRecord log files into FastText embeddings for 
analysis and visualization.
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
    """Tokenize log entries for FastText training."""
    print("Tokenizing log entries...")
    df['tokens'] = df['log'].apply(lambda x: simple_preprocess(str(x)))
    return df

def train_fasttext_model(corpus, vector_size=VECTOR_SIZE, window=5, min_count=1, epochs=10):
    """Train a FastText model on the corpus."""
    print(f"Training FastText model on {len(corpus)} documents...")
    model = FastText(
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=multiprocessing.cpu_count() - 1,  # Use all cores except one
        seed=RANDOM_SEED
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
    
    # Create custom color palette with "normal" as green
    unique_labels = df['label'].unique()
    
    # Create a color palette for non-normal classes
    other_labels = [label for label in unique_labels if label != "normal"]
    other_colors = sns.color_palette("husl", len(other_labels))
    
    # Create a dictionary mapping each label to its color
    color_dict = {}
    color_idx = 0
    
    for label in unique_labels:
        if label == "normal":
            color_dict[label] = "green"  # Set normal to green
        else:
            color_dict[label] = other_colors[color_idx]
            color_idx += 1
    
    # Use the custom palette in the scatterplot
    sns.scatterplot(x='x', y='y', hue='label', data=df, palette=color_dict)
    
    plt.title('t-SNE Visualization of FastText Log Embeddings')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {output_file}")
    else:
        plt.show()
    
    plt.close()  # Properly close figure to free memory

def main():
    # Load and preprocess data from TFRecord files
    df = load_tfrecord_files()
    df = preprocess_logs(df)
    
    # Split data (80% train, 20% test)
    train_df = df.sample(frac=0.8, random_state=RANDOM_SEED)
    test_df = df.drop(train_df.index)
    
    # Train FastText model
    model_path = MODEL_DIR / "fasttext_model.bin"
    
    if model_path.exists():
        print(f"Loading existing model from {model_path}")
        model = FastText.load(str(model_path))
    else:
        model = train_fasttext_model(train_df['tokens'].tolist())
        model.save(str(model_path))
        print(f"Saved model to {model_path}")
    
    # Generate embeddings
    train_embeddings = generate_embeddings(model, train_df['tokens'].tolist())
    test_embeddings = generate_embeddings(model, test_df['tokens'].tolist())
    
    # Get labels
    train_labels = train_df['label'].tolist()
    test_labels = test_df['label'].tolist()
    
    # Save embeddings
    with open(OUTPUT_DIR / "train_embeddings.pkl", 'wb') as f:
        pickle.dump(train_embeddings, f)
    
    with open(OUTPUT_DIR / "test_embeddings.pkl", 'wb') as f:
        pickle.dump(test_embeddings, f)
        
    # Save labels (in separate files as requested)
    with open(OUTPUT_DIR / "train_labels.pkl", 'wb') as f:
        pickle.dump(train_labels, f)
    
    with open(OUTPUT_DIR / "test_labels.pkl", 'wb') as f:
        pickle.dump(test_labels, f)
    
    print(f"Saved embeddings and labels to {OUTPUT_DIR}")
    print(f"  - Embeddings: train_embeddings.pkl, test_embeddings.pkl")
    print(f"  - Labels: train_labels.pkl, test_labels.pkl")
    
    # Visualize (sample up to 5000 points to avoid overcrowding)
    sample_size = min(5000, len(train_embeddings))
    sample_idx = np.random.choice(len(train_embeddings), sample_size, replace=False)
    
    visualize_embeddings(
        train_embeddings[sample_idx], 
        train_df['label'].iloc[sample_idx].tolist(),
        output_file=OUTPUT_DIR / "embedding_visualization.png"
    )
    
    print("FastText embedding processing complete!")

if __name__ == "__main__":
    main()
