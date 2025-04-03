#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TF-IDF Embedding for Log Analysis
---------------------------------
Converts processed TFRecord log files into TF-IDF embeddings for 
analysis and visualization. Optimized for Apple Silicon (M1/M2/M3) processors.
"""

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
import seaborn as sns
import json
import multiprocessing
import joblib
from scipy import sparse

# Configuration
OUTPUT_DIR = Path("embeddings")
MODEL_DIR = Path("models")
PROCESSED_DIR = Path("processed")
VECTOR_SIZE = 100
RANDOM_SEED = 42
MAX_FEATURES = 5000  # Maximum number of features for TF-IDF

# Apple Silicon optimizations
CPU_COUNT = os.cpu_count()
N_JOBS = max(1, CPU_COUNT - 1) if CPU_COUNT else -1  # Use all cores except one

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

def create_tfidf_vectorizer(corpus, max_features=MAX_FEATURES):
    """Create and train a TF-IDF vectorizer."""
    print(f"Creating TF-IDF vectorizer with {max_features} features...")
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=2,           # Minimum document frequency
        max_df=0.9,         # Maximum document frequency
        ngram_range=(1, 2), # Use unigrams and bigrams
        sublinear_tf=True   # Apply sublinear TF scaling (log scaling)
    )
    
    # Fit the vectorizer on the corpus
    vectorizer.fit(corpus)
    print(f"TF-IDF vocabulary size: {len(vectorizer.vocabulary_)}")
    
    return vectorizer

def generate_embeddings(vectorizer, corpus):
    """Generate TF-IDF embeddings for documents."""
    print("Generating TF-IDF embeddings...")
    
    # Process in batches to avoid memory issues
    batch_size = 10000
    num_batches = (len(corpus) + batch_size - 1) // batch_size
    
    all_embeddings = []
    
    for i in tqdm(range(num_batches)):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(corpus))
        batch = corpus[start_idx:end_idx]
        
        # Transform batch to TF-IDF vectors
        batch_embeddings = vectorizer.transform(batch)
        
        all_embeddings.append(batch_embeddings)
    
    # Combine all batches
    if num_batches == 1:
        return all_embeddings[0]
    else:
        return sparse.vstack(all_embeddings)

def reduce_dimensions(embeddings, n_components=VECTOR_SIZE):
    """Reduce dimensionality of embeddings to make them manageable."""
    print(f"Reducing TF-IDF dimensions from {embeddings.shape[1]} to {n_components}...")
    
    svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_SEED)
    reduced_embeddings = svd.fit_transform(embeddings)
    
    # Calculate and display explained variance
    explained_variance = svd.explained_variance_ratio_.sum() * 100
    print(f"Explained variance: {explained_variance:.2f}%")
    
    return reduced_embeddings, svd

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
    tsne = TSNE(
        n_components=2, 
        random_state=RANDOM_SEED, 
        perplexity=30,
        n_jobs=N_JOBS  # Use multiple cores for faster processing
    )
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
    
    plt.title('t-SNE Visualization of TF-IDF Log Embeddings')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {output_file}")
    else:
        plt.show()
    
    plt.close()  # Properly close figure to free memory

def main():
    # Load processed data from TFRecord files
    df = load_tfrecord_files()
    
    # Split data (80% train, 20% test)
    train_df = df.sample(frac=0.8, random_state=RANDOM_SEED)
    test_df = df.drop(train_df.index)
    
    # Create or load TF-IDF vectorizer
    vectorizer_path = MODEL_DIR / "tfidf_vectorizer.joblib"
    
    if vectorizer_path.exists():
        print(f"Loading existing TF-IDF vectorizer from {vectorizer_path}")
        vectorizer = joblib.load(vectorizer_path)
    else:
        vectorizer = create_tfidf_vectorizer(train_df['log'].tolist())
        # Save the vectorizer for future use
        joblib.dump(vectorizer, vectorizer_path)
        print(f"Saved TF-IDF vectorizer to {vectorizer_path}")
    
    # Generate TF-IDF embeddings
    train_sparse_embeddings = generate_embeddings(vectorizer, train_df['log'].tolist())
    test_sparse_embeddings = generate_embeddings(vectorizer, test_df['log'].tolist())
    
    # Reduce dimensions to make embeddings more manageable
    svd_path = MODEL_DIR / "tfidf_svd.joblib"
    
    if svd_path.exists() and train_sparse_embeddings.shape[1] > VECTOR_SIZE:
        print(f"Loading existing SVD model from {svd_path}")
        svd = joblib.load(svd_path)
        train_embeddings = svd.transform(train_sparse_embeddings)
        test_embeddings = svd.transform(test_sparse_embeddings)
    else:
        train_embeddings, svd = reduce_dimensions(train_sparse_embeddings)
        test_embeddings = svd.transform(test_sparse_embeddings)
        # Save the SVD model for future use
        joblib.dump(svd, svd_path)
        print(f"Saved SVD model to {svd_path}")
    
    # Get labels
    train_labels = train_df['label'].tolist()
    test_labels = test_df['label'].tolist()
    
    # Save embeddings
    print("Saving embeddings to disk...")
    with open(OUTPUT_DIR / "tfidf_train_embeddings.pkl", 'wb') as f:
        pickle.dump(train_embeddings, f)
    
    with open(OUTPUT_DIR / "tfidf_test_embeddings.pkl", 'wb') as f:
        pickle.dump(test_embeddings, f)
        
    # Save labels
    with open(OUTPUT_DIR / "tfidf_train_labels.pkl", 'wb') as f:
        pickle.dump(train_labels, f)
    
    with open(OUTPUT_DIR / "tfidf_test_labels.pkl", 'wb') as f:
        pickle.dump(test_labels, f)
    
    print(f"Saved embeddings and labels to {OUTPUT_DIR}")
    print(f"  - Embeddings: tfidf_train_embeddings.pkl, tfidf_test_embeddings.pkl")
    print(f"  - Labels: tfidf_train_labels.pkl, tfidf_test_labels.pkl")
    
    # Visualize (sample up to 5000 points to avoid overcrowding)
    sample_size = min(5000, len(train_embeddings))
    sample_idx = np.random.choice(len(train_embeddings), sample_size, replace=False)
    
    visualize_embeddings(
        train_embeddings[sample_idx], 
        train_df['label'].iloc[sample_idx].tolist(),
        output_file=OUTPUT_DIR / "tfidf_visualization.png"
    )
    
    print("TF-IDF embedding processing complete!")

if __name__ == "__main__":
    main()