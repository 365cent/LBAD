#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GAN-based Data Augmentation for Log Data
----------------------------------------
This script implements GAN-based data augmentation for log data,
generating synthetic log embeddings to address class imbalance.
Optimized for Apple Silicon (M1/M2/M3) processors.
"""

import sys
import json
import pickle
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LeakyReLU, Dropout, Input
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tqdm import tqdm
import os

# Project paths
ROOT = Path(__file__).resolve().parent.parent
EMB, MOD, RES = ROOT / 'embeddings', ROOT / 'models', ROOT / 'results'
AUG = ROOT / 'augmented'  # For storing augmented data
[d.mkdir(exist_ok=True) for d in (EMB, MOD, RES, AUG)]

# For Apple Silicon optimization
CPU_COUNT = os.cpu_count()
if CPU_COUNT:
    N_JOBS = max(1, CPU_COUNT - 1)  # Leave one core free
else:
    N_JOBS = -1  # Use all cores

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

def parse_labels(label_strings):
    """Convert JSON label strings to classification labels."""
    parsed = []
    for label in label_strings:
        try:
            data = json.loads(label)
            if isinstance(data, list):
                parsed.append("normal" if not data else data[0])
            else:
                parsed.append("unknown")
        except:
            parsed.append("unknown")
    return parsed

def load_data():
    """Load embeddings and labels from separate pickle files."""
    print("Loading data...")
    
    # Load embeddings
    try:
        with open(EMB / 'train_embeddings.pkl', 'rb') as f: 
            X_train = pickle.load(f)
        with open(EMB / 'test_embeddings.pkl', 'rb') as f: 
            X_test = pickle.load(f)
    except FileNotFoundError:
        print("Error: Embedding files not found. Run fasttext_embedding.py first.")
        sys.exit(1)
    
    # Load and parse labels
    try:
        with open(EMB / 'train_labels.pkl', 'rb') as f:
            y_train_raw = pickle.load(f)
        with open(EMB / 'test_labels.pkl', 'rb') as f:
            y_test_raw = pickle.load(f)
        
        y_train = parse_labels(y_train_raw)
        y_test = parse_labels(y_test_raw)
    except FileNotFoundError:
        print("Error: Label files not found. Run fasttext_embedding.py first.")
        sys.exit(1)
    
    print(f"Loaded {len(X_train)} training samples, {len(X_test)} test samples")
    
    # Display label distribution
    label_dist = pd.Series(y_train).value_counts().to_dict()
    print(f"Training label distribution: {label_dist}")
    
    return X_train, y_train, X_test, y_test

def get_class_embeddings(embeddings, labels, target_class):
    """Extract embeddings for a specific class."""
    # Convert to numpy arrays for consistent processing
    embeddings_array = np.array(embeddings)
    labels_array = np.array(labels)
    
    # Get indices for the target class
    class_indices = np.where(labels_array == target_class)[0]
    return embeddings_array[class_indices]

def build_generator(input_dim, output_dim):
    """Build the generator model."""
    model = Sequential()
    
    # First hidden layer
    model.add(Dense(128, input_dim=input_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))
    
    # Second hidden layer
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))
    
    # Third hidden layer
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))
    
    # Output layer
    model.add(Dense(output_dim, activation='tanh'))
    
    return model

def build_discriminator(input_dim):
    """Build the discriminator model."""
    model = Sequential()
    
    # First hidden layer
    model.add(Dense(512, input_dim=input_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))
    
    # Second hidden layer
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))
    
    # Third hidden layer
    model.add(Dense(128))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))
    
    # Output layer
    model.add(Dense(1, activation='sigmoid'))
    
    return model

def build_gan(generator, discriminator):
    """Build the GAN model."""
    # Make the discriminator non-trainable for the combined model
    discriminator.trainable = False
    
    # GAN input (noise) and output (generated embedding)
    gan_input = Input(shape=(generator.input_shape[1],))
    generated_embedding = generator(gan_input)
    gan_output = discriminator(generated_embedding)
    
    # Define GAN model
    gan = Model(gan_input, gan_output)
    
    return gan

def train_gan(generator, discriminator, gan, real_embeddings, noise_dim, epochs=2000, batch_size=32):
    """Train the GAN model."""
    # Normalize the embeddings to [-1, 1] range (tanh output range)
    scaler = StandardScaler()
    scaled_embeddings = scaler.fit_transform(real_embeddings)
    
    # Get efficient batch size (32 or smaller if not enough samples)
    batch_size = min(batch_size, len(scaled_embeddings))
    if batch_size < 4:  # Too few samples to train a GAN effectively
        print("Warning: Too few samples for effective GAN training")
        return None, None, None, scaler
    
    # Adversarial ground truths
    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))
    
    # Training history
    history = {
        'd_loss': [],
        'g_loss': []
    }
    
    # Start training
    print(f"Training GAN for {epochs} epochs with batch size {batch_size}...")
    for epoch in tqdm(range(epochs)):
        # ---------------------
        #  Train Discriminator
        # ---------------------
        
        # Select a random batch of real embeddings
        idx = np.random.randint(0, scaled_embeddings.shape[0], batch_size)
        real_batch = scaled_embeddings[idx]
        
        # Generate a batch of fake embeddings
        noise = np.random.normal(0, 1, (batch_size, noise_dim))
        fake_batch = generator.predict(noise, verbose=0)
        
        # Train the discriminator
        d_loss_real = discriminator.train_on_batch(real_batch, valid)
        d_loss_fake = discriminator.train_on_batch(fake_batch, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        # ---------------------
        #  Train Generator
        # ---------------------
        
        # Generate a batch of noise
        noise = np.random.normal(0, 1, (batch_size, noise_dim))
        
        # Train the generator
        g_loss = gan.train_on_batch(noise, valid)
        
        # Store losses
        history['d_loss'].append(d_loss)
        history['g_loss'].append(g_loss)
        
        # Progress report every 10% of epochs
        if epoch % (epochs // 10) == 0:
            if isinstance(d_loss, list):
                d_loss_val = d_loss[0]  # Extract loss value if it's a list
            else:
                d_loss_val = d_loss
            
            # Fix the print statement to handle numpy arrays properly
            if isinstance(d_loss_val, np.ndarray):
                d_loss_display = d_loss_val.mean() if d_loss_val.size > 0 else 0.0
            else:
                d_loss_display = d_loss_val
            
            if isinstance(g_loss, np.ndarray):
                g_loss_display = g_loss.mean() if g_loss.size > 0 else 0.0
            else:
                g_loss_display = g_loss
            
            print(f"Epoch {epoch}/{epochs} [D loss: {d_loss_display:.4f}] [G loss: {g_loss_display:.4f}]")
    
    return generator, discriminator, history, scaler

def generate_synthetic_embeddings(generator, scaler, noise_dim, n_samples):
    """Generate synthetic embeddings using the trained generator."""
    # Generate noise
    noise = np.random.normal(0, 1, (n_samples, noise_dim))
    
    # Generate synthetic embeddings
    synthetic_embeddings = generator.predict(noise, verbose=0)
    
    # Inverse transform to original scale
    synthetic_embeddings = scaler.inverse_transform(synthetic_embeddings)
    
    return synthetic_embeddings

def plot_training_history(history, target_class):
    """Plot GAN training history."""
    plt.figure(figsize=(10, 6))
    
    # Extract losses (ensuring they're scalar values)
    d_loss = history['d_loss']
    if isinstance(d_loss[0], list):
        d_loss = [x[0] for x in d_loss]  # Extract loss value if it's a list
        
    plt.plot(d_loss, label='Discriminator Loss')
    plt.plot(history['g_loss'], label='Generator Loss')
    plt.title(f'GAN Training History for {target_class}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(RES / f'gan_history_{target_class}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training history saved to {RES}/gan_history_{target_class}.png")

def plot_embedding_distribution(real_embeddings, synthetic_embeddings, target_class):
    """Plot distribution of real and synthetic embeddings using PCA."""
    # Combine real and synthetic embeddings
    combined_embeddings = np.vstack([real_embeddings, synthetic_embeddings])
    
    # Apply PCA
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(combined_embeddings)
    
    # Split back into real and synthetic
    n_real = real_embeddings.shape[0]
    reduced_real = reduced_embeddings[:n_real]
    reduced_synthetic = reduced_embeddings[n_real:]
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.scatter(reduced_real[:, 0], reduced_real[:, 1], alpha=0.7, label='Real', color='blue')
    plt.scatter(reduced_synthetic[:, 0], reduced_synthetic[:, 1], alpha=0.7, label='Synthetic', color='red')
    plt.title(f'PCA of Real and Synthetic Embeddings for {target_class}')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(RES / f'gan_pca_{target_class}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"PCA visualization saved to {RES}/gan_pca_{target_class}.png")

def main():
    """Main function to orchestrate the GAN-based data augmentation workflow."""
    parser = argparse.ArgumentParser(description='GAN-based data augmentation for log analysis')
    parser.add_argument('--epochs', type=int, default=2000, help='Number of training epochs')
    parser.add_argument('--noise-dim', type=int, default=100, help='Dimension of noise vector')
    parser.add_argument('--threshold', type=float, default=0.1, 
                      help='Threshold for minority class (fraction of majority class)')
    parser.add_argument('--include-normal', action='store_true', 
                      help='Include normal class in augmentation')
    args = parser.parse_args()
    
    # Load data
    X_train, y_train, _, _ = load_data()
    
    # Identify class distribution
    class_counts = pd.Series(y_train).value_counts()
    print("\nClass distribution:")
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count}")
    
    # Define majority and minority classes
    majority_class_count = class_counts.max()
    majority_class = class_counts.idxmax()
    minority_threshold = args.threshold * majority_class_count
    
    minority_classes = [class_name for class_name, count in class_counts.items() 
                       if count < minority_threshold and 
                       (args.include_normal or class_name != 'normal')]
    
    print(f"\nIdentified {len(minority_classes)} minority classes: {minority_classes}")
    print(f"Majority class: {majority_class} with {majority_class_count} samples")
    print(f"Threshold for minority class: {minority_threshold} samples")
    
    # Set GAN parameters
    noise_dim = args.noise_dim
    embedding_dim = X_train.shape[1]
    
    # Track all generated synthetic data
    all_synthetic_embeddings = []
    all_synthetic_labels = []
    
    # Process each minority class
    for target_class in minority_classes:
        print(f"\n{'='*60}")
        print(f"Processing minority class: {target_class}")
        print(f"{'='*60}")
        
        # Get real embeddings for the target class
        real_class_embeddings = get_class_embeddings(X_train, y_train, target_class)
        real_count = len(real_class_embeddings)
        
        print(f"Found {real_count} real samples for class '{target_class}'")
        
        # Skip if too few samples (less than 5)
        if real_count < 5:
            print(f"Too few samples for class '{target_class}', skipping GAN training")
            continue
        
        # Build and compile the discriminator
        discriminator = build_discriminator(embedding_dim)
        discriminator.compile(loss='binary_crossentropy', 
                              optimizer=Adam(0.0002, 0.5), 
                              metrics=['accuracy'])
        
        # Build the generator
        generator = build_generator(noise_dim, embedding_dim)
        
        # Build and compile the GAN
        gan = build_gan(generator, discriminator)
        gan.compile(loss='binary_crossentropy', 
                   optimizer=Adam(0.0002, 0.5))
        
        # Train the GAN
        generator, discriminator, history, scaler = train_gan(
            generator, discriminator, gan, real_class_embeddings, noise_dim,
            epochs=args.epochs, batch_size=min(32, real_count)
        )
        
        # Skip if training failed
        if generator is None:
            print(f"GAN training failed for class '{target_class}', skipping")
            continue
        
        # Save the trained models
        generator.save(MOD / f"gan_generator_{target_class}.h5")
        discriminator.save(MOD / f"gan_discriminator_{target_class}.h5")
        
        # Plot training history
        plot_training_history(history, target_class)
        
        # Calculate how many synthetic samples to generate 
        n_synthetic = majority_class_count - real_count
        print(f"Generating {n_synthetic} synthetic samples for class '{target_class}'")
        
        # Generate synthetic embeddings
        synthetic_embeddings = generate_synthetic_embeddings(
            generator, scaler, noise_dim, n_synthetic
        )
        
        # Plot embedding distribution
        plot_embedding_distribution(real_class_embeddings, synthetic_embeddings, target_class)
        
        # Store synthetic embeddings and labels
        all_synthetic_embeddings.append(synthetic_embeddings)
        all_synthetic_labels.extend([target_class] * n_synthetic)
    
    # Save combined synthetic data if any was generated
    if all_synthetic_embeddings:
        combined_synthetic_embeddings = np.vstack(all_synthetic_embeddings)
        combined_synthetic_labels = np.array(all_synthetic_labels)
        
        # Save synthetic embeddings
        with open(AUG / "synthetic_embeddings.pkl", 'wb') as f:
            pickle.dump(combined_synthetic_embeddings, f)
        
        # Save synthetic labels 
        with open(AUG / "synthetic_labels.pkl", 'wb') as f:
            pickle.dump(combined_synthetic_labels, f)
        
        # Also save combined real + synthetic data
        combined_X_train = np.vstack([X_train, combined_synthetic_embeddings])
        combined_y_train = np.concatenate([y_train, combined_synthetic_labels])
        
        with open(AUG / "augmented_train_embeddings.pkl", 'wb') as f:
            pickle.dump(combined_X_train, f)
        
        with open(AUG / "augmented_train_labels.pkl", 'wb') as f:
            pickle.dump(combined_y_train, f)
        
        print(f"\nGenerated a total of {len(combined_synthetic_labels)} synthetic samples")
        print(f"Saved synthetic data to {AUG}/synthetic_embeddings.pkl and {AUG}/synthetic_labels.pkl")
        print(f"Saved augmented training data (real + synthetic) to {AUG}/augmented_train_*.pkl")
        
        # Show final class distribution after augmentation
        augmented_dist = pd.Series(combined_y_train).value_counts()
        print("\nClass distribution after augmentation:")
        for class_name, count in augmented_dist.items():
            orig_count = class_counts.get(class_name, 0)
            added = count - orig_count
            print(f"  {class_name}: {count} (+{added})")
    
    else:
        print("\nNo synthetic embeddings were generated")
    
    print("\nGAN-based data augmentation complete!")

if __name__ == "__main__":
    main()
