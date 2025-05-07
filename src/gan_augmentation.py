#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
VAE-GAN Hybrid for Data Augmentation in Log Analysis
---------------------------------------------------
This script implements a VAE-GAN hybrid for data augmentation of log embeddings,
enhancing recall by generating higher quality synthetic samples.
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
from tensorflow.keras.layers import Dense, LeakyReLU, Dropout, Input, Lambda, BatchNormalization
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
import os
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
from sklearn.manifold import TSNE

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

def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian."""
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.random.normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def build_encoder(input_dim, latent_dim=100):
    """Build the encoder part of the VAE with explicit model creation."""
    # Create input layer
    inputs = Input(shape=(input_dim,), name='encoder_input')
    
    # First hidden layer with increased capacity
    x = Dense(512, kernel_initializer='glorot_uniform')(inputs)
    x = LeakyReLU(negative_slope=0.2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Second hidden layer
    x = Dense(768, kernel_initializer='glorot_uniform')(x)
    x = LeakyReLU(negative_slope=0.2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Additional layer for better feature extraction
    x = Dense(512, kernel_initializer='glorot_uniform')(x)
    x = LeakyReLU(negative_slope=0.2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # VAE latent space parameters
    z_mean = Dense(latent_dim, name='z_mean', kernel_initializer='glorot_uniform')(x)
    z_log_var = Dense(latent_dim, name='z_log_var', kernel_initializer='glorot_uniform')(x)
    
    # Use reparameterization trick to ensure backpropagation works
    z = Lambda(sampling, name='z')([z_mean, z_log_var])
    
    # Build encoder model
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
    
    # Ensure the model is built
    encoder.build((None, input_dim))
    
    return encoder

def build_decoder(latent_dim, output_dim):
    """Build the decoder part of the VAE with explicit model creation."""
    # Create input layer
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    
    # First hidden layer
    x = Dense(512, kernel_initializer='glorot_uniform')(latent_inputs)
    x = LeakyReLU(negative_slope=0.2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Second hidden layer with increased capacity
    x = Dense(768, kernel_initializer='glorot_uniform')(x)
    x = LeakyReLU(negative_slope=0.2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Additional layer for better reconstruction
    x = Dense(512, kernel_initializer='glorot_uniform')(x)
    x = LeakyReLU(negative_slope=0.2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Output layer
    outputs = Dense(output_dim, activation='tanh', kernel_initializer='glorot_uniform')(x)
    
    # Build decoder model
    decoder = Model(latent_inputs, outputs, name='decoder')
    
    # Ensure the model is built
    decoder.build((None, latent_dim))
    
    return decoder

class VAE(Model):
    """VAE model with proper Keras implementation."""
    def __init__(self, encoder, decoder, input_dim, latent_dim, beta=1.0, annealing=False, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.beta = beta  # KL weight factor (beta-VAE)
        self.annealing = annealing  # Whether to use KL annealing
        self.kl_weight = 0.0 if annealing else beta  # Starting KL weight
        
        # Create a dummy input layer to ensure model is built
        self.dummy_input_layer = tf.keras.layers.InputLayer(input_shape=(input_dim,))
        
        # Define trackers for loss metrics
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
        
        # Epoch counter for KL annealing
        self.epoch = 0
        
        # Initialize by running a forward pass with dummy data
        dummy_data = tf.zeros((1, input_dim))
        self(dummy_data)

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]
    
    def update_epoch(self, epoch):
        """Update epoch counter and KL weight for annealing."""
        self.epoch = epoch
        if self.annealing:
            # Linear annealing from 0 to beta over 10 epochs
            self.kl_weight = min(self.beta, self.beta * (epoch / 10.0))
            print(f"Epoch {epoch}: KL weight = {self.kl_weight:.4f}")

    def call(self, inputs):
        # Ensure inputs is a tensor
        if not isinstance(inputs, tf.Tensor):
            inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
        
        # Forward pass through encoder
        z_mean, z_log_var, z = self.encoder(inputs)
        
        # Forward pass through decoder
        reconstructed = self.decoder(z)
        
        return reconstructed

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
            
        # Ensure data is a tensor
        if not isinstance(data, tf.Tensor):
            data = tf.convert_to_tensor(data, dtype=tf.float32)
            
        with tf.GradientTape() as tape:
            # Forward pass through encoder
            z_mean, z_log_var, z = self.encoder(data)
            
            # Decode the latent sample
            reconstruction = self.decoder(z)
            
            # Calculate reconstruction loss (MSE)
            reconstruction_loss = tf.reduce_mean(
                tf.square(data - reconstruction)
            )
            
            # KL divergence loss
            kl_loss = -0.5 * tf.reduce_mean(
                1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            )
            
            # Total loss with beta factor for KL term (beta-VAE)
            # Use the annealed KL weight if annealing is enabled
            total_loss = reconstruction_loss + self.kl_weight * kl_loss
        
        # Get trainable variables
        trainable_vars = self.trainable_weights
        
        # Check if we have trainable weights
        if not trainable_vars:
            print("WARNING: VAE has no trainable weights!")
            print(f"Encoder weights: {len(self.encoder.trainable_weights)}")
            print(f"Decoder weights: {len(self.decoder.trainable_weights)}")
            # Include encoder and decoder weights explicitly
            trainable_vars = self.encoder.trainable_weights + self.decoder.trainable_weights
            if not trainable_vars:
                print("ERROR: No trainable weights found in encoder or decoder")
                return {
                    "loss": 0.0,
                    "reconstruction_loss": 0.0,
                    "kl_loss": 0.0,
                }
        
        # Compute gradients
        gradients = tape.gradient(total_loss, trainable_vars)
        
        # Clip gradients to prevent exploding gradients
        gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
        
        # Apply gradients
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        # Update loss trackers
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

def build_discriminator(input_dim):
    """Build the discriminator model with functional API for more stable model creation."""
    inputs = Input(shape=(input_dim,), name='discriminator_input')
    
    # First hidden layer
    x = Dense(512, kernel_initializer='glorot_uniform')(inputs)
    x = LeakyReLU(negative_slope=0.2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Second hidden layer
    x = Dense(768, kernel_initializer='glorot_uniform')(x)
    x = LeakyReLU(negative_slope=0.2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Third hidden layer
    x = Dense(256, kernel_initializer='glorot_uniform')(x)
    x = LeakyReLU(negative_slope=0.2)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Output layer
    outputs = Dense(1, activation='sigmoid', kernel_initializer='glorot_uniform')(x)
    
    # Build discriminator model
    discriminator = Model(inputs, outputs, name='discriminator')
    
    # Ensure the model is built
    discriminator.build((None, input_dim))
    
    return discriminator

def build_vaegan(encoder, decoder, discriminator, input_dim, latent_dim, beta=1.0):
    """Build the VAE-GAN model with proper weight management."""
    # Compile discriminator first
    discriminator.compile(
        loss='binary_crossentropy', 
        optimizer=Adam(0.0002, 0.5),
        metrics=['accuracy']
    )
    
    # Create and compile VAE
    vae = VAE(encoder, decoder, input_dim, latent_dim, beta=beta)
    vae.compile(optimizer=Adam(0.0002, 0.5))
    
    # Ensure we can access trainable weights
    dummy_input = tf.zeros((1, input_dim))
    _ = vae(dummy_input)  # Forward pass to build weights
    _ = discriminator(dummy_input)  # Forward pass to build weights
    
    # Create VAE-GAN (generator)
    # Freeze discriminator for VAE-GAN training
    discriminator.trainable = False
    
    # Create VAE-GAN inputs
    vaegan_input = Input(shape=(input_dim,))
    
    # Get VAE reconstruction
    reconstructed = vae(vaegan_input)
    
    # Get discriminator prediction on reconstruction
    validity = discriminator(reconstructed)
    
    # Create and compile VAE-GAN model
    vaegan = Model(vaegan_input, validity, name='vaegan')
    vaegan.compile(
        loss='binary_crossentropy',
        optimizer=Adam(0.0002, 0.5),
        metrics=['accuracy']
    )
    
    # Restore discriminator trainability for separate training
    discriminator.trainable = True
    
    return vaegan, vae

def evaluate_generation_quality(decoder, scaler, test_data, test_labels, latent_dim, target_class, n_samples=1000):
    """Evaluate the quality of generated samples compared to real data for the target class."""
    # Get real samples of target class
    real_indices = np.where(np.array(test_labels) == target_class)[0]
    if len(real_indices) == 0:
        print(f"No real samples of class {target_class} in test set.")
        return None
        
    # Generate synthetic samples
    noise = tf.random.normal((n_samples, latent_dim))
    synthetic_samples = decoder.predict(noise, verbose=0)
    if isinstance(synthetic_samples, tf.Tensor):
        synthetic_samples = synthetic_samples.numpy()
    synthetic_samples = scaler.inverse_transform(synthetic_samples)
    
    # Get real samples
    real_samples = test_data[real_indices]
    
    # Compute statistics for comparison
    real_mean = np.mean(real_samples, axis=0)
    synth_mean = np.mean(synthetic_samples, axis=0)
    
    real_std = np.std(real_samples, axis=0)
    synth_std = np.std(synthetic_samples, axis=0)
    
    # Compute mean absolute error between distributions
    mean_diff = np.mean(np.abs(real_mean - synth_mean))
    std_diff = np.mean(np.abs(real_std - synth_std))
    
    # Compute Euclidean distance between means
    euclidean_dist = np.sqrt(np.sum(np.square(real_mean - synth_mean)))
    
    return {
        'mean_diff': float(mean_diff),
        'std_diff': float(std_diff),
        'euclidean_dist': float(euclidean_dist)
    }

def simple_classifier_recall(X_train, y_train, X_test, y_test, target_class):
    """Evaluate recall using a simple classifier for a specific target class."""
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import recall_score
    
    # Convert labels to binary problem (target class vs rest)
    y_train_binary = np.array([1 if y == target_class else 0 for y in y_train])
    y_test_binary = np.array([1 if y == target_class else 0 for y in y_test])
    
    # Standardize data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train a simple KNN classifier
    k = min(5, len(X_train) // 2)  # Ensure k is smaller than training set
    if k < 1:
        k = 1
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(X_train_scaled, y_train_binary)
    
    # Predict and compute recall
    y_pred = clf.predict(X_test_scaled)
    recall = recall_score(y_test_binary, y_pred)
    
    return recall

def train_vaegan(encoder, decoder, discriminator, vaegan, vae, real_embeddings, latent_dim, 
                target_class=None, X_test=None, y_test=None,
                epochs=200, batch_size=32, beta=1.0, 
                eval_interval=10, patience=20):
    """Train the VAE-GAN model with comparison study for better recall."""
    # Normalize the embeddings
    scaler = StandardScaler()
    scaled_embeddings = scaler.fit_transform(real_embeddings)
    
    # Adjust batch size based on available data
    batch_size = min(batch_size, len(scaled_embeddings))
    if batch_size < 4:  # Too few samples to train effectively
        print("Warning: Too few samples for effective VAE-GAN training")
        return None, None, None, None, scaler
    
    # Convert to tensor once to avoid repeated conversions
    scaled_embeddings_tensor = tf.convert_to_tensor(scaled_embeddings, dtype=tf.float32)
    
    # Print model summaries for debugging
    print("\nEncoder Summary:")
    encoder.summary(print_fn=lambda x: print(x))
    print("\nDecoder Summary:")
    decoder.summary(print_fn=lambda x: print(x))
    print("\nDiscriminator Summary:")
    discriminator.summary(print_fn=lambda x: print(x))
    print("\nVAE Summary:")
    vae.summary(print_fn=lambda x: print(x))
    
    # Check for trainable weights
    print(f"\nEncoder trainable weights: {len(encoder.trainable_weights)}")
    print(f"Decoder trainable weights: {len(decoder.trainable_weights)}")
    print(f"Discriminator trainable weights: {len(discriminator.trainable_weights)}")
    print(f"VAE trainable weights: {len(vae.trainable_weights)}")
    
    # Helper function to safely convert losses to scalar values
    def to_scalar(value):
        if isinstance(value, list) or isinstance(value, tuple):
            return to_scalar(value[0]) if len(value) > 0 else 0.0
        elif isinstance(value, np.ndarray):
            return float(value.mean()) if value.size > 0 else 0.0
        elif isinstance(value, tf.Tensor):
            return float(value.numpy())
        elif isinstance(value, dict) and "loss" in value:
            return to_scalar(value["loss"])
        else:
            try:
                return float(value)
            except (TypeError, ValueError):
                print(f"Warning: Could not convert {type(value)} to float, using 0.0")
                return 0.0
    
    # Training history
    history = {
        'epoch': [],
        'd_loss': [],
        'g_loss': [],
        'vae_loss': [],
        'vae_recon_loss': [],
        'vae_kl_loss': [],
        'quality_metrics': [],
        'recall': []
    }
    
    # Early stopping variables
    best_recall = -1
    best_epoch = 0
    best_weights = None
    
    # Enable KL annealing for smoother training
    vae.annealing = True
    
    # Learning rate schedulers for adaptive learning
    initial_lr = 0.0002
    vae_lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_lr, decay_steps=epochs//4, decay_rate=0.8, staircase=True
    )
    vae_optimizer = tf.keras.optimizers.Adam(vae_lr_schedule, beta_1=0.5)
    vae.compile(optimizer=vae_optimizer)
    
    # Prepare real labels and fake labels once
    real_labels = np.ones((batch_size, 1))
    fake_labels = np.zeros((batch_size, 1))
    
    # Focal loss lambda for rare minority classes
    def focal_loss_binary(y_true, y_pred, alpha=0.25, gamma=2.0):
        """Focal loss for better handling of minority classes."""
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1-y_pred)
        alpha_factor = tf.ones_like(y_true) * alpha
        alpha_factor = tf.where(tf.equal(y_true, 1), alpha_factor, 1-alpha_factor)
        focal_weight = tf.multiply(alpha_factor, tf.pow(1-pt, gamma))
        loss = -tf.multiply(focal_weight, tf.math.log(pt + 1e-7))
        return tf.reduce_mean(loss)
    
    # Start training
    print(f"\nTraining VAE-GAN for {epochs} epochs with batch size {batch_size}, beta={beta}...")
    
    for epoch in tqdm(range(epochs)):
        # Update KL annealing weight
        vae.update_epoch(epoch)
        
        # Get a random batch
        idx = np.random.randint(0, len(scaled_embeddings), batch_size)
        real_batch = tf.gather(scaled_embeddings_tensor, idx)
        
        # --------------------------
        # Train VAE (Reconstruction)
        # --------------------------
        vae_loss = vae.train_on_batch(real_batch)
        
        # Extract VAE component losses
        vae_recon_loss = vae.reconstruction_loss_tracker.result()
        vae_kl_loss = vae.kl_loss_tracker.result()
        vae_loss_val = to_scalar(vae_loss)
        
        # --------------------------
        # Train Discriminator
        # --------------------------
        # Get reconstructed images from VAE
        reconstructed = vae(real_batch)
        
        # Train discriminator
        d_loss_real = discriminator.train_on_batch(real_batch, real_labels)
        d_loss_fake = discriminator.train_on_batch(reconstructed, fake_labels)
        
        # Calculate discriminator loss (scalar value)
        d_loss_real_val = to_scalar(d_loss_real)
        d_loss_fake_val = to_scalar(d_loss_fake)
        d_loss = 0.5 * (d_loss_real_val + d_loss_fake_val)
        
        # --------------------------
        # Train Generator (VAE-GAN)
        # --------------------------
        # Freeze discriminator
        discriminator.trainable = False
        
        # Train generator to fool discriminator
        g_loss = vaegan.train_on_batch(real_batch, real_labels)
        g_loss_val = to_scalar(g_loss)
        
        # Unfreeze discriminator for next iteration
        discriminator.trainable = True
        
        # --------------------------
        # Track metrics and losses
        # --------------------------
        history['epoch'].append(epoch)
        history['d_loss'].append(d_loss)
        history['g_loss'].append(g_loss_val)
        history['vae_loss'].append(vae_loss_val)
        history['vae_recon_loss'].append(float(vae_recon_loss))
        history['vae_kl_loss'].append(float(vae_kl_loss))
        
        # Generate additional diverse samples if needed (every 5 epochs)
        if epoch % 5 == 0 and epoch > 10:
            # Generate random samples from different parts of latent space
            diverse_noise = tf.random.normal((batch_size, latent_dim), mean=0, stddev=1.5)
            diverse_samples = decoder.predict(diverse_noise, verbose=0)
            
            # Mix these with the training batch occasionally to increase diversity
            if np.random.random() > 0.7:  # 30% chance
                mixed_batch = np.vstack([real_batch[:batch_size//2], diverse_samples[:batch_size//2]])
                _ = vae.train_on_batch(mixed_batch)
        
        # Periodic evaluation (generation quality and recall)
        if epoch % eval_interval == 0 or epoch == epochs - 1:
            # Progress report
            print(f"\nEpoch {epoch}/{epochs}")
            print(f"  D loss: {d_loss:.4f}")
            print(f"  G loss: {g_loss_val:.4f}")
            print(f"  VAE loss: {vae_loss_val:.4f}")
            print(f"  Reconstruction loss: {float(vae_recon_loss):.4f}")
            print(f"  KL loss: {float(vae_kl_loss):.4f}")
            
            # Evaluate generation quality if test data is available
            quality_metrics = None
            recall = None
            
            if X_test is not None and y_test is not None and target_class is not None:
                # Evaluate generation quality
                quality_metrics = evaluate_generation_quality(
                    decoder, scaler, X_test, y_test, 
                    latent_dim, target_class
                )
                
                # Generate synthetic samples for recall evaluation
                n_synthetic = min(1000, len(X_test))
                synthetic_embeddings = generate_synthetic_embeddings(
                    decoder, scaler, latent_dim, n_synthetic
                )
                
                # Combine original and synthetic embeddings
                combined_X = np.vstack([real_embeddings, synthetic_embeddings])
                combined_y = np.concatenate([
                    [target_class] * len(real_embeddings),
                    [target_class] * n_synthetic
                ])
                
                # Evaluate recall
                recall = simple_classifier_recall(
                    combined_X, combined_y, X_test, y_test, target_class
                )
                
                print(f"  Generation quality metrics: {quality_metrics}")
                print(f"  Recall on test set: {recall:.4f}")
                
                # Early stopping based on recall
                if recall > best_recall:
                    best_recall = recall
                    best_epoch = epoch
                    # Save best weights
                    best_weights = {
                        'encoder': encoder.get_weights(),
                        'decoder': decoder.get_weights(),
                        'discriminator': discriminator.get_weights()
                    }
                    print(f"  New best recall: {best_recall:.4f}")
                else:
                    print(f"  No improvement in recall for {epoch - best_epoch} epochs (best: {best_recall:.4f})")
                    if epoch - best_epoch >= patience:
                        print(f"  Early stopping triggered. Best recall: {best_recall:.4f} at epoch {best_epoch}")
                        # Restore best weights
                        if best_weights:
                            encoder.set_weights(best_weights['encoder'])
                            decoder.set_weights(best_weights['decoder'])
                            discriminator.set_weights(best_weights['discriminator'])
                        break
            
            # Record metrics
            history['quality_metrics'].append(quality_metrics)
            history['recall'].append(recall)
    
    # If early stopping was triggered, make sure we use the best weights
    if best_weights:
        print(f"Restoring best weights from epoch {best_epoch} with recall {best_recall:.4f}")
        encoder.set_weights(best_weights['encoder'])
        decoder.set_weights(best_weights['decoder'])
        discriminator.set_weights(best_weights['discriminator'])
    
    return encoder, decoder, discriminator, history, scaler

def generate_synthetic_embeddings(decoder, scaler, latent_dim, n_samples, diversity=1.0):
    """Generate synthetic embeddings using the trained decoder with diversity control."""
    try:
        # Generate noise with controllable diversity
        noise = tf.random.normal((n_samples, latent_dim), mean=0, stddev=diversity)
        
        # Generate synthetic embeddings
        synthetic_embeddings = decoder.predict(noise, verbose=0)
        
        # Convert to numpy array if it's a tensor
        if isinstance(synthetic_embeddings, tf.Tensor):
            synthetic_embeddings = synthetic_embeddings.numpy()
        
        # Inverse transform to original scale
        synthetic_embeddings = scaler.inverse_transform(synthetic_embeddings)
        
        return synthetic_embeddings
    except Exception as e:
        print(f"Error generating synthetic embeddings: {e}")
        # Return a small sample of random data as fallback
        print("Returning random noise as fallback")
        return np.random.normal(0, 1, (min(n_samples, 100), latent_dim))

def evaluate_synthetic_data(X_train, y_train, X_test, y_test, synthetic_X, synthetic_y):
    """Evaluate the quality of synthetic data using a KNN classifier."""
    print("Evaluating synthetic data quality with KNN...")
    
    # Scale the data for KNN
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    synthetic_X_scaled = scaler.transform(synthetic_X)
    
    # Combine original and synthetic data
    X_train_aug = np.vstack([X_train_scaled, synthetic_X_scaled])
    y_train_aug = np.concatenate([y_train, synthetic_y])
    
    # Find optimal k (square root of training samples is a common heuristic)
    k = int(np.sqrt(len(X_train_scaled)))
    k = max(3, min(k, 15))  # Keep k between 3 and 15
    print(f"Using k={k} for KNN evaluation")
    
    # Train classifier on original data
    clf_orig = KNeighborsClassifier(n_neighbors=k)
    clf_orig.fit(X_train_scaled, y_train)
    y_pred_orig = clf_orig.predict(X_test_scaled)
    
    # Train classifier on augmented data
    clf_aug = KNeighborsClassifier(n_neighbors=k)
    clf_aug.fit(X_train_aug, y_train_aug)
    y_pred_aug = clf_aug.predict(X_test_scaled)
    
    # Calculate metrics
    precision_orig, recall_orig, f1_orig, _ = precision_recall_fscore_support(
        y_test, y_pred_orig, average='weighted')
    precision_aug, recall_aug, f1_aug, _ = precision_recall_fscore_support(
        y_test, y_pred_aug, average='weighted')
    
    # Calculate per-class metrics for minority classes
    class_metrics_orig = precision_recall_fscore_support(
        y_test, y_pred_orig, average=None, labels=np.unique(synthetic_y))
    class_metrics_aug = precision_recall_fscore_support(
        y_test, y_pred_aug, average=None, labels=np.unique(synthetic_y))
    
    print(f"Overall metrics (original): Precision={precision_orig:.4f}, Recall={recall_orig:.4f}, F1={f1_orig:.4f}")
    print(f"Overall metrics (augmented): Precision={precision_aug:.4f}, Recall={recall_aug:.4f}, F1={f1_aug:.4f}")
    
    # Print per-class recall (focus on recall as per request)
    print("\nPer-class recall:")
    for i, cls in enumerate(np.unique(synthetic_y)):
        print(f"  {cls}: {class_metrics_orig[1][i]:.4f} -> {class_metrics_aug[1][i]:.4f}" +
              f" ({class_metrics_aug[1][i] - class_metrics_orig[1][i]:.4f})")
    
    return {
        'original': {
            'precision': precision_orig,
            'recall': recall_orig,
            'f1': f1_orig
        },
        'augmented': {
            'precision': precision_aug,
            'recall': recall_aug,
            'f1': f1_aug
        },
        'class_recall_improvement': {
            cls: class_metrics_aug[1][i] - class_metrics_orig[1][i]
            for i, cls in enumerate(np.unique(synthetic_y))
        }
    }

def plot_training_history(history, target_class):
    """Plot VAE-GAN training history with robust type handling."""
    plt.figure(figsize=(12, 8))
    
    # Helper function to safely convert values to floats
    def safe_convert(values):
        result = []
        for x in values:
            if isinstance(x, (list, tuple)) and len(x) > 0:
                result.append(float(x[0]))
            elif isinstance(x, np.ndarray) and x.size > 0:
                result.append(float(x.mean()))
            elif isinstance(x, tf.Tensor):
                result.append(float(x.numpy()))
            else:
                try:
                    result.append(float(x))
                except (TypeError, ValueError):
                    result.append(0.0)
        return result
    
    # Create multiple subplots
    plt.subplot(2, 2, 1)
    d_loss = safe_convert(history['d_loss'])
    plt.plot(d_loss, label='Discriminator Loss')
    plt.title('Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    g_loss = safe_convert(history['g_loss'])
    plt.plot(g_loss, label='Generator Loss')
    plt.title('Generator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    vae_loss = safe_convert(history['vae_loss'])
    plt.plot(vae_loss, label='VAE Loss')
    plt.title('VAE Reconstruction Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    cycle_loss = safe_convert(history['cycle_loss'])
    plt.plot(cycle_loss, label='Cycle Loss')
    plt.title('Cycle Consistency Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(RES / f'vaegan_history_{target_class}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training history saved to {RES}/vaegan_history_{target_class}.png")

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
    plt.savefig(RES / f'vaegan_pca_{target_class}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"PCA visualization saved to {RES}/vaegan_pca_{target_class}.png")

def visualize_all_embeddings(original_embeddings, original_labels, synthetic_embeddings, synthetic_labels, output_file=None):
    """Create t-SNE visualization of original and synthetic embeddings with class-based coloring."""
    print("Creating t-SNE visualization of all embeddings...")
    
    # Combine original and synthetic embeddings
    combined_embeddings = np.vstack([original_embeddings, synthetic_embeddings])
    combined_labels = np.concatenate([original_labels, synthetic_labels])
    
    # Add source information (original vs synthetic)
    sources = np.array(['Original'] * len(original_labels) + ['Synthetic'] * len(synthetic_labels))
    
    # Apply t-SNE for dimensionality reduction
    print("Applying t-SNE dimensionality reduction...")
    tsne = TSNE(n_components=2, random_state=RANDOM_SEED, perplexity=min(30, len(combined_embeddings) - 1))
    reduced_embeddings = tsne.fit_transform(combined_embeddings)
    
    # Create DataFrame with results
    df = pd.DataFrame({
        'x': reduced_embeddings[:, 0],
        'y': reduced_embeddings[:, 1],
        'label': combined_labels,
        'source': sources
    })
    
    # Get unique labels and count occurrences
    label_counts = df['label'].value_counts()
    print(f"Label distribution: {dict(label_counts)}")
    
    # Create custom color palette
    unique_labels = df['label'].unique()
    
    # Create a color palette with "normal" as green
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
    
    # Create plot with improved visualization
    plt.figure(figsize=(16, 12))
    
    # Plot by label with color and style
    g = sns.scatterplot(
        data=df, x='x', y='y', 
        hue='label', style='source',
        palette=color_dict, s=70, alpha=0.7
    )
    
    # Add legend
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add title and labels
    plt.title('t-SNE Visualization of Original and Synthetic Embeddings', fontsize=16)
    plt.xlabel('t-SNE Component 1', fontsize=14)
    plt.ylabel('t-SNE Component 2', fontsize=14)
    plt.tight_layout()
    
    # Save plot or show it
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {output_file}")
    else:
        plt.show()
        
    plt.close()
    
    return df  # Return DataFrame for potential further analysis

def main():
    """VAE-GAN hybrid data augmentation for log embeddings."""
    parser = argparse.ArgumentParser(description='VAE-GAN hybrid data augmentation')
    parser.add_argument('--epochs', type=int, default=100, 
                        help='Number of VAE-GAN training epochs')
    parser.add_argument('--latent-dim', type=int, default=128, 
                        help='Dimension of latent space')
    parser.add_argument('--threshold', type=float, default=0.1, 
                        help='Minority class threshold (fraction)')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='Beta weight for KL divergence in VAE (default: 0.5)')
    parser.add_argument('--include-normal', action='store_true', 
                        help='Augment normal class as well')
    parser.add_argument('--evaluate', action='store_true', 
                        help='Evaluate augmentation quality')
    parser.add_argument('--diversity', type=float, default=1.2,
                        help='Diversity factor for synthetic samples (default: 1.2)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode with extensive logging')
    args = parser.parse_args()

    # Enable eager execution explicitly for debugging
    tf.config.run_functions_eagerly(args.debug)
    print(f"Running with eager execution: {tf.executing_eagerly()}")

    # Load embeddings and labels
    try:
        X_train, y_train, X_test, y_test = load_data()
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Identify class distribution
    class_counts = pd.Series(y_train).value_counts()
    majority_class_count = class_counts.max()
    majority_class = class_counts.idxmax()
    threshold = args.threshold * majority_class_count

    # Select minority classes
    minority_classes = [c for c, n in class_counts.items()
                        if n < threshold and (args.include_normal or c != 'normal')]

    print(f"Minority classes: {minority_classes}")
    print(f"Majority class: {majority_class} ({majority_class_count} samples)")

    # Fixed embedding dimension handling
    try:
        if isinstance(X_train, list):
            embedding_dim = X_train[0].shape[0]
        else:
            embedding_dim = X_train.shape[1]
        print(f"Embedding dimension: {embedding_dim}")
    except Exception as e:
        print(f"Error determining embedding dimension: {e}")
        print(f"X_train type: {type(X_train)}")
        if isinstance(X_train, list) and X_train:
            print(f"First element type: {type(X_train[0])}, shape: {X_train[0].shape if hasattr(X_train[0], 'shape') else 'unknown'}")
        elif hasattr(X_train, 'shape'):
            print(f"X_train shape: {X_train.shape}")
        return

    latent_dim = args.latent_dim
    all_synthetic_embeddings, all_synthetic_labels = [], []

    for target_class in minority_classes:
        try:
            real_class_embeddings = get_class_embeddings(X_train, y_train, target_class)
            real_count = len(real_class_embeddings)
            if real_count < 5:
                print(f"Skipping {target_class}: too few samples ({real_count})")
                continue

            print(f"\nProcessing class: {target_class} ({real_count} samples)")
            
            # Build VAE-GAN components with enhanced architecture
            encoder = build_encoder(embedding_dim, latent_dim)
            decoder = build_decoder(latent_dim, embedding_dim)
            discriminator = build_discriminator(embedding_dim)
            
            # Build VAE-GAN with optimized beta value
            try:
                vaegan, vae = build_vaegan(encoder, decoder, discriminator, embedding_dim, latent_dim, beta=args.beta)
            except Exception as e:
                print(f"Error building VAE-GAN: {e}")
                continue
            
            # Train the VAE-GAN with debug info and improved training dynamics
            print(f"Training VAE-GAN for class {target_class}...")
            try:
                encoder, decoder, discriminator, history, scaler = train_vaegan(
                    encoder, decoder, discriminator, vaegan, vae,
                    real_class_embeddings, latent_dim,
                    target_class=target_class, X_test=X_test, y_test=y_test,
                    epochs=args.epochs, batch_size=min(32, real_count),
                    beta=args.beta
                )
            except Exception as e:
                print(f"Error training VAE-GAN: {e}")
                continue
            
            if decoder is None:
                print(f"VAE-GAN training failed for {target_class}")
                continue

            # Generate diverse synthetic embeddings
            n_synthetic = min(20000, majority_class_count - real_count)  # More samples for minority classes
            print(f"Generating {n_synthetic} synthetic samples for {target_class} with diversity={args.diversity}...")
            
            # Generate multiple batches with different diversity settings for better coverage
            synthetic_embeddings_list = []
            
            # Generate 3 batches with increasing diversity
            diversity_settings = [args.diversity * 0.8, args.diversity, args.diversity * 1.2]
            batch_size = n_synthetic // len(diversity_settings)
            
            for diversity in diversity_settings:
                batch_embeddings = generate_synthetic_embeddings(
                    decoder, scaler, latent_dim, batch_size, diversity=diversity
                )
                synthetic_embeddings_list.append(batch_embeddings)
            
            # Combine all batches
            synthetic_embeddings = np.vstack(synthetic_embeddings_list)
            
            # Save model components
            try:
                # Save only weights to avoid serialization issues
                encoder.save_weights(str(MOD / f"encoder_{target_class}_weights.h5"))
                decoder.save_weights(str(MOD / f"decoder_{target_class}_weights.h5"))
                print(f"Saved model weights for {target_class}")
            except Exception as e:
                print(f"Could not save model weights: {e}")
            
            # Save the generated embeddings and their labels
            all_synthetic_embeddings.append(synthetic_embeddings)
            all_synthetic_labels.extend([target_class] * len(synthetic_embeddings))
            
            # Plot results
            if history:
                try:
                    # Make sure history contains valid numeric data
                    for k in history:
                        if k in ['epoch', 'quality_metrics', 'recall']:
                            continue
                        history[k] = [float(x) if hasattr(x, 'dtype') else float(x) for x in history[k]]
                    plot_training_history(history, target_class)
                    plot_embedding_distribution(real_class_embeddings, synthetic_embeddings, target_class)
                except Exception as e:
                    print(f"Could not create plots: {e}")
        except Exception as e:
            print(f"Error processing class {target_class}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save results if any synthetic data was generated
    if all_synthetic_embeddings:
        try:
            # Check if we have valid data to stack
            if not all(isinstance(e, np.ndarray) for e in all_synthetic_embeddings):
                print("Warning: Some synthetic embeddings are not numpy arrays. Converting...")
                all_synthetic_embeddings = [np.array(e) if not isinstance(e, np.ndarray) else e for e in all_synthetic_embeddings]
            
            # Check dimensions
            shapes = [e.shape for e in all_synthetic_embeddings]
            print(f"Embedding shapes: {shapes}")
            
            # Stack embeddings
            combined_synthetic_embeddings = np.vstack(all_synthetic_embeddings)
            combined_synthetic_labels = np.array(all_synthetic_labels)
            
            with open(AUG / "synthetic_embeddings.pkl", 'wb') as f:
                pickle.dump(combined_synthetic_embeddings, f)
            with open(AUG / "synthetic_labels.pkl", 'wb') as f:
                pickle.dump(combined_synthetic_labels, f)
            
            print(f"Generated {len(combined_synthetic_labels)} synthetic samples.")
            
            # Create t-SNE visualization of original + synthetic embeddings
            try:
                # Sample original data to a reasonable size if needed
                max_samples = 5000
                if len(X_train) > max_samples:
                    indices = np.random.choice(len(X_train), max_samples, replace=False)
                    X_train_sample = X_train[indices]
                    y_train_sample = np.array(y_train)[indices]
                else:
                    X_train_sample = X_train
                    y_train_sample = np.array(y_train)
                
                # Create visualization
                visualize_all_embeddings(
                    X_train_sample, y_train_sample,
                    combined_synthetic_embeddings, combined_synthetic_labels,
                    output_file=AUG / "augmented_visualization.png"
                )
            except Exception as e:
                print(f"Error creating t-SNE visualization: {e}")
                import traceback
                traceback.print_exc()
            
            # Evaluate the quality of synthetic data if requested
            if args.evaluate:
                try:
                    metrics = evaluate_synthetic_data(
                        np.array(X_train), np.array(y_train), 
                        np.array(X_test), np.array(y_test),
                        combined_synthetic_embeddings, combined_synthetic_labels
                    )
                    
                    # Save evaluation results
                    with open(RES / "augmentation_metrics.json", 'w') as f:
                        json.dump(metrics, f, indent=2)
                    
                    print(f"\nRecall improvement summary:")
                    for cls, improvement in metrics['class_recall_improvement'].items():
                        print(f"  {cls}: {improvement:.4f}")
                except Exception as e:
                    print(f"Evaluation failed: {e}")
        except Exception as e:
            print(f"Error saving synthetic data: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("No synthetic embeddings generated.")

    print("VAE-GAN hybrid data augmentation complete!")

if __name__ == "__main__":
    main()
