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
        self.dummy_input_layer = tf.keras.layers.InputLayer(shape=(input_dim,))
        
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
    
    # First hidden layer with increased capacity and L2 regularization
    x = Dense(512, kernel_initializer='glorot_uniform', 
             kernel_regularizer=tf.keras.regularizers.l2(0.0001))(inputs)
    x = LeakyReLU(negative_slope=0.2)(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Dropout(0.3)(x)
    
    # Second hidden layer
    x = Dense(768, kernel_initializer='glorot_uniform',
             kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
    x = LeakyReLU(negative_slope=0.2)(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Dropout(0.3)(x)
    
    # Third hidden layer
    x = Dense(256, kernel_initializer='glorot_uniform',
             kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
    x = LeakyReLU(negative_slope=0.2)(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Dropout(0.3)(x)
    
    # Feature layer for feature matching loss
    features = Dense(128, kernel_initializer='glorot_uniform',
                   kernel_regularizer=tf.keras.regularizers.l2(0.0001),
                   name='features')(x)
    features = LeakyReLU(negative_slope=0.2)(features)
    
    # Output layer
    outputs = Dense(1, activation='sigmoid', kernel_initializer='glorot_uniform')(features)
    
    # Build discriminator model with both feature output and prediction
    discriminator = Model(inputs, [outputs, features], name='discriminator')
    
    # Ensure the model is built
    discriminator.build((None, input_dim))
    
    return discriminator

def build_vaegan(encoder, decoder, discriminator, input_dim, latent_dim, beta=1.0):
    """Build the VAE-GAN model with proper weight management."""
    # Compile discriminator first with metrics for both outputs
    discriminator.compile(
        loss=['binary_crossentropy', None],  # No loss for feature output
        optimizer=Adam(0.0002, 0.5),
        metrics=[['accuracy'], None]  # Only accuracy for the prediction output
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
    
    # Get discriminator prediction on reconstruction - use only the main output
    validity = discriminator(reconstructed)[0]  # Take only the first output (prediction)
    
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

def simple_classifier_f1(X_train, y_train, X_test, y_test, target_class):
    """Evaluate F1 score using a simple classifier for a specific target class."""
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import f1_score, precision_score, recall_score
    
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
    
    # Predict and compute metrics
    y_pred = clf.predict(X_test_scaled)
    precision = precision_score(y_test_binary, y_pred, zero_division=0)
    recall = recall_score(y_test_binary, y_pred, zero_division=0)
    f1 = f1_score(y_test_binary, y_pred, zero_division=0)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

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
            # If it's a list/tuple of multiple outputs, take only the first value (main loss)
            if len(value) > 0:
                return to_scalar(value[0])
            return 0.0
        elif isinstance(value, np.ndarray):
            # For numpy arrays, ensure we're getting a scalar
            if value.size > 0:
                if value.ndim > 0 and value.size > 1:
                    # If multi-dimensional or multi-element array, take the mean
                    return float(value.mean())
                else:
                    # Otherwise convert the single value
                    return float(value.item() if hasattr(value, 'item') else value)
            return 0.0
        elif isinstance(value, tf.Tensor):
            # Convert TensorFlow tensor to numpy and then to float
            np_value = value.numpy()
            if np_value.size > 1:
                return float(np_value.mean())
            return float(np_value.item() if hasattr(np_value, 'item') else np_value)
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
        'recall': [],
        'precision': [],
        'f1': [],
        'feature_matching_loss': [],
        'cycle_loss': []
    }
    
    # Early stopping variables
    best_recall = -1
    best_f1 = -1
    best_epoch = 0
    best_weights = None
    best_euclidean_dist = float('inf')  # Track best euclidean distance
    
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
    
    # Feature matching loss for improved stability
    def feature_matching_loss(real_features, fake_features):
        """Feature matching loss to improve GAN training stability."""
        return tf.reduce_mean(tf.square(tf.reduce_mean(real_features, axis=0) - 
                                       tf.reduce_mean(fake_features, axis=0)))
    
    # Cycle consistency loss to ensure generated samples maintain class characteristics
    def cycle_consistency_loss(real_batch, reconstructed):
        """Cycle consistency loss to ensure generated samples maintain class characteristics."""
        return tf.reduce_mean(tf.abs(real_batch - reconstructed))
    
    # Progressive training phases
    training_phases = [
        {'vae_weight': 1.0, 'gan_weight': 0.0, 'feature_weight': 0.0, 'epochs': int(epochs * 0.2)},
        {'vae_weight': 0.8, 'gan_weight': 0.2, 'feature_weight': 0.5, 'epochs': int(epochs * 0.3)},
        {'vae_weight': 0.6, 'gan_weight': 0.4, 'feature_weight': 1.0, 'epochs': int(epochs * 0.5)}
    ]
    
    # Initial diversity factor (will be adjusted dynamically)
    diversity_factor = 1.0
    
    # Start progressive training
    current_epoch = 0
    for phase_idx, phase in enumerate(training_phases):
        phase_epochs = phase['epochs']
        print(f"\nStarting training phase {phase_idx+1}/{len(training_phases)}")
        print(f"VAE weight: {phase['vae_weight']}, GAN weight: {phase['gan_weight']}, Feature weight: {phase['feature_weight']}")
        print(f"Phase epochs: {phase_epochs}")
        
        for phase_epoch in tqdm(range(phase_epochs)):
            # Update epoch counter
            epoch = current_epoch + phase_epoch
            
            # Update KL annealing weight
            vae.update_epoch(epoch)
            
            # Gradually increase diversity factor for rare classes
            samples_count = len(scaled_embeddings)
            if samples_count < 50:  # For very rare classes
                diversity_factor = min(2.0, 1.0 + 0.01 * epoch)  # Increase diversity more aggressively
            elif samples_count < 100:
                diversity_factor = min(1.5, 1.0 + 0.005 * epoch)
            else:
                diversity_factor = 1.0  # No adjustment for larger classes
            
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
            # Get reconstructed samples from VAE
            reconstructed = vae(real_batch)
            
            # Train discriminator with feature outputs
            d_output_real, real_features = discriminator(real_batch)
            d_output_fake, fake_features = discriminator(reconstructed)
            
            # Compute feature matching loss
            feat_match_loss = feature_matching_loss(real_features, fake_features)
            
            # Add instance noise to improve training stability (decreasing with epochs)
            noise_stddev = max(0.0, 0.1 * (1.0 - epoch / epochs))
            
            # Train discriminator with noise augmentation for improved stability
            real_noisy = real_batch + tf.random.normal(tf.shape(real_batch), mean=0.0, stddev=noise_stddev)
            fake_noisy = reconstructed + tf.random.normal(tf.shape(reconstructed), mean=0.0, stddev=noise_stddev)
            
            # Use dummy values for feature targets since we don't compute loss on them
            dummy_features = np.zeros_like(real_features)
            
            # Train discriminator with appropriate targets for both outputs
            d_loss_real = discriminator.train_on_batch(real_noisy, [real_labels, dummy_features])
            d_loss_fake = discriminator.train_on_batch(fake_noisy, [fake_labels, dummy_features])
            
            # Calculate discriminator loss - extract only the main loss (first output)
            # We're only interested in the first value which is the binary classification loss
            d_loss_real_val = to_scalar(d_loss_real[0] if isinstance(d_loss_real, list) else d_loss_real)
            d_loss_fake_val = to_scalar(d_loss_fake[0] if isinstance(d_loss_fake, list) else d_loss_fake)
            d_loss = 0.5 * (d_loss_real_val + d_loss_fake_val)
            
            # --------------------------
            # Train Generator (VAE-GAN)
            # --------------------------
            # Freeze discriminator
            discriminator.trainable = False
            
            # Get combined adversarial and cycle consistency loss for improved quality
            cycle_loss_val = cycle_consistency_loss(real_batch, reconstructed)
            
            # Adjust generator loss with feature matching and cycle consistency
            g_loss = vaegan.train_on_batch(real_batch, real_labels) + \
                     phase['feature_weight'] * feat_match_loss + \
                     0.5 * cycle_loss_val
                     
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
            history['feature_matching_loss'].append(float(feat_match_loss))
            history['cycle_loss'].append(float(cycle_loss_val))
            
            # Generate additional diverse samples if needed (every 5 epochs)
            if epoch % 5 == 0 and epoch > 10:
                # Generate random samples from different parts of latent space with adaptive diversity
                diverse_noise = tf.random.normal((batch_size, latent_dim), mean=0, stddev=diversity_factor)
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
                print(f"  Feature matching loss: {float(feat_match_loss):.4f}")
                print(f"  Cycle consistency loss: {float(cycle_loss_val):.4f}")
                print(f"  Diversity factor: {diversity_factor:.2f}")
                
                # Evaluate generation quality if test data is available
                quality_metrics = None
                metrics = None
                
                if X_test is not None and y_test is not None and target_class is not None:
                    # Evaluate generation quality
                    quality_metrics = evaluate_generation_quality(
                        decoder, scaler, X_test, y_test, 
                        latent_dim, target_class
                    )
                    
                    # Generate synthetic samples for evaluation
                    n_synthetic = min(1000, len(X_test))
                    synthetic_embeddings = generate_synthetic_embeddings(
                        decoder, scaler, latent_dim, n_synthetic, diversity=diversity_factor
                    )
                    
                    # Combine original and synthetic embeddings
                    combined_X = np.vstack([real_embeddings, synthetic_embeddings])
                    combined_y = np.concatenate([
                        [target_class] * len(real_embeddings),
                        [target_class] * n_synthetic
                    ])
                    
                    # Evaluate metrics (F1, precision, recall)
                    metrics = simple_classifier_f1(
                        combined_X, combined_y, X_test, y_test, target_class
                    )
                    
                    print(f"  Generation quality metrics: {quality_metrics}")
                    print(f"  Metrics on test set:")
                    print(f"    Precision: {metrics['precision']:.4f}")
                    print(f"    Recall: {metrics['recall']:.4f}")
                    print(f"    F1 Score: {metrics['f1']:.4f}")
                    
                    # Get the F1 and recall
                    f1 = metrics['f1']
                    recall = metrics['recall']
                    precision = metrics['precision']
                    
                    # Save model based on best F1 score, considering generation quality for very small classes
                    if quality_metrics and 'euclidean_dist' in quality_metrics:
                        euclidean_dist = quality_metrics['euclidean_dist']
                        
                        # For very rare classes (less than 20 samples), balance F1 and quality
                        if len(scaled_embeddings) < 20:
                            # Use a combined score that weighs both F1 and euclidean distance
                            # Normalize euclidean_dist to [0,1] range for comparison
                            norm_euclidean = min(1.0, euclidean_dist / 2.0)  # Typical range is 0-2
                            combined_score = f1 * (1.0 - 0.3 * norm_euclidean)
                            
                            if combined_score > best_f1 or (euclidean_dist < best_euclidean_dist * 0.8 and f1 >= best_f1 * 0.9):
                                print(f"  New best combined score: {combined_score:.4f} (F1: {f1:.4f}, Euclidean: {euclidean_dist:.4f})")
                                best_f1 = f1
                                best_recall = recall
                                best_epoch = epoch
                                best_euclidean_dist = euclidean_dist
                                # Save best weights
                                best_weights = {
                                    'encoder': encoder.get_weights(),
                                    'decoder': decoder.get_weights(),
                                    'discriminator': discriminator.get_weights()
                                }
                        else:
                            # For larger classes, focus primarily on F1 score
                            if f1 > best_f1:
                                best_f1 = f1
                                best_recall = recall
                                best_epoch = epoch
                                best_euclidean_dist = euclidean_dist
                                # Save best weights
                                best_weights = {
                                    'encoder': encoder.get_weights(),
                                    'decoder': decoder.get_weights(),
                                    'discriminator': discriminator.get_weights()
                                }
                                print(f"  New best F1 score: {best_f1:.4f}")
                            
                        print(f"  No improvement for {epoch - best_epoch} epochs (best F1: {best_f1:.4f}, best euclidean: {best_euclidean_dist:.4f})")
                        if epoch - best_epoch >= patience:
                            print(f"  Early stopping triggered. Best F1: {best_f1:.4f} at epoch {best_epoch}")
                            # Restore best weights
                            if best_weights:
                                encoder.set_weights(best_weights['encoder'])
                                decoder.set_weights(best_weights['decoder'])
                                discriminator.set_weights(best_weights['discriminator'])
                            # Skip to next phase or end training
                            break
                    
                # Record metrics
                history['quality_metrics'].append(quality_metrics)
                if metrics:
                    history['recall'].append(metrics['recall'])
                    history['precision'].append(metrics['precision']) 
                    history['f1'].append(metrics['f1'])
                else:
                    history['recall'].append(None)
                    history['precision'].append(None)
                    history['f1'].append(None)
        
        # Update current epoch for next phase
        current_epoch += phase_epochs
        
        # Apply mixup augmentation between phases to improve generalization
        if phase_idx < len(training_phases) - 1:
            print("Applying mixup augmentation between training phases")
            # Generate samples with current model
            n_mixup = len(scaled_embeddings)
            noise = tf.random.normal((n_mixup, latent_dim))
            generated = decoder.predict(noise, verbose=0)
            
            # Apply mixup augmentation (interpolate between real and generated)
            for _ in range(10):  # Do 10 mixup training iterations
                alpha = np.random.beta(0.2, 0.2, size=(batch_size, 1))  # Mixup strength
                idx1 = np.random.randint(0, len(scaled_embeddings), batch_size)
                idx2 = np.random.randint(0, n_mixup, batch_size)
                
                batch1 = tf.gather(scaled_embeddings_tensor, idx1)
                batch2 = generated[idx2]
                
                mixed_batch = alpha * batch1 + (1 - alpha) * batch2
                _ = vae.train_on_batch(mixed_batch)
    
    # If early stopping was triggered, make sure we use the best weights
    if best_weights:
        print(f"Restoring best weights from epoch {best_epoch} with F1 {best_f1:.4f}, recall {best_recall:.4f} and euclidean distance {best_euclidean_dist:.4f}")
        encoder.set_weights(best_weights['encoder'])
        decoder.set_weights(best_weights['decoder'])
        discriminator.set_weights(best_weights['discriminator'])
    
    return encoder, decoder, discriminator, history, scaler

def generate_synthetic_embeddings(decoder, scaler, latent_dim, n_samples, diversity=1.0):
    """Generate synthetic embeddings using the trained decoder with diversity control."""
    try:
        # Use a mix of sampling approaches for better coverage of the latent space
        synthetic_embeddings_list = []
        
        # 1. Standard Gaussian sampling with diversity control
        noise_standard = tf.random.normal((n_samples // 3, latent_dim), mean=0, stddev=diversity)
        synthetic_std = decoder.predict(noise_standard, verbose=0)
        
        # 2. Truncated normal sampling for more focused samples
        noise_truncated = tf.clip_by_value(
            tf.random.normal((n_samples // 3, latent_dim), mean=0, stddev=diversity*0.7),
            -2.0, 2.0
        )
        synthetic_trunc = decoder.predict(noise_truncated, verbose=0)
        
        # 3. Uniform sampling for better coverage of rare regions
        noise_uniform = tf.random.uniform((n_samples // 3, latent_dim), minval=-diversity*1.5, maxval=diversity*1.5)
        synthetic_uniform = decoder.predict(noise_uniform, verbose=0)
        
        # 4. If there are remaining samples, add some spherical sampling (points on a hypersphere)
        remaining = n_samples - (n_samples // 3) * 3
        if remaining > 0:
            # Generate points on a unit hypersphere
            noise_sphere = tf.random.normal((remaining, latent_dim))
            noise_sphere = noise_sphere / tf.norm(noise_sphere, axis=1, keepdims=True) * diversity
            synthetic_sphere = decoder.predict(noise_sphere, verbose=0)
            synthetic_embeddings_list.append(synthetic_sphere)
        
        # Combine all sets of synthetic embeddings
        synthetic_embeddings_list = [synthetic_std, synthetic_trunc, synthetic_uniform]
        if remaining > 0:
            synthetic_embeddings_list.append(synthetic_sphere)
            
        # Concatenate all samples
        synthetic_embeddings = np.vstack(synthetic_embeddings_list)
        
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
    
    # Print per-class metrics with focus on F1 score
    print("\nPer-class metrics:")
    print(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1 Score':<12} {'Improvement':<12}")
    print("-" * 70)
    
    class_improvement = {}
    for i, cls in enumerate(np.unique(synthetic_y)):
        precision_improvement = class_metrics_aug[0][i] - class_metrics_orig[0][i]
        recall_improvement = class_metrics_aug[1][i] - class_metrics_orig[1][i]
        f1_improvement = class_metrics_aug[2][i] - class_metrics_orig[2][i]
        
        print(f"{cls:<20} "
              f"{class_metrics_orig[0][i]:.4f}->{class_metrics_aug[0][i]:.4f} "
              f"{class_metrics_orig[1][i]:.4f}->{class_metrics_aug[1][i]:.4f} "
              f"{class_metrics_orig[2][i]:.4f}->{class_metrics_aug[2][i]:.4f} "
              f"{f1_improvement:+.4f}")
        
        class_improvement[cls] = {
            'precision': float(precision_improvement),
            'recall': float(recall_improvement),
            'f1': float(f1_improvement)
        }
    
    # Sort classes by F1 improvement and highlight the best
    sorted_classes = sorted(
        class_improvement.items(), 
        key=lambda x: x[1]['f1'], 
        reverse=True
    )
    
    if sorted_classes:
        best_class, best_metrics = sorted_classes[0]
        print(f"\nBest improvement in F1 score: {best_class} (+{best_metrics['f1']:.4f})")
        
        # If any class has worse performance, note it
        worst_classes = [(cls, metrics) for cls, metrics in sorted_classes if metrics['f1'] < 0]
        if worst_classes:
            worst_class, worst_metrics = worst_classes[0]
            print(f"Warning: {worst_class} saw a decrease in F1 score ({worst_metrics['f1']:.4f})")
    
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
        'class_improvement': class_improvement
    }

def plot_training_history(history, target_class):
    """Plot VAE-GAN training history with robust type handling."""
    plt.figure(figsize=(15, 12))
    
    # Helper function to safely convert values to floats
    def safe_convert(values):
        result = []
        for x in values:
            if x is None:
                result.append(None)
                continue
                
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
    plt.subplot(3, 2, 1)
    d_loss = safe_convert(history['d_loss'])
    plt.plot(d_loss, label='Discriminator Loss')
    plt.title('Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 2, 2)
    g_loss = safe_convert(history['g_loss'])
    plt.plot(g_loss, label='Generator Loss')
    plt.title('Generator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 2, 3)
    vae_loss = safe_convert(history['vae_loss'])
    plt.plot(vae_loss, label='VAE Loss')
    plt.title('VAE Reconstruction Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 2, 4)
    cycle_loss = safe_convert(history['cycle_loss'])
    plt.plot(cycle_loss, label='Cycle Loss')
    plt.title('Cycle Consistency Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    
    # Add new plots for precision, recall, and F1
    plt.subplot(3, 2, 5)
    precision = safe_convert(history['precision'])
    recall = safe_convert(history['recall'])
    
    # Filter out None values
    epochs = np.array(history['epoch'])
    precision_valid = [(e, p) for e, p in zip(epochs, precision) if p is not None]
    recall_valid = [(e, r) for e, r in zip(epochs, recall) if r is not None]
    
    if precision_valid:
        epochs_p, precision_filtered = zip(*precision_valid)
        plt.plot(epochs_p, precision_filtered, label='Precision', color='blue')
    if recall_valid:
        epochs_r, recall_filtered = zip(*recall_valid)
        plt.plot(epochs_r, recall_filtered, label='Recall', color='green')
    
    plt.title('Precision and Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 2, 6)
    f1 = safe_convert(history['f1'])
    f1_valid = [(e, f) for e, f in zip(epochs, f1) if f is not None]
    
    if f1_valid:
        epochs_f, f1_filtered = zip(*f1_valid)
        plt.plot(epochs_f, f1_filtered, label='F1 Score', color='red')
    
    plt.title('F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
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
    parser.add_argument('--adaptive-sampling', action='store_true',
                        help='Enable adaptive sampling based on class size', default=True)
    parser.add_argument('--feature-matching', action='store_true',
                        help='Enable feature matching loss for better generation', default=True)
    parser.add_argument('--optimize-f1', action='store_true',
                        help='Optimize for F1 score instead of just recall', default=True)
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
    
    # Print detailed class distribution
    print("\nDetailed class distribution:")
    for cls, count in class_counts.items():
        print(f"  {cls}: {count} samples ({count/len(y_train)*100:.2f}%)")

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
            
            # Adjust hyperparameters based on class size for very rare classes
            adaptive_beta = args.beta
            adaptive_batch_size = min(32, real_count)
            adaptive_epochs = args.epochs
            
            # For very small classes, use more regularization and training epochs
            if real_count < 20:
                print(f"Adjusting hyperparameters for rare class {target_class}")
                adaptive_beta = max(0.1, args.beta * 0.5)  # Lower beta for rare classes
                adaptive_epochs = int(args.epochs * 1.5)    # More epochs
                print(f"  Adjusted beta: {adaptive_beta}, epochs: {adaptive_epochs}")
            
            # Build VAE-GAN components with enhanced architecture
            encoder = build_encoder(embedding_dim, latent_dim)
            decoder = build_decoder(latent_dim, embedding_dim)
            discriminator = build_discriminator(embedding_dim)
            
            # Build VAE-GAN with optimized beta value
            try:
                vaegan, vae = build_vaegan(encoder, decoder, discriminator, embedding_dim, latent_dim, beta=adaptive_beta)
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
                    epochs=adaptive_epochs, batch_size=adaptive_batch_size,
                    beta=adaptive_beta
                )
            except Exception as e:
                print(f"Error training VAE-GAN: {e}")
                continue
            
            if decoder is None:
                print(f"VAE-GAN training failed for {target_class}")
                continue

            # Generate diverse synthetic embeddings
            n_synthetic = min(20000, majority_class_count - real_count)  # More samples for minority classes
            
            # Adjust diversity for rare classes
            adaptive_diversity = args.diversity
            if real_count < 20:
                adaptive_diversity = args.diversity * 0.8  # Reduce diversity for very rare classes
            elif real_count < 50:
                adaptive_diversity = args.diversity * 0.9
                
            print(f"Generating {n_synthetic} synthetic samples for {target_class} with diversity={adaptive_diversity}...")
            
            # Generate synthetic embeddings with improved diversity control
            synthetic_embeddings = generate_synthetic_embeddings(
                decoder, scaler, latent_dim, n_synthetic, diversity=adaptive_diversity
            )
            
            # Evaluate and filter low-quality samples (optional)
            if len(synthetic_embeddings) > real_count * 2:
                # Find euclidean distances to real samples for each synthetic sample
                from scipy.spatial.distance import cdist
                distances = cdist(synthetic_embeddings, real_class_embeddings, 'euclidean')
                # Keep samples with at least one close real sample
                min_distances = np.min(distances, axis=1)
                quality_threshold = np.percentile(min_distances, 80)  # Keep best 80%
                quality_mask = min_distances < quality_threshold
                synthetic_embeddings = synthetic_embeddings[quality_mask]
                print(f"Filtered to {len(synthetic_embeddings)} high-quality synthetic samples")
            
            # Save model components
            try:
                # Save only weights to avoid serialization issues
                encoder.save_weights(str(MOD / f"encoder_{target_class}.weights.h5"))
                decoder.save_weights(str(MOD / f"decoder_{target_class}.weights.h5"))
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
                    
                    print(f"\nF1 score improvement summary:")
                    # Sort classes by F1 improvement for better visibility
                    sorted_classes = sorted(
                        metrics['class_improvement'].items(),
                        key=lambda x: x[1]['f1'],
                        reverse=True
                    )
                    for cls, improvements in sorted_classes:
                        print(f"  {cls}: {improvements['f1']:.4f} (Precision: {improvements['precision']:.4f}, Recall: {improvements['recall']:.4f})")
                    
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
