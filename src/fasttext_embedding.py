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
from collections import Counter
import argparse

# Configuration
OUTPUT_DIR = Path("embeddings")
MODEL_DIR = Path("models")
PROCESSED_DIR = Path("processed")
VECTOR_SIZE = 100
RANDOM_SEED = 42

# Map of log types to potential attack types
LOG_TYPE_ATTACKS = {
    'dns': [
        'traceroute', 
        'dns_scan', 
        'service_scan', 
        'dnsteal-received', 
        'dnsteal-dropped'
    ],
    'network': [ # Covers firewall, vpn, and potentially generic network captures
        'traceroute', 
        'network_scan', 
        'dns_scan', 
        'service_scan',
        'port_scan', # More specific than network_scan
        'ddos', 
        'tcp_syn_flood', 
        'udp_flood', 
        'icmp_flood'
    ],
    'web': [
        'webshell_cmd', 
        'webshell_upload', 
        'dirb', 
        'wordpress_database_dump',
        'wordpress_scan',
        'sql_injection',
        'xss', # Cross-Site Scripting
        'directory_traversal',
        'command_injection' # OS Command Injection via web
    ],
    'error': [ # Errors can sometimes indicate attempted or successful attacks
        'dirb', 
        'wordpress_scan',
        'sql_injection_error', # e.g., SQL syntax error from failed attempt
        'auth_failure_reflected_in_error' # e.g., permission denied errors
    ],
    'monitoring': [ # Covers kernel, systemd - broader system activity
        'password_cracking',
        'malware_execution', # Generic malware activity detected by system monitors
        'suspicious_process'
    ],
    'auth': [
        'login_as_system_user', 
        'reverse_shell',
        'ssh_bruteforce',
        'ftp_bruteforce',
        'password_cracking' # Can also be seen here
    ],
    'audit': [
        'root_command_execution', 
        'login_as_system_user', 
        'dnsteal-received', 
        'dnsteal-dropped',
        'privilege_escalation',
        'unauthorized_file_access'
    ],
    'ids': [ # Attacks specifically flagged by IDS/IPS systems
        'port_scan_alert',
        'web_attack_alert', # Generic, could be SQLi, XSS etc.
        'malware_signature_detected',
        'ddos_alert',
        'bruteforce_alert',
        'policy_violation',
        'exploit_detected'
    ]
}

# Create a sorted list of all unique attack types across all log categories
ALL_UNIQUE_ATTACKS = sorted(list(set(attack for attacks in LOG_TYPE_ATTACKS.values() for attack in attacks)))

# Map of directory names to logical log types
DIR_TO_LOG_TYPE = {
    'dns': 'dns',
    'web': 'web',
    'auth': 'auth',
    'audit': 'audit',
    'firewall': 'network',
    'vpn': 'network',
    'kernel': 'monitoring',
    'systemd': 'monitoring',
    'error': 'error',
    'ids': 'ids',  # Added mapping for the new ids type
    'snort': 'ids', # Example if you have a 'snort' directory
    'suricata': 'ids' # Example if you have a 'suricata' directory
}

def parse_example(example):
    """Parse a TensorFlow Example protocol buffer."""
    feature_description = {
        'l': tf.io.FixedLenFeature([], tf.string),  # log
        'y': tf.io.FixedLenFeature([], tf.string),  # label
    }
    return tf.io.parse_single_example(example, feature_description)

def load_tfrecord_files(directory=PROCESSED_DIR, log_type_filter=None):
    """Load TFRecord files from directory into a DataFrame, optionally filtered by log type."""
    if log_type_filter:
        # Check if this is a physical directory name or a logical log type
        if log_type_filter in DIR_TO_LOG_TYPE.values():
            # It's a logical type, find all matching directories
            matching_dirs = [k for k, v in DIR_TO_LOG_TYPE.items() if v == log_type_filter]
            print(f"Loading TFRecord files for logical log type '{log_type_filter}' from directories: {matching_dirs}")
            
            tfrecord_files = []
            for dir_name in matching_dirs:
                log_type_dir_path = directory / dir_name
                if log_type_dir_path.exists():
                    tfrecord_files.extend(log_type_dir_path.glob("*.tfrecord"))
        else:
            # It's a physical directory name
            log_type_dir_path = directory / log_type_filter
            print(f"Loading TFRecord files from directory '{log_type_filter}' at {log_type_dir_path}")
            if not log_type_dir_path.exists():
                raise FileNotFoundError(f"No directory found for '{log_type_filter}' at {log_type_dir_path}")
            tfrecord_files = list(log_type_dir_path.glob("*.tfrecord"))
    else:
        print(f"Loading all TFRecord files from {directory}...")
        tfrecord_files = []
        # Check each log type directory
        for log_dir_path in directory.iterdir():
            if log_dir_path.is_dir():
                tfrecord_files.extend(log_dir_path.glob("*.tfrecord"))
    
    if not tfrecord_files:
        if log_type_filter:
            raise FileNotFoundError(f"No TFRecord files found for log type '{log_type_filter}'")
        else:
            raise FileNotFoundError(f"No TFRecord files found in {directory}")
    
    all_logs = []
    all_labels_json = []
    all_logical_log_types = []  # Store logical log type for each entry
    
    for file_path in tfrecord_files:
        try:
            # Determine the physical directory name (e.g., 'dns', 'web')
            physical_dir_name = file_path.parent.name
            # Map physical directory name to its logical log type (e.g., 'dns' -> 'dns', 'firewall' -> 'network')
            logical_log_type = DIR_TO_LOG_TYPE.get(physical_dir_name, physical_dir_name) # Fallback to physical if no mapping
            
            print(f"Processing {file_path} (physical dir: {physical_dir_name}, logical type: {logical_log_type})")
            
            # Load dataset with GZIP compression
            dataset = tf.data.TFRecordDataset(str(file_path), compression_type="GZIP")
            
            for raw_record in dataset:
                try:
                    parsed = parse_example(raw_record)
                    log = parsed['l'].numpy().decode('utf-8')
                    label_json = parsed['y'].numpy().decode('utf-8')
                    
                    # Store the raw JSON label and the determined logical log type
                    all_logs.append(log)
                    all_labels_json.append(label_json) # Keep as JSON string for now
                    all_logical_log_types.append(logical_log_type)
                except tf.errors.OpError as e: # Catch TensorFlow specific operational errors
                    print(f"TensorFlow OpError parsing record in {file_path}: {e}. Skipping record.")
                    continue # Skip to the next record
                except Exception as e: # Broader exception for safety during parsing or decoding
                    print(f"Error parsing/processing record in {file_path}: {e}. Skipping record.")
                    continue # Skip to the next record
                
        except Exception as e: # Catch errors related to a whole file (e.g., file not found, permission)
            print(f"Error processing file {file_path}: {e}")
    
    print(f"Loaded {len(all_logs)} log entries")
    return pd.DataFrame({
        'log': all_logs, 
        'label_json': all_labels_json, # Store JSON string
        'log_type': all_logical_log_types # This is the logical log type
    })

def create_attack_binary_vector(label_json_str, log_type_for_attacks, global_attack_list=None):
    """
    Convert labels from JSON string into a binary vector.
    Uses global_attack_list if provided, otherwise uses log_type_for_attacks to find specific attacks.
    """
    try:
        labels = json.loads(label_json_str) # Actual attacks present in this log
        if not isinstance(labels, list): # Ensure labels are a list (e.g. from label file)
            labels = [labels]
    except json.JSONDecodeError:
        print(f"Warning: Could not parse label JSON: {label_json_str}. Assuming no attacks.")
        labels = []

    if global_attack_list:
        # Use the global, sorted list of all unique attacks
        target_attack_types = global_attack_list
    elif log_type_for_attacks in LOG_TYPE_ATTACKS:
        # Use the attacks specific to this logical log type
        target_attack_types = LOG_TYPE_ATTACKS[log_type_for_attacks]
    else:
        # Log type not recognized for specific attacks, and no global list provided.
        # This case should ideally be avoided by ensuring LOG_TYPE_ATTACKS is comprehensive
        # or by always providing global_attack_list when appropriate.
        print(f"Warning: Log type '{log_type_for_attacks}' not in LOG_TYPE_ATTACKS and no global_attack_list provided. Returning empty vector.")
        return np.array([], dtype=int) # Or handle as error, or return zeros of a default max length

    binary_vector = np.zeros(len(target_attack_types), dtype=int)
    for i, attack_name in enumerate(target_attack_types):
        if attack_name in labels:
            binary_vector[i] = 1
    return binary_vector

def preprocess_logs_and_labels(df, use_global_attack_list=False):
    """Tokenize log entries and create binary label vectors."""
    print("Tokenizing log entries and processing labels...")
    
    # Tokenize logs
    df['tokens'] = df['log'].apply(lambda x: simple_preprocess(str(x)))
    
    if use_global_attack_list:
        print(f"Generating binary labels using global attack list: {ALL_UNIQUE_ATTACKS}")
        df['binary_labels'] = df.apply(
            lambda row: create_attack_binary_vector(row['label_json'], row['log_type'], global_attack_list=ALL_UNIQUE_ATTACKS),
            axis=1
        )
    else:
        print("Generating binary labels using log-type specific attack lists.")
        df['binary_labels'] = df.apply(
            lambda row: create_attack_binary_vector(row['label_json'], row['log_type']),
            axis=1
        )
    
    # Ensure all binary_labels are numpy arrays and handle potential empty arrays from create_attack_binary_vector
    # This is important if a log_type was not in LOG_TYPE_ATTACKS and global_attack_list was not used.
    # For XGBoost, all label vectors in a dataset should ideally have the same length.
    # If using global_attack_list, this is naturally handled.
    # If not, and some log types had no defined attacks, their vectors might be empty or different lengths.
    # The current create_attack_binary_vector returns np.array([]) in such cases.
    # We might need to pad these or ensure LOG_TYPE_ATTACKS is exhaustive for per-log-type processing.
    
    # For simplicity, we assume that if not using global_attack_list, each log_type processed will have defined attacks.
    # If global_attack_list is True, all vectors will have length len(ALL_UNIQUE_ATTACKS).

    print("Log type distribution in the current DataFrame:")
    print(df['log_type'].value_counts())
    
    return df

def train_fasttext_model(corpus, vector_size=VECTOR_SIZE, window=5, min_count=1, epochs=10, model_name_suffix=""):
    """Train a FastText model on the corpus and save it."""
    model_name = f"fasttext_model{model_name_suffix}.bin"
    model_path = MODEL_DIR / model_name
    
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Training FastText model '{model_name}' on {len(corpus)} documents...")
    model = FastText(
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=multiprocessing.cpu_count() - 1,
        seed=RANDOM_SEED
    )
    
    model.build_vocab(corpus_iterable=corpus)
    model.train(
        corpus_iterable=corpus,
        total_examples=len(corpus),
        epochs=epochs
    )
    
    model.save(str(model_path))
    print(f"Saved trained model to {model_path}")
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

def visualize_embeddings(embeddings, label_json_list, log_types_for_labels, output_file=None, use_global_attacks_for_display=False):
    """Create t-SNE visualization of embeddings. Labels for display can be simplified."""
    print("Creating t-SNE visualization...")
    
    # For visualization, we simplify labels to the first attack or "normal"
    # or use a summary if using global attacks.
    display_labels = []
    for i, label_json_str in enumerate(label_json_list):
        try:
            actual_labels = json.loads(label_json_str) # list of attacks
            if not actual_labels:
                display_labels.append("normal")
            elif use_global_attacks_for_display:
                 # If using global, just indicate "attack" vs "normal" for simplicity in viz
                display_labels.append("attack" if actual_labels else "normal")
            else:
                # Use first attack type for specific log type visualization
                display_labels.append(actual_labels[0] if actual_labels else "normal")
        except json.JSONDecodeError:
            display_labels.append("unknown_label_format")

    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=RANDOM_SEED, perplexity=min(30, len(embeddings)-1 if len(embeddings)>1 else 1)) # Adjust perplexity
    reduced = tsne.fit_transform(embeddings)
    
    # Create plot with improved visualization
    plt.figure(figsize=(12, 10))
    
    # Create DataFrame with results
    df_plot = pd.DataFrame({
        'x': reduced[:, 0],
        'y': reduced[:, 1],
        'label': display_labels,
        'log_type': log_types_for_labels # This is the logical log type
    })
    
    # Get unique labels and count occurrences
    label_counts = df_plot['label'].value_counts()
    print(f"Visualization label distribution: {dict(label_counts)}")
    
    # Create a color palette for labels
    unique_display_labels = df_plot['label'].unique()
    palette_colors = sns.color_palette("husl", len(unique_display_labels))
    color_map = {label: palette_colors[i] for i, label in enumerate(unique_display_labels)}
    if "normal" in color_map: # Ensure 'normal' is consistently green if present
        color_map["normal"] = "green"
    
    # Create the plot
    sns.scatterplot(
        x='x', 
        y='y', 
        hue='label', 
        style='log_type', # Can show log_type as different shapes
        data=df_plot, 
        palette=color_map,
        alpha=0.7,
        s=50 # marker size
    )
    plt.title('t-SNE Visualization of FastText Log Embeddings')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout(rect=[0,0,0.85,1]) # Adjust layout to make space for legend
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {output_file}")
    else:
        plt.show()
    
    plt.close()  # Properly close figure to free memory

def find_available_logical_log_types():
    """Discover available logical log types by examining the processed directory structure and DIR_TO_LOG_TYPE mapping."""
    available_physical_dirs = set()
    if PROCESSED_DIR.exists():
        for path in PROCESSED_DIR.iterdir():
            if path.is_dir() and list(path.glob("*.tfrecord")):
                available_physical_dirs.add(path.name)
    
    # Map physical dirs to logical types, then get unique logical types
    available_logical_types = set()
    for physical_dir in available_physical_dirs:
        logical_type = DIR_TO_LOG_TYPE.get(physical_dir)
        if logical_type: # Only include if it maps to a known logical type
            available_logical_types.add(logical_type)
        else:
            # If a directory in processed/ doesn't map to a logical type,
            # we might want to treat it as its own type or log a warning.
            # For now, we only process known logical types for per-type models.
            print(f"Warning: Directory '{physical_dir}' in processed folder does not map to a known logical log type. It will be included in 'all_combined' but not processed individually unless it's also a key in LOG_TYPE_ATTACKS.")
            # Option: add it as a distinct logical type if it's also in LOG_TYPE_ATTACKS.
            if physical_dir in LOG_TYPE_ATTACKS:
                 available_logical_types.add(physical_dir)

    # We should process based on keys in LOG_TYPE_ATTACKS as these define our label structures.
    # Or, more robustly, iterate unique values from DIR_TO_LOG_TYPE whose keys are present as physical dirs.
    defined_logical_types = set(LOG_TYPE_ATTACKS.keys())
    
    # The types we can process individually are those that are logical types AND have defined attacks
    # AND correspond to actual data.
    
    # Let's refine: find physical dirs with data, map to logical types, then filter by those in LOG_TYPE_ATTACKS.
    # This ensures we only try to build per-type models for which we have data and attack definitions.
    
    processable_logical_types = set()
    if PROCESSED_DIR.exists():
        for physical_dir_path in PROCESSED_DIR.iterdir():
            if physical_dir_path.is_dir() and list(physical_dir_path.glob("*.tfrecord")):
                physical_dir_name = physical_dir_path.name
                logical_type = DIR_TO_LOG_TYPE.get(physical_dir_name)
                if logical_type and logical_type in LOG_TYPE_ATTACKS:
                    processable_logical_types.add(logical_type)
                elif physical_dir_name in LOG_TYPE_ATTACKS: # If physical dir name itself is a defined log type
                     processable_logical_types.add(physical_dir_name)

    return sorted(list(processable_logical_types))

def print_data_distribution(df_with_binary_labels, processing_context_name, using_global_attacks=False):
    """Print distribution of attack types in the dataset based on 'binary_labels' and 'log_type' columns."""
    print(f"\nData distribution for context: '{processing_context_name}':")

    if using_global_attacks:
        # Distribution based on the global attack list
        binary_labels_matrix = np.array(df_with_binary_labels['binary_labels'].tolist())
        if binary_labels_matrix.ndim == 1: # Handle case of single row or malformed
             if len(binary_labels_matrix) > 0 and isinstance(binary_labels_matrix[0], np.ndarray): # list of arrays
                binary_labels_matrix = np.vstack(binary_labels_matrix)
             else: # single array
                binary_labels_matrix = binary_labels_matrix.reshape(1, -1) if binary_labels_matrix.size > 0 else np.array([[0]*len(ALL_UNIQUE_ATTACKS)])

        if binary_labels_matrix.size == 0 or binary_labels_matrix.shape[1] != len(ALL_UNIQUE_ATTACKS):
            print(f"  Warning: Binary labels for global context seem malformed or empty. Expected {len(ALL_UNIQUE_ATTACKS)} attack types.")
            # Attempt to print based on what's there, but with a warning.
            # This might happen if df was empty or create_attack_binary_vector had issues.
            if binary_labels_matrix.size > 0 :
                print(f"  Actual shape: {binary_labels_matrix.shape}")
            # return # Or try to proceed cautiously

        print(f"  Attack types (global): {ALL_UNIQUE_ATTACKS}")
        if binary_labels_matrix.size > 0:
            attack_counts = np.sum(binary_labels_matrix, axis=0)
            total_logs = len(binary_labels_matrix)
            print("  Attack distribution (global):")
            for i, attack_type in enumerate(ALL_UNIQUE_ATTACKS):
                count = attack_counts[i] if i < len(attack_counts) else 0
                percentage = (count / total_logs) * 100 if total_logs > 0 else 0
                print(f"    {attack_type}: {count} occurrences ({percentage:.2f}%)")
            
            logs_with_any_attack = np.sum(np.any(binary_labels_matrix, axis=1))
            percentage_with_attack = (logs_with_any_attack / total_logs) * 100 if total_logs > 0 else 0
            print(f"  Logs with any attack: {logs_with_any_attack} ({percentage_with_attack:.2f}%)")
            print(f"  Normal logs: {total_logs - logs_with_any_attack} ({100 - percentage_with_attack:.2f}%)")
            print(f"  Total logs: {total_logs}")
        else:
            print("  No data to report for global attack distribution.")

    else: # Per-log-type distribution
        for log_type_name, group in df_with_binary_labels.groupby('log_type'):
            if log_type_name not in LOG_TYPE_ATTACKS:
                print(f"  Skipping distribution for '{log_type_name}': No attack types defined in LOG_TYPE_ATTACKS.")
                continue

            current_log_type_attacks = LOG_TYPE_ATTACKS[log_type_name]
            binary_labels_matrix = np.array(group['binary_labels'].tolist())
            
            if binary_labels_matrix.ndim == 1: # As above, ensure 2D
                 if len(binary_labels_matrix) > 0 and isinstance(binary_labels_matrix[0], np.ndarray):
                    binary_labels_matrix = np.vstack(binary_labels_matrix)
                 else:
                    binary_labels_matrix = binary_labels_matrix.reshape(1, -1) if binary_labels_matrix.size > 0 else np.array([[0]*len(current_log_type_attacks)])

            if binary_labels_matrix.size == 0 or binary_labels_matrix.shape[1] != len(current_log_type_attacks):
                 print(f"  Warning: Binary labels for log type '{log_type_name}' seem malformed. Expected {len(current_log_type_attacks)} features, got {binary_labels_matrix.shape[1] if binary_labels_matrix.size > 0 else 'empty'}.")
                 # continue # Skip this problematic group

            print(f"  Log Type: '{log_type_name}' (Attacks: {current_log_type_attacks})")
            if binary_labels_matrix.size > 0 :
                attack_counts = np.sum(binary_labels_matrix, axis=0)
                total_logs = len(binary_labels_matrix)
                print("    Attack distribution:")
                for i, attack_type in enumerate(current_log_type_attacks):
                    count = attack_counts[i] if i < len(attack_counts) else 0
                    percentage = (count / total_logs) * 100 if total_logs > 0 else 0
                    print(f"      {attack_type}: {count} occurrences ({percentage:.2f}%)")

                logs_with_any_attack = np.sum(np.any(binary_labels_matrix, axis=1))
                percentage_with_attack = (logs_with_any_attack / total_logs) * 100 if total_logs > 0 else 0
                print(f"    Logs with any attack: {logs_with_any_attack} ({percentage_with_attack:.2f}%)")
                print(f"    Normal logs: {total_logs - logs_with_any_attack} ({100 - percentage_with_attack:.2f}%)")
                print(f"    Total logs for '{log_type_name}': {total_logs}")
            else:
                print(f"    No data to report for '{log_type_name}'.")

def main():
    parser = argparse.ArgumentParser(description="Generate FastText embeddings for log data, processing specific or all log types.")
    parser.add_argument(
        "--log-type",
        type=str,
        choices=list(LOG_TYPE_ATTACKS.keys()),  # Use defined logical types that have attack mappings
        default=None,
        help="Process only this specific logical log type. If not specified, processes all available types individually and then a combined model."
    )
    args = parser.parse_args()

    # Ensure output directories exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    types_to_process_individually = []
    run_combined_processing = False

    if args.log_type:
        print(f"Processing based on --log-type: {args.log_type}")
        available_types_with_data = find_available_logical_log_types()
        if args.log_type in available_types_with_data:
            types_to_process_individually = [args.log_type]
        else:
            print(f"Error: Specified --log-type '{args.log_type}' is not in the list of available and processable log types.")
            print(f"Available and processable types are: {available_types_with_data if available_types_with_data else 'None'}")
            # types_to_process_individually remains empty, so the individual loop won't run for this type
        run_combined_processing = False  # Do not run combined if a specific type is specified
    else: # No specific log type, run default behavior (all individuals + combined)
        types_to_process_individually = find_available_logical_log_types()
        if not types_to_process_individually:
            print("No processable logical log types found with corresponding data and attack definitions for individual processing.")
        else:
            print(f"Found {len(types_to_process_individually)} processable logical log types for individual processing: {', '.join(types_to_process_individually)}")
        run_combined_processing = True # Default is to also run combined processing

    # --- Individual Log Type Processing Loop ---
    if types_to_process_individually:
        for logical_log_type_name in types_to_process_individually:
            print(f"\n{'='*50}\nProcessing logical log type: {logical_log_type_name}\n{'='*50}")
            try:
                # Load data specifically for this logical log type
                # load_tfrecord_files uses DIR_TO_LOG_TYPE to find relevant physical dirs
                df_log_type = load_tfrecord_files(log_type_filter=logical_log_type_name)
                
                if df_log_type.empty:
                    print(f"No data loaded for logical log type '{logical_log_type_name}'. Skipping.")
                    continue

                # Preprocess: tokenize and create binary labels specific to this log type's attacks
                df_log_type = preprocess_logs_and_labels(df_log_type, use_global_attack_list=False)

                if df_log_type.empty or 'tokens' not in df_log_type.columns or df_log_type['tokens'].empty:
                    print(f"No tokens generated for '{logical_log_type_name}'. Skipping model training and embedding.")
                    continue
                
                # Check if binary_labels column exists and is not all empty/problematic
                if 'binary_labels' not in df_log_type or not any(bl.size > 0 for bl in df_log_type['binary_labels']):
                    print(f"Warning: No valid binary labels generated for '{logical_log_type_name}'. Check LOG_TYPE_ATTACKS definition for this type.")
                    # Decide if to proceed without labels or skip
                
                model_suffix = f"_{logical_log_type_name.replace('/', '_')}" # Sanitize suffix
                model_path = MODEL_DIR / f"fasttext_model{model_suffix}.bin"

                if model_path.exists():
                    print(f"Loading existing model from {model_path}")
                    model = FastText.load(str(model_path))
                else:
                    model = train_fasttext_model(df_log_type['tokens'].tolist(), model_name_suffix=model_suffix)

                embeddings = generate_embeddings(model, df_log_type['tokens'].tolist())
                
                # Extract binary labels (already created based on log-type specific attacks)
                # Ensure they are in a consistent NumPy array format
                try:
                    binary_labels_list = df_log_type['binary_labels'].tolist()
                    # Filter out any potentially empty arrays if create_attack_binary_vector returned them
                    # and the original log_type was not in LOG_TYPE_ATTACKS (should be caught by find_available_logical_log_types)
                    valid_binary_labels = [bl for bl in binary_labels_list if isinstance(bl, np.ndarray) and bl.ndim > 0]
                    if not valid_binary_labels: # if all were empty or problematic
                         raise ValueError("No valid binary labels to stack.")
                    binary_labels = np.array(valid_binary_labels)
                    if binary_labels.ndim == 1: # If it results in a 1D array (e.g. list of scalars, unlikely here)
                        # This logic might need adjustment based on how create_attack_binary_vector handles errors
                        # For now, assume it's a list of 1D arrays that can be stacked
                        if len(binary_labels) > 0 and isinstance(binary_labels[0], np.ndarray):
                             binary_labels = np.vstack(binary_labels) # Stack list of 1D arrays into 2D
                        else: # if it's truly a flat 1D array (e.g. error case)
                             binary_labels = binary_labels.reshape(len(df_log_type), -1) if binary_labels.size > 0 else np.array([[]])

                except Exception as e:
                    print(f"Error preparing binary labels for saving/distribution print for {logical_log_type_name}: {e}")
                    print(f"Sample of binary_labels column: {df_log_type['binary_labels'].head()}")
                    # Fallback to empty array or skip saving labels if critical
                    binary_labels = np.array([])

                print_data_distribution(df_log_type, f"log_type_{logical_log_type_name}", using_global_attacks=False)

                output_log_type_dir = OUTPUT_DIR / logical_log_type_name.replace('/', '_') # Sanitize path
                output_log_type_dir.mkdir(parents=True, exist_ok=True)

                with open(output_log_type_dir / "embeddings.pkl", 'wb') as f:
                    pickle.dump(embeddings, f)
                if binary_labels.size > 0 : # Only save if labels are valid
                    with open(output_log_type_dir / "labels.pkl", 'wb') as f:
                        pickle.dump(binary_labels, f)
                    print(f"Saved embeddings and binary labels to {output_log_type_dir}")
                else:
                    print(f"Embeddings saved to {output_log_type_dir}. Labels were empty or invalid, not saved.")

                if embeddings.size > 0:
                    sample_size = min(5000, len(embeddings))
                    sample_idx = np.random.choice(len(embeddings), sample_size, replace=False) if len(embeddings) > 0 else []
                    
                    if len(sample_idx) > 0:
                        visualize_embeddings(
                            embeddings[sample_idx],
                            df_log_type['label_json'].iloc[sample_idx].tolist(),
                            df_log_type['log_type'].iloc[sample_idx].tolist(), # Pass logical log_types for styling
                            output_file=output_log_type_dir / "visualization.png",
                            use_global_attacks_for_display=False
                        )
                else:
                    print("No embeddings generated, skipping visualization.")

            except FileNotFoundError as e:
                print(f"Skipping '{logical_log_type_name}': {e}")
            except Exception as e:
                print(f"Error processing log type {logical_log_type_name}: {e}")
                import traceback
                traceback.print_exc()
    else:
        if args.log_type: # Specific type was requested but was not found/valid. Error already printed.
             print(f"No individual processing performed for '{args.log_type}'.")
        # else: No specific type requested, and none found. Initial message about "No processable..." was already printed.

    # --- Combined Log Type Processing ---
    if run_combined_processing:
        print(f"\n{'='*50}\nProcessing all log types combined\n{'='*50}")
        try:
            df_all = load_tfrecord_files() # Load all data
            if df_all.empty:
                print("No data loaded for 'all combined'. Skipping.")
            else:
                # Preprocess: tokenize and create binary labels using the GLOBAL list of all attacks
                df_all = preprocess_logs_and_labels(df_all, use_global_attack_list=True)

                if df_all.empty or 'tokens' not in df_all.columns or df_all['tokens'].empty:
                    print("No tokens generated for 'all combined'. Skipping model training and embedding.")
                else:
                    model_suffix_all = "_all_combined"
                    model_path_all = MODEL_DIR / f"fasttext_model{model_suffix_all}.bin"

                    if model_path_all.exists():
                        print(f"Loading existing combined model from {model_path_all}")
                        model_all = FastText.load(str(model_path_all))
                    else:
                        model_all = train_fasttext_model(df_all['tokens'].tolist(), model_name_suffix=model_suffix_all)

                    embeddings_all = generate_embeddings(model_all, df_all['tokens'].tolist())
                    
                    # Binary labels are already created using the global attack list
                    binary_labels_all_list = df_all['binary_labels'].tolist()
                    # Ensure it's a 2D numpy array of the correct shape
                    if not all(isinstance(bl, np.ndarray) and bl.shape == (len(ALL_UNIQUE_ATTACKS),) for bl in binary_labels_all_list if bl.size >0 ):
                        # This check might be too strict if some rows had zero attacks and create_attack_binary_vector returned empty or different shape
                        # Re-stacking might be safer if shapes are consistent but not necessarily all full
                         print(f"Warning: Some binary labels for 'all_combined' may not conform to expected shape ({len(ALL_UNIQUE_ATTACKS)},). Attempting to stack.")
                    
                    # Filter out potentially problematic (e.g. not np.ndarray) entries before stacking
                    valid_labels_for_stacking = [bl for bl in binary_labels_all_list if isinstance(bl, np.ndarray) and bl.ndim > 0]

                    if not valid_labels_for_stacking:
                        print("Error: No valid binary labels to stack for 'all_combined'. Labels will be empty.")
                        binary_labels_all = np.array([[]]*len(df_all)) # Or handle as error
                    else:
                        try:
                            binary_labels_all = np.array(valid_labels_for_stacking)
                            # Ensure it's 2D, even if only one row
                            if binary_labels_all.ndim == 1 and isinstance(binary_labels_all[0], np.ndarray): # list of arrays that became object array
                                binary_labels_all = np.vstack(binary_labels_all)
                            elif binary_labels_all.ndim == 1 and binary_labels_all.size > 0 : # flat array
                                 binary_labels_all = binary_labels_all.reshape(len(df_all), -1)

                            # Final check for consistent shape if not empty
                            if binary_labels_all.size > 0 and binary_labels_all.shape[1] != len(ALL_UNIQUE_ATTACKS):
                                 print(f"CRITICAL WARNING: 'all_combined' labels matrix shape {binary_labels_all.shape} does not match num global attacks {len(ALL_UNIQUE_ATTACKS)}. XGBoost will likely fail.")
                                 # This indicates a fundamental issue in label generation for the combined set.
                        except Exception as e_stack:
                             print(f"Error stacking binary labels for 'all_combined': {e_stack}. Labels will be empty.")
                             binary_labels_all = np.array([[]]*len(df_all))

                    print_data_distribution(df_all, "all_combined", using_global_attacks=True)

                    OUTPUT_DIR.mkdir(parents=True, exist_ok=True) # Ensure main output dir exists
                    with open(OUTPUT_DIR / "embeddings_all_combined.pkl", 'wb') as f:
                        pickle.dump(embeddings_all, f)
                    
                    if binary_labels_all.size > 0 and binary_labels_all.shape[1] == len(ALL_UNIQUE_ATTACKS):
                        with open(OUTPUT_DIR / "labels_all_combined.pkl", 'wb') as f:
                            pickle.dump(binary_labels_all, f)
                        print(f"Saved combined embeddings and global binary labels to {OUTPUT_DIR}")
                    else:
                         print(f"Combined embeddings saved to {OUTPUT_DIR}. Global labels were not saved due to errors or empty.")

                    if embeddings_all.size > 0:
                        sample_size = min(5000, len(embeddings_all))
                        sample_idx = np.random.choice(len(embeddings_all), sample_size, replace=False) if len(embeddings_all) > 0 else []

                        if len(sample_idx) > 0:
                            visualize_embeddings(
                                embeddings_all[sample_idx],
                                df_all['label_json'].iloc[sample_idx].tolist(),
                                df_all['log_type'].iloc[sample_idx].tolist(), # Pass logical log_types for styling
                                output_file=OUTPUT_DIR / "visualization_all_combined.png",
                                use_global_attacks_for_display=True # Use simplified "attack" vs "normal" for combined viz
                            )
                    else:
                        print("No 'all_combined' embeddings generated, skipping visualization.")
        
        except FileNotFoundError as e:
            print(f"Skipping 'all combined': {e}")
        except Exception as e:
            print(f"Error processing all log types combined: {e}")
            import traceback
            traceback.print_exc()
    elif args.log_type: # Combined processing was skipped because a specific log type was requested
        print(f"\nSkipping 'all log types combined' processing because --log-type '{args.log_type}' was specified.")

    print("\nFastText embedding processing complete!")

if __name__ == "__main__":
    main()
