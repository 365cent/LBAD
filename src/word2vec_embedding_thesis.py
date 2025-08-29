#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Word2Vec Embeddings for Log Analysis (Thesis-aligned)
- Reads processed TFRecords (per-log-type)
- Trains a Word2Vec model on logs tokens
- Produces per-log-type outputs mirroring FastText/LogBERT format:
  embeddings/word2vec/<log_type>/log_<log_type>.pkl
  embeddings/word2vec/<log_type>/label_<log_type>.pkl
  embeddings/word2vec/<log_type>/attack_types_<log_type>.txt
"""

import argparse
import json
import os
import time
import signal
import sys
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure Matplotlib can write cache/config on HPC
try:
    if "MPLCONFIGDIR" not in os.environ:
        _mpl_dir = Path.cwd() / ".mplconfig"
        _mpl_dir.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(_mpl_dir)
except Exception:
    pass

import tensorflow as tf
try:
    from gensim.models import Word2Vec
    from gensim.utils import simple_preprocess
    HAS_GENSIM = True
except ImportError:
    HAS_GENSIM = False
    print("Warning: gensim library not available, Word2Vec embeddings disabled")
from halo import Halo
import pickle

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns

# Configuration
OUTPUT_DIR = Path("embeddings") / "word2vec"
PROCESSED_DIR = Path("processed")
CHECKPOINT_DIR = Path("checkpoints") / "word2vec"
VECTOR_SIZE = 200
WINDOW = 5
MIN_COUNT = 1
WORKERS = 4
EPOCHS = 10

# Global variables for emergency checkpoint saving
_current_checkpoint_state = None
_cleanup_functions = []

def signal_handler(signum, frame):
    """Handle SIGTERM/SIGINT by saving emergency checkpoint."""
    print(f"\n⚠️  Received signal {signum} - saving emergency checkpoint...")
    
    if _current_checkpoint_state:
        try:
            log_type = _current_checkpoint_state.get('log_type')
            stage = _current_checkpoint_state.get('stage', 'unknown')
            data_hash = _current_checkpoint_state.get('data_hash')
            progress_data = _current_checkpoint_state.get('progress_data', {})
            
            print(f"💾 Saving emergency checkpoint for {log_type} at stage: {stage}")
            save_checkpoint(log_type, f"emergency_{stage}", progress_data, data_hash)
            print(f"✅ Emergency checkpoint saved")
            
        except Exception as e:
            print(f"❌ Error in emergency checkpoint handler: {e}")
    
    # Run cleanup functions
    for cleanup_func in _cleanup_functions:
        try:
            cleanup_func()
        except:
            pass
    
    print("🔄 Emergency checkpoint complete. Exiting...")
    sys.exit(1)

# Register signal handlers for graceful shutdown
signal.signal(signal.SIGTERM, signal_handler)  # SLURM termination
signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C

def register_cleanup_function(func):
    """Register a function to be called on emergency exit."""
    global _cleanup_functions
    _cleanup_functions.append(func)


def parse_tfrecord(example: tf.Tensor):
	feature_description = {
		"l": tf.io.FixedLenFeature([], tf.string),
		"y": tf.io.FixedLenFeature([], tf.string),
	}
	return tf.io.parse_single_example(example, feature_description)


def load_tfrecord_files(directory=PROCESSED_DIR, log_type_filter=None) -> pd.DataFrame:
	if log_type_filter:
		log_type_dir_path = directory / log_type_filter
		if not log_type_dir_path.exists():
			raise FileNotFoundError(f"No directory found for '{log_type_filter}'")
		files = list(log_type_dir_path.glob("*.tfrecord"))
		if not files:
			raise FileNotFoundError(f"No TFRecord files found for log type '{log_type_filter}'")
	else:
		files = []
		for log_dir_path in directory.iterdir():
			if log_dir_path.is_dir():
				files.extend(log_dir_path.glob("*.tfrecord"))
		if not files:
			raise FileNotFoundError(f"No TFRecord files found in {directory}")

	all_logs, all_labels_json, all_log_types = [], [], []
	spinner = Halo(text='Loading TFRecords', spinner='dots')
	spinner.start()
	for file_path in files:
		log_type = file_path.parent.name
		dataset = tf.data.TFRecordDataset(str(file_path), compression_type="GZIP")
		dataset = dataset.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
		dataset = dataset.prefetch(tf.data.AUTOTUNE)
		for parsed in dataset:
			all_logs.append(parsed["l"].numpy().decode("utf-8"))
			all_labels_json.append(parsed["y"].numpy().decode("utf-8"))
			all_log_types.append(log_type)
	spinner.succeed(f"Loaded {len(all_logs)} entries")
	return pd.DataFrame({"log": all_logs, "label_json": all_labels_json, "log_type": all_log_types})


def normalize_label(label: str) -> str:
	if not label:
		return label
	return label.replace('-', '_').lower().strip()


def get_labels_from_json(label_json_str):
	try:
		labels = json.loads(label_json_str)
		if not isinstance(labels, list):
			labels = [labels]
		return {normalize_label(l) for l in labels if l}
	except json.JSONDecodeError:
		return set()


def collect_unique_labels(df: pd.DataFrame):
	labels = set()
	for s in df['label_json']:
		labels.update(get_labels_from_json(s))
	labels.discard('')
	labels.discard(None)
	return sorted(list(labels))


def create_binary_label_vector(label_json_str, all_attack_types):
	labels = get_labels_from_json(label_json_str)
	vec = np.zeros(len(all_attack_types), dtype=np.int8)
	if labels:
		idxs = [i for i, a in enumerate(all_attack_types) if a in labels]
		vec[idxs] = 1
	return vec


def tokenize_logs(logs):
	return [simple_preprocess(text) for text in logs]


def train_word2vec(corpus_tokens):
	model = Word2Vec(
		corpus_tokens,
		vector_size=VECTOR_SIZE,
		window=WINDOW,
		min_count=MIN_COUNT,
		workers=WORKERS,
		epochs=EPOCHS,
		sg=1,
	)
	return model


def embed_tokens(model: Word2Vec, tokens_list):
	embeddings = []
	for tokens in tokens_list:
		vecs = [model.wv[t] for t in tokens if t in model.wv]
		if vecs:
			emb = np.mean(np.vstack(vecs), axis=0).astype(np.float32)
		else:
			emb = np.zeros(model.vector_size, dtype=np.float32)
		embeddings.append(emb)
	return np.array(embeddings, dtype=np.float32)


def generate_data_hash(df):
	"""Generate a hash of the dataset for checkpoint validation."""
	# Create a hash based on log content and labels
	content = f"{len(df)}_{df['log'].iloc[0] if len(df) > 0 else ''}_{df['log'].iloc[-1] if len(df) > 0 else ''}"
	data_hash = hashlib.md5(content.encode()).hexdigest()[:16]
	print(f"🔍 Generated data hash: {data_hash} (based on {len(df)} entries)")
	return data_hash

def save_checkpoint(log_type: str, stage: str, data: dict, data_hash: str):
	"""Save checkpoint for resumeable processing."""
	CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
	checkpoint_file = CHECKPOINT_DIR / f"{log_type}_{stage}_{data_hash}.pkl"
	
	checkpoint_data = {
		'log_type': log_type,
		'stage': stage,
		'data_hash': data_hash,
		'timestamp': time.time(),
		'data': data
	}
	
	with open(checkpoint_file, 'wb') as f:
		pickle.dump(checkpoint_data, f, protocol=pickle.HIGHEST_PROTOCOL)
	
	print(f"💾 Checkpoint saved: {checkpoint_file.name}")
	return checkpoint_file

def load_checkpoint(log_type: str, stage: str, data_hash: str):
	"""Load checkpoint if it exists and matches the data hash."""
	if not CHECKPOINT_DIR.exists():
		return None
	
	checkpoint_file = CHECKPOINT_DIR / f"{log_type}_{stage}_{data_hash}.pkl"
	
	if checkpoint_file.exists():
		try:
			with open(checkpoint_file, 'rb') as f:
				checkpoint_data = pickle.load(f)
			
			# Validate checkpoint
			if (checkpoint_data['log_type'] == log_type and 
				checkpoint_data['stage'] == stage and 
				checkpoint_data['data_hash'] == data_hash):
				
				age_hours = (time.time() - checkpoint_data['timestamp']) / 3600
				print(f"📂 Found checkpoint: {checkpoint_file.name} (age: {age_hours:.1f}h)")
				return checkpoint_data['data']
		except Exception as e:
			print(f"⚠️  Checkpoint loading failed: {e}")
			# Remove corrupted checkpoint
			checkpoint_file.unlink(missing_ok=True)
	
	return None

def cleanup_old_checkpoints(log_type: str, keep_latest: int = 3):
	"""Clean up old checkpoints, keeping only the latest ones."""
	if not CHECKPOINT_DIR.exists():
		return
	
	# Find all checkpoints for this log type
	pattern = f"{log_type}_*.pkl"
	checkpoints = list(CHECKPOINT_DIR.glob(pattern))
	
	if len(checkpoints) > keep_latest:
		# Sort by modification time, keep latest
		checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
		
		for old_checkpoint in checkpoints[keep_latest:]:
			old_checkpoint.unlink(missing_ok=True)
			print(f"🗑️  Cleaned up old checkpoint: {old_checkpoint.name}")

def save_incremental_checkpoint(log_type: str, stage: str, data_hash: str, progress_data: dict):
	"""Save incremental checkpoint with progress information."""
	CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
	progress_pct = progress_data.get('progress_pct', 0)
	checkpoint_file = CHECKPOINT_DIR / f"{log_type}_incremental_{progress_pct}pct_{stage}_{data_hash}.pkl"
	
	checkpoint_data = {
		'log_type': log_type,
		'stage': f"incremental_{stage}",
		'data_hash': data_hash,
		'timestamp': time.time(),
		'progress_pct': progress_pct,
		'data': progress_data
	}
	
	try:
		with open(checkpoint_file, 'wb') as f:
			pickle.dump(checkpoint_data, f, protocol=pickle.HIGHEST_PROTOCOL)
		
		file_size_mb = checkpoint_file.stat().st_size / (1024 * 1024)
		print(f"💾 Incremental checkpoint saved: {progress_pct}% complete ({file_size_mb:.1f}MB)")
		return checkpoint_file
	except Exception as e:
		print(f"❌ Failed to save incremental checkpoint: {e}")
		return None

def load_latest_checkpoint(log_type: str, stage: str, data_hash: str):
	"""Load the latest checkpoint for a given stage, with tolerance for different hashes."""
	if not CHECKPOINT_DIR.exists():
		return None
	
	# Try exact match first
	exact_checkpoint = load_checkpoint(log_type, stage, data_hash)
	if exact_checkpoint:
		return exact_checkpoint
	
	# Look for incremental checkpoints with any hash
	pattern = f"{log_type}_incremental_*pct_{stage}_*.pkl"
	incremental_checkpoints = list(CHECKPOINT_DIR.glob(pattern))
	
	if incremental_checkpoints:
		# Sort by progress percentage (highest first)
		def extract_progress(path):
			try:
				parts = path.stem.split('_')
				for part in parts:
					if part.endswith('pct'):
						return int(part[:-3])
			except:
				return 0
			return 0
		
		incremental_checkpoints.sort(key=extract_progress, reverse=True)
		
		# Try loading the highest progress checkpoint
		for checkpoint_file in incremental_checkpoints:
			try:
				with open(checkpoint_file, 'rb') as f:
					checkpoint_data = pickle.load(f)
				
				if checkpoint_data.get('log_type') == log_type:
					progress_pct = checkpoint_data.get('progress_pct', 0)
					age_hours = (time.time() - checkpoint_data.get('timestamp', time.time())) / 3600
					
					if checkpoint_data.get('data_hash') != data_hash:
						print(f"⚠️  Loading checkpoint with different data hash (tolerance mode)")
						print(f"   Checkpoint hash: {checkpoint_data.get('data_hash', 'unknown')}")
						print(f"   Current hash: {data_hash}")
					
					print(f"✅ Loaded incremental checkpoint: {progress_pct}% complete (age: {age_hours:.1f}h)")
					return checkpoint_data['data']
				
			except Exception as e:
				print(f"⚠️  Failed to load {checkpoint_file.name}: {e}")
				continue
	
	return None


def save_embeddings_and_labels(df: pd.DataFrame, out_dir: Path, log_type: str):
	out_dir.mkdir(parents=True, exist_ok=True)
	# Save embeddings
	X = np.vstack(df['log_embedding'].tolist()).astype(np.float32)
	with open(out_dir / f"log_{log_type}.pkl", 'wb') as f:
		pickle.dump(X, f, protocol=pickle.HIGHEST_PROTOCOL)
	# Save labels
	vectors = np.vstack(df['binary_labels'].tolist()).astype(np.int8) if 'binary_labels' in df else None
	if vectors is not None and vectors.size > 0:
		classes = df.attrs.get('attack_types', [])
		label_data = {
			'vectors': vectors,
			'classes': classes,
			'description': 'Binary multi-label vectors where [0 1 0] means only the second class is present'
		}
		with open(out_dir / f"label_{log_type}.pkl", 'wb') as f:
			pickle.dump(label_data, f, protocol=pickle.HIGHEST_PROTOCOL)
		# Human-readable mapping
		with open(out_dir / f"attack_types_{log_type}.txt", 'w', encoding='utf-8') as f:
			f.write(f"Attack Types for {log_type}\n")
			for i, c in enumerate(classes):
				f.write(f"{i},{c}\n")


def visualize_embeddings(df: pd.DataFrame, out_path: Path):
	"""Create a balanced t-SNE visualization similar to FastText/LogBERT."""
	MAX_TOTAL_POINTS = 30000
	MAX_POINTS_PER_CLASS = 1500

	# Build labels for viz
	viz_labels = []
	for s in df['label_json']:
		labels = get_labels_from_json(s)
		viz_labels.append("normal" if not labels else ", ".join(sorted(labels)))

	# Balanced sampling
	label_to_idx = {}
	for i, lbl in enumerate(viz_labels):
		label_to_idx.setdefault(lbl, []).append(i)

	selected = []
	for lbl, idxs in label_to_idx.items():
		if len(idxs) > MAX_POINTS_PER_CLASS:
			selected.extend(np.random.choice(idxs, MAX_POINTS_PER_CLASS, replace=False))
		else:
			selected.extend(idxs)

	if len(selected) > MAX_TOTAL_POINTS:
		selected = list(np.random.choice(selected, MAX_TOTAL_POINTS, replace=False))

	X = np.vstack(df['log_embedding'].iloc[selected]).astype(np.float32)
	labels = [viz_labels[i] for i in selected]

	perplexity = min(50, max(5, len(X)//1000))
	tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=500, learning_rate='auto', init='pca', method='barnes_hut', random_state=42)
	Y = tsne.fit_transform(X)

	# Plot
	plt.figure(figsize=(16, 10))
	df_plot = pd.DataFrame({'x': Y[:,0], 'y': Y[:,1], 'label': labels})
	uniq = sorted(df_plot['label'].unique())
	palette = sns.color_palette("husl", len(uniq))
	color_map = {lbl: palette[i] for i, lbl in enumerate(uniq)}
	if "normal" in color_map:
		color_map["normal"] = "green"
	for lbl in uniq:
		mask = df_plot['label'] == lbl
		plt.scatter(df_plot.loc[mask, 'x'], df_plot.loc[mask, 'y'], c=[color_map[lbl]], s=12, alpha=0.6, edgecolors='none', label=lbl)
	plt.title(f't-SNE: Word2Vec Log Embeddings ({len(uniq)} classes)')
	plt.xlabel('t-SNE-1'); plt.ylabel('t-SNE-2')
	if len(uniq) <= 20:
		plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
		plt.tight_layout(rect=[0,0,0.85,1])
	else:
		plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=6, ncol=2)
		plt.tight_layout(rect=[0,0,0.8,1])
	plt.savefig(out_path, dpi=150, bbox_inches='tight')
	plt.close()


def check_existing_outputs(out_dir: Path, log_type: str) -> dict:
	"""Check which outputs already exist and are valid."""
	status = {
		'log_pkl': False,
		'label_pkl': False,
		'attack_txt': False,
		'viz_png': False,
		'model_pkl': False,
		'progress_json': False
	}
	
	log_pkl = out_dir / f"log_{log_type}.pkl"
	label_pkl = out_dir / f"label_{log_type}.pkl"
	attack_txt = out_dir / f"attack_types_{log_type}.txt"
	viz_png = out_dir / "visualization.png"
	model_pkl = out_dir / "word2vec_model.pkl"
	progress_json = out_dir / "progress.json"
	
	# Check file existence and size
	if log_pkl.exists() and log_pkl.stat().st_size > 0:
		status['log_pkl'] = True
	if label_pkl.exists() and label_pkl.stat().st_size > 0:
		status['label_pkl'] = True
	if attack_txt.exists() and attack_txt.stat().st_size > 0:
		status['attack_txt'] = True
	if viz_png.exists() and viz_png.stat().st_size > 0:
		status['viz_png'] = True
	if model_pkl.exists() and model_pkl.stat().st_size > 0:
		status['model_pkl'] = True
	if progress_json.exists():
		status['progress_json'] = True
	
	return status

def save_progress(out_dir: Path, log_type: str, stage: str, details: dict = None):
	"""Save progress checkpoint."""
	progress = {
		'log_type': log_type,
		'stage': stage,
		'timestamp': time.time(),
		'time_str': time.strftime('%Y-%m-%d %H:%M:%S'),
		'details': details or {}
	}
	
	progress_file = out_dir / "progress.json"
	with open(progress_file, 'w') as f:
		json.dump(progress, f, indent=2)

def load_progress(out_dir: Path) -> dict:
	"""Load progress checkpoint if it exists."""
	progress_file = out_dir / "progress.json"
	if progress_file.exists():
		try:
			with open(progress_file, 'r') as f:
				return json.load(f)
		except Exception:
			return {}
	return {}

def save_word2vec_model(model: Word2Vec, out_dir: Path):
	"""Save the Word2Vec model for reuse."""
	model_file = out_dir / "word2vec_model.pkl"
	with open(model_file, 'wb') as f:
		pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_word2vec_model(out_dir: Path):
	"""Load an existing Word2Vec model."""
	model_file = out_dir / "word2vec_model.pkl"
	if model_file.exists():
		try:
			with open(model_file, 'rb') as f:
				return pickle.load(f)
		except Exception:
			return None
	return None

def main():
	parser = argparse.ArgumentParser(description="Generate Word2Vec embeddings per log type (thesis) - Resumeable with Checkpoints")
	parser.add_argument("--log-type", type=str, default=None, help="Process only this specific log type")
	parser.add_argument("--output-subdir", type=str, default=None, help="Override default subdir under embeddings/")
	parser.add_argument("--force-restart", action="store_true", help="Force restart from beginning, ignoring existing outputs and checkpoints")
	parser.add_argument("--clean-checkpoints", action="store_true", help="Clean up all checkpoints before starting")
	parser.add_argument("--clean-incremental", action="store_true", help="Clean up incremental checkpoints only")
	parser.add_argument("--list-checkpoints", action="store_true", help="List available checkpoints and exit")
	args = parser.parse_args()
	
	# Check if gensim library is available
	if not HAS_GENSIM:
		print("❌ gensim library not available. Please install it:")
		print("   pip install gensim")
		return
	
	print("🔄 Auto-Resume System: Saves checkpoints every stage, auto-recovers from crashes")
	
	# Handle checkpoint management options
	if args.list_checkpoints:
		print("\n📂 Available Checkpoints:")
		print("="*50)
		if CHECKPOINT_DIR.exists():
			checkpoints = list(CHECKPOINT_DIR.glob("*.pkl"))
			if checkpoints:
				for cp in sorted(checkpoints, key=lambda x: x.stat().st_mtime, reverse=True):
					age_hours = (time.time() - cp.stat().st_mtime) / 3600
					size_mb = cp.stat().st_size / (1024 * 1024)
					print(f"  {cp.name} (age: {age_hours:.1f}h, size: {size_mb:.1f}MB)")
			else:
				print("  No checkpoints found")
		else:
			print("  Checkpoint directory doesn't exist")
		return
	
	if args.clean_checkpoints:
		import shutil
		if CHECKPOINT_DIR.exists():
			shutil.rmtree(CHECKPOINT_DIR)
			print("🗑️  Cleaned up all checkpoints")
	elif args.clean_incremental:
		if CHECKPOINT_DIR.exists():
			incremental_files = list(CHECKPOINT_DIR.glob("*_incremental_*pct_*.pkl"))
			for f in incremental_files:
				f.unlink(missing_ok=True)
			print(f"🗑️  Cleaned up {len(incremental_files)} incremental checkpoints")
	
	global OUTPUT_DIR
	if args.output_subdir:
		OUTPUT_DIR = Path("embeddings") / args.output_subdir
	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
	
	# Determine types
	if args.log_type:
		log_types = [args.log_type]
	else:
		log_types = sorted([p.name for p in PROCESSED_DIR.iterdir() if p.is_dir() and list(p.glob('*.tfrecord'))])
	if not log_types:
		print("No log types found under processed/")
		return
	
	print(f"🔤 Word2Vec Embedding Generation (Thesis-aligned)")
	print(f"📁 Output directory: {OUTPUT_DIR}")
	print(f"📊 Log types to process: {log_types}")
	if args.force_restart:
		print("🔄 Force restart enabled - will overwrite existing outputs")
	
	# Global Word2Vec model training (with reuse capability)
	global_model = None
	global_model_path = OUTPUT_DIR / "global_word2vec_model.pkl"
	
	if not args.force_restart and global_model_path.exists():
		print("📦 Loading existing global Word2Vec model...")
		try:
			with open(global_model_path, 'rb') as f:
				global_model = pickle.load(f)
			print(f"✅ Loaded existing global Word2Vec model from {global_model_path}")
		except Exception as e:
			print(f"⚠️  Failed to load existing model: {e}")
			global_model = None
	
	if global_model is None:
		print("🤖 Training new global Word2Vec model...")
		spinner = Halo(text='Loading all logs for global Word2Vec training', spinner='dots')
		spinner.start()
		all_df = load_tfrecord_files()
		global_tokens = tokenize_logs(all_df['log'])
		spinner.succeed(f"Loaded {len(all_df)} entries; training Word2Vec")
		
		train_start = time.time()
		global_model = train_word2vec(global_tokens)
		train_time = time.time() - train_start
		
		# Save global model for reuse
		with open(global_model_path, 'wb') as f:
			pickle.dump(global_model, f, protocol=pickle.HIGHEST_PROTOCOL)
		print(f"💾 Saved global Word2Vec model to {global_model_path}")
		print(f"⏱️  Training time: {train_time:.2f} seconds")
	
	# Process per log type with intelligent resume
	completed_count = 0
	for i, lt in enumerate(log_types, 1):
		print(f"\n{'='*60}")
		print(f"Processing log type {i}/{len(log_types)}: {lt}")
		print(f"{'='*60}")
		
		out_dir = OUTPUT_DIR / lt
		out_dir.mkdir(parents=True, exist_ok=True)
		
		# Check existing outputs
		if not args.force_restart:
			status = check_existing_outputs(out_dir, lt)
			progress = load_progress(out_dir)
			
			# Skip if everything is complete
			if all(status.values()):
				print(f"✅ All outputs already exist for '{lt}', skipping")
				print(f"   📁 {out_dir}")
				if progress:
					print(f"   📅 Last completed: {progress.get('time_str', 'unknown')}")
				completed_count += 1
				continue
			
			# Show what exists
			if any(status.values()):
				print(f"📋 Existing outputs for '{lt}':")
				for output, exists in status.items():
					status_icon = "✅" if exists else "❌"
					print(f"   {status_icon} {output}")
				if progress:
					print(f"   📅 Last stage: {progress.get('stage', 'unknown')} at {progress.get('time_str', 'unknown')}")
		
		# Start processing
		process_start = time.time()
		save_progress(out_dir, lt, "started", {"start_time": process_start})
		
		try:
			# Generate data hash for checkpoint validation
			initial_df = load_tfrecord_files(log_type_filter=lt)
			data_hash = generate_data_hash(initial_df)
			
			# Set up emergency checkpoint state
			global _current_checkpoint_state
			_current_checkpoint_state = {
				'log_type': lt,
				'data_hash': data_hash,
				'stage': 'data_loading',
				'progress_data': {}
			}
			
			# Check for existing checkpoints
			if not args.force_restart:
				# Try to resume from latest checkpoint
				resume_data = load_latest_checkpoint(lt, "processing", data_hash)
				if resume_data:
					print(f"🔄 Resuming from checkpoint...")
					
					# Load data state from checkpoint
					df = resume_data.get('df')
					if df is not None:
						attack_types = resume_data.get('attack_types', [])
						checkpoint_stage = resume_data.get('current_stage', 'data_loaded')
						
						print(f"   📂 Resumed at stage: {checkpoint_stage}")
						print(f"   📊 Data: {len(df)} entries, {len(attack_types)} attack types")
						
						# Skip to appropriate stage based on checkpoint
						if checkpoint_stage == 'completed':
							print(f"✅ Found completed checkpoint for {lt}, skipping")
							completed_count += 1
							continue
						elif checkpoint_stage in ['data_saved', 'labels_processed']:
							# Data already processed, just need visualization
							print(f"📊 Creating t-SNE visualization...")
							viz_start = time.time()
							visualize_embeddings(df, out_dir / "visualization.png")
							viz_time = time.time() - viz_start
							print(f"   ✅ Visualization saved in {viz_time:.2f}s")
							
							# Save local model copy
							save_word2vec_model(global_model, out_dir)
							
							# Update final progress
							total_time = time.time() - process_start
							final_details = {"total_time": total_time, "resumed_from_checkpoint": True}
							save_progress(out_dir, lt, "completed", final_details)
							completed_count += 1
							continue
			
			# Load data (use pre-loaded if from checkpoint, else load fresh)
			if 'df' not in locals() or df is None:
				print(f"📥 Loading TFRecord data for {lt}...")
				load_start = time.time()
				df = initial_df
				load_time = time.time() - load_start
				print(f"   ✅ Loaded {len(df)} entries in {load_time:.2f}s")
				save_progress(out_dir, lt, "data_loaded", {"samples": len(df), "load_time": load_time})
				
				# Save checkpoint after data loading
				checkpoint_data = {
					'df': df,
					'current_stage': 'data_loaded',
					'load_time': load_time
				}
				save_checkpoint(lt, "processing", checkpoint_data, data_hash)
				_current_checkpoint_state.update({
					'stage': 'data_loaded', 
					'progress_data': checkpoint_data
				})
			
			# Tokenize
			if 'tokens' not in df.columns:
				print(f"🔤 Tokenizing logs...")
				token_start = time.time()
				df['tokens'] = tokenize_logs(df['log'])
				token_time = time.time() - token_start
				print(f"   ✅ Tokenization completed in {token_time:.2f}s")
				save_progress(out_dir, lt, "tokenized", {"tokenization_time": token_time})
				
				# Save checkpoint after tokenization
				checkpoint_data = {
					'df': df,
					'current_stage': 'tokenized',
					'token_time': token_time
				}
				save_checkpoint(lt, "processing", checkpoint_data, data_hash)
				_current_checkpoint_state.update({
					'stage': 'tokenized',
					'progress_data': checkpoint_data
				})
			
			# Generate embeddings
			if 'log_embedding' not in df.columns:
				print(f"🎯 Generating embeddings...")
				embed_start = time.time()
				
				# Process embeddings in batches with incremental checkpoints
				embeddings_list = []
				batch_size = 1000  # Process in batches for large datasets
				num_batches = (len(df) + batch_size - 1) // batch_size
				
				for batch_idx in range(num_batches):
					start_idx = batch_idx * batch_size
					end_idx = min(start_idx + batch_size, len(df))
					batch_tokens = df['tokens'].iloc[start_idx:end_idx].tolist()
					
					batch_embeddings = embed_tokens(global_model, batch_tokens)
					embeddings_list.extend(batch_embeddings)
					
					# Save incremental checkpoint every 10% or 5 batches
					progress_pct = int((end_idx / len(df)) * 100)
					if batch_idx % max(1, num_batches // 10) == 0 and batch_idx > 0:
						incremental_data = {
							'df': df.iloc[:end_idx].copy(),
							'embeddings_processed': end_idx,
							'current_stage': 'embedding_generation',
							'progress_pct': progress_pct,
							'batch_idx': batch_idx
						}
						incremental_data['df']['log_embedding'] = embeddings_list
						save_incremental_checkpoint(lt, "embedding", data_hash, incremental_data)
					
					# Update emergency checkpoint state
					_current_checkpoint_state.update({
						'stage': 'embedding_generation',
						'progress_data': {
							'embeddings_processed': end_idx,
							'total_samples': len(df),
							'progress_pct': progress_pct
						}
					})
				
				df['log_embedding'] = embeddings_list
				embed_time = time.time() - embed_start
				print(f"   ✅ Embeddings generated in {embed_time:.2f}s")
				save_progress(out_dir, lt, "embeddings_generated", {"embedding_time": embed_time})
				
				# Save checkpoint after embedding generation
				checkpoint_data = {
					'df': df,
					'current_stage': 'embeddings_generated',
					'embed_time': embed_time
				}
				save_checkpoint(lt, "processing", checkpoint_data, data_hash)
				_current_checkpoint_state.update({
					'stage': 'embeddings_generated',
					'progress_data': checkpoint_data
				})
			
			# Process labels
			if 'binary_labels' not in df.columns:
				print(f"🏷️  Processing labels...")
				label_start = time.time()
				attack_types = collect_unique_labels(df)
				df['binary_labels'] = [create_binary_label_vector(s, attack_types) for s in df['label_json']]
				df.attrs['attack_types'] = attack_types
				label_time = time.time() - label_start
				print(f"   ✅ Processed {len(attack_types)} attack types in {label_time:.2f}s")
				print(f"   📊 Attack types: {', '.join(attack_types) if len(attack_types) <= 5 else f'{len(attack_types)} types'}")
				save_progress(out_dir, lt, "labels_processed", {"attack_types": attack_types, "label_time": label_time})
				
				# Save checkpoint after label processing
				checkpoint_data = {
					'df': df,
					'attack_types': attack_types,
					'current_stage': 'labels_processed',
					'label_time': label_time
				}
				save_checkpoint(lt, "processing", checkpoint_data, data_hash)
				_current_checkpoint_state.update({
					'stage': 'labels_processed',
					'progress_data': checkpoint_data
				})
			else:
				attack_types = df.attrs.get('attack_types', [])
			
			# Save embeddings and labels
			print(f"💾 Saving embeddings and labels...")
			save_start = time.time()
			save_embeddings_and_labels(df, out_dir, lt)
			save_time = time.time() - save_start
			print(f"   ✅ Saved in {save_time:.2f}s")
			save_progress(out_dir, lt, "data_saved", {"save_time": save_time})
			
			# Save checkpoint after data saving
			checkpoint_data = {
				'df': df,
				'attack_types': attack_types,
				'current_stage': 'data_saved',
				'save_time': save_time
			}
			save_checkpoint(lt, "processing", checkpoint_data, data_hash)
			_current_checkpoint_state.update({
				'stage': 'data_saved',
				'progress_data': checkpoint_data
			})
			
			# Create visualization
			print(f"📊 Creating t-SNE visualization...")
			viz_start = time.time()
			visualize_embeddings(df, out_dir / "visualization.png")
			viz_time = time.time() - viz_start
			print(f"   ✅ Visualization saved in {viz_time:.2f}s")
			save_progress(out_dir, lt, "visualization_created", {"viz_time": viz_time})
			
			# Save local model copy
			save_word2vec_model(global_model, out_dir)
			
			# Final progress update
			total_time = time.time() - process_start
			final_details = {
				"total_time": total_time,
				"samples_processed": len(df),
				"attack_types_count": len(attack_types),
				"embedding_dimension": VECTOR_SIZE,
				"timing_breakdown": {
					"data_loading": locals().get('load_time', 0),
					"tokenization": locals().get('token_time', 0),
					"embedding_generation": locals().get('embed_time', 0),
					"label_processing": locals().get('label_time', 0),
					"data_saving": save_time,
					"visualization": viz_time
				}
			}
			save_progress(out_dir, lt, "completed", final_details)
			
			# Save final checkpoint
			final_checkpoint_data = {
				'current_stage': 'completed',
				'final_details': final_details
			}
			save_checkpoint(lt, "processing", final_checkpoint_data, data_hash)
			
			# Clean up old checkpoints after successful completion
			cleanup_old_checkpoints(lt)
			
			print(f"✅ Completed processing {lt}")
			print(f"   📁 Output directory: {out_dir}")
			print(f"   ⏱️  Total time: {total_time:.2f}s")
			print(f"   📊 Processed {len(df):,} samples → {VECTOR_SIZE}D embeddings")
			completed_count += 1
			
			# Clear emergency checkpoint state
			_current_checkpoint_state = None
			
		except Exception as e:
			error_details = {"error": str(e), "error_time": time.time()}
			save_progress(out_dir, lt, "error", error_details)
			print(f"❌ Error processing {lt}: {e}")
			import traceback
			traceback.print_exc()
			continue
	
	# Final summary
	print(f"\n{'='*60}")
	print(f"🎉 Word2Vec Embedding Generation Complete")
	print(f"{'='*60}")
	print(f"✅ Successfully processed: {completed_count}/{len(log_types)} log types")
	print(f"📁 Output directory: {OUTPUT_DIR}")
	print(f"📦 Global model saved: {global_model_path}")
	
	if completed_count < len(log_types):
		failed = len(log_types) - completed_count
		print(f"⚠️  {failed} log types had errors - check individual progress.json files")
	
	print(f"\n💾 Checkpoint Features:")
	print(f"   - Auto-resume: Continues from last checkpoint on interruption")
	print(f"   - Stage checkpoints: Data loading, tokenization, embeddings, labels")
	print(f"   - Incremental checkpoints: Every 10% during embedding generation")
	print(f"   - Emergency recovery: SIGTERM/SIGINT signal handling")
	print(f"   - Checkpoint directory: {CHECKPOINT_DIR}")
	
	print(f"\n🔍 Next steps:")
	print(f"   - Check embeddings: ls -la {OUTPUT_DIR}/*/")
	print(f"   - Run models: make run-method LOG_TYPE=<type> METHOD=word2vec")
	print(f"   - View visualizations: {OUTPUT_DIR}/*/visualization.png")
	print(f"   - List checkpoints: python {__file__} --list-checkpoints")
	print(f"   - Clean checkpoints: python {__file__} --clean-checkpoints")

if __name__ == "__main__":
	main()