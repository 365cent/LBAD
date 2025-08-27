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
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from halo import Halo
import pickle

# Configuration
OUTPUT_DIR = Path("embeddings") / "word2vec"
PROCESSED_DIR = Path("processed")
VECTOR_SIZE = 200
WINDOW = 5
MIN_COUNT = 1
WORKERS = 4
EPOCHS = 10


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


def main():
	parser = argparse.ArgumentParser(description="Generate Word2Vec embeddings per log type (thesis)")
	parser.add_argument("--log-type", type=str, default=None, help="Process only this specific log type")
	parser.add_argument("--output-subdir", type=str, default=None, help="Override default subdir under embeddings/")
	args = parser.parse_args()
	
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
	
	# Train a single corpus Word2Vec model on all logs for consistency
	spinner = Halo(text='Loading all logs for global Word2Vec training', spinner='dots')
	spinner.start()
	all_df = load_tfrecord_files()
	global_tokens = tokenize_logs(all_df['log'])
	spinner.succeed(f"Loaded {len(all_df)} entries; training Word2Vec")
	model = train_word2vec(global_tokens)
	
	# Process per log type
	for lt in log_types:
		print(f"\n{'='*50}\nProcessing log type: {lt}\n{'='*50}")
		# Pre-check: skip if expected outputs already exist and appear valid
		out_dir = OUTPUT_DIR / lt
		log_pkl = out_dir / f"log_{lt}.pkl"
		label_pkl = out_dir / f"label_{lt}.pkl"
		attack_txt = out_dir / f"attack_types_{lt}.txt"
		if log_pkl.exists() and log_pkl.stat().st_size > 0 and \
		   label_pkl.exists() and label_pkl.stat().st_size > 0 and \
		   attack_txt.exists() and attack_txt.stat().st_size > 0:
			print(f"Outputs already exist for '{lt}', skipping.")
			continue
		df = load_tfrecord_files(log_type_filter=lt)
		# Tokenize
		df['tokens'] = tokenize_logs(df['log'])
		# Embeddings
		df['log_embedding'] = list(embed_tokens(model, df['tokens']))
		# Labels
		attack_types = collect_unique_labels(df)
		df['binary_labels'] = [create_binary_label_vector(s, attack_types) for s in df['label_json']]
		df.attrs['attack_types'] = attack_types
		# Save
		save_embeddings_and_labels(df, out_dir, lt)
		print(f"Saved to {out_dir}")

if __name__ == "__main__":
	main()