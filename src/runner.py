#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Thesis Experiment Runner

- Generate embeddings: LogBERT (2314D), FastText (300D), Word2Vec (200D)
- Train/evaluate unsupervised transformer on each embedding family
- Train/evaluate supervised baselines (RF/LR/KNN/XGB) on each embedding family
- Optional binary fallback with SMOTE if multi-label fails to reach targets

Usage examples:
  python src/runner.py --log-type wp-error --all
  python src/runner.py --log-type all --embeddings logbert fasttext word2vec --unsupervised --supervised
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List

import pickle
from experiment_utils import get_embeddings_dir, find_available_log_types


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC = PROJECT_ROOT / "src"
EMBEDDINGS = PROJECT_ROOT / "embeddings"
RESULTS = PROJECT_ROOT / "results"


def run(cmd: List[str]):
	print(f"$ {' '.join(cmd)}")
	return subprocess.run(cmd, check=True)


def generate_embeddings(embedding: str, log_type: str = None):
	if embedding == "logbert":
		cmd = [sys.executable, str(SRC / "logbert_embeddings.py"), "--output-subdir", "logbert"]
		if log_type:
			cmd += ["--log-type", log_type]
		run(cmd)
	elif embedding == "fasttext":
		cmd = [sys.executable, str(SRC / "fasttext_embedding.py"), "--output-subdir", "fasttext"]
		if log_type:
			cmd += ["--log-type", log_type]
		run(cmd)
	elif embedding == "word2vec":
		cmd = [sys.executable, str(SRC / "word2vec_embedding_thesis.py"), "--output-subdir", "word2vec"]
		if log_type:
			cmd += ["--log-type", log_type]
		run(cmd)
	else:
		raise ValueError(f"Unknown embedding: {embedding}")


def run_unsupervised_transformer(embedding: str, log_type: str):
	# Reuse transformer.py, which reads embeddings from embeddings/<log_type>/ by default
	# We symlink or copy paths to legacy layout if needed.
	source_dir = get_embeddings_dir(embedding, log_type)
	legacy_dir = EMBEDDINGS / log_type
	legacy_dir.mkdir(parents=True, exist_ok=True)
	# Always sync files so each embedding family is evaluated correctly
	for name in [f"log_{log_type}.pkl", f"label_{log_type}.pkl"]:
		src = source_dir / name
		dst = legacy_dir / name
		if src.exists():
			import shutil
			try:
				shutil.copy2(src, dst)
			except Exception:
				shutil.copyfile(src, dst)
	cmd = [sys.executable, str(SRC / "transformer.py"), "--log-type", log_type, "--use-enhanced-features"]
	run(cmd)


def run_supervised_baselines(embedding: str, log_type: str):
	# xgboost_ml expects embeddings/<log_type> legacy layout
	source_dir = get_embeddings_dir(embedding, log_type)
	legacy_dir = EMBEDDINGS / log_type
	legacy_dir.mkdir(parents=True, exist_ok=True)
	# Copy thesis files to legacy names for compatibility
	import shutil
	for name in [f"log_{log_type}.pkl", f"label_{log_type}.pkl"]:
		src = source_dir / name
		dst = legacy_dir / name
		if src.exists():
			# Always overwrite so each embedding family is evaluated fairly
			try:
				shutil.copy2(src, dst)
			except Exception:
				shutil.copyfile(src, dst)
	# Create xgboost_compat files if needed
	xgb_X = legacy_dir / "embeddings.pkl"
	xgb_y = legacy_dir / "labels.pkl"
	if not xgb_X.exists() and (legacy_dir / f"log_{log_type}.pkl").exists():
		shutil.copy2(legacy_dir / f"log_{log_type}.pkl", xgb_X)
	if not xgb_y.exists() and (legacy_dir / f"label_{log_type}.pkl").exists():
		# xgboost_ml expects raw arrays, but our label file is a dict; write a raw matrix copy
		try:
			with open(legacy_dir / f"label_{log_type}.pkl", 'rb') as f:
				label_data = pickle.load(f)
			import pickle as _pkl
			with open(xgb_y, 'wb') as g:
				if isinstance(label_data, dict) and 'vectors' in label_data:
					_pkl.dump(label_data['vectors'], g)
				else:
					_pkl.dump(label_data, g)
		except Exception:
			pass
	# Also write key.txt mapping for readability
	key_file = legacy_dir / "key.txt"
	if not key_file.exists() and (legacy_dir / f"label_{log_type}.pkl").exists():
		try:
			with open(legacy_dir / f"label_{log_type}.pkl", 'rb') as f:
				label_data = pickle.load(f)
			classes = label_data.get('classes', []) if isinstance(label_data, dict) else []
			with open(key_file, 'w') as kf:
				for i, c in enumerate(classes):
					kf.write(f"{i},{c}\n")
		except Exception:
			pass
	# Run ml_models.py on this log_type
	cmd = [sys.executable, str(SRC / "ml_models.py"), "--log-type", log_type, "--model", "all"]
	run(cmd)
	# Run xgboost specialized
	cmd = [sys.executable, str(SRC / "xgboost_ml.py"), "--log-type", log_type]
	try:
		run(cmd)
	except Exception:
		pass


def main():
	parser = argparse.ArgumentParser(description="Run thesis experiments end-to-end")
	parser.add_argument("--log-type", type=str, default=None, help="Specific log type to process (default: all)")
	parser.add_argument("--embeddings", nargs="*", default=["logbert", "fasttext", "word2vec"], help="Embeddings to generate/use")
	parser.add_argument("--unsupervised", action="store_true", help="Run unsupervised transformer")
	parser.add_argument("--supervised", action="store_true", help="Run supervised baselines")
	parser.add_argument("--all", action="store_true", help="Run embeddings + unsupervised + supervised")
	args = parser.parse_args()
	
	if args.all:
		args.unsupervised = True
		args.supervised = True
	
	# Determine log types from processed
	if args.log_type and args.log_type != "all":
		log_types = [args.log_type]
	else:
		log_types = find_available_log_types()
		if not log_types:
			print("No log types discovered under processed/")
			return
	
	# Generate embeddings
	for emb in args.embeddings:
		for lt in log_types:
			generate_embeddings(emb, lt)
	
	# Run models
	for emb in args.embeddings:
		for lt in log_types:
			if args.unsupervised:
				run_unsupervised_transformer(emb, lt)
			if args.supervised:
				run_supervised_baselines(emb, lt)
				# Optional: binary fallback baseline
				try:
					run([sys.executable, str(SRC / "supervised_binary.py"), "--log-type", lt, "--pos-ratio", "0.5"]) 
				except Exception:
					pass
	
	# TODO: gather and aggregate summaries across runs
	print("All requested runs completed. Check results/ for outputs.")

if __name__ == "__main__":
	main()
