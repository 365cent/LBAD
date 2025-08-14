#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Experiment utilities for thesis-scale orchestration.

- Standardize embeddings root handling (e.g., embeddings/logbert/<type>)
- Consistent discovery of available log types from processed TFRecords
- Common helpers for loading/saving paths
"""

from pathlib import Path
from typing import List, Tuple


DEFAULT_PROCESSED_DIR = Path("processed")
DEFAULT_EMBEDDINGS_ROOT = Path("embeddings")


def get_embeddings_dir(embedding_method: str = None, log_type: str = None, embeddings_root: Path = DEFAULT_EMBEDDINGS_ROOT) -> Path:
	"""Return directory where embeddings for a given method/log_type live.
	- If embedding_method is provided, use embeddings/<method>/<log_type>
	- Else fallback to legacy embeddings/<log_type>
	"""
	root = embeddings_root
	if embedding_method:
		root = embeddings_root / embedding_method
	return root / log_type if log_type else root


def get_embedding_filepaths(embedding_method: str, log_type: str, embeddings_root: Path = DEFAULT_EMBEDDINGS_ROOT) -> Tuple[Path, Path]:
	"""Return (log_embeddings.pkl, label.pkl) paths for a method/log_type."""
	dir_path = get_embeddings_dir(embedding_method, log_type, embeddings_root)
	log_path = dir_path / f"log_{log_type}.pkl"
	label_path = dir_path / f"label_{log_type}.pkl"
	return log_path, label_path


def find_available_log_types(processed_dir: Path = DEFAULT_PROCESSED_DIR) -> List[str]:
	"""Find log types that have TFRecord data under processed/.
	Return sorted list of directory names containing any .tfrecord files.
	"""
	if not processed_dir.exists():
		return []
	log_types: List[str] = []
	for sub in processed_dir.iterdir():
		if sub.is_dir() and list(sub.glob("*.tfrecord")):
			log_types.append(sub.name)
	return sorted(log_types)