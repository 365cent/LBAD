#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Minimal LogBERT-style embedding generation.

This script mirrors the original LogBERT workflow by tokenizing each log line
with the public ``bert-base-uncased`` model and collecting the 768-dimensional
CLS embedding.  Outputs follow the exact same structure used by the
FastText/Word2Vec pipelines so downstream consumers can treat all embeddings
interchangeably:

    embeddings/logbert/<log_type>/
        ├── log_<log_type>.pkl      # float32 numpy array (N, 768)
        ├── label_<log_type>.pkl    # attack vectors + metadata
        └── attack_types_<log_type>.txt

No additional feature engineering or dimensionality expansion is performed –
each sample is represented solely by the CLS token from the BERT encoder.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from itertools import repeat
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from halo import Halo

try:
    from transformers import BertModel, BertTokenizer
    HAS_TRANSFORMERS = True
except ImportError:  # pragma: no cover - environment fallback
    HAS_TRANSFORMERS = False


_BASE_DIR = Path(__file__).resolve().parent.parent
try:
    if "MPLCONFIGDIR" not in os.environ:
        mpl_dir = _BASE_DIR / ".mplconfig"
        mpl_dir.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(mpl_dir)
except Exception:  # pragma: no cover - best effort only
    pass

HF_CACHE_DIR = _BASE_DIR / "hf_cache"
try:
    HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    resolved = str(HF_CACHE_DIR.resolve())
    os.environ.setdefault("HF_HOME", resolved)
    os.environ.setdefault("TRANSFORMERS_CACHE", resolved)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", resolved)
except Exception:  # pragma: no cover - best effort only
    pass


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTPUT_ROOT = Path("embeddings") / "logbert"
PROCESSED_DIR = Path("processed")
VECTOR_SIZE = 768
MAX_SEQ_LENGTH = 128
DEFAULT_BATCH_SIZE = 32


# ---------------------------------------------------------------------------
# TFRecord helpers (shared with other embedding scripts)
# ---------------------------------------------------------------------------


def parse_example(example: tf.Tensor) -> Mapping[str, tf.Tensor]:
    feature_description = {
        "l": tf.io.FixedLenFeature([], tf.string),
        "y": tf.io.FixedLenFeature([], tf.string),
    }
    return tf.io.parse_single_example(example, feature_description)


def load_tfrecord_files(
    directory: Path = PROCESSED_DIR,
    log_type_filter: Optional[str] = None,
) -> pd.DataFrame:
    """Load TFRecord files and return a DataFrame with logs and labels."""

    if log_type_filter:
        target_dir = directory / log_type_filter
        if not target_dir.exists():
            raise FileNotFoundError(f"No directory found for '{log_type_filter}'")
        tfrecord_files = list(target_dir.glob("*.tfrecord"))
        if not tfrecord_files:
            raise FileNotFoundError(f"No TFRecord files found for log type '{log_type_filter}'")
    else:
        tfrecord_files = [
            tfrecord
            for subdir in directory.iterdir()
            if subdir.is_dir()
            for tfrecord in subdir.glob("*.tfrecord")
        ]
        if not tfrecord_files:
            raise FileNotFoundError(f"No TFRecord files found in {directory}")

    spinner = Halo(text="Loading TFRecord files", spinner="dots")
    spinner.start()

    dataset_options = tf.data.Options()
    dataset_options.experimental_deterministic = False
    autotune = getattr(tf.data, "AUTOTUNE", tf.data.experimental.AUTOTUNE)

    all_logs: List[str] = []
    all_labels: List[str] = []
    all_types: List[str] = []

    for idx, file_path in enumerate(tfrecord_files, start=1):
        spinner.text = f"Reading {idx}/{len(tfrecord_files)}: {file_path.name}"
        dataset = tf.data.TFRecordDataset(
            str(file_path),
            compression_type="GZIP",
            num_parallel_reads=autotune,
        )
        dataset = dataset.with_options(dataset_options)
        dataset = dataset.map(parse_example, num_parallel_calls=autotune)
        dataset = dataset.batch(1024).prefetch(autotune)

        for batch in dataset:
            logs = [log.decode("utf-8") for log in batch["l"].numpy()]
            labels = [label.decode("utf-8") for label in batch["y"].numpy()]
            all_logs.extend(logs)
            all_labels.extend(labels)
            all_types.extend(repeat(file_path.parent.name, len(logs)))

    spinner.succeed(f"Loaded {len(all_logs)} log entries")

    return pd.DataFrame({
        "log": all_logs,
        "label_json": all_labels,
        "log_type": all_types,
    })


# ---------------------------------------------------------------------------
# Label helpers (identical philosophy to FastText/Word2Vec scripts)
# ---------------------------------------------------------------------------


def normalize_label(label: Optional[str]) -> Optional[str]:
    if not label:
        return label
    return label.replace("-", "_").lower().strip()


def get_labels_from_json(label_json_str: str) -> Set[str]:
    try:
        labels = json.loads(label_json_str)
        if not isinstance(labels, list):
            labels = [labels]
        return {normalize_label(lbl) for lbl in labels if lbl}
    except json.JSONDecodeError:
        return set()


def collect_unique_labels_from_data(df: pd.DataFrame) -> List[str]:
    uniques: Set[str] = set()
    for label_json in df["label_json"]:
        uniques.update(get_labels_from_json(label_json))
    uniques.discard("")
    return sorted(uniques)


def create_binary_label_vector(
    label_json_str: str,
    *,
    attack_types: Sequence[str],
    attack_lookup: Optional[Dict[str, int]] = None,
) -> np.ndarray:
    if not attack_types:
        return np.zeros(0, dtype=np.int8)

    lookup = attack_lookup if attack_lookup is not None else {
        attack: idx for idx, attack in enumerate(attack_types)
    }

    vector = np.zeros(len(attack_types), dtype=np.int8)
    for label in get_labels_from_json(label_json_str):
        idx = lookup.get(label)
        if idx is not None:
            vector[idx] = 1
    return vector


# ---------------------------------------------------------------------------
# Embedding generation using BERT CLS vectors
# ---------------------------------------------------------------------------


def process_embeddings(
    df: pd.DataFrame,
    tokenizer: BertTokenizer,
    model: BertModel,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Return CLS embeddings and binary attack vectors for the provided logs."""

    attack_types = collect_unique_labels_from_data(df)
    attack_lookup = {attack: idx for idx, attack in enumerate(attack_types)}

    num_samples = len(df)
    embeddings = np.zeros((num_samples, VECTOR_SIZE), dtype=np.float32)
    label_vectors = (
        np.zeros((num_samples, len(attack_types)), dtype=np.int8)
        if attack_types else np.zeros((num_samples, 0), dtype=np.int8)
    )

    device = next(model.parameters()).device
    model.eval()

    spinner = Halo(text="Encoding logs with BERT", spinner="dots")
    spinner.start()

    for start in range(0, num_samples, batch_size):
        end = min(start + batch_size, num_samples)
        texts = df["log"].iloc[start:end].tolist()
        encodings = tokenizer(
            texts,
            max_length=MAX_SEQ_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        encodings = {key: value.to(device) for key, value in encodings.items()}

        with torch.no_grad():
            outputs = model(**encodings)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]

        embeddings[start:end] = cls_embeddings.cpu().numpy().astype(np.float32)

        if attack_types:
            for offset, label_json in enumerate(df["label_json"].iloc[start:end]):
                label_vectors[start + offset] = create_binary_label_vector(
                    label_json,
                    attack_types=attack_types,
                    attack_lookup=attack_lookup,
                )

        spinner.text = f"Encoded {end}/{num_samples} logs"

    spinner.succeed("BERT encoding complete")
    return embeddings, label_vectors, attack_types


# ---------------------------------------------------------------------------
# Saving utilities (identical structure to other embedding scripts)
# ---------------------------------------------------------------------------


def save_embeddings_and_labels(
    *,
    log_embeddings: np.ndarray,
    label_vectors: np.ndarray,
    attack_types: Sequence[str],
    output_dir: Path,
    log_type: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    log_filename = f"log_{log_type}.pkl"
    label_filename = f"label_{log_type}.pkl"
    attack_info_filename = f"attack_types_{log_type}.txt"

    with open(output_dir / log_filename, "wb") as handle:
        pickle.dump(log_embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)

    label_payload = {
        "vectors": label_vectors,
        "classes": list(attack_types),
        "description": "Binary multi-label vectors where [0 1 0] means only the second class is present",
    }
    with open(output_dir / label_filename, "wb") as handle:
        pickle.dump(label_payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(output_dir / attack_info_filename, "w", encoding="utf-8") as handle:
        handle.write(f"Attack Types and Column Mapping for {log_type}\n")
        handle.write("=" * 60 + "\n\n")
        handle.write("Binary Vector Format:\n")
        handle.write("- Each row is a binary vector representing one log entry\n")
        handle.write("- [0 1 0] means only the second attack type is present\n")
        handle.write("- Multiple attacks can be present simultaneously\n\n")
        handle.write(f"Total attack types: {len(attack_types)}\n")
        handle.write(f"Vector dimension: {label_vectors.shape[1]}\n\n")

        if attack_types:
            handle.write("Column Mapping (Index -> Attack Type):\n")
            handle.write("-" * 40 + "\n")
            for idx, attack in enumerate(attack_types):
                handle.write(f"Column {idx:2d}: {attack}\n")
        else:
            handle.write("No attack types detected for this log type.\n")

    print(f"\nSaved files for {log_type}:")
    print(f"  - {log_filename}: Log embeddings {log_embeddings.shape}")
    print(f"  - {label_filename}: Binary label vectors {label_vectors.shape}")
    print(f"  - {attack_info_filename}: Attack types and column mapping")


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def find_available_log_types(directory: Path = PROCESSED_DIR) -> List[str]:
    if not directory.exists():
        return []
    return sorted([item.name for item in directory.iterdir() if item.is_dir()])


def display_data_distribution(df: pd.DataFrame, *, log_type: str) -> None:
    total = len(df)
    print(f"\n{'=' * 20} Data Distribution for '{log_type}' {'=' * 20}")
    print(f"Total log entries: {total}")

    attack_counts: Dict[str, int] = {}
    normal = 0

    for label_json in df["label_json"]:
        labels = get_labels_from_json(label_json)
        if labels:
            for label in labels:
                attack_counts[label] = attack_counts.get(label, 0) + 1
        else:
            normal += 1

    attack = total - normal
    attack_pct = (attack / total * 100.0) if total else 0.0
    normal_pct = (normal / total * 100.0) if total else 0.0

    print(f"Attack samples: {attack} ({attack_pct:.2f}%)")
    print(f"Normal samples: {normal} ({normal_pct:.2f}%)")

    if attack_counts:
        print("\nAttack label counts:")
        for label, count in sorted(attack_counts.items(), key=lambda kv: kv[1], reverse=True):
            pct = (count / total * 100.0) if total else 0.0
            print(f"  {label}: {count} ({pct:.2f}%)")
    print("=" * 64)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate BERT CLS embeddings for log data (LogBERT-style)"
    )
    parser.add_argument("--log-type", type=str, default=None, help="Only process this log type")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Embedding batch size")
    parser.add_argument("--output-subdir", type=str, default=None, help="Optional subdirectory under embeddings/")
    args = parser.parse_args()

    if not HAS_TRANSFORMERS:
        print("❌ transformers library not available. Please install it:")
        print("   pip install transformers torch")
        return

    output_dir = OUTPUT_ROOT
    if args.output_subdir:
        output_dir = Path("embeddings") / args.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    model.to(device)

    available_types = find_available_log_types()
    if not available_types:
        print(f"No TFRecord data found in {PROCESSED_DIR}")
        return

    if args.log_type:
        if args.log_type not in available_types:
            print(f"Log type '{args.log_type}' not found. Available: {', '.join(available_types)}")
            return
        types_to_process = [args.log_type]
    else:
        types_to_process = available_types

    for log_type in types_to_process:
        print(f"\n{'=' * 50}\nProcessing log type: {log_type}\n{'=' * 50}")
        try:
            df = load_tfrecord_files(log_type_filter=log_type)
            if df.empty:
                print(f"No data found for log type '{log_type}'. Skipping.")
                continue

            display_data_distribution(df, log_type=log_type)

            embeddings, label_vectors, attack_types = process_embeddings(
                df,
                tokenizer,
                model,
                batch_size=args.batch_size,
            )

            save_embeddings_and_labels(
                log_embeddings=embeddings,
                label_vectors=label_vectors,
                attack_types=attack_types,
                output_dir=output_dir / log_type,
                log_type=log_type,
            )

        except Exception as exc:  # pragma: no cover - defensive guard rail
            print(f"Error processing log type '{log_type}': {exc}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
