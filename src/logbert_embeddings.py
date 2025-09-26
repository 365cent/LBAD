#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LogBERT Embedding for Log Analysis - Pre-trained BERT CLS token features

This implementation mirrors the optimized FastText/Word2Vec pipelines so all
embeddings share the exact same output format and downstream contracts.

Highlights
----------
- Uses a pre-trained BERT model to extract CLS, mean, max, and attention
  features (2314D vectors) in batches with GPU support when available
- Dynamic-programming style caching for both tokenizer outputs and label
  vector construction to avoid redundant work on repeated log lines
- Streamlined data loading that reuses the high-throughput TFRecord reader
  used by the FastText pipeline for consistent performance
- Saves `log_*.pkl`, `label_*.pkl`, and `attack_types_*.txt` files with the
  same structure expected by `transformer.py` and other consumers
- Provides optional t-SNE visualization with balanced sampling identical to
  the FastText workflow

The generated 2314-dimensional embedding is composed of:
    * 768D CLS token
    * 768D mean pooling
    * 768D max pooling
    * 10D attention-top-k summary
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from dataclasses import dataclass
from functools import lru_cache
from itertools import repeat
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Optional imports with graceful degradation
# ---------------------------------------------------------------------------

try:
    from transformers import BertModel, BertTokenizer
    HAS_TRANSFORMERS = True
except ImportError:  # pragma: no cover - handled at runtime
    HAS_TRANSFORMERS = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError as exc:  # pragma: no cover - environment specific
    raise RuntimeError("matplotlib and seaborn are required for visualization") from exc

from halo import Halo

# ---------------------------------------------------------------------------
# Environment configuration for cache directories (mirrors FastText pipeline)
# ---------------------------------------------------------------------------

try:
    if "MPLCONFIGDIR" not in os.environ:
        mpl_dir = Path.cwd() / ".mplconfig"
        mpl_dir.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(mpl_dir)
except Exception:  # pragma: no cover - best effort only
    pass

HF_CACHE_DIR = Path("hf_cache")
try:
    HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    resolved = str(HF_CACHE_DIR.resolve())
    os.environ.setdefault("HF_HOME", resolved)
    os.environ.setdefault("TRANSFORMERS_CACHE", resolved)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", resolved)
except Exception:  # pragma: no cover - best effort only
    pass

# ---------------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------------

OUTPUT_DIR = Path("embeddings")
PROCESSED_DIR = Path("processed")
VECTOR_SIZE = 2314  # CLS + mean + max + attention top-k
MAX_SEQ_LENGTH = 128
DEFAULT_BATCH_SIZE = 8
TOP_K_ATTENTION = 10

# ---------------------------------------------------------------------------
# TFRecord utilities (shared philosophy with fasttext_embedding.py)
# ---------------------------------------------------------------------------


def parse_example(example: tf.Tensor) -> Mapping[str, tf.Tensor]:
    feature_description = {
        "l": tf.io.FixedLenFeature([], tf.string),  # log
        "y": tf.io.FixedLenFeature([], tf.string),  # label JSON
    }
    return tf.io.parse_single_example(example, feature_description)


def load_tfrecord_files(
    directory: Path = PROCESSED_DIR,
    log_type_filter: Optional[str] = None,
) -> pd.DataFrame:
    """Load TFRecord files into a DataFrame with batched TF pipeline."""

    if log_type_filter:
        log_dir = directory / log_type_filter
        if not log_dir.exists():
            raise FileNotFoundError(f"No directory found for '{log_type_filter}'")
        tfrecord_files = list(log_dir.glob("*.tfrecord"))
        if not tfrecord_files:
            raise FileNotFoundError(f"No TFRecord files found for log type '{log_type_filter}'")
    else:
        tfrecord_files = [file for subdir in directory.iterdir() if subdir.is_dir()
                          for file in subdir.glob("*.tfrecord")]
        if not tfrecord_files:
            raise FileNotFoundError(f"No TFRecord files found in {directory}")

    spinner = Halo(text="Loading TFRecord files", spinner="dots")
    spinner.start()

    options = tf.data.Options()
    options.experimental_deterministic = False
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
        dataset = dataset.with_options(options)
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
# Label utilities with dynamic-programming style caching
# ---------------------------------------------------------------------------


def normalize_label(label: Optional[str]) -> Optional[str]:
    if not label:
        return label
    return label.replace("-", "_").lower().strip()


@lru_cache(maxsize=200_000)
def _cached_label_tuple(label_json_str: str) -> Tuple[str, ...]:
    try:
        labels = json.loads(label_json_str)
        if not isinstance(labels, list):
            labels = [labels]
        normalized = sorted(normalize_label(label) for label in labels if label)
        return tuple(lbl for lbl in normalized if lbl)
    except json.JSONDecodeError:
        return tuple()


def get_labels_from_json(label_json_str: str) -> Set[str]:
    return set(_cached_label_tuple(label_json_str))


def collect_unique_labels_from_data(
    df: pd.DataFrame,
    *,
    show_progress: bool = True,
) -> List[str]:
    spinner = Halo(text="Collecting unique labels", spinner="dots") if show_progress else None
    if spinner:
        spinner.start()

    unique_labels: Set[str] = set()
    for label_json_str in df["label_json"]:
        unique_labels.update(get_labels_from_json(label_json_str))

    unique_labels.discard("")
    unique_labels.discard(None)  # type: ignore[arg-type]

    if spinner:
        spinner.succeed(f"Found {len(unique_labels)} unique attack types")
    return sorted(unique_labels)


def build_label_vector_factory(attack_types: Sequence[str]) -> Callable[[str], np.ndarray]:
    if not attack_types:
        empty = np.zeros(0, dtype=np.int8)
        return lambda _label_json: empty.copy()

    lookup = {attack: idx for idx, attack in enumerate(attack_types)}
    cache: Dict[str, np.ndarray] = {}

    def _vectorize(label_json_str: str) -> np.ndarray:
        cached = cache.get(label_json_str)
        if cached is not None:
            return cached.copy()
        labels = get_labels_from_json(label_json_str)
        vector = np.zeros(len(attack_types), dtype=np.int8)
        for label in labels:
            idx = lookup.get(label)
            if idx is not None:
                vector[idx] = 1
        cache[label_json_str] = vector
        return vector.copy()

    return _vectorize


# ---------------------------------------------------------------------------
# Device handling and tokenizer dataset with memoization
# ---------------------------------------------------------------------------


def get_device() -> torch.device:
    if torch.cuda.is_available():
        print("Using CUDA device")
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        print("Using MPS device")
        return torch.device("mps")
    print("Using CPU device")
    return torch.device("cpu")


@dataclass
class TokenizerBundle:
    tokenizer: BertTokenizer
    cache: Callable[[str], Tuple[Tuple[int, ...], Tuple[int, ...]]]


def build_tokenizer_bundle(tokenizer: BertTokenizer, max_length: int = MAX_SEQ_LENGTH) -> TokenizerBundle:
    @lru_cache(maxsize=50_000)
    def _tokenize(text: str) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        encoding = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = tuple(int(x) for x in encoding["input_ids"].squeeze(0).tolist())
        attention_mask = tuple(int(x) for x in encoding["attention_mask"].squeeze(0).tolist())
        return input_ids, attention_mask

    return TokenizerBundle(tokenizer=tokenizer, cache=_tokenize)


class LogBERTDataset(Dataset):
    def __init__(self, texts: Sequence[str], tokenizer_bundle: TokenizerBundle):
        self.texts = list(texts)
        self.bundle = tokenizer_bundle

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        input_ids, attention_mask = self.bundle.cache(text)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "idx": torch.tensor(idx, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# BERT embedding extraction
# ---------------------------------------------------------------------------


def load_pretrained_logbert(device: torch.device) -> Tuple[BertModel, BertTokenizer]:
    spinner = Halo(text="Loading pre-trained BERT model", spinner="dots")
    spinner.start()
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    model.to(device)
    model.eval()
    spinner.succeed("Loaded pre-trained BERT (bert-base-uncased)")
    return model, tokenizer


def extract_bert_embeddings(
    df: pd.DataFrame,
    model: BertModel,
    tokenizer_bundle: TokenizerBundle,
    device: torch.device,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> np.ndarray:
    if df.empty:
        return np.zeros((0, VECTOR_SIZE), dtype=np.float32)

    dataset = LogBERTDataset(df["log"].tolist(), tokenizer_bundle)

    num_workers = 0 if device.type == "mps" else min(4, os.cpu_count() or 1)
    pin_memory = device.type == "cuda"

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    embeddings = np.empty((len(dataset), VECTOR_SIZE), dtype=np.float32)

    spinner = Halo(text="Extracting BERT embeddings", spinner="dots")
    spinner.start()

    top_k = TOP_K_ATTENTION

    for batch_idx, batch in enumerate(dataloader, start=1):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        indices = batch["idx"].cpu().numpy()

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                output_attentions=True,
                return_dict=True,
            )

        hidden_states = outputs.last_hidden_state  # (batch, seq_len, 768)
        cls_tokens = hidden_states[:, 0, :]

        mask = attention_mask.unsqueeze(-1).type_as(hidden_states)
        lengths = mask.sum(dim=1).clamp(min=1.0)
        mean_pool = (hidden_states * mask).sum(dim=1) / lengths

        # Replace masked positions with very negative numbers before max
        masked_hidden = hidden_states.masked_fill(mask == 0, -1e9)
        max_pool = masked_hidden.max(dim=1).values
        max_pool = torch.where(torch.isfinite(max_pool), max_pool, torch.zeros_like(max_pool))

        # Attention summary (average heads from final layer, CLS to tokens)
        attentions = outputs.attentions[-1].mean(dim=1)  # (batch, seq_len, seq_len)
        cls_attention = attentions[:, 0, :] * attention_mask
        cls_attention[:, 0] = 0.0  # drop self-attention weight

        effective_k = min(top_k, cls_attention.size(1) - 1)
        if effective_k > 0:
            top_values, _ = torch.topk(cls_attention, k=effective_k, dim=1)
            if effective_k < top_k:
                pad = torch.zeros((top_values.size(0), top_k - effective_k), device=top_values.device)
                top_values = torch.cat([top_values, pad], dim=1)
        else:
            top_values = torch.zeros((cls_attention.size(0), top_k), device=cls_attention.device)

        batch_embedding = torch.cat([cls_tokens, mean_pool, max_pool, top_values], dim=1)
        embeddings[indices] = batch_embedding.cpu().numpy().astype(np.float32)

        spinner.text = f"Processing batch {batch_idx}/{len(dataloader)}"

        del outputs, hidden_states, cls_tokens, mean_pool, max_pool, attentions, cls_attention, top_values
        torch.cuda.empty_cache() if device.type == "cuda" else None

    spinner.succeed("Embedding extraction complete")
    return embeddings


# ---------------------------------------------------------------------------
# Label processing & visualization (shared with FastText pipeline philosophy)
# ---------------------------------------------------------------------------


def process_embeddings(
    df: pd.DataFrame,
    model: BertModel,
    tokenizer_bundle: TokenizerBundle,
    device: torch.device,
    *,
    use_global_attack_list: bool = False,
) -> pd.DataFrame:
    df = df.copy()

    embeddings = extract_bert_embeddings(df, model, tokenizer_bundle, device)
    df["log_embedding"] = [row for row in embeddings]

    if use_global_attack_list:
        attack_types = collect_unique_labels_from_data(df, show_progress=False)
        vectorize = build_label_vector_factory(attack_types)
        df["binary_labels"] = [vectorize(label_json) for label_json in df["label_json"]]
        df.attrs["attack_types"] = attack_types
    else:
        log_type_to_attacks: Dict[str, List[str]] = {}
        vectorizers: Dict[str, Callable[[str], np.ndarray]] = {}
        for log_type, group_df in df.groupby("log_type"):
            attacks = collect_unique_labels_from_data(group_df, show_progress=False)
            log_type_to_attacks[log_type] = attacks
            vectorizers[log_type] = build_label_vector_factory(attacks)
        binary_labels: List[np.ndarray] = []
        for row in df.itertuples(index=False):
            vec = vectorizers.get(row.log_type)
            binary_labels.append(vec(row.label_json) if vec else np.zeros(0, dtype=np.int8))
        df["binary_labels"] = binary_labels
        df.attrs["log_type_to_attacks"] = log_type_to_attacks

    return df


def visualize_embeddings(df: pd.DataFrame, output_file: Optional[Path] = None) -> None:
    MAX_TOTAL_POINTS = 50_000
    MAX_POINTS_PER_CLASS = 1_500

    spinner = Halo(text="Preparing visualization data", spinner="dots")
    spinner.start()

    viz_labels: List[str] = []
    for label_json in df["label_json"]:
        labels = get_labels_from_json(label_json)
        viz_labels.append(", ".join(sorted(labels)) if labels else "normal")

    df = df.copy()
    df["viz_label"] = viz_labels

    spinner.text = "Applying balanced sampling"
    rng = np.random.default_rng(42)
    label_to_indices: Dict[str, List[int]] = {}
    for idx, label in enumerate(viz_labels):
        label_to_indices.setdefault(label, []).append(idx)

    selected: List[int] = []
    for label, indices in label_to_indices.items():
        if len(indices) > MAX_POINTS_PER_CLASS:
            selected.extend(rng.choice(indices, MAX_POINTS_PER_CLASS, replace=False))
        else:
            selected.extend(indices)

    if len(selected) > MAX_TOTAL_POINTS:
        selected = list(rng.choice(selected, MAX_TOTAL_POINTS, replace=False))

    embeddings = np.vstack([df.iloc[i]["log_embedding"] for i in selected]).astype(np.float32)
    sampled_labels = [viz_labels[i] for i in selected]
    sampled_types = [df.iloc[i]["log_type"] for i in selected]

    spinner.text = f"Running t-SNE on {len(embeddings)} samples"
    perplexity = min(50, max(5, len(embeddings) // 1000))

    from sklearn.manifold import TSNE  # local import to avoid cost when unused

    reduced = TSNE(
        n_components=2,
        perplexity=perplexity,
        n_iter=500,
        learning_rate="auto",
        init="pca",
        method="barnes_hut",
        random_state=42,
    ).fit_transform(embeddings)

    df_plot = pd.DataFrame({
        "x": reduced[:, 0],
        "y": reduced[:, 1],
        "label": sampled_labels,
        "log_type": sampled_types,
    })

    spinner.succeed("t-SNE complete")

    label_counts = df_plot["label"].value_counts()
    print(f"\nVisualization includes {len(label_counts)} label combinations (top 10 shown):")
    for label, count in label_counts.head(10).items():
        percentage = (count / len(df_plot)) * 100
        print(f"  {label}: {count} ({percentage:.2f}%)")
    if len(label_counts) > 10:
        print(f"  ... and {len(label_counts) - 10} more")

    unique_labels = sorted(df_plot["label"].unique())
    palette = sns.color_palette("husl", len(unique_labels))
    color_map = {label: palette[i] for i, label in enumerate(unique_labels)}
    if "normal" in color_map:
        color_map["normal"] = "green"

    spinner = Halo(text="Creating visualization plot", spinner="dots")
    spinner.start()

    plt.figure(figsize=(16, 10))
    for label in unique_labels:
        subset = df_plot[df_plot["label"] == label]
        plt.scatter(
            subset["x"],
            subset["y"],
            c=[color_map[label]],
            label=label,
            alpha=0.6,
            s=20,
            edgecolors="none",
        )
    plt.title(f"t-SNE Visualization: LogBERT Embeddings (All {len(unique_labels)} Classes)")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")

    if len(unique_labels) <= 20:
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
        plt.tight_layout(rect=[0, 0, 0.85, 1])
    else:
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=6, ncol=2)
        plt.tight_layout(rect=[0, 0, 0.8, 1])

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        spinner.succeed(f"Saved visualization to {output_file}")
    else:
        plt.show()
        spinner.succeed("Displayed visualization")

    plt.close()


# ---------------------------------------------------------------------------
# Persistence helpers (identical to FastText format expectations)
# ---------------------------------------------------------------------------


def save_embeddings_and_labels(df: pd.DataFrame, output_dir: Path, log_type: str) -> None:
    spinner = Halo(text=f"Saving embeddings for {log_type}", spinner="dots")
    spinner.start()

    output_dir.mkdir(parents=True, exist_ok=True)

    log_embeddings = np.vstack(df["log_embedding"].tolist()).astype(np.float32)
    log_filename = output_dir / f"log_{log_type}.pkl"
    spinner.text = f"Writing {log_filename.name}"
    with open(log_filename, "wb") as f:
        pickle.dump(log_embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)

    if "binary_labels" in df.columns:
        valid = [vec for vec in df["binary_labels"] if isinstance(vec, np.ndarray) and vec.size > 0]
        if valid:
            binary_matrix = np.vstack(valid).astype(np.int8)
            label_filename = output_dir / f"label_{log_type}.pkl"

            classes: List[str] = []
            if "attack_types" in df.attrs:
                classes = list(df.attrs["attack_types"])
            elif "log_type_to_attacks" in df.attrs:
                classes = list(df.attrs["log_type_to_attacks"].get(log_type, []))

            label_payload = {
                "vectors": binary_matrix,
                "classes": classes,
                "description": "Binary multi-label vectors where [0 1 0] means only the second class is present",
            }

            spinner.text = f"Writing {label_filename.name}"
            with open(label_filename, "wb") as f:
                pickle.dump(label_payload, f, protocol=pickle.HIGHEST_PROTOCOL)

            attack_info = output_dir / f"attack_types_{log_type}.txt"
            spinner.text = f"Writing {attack_info.name}"
            with open(attack_info, "w", encoding="utf-8") as f:
                f.write(f"Attack Types and Column Mapping for {log_type}\n")
                f.write("=" * 60 + "\n\n")
                f.write("Binary Vector Format:\n")
                f.write("- Each row is a binary vector representing one log entry\n")
                f.write("- [0 1 0] means only the second attack type is present\n")
                f.write("- Multiple attacks: [1 0 1] -> first and third attacks\n\n")
                f.write(f"Total attack types: {len(classes)}\n")
                f.write(f"Vector dimension: {len(classes)}\n\n")
                if classes:
                    f.write("Column Mapping (Index -> Attack Type):\n")
                    f.write("-" * 40 + "\n")
                    for idx, attack in enumerate(classes):
                        f.write(f"Column {idx:2d}: {attack}\n")
                    
                    f.write("\nExample Interpretations:\n")
                    f.write("-" * 25 + "\n")
                    example = np.zeros(len(classes), dtype=np.int8)
                    if len(classes) >= 1:
                        example[:1] = [1]
                        f.write(f"{example.tolist()} -> Only '{classes[0]}' attack present\n")
                        example[:1] = [0]
                    if len(classes) >= 2:
                        example[:2] = [0, 1]
                        f.write(f"{example.tolist()} -> Only '{classes[1]}' attack present\n")
                        example[:2] = [1, 1]
                        f.write(f"{example.tolist()} -> Both '{classes[0]}' and '{classes[1]}' present\n")
                    f.write(f"{np.zeros(len(classes), dtype=np.int8).tolist()} -> Normal log\n")
                else:
                    f.write("No attack types found for this log type.\n")
                f.write("\nGenerated by LogBERT embeddings extraction\n")
                f.write("Embedding dimension: 2314D (CLS + mean + max + attention top-k)\n")

            spinner.succeed(f"Saved embeddings and labels for {log_type}")
            print(f"\nSaved files for {log_type}:")
            print(f"  - {log_filename.name}: Log embeddings {log_embeddings.shape} (2314D)")
            print(f"  - {label_filename.name}: Binary label vectors {binary_matrix.shape}")
            print(f"  - {attack_info.name}: Attack mapping")
            print(f"  - Classes: {classes}")
        else:
            spinner.warn(f"No valid binary labels found for {log_type}")
    else:
        spinner.warn(f"No binary labels found for {log_type}, saved only log embeddings")


def display_data_distribution(df: pd.DataFrame, log_type: str = "all combined") -> Tuple[int, int]:
    print(f"\n{'=' * 20} Data Distribution for '{log_type}' {'=' * 20}")
    total = len(df)
    print(f"Total log entries: {total}")

    attack_counts: Dict[str, int] = {}
    normal = 0
    attack_rows = 0

    for label_json in df["label_json"]:
        labels = get_labels_from_json(label_json)
        if labels:
            attack_rows += 1
            for label in labels:
                attack_counts[label] = attack_counts.get(label, 0) + 1
        else:
            normal += 1

    if attack_counts:
        print("\nAttack type distribution:")
        for attack, count in sorted(attack_counts.items(), key=lambda x: x[1], reverse=True):
            pct = (count / total) * 100 if total else 0
            print(f"  {attack}: {count} ({pct:.2f}%)")

    attack_pct = (attack_rows / total) * 100 if total else 0
    normal_pct = (normal / total) * 100 if total else 0
    print(f"\nLogs with attacks: {attack_rows} ({attack_pct:.2f}%)")
    print(f"Normal logs: {normal} ({normal_pct:.2f}%)")

    if log_type == "all combined":
        print("\nLog type distribution:")
        for lt, count in df["log_type"].value_counts().items():
            pct = (count / total) * 100 if total else 0
            print(f"  {lt}: {count} ({pct:.2f}%)")

    print("=" * 70 + "\n")
    return attack_rows, normal


def find_available_log_types() -> List[str]:
    if not PROCESSED_DIR.exists():
        return []
    return sorted([
        path.name
        for path in PROCESSED_DIR.iterdir()
        if path.is_dir() and any(path.glob("*.tfrecord"))
    ])


# ---------------------------------------------------------------------------
# Main CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate LogBERT embeddings using pre-trained BERT"
    )
    parser.add_argument("--log-type", type=str, default=None, help="Process only this specific log type")
    parser.add_argument("--output-subdir", type=str, default=None, help="Optional subdirectory under embeddings/ (e.g., 'logbert')")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size for BERT inference (default: 8)")
    args = parser.parse_args()

    if not HAS_TRANSFORMERS:
        print("❌ transformers library not available. Please install it:")
        print("   pip install transformers")
        return

    global OUTPUT_DIR
    if args.output_subdir:
        OUTPUT_DIR = Path("embeddings") / args.output_subdir
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    available_types = find_available_log_types()
    if not available_types:
        print("No log types with data found.")
        return

    print(f"Found {len(available_types)} log types: {', '.join(available_types)}")

    if args.log_type:
        if args.log_type not in available_types:
            print(f"Log type '{args.log_type}' not found. Available: {', '.join(available_types)}")
            return
        log_types = [args.log_type]
        run_combined = False
    else:
        log_types = available_types
        run_combined = False

    device = get_device()
    model, tokenizer = load_pretrained_logbert(device)
    tokenizer_bundle = build_tokenizer_bundle(tokenizer, MAX_SEQ_LENGTH)

    for log_type in log_types:
        print(f"\n{'=' * 50}\nProcessing log type: {log_type}\n{'=' * 50}")
        try:
            df = load_tfrecord_files(log_type_filter=log_type)
            if df.empty:
                print(f"No data for log type '{log_type}'. Skipping.")
                continue

            display_data_distribution(df, log_type)
            df = process_embeddings(df, model, tokenizer_bundle, device, use_global_attack_list=False)

            output_dir = OUTPUT_DIR / log_type
            save_embeddings_and_labels(df, output_dir, log_type)

            visualize_embeddings(df, output_file=output_dir / "visualization.png")
        except Exception as exc:  # pragma: no cover - runtime guard
            print(f"Error processing log type {log_type}: {exc}")
            import traceback
            traceback.print_exc()

    if run_combined:
        print(f"\n{'=' * 50}\nProcessing all log types combined\n{'=' * 50}")
        try:
            df_all = load_tfrecord_files()
            if df_all.empty:
                print("No data found for combined log types")
            else:
                display_data_distribution(df_all, "all combined")
                df_all = process_embeddings(df_all, model, tokenizer_bundle, device, use_global_attack_list=True)
                save_embeddings_and_labels(df_all, OUTPUT_DIR, "all_combined")
                visualize_embeddings(df_all, output_file=OUTPUT_DIR / "visualization_all_combined.png")
        except Exception as exc:
            print(f"Error processing combined log types: {exc}")
            import traceback
            traceback.print_exc()

    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
