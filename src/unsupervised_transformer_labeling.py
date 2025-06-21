#!/usr/bin/env python3
"""Unsupervised log labelling using a transformer encoder.

Logs are embedded with a pre-trained sentence-transformer model and
clustered with KMeans. Cluster assignments are mapped to the majority
true label for evaluation.
"""

from pathlib import Path
import json
import pickle
from typing import List

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModel
import torch
import tensorflow as tf

PROCESSED_DIR = Path("processed")
BATCH_SIZE = 32
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def parse_example(example_proto):
    feature_description = {"l": tf.io.FixedLenFeature([], tf.string), "y": tf.io.FixedLenFeature([], tf.string)}
    example = tf.io.parse_single_example(example_proto, feature_description)
    return example["l"], example["y"]


def load_data() -> tuple[list[str], list[str]]:
    tfrecord_files = list(PROCESSED_DIR.glob("**/*.tfrecord"))
    if not tfrecord_files:
        raise FileNotFoundError("No TFRecord files found")
    ds = tf.data.TFRecordDataset(tfrecord_files, compression_type="GZIP").map(parse_example)
    logs = [l.numpy().decode("utf-8") for l, _ in ds]
    labels = [y.numpy().decode("utf-8") for _, y in ds]
    return logs, labels


def embed_texts(texts: List[str]) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    embeddings = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            tokens = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
            output = model(**tokens)
            vec = output.last_hidden_state.mean(dim=1)
            embeddings.append(vec.cpu().numpy())
    return np.vstack(embeddings)


def main():
    logs, labels = load_data()
    vectors = embed_texts(logs)
    n_clusters = len(set(labels)) or 2
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    cluster_ids = kmeans.fit_predict(vectors)

    # map cluster -> majority true label
    mapping = {}
    for cid in range(n_clusters):
        indices = [i for i, c in enumerate(cluster_ids) if c == cid]
        true = [labels[i] for i in indices]
        if true:
            mapping[cid] = max(set(true), key=true.count)
        else:
            mapping[cid] = "unknown"
    pred_labels = [mapping[c] for c in cluster_ids]
    score = f1_score(labels, pred_labels, average="weighted")
    print(f"Unsupervised F1-score: {score:.4f}")

    with open("results/unsupervised_labeling_score.txt", "w") as f:
        f.write(str(score))


if __name__ == "__main__":
    main()
