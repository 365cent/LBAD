#!/usr/bin/env python3
"""Generate log embeddings using LogBERT and KnowLog models.

This script loads processed TFRecord logs, fine-tunes the chosen model
with a masked language modeling objective (self-supervised), then
extracts embeddings for each log entry.
"""

from pathlib import Path
import json
import pickle
from typing import List

import numpy as np
import tensorflow as tf
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

PROCESSED_DIR = Path("processed")
OUTPUT_DIR = Path("embeddings")
BATCH_SIZE = 16
EPOCHS = 1


def parse_example(example_proto):
    feature_description = {
        "l": tf.io.FixedLenFeature([], tf.string),
        "y": tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(example_proto, feature_description)
    return example["l"], example["y"]


def load_logs() -> List[str]:
    tfrecord_files = list(PROCESSED_DIR.glob("**/*.tfrecord"))
    if not tfrecord_files:
        raise FileNotFoundError("No TFRecord files found in 'processed' directory")

    dataset = tf.data.TFRecordDataset(tfrecord_files, compression_type="GZIP")
    dataset = dataset.map(parse_example)

    logs = [log.numpy().decode("utf-8") for log, _ in dataset]
    labels = [label.numpy().decode("utf-8") for _, label in dataset]
    return logs, labels


def fine_tune_model(model_name: str, texts: List[str]) -> AutoModelForMaskedLM:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)

    tokens = tokenizer(texts, truncation=True, padding=True)
    tf_dataset = tf.data.Dataset.from_tensor_slices(tokens)

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True)
    training_args = TrainingArguments(
        output_dir="tmp_finetune",
        per_device_train_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        logging_steps=10,
        learning_rate=5e-5,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tf_dataset,
        data_collator=collator,
    )
    trainer.train()
    return tokenizer, model


def embed_logs(tokenizer, model, texts: List[str]) -> np.ndarray:
    embeddings = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            tokens = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
            outputs = model(**tokens, output_hidden_states=True)
            hidden = outputs.hidden_states[-1].mean(dim=1)
            embeddings.append(hidden.cpu().numpy())
    return np.vstack(embeddings)


def process_model(model_name: str, logs: List[str], labels: List[str]):
    print(f"Processing {model_name} ...")
    tokenizer, model = fine_tune_model(model_name, logs)
    vectors = embed_logs(tokenizer, model, logs)

    out_dir = OUTPUT_DIR / model_name.split("/")[-1]
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "log_embeddings.pkl", "wb") as f:
        pickle.dump(vectors, f)
    with open(out_dir / "labels.json", "w") as f:
        json.dump(labels, f)
    print(f"Saved embeddings to {out_dir}")


def main():
    logs, labels = load_logs()
    for model in ["Sirapatsorn/Spark_Log_Analysis-logbert", "deeperbreath/knowlog-bert"]:
        try:
            process_model(model, logs, labels)
        except Exception as exc:
            print(f"Failed to process {model}: {exc}")


if __name__ == "__main__":
    main()
