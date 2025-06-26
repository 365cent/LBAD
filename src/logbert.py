#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForMaskedLM, BertConfig
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from pathlib import Path

# Configuration
MAX_SEQ_LENGTH = 256
WINDOW_SIZE = 10
BATCH_SIZE = 32
LEARNING_RATE = 4e-5
NUM_EPOCHS = 4
OUTPUT_DIR = Path("embeddings")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Device setup
def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

device = get_device()
print(f"Using device: {device}")

# Model setup
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
config = BertConfig(
    vocab_size=len(tokenizer),
    hidden_size=512,
    num_hidden_layers=4,
    num_attention_heads=8,
    intermediate_size=2048,
    max_position_embeddings=MAX_SEQ_LENGTH,
)
model = BertForMaskedLM(config).to(device)

# Dummy dataset loader class
class DummyLogDataset(torch.utils.data.Dataset):
    def __init__(self, size=1000):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        input_ids = torch.randint(0, len(tokenizer), (MAX_SEQ_LENGTH,))
        attention_mask = torch.ones(MAX_SEQ_LENGTH)
        labels = torch.randint(0, 2, (1,), dtype=torch.float)
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

# Data loading
train_dataset, val_dataset = train_test_split(range(1000), test_size=0.2, random_state=42)
train_loader = DataLoader(DummyLogDataset(len(train_dataset)), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(DummyLogDataset(len(val_dataset)), batch_size=BATCH_SIZE)

# Optimizer
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
criterion = nn.BCEWithLogitsLoss()

# Training Loop
for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    train_loss = 0
    print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
    for batch in tqdm(train_loader, desc="Training"):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)
    print(f"Average training loss: {avg_train_loss:.4f}")

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss

            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    print(f"Validation loss: {avg_val_loss:.4f}")

print("\nTraining completed successfully!")

# Extract embeddings for unsupervised learning tasks
print("\nExtracting embeddings for downstream tasks...")
model.eval()
all_embeddings = []
all_sequence_ids = []

with torch.no_grad():
    sequence_id = 0
    for batch in tqdm(val_loader, desc="Extracting embeddings"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        # Get hidden states from BERT (use last hidden layer)
        outputs = model.bert(input_ids, attention_mask=attention_mask)
        # Use [CLS] token embedding (first token) as sequence representation
        cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        all_embeddings.append(cls_embeddings)
        # Create sequence IDs for this batch
        batch_sequence_ids = np.arange(sequence_id, sequence_id + cls_embeddings.shape[0])
        all_sequence_ids.append(batch_sequence_ids)
        sequence_id += cls_embeddings.shape[0]

# Concatenate all embeddings and sequence IDs
embeddings = np.concatenate(all_embeddings, axis=0)
sequence_ids = np.concatenate(all_sequence_ids, axis=0)

# Save embeddings in compressed format for downstream tasks
output_file = OUTPUT_DIR / "logbert_embeddings.npz"
np.savez_compressed(
    output_file,
    embeddings=embeddings,
    sequence_ids=sequence_ids
)

print(f"Embeddings saved to '{output_file}'.")
print(f"Embedding shape: {embeddings.shape}")
print(f"Number of sequences: {len(sequence_ids)}")

# Example usage for unsupervised learning
print("\nExample: Loading embeddings for clustering...")
data = np.load(output_file)
loaded_embeddings = data["embeddings"]
loaded_sequence_ids = data["sequence_ids"]

from sklearn.cluster import KMeans

# Example: Perform clustering
num_clusters = 5
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(loaded_embeddings)

print(f"Cluster labels: {cluster_labels[:20]}...")  # Show first 20 labels
print(f"Unique clusters found: {np.unique(cluster_labels)}")