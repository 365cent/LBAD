import pickle
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import random_split, TensorDataset, DataLoader
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, accuracy_score
import halo
from collections import defaultdict

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

hierarchy = {
    "foothold": {"attacker_http": ["dirb", "webshell_cmd", "webshell_upload"]},
    "escalate": {"escalated_command": ["escalated_sudo_session"], "attacker_change_user": [], "reverse_shell": []},
    "attacker_vpn": {},
    "dnsteal": {"dnsteal-received": [], "dnsteal-dropped": [], "exfiltration-service": []}
}

class HierarchicalTransformer(nn.Module):
    def __init__(self, hierarchy, hidden_dim=512, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.hierarchy = hierarchy
        
        # Input projections with dropout
        self.cls_proj = nn.Linear(768, hidden_dim)
        self.mean_proj = nn.Linear(768, hidden_dim)
        self.max_proj = nn.Linear(768, hidden_dim)
        self.attn_proj = nn.Linear(10, hidden_dim)
        self.input_dropout = nn.Dropout(dropout)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Encoder with dropout
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=num_heads, 
            batch_first=True,
            dropout=dropout,
            dim_feedforward=hidden_dim * 4
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Bottleneck & decoder
        self.bottleneck = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2314)
        )

        # Anomaly detection head (binary classification)
        self.anomaly_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1)
        )
        
        # Hierarchical heads
        self.heads = nn.ModuleDict()
        self._build_heads(self.hierarchy, dropout)

    def _build_heads(self, hierarchy, dropout):
        """Build classification heads for each node in hierarchy"""
        for parent, children in hierarchy.items():
            self.heads[parent] = nn.Sequential(
                nn.Linear(self.bottleneck[0].out_features, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 1)
            )
            if isinstance(children, dict):
                for child, leaves in children.items():
                    self.heads[child] = nn.Sequential(
                        nn.Linear(self.bottleneck[0].out_features, 64),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(64, 1)
                    )
                    for leaf in leaves:
                        self.heads[leaf] = nn.Sequential(
                            nn.Linear(self.bottleneck[0].out_features, 64),
                            nn.ReLU(),
                            nn.Dropout(dropout),
                            nn.Linear(64, 1)
                        )

    def forward(self, cls_tokens, mean_pooling, max_pooling, attn):
        # Project inputs
        cls_h = self.cls_proj(cls_tokens)
        mean_h = self.mean_proj(mean_pooling)
        max_h = self.max_proj(max_pooling)
        attn_h = self.attn_proj(attn)
        
        # Apply dropout to inputs
        cls_h = self.input_dropout(cls_h)
        mean_h = self.input_dropout(mean_h)
        max_h = self.input_dropout(max_h)
        attn_h = self.input_dropout(attn_h)

        # Combine and normalize
        combined = torch.stack([cls_h, mean_h, max_h, attn_h], dim=1)
        combined = self.layer_norm(combined)
        
        # Encode
        h_enc = self.encoder(combined)
        pooled = h_enc.mean(1)

        # Bottleneck
        z = self.bottleneck(pooled)
        
        # Reconstruction
        recon = self.decoder(z)
        
        # Anomaly detection
        anomaly_score = torch.sigmoid(self.anomaly_head(z))
        
        # Hierarchical outputs
        outputs = {name: torch.sigmoid(head(z)) for name, head in self.heads.items()}
        outputs['anomaly'] = anomaly_score
        
        return recon, z, outputs

def load_logbert_embeddings():
    embeddings_dir = Path("embeddings/logbert")
    embeddings, labels = {}, {}
    for embed_dir in [d for d in embeddings_dir.iterdir() if d.is_dir()]:
        log_type = embed_dir.name
        log_pkl = embed_dir / f"log_{log_type}.pkl"
        label_pkl = embed_dir / f"label_{log_type}.pkl"
        if log_pkl.exists() and label_pkl.exists():
            with open(log_pkl, 'rb') as f:
                log_data = pickle.load(f)
            with open(label_pkl, 'rb') as f:
                label_data = pickle.load(f)
            embeddings[log_type] = log_data
            labels[log_type] = label_data
    print(f"Loaded LogBERT embeddings for {len(embeddings)} log types")
    return embeddings, labels

def prepare_labels(labels_dict, n_samples):
    """Convert label dictionary to binary anomaly labels"""
    if 'vectors' in labels_dict:
        # Sum across all anomaly types - if any is 1, it's anomalous
        anomaly_labels = (labels_dict['vectors'].sum(axis=1) > 0).astype(float)
        return anomaly_labels[:n_samples]
    else:
        # If no labels, assume all normal
        return np.zeros(n_samples)

def load_datasets(embeddings, labels, batch_size=128):
    datasets = {}
    for log_type, log_vectors in embeddings.items():
        cls_tokens = log_vectors[:, :768]
        mean_pooling = log_vectors[:, 768:1536]
        max_pooling = log_vectors[:, 1536:2304]
        attn = log_vectors[:, 2304:]
        
        # Get binary labels (0=normal, 1=anomaly)
        binary_labels = prepare_labels(labels[log_type], log_vectors.shape[0])
        
        dataset = TensorDataset(
            torch.from_numpy(cls_tokens).float(),
            torch.from_numpy(mean_pooling).float(),
            torch.from_numpy(max_pooling).float(),
            torch.from_numpy(attn).float(),
            torch.from_numpy(binary_labels).float()
        )
        
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        datasets[log_type] = {
            'loader': loader,
            'num_samples': log_vectors.shape[0],
            'num_anomalies': int(binary_labels.sum()),
            'num_normal': int((binary_labels == 0).sum())
        }
        
        total = log_vectors.shape[0]
        normal = datasets[log_type]['num_normal']
        anomalies = datasets[log_type]['num_anomalies']
        print(f"[{log_type}] {total} samples | normal: {normal} ({normal/total:.2%}) | anomalies: {anomalies} ({anomalies/total:.2%})")
    
    return datasets

def split_dataset(dataset, test_ratio=0.2, batch_size=128):
    n_total = len(dataset)
    n_test = max(1, int(n_total * test_ratio))
    n_train = n_total - n_test
    train_set, test_set = random_split(dataset, [n_train, n_test])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def train_model(model, train_loader, test_loader, epochs=10, lr=1e-3):
    """Combined training for autoencoder and anomaly detection"""
    spinner = halo.Halo(text="Training model", spinner="dots")
    spinner.start()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    
    mse_criterion = nn.MSELoss()
    bce_criterion = nn.BCELoss()
    
    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 5
    
    for epoch in range(epochs):
        model.train()
        train_losses = []
        
        for batch in train_loader:
            cls_tokens, mean_pooling, max_pooling, attn, labels = batch
            cls_tokens = cls_tokens.to(device)
            mean_pooling = mean_pooling.to(device)
            max_pooling = max_pooling.to(device)
            attn = attn.to(device)
            labels = labels.to(device).unsqueeze(1)
            
            # Forward pass
            recon, z, outputs = model(cls_tokens, mean_pooling, max_pooling, attn)
            
            # Reconstruction loss
            feats = torch.cat([cls_tokens, mean_pooling, max_pooling, attn], dim=1)
            recon_loss = mse_criterion(recon, feats)
            
            # Anomaly detection loss
            anomaly_loss = bce_criterion(outputs['anomaly'], labels)
            
            # Combined loss with weighting
            total_loss = recon_loss + 2.0 * anomaly_loss  # Weight anomaly detection more
            
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_losses.append(total_loss.item())
        
        # Validation
        model.eval()
        val_losses = []
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                cls_tokens, mean_pooling, max_pooling, attn, labels = batch
                cls_tokens = cls_tokens.to(device)
                mean_pooling = mean_pooling.to(device)
                max_pooling = max_pooling.to(device)
                attn = attn.to(device)
                labels = labels.to(device).unsqueeze(1)
                
                recon, z, outputs = model(cls_tokens, mean_pooling, max_pooling, attn)
                
                feats = torch.cat([cls_tokens, mean_pooling, max_pooling, attn], dim=1)
                recon_loss = mse_criterion(recon, feats)
                anomaly_loss = bce_criterion(outputs['anomaly'], labels)
                total_loss = recon_loss + 2.0 * anomaly_loss
                
                val_losses.append(total_loss.item())
                val_preds.extend((outputs['anomaly'].cpu().numpy() > 0.5).astype(int))
                val_labels.extend(labels.cpu().numpy().astype(int))
        
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        val_acc = accuracy_score(val_labels, val_preds)
        
        scheduler.step(avg_val_loss)
        
        spinner.text = f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.3f}"
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                spinner.text = f"Early stopping at epoch {epoch+1}"
                break
    
    spinner.stop_and_persist(text="Model training completed")
    return model

def evaluate(model, test_loader):
    """Evaluate model performance"""
    model.eval()
    y_true = []
    y_pred = []
    anomaly_scores = []
    
    with torch.no_grad():
        for batch in test_loader:
            cls_tokens, mean_pooling, max_pooling, attn, labels = batch
            cls_tokens = cls_tokens.to(device)
            mean_pooling = mean_pooling.to(device)
            max_pooling = max_pooling.to(device)
            attn = attn.to(device)
            
            _, _, outputs = model(cls_tokens, mean_pooling, max_pooling, attn)
            
            preds = (outputs['anomaly'].cpu().numpy() > 0.5).astype(int)
            scores = outputs['anomaly'].cpu().numpy()
            
            y_pred.extend(preds.flatten())
            y_true.extend(labels.numpy().astype(int))
            anomaly_scores.extend(scores.flatten())
    
    # Calculate metrics
    print("\n=== Evaluation Results ===")
    print(classification_report(y_true, y_pred, target_names=['Normal', 'Anomaly'], digits=3))
    
    # Additional metrics
    from sklearn.metrics import roc_auc_score, confusion_matrix
    if len(np.unique(y_true)) > 1:  # Only calculate AUC if we have both classes
        auc = roc_auc_score(y_true, anomaly_scores)
        print(f"ROC-AUC Score: {auc:.3f}")
        
        cm = confusion_matrix(y_true, y_pred)
        print("\nConfusion Matrix:")
        print(f"TN: {cm[0,0]}, FP: {cm[0,1]}")
        print(f"FN: {cm[1,0]}, TP: {cm[1,1]}")

def main():
    # Check if embeddings exist
    if not Path("embeddings/logbert").exists():
        print("No embeddings found. Run preprocessing first.")
        return
    
    # Create models directory if it doesn't exist
    Path("models").mkdir(exist_ok=True)
    
    # Load data
    embeddings, labels = load_logbert_embeddings()
    datasets = load_datasets(embeddings, labels)
    
    # Train and evaluate for each log type
    for log_type, data in datasets.items():
        print(f"\n{'='*50}")
        print(f"Processing {log_type}")
        print('='*50)
        
        # Split dataset
        train_loader, test_loader = split_dataset(data['loader'].dataset, test_ratio=0.2)
        print(f"Train samples: {len(train_loader.dataset)}, Test samples: {len(test_loader.dataset)}")
        
        # Initialize model
        model = HierarchicalTransformer(hierarchy, hidden_dim=512, num_heads=8, num_layers=3, dropout=0.1).to(device)
        
        # Train model
        model = train_model(model, train_loader, test_loader, epochs=15, lr=5e-4)
        
        # Evaluate
        evaluate(model, test_loader)
        
        # Save model
        model_path = f"models/transformer_{log_type}.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'log_type': log_type,
            'hierarchy': hierarchy
        }, model_path)
        print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()