import pickle
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.metrics import precision_recall_fscore_support, jaccard_score
import halo

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Optimize for M2 GPU if available
if device.type == "mps":
    # Enable optimized memory format for MPS
    torch.backends.mps.allow_tf32 = True
    print("Enabled MPS optimizations for M2 GPU")

hierarchy = {
    "foothold": {"attacker_http": ["dirb", "webshell_cmd", "webshell_upload"]},
    "escalate": {"escalated_command": ["escalated_sudo_session"], "attacker_change_user": [], "reverse_shell": []},
    "attacker_vpn": {},
    "dnsteal": {"dnsteal-received": [], "dnsteal-dropped": [], "exfiltration-service": []}
}

class HierarchicalTransformer(nn.Module):
    def __init__(self, hierarchy, hidden_dim=512, num_heads=4, num_layers=2):
        super().__init__()
        self.hierarchy = hierarchy
        self.cls_proj, self.mean_proj = nn.Linear(768, hidden_dim), nn.Linear(768, hidden_dim)
        self.max_proj, self.attn_proj = nn.Linear(768, hidden_dim), nn.Linear(10, hidden_dim)

        # encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # bottleneck & decoder
        self.bottleneck = nn.Linear(hidden_dim, hidden_dim // 2)
        self.decoder = nn.Sequential(nn.Linear(hidden_dim // 2, hidden_dim), nn.ReLU(),
                                     nn.Linear(hidden_dim, 2314))

        # hierarchical heads and parent-child mappings
        self.heads = nn.ModuleDict()
        self.parent_child_map = {}
        self.node_to_index = {}
        self._build_heads(self.hierarchy)

    def _build_heads(self, hierarchy, parent=None):
        idx = len(self.node_to_index)
        for node, children in hierarchy.items():
            self.heads[node] = nn.Linear(self.bottleneck.out_features, 1)
            self.node_to_index[node] = idx
            idx += 1
            if parent:
                if parent not in self.parent_child_map:
                    self.parent_child_map[parent] = []
                self.parent_child_map[parent].append(node)
            
            if isinstance(children, dict):
                for child, leaves in children.items():
                    self.heads[child] = nn.Linear(self.bottleneck.out_features, 1)
                    self.node_to_index[child] = idx
                    idx += 1
                    if node not in self.parent_child_map:
                        self.parent_child_map[node] = []
                    self.parent_child_map[node].append(child)
                    
                    for leaf in leaves:
                        self.heads[leaf] = nn.Linear(self.bottleneck.out_features, 1)
                        self.node_to_index[leaf] = idx
                        idx += 1
                        if child not in self.parent_child_map:
                            self.parent_child_map[child] = []
                        self.parent_child_map[child].append(leaf)

    def forward(self, cls_tokens, mean_pooling, max_pooling, attn):
        cls_h, mean_h = self.cls_proj(cls_tokens), self.mean_proj(mean_pooling)
        max_h, attn_h = self.max_proj(max_pooling), self.attn_proj(attn)

        combined = torch.stack([cls_h, mean_h, max_h, attn_h], dim=1)
        h_enc = self.encoder(combined)
        pooled = h_enc.mean(1)

        z = self.bottleneck(pooled)
        recon = self.decoder(z)
        
        # compute logits (not sigmoid yet for BCEWithLogitsLoss)
        outputs = {name: head(z) for name, head in self.heads.items()}
        return recon, z, outputs

    def hierarchy_consistency_loss(self, outputs):
        """Penalty if child probability > parent probability"""
        consistency_loss = 0.0
        for parent, children in self.parent_child_map.items():
            if parent in outputs:
                parent_prob = torch.sigmoid(outputs[parent])
                for child in children:
                    if child in outputs:
                        child_prob = torch.sigmoid(outputs[child])
                        # penalize when child > parent
                        violation = F.relu(child_prob - parent_prob)
                        consistency_loss += violation.mean()
        return consistency_loss

    def propagate_labels(self, outputs):
        """Propagate active parent labels to children"""
        propagated = {}
        for name, logits in outputs.items():
            propagated[name] = torch.sigmoid(logits)
        
        # if parent active, activate children
        for parent, children in self.parent_child_map.items():
            if parent in propagated:
                parent_active = (propagated[parent] > 0.5).float()
                for child in children:
                    if child in propagated:
                        propagated[child] = torch.maximum(propagated[child], parent_active)
        return propagated

def load_logbert_embeddings():
    embeddings_dir = Path("embeddings/logbert")
    if not embeddings_dir.exists():
        raise FileNotFoundError(f"Embeddings not found: {embeddings_dir}")
    
    embeddings, labels, skipped = {}, {}, []
    
    def safe_load(path):
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except (EOFError, pickle.PickleError, Exception) as e:
            print(f"✗ {path.name}: {e}")
            return None
    
    for embed_dir in [d for d in embeddings_dir.iterdir() if d.is_dir()]:
        log_type = embed_dir.name
        log_pkl, label_pkl = embed_dir / f"log_{log_type}.pkl", embed_dir / f"label_{log_type}.pkl"
        
        if not (log_pkl.exists() and label_pkl.exists()):
            continue
            
        print(f"Loading {log_type}...", end=" ")
        
        log_data = safe_load(log_pkl)
        label_data = safe_load(label_pkl)
        
        if log_data is not None and label_data is not None:
            embeddings[log_type] = log_data
            labels[log_type] = label_data
            print("✓")
        else:
            skipped.append(log_type)
    
    if not embeddings:
        raise ValueError("No valid embeddings could be loaded. All pickle files appear corrupted.")
    
    print(f"Loaded {len(embeddings)} log types" + (f" (skipped {len(skipped)})" if skipped else ""))
    return embeddings, labels

def flatten_hierarchy(h, parent=None):
    """Flatten hierarchy to list of all nodes"""
    for node, children in h.items():
        yield node
        if isinstance(children, dict):
            for child, leaves in children.items():
                yield child
                for leaf in leaves:
                    yield leaf

def create_multilabel_targets(labels_dict, hierarchy):
    """Convert label vectors to multi-label targets aligned with hierarchy"""
    if 'vectors' not in labels_dict:
        return None
    
    label_matrix = labels_dict['vectors']
    n_samples = label_matrix.shape[0]
    
    # create mapping from flat index to node name
    node_list = list(flatten_hierarchy(hierarchy))
    n_nodes = len(node_list)
    
    # create aligned multi-label matrix
    targets = np.zeros((n_samples, n_nodes), dtype=np.float32)
    
    # map existing labels to new structure (if dimensions match)
    if label_matrix.shape[1] <= n_nodes:
        targets[:, :label_matrix.shape[1]] = label_matrix
    
    return targets

def load_datasets(embeddings, labels, batch_size=128):
    datasets = {}
    for log_type, log_vectors in embeddings.items():
        cls_tokens, mean_pooling = log_vectors[:, :768], log_vectors[:, 768:1536]
        max_pooling, attn = log_vectors[:, 1536:2304], log_vectors[:, 2304:]
        
        # create multi-label targets
        if log_type in labels:
            targets = create_multilabel_targets(labels[log_type], hierarchy)
        else:
            targets = np.zeros((log_vectors.shape[0], len(list(flatten_hierarchy(hierarchy)))))
        
        if targets is None:
            targets = np.zeros((log_vectors.shape[0], len(list(flatten_hierarchy(hierarchy)))))
        
        dataset = TensorDataset(
            torch.from_numpy(cls_tokens).float(),
            torch.from_numpy(mean_pooling).float(),
            torch.from_numpy(max_pooling).float(),
            torch.from_numpy(attn).float(),
            torch.from_numpy(targets).float()
        )
        
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        datasets[log_type] = {'loader': loader, 'num_samples': log_vectors.shape[0]}
        
        total = log_vectors.shape[0]
        anomalies = (targets.sum(axis=1) > 0).sum()
        print(f"[{log_type}] {total} samples | anomalies: {anomalies} ({anomalies/total:.2%})")
        
        # Show per-class distribution
        node_names = list(flatten_hierarchy(hierarchy))
        normal_count = total - anomalies
        print("Per-class distribution:")
        print(f"  normal: {normal_count} ({normal_count/total:.2%})")
        for i, name in enumerate(node_names):
            if i < targets.shape[1]:
                count = int(targets[:, i].sum())
                if count > 0:
                    print(f"  {name}: {count} ({count/total:.2%})")
    
    return datasets

def train_model(model, train_loader, val_loader, epochs=10):
    """
    Trains a hierarchical model with an integrated, concise data balancing strategy (SMOTE).
    """
    spinner = halo.Halo(text="Training hierarchical model", spinner="dots")
    spinner.start()

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=2e-4, epochs=epochs, steps_per_epoch=len(train_loader))
    mse_loss = nn.MSELoss()
    use_amp = device.type in ["cuda", "mps"]
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # --- Data Loading and Balancing Stage ---
    with torch.no_grad():
        spinner.text = "Loading and balancing data..."
        Xc, Xm, Xmx, Xa, Y = [], [], [], [], []
        for c, m, x, a, t in train_loader:
            Xc.append(c); Xm.append(m); Xmx.append(x); Xa.append(a); Y.append(t)
        Xc = torch.cat(Xc); Xm = torch.cat(Xm); Xmx = torch.cat(Xmx); Xa = torch.cat(Xa); Y = torch.cat(Y)
        
        F_full = torch.cat([Xc, Xm, Xmx, Xa], dim=1)
        L = Y.shape[1]
        n_initial = Y.shape[0]
        pre_counts = Y.sum(0).cpu().numpy()

        # 1. Downsample over-represented classes
        downsample_cap = 2000
        keep = torch.ones(n_initial, dtype=torch.bool)
        g = torch.Generator().manual_seed(42)
        for j in range(L):
            idx = torch.nonzero(Y[:, j] == 1, as_tuple=True)[0]
            if idx.numel() > downsample_cap:
                drop_idx = idx[torch.randperm(idx.numel(), generator=g)[downsample_cap:]]
                keep[drop_idx] = False
        
        F_full, Y = F_full[keep], Y[keep]

        # 2. Vectorized SMOTE for under-represented classes
        oversample_target = 8000
        F_new, Y_new = [F_full], [Y]
        for j in range(L):
            idx = torch.nonzero(Y[:, j] == 1, as_tuple=True)[0]
            c = idx.numel()
            needed = max(0, oversample_target - c)
            if c < 2 or needed == 0:
                continue
            
            a = idx[torch.randint(0, c, (needed,))]
            b = idx[torch.randint(0, c, (needed,))]
            lam = torch.rand(needed, 1)
            F_new.append(F_full[a] * lam + F_full[b] * (1 - lam))
            Y_new.append(torch.maximum(Y[a], Y[b]))
        
        Fb, Yb = torch.cat(F_new), torch.cat(Y_new)

        # 3. Report results and create new DataLoader
        post_counts = Yb.sum(0).cpu().numpy()
        summary_df = pd.DataFrame({
            'Label': [f'L_{i}' for i in range(L)],
            'Before': pre_counts,
            'After': post_counts
        })
        spinner.succeed("Data balancing complete.")
        print(summary_df.to_string())
        print(f"\nDataset size: {n_initial} -> {Fb.shape[0]} samples.")

        d1, d2, d3 = Xc.shape[1], Xm.shape[1], Xmx.shape[1]
        Xc_b, Xm_b = Fb[:, :d1], Fb[:, d1:d1+d2]
        Xmx_b, Xa_b = Fb[:, d1+d2:d1+d2+d3], Fb[:, d1+d2+d3:]
        train_loader = DataLoader(
            TensorDataset(Xc_b, Xm_b, Xmx_b, Xa_b, Yb),
            batch_size=getattr(train_loader, "batch_size", 1024),
            shuffle=True, drop_last=True, pin_memory=True
        )

        pos_weights = Yb.shape[0] / (2 * Yb.sum(0).clamp(min=1))
        bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weights.to(device))

    # --- Model Training Stage ---
    best_val_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        train_losses = []
        spinner.start(f"Epoch {epoch+1}/{epochs}")
        for cls_tokens, mean_pooling, max_pooling, attn, targets in train_loader:
            cls_tokens, mean_pooling, max_pooling, attn, targets = \
                cls_tokens.to(device), mean_pooling.to(device), max_pooling.to(device), attn.to(device), targets.to(device)

            with torch.cuda.amp.autocast(enabled=use_amp):
                recon, z, outputs = model(cls_tokens, mean_pooling, max_pooling, attn)
                feats = torch.cat([cls_tokens, mean_pooling, max_pooling, attn], dim=1)
                recon_loss = mse_loss(recon, feats)
                logits = torch.cat([v for v in outputs.values()], dim=1) if isinstance(outputs, dict) else outputs
                ml_loss = bce_loss(logits, targets)
                hier_loss = model.hierarchy_consistency_loss(outputs)
                total_loss = 0.1 * recon_loss + ml_loss + 0.1 * hier_loss

            if not torch.isfinite(total_loss): continue
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(total_loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(total_loss.item())

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for cls_tokens, mean_pooling, max_pooling, attn, targets in val_loader:
                cls_tokens, mean_pooling, max_pooling, attn, targets = \
                    cls_tokens.to(device), mean_pooling.to(device), max_pooling.to(device), attn.to(device), targets.to(device)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    recon, _, outputs = model(cls_tokens, mean_pooling, max_pooling, attn)
                    feats = torch.cat([cls_tokens, mean_pooling, max_pooling, attn], dim=1)
                    recon_loss = mse_loss(recon, feats)
                    logits = torch.cat([v for v in outputs.values()], dim=1) if isinstance(outputs, dict) else outputs
                    ml_loss = bce_loss(logits, targets)
                    hier_loss = model.hierarchy_consistency_loss(outputs)
                    total_loss = 0.1 * recon_loss + ml_loss + 0.1 * hier_loss
                if torch.isfinite(total_loss): val_losses.append(total_loss.item())

        avg_train_loss = np.mean(train_losses) if train_losses else 0.0
        avg_val_loss = np.mean(val_losses) if val_losses else 0.0
        spinner.text = f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}"
        if avg_val_loss < best_val_loss: best_val_loss = avg_val_loss

    spinner.succeed(f"Training completed. Best validation loss: {best_val_loss:.4f}")
    return model

def evaluate_model(model, test_loader, log_type):
    model.eval()
    all_preds, all_targets = [], []
    node_names = list(model.node_to_index.keys())
    
    with torch.no_grad():
        for cls_tokens, mean_pooling, max_pooling, attn, targets in test_loader:
            cls_tokens = cls_tokens.to(device)
            mean_pooling = mean_pooling.to(device)
            max_pooling = max_pooling.to(device)
            attn = attn.to(device)
            
            _, _, outputs = model(cls_tokens, mean_pooling, max_pooling, attn)
            
            # apply hierarchy propagation
            propagated = model.propagate_labels(outputs)
            
            # collect predictions
            batch_preds = np.zeros((cls_tokens.shape[0], len(node_names)))
            for name, probs in propagated.items():
                if name in model.node_to_index:
                    idx = model.node_to_index[name]
                    batch_preds[:, idx] = (probs.cpu().numpy() > 0.5).astype(int).squeeze()
            
            all_preds.append(batch_preds)
            all_targets.append(targets.cpu().numpy())
    
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    
    # per-class metrics
    print("\n=== Per-Class Metrics ===")
    print(f"{'Class':<25} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 55)
    
    class_metrics = []
    for i, name in enumerate(node_names):
        if i < all_targets.shape[1]:
            y_true = all_targets[:, i]
            y_pred = all_preds[:, i]
            
            if y_true.sum() > 0:  # only evaluate if class has samples
                prec, rec, f1, _ = precision_recall_fscore_support(
                    y_true, y_pred, average='binary', zero_division=0
                )
                class_metrics.append([prec, rec, f1])
                print(f"{name:<25} {prec:>10.3f} {rec:>10.3f} {f1:>10.3f}")
    
    # overall metrics
    print("\n=== Overall Metrics ===")
    
    # micro-averaged (global)
    micro_prec, micro_rec, micro_f1, _ = precision_recall_fscore_support(
        all_targets.ravel(), all_preds.ravel(), average='micro', zero_division=0
    )
    print(f"Micro-averaged: Precision={micro_prec:.3f}, Recall={micro_rec:.3f}, F1={micro_f1:.3f}")
    
    # macro-averaged
    if class_metrics:
        macro_metrics = np.mean(class_metrics, axis=0)
        print(f"Macro-averaged: Precision={macro_metrics[0]:.3f}, Recall={macro_metrics[1]:.3f}, F1={macro_metrics[2]:.3f}")
    
    # jaccard score (multi-label)
    jaccard = jaccard_score(all_targets, all_preds, average='samples', zero_division=0)
    print(f"Jaccard Score (samples): {jaccard:.3f}")
    
    # anomaly detection (any attack vs none)
    any_attack_true = (all_targets.sum(axis=1) > 0).astype(int)
    any_attack_pred = (all_preds.sum(axis=1) > 0).astype(int)
    
    if len(np.unique(any_attack_true)) > 1:
        anomaly_prec, anomaly_rec, anomaly_f1, _ = precision_recall_fscore_support(
            any_attack_true, any_attack_pred, average='binary', zero_division=0
        )
        print(f"\nAnomaly Detection: Precision={anomaly_prec:.3f}, Recall={anomaly_rec:.3f}, F1={anomaly_f1:.3f}")
    
    # Save evaluation report
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    report_path = results_dir / f"hierarchical_{log_type}_evaluation_{timestamp}.txt"
    
    with open(report_path, 'w') as f:
        f.write(f"Hierarchical Transformer Evaluation Report\n")
        f.write(f"Log Type: {log_type}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Dataset: {len(all_targets)} test samples\n")
        f.write("="*50 + "\n\n")
        
        f.write("Per-Class Metrics:\n")
        f.write(f"{'Class':<25} {'Precision':>10} {'Recall':>10} {'F1':>10}\n")
        f.write("-" * 55 + "\n")
        
        for i, name in enumerate(node_names):
            if i < all_targets.shape[1]:
                y_true = all_targets[:, i]
                y_pred = all_preds[:, i]
                if y_true.sum() > 0:
                    prec, rec, f1, _ = precision_recall_fscore_support(
                        y_true, y_pred, average='binary', zero_division=0
                    )
                    f.write(f"{name:<25} {prec:>10.3f} {rec:>10.3f} {f1:>10.3f}\n")
        
        f.write(f"\nOverall Metrics:\n")
        f.write(f"Micro-averaged: Precision={micro_prec:.3f}, Recall={micro_rec:.3f}, F1={micro_f1:.3f}\n")
        if class_metrics:
            macro_metrics = np.mean(class_metrics, axis=0)
            f.write(f"Macro-averaged: Precision={macro_metrics[0]:.3f}, Recall={macro_metrics[1]:.3f}, F1={macro_metrics[2]:.3f}\n")
        f.write(f"Jaccard Score: {jaccard:.3f}\n")
        if len(np.unique(any_attack_true)) > 1:
            f.write(f"Anomaly Detection: Precision={anomaly_prec:.3f}, Recall={anomaly_rec:.3f}, F1={anomaly_f1:.3f}\n")
    
    print(f"\nEvaluation report saved: {report_path}")

def main():
    # load data
    embeddings, labels = load_logbert_embeddings()
    datasets = load_datasets(embeddings, labels)
    
    for log_type, data in datasets.items():
        print(f"\n{'='*50}")
        print(f"Processing {log_type}")
        print('='*50)
        
        # Stratified split to preserve class distribution
        dataset = data['loader'].dataset
        all_targets = torch.stack([dataset[i][4] for i in range(len(dataset))])
        anomaly_mask = (all_targets.sum(dim=1) > 0)
        
        # Get indices for normal and anomaly samples
        normal_idx = torch.nonzero(~anomaly_mask, as_tuple=True)[0]
        anomaly_idx = torch.nonzero(anomaly_mask, as_tuple=True)[0]
        
        # Stratified sampling
        def stratified_split(indices, train_ratio=0.8, val_ratio=0.1):
            n = len(indices)
            n_train = int(train_ratio * n)
            n_val = int(val_ratio * n)
            perm = torch.randperm(n)
            return indices[perm[:n_train]], indices[perm[n_train:n_train+n_val]], indices[perm[n_train+n_val:]]
        
        normal_train, normal_val, normal_test = stratified_split(normal_idx)
        anomaly_train, anomaly_val, anomaly_test = stratified_split(anomaly_idx)
        
        train_indices = torch.cat([normal_train, anomaly_train])
        val_indices = torch.cat([normal_val, anomaly_val])
        test_indices = torch.cat([normal_test, anomaly_test])
        
        train_set = torch.utils.data.Subset(dataset, train_indices)
        val_set = torch.utils.data.Subset(dataset, val_indices)
        test_set = torch.utils.data.Subset(dataset, test_indices)
        train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=128, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=128, shuffle=False)
        
        print(f"Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")
        
        # initialize and train model
        model = HierarchicalTransformer(hierarchy).to(device)
        model = train_model(model, train_loader, val_loader, epochs=10)
        
        # evaluate
        evaluate_model(model, test_loader, log_type)
        
        # save model
        Path("models").mkdir(exist_ok=True)
        torch.save(model.state_dict(), f"models/hierarchical_{log_type}.pth")
        print(f"\nModel saved to models/hierarchical_{log_type}.pth")

if __name__ == "__main__":
    main()