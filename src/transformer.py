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
        print("Per-class distribution:")
        for i, name in enumerate(node_names):
            if i < targets.shape[1]:
                count = int(targets[:, i].sum())
                if count > 0:
                    print(f"  {name}: {count} ({count/total:.2%})")
    
    return datasets

def train_model(model, train_loader, val_loader, epochs=10, lambda_recon=1.0, lambda_hier=0.5):
    spinner = halo.Halo(text="Training hierarchical model", spinner="dots")
    spinner.start()

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)  # Fast + stable
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=2e-4, epochs=epochs, steps_per_epoch=len(train_loader))
    mse_loss = nn.MSELoss()
    use_amp = device.type in ["cuda", "mps"]  # Enable for M2
    scaler = torch.amp.GradScaler(enabled=use_amp and device.type == "cuda")

    # Feature-aware SMOTE with majority downsampling
    with torch.no_grad():
        Xc, Xm, Xmx, Xa, Y = [], [], [], [], []
        for c, m, x, a, t in train_loader:
            Xc.append(c.cpu()); Xm.append(m.cpu()); Xmx.append(x.cpu()); Xa.append(a.cpu()); Y.append(t.cpu())
        Xc = torch.cat(Xc); Xm = torch.cat(Xm); Xmx = torch.cat(Xmx); Xa = torch.cat(Xa); Y = torch.cat(Y)
        
        # Use mean & max pooling for feature-aware sampling (more semantic)
        F_semantic = torch.cat([Xm, Xmx], dim=1)  # Focus on semantic features
        F_full = torch.cat([Xc, Xm, Xmx, Xa], dim=1)
        n, L = Y.shape
        counts = Y.sum(0)
        
        # Aggressive majority downsampling (cap at 10% of max class)
        max_count = int(counts.max().item())
        downsample_cap = max(max_count // 10, 1000)  # Cap majority at 10% or 1k min
        
        # Fast stratified downsampling
        keep = torch.ones(n, dtype=torch.bool)
        g = torch.Generator().manual_seed(42)
        
        for j in range(L):
            idx = torch.nonzero(Y[:, j] == 1, as_tuple=True)[0]
            c = idx.numel()
            if c > downsample_cap:
                # Fast random sampling for speed
                drop = idx[torch.randperm(c, generator=g)[downsample_cap:]]
                keep[drop] = False
        
        F_full, F_semantic, Y = F_full[keep], F_semantic[keep], Y[keep]
        
        # Balanced oversampling target
        remaining_counts = Y.sum(0)
        target = max(int(remaining_counts.median().item()), 500)  # Balanced target
        
        F_new, Y_new = [F_full], [Y]
        for j in range(L):
            idx = torch.nonzero(Y[:, j] == 1, as_tuple=True)[0]
            c = idx.numel()
            if c < 2:
                continue
            need = max(0, target - c)
            if need == 0:
                continue
            
            # Fast vectorized SMOTE
            a = idx[torch.randint(0, c, (need,))]
            b = idx[torch.randint(0, c, (need,))]
            lam = torch.rand(need, 1)
            synth_features = F_full[a] * lam + F_full[b] * (1 - lam)
            synth_labels = torch.maximum(Y[a], Y[b])
            F_new.append(synth_features)
            Y_new.append(synth_labels)
        
        Fb, Yb = torch.cat(F_new), torch.cat(Y_new)

        d1, d2, d3, d4 = Xc.shape[1], Xm.shape[1], Xmx.shape[1], Xa.shape[1]
        Xc_b, Xm_b = Fb[:, :d1], Fb[:, d1:d1+d2]
        Xmx_b, Xa_b = Fb[:, d1+d2:d1+d2+d3], Fb[:, d1+d2+d3:]
        train_loader = DataLoader(
            TensorDataset(Xc_b, Xm_b, Xmx_b, Xa_b, Yb),
            batch_size=getattr(train_loader, "batch_size", 1024),
            shuffle=True, drop_last=False, pin_memory=True
        )

        # names for distribution (from a single forward pass)
        c0, m0, x0, a0, _ = next(iter(train_loader))
        c0, m0, x0, a0 = c0.to(device), m0.to(device), x0.to(device), a0.to(device)
        _, _, outs0 = model(c0, m0, x0, a0)
        label_names = list(outs0.keys()) if isinstance(outs0, dict) else [f"label_{i}" for i in range(Yb.shape[1])]

        # Show training split distribution changes  
        N_train, N_new = Y.shape[0], Yb.shape[0]
        normal_train = int((Y.sum(1) == 0).sum().item())
        normal_new = int((Yb.sum(1) == 0).sum().item())
        
        print(f"\nTraining split rebalancing ({N_train} → {N_new} samples):")
        print(f"{'Class':<20} {'Train Split':<12} {'After SMOTE':<12} {'Change'}")
        print("-" * 55)
        
        # Normal samples
        change = normal_new - normal_train
        sign = "++" if change > 0 else "--" if change < 0 else "  "
        print(f"{'normal':<20} {normal_train:>5} ({100.0*normal_train/N_train:>4.1f}%) {normal_new:>5} ({100.0*normal_new/N_new:>4.1f}%) {sign}{abs(change)}")
        
        # Attack classes and compute class weights
        pos_weights = []
        for j, name in enumerate(label_names):
            cnt_train = int(Y[:, j].sum().item())
            cnt_new = int(Yb[:, j].sum().item())
            if cnt_train > 0 or cnt_new > 0:
                change = cnt_new - cnt_train
                sign = "++" if change > 0 else "--" if change < 0 else "  "
                print(f"{name:<20} {cnt_train:>5} ({100.0*cnt_train/N_train:>4.1f}%) {cnt_new:>5} ({100.0*cnt_new/N_new:>4.1f}%) {sign}{abs(change)}")
            
            # Balanced weights
            pos_count = max(cnt_new, 1)
            neg_count = max(N_new - pos_count, 1)
            pos_weight = min(neg_count / pos_count, 50.0)
            pos_weights.append(pos_weight)
        
        bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weights, device=device))

    # Train
    best_val_loss = float('inf')
    torch.set_float32_matmul_precision('high') if hasattr(torch, "set_float32_matmul_precision") else None

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for cls_tokens, mean_pooling, max_pooling, attn, targets in train_loader:
            cls_tokens = cls_tokens.to(device, non_blocking=True)
            mean_pooling = mean_pooling.to(device, non_blocking=True)
            max_pooling = max_pooling.to(device, non_blocking=True)
            attn = attn.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                recon, z, outputs = model(cls_tokens, mean_pooling, max_pooling, attn)
                feats = torch.cat([cls_tokens, mean_pooling, max_pooling, attn], dim=1)
                recon_loss = mse_loss(recon, feats)
                if isinstance(outputs, dict):
                    logits = torch.cat([v.view(v.size(0), -1) for v in outputs.values()], dim=1)
                else:
                    logits = outputs  # assume (B, L)
                ml_loss = bce_loss(logits, targets)
                hier_loss = model.hierarchy_consistency_loss(outputs)
                
                # Balanced loss scaling to prevent NaN
                total_loss = 0.1 * recon_loss + ml_loss + 0.1 * hier_loss
                
                # Check for NaN/inf and skip if found
                if not torch.isfinite(total_loss):
                    print(f"Warning: Non-finite loss detected, skipping batch")
                    continue

            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                # More aggressive gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                if torch.isfinite(grad_norm):
                    scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                if torch.isfinite(grad_norm):
                    optimizer.step()

            train_losses.append(total_loss.item())

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for cls_tokens, mean_pooling, max_pooling, attn, targets in val_loader:
                cls_tokens = cls_tokens.to(device, non_blocking=True)
                mean_pooling = mean_pooling.to(device, non_blocking=True)
                max_pooling = max_pooling.to(device, non_blocking=True)
                attn = attn.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    recon, z, outputs = model(cls_tokens, mean_pooling, max_pooling, attn)
                    feats = torch.cat([cls_tokens, mean_pooling, max_pooling, attn], dim=1)
                    recon_loss = mse_loss(recon, feats)
                    if isinstance(outputs, dict):
                        logits = torch.cat([v.view(v.size(0), -1) for v in outputs.values()], dim=1)
                    else:
                        logits = outputs
                    ml_loss = bce_loss(logits, targets)
                    hier_loss = model.hierarchy_consistency_loss(outputs)
                    total_loss = 0.1 * recon_loss + ml_loss + 0.1 * hier_loss
                    
                    # Skip if non-finite
                    if torch.isfinite(total_loss):
                        val_losses.append(total_loss.item())

        avg_train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        avg_val_loss = float(np.mean(val_losses)) if val_losses else avg_train_loss
        
        # Progress checkpoints every 5% [[memory:4887036]]
        if (epoch + 1) % max(1, epochs // 20) == 0:
            spinner.text = f"Epoch {epoch+1}/{epochs} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}"
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

    spinner.stop_and_persist(text="Training completed")
    return model

def evaluate_hierarchical(model, test_loader):
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

def main():
    # load data
    embeddings, labels = load_logbert_embeddings()
    datasets = load_datasets(embeddings, labels)
    
    for log_type, data in datasets.items():
        print(f"\n{'='*50}")
        print(f"Processing {log_type}")
        print('='*50)
        
        # split dataset
        dataset = data['loader'].dataset
        n_total = len(dataset)
        n_val = int(0.1 * n_total)
        n_test = int(0.1 * n_total)
        n_train = n_total - n_val - n_test
        
        train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test])
        train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=128, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=128, shuffle=False)
        
        print(f"Train: {n_train}, Val: {n_val}, Test: {n_test}")
        
        # initialize and train model
        model = HierarchicalTransformer(hierarchy).to(device)
        model = train_model(model, train_loader, val_loader, epochs=10)
        
        # evaluate
        evaluate_hierarchical(model, test_loader)
        
        # save model
        Path("models").mkdir(exist_ok=True)
        torch.save(model.state_dict(), f"models/hierarchical_{log_type}.pth")
        print(f"\nModel saved to models/hierarchical_{log_type}.pth")

if __name__ == "__main__":
    main()