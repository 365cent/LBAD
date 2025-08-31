import pickle
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.metrics import precision_recall_fscore_support, jaccard_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import KMeansSMOTE
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import halo

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

if device.type == "mps":
    torch.backends.mps.allow_tf32 = True
    print("Enabled MPS optimizations for Silicon GPU")
else if device.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high") 
    print("Enabled CUDA optimizations for NVIDIA GPU")

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

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.bottleneck = nn.Linear(hidden_dim, hidden_dim // 2)
        self.decoder = nn.Sequential(nn.Linear(hidden_dim // 2, hidden_dim), nn.ReLU(),
                                     nn.Linear(hidden_dim, 2314))

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
        outputs = {name: torch.clamp(head(z), -10.0, 10.0) for name, head in self.heads.items()}
        return recon, z, outputs

    def hierarchy_consistency_loss(self, outputs):
        """Penalty if child probability > parent probability"""
        consistency_loss = torch.tensor(0.0, device=next(iter(outputs.values())).device)
        count = 0

        for parent, children in self.parent_child_map.items():
            if parent in outputs:
                # Clamp logits before sigmoid for numerical stability
                parent_logits = torch.clamp(outputs[parent], -10.0, 10.0)
                parent_prob = torch.sigmoid(parent_logits)

                for child in children:
                    if child in outputs:
                        child_logits = torch.clamp(outputs[child], -10.0, 10.0)
                        child_prob = torch.sigmoid(child_logits)
                        violation = F.relu(child_prob - parent_prob)
                        consistency_loss += violation.mean()
                        count += 1

        # Return average if we have any violations, otherwise return 0
        return consistency_loss / max(count, 1)

    def propagate_labels(self, outputs):
        """Propagate active parent labels to children"""
        propagated = {}
        for name, logits in outputs.items():
            propagated[name] = torch.sigmoid(logits)
        
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

def smote_data(train_loader, val_loader):
    # Apply SMOTE to highly imbalanced embeddings with block-aware preprocessing
    target_ratio = 0.2          # target minority:majority after oversampling (per class)
    max_neg_per_pos = 20        # negatives kept per positive in the per-class subset
    min_pos = 5                # skip classes with fewer positives than this
    pca_dims = (128, 128, 128, 10)  # CLS / MEAN / MAX / ATTN
    random_state = 42

    spinner = halo.Halo(text="Applying SMOTE", spinner="dots")
    spinner.start()
    msg = "SMOTE completed"

    try:
        # Extract all training data
        all_cls, all_mean, all_max, all_attn, all_targets = [], [], [], [], []
        for cls_b, mean_b, max_b, attn_b, y_b in train_loader:
            all_cls.append(cls_b.detach().cpu().numpy())
            all_mean.append(mean_b.detach().cpu().numpy())
            all_max.append(max_b.detach().cpu().numpy())
            all_attn.append(attn_b.detach().cpu().numpy())
            all_targets.append(y_b.detach().cpu().numpy())

        # Stack into arrays
        cls = np.vstack(all_cls)
        mean = np.vstack(all_mean)
        maxp = np.vstack(all_max)
        attn = np.vstack(all_attn)
        y = np.vstack(all_targets)  # shape [N, L]
        N, L = y.shape

        # Standardize each component
        scaler_cls = StandardScaler().fit(cls)
        scaler_mean = StandardScaler().fit(mean)
        scaler_max = StandardScaler().fit(maxp)
        scaler_attn = StandardScaler().fit(attn)

        cls_s = scaler_cls.transform(cls)
        mean_s = scaler_mean.transform(mean)
        max_s = scaler_max.transform(maxp)
        attn_s = scaler_attn.transform(attn)

        # Equalize block energy (so ATTN is not drowned by 768D blocks)
        def eq(A): return A * (1.0 / np.sqrt(A.shape[1]))
        cls_eq, mean_eq, max_eq, attn_eq = map(eq, (cls_s, mean_s, max_s, attn_s))

        # PCA with reasonable dimensions
        p_cls, p_mean, p_max, p_attn = pca_dims
        pca_cls  = PCA(n_components=min(p_cls,  cls_eq.shape[1]),  svd_solver='auto').fit(cls_eq)
        pca_mean = PCA(n_components=min(p_mean, mean_eq.shape[1]), svd_solver='auto').fit(mean_eq)
        pca_max  = PCA(n_components=min(p_max,  max_eq.shape[1]),  svd_solver='auto').fit(max_eq)
        pca_attn = PCA(n_components=min(p_attn, attn_eq.shape[1]), svd_solver='auto').fit(attn_eq)

        # Combine PCA features
        Z_cls  = pca_cls.transform(cls_eq)
        Z_mean = pca_mean.transform(mean_eq)
        Z_max  = pca_max.transform(max_eq)
        Z_attn = pca_attn.transform(attn_eq)
        Xr = np.hstack([Z_cls, Z_mean, Z_max, Z_attn])  # reduced space

        # Oversample per class with caps
        rng = np.random.default_rng(random_state)
        synth_Z, synth_Y = [], []

        for c in range(L):
            # Build one-vs-rest vector
            y_bin = y[:, c].astype(int)

            # Skip tiny classes
            pos_idx = np.flatnonzero(y_bin == 1)
            if len(pos_idx) < min_pos:
                continue

            # Limit negatives to bound runtime
            neg_idx = np.flatnonzero(y_bin == 0)
            k_neg = min(len(neg_idx), max_neg_per_pos * len(pos_idx))
            if k_neg == 0:
                continue
            neg_keep = rng.choice(neg_idx, size=k_neg, replace=False)

            # Subset features and labels
            sub_idx = np.concatenate([pos_idx, neg_keep])
            X_sub = Xr[sub_idx]
            y_sub = y_bin[sub_idx]

            # Adjust k_neighbors based on positives
            k_neighbors = max(2, min(5, len(pos_idx) - 1))

            # Run KMeansSMOTE with target ratio (does not fully balance)
            try:
                smote = KMeansSMOTE(
                    sampling_strategy=target_ratio,
                    k_neighbors=k_neighbors,
                    random_state=random_state,
                    cluster_balance_threshold=0.1,
                    kmeans_estimator=25
                )
                X_res, y_res = smote.fit_resample(X_sub, y_sub)

                # Take only the newly created positives (appended at the end)
                n_pos_before = int(y_sub.sum())
                n_pos_after = int(y_res.sum())
                n_new = n_pos_after - n_pos_before
                if n_new > 0:
                    pos_rows = np.flatnonzero(y_res == 1)
                    new_pos_rows = pos_rows[-n_new:]
                    Z_new = X_res[new_pos_rows]

                    # Reconstruct full target rows (one-hot for this class)
                    Y_new = np.zeros((n_new, L), dtype=y.dtype)
                    Y_new[:, c] = 1

                    synth_Z.append(Z_new)
                    synth_Y.append(Y_new)
            except Exception as e:
                print(f"SMOTE failed for class {c}: {e}")
                continue

        # No augmentation case
        if not synth_Z:
            msg = "SMOTE skipped (no eligible classes)"
            return train_loader

        Z_syn = np.vstack(synth_Z)
        Y_syn = np.vstack(synth_Y)

        # Split PCA features back
        d1, d2, d3 = Z_cls.shape[1], Z_mean.shape[1], Z_max.shape[1]
        zc  = Z_syn[:, :d1]
        zm  = Z_syn[:, d1:d1+d2]
        zmx = Z_syn[:, d1+d2:d1+d2+d3]
        za  = Z_syn[:, d1+d2+d3:]

        # Inverse PCA transform (synthetic only)
        cls_rec  = pca_cls.inverse_transform(zc)
        mean_rec = pca_mean.inverse_transform(zm)
        max_rec  = pca_max.inverse_transform(zmx)
        attn_rec = pca_attn.inverse_transform(za)

        # Inverse scaling
        cls_final  = scaler_cls.inverse_transform(cls_rec)
        mean_final = scaler_mean.inverse_transform(mean_rec)
        max_final  = scaler_max.inverse_transform(max_rec)
        attn_final = scaler_attn.inverse_transform(attn_rec)

        # Create augmented arrays
        cls_aug  = np.vstack([cls,  cls_final])
        mean_aug = np.vstack([mean, mean_final])
        max_aug  = np.vstack([maxp, max_final])
        attn_aug = np.vstack([attn, attn_final])
        y_aug    = np.vstack([y,    Y_syn])

        # Create dataset
        batch_size = getattr(train_loader, 'batch_size', 128)
        aug_dataset = TensorDataset(
            torch.from_numpy(cls_aug).float(),
            torch.from_numpy(mean_aug).float(),
            torch.from_numpy(max_aug).float(),
            torch.from_numpy(attn_aug).float(),
            torch.from_numpy(y_aug).float()
        )

        # Print data distribution after SMOTE
        print(f"\nData distribution after SMOTE: {len(aug_dataset)} samples")
        y_aug_tensor = torch.from_numpy(y_aug)
        normal_aug = (y_aug_tensor.sum(dim=1) == 0).sum().item()
        print(f"  normal: {normal_aug} ({normal_aug/len(aug_dataset):.2%})")
        node_names = []
        def collect_names(hier):
            for node, children in hier.items():
                node_names.append(node)
                if isinstance(children, dict):
                    for child, leaves in children.items():
                        node_names.append(child)
                        node_names.extend(leaves)
        collect_names(hierarchy)
        for i, name in enumerate(node_names[:y_aug.shape[1]]):
            count = int(y_aug[:, i].sum())
            if count > 0:
                print(f"  {name}: {count} ({count/len(aug_dataset):.2%})")


        # Return new dataloader
        return DataLoader(aug_dataset, batch_size=batch_size, shuffle=True)

    except Exception as e:
        msg = f"SMOTE failed"
        print(f"[SMOTE] error: {e}")
        return train_loader
    finally:
        # Stop spinner
        try:
            spinner.stop_and_persist(text=msg)
        except Exception:
            pass

def train_model(model, train_loader, val_loader, epochs=10, lambda_recon=1.0, lambda_hier=0.5):
    train = smote_data(train_loader, val_loader)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)  # More stable learning rate
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-4, epochs=epochs, steps_per_epoch=len(train))
    mse_loss = nn.MSELoss()
    use_amp = device.type in ["cuda", "mps"]  # Enable for M2
    scaler = torch.amp.GradScaler(enabled=use_amp and device.type == "cuda")

    spinner = halo.Halo(text="Training hierarchical model", spinner="dots")
    spinner.start()

    # Get label names and compute class weights from original training data
    with torch.no_grad():
        c0, m0, x0, a0, _ = next(iter(train_loader))
        c0, m0, x0, a0 = c0.to(device), m0.to(device), x0.to(device), a0.to(device)
        _, _, outs0 = model(c0, m0, x0, a0)
        label_names = list(outs0.keys()) if isinstance(outs0, dict) else [f"label_{i}" for i in range(len(outs0))]
        all_targets = []
        for _, _, _, _, targets in train:  # Use SMOTE-augmented data for weights
            all_targets.append(targets.cpu())
        Y = torch.cat(all_targets)

        # Compute balanced class weights with proper bounds
        pos_weights = []
        for j in range(len(label_names)):
            cnt = int(Y[:, j].sum().item())
            if cnt > 0:
                # Balanced weight: inverse frequency with reasonable bounds
                weight = (Y.shape[0] - cnt) / max(cnt, 1)
                weight = min(max(weight, 1.0), 10.0)
            else:
                weight = 1.0  # Default weight for classes with no samples
            pos_weights.append(weight)

        # Use focal loss for better stability with imbalanced data
        class FocalBCEWithLogitsLoss(nn.BCEWithLogitsLoss):
            def __init__(self, alpha=1.0, gamma=2.0, **kwargs):
                super().__init__(**kwargs)
                self.alpha = alpha
                self.gamma = gamma

            def forward(self, input, target):
                bce_loss = super().forward(input, target)
                pt = torch.exp(-bce_loss)
                focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
                return focal_loss

        bce_loss = FocalBCEWithLogitsLoss(alpha=1.0, gamma=1.0, pos_weight=torch.tensor(pos_weights, device=device))

    # Initialize model weights properly
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    model.apply(init_weights)

    best_val_loss = float('inf')
    torch.set_float32_matmul_precision('high') if hasattr(torch, "set_float32_matmul_precision") else None

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for cls_tokens, mean_pooling, max_pooling, attn, targets in train:
            cls_tokens = cls_tokens.to(device, non_blocking=True)
            mean_pooling = mean_pooling.to(device, non_blocking=True)
            max_pooling = max_pooling.to(device, non_blocking=True)
            attn = attn.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                recon, z, outputs = model(cls_tokens, mean_pooling, max_pooling, attn)

                feats = torch.cat([cls_tokens, mean_pooling, max_pooling, attn], dim=1)
                recon_loss = mse_loss(recon, feats)
                recon_loss = torch.clamp(recon_loss, 0, 100.0)

                if isinstance(outputs, dict):
                    if outputs:
                        logits = torch.cat([v.view(v.size(0), -1) for v in outputs.values()], dim=1)
                    else:
                        # Fallback for empty outputs dict
                        logits = torch.zeros(targets.size(0), len(label_names), device=device)
                else:
                    logits = outputs

                logits = torch.clamp(logits, -10.0, 10.0)
                ml_loss = bce_loss(logits, targets)
                ml_loss = torch.clamp(ml_loss, 0, 100.0)

                hier_loss = model.hierarchy_consistency_loss(outputs)
                hier_loss = torch.clamp(hier_loss, 0, 10.0)

                total_loss = 0.001 * recon_loss + ml_loss + 0.01 * hier_loss

            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            scheduler.step()

            train_losses.append(total_loss.item())
        model.eval()
        val_losses = []
        with torch.no_grad():
            for cls_tokens, mean_pooling, max_pooling, attn, targets in val_loader:
                cls_tokens = cls_tokens.to(device, non_blocking=True)
                mean_pooling = mean_pooling.to(device, non_blocking=True)
                max_pooling = max_pooling.to(device, non_blocking=True)
                attn = attn.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    recon, z, outputs = model(cls_tokens, mean_pooling, max_pooling, attn)
                    feats = torch.cat([cls_tokens, mean_pooling, max_pooling, attn], dim=1)
                    recon_loss = mse_loss(recon, feats)

                    recon_loss = torch.clamp(recon_loss, 0, 100.0)

                    if isinstance(outputs, dict):
                        if outputs:
                            logits = torch.cat([v.view(v.size(0), -1) for v in outputs.values()], dim=1)
                        else:
                            # Fallback for empty outputs dict
                            logits = torch.zeros(targets.size(0), len(label_names), device=device)
                    else:
                        logits = outputs

                    logits = torch.clamp(logits, -10.0, 10.0)

                    ml_loss = bce_loss(logits, targets)
                    ml_loss = torch.clamp(ml_loss, 0, 100.0)

                    hier_loss = model.hierarchy_consistency_loss(outputs)
                    hier_loss = torch.clamp(hier_loss, 0, 10.0)

                    total_loss = 0.001 * recon_loss + ml_loss + 0.01 * hier_loss

                    val_losses.append(total_loss.item())

        avg_train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        avg_val_loss = float(np.mean(val_losses)) if val_losses else avg_train_loss
        
        # Progress checkpoints every 5% [[memory:4887036]]
        if (epoch + 1) % max(1, epochs // 20) == 0:
            spinner.text = f"Epoch {epoch+1}/{epochs} | Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f}"
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

    spinner.stop_and_persist(text="Training completed")
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
            propagated = model.propagate_labels(outputs)
            batch_preds = np.zeros((cls_tokens.shape[0], len(node_names)))
            for name, probs in propagated.items():
                if name in model.node_to_index:
                    idx = model.node_to_index[name]
                    batch_preds[:, idx] = (probs.cpu().numpy() > 0.5).astype(int).squeeze()
            
            all_preds.append(batch_preds)
            all_targets.append(targets.cpu().numpy())
    
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    
    print("\n=== Per-Class Metrics ===")
    print(f"{'Class':<25} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 55)

    class_metrics = []
    for i, name in enumerate(node_names):
        if i < all_targets.shape[1]:
            y_true = all_targets[:, i]
            y_pred = all_preds[:, i]

            if y_true.sum() > 0:
                prec, rec, f1, _ = precision_recall_fscore_support(
                    y_true, y_pred, average='binary', zero_division=0
                )
                class_metrics.append([prec, rec, f1])
                print(f"{name:<25} {prec:>10.3f} {rec:>10.3f} {f1:>10.3f}")
    
    print("\n=== Overall Metrics ===")

    micro_prec, micro_rec, micro_f1, _ = precision_recall_fscore_support(
        all_targets.ravel(), all_preds.ravel(), average='micro', zero_division=0
    )
    print(f"Micro-averaged: Precision={micro_prec:.3f}, Recall={micro_rec:.3f}, F1={micro_f1:.3f}")

    if class_metrics:
        macro_metrics = np.mean(class_metrics, axis=0)
        print(f"Macro-averaged: Precision={macro_metrics[0]:.3f}, Recall={macro_metrics[1]:.3f}, F1={macro_metrics[2]:.3f}")

    jaccard = jaccard_score(all_targets, all_preds, average='samples', zero_division=0)
    print(f"Jaccard Score (samples): {jaccard:.3f}")

    any_attack_true = (all_targets.sum(axis=1) > 0).astype(int)
    any_attack_pred = (all_preds.sum(axis=1) > 0).astype(int)
    
    if len(np.unique(any_attack_true)) > 1:
        anomaly_prec, anomaly_rec, anomaly_f1, _ = precision_recall_fscore_support(
            any_attack_true, any_attack_pred, average='binary', zero_division=0
        )
        print(f"\nAnomaly Detection: Precision={anomaly_prec:.3f}, Recall={anomaly_rec:.3f}, F1={anomaly_f1:.3f}")
    
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
    embeddings, labels = load_logbert_embeddings()
    datasets = load_datasets(embeddings, labels)
    
    for log_type, data in datasets.items():
        print(f"\n{'='*50}")
        print(f"Processing {log_type}")
        print('='*50)
        
        dataset = data['loader'].dataset
        all_targets = torch.stack([dataset[i][4] for i in range(len(dataset))])
        anomaly_mask = (all_targets.sum(dim=1) > 0)

        normal_idx = torch.nonzero(~anomaly_mask, as_tuple=True)[0]
        anomaly_idx = torch.nonzero(anomaly_mask, as_tuple=True)[0]

        def stratified_split(indices, train_ratio=0.8, val_ratio=0.1):
            n = len(indices)
            if n == 0:
                return torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)

            n_train = int(train_ratio * n)
            n_val = int(val_ratio * n)

            # Ensure we have at least 1 sample for each split if possible
            n_train = max(1, min(n_train, n - 2)) if n > 2 else n_train
            n_val = max(1, min(n_val, n - n_train - 1)) if n > n_train + 1 else n_val

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

        model = HierarchicalTransformer(hierarchy).to(device)
        model = train_model(model, train_loader, val_loader, epochs=10)

        evaluate_model(model, test_loader, log_type)

        Path("models").mkdir(exist_ok=True)
        torch.save(model.state_dict(), f"models/hierarchical_{log_type}.pth")
        print(f"\nModel saved to models/hierarchical_{log_type}.pth")

if __name__ == "__main__":
    main()