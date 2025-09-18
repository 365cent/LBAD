import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import halo
import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.metrics import precision_recall_fscore_support, jaccard_score
from sklearn.preprocessing import StandardScaler
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from imblearn.over_sampling import BorderlineSMOTE, KMeansSMOTE
from imblearn.under_sampling import RandomUnderSampler

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

if device.type == "mps":
    torch.backends.mps.allow_tf32 = True
    print("Enabled MPS optimizations for Silicon GPU")
elif device.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high") 
    print("Enabled CUDA optimizations for NVIDIA GPU")
    torch.backends.cudnn.benchmark = True

def safe_load(path):
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except (EOFError, pickle.PickleError, Exception) as e:
        print(f"✗ {Path(path).name}: {e}")
        return None

hierarchy = {
    "foothold": {"attacker_http": ["dirb", "webshell_cmd", "webshell_upload"]},
    "escalate": {"escalated_command": ["escalated_sudo_session"], "attacker_change_user": [], "reverse_shell": []},
    "attacker_vpn": {},
    "dnsteal": {"dnsteal-received": [], "dnsteal-dropped": [], "exfiltration-service": []}
}


@dataclass(frozen=True)
class TransformerConfig:
    """Configuration for the hierarchical single-layer transformer."""

    hidden_dim: int = 384
    bottleneck_dim: int = 192
    num_heads: int = 4
    dropout: float = 0.1
    attn_input_dim: int = 10

class HierarchicalTransformer(nn.Module):
    """Hierarchical transformer with single-layer encoder and DP smoothing."""

    def __init__(
        self,
        hierarchy: Mapping[str, Mapping[str, Sequence[str]]],
        config: Optional[TransformerConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config or TransformerConfig()
        self.hierarchy = hierarchy
        self.hidden_dim = self.config.hidden_dim
        self.reconstruction_dim = 768 * 3 + self.config.attn_input_dim

        def _build_proj(input_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(input_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.GELU(),
                nn.Dropout(self.config.dropout),
            )

        self.cls_proj = _build_proj(768)
        self.mean_proj = _build_proj(768)
        self.max_proj = _build_proj(768)
        self.attn_proj = _build_proj(self.config.attn_input_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.config.num_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=self.config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.sequence_dropout = nn.Dropout(self.config.dropout)
        self.post_encoder_norm = nn.LayerNorm(self.hidden_dim)

        self.bottleneck = nn.Linear(self.hidden_dim, self.config.bottleneck_dim)
        self.decoder = nn.Sequential(
            nn.Linear(self.config.bottleneck_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.hidden_dim, self.reconstruction_dim),
        )

        self.heads = nn.ModuleDict()
        self.children_map: Dict[str, List[str]] = {}
        self.parent_map: Dict[str, str] = {}
        self.node_to_index: Dict[str, int] = {}
        self.root_nodes: List[str] = []
        self.topological_order: List[str] = []
        self._build_heads(self.hierarchy)
        self._reverse_topological_order = list(reversed(self.topological_order))

    @property
    def device(self) -> torch.device:
        """Returns the primary device for the model."""

        return next(self.parameters()).device

    def _register_node(self, name: str, parent: Optional[str]) -> None:
        """Registers a node in the hierarchy and creates a linear head if needed."""

        if name not in self.heads:
            self.heads[name] = nn.Linear(self.config.bottleneck_dim, 1)
        if name not in self.node_to_index:
            self.node_to_index[name] = len(self.node_to_index)
            self.topological_order.append(name)
        if parent is None:
            if name not in self.root_nodes:
                self.root_nodes.append(name)
            return

        children = self.children_map.setdefault(parent, [])
        if name not in children:
            children.append(name)
        self.parent_map[name] = parent

    def _build_heads(
        self,
        tree: Mapping[str, Mapping[str, Sequence[str]]],
        parent: Optional[str] = None,
    ) -> None:
        """Recursively builds linear heads for each hierarchical node."""

        for node, children in tree.items():
            self._register_node(node, parent)
            if isinstance(children, Mapping):
                self._build_heads(children, node)
            elif isinstance(children, (list, tuple, set)):
                for child in children:
                    self._register_node(str(child), node)
            elif children is None:
                continue
            else:
                raise TypeError(f"Unsupported hierarchy type for node '{node}': {type(children)}")

    def forward(
        self,
        cls_tokens: torch.Tensor,
        mean_pooling: torch.Tensor,
        max_pooling: torch.Tensor,
        attn: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Runs a forward pass and returns reconstruction, latent, and logits."""

        if attn.shape[-1] != self.config.attn_input_dim:
            raise ValueError(
                f"Attention features expected dim {self.config.attn_input_dim}, got {attn.shape[-1]}"
            )

        cls_h = self.cls_proj(cls_tokens)
        mean_h = self.mean_proj(mean_pooling)
        max_h = self.max_proj(max_pooling)
        attn_h = self.attn_proj(attn)

        combined = torch.stack([cls_h, mean_h, max_h, attn_h], dim=1)
        combined = self.sequence_dropout(combined)
        encoded = self.encoder(combined)
        pooled = encoded.mean(dim=1)
        pooled = self.post_encoder_norm(pooled)

        z = F.gelu(self.bottleneck(pooled))
        recon = self.decoder(z)

        outputs = {name: torch.clamp(head(z), -10.0, 10.0) for name, head in self.heads.items()}
        return recon, z, outputs

    def _dp_probabilities(
        self, outputs: Mapping[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Returns raw and hierarchy-consistent probabilities via DP smoothing."""

        if not outputs:
            return {}, {}

        base_probs: Dict[str, torch.Tensor] = {}
        for name, logits in outputs.items():
            base_probs[name] = torch.sigmoid(torch.clamp(logits, -10.0, 10.0))

        aggregated: Dict[str, torch.Tensor] = {}
        for name in self._reverse_topological_order:
            if name not in base_probs:
                continue
            prob = base_probs[name]
            children = self.children_map.get(name, [])
            if children:
                child_probs = [aggregated[child] for child in children if child in aggregated]
                if child_probs:
                    child_stack = torch.stack(child_probs, dim=0)
                    child_summary = child_stack.max(dim=0).values
                    prob = torch.maximum(prob, child_summary)
            aggregated[name] = prob

        for name, prob in base_probs.items():
            aggregated.setdefault(name, prob)

        return base_probs, aggregated

    def hierarchy_consistency_loss(self, outputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Encourages logits to respect the hierarchy using DP smoothing."""

        if not outputs:
            return torch.tensor(0.0, device=self.device)

        base_probs, aggregated = self._dp_probabilities(outputs)
        losses = []
        for name, base in base_probs.items():
            target = aggregated[name]
            losses.append(F.smooth_l1_loss(base, target, beta=0.05, reduction="mean"))
        return torch.stack(losses).mean()

    def propagate_labels(self, outputs: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Returns hierarchy-consistent probabilities for inference."""

        _, aggregated = self._dp_probabilities(outputs)
        return aggregated

def load_logbert_embeddings(target_log_type: str = None):
    embeddings_dir = Path("embeddings/logbert")
    if not embeddings_dir.exists():
        raise FileNotFoundError(f"Embeddings not found: {embeddings_dir}")
    
    embeddings, labels, skipped = {}, {}, []
    
    if target_log_type:
        embed_dir = embeddings_dir / target_log_type
        log_pkl, label_pkl = embed_dir / f"log_{target_log_type}.pkl", embed_dir / f"label_{target_log_type}.pkl"
        if not (log_pkl.exists() and label_pkl.exists()):
            raise FileNotFoundError(f"Embeddings for {target_log_type} not found in {embeddings_dir}")
        print(f"Loading {target_log_type}...", end=" ")
        log_data = safe_load(log_pkl)
        label_data = safe_load(label_pkl)
        if log_data is None or label_data is None:
            raise ValueError(f"Failed to load embeddings for {target_log_type}")
        embeddings[target_log_type] = log_data
        labels[target_log_type] = label_data
        print("✓")
    else:
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

def load_fasttext_embeddings(target_log_type: str = None):
    embeddings_dir = Path("embeddings/fasttext")
    if not embeddings_dir.exists():
        raise FileNotFoundError(f"Embeddings not found: {embeddings_dir}")
    
    embeddings, labels, skipped = {}, {}, []
    
    if target_log_type:
        embed_dir = embeddings_dir / target_log_type
        log_pkl, label_pkl = embed_dir / f"log_{target_log_type}.pkl", embed_dir / f"label_{target_log_type}.pkl"
        if not (log_pkl.exists() and label_pkl.exists()):
            raise FileNotFoundError(f"Embeddings for {target_log_type} not found in {embeddings_dir}")
        print(f"Loading {target_log_type}...", end=" ")
        log_data = safe_load(log_pkl)
        label_data = safe_load(label_pkl)
        if log_data is None or label_data is None:
            raise ValueError(f"Failed to load embeddings for {target_log_type}")
        embeddings[target_log_type] = log_data
        labels[target_log_type] = label_data
        print("✓")
    else:
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

def load_word2vec_embeddings(target_log_type: str = None):
    embeddings_dir = Path("embeddings/word2vec")
    if not embeddings_dir.exists():
        raise FileNotFoundError(f"Embeddings not found: {embeddings_dir}")
    
    embeddings, labels, skipped = {}, {}, []
    if target_log_type:
        embed_dir = embeddings_dir / target_log_type
        log_pkl, label_pkl = embed_dir / f"log_{target_log_type}.pkl", embed_dir / f"label_{target_log_type}.pkl"
        if not (log_pkl.exists() and label_pkl.exists()):
            raise FileNotFoundError(f"Embeddings for {target_log_type} not found in {embeddings_dir}")
        print(f"Loading {target_log_type}...", end=" ")
        log_data = safe_load(log_pkl)
        label_data = safe_load(label_pkl)
        if log_data is None or label_data is None:
            raise ValueError(f"Failed to load embeddings for {target_log_type}")
        embeddings[target_log_type] = log_data
        labels[target_log_type] = label_data
        print("✓")
    else:
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
    """Yields each node in the hierarchy in a flat traversal."""
    for node, children in h.items():
        yield node
        if isinstance(children, dict):
            for child, leaves in children.items():
                yield child
                for leaf in leaves:
                    yield leaf

def create_multilabel_targets(labels_dict, hierarchy):
    """Aligns label vectors with the flattened hierarchy layout."""
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
        
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=(device.type=="cuda"), persistent_workers=True, prefetch_factor=4)
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
    """Performs lightweight SMOTE/undersampling to mitigate class imbalance."""
    spinner = halo.Halo(text="Applying balanced resampling", spinner="dots")
    spinner.start()

    try:
        # Extract training data
        all_data = [[], [], [], [], []]
        for batch in train_loader:
            for i, tensor in enumerate(batch):
                all_data[i].append(tensor.detach().cpu().numpy())
        cls, mean, maxp, attn, y = [np.vstack(data) for data in all_data]

        if int(y.sum()) == 0:
            spinner.stop_and_persist(text="Skipping resampling (only normal samples detected)")
            return train_loader

        # Safety cap: limit rows used for resampling to avoid OOM/very long runs
        n_total = y.shape[0]
        max_rows = 400_000 if device.type == "cuda" else 200_000
        if n_total > max_rows:
            rng = np.random.default_rng(42)
            idx = rng.choice(n_total, size=max_rows, replace=False)
            cls, mean, maxp, attn, y = cls[idx], mean[idx], maxp[idx], attn[idx], y[idx]
        
        # Preprocessing
        scalers = [StandardScaler().fit(data) for data in [cls, mean, maxp, attn]]
        scaled = [scaler.transform(data) for scaler, data in zip(scalers, [cls, mean, maxp, attn])]
        
        # Dimensionality reduction for efficiency
        pca_dims = [min(48, scaled[0].shape[1]), min(48, scaled[1].shape[1]), min(48, scaled[2].shape[1]), min(8,  scaled[3].shape[1])]  # replace
        pcas = [PCA(n_components=dim, svd_solver='auto').fit(data) 
                for dim, data in zip(pca_dims, scaled)]
        X_reduced = np.hstack([pca.transform(data) for pca, data in zip(pcas, scaled)])

        # Process each class
        X_final, y_final = [], []
        
        for c in range(y.shape[1]):
            pos_mask = y[:, c] == 1
            pos_count = int(pos_mask.sum())
            
            if pos_count == 0:
                continue
                
            neg_mask = y[:, c] == 0
            neg_count = int(neg_mask.sum())
            
            # Determine target counts for balance
            total_samples = pos_count + neg_count
            target_per_class = int(np.sqrt(total_samples * pos_count))
            target_per_class = max(target_per_class, pos_count)  # Never undersample minority
            
            # Prepare class-specific data
            # Subsample negatives to keep class-specific processing bounded
            rng = np.random.default_rng(42 + c)
            pos_idx = np.where(pos_mask)[0]
            neg_idx = np.where(neg_mask)[0]
            neg_keep = min(len(neg_idx), max(3 * len(pos_idx), 10000))
            if neg_keep > 0:
                neg_idx_sampled = rng.choice(neg_idx, size=neg_keep, replace=False)
                class_idx = np.concatenate([pos_idx, neg_idx_sampled])
            else:
                class_idx = pos_idx
            X_class = X_reduced[class_idx]
            y_class = y[class_idx, c].astype(int)
            
            # Apply resampling pipeline
            try:
                # Determine strategy based on class balance
                if pos_count <= 1:
                    X_resampled, y_resampled = X_class, y_class
                elif pos_count < target_per_class:
                    k_neighbors = min(3, pos_count - 1)
                    if pos_count < 5:
                        smote = KMeansSMOTE(
                            sampling_strategy="auto",
                            random_state=42,
                            k_neighbors=max(1, k_neighbors),
                            cluster_balance_threshold=0.1,
                        )
                    else:
                        oversample_ratio = min(target_per_class / max(pos_count, 1), 1.5)
                        smote = BorderlineSMOTE(
                            sampling_strategy={
                                1: min(int(pos_count * oversample_ratio), pos_count + 20000)
                            },
                            k_neighbors=max(2, k_neighbors),
                            random_state=42,
                            kind="borderline-1",
                        )
                    X_resampled, y_resampled = smote.fit_resample(X_class, y_class)
                else:
                    X_resampled, y_resampled = X_class, y_class
                
                # Undersample majority class if needed
                if (y_resampled == 0).sum() > target_per_class:
                    undersampler = RandomUnderSampler(
                        sampling_strategy={0: target_per_class},
                        random_state=42
                    )
                    X_resampled, y_resampled = undersampler.fit_resample(X_resampled, y_resampled)
                
                # Store resampled data for this class
                y_class_matrix = np.zeros((len(y_resampled), y.shape[1]))
                y_class_matrix[:, c] = y_resampled
                
                X_final.append(X_resampled)
                y_final.append(y_class_matrix)
                
            except Exception:
                # Fallback: use original data
                y_class_matrix = np.zeros((len(y_class), y.shape[1]))
                y_class_matrix[:, c] = y_class
                X_final.append(X_class)
                y_final.append(y_class_matrix)

        if not X_final:
            return train_loader

        # Combine all resampled classes (skip dedup to save RAM) and optionally cap size
        X_final = np.vstack(X_final)
        y_final = np.vstack(y_final)
        max_out = 1_000_000 if device.type == "cuda" else 500_000
        if len(X_final) > max_out:
            rng = np.random.default_rng(123)
            keep = rng.choice(len(X_final), size=max_out, replace=False)
            X_final = X_final[keep]
            y_final = y_final[keep]
        
        # Inverse transform to original feature space
        splits = np.cumsum([pca.n_components_ for pca in pcas[:-1]])
        x_blocks = np.split(X_final, splits, axis=1)
        # Batched inverse-transform to limit peak memory
        def _inv(pca, scaler, block, bs=200000):
            out = []
            for i in range(0, len(block), bs):
                out.append(scaler.inverse_transform(pca.inverse_transform(block[i:i+bs])))
            return np.vstack(out)
        final_features = [_inv(p, s, b, bs=100_000 if device.type=="cuda" else 50_000) for (p, s, b) in zip(pcas, scalers, x_blocks)]
        
        # Create balanced dataset
        dataset = TensorDataset(*[torch.from_numpy(data).float() for data in final_features + [y_final]])
        
        print(f"\nBalanced dataset: {len(dataset)} samples")
        anomaly_count = (torch.from_numpy(y_final).sum(dim=1) > 0).sum().item()
        print(f"  normal: {len(dataset) - anomaly_count} ({(len(dataset) - anomaly_count)/len(dataset):.2%})")
        print(f"  anomalies: {anomaly_count} ({anomaly_count/len(dataset):.2%})")

        return DataLoader(dataset, batch_size=getattr(train_loader, 'batch_size', 128), shuffle=True, num_workers=8, pin_memory=(device.type=="cuda"), persistent_workers=True, prefetch_factor=4)

    except Exception as e:
        print(f"Resampling failed: {e}")
        return train_loader
    finally:
        try:
            spinner.stop_and_persist(text="Resampling completed")
        except:
            pass

def train_model(model, train_loader, val_loader, epochs=10, lambda_recon=0.05, lambda_hier=0.02):
    """Trains the transformer with reconstruction and hierarchy regularizers."""
    train = smote_data(train_loader, val_loader)
    if len(train) == 0:
        train = train_loader
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4, fused=(device.type=="cuda"))
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

            with torch.cuda.amp.autocast(enabled=(device.type=="cuda")):
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

                total_loss = lambda_recon * recon_loss + ml_loss + lambda_hier * hier_loss

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
                # Use CUDA autocast only when CUDA; avoid triton/inductor path on CPU/MPS
                use_cuda_amp = (device.type=="cuda")
                ctx = torch.cuda.amp.autocast(enabled=use_cuda_amp)
                with ctx:
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

                    total_loss = lambda_recon * recon_loss + ml_loss + lambda_hier * hier_loss

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

def evaluate_model(model, test_loader, log_type, embedding_type=""):
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
    
    # Include embedding type in filename if specified
    embedding_suffix = f"_{embedding_type}" if embedding_type else ""
    report_path = results_dir / f"hierarchical_{log_type}{embedding_suffix}_evaluation_{timestamp}.txt"
    
    with open(report_path, 'w') as f:
        f.write(f"Hierarchical Transformer Evaluation Report\n")
        f.write(f"Log Type: {log_type}\n")
        if embedding_type:
            f.write(f"Embedding Type: {embedding_type}\n")
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
    
    parser = argparse.ArgumentParser(description='Train hierarchical transformer on log embeddings')
    parser.add_argument('--embedding-type', type=str, default='all', 
                       choices=['all', 'logbert', 'fasttext', 'word2vec'],
                       help='Type of embeddings to use (default: all)')
    parser.add_argument('--log-type', type=str, default=None,
                       help='Specific log type to process (processes all if not specified)')
    parser.add_argument('--sample-size', type=int, default=None,
                       help='Subsample size for processing (processes full dataset if not specified)')
    
    args = parser.parse_args()
    
    # Define embedding loading order
    embedding_types = ['fasttext', 'word2vec', 'logbert'] if args.embedding_type == 'all' else [args.embedding_type]
    
    for embedding_type in embedding_types:
        print(f"\n{'='*60}")
        print(f"Processing with {embedding_type} embeddings")
        print('='*60)
        
        # Load embeddings based on type
        try:
            if embedding_type == 'logbert':
                embeddings, labels = load_logbert_embeddings(args.log_type)
            elif embedding_type == 'fasttext':
                embeddings, labels = load_fasttext_embeddings(args.log_type)
            elif embedding_type == 'word2vec':
                embeddings, labels = load_word2vec_embeddings(args.log_type)
            else:
                print(f"Unsupported embedding type: {embedding_type}, skipping...")
                continue
        except Exception as e:
            print(f"Failed to load {embedding_type} embeddings: {e}")
            continue
        
        # If a specific log type was requested but not found, skip
        if args.log_type and args.log_type not in embeddings:
            print(f"Log type '{args.log_type}' not found in {embedding_type} embeddings: {list(embeddings.keys())}")
            continue
        
        datasets = load_datasets(embeddings, labels)
        
        for log_type, data in datasets.items():
            print(f"\n{'='*50}")
            print(f"Processing {log_type} with {embedding_type} embeddings")
            print('='*50)
            
            dataset = data['loader'].dataset
            
            # Apply subsampling if specified
            if args.sample_size and args.sample_size < len(dataset):
                print(f"Subsampling {args.sample_size} samples from {len(dataset)} total samples")
                indices = torch.randperm(len(dataset))[:args.sample_size]
                dataset = torch.utils.data.Subset(dataset, indices)
            
            all_targets = torch.stack([dataset[i][4] for i in range(len(dataset))])
            anomaly_mask = (all_targets.sum(dim=1) > 0)

            normal_idx = torch.nonzero(~anomaly_mask, as_tuple=True)[0]
            anomaly_idx = torch.nonzero(anomaly_mask, as_tuple=True)[0]

            def stratified_split(indices: torch.Tensor, train_ratio: float = 0.8, val_ratio: float = 0.1):
                n = len(indices)
                if n == 0:
                    empty = torch.tensor([], dtype=torch.long)
                    return empty, empty, empty

                n_train = int(train_ratio * n)
                n_val = int(val_ratio * n)

                n_train = max(1, min(n_train, n - 2)) if n > 2 else n_train
                n_val = max(1, min(n_val, n - n_train - 1)) if n > n_train + 1 else n_val

                perm = torch.randperm(n)
                train_split = indices[perm[:n_train]]
                val_split = indices[perm[n_train:n_train + n_val]]
                test_split = indices[perm[n_train + n_val:]]
                return train_split, val_split, test_split

            def concat_indices(chunks: List[torch.Tensor]) -> torch.Tensor:
                filtered = [chunk for chunk in chunks if len(chunk) > 0]
                if not filtered:
                    return torch.tensor([], dtype=torch.long)
                return torch.cat(filtered)

            normal_train, normal_val, normal_test = stratified_split(normal_idx)
            _, anomaly_val, anomaly_test = stratified_split(anomaly_idx)

            if len(normal_train) == 0:
                print("Insufficient normal samples for training; defaulting to all normal logs.")
                normal_train = normal_idx

            train_indices = normal_train
            val_indices = concat_indices([normal_val, anomaly_val])
            test_indices = concat_indices([normal_test, anomaly_test])

            if len(train_indices) == 0:
                raise ValueError("Dataset does not contain normal samples for unsupervised training.")

            if len(val_indices) == 0 and len(train_indices) > 1:
                val_indices = train_indices[: max(1, len(train_indices) // 10)]

            if len(test_indices) == 0:
                test_indices = normal_idx if len(normal_idx) > 0 else anomaly_idx

            train_set = torch.utils.data.Subset(dataset, train_indices)
            val_set = torch.utils.data.Subset(dataset, val_indices)
            test_set = torch.utils.data.Subset(dataset, test_indices)
            train_loader = DataLoader(train_set, batch_size=128, shuffle=True,  num_workers=8, pin_memory=(device.type=="cuda"), persistent_workers=True, prefetch_factor=4)
            val_loader   = DataLoader(val_set,   batch_size=128, shuffle=False, num_workers=8, pin_memory=(device.type=="cuda"), persistent_workers=True, prefetch_factor=4)
            test_loader  = DataLoader(test_set,  batch_size=128, shuffle=False, num_workers=8, pin_memory=(device.type=="cuda"), persistent_workers=True, prefetch_factor=4)
            
            print(f"Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")

            model = HierarchicalTransformer(hierarchy).to(device)
            if device.type == "cuda":
                try:
                    import triton  # type: ignore  # noqa: F401
                    model = torch.compile(model, mode="max-autotune")
                except Exception:
                    print("Triton not available or compile failed; using eager mode.")
            model = train_model(model, train_loader, val_loader, epochs=10)

            evaluate_model(model, test_loader, log_type, embedding_type)

            Path("models").mkdir(exist_ok=True)
            model_path = f"models/hierarchical_{log_type}_{embedding_type}.pth"
            torch.save(model.state_dict(), model_path)
            print(f"\nModel saved to {model_path}")

if __name__ == "__main__":
    main()
