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
from imblearn.over_sampling import KMeansSMOTE, BorderlineSMOTE
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import halo
import argparse

# Windowing constants (no CLI changes)
STRIDE = 8

# Cost-sensitive penalty constants
LAMBDA_DOWN = 3.0      # penalty multiplier for severe false negatives
LAMBDA_INTRA = 1.3     # smaller weight for within-parent subclass mistakes
LAMBDA_HIER = 0.01     # consistency loss weight

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# Device-specific constants (defined after device)
# Use low-spec preset for MPS, full spec for CUDA
WINDOW_SIZE = 16 if device.type == "mps" else 32
ALPHA_RECON = 0.0 if device.type == "mps" else 0.001    # disable recon for MPS, enable for CUDA

# Low-spec model parameters for MPS
HIDDEN_DIM = 192 if device.type == "mps" else 512
NUM_HEADS = 3 if device.type == "mps" else 4
NUM_LAYERS = 1 if device.type == "mps" else 2
BATCH_SIZE = 32 if device.type == "mps" else 128
EPOCHS = 5 if device.type == "mps" else 10
NUM_WORKERS = 0 if device.type == "mps" else 8
PERSISTENT_WORKERS = False if device.type == "mps" else True
print(f"Using device: {device}")

if device.type == "mps":
    torch.backends.mps.allow_tf32 = True
    print("Enabled MPS optimizations for Silicon GPU")
    print(f"Low-spec preset: WINDOW_SIZE={WINDOW_SIZE}, HIDDEN_DIM={HIDDEN_DIM}, BATCH_SIZE={BATCH_SIZE}, EPOCHS={EPOCHS}")
elif device.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high") 
    print("Enabled CUDA optimizations for NVIDIA GPU")
    print(f"Full-spec preset: WINDOW_SIZE={WINDOW_SIZE}, HIDDEN_DIM={HIDDEN_DIM}, BATCH_SIZE={BATCH_SIZE}, EPOCHS={EPOCHS}")
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

# Top-level classes and severity mapping for cost-sensitive loss
TOP_LEVEL_CLASSES = ["normal", "foothold", "escalate", "attacker_vpn", "dnsteal"]
SEVERITY_MAP = {
    "normal": 0,
    "foothold": 1,
    "attacker_vpn": 2,
    "escalate": 3,
    "dnsteal": 3
}

class HierarchicalTransformer(nn.Module):
    def __init__(self, hierarchy, hidden_dim=None, num_heads=None, num_layers=None,
                 cls_in=768, mean_in=768, max_in=768, attn_in=10, max_seq_len=None):
        # Use device-specific defaults if not provided
        if hidden_dim is None:
            hidden_dim = HIDDEN_DIM
        if num_heads is None:
            num_heads = NUM_HEADS
        if num_layers is None:
            num_layers = NUM_LAYERS
        if max_seq_len is None:
            max_seq_len = WINDOW_SIZE
        super().__init__()
        self.hierarchy = hierarchy
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        
        # Feature projection layers (reused for temporal encoding)
        self.cls_proj = nn.Linear(cls_in, hidden_dim)
        self.mean_proj = nn.Linear(mean_in, hidden_dim)
        self.max_proj = nn.Linear(max_in, hidden_dim)
        self.attn_proj = nn.Linear(attn_in, hidden_dim)
        
        # Positional embeddings for temporal sequences
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, hidden_dim) * 0.02)
        
        # Temporal transformer encoder for LogBERT sequences
        temporal_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True)
        self.temporal_encoder = nn.TransformerEncoder(temporal_layer, num_layers=num_layers)
        
        # Attention pooling for temporal aggregation
        self.attention_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Original encoder for non-temporal features (FastText/Word2Vec)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.input_recon_dim = cls_in + mean_in + max_in + attn_in
        self.bottleneck = nn.Linear(hidden_dim, hidden_dim // 2)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.input_recon_dim)
        )

        # Top-level multi-label head
        self.top_level_head = nn.Linear(self.bottleneck.out_features, len(TOP_LEVEL_CLASSES))
        
        # Hierarchy heads
        self.heads = nn.ModuleDict()
        self.parent_child_map = {}
        self.node_to_index = {}
        self._build_heads(self.hierarchy)
        
        # Enable gradient checkpointing for MPS to save memory
        if device.type == "mps":
            self._enable_checkpointing()

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

    def _enable_checkpointing(self):
        """Enable gradient checkpointing for temporal encoder to save memory on MPS"""
        def checkpoint_wrapper(layer):
            original_forward = layer.forward
            def checkpointed_forward(src, src_mask=None, src_key_padding_mask=None):
                return torch.utils.checkpoint.checkpoint(
                    original_forward, src, src_mask, src_key_padding_mask, use_reentrant=False
                )
            layer.forward = checkpointed_forward
            return layer
        
        # Apply checkpointing to temporal encoder layers
        for i, layer in enumerate(self.temporal_encoder.layers):
            self.temporal_encoder.layers[i] = checkpoint_wrapper(layer)

    def forward(self, cls_tokens, mean_pooling, max_pooling, attn, attention_mask=None):
        batch_size = cls_tokens.shape[0]
        
        # Check if this is temporal (windowed) input
        is_temporal = cls_tokens.dim() == 3  # (B, T, D) vs (B, D)
        
        if is_temporal:
            # Temporal processing for LogBERT windows
            seq_len = cls_tokens.shape[1]
            
            # Project each time step's features
            cls_h = self.cls_proj(cls_tokens)  # (B, T, hidden_dim)
            mean_h = self.mean_proj(mean_pooling)  # (B, T, hidden_dim)
            max_h = self.max_proj(max_pooling)  # (B, T, hidden_dim)
            attn_h = self.attn_proj(attn)  # (B, T, hidden_dim)
            
            # Sum features at each time step
            h_t = cls_h + mean_h + max_h + attn_h  # (B, T, hidden_dim)
            
            # Add positional embeddings
            h_t = h_t + self.pos_embedding[:, :seq_len, :]
            
            # Apply temporal transformer encoder
            if attention_mask is not None:
                # Convert boolean mask to attention mask for transformer
                attn_mask = ~attention_mask  # Transformer uses True for positions to ignore
            else:
                attn_mask = None
                
            h_enc = self.temporal_encoder(h_t, src_key_padding_mask=attn_mask)
            
            # Attention pooling over time
            attn_weights = self.attention_pool(h_enc)  # (B, T, 1)
            if attention_mask is not None:
                attn_weights = attn_weights.masked_fill(~attention_mask.unsqueeze(-1), float('-inf'))
            attn_weights = torch.softmax(attn_weights, dim=1)
            pooled = (h_enc * attn_weights).sum(dim=1)  # (B, hidden_dim)
            
            # For reconstruction, use time-averaged raw features
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1)
                raw_feats = torch.cat([cls_tokens, mean_pooling, max_pooling, attn], dim=-1)
                raw_feats_masked = raw_feats * mask_expanded.float()
                seq_lengths = attention_mask.sum(dim=1, keepdim=True).float()
                recon_target = raw_feats_masked.sum(dim=1) / seq_lengths.clamp(min=1)
            else:
                raw_feats = torch.cat([cls_tokens, mean_pooling, max_pooling, attn], dim=-1)
                recon_target = raw_feats.mean(dim=1)
        else:
            # Non-temporal processing for FastText/Word2Vec (original path)
            cls_h, mean_h = self.cls_proj(cls_tokens), self.mean_proj(mean_pooling)
            max_h, attn_h = self.max_proj(max_pooling), self.attn_proj(attn)

            combined = torch.stack([cls_h, mean_h, max_h, attn_h], dim=1)
            h_enc = self.encoder(combined)
            pooled = h_enc.mean(1)
            
            recon_target = torch.cat([cls_tokens, mean_pooling, max_pooling, attn], dim=1)

        # Bottleneck and reconstruction
        z = self.bottleneck(pooled)
        recon = self.decoder(z)
        
        # Top-level predictions
        top_logits = torch.clamp(self.top_level_head(z), -10.0, 10.0)
        
        # Hierarchy sub-head predictions
        sub_outputs = {name: torch.clamp(head(z), -10.0, 10.0) for name, head in self.heads.items()}
        
        return recon, z, top_logits, sub_outputs, recon_target

    def hierarchy_consistency_loss(self, sub_outputs):
        """Tighter hierarchy consistency: child prob ≤ parent prob with margin"""
        if not sub_outputs:
            return torch.tensor(0.0, device=next(iter(self.heads.parameters())).device)
            
        consistency_loss = torch.tensor(0.0, device=next(iter(sub_outputs.values())).device)
        count = 0

        for parent, children in self.parent_child_map.items():
            if parent in sub_outputs:
                parent_logits = torch.clamp(sub_outputs[parent], -10.0, 10.0)
                parent_prob = torch.sigmoid(parent_logits)

                for child in children:
                    if child in sub_outputs:
                        child_logits = torch.clamp(sub_outputs[child], -10.0, 10.0)
                        child_prob = torch.sigmoid(child_logits)
                        # Encourage margin: child should be notably less than parent
                        margin = 0.1
                        violation = F.relu(child_prob - parent_prob + margin)
                        consistency_loss += violation.mean()
                        count += 1

        return consistency_loss / max(count, 1)

    def decode_hierarchical(self, top_logits, sub_outputs, threshold=0.5):
        """Top-down hierarchical decoding with consistency enforcement"""
        batch_size = top_logits.shape[0]
        
        # First decode top-level
        top_probs = torch.sigmoid(top_logits)
        top_preds = (top_probs > threshold).float()
        
        # Initialize final predictions
        all_preds = {}
        
        # Add top-level predictions (excluding 'normal')
        for i, class_name in enumerate(TOP_LEVEL_CLASSES):
            if class_name != "normal":
                all_preds[class_name] = top_preds[:, i]
        
        # Decode sub-classes with parent-gated logits
        for parent, children in self.parent_child_map.items():
            if parent in TOP_LEVEL_CLASSES:
                p_idx = TOP_LEVEL_CLASSES.index(parent)
                p_prob = top_probs[:, p_idx].unsqueeze(-1)  # (B,1)
                for child in children:
                    if child in sub_outputs:
                        c_prob = torch.sigmoid(torch.clamp(sub_outputs[child], -10.0, 10.0))
                        c_prob = c_prob * p_prob  # gate by parent
                        all_preds[child] = (c_prob > threshold).float().squeeze(-1)
                        
                        # Process grandchildren
                        if child in self.parent_child_map:
                            child_pred_for_gc = all_preds[child]  # Get the child prediction we just set
                            for grandchild in self.parent_child_map[child]:
                                if grandchild in sub_outputs:
                                    gc_logits = torch.clamp(sub_outputs[grandchild], -10.0, 10.0)
                                    gc_prob = torch.sigmoid(gc_logits)
                                    gc_pred = (gc_prob > threshold).float() * child_pred_for_gc
                                    all_preds[grandchild] = gc_pred.squeeze(-1) if gc_pred.dim() > 1 else gc_pred
        
        return all_preds

    def optimistic_event_mapping(self, window_preds, window_to_event_map):
        """Map window predictions back to events using optimistic (max) aggregation"""
        if not window_to_event_map:
            return window_preds
            
        event_preds = {}
        max_event_idx = max(max(events) for events in window_to_event_map.values()) if window_to_event_map else 0
        
        for class_name in window_preds:
            if class_name in window_preds:
                # Initialize event predictions
                event_pred = torch.zeros(max_event_idx + 1, device=window_preds[class_name].device)
                
                # For each window, update covered events with max probability
                for window_idx, event_indices in window_to_event_map.items():
                    if window_idx < len(window_preds[class_name]):
                        window_prob = window_preds[class_name][window_idx]
                        for event_idx in event_indices:
                            event_pred[event_idx] = torch.maximum(event_pred[event_idx], window_prob)
                
                event_preds[class_name] = event_pred
        
        return event_preds

def create_sliding_windows(data, labels, window_size=WINDOW_SIZE, stride=STRIDE):
    """Create sliding windows for temporal processing with optimistic labels"""
    n_samples = data.shape[0]
    if n_samples <= window_size:
        # If we have fewer samples than window size, pad and create one window
        pad_size = window_size - n_samples
        if data.ndim == 2:
            padded_data = np.concatenate([np.zeros((pad_size, data.shape[1]), dtype=data.dtype), data], axis=0)
        else:  # 3D for multi-feature data
            padded_data = np.concatenate([np.zeros((pad_size,) + data.shape[1:], dtype=data.dtype), data], axis=0)
        
        padded_labels = np.concatenate([np.zeros((pad_size, labels.shape[1]), dtype=labels.dtype), labels], axis=0)
        attention_mask = np.concatenate([np.zeros(pad_size, dtype=bool), np.ones(n_samples, dtype=bool)])
        
        return padded_data[None, ...], padded_labels[None, ...], attention_mask[None, ...], {0: list(range(n_samples))}
    
    # Create overlapping windows
    windows_data = []
    windows_labels = []
    attention_masks = []
    window_to_event_map = {}
    
    for i in range(0, n_samples - window_size + 1, stride):
        end_idx = min(i + window_size, n_samples)
        actual_size = end_idx - i
        
        if actual_size < window_size:
            # Left-pad the final window
            pad_size = window_size - actual_size
            if data.ndim == 2:
                window_data = np.concatenate([
                    np.zeros((pad_size, data.shape[1]), dtype=data.dtype),
                    data[i:end_idx]
                ], axis=0)
            else:
                window_data = np.concatenate([
                    np.zeros((pad_size,) + data.shape[1:], dtype=data.dtype),
                    data[i:end_idx]
                ], axis=0)
            
            window_labels = np.concatenate([
                np.zeros((pad_size, labels.shape[1]), dtype=labels.dtype),
                labels[i:end_idx]
            ], axis=0)
            
            attention_mask = np.concatenate([
                np.zeros(pad_size, dtype=bool),
                np.ones(actual_size, dtype=bool)
            ])
            
            event_indices = list(range(i, end_idx))
        else:
            window_data = data[i:end_idx]
            window_labels = labels[i:end_idx]
            attention_mask = np.ones(window_size, dtype=bool)
            event_indices = list(range(i, end_idx))
        
        # Optimistic OR for window labels: if any event has label, window has label
        window_label = np.any(window_labels, axis=0).astype(np.float32)
        
        windows_data.append(window_data)
        windows_labels.append(window_label)
        attention_masks.append(attention_mask)
        window_to_event_map[len(windows_data) - 1] = event_indices
    
    return (np.array(windows_data), np.array(windows_labels), 
            np.array(attention_masks), window_to_event_map)

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
    """Flatten hierarchy to list of all nodes"""
    for node, children in h.items():
        yield node
        if isinstance(children, dict):
            for child, leaves in children.items():
                yield child
                for leaf in leaves:
                    yield leaf

def build_maps(h):
    """Return node_list, name->idx, parent_map, children_map, descendants_map"""
    node_list = list(flatten_hierarchy(h))
    name_to_idx = {n:i for i,n in enumerate(node_list)}
    parent = {}
    children = {}

    for top, sub in h.items():
        children.setdefault(top, [])
        if isinstance(sub, dict):
            for child, leaves in sub.items():
                children[top].append(child)
                parent[child] = top
                children.setdefault(child, [])
                for leaf in leaves:
                    children[child].append(leaf)
                    parent[leaf] = child

    # compute descendants for each top-level (all levels below)
    descendants = {n:set() for n in node_list}
    def dfs(n):
        for c in children.get(n, []):
            descendants[n].add(c)
            descendants[n] |= dfs(c)
        return descendants[n]
    for n in node_list:
        dfs(n)

    return node_list, name_to_idx, parent, children, descendants

def propagate_targets_up_np(targets, hierarchy):
    """In-place propagate positives from leaves to ancestors."""
    node_list, name_to_idx, parent, children, _ = build_maps(hierarchy)
    # 2 passes are enough on this 3-level tree
    for _ in range(2):
        for child, p in parent.items():
            if child in name_to_idx and p in name_to_idx:
                ci = name_to_idx[child]
                pi = name_to_idx[p]
                targets[:, pi] = np.maximum(targets[:, pi], targets[:, ci])
    return targets

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

def make_weighted_sampler(dataset):
    """Create weighted sampler to avoid all-normal batches on FastText/Word2Vec"""
    # targets at index 4
    Y = torch.stack([dataset[i][4] for i in range(len(dataset))])
    has_pos = (Y.sum(dim=1) > 0).float()
    # weight normals small, anomalies large
    w = torch.where(has_pos > 0, torch.full_like(has_pos, 50.0), torch.full_like(has_pos, 1.0))
    return torch.utils.data.WeightedRandomSampler(weights=w.double(), num_samples=len(dataset), replacement=True)

def find_best_thresholds(y_true, y_prob):
    """Find optimal thresholds per class for rare labels"""
    taus = np.linspace(0.05, 0.5, 10)  # focus on low thresholds for rare classes
    best = []
    for j in range(y_true.shape[1]):
        yt, yp = y_true[:, j], y_prob[:, j]
        if yt.sum() == 0:
            best.append(0.5); continue
        f1s = [precision_recall_fscore_support(yt, (yp>=t).astype(int), average='binary', zero_division=0)[2] for t in taus]
        best.append(float(taus[int(np.argmax(f1s))]))
    return np.array(best, dtype=np.float32)

def load_datasets(embeddings, labels, batch_size=128):
    datasets = {}
    for log_type, log_vectors in embeddings.items():
        dim = log_vectors.shape[1]
        is_logbert = (dim == 2314)
        
        if is_logbert:
            # LogBERT enhanced - apply windowing
            cls_tokens, mean_pooling = log_vectors[:, :768], log_vectors[:, 768:1536]
            max_pooling, attn = log_vectors[:, 1536:2304], log_vectors[:, 2304:]
            in_dims = (768, 768, 768, 10)
            
            # Prepare targets for windowing
            if log_type in labels:
                targets = create_multilabel_targets(labels[log_type], hierarchy)
            else:
                targets = np.zeros((log_vectors.shape[0], len(list(flatten_hierarchy(hierarchy)))))
            
            if targets is None:
                targets = np.zeros((log_vectors.shape[0], len(list(flatten_hierarchy(hierarchy)))))
            
            # 🔧 NEW: propagate labels upward so parents are positive when any descendant is positive
            targets = propagate_targets_up_np(targets, hierarchy)
            
            # Create sliding windows for LogBERT data
            print(f"Creating temporal windows for {log_type} (LogBERT)...")
            
            # Stack features for windowing
            features_stacked = np.stack([cls_tokens, mean_pooling, max_pooling], axis=-1)  # (N, feat_dim, 3)
            attn_expanded = attn  # (N, 10)
            
            # Apply windowing
            windowed_features, windowed_labels, attention_masks, window_to_event_map = create_sliding_windows(
                features_stacked, targets, WINDOW_SIZE, STRIDE
            )
            
            # Unstack windowed features
            windowed_cls = windowed_features[:, :, :, 0]      # (n_windows, window_size, 768)
            windowed_mean = windowed_features[:, :, :, 1]     # (n_windows, window_size, 768)  
            windowed_max = windowed_features[:, :, :, 2]      # (n_windows, window_size, 768)
            
            # Handle attention features (need to window separately due to different dim)
            windowed_attn, _, attn_masks, _ = create_sliding_windows(attn_expanded, targets, WINDOW_SIZE, STRIDE)
            windowed_attn = windowed_attn[:, :, :10]  # Take only attention features
            
            dataset = TensorDataset(
                torch.from_numpy(windowed_cls).float(),
                torch.from_numpy(windowed_mean).float(),
                torch.from_numpy(windowed_max).float(),
                torch.from_numpy(windowed_attn).float(),
                torch.from_numpy(windowed_labels).float(),
                torch.from_numpy(attention_masks).bool()
            )
            
            total_windows = len(windowed_labels)
            total_events = log_vectors.shape[0]
            anomaly_windows = (windowed_labels.sum(axis=1) > 0).sum()
            
            print(f"[{log_type}] {total_events} events → {total_windows} windows | anomaly windows: {anomaly_windows} ({anomaly_windows/total_windows:.2%})")
            
        else:
            # FastText / Word2Vec etc. - keep single-row behavior (T=1)
            cls_tokens = log_vectors
            feat_dim = dim
            zero = np.zeros((log_vectors.shape[0], feat_dim), dtype=log_vectors.dtype)
            mean_pooling, max_pooling = zero, zero
            attn = np.zeros((log_vectors.shape[0], 10), dtype=log_vectors.dtype)
            in_dims = (feat_dim, feat_dim, feat_dim, 10)
            
            if log_type in labels:
                targets = create_multilabel_targets(labels[log_type], hierarchy)
            else:
                targets = np.zeros((log_vectors.shape[0], len(list(flatten_hierarchy(hierarchy)))))
            
            if targets is None:
                targets = np.zeros((log_vectors.shape[0], len(list(flatten_hierarchy(hierarchy)))))
            
            # 🔧 NEW: propagate labels upward so parents are positive when any descendant is positive
            targets = propagate_targets_up_np(targets, hierarchy)
            
            dataset = TensorDataset(
                torch.from_numpy(cls_tokens).float(),
                torch.from_numpy(mean_pooling).float(),
                torch.from_numpy(max_pooling).float(),
                torch.from_numpy(attn).float(),
                torch.from_numpy(targets).float()
            )
            
            total = log_vectors.shape[0]
            anomalies = (targets.sum(axis=1) > 0).sum()
            print(f"[{log_type}] {total} samples | anomalies: {anomalies} ({anomalies/total:.2%})")
        
        # Set prefetch_factor only when using multiprocessing
        prefetch_factor = 4 if NUM_WORKERS > 0 else None
        
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                           num_workers=NUM_WORKERS, pin_memory=(device.type=="cuda"), 
                           persistent_workers=PERSISTENT_WORKERS, prefetch_factor=prefetch_factor)
        datasets[log_type] = {
            'loader': loader, 
            'num_samples': total_windows if is_logbert else log_vectors.shape[0], 
            'in_dims': in_dims,
            'is_temporal': is_logbert,
            'window_to_event_map': window_to_event_map if is_logbert else None
        }
        
        # Print class distribution
        if is_logbert:
            targets_for_dist = windowed_labels
            total_for_dist = total_windows
        else:
            targets_for_dist = targets
            total_for_dist = total
            
        node_names = list(flatten_hierarchy(hierarchy))
        normal_count = total_for_dist - (targets_for_dist.sum(axis=1) > 0).sum()
        print("Per-class distribution:")
        print(f"  normal: {normal_count} ({normal_count/total_for_dist:.2%})")
        for i, name in enumerate(node_names):
            if i < targets_for_dist.shape[1]:
                count = int(targets_for_dist[:, i].sum())
                if count > 0:
                    print(f"  {name}: {count} ({count/total_for_dist:.2%})")
    
    return datasets

def smote_data(train_loader, val_loader):
    """Apply balanced resampling to address class imbalance (on window-level pooled features for temporal data)"""
    spinner = halo.Halo(text="Applying balanced resampling", spinner="dots")
    spinner.start()

    try:
        # Extract training data
        is_temporal = len(next(iter(train_loader))) == 6  # Has attention mask
        
        if is_temporal:
            # For temporal data, extract and pool to get per-window features
            all_data = [[], [], [], [], [], []]
            for batch in train_loader:
                for i, tensor in enumerate(batch):
                    all_data[i].append(tensor.detach().cpu().numpy())
            cls, mean, maxp, attn, y, masks = [np.concatenate(data, axis=0) for data in all_data]
            
            # Pool temporal features to get per-window representations for SMOTE
            # Use attention-weighted mean pooling
            masks_expanded = masks.astype(np.float32)  # (B, T)
            seq_lengths = masks_expanded.sum(axis=1, keepdim=True)  # (B, 1)
            
            # Pool each feature type
            cls = (cls * masks_expanded[..., None]).sum(axis=1) / np.maximum(seq_lengths, 1)
            mean = (mean * masks_expanded[..., None]).sum(axis=1) / np.maximum(seq_lengths, 1)
            maxp = (maxp * masks_expanded[..., None]).sum(axis=1) / np.maximum(seq_lengths, 1)
            attn = (attn * masks_expanded[..., None]).sum(axis=1) / np.maximum(seq_lengths, 1)
        else:
            # Non-temporal data
            all_data = [[], [], [], [], []]
            for batch in train_loader:
                for i, tensor in enumerate(batch):
                    all_data[i].append(tensor.detach().cpu().numpy())
            cls, mean, maxp, attn, y = [np.vstack(data) for data in all_data]

        # Safety cap: limit rows used for resampling to avoid OOM/very long runs
        n_total = y.shape[0]
        max_rows = 400_000 if device.type == "cuda" else 200_000
        if n_total > max_rows:
            rng = np.random.default_rng(42)
            idx = rng.choice(n_total, size=max_rows, replace=False)
            cls, mean, maxp, attn, y = cls[idx], mean[idx], maxp[idx], attn[idx], y[idx]
        
        # Preprocessing (skip zero-width blocks)
        blocks = [cls, mean, maxp, attn]
        scalers, scaled = [], []
        for b in blocks:
            if b.shape[1] > 0:
                sc = StandardScaler().fit(b)
                scalers.append(sc)
                scaled.append(sc.transform(b))
            else:
                scalers.append(None)
                scaled.append(b)
        
        # Dimensionality reduction for efficiency
        pca_dims = [min(48, s.shape[1]) if s.shape[1] > 0 else 0 for s in scaled]
        pcas = [PCA(n_components=dim, svd_solver='auto').fit(s) if dim > 0 else None for dim, s in zip(pca_dims, scaled)]
        reduced_parts = []
        for p, s in zip(pcas, scaled):
            if p is None:
                reduced_parts.append(np.zeros((s.shape[0], 0), dtype=s.dtype))
            else:
                reduced_parts.append(p.transform(s))
        X_reduced = np.hstack(reduced_parts)

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
                if pos_count < target_per_class:
                    # Oversample minority class
                    k_neighbors = min(3, pos_count - 1) if pos_count > 1 else 1
                    oversample_ratio = min(target_per_class / max(pos_count, 1), 1.5)  # replace
                    
                    smote = BorderlineSMOTE(
                        sampling_strategy={1: min(int(pos_count * oversample_ratio), pos_count + 20000)},
                        k_neighbors=max(2, k_neighbors),
                        random_state=42,
                        kind='borderline-1'
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
        splits = np.cumsum([p.n_components_ if p is not None else 0 for p in pcas[:-1]])
        x_blocks = np.split(X_final, splits, axis=1)
        # Batched inverse-transform to limit peak memory
        def _inv(pca, scaler, block, bs=200000):
            out = []
            for i in range(0, len(block), bs):
                chunk = block[i:i+bs]
                if pca is None or scaler is None or chunk.shape[1] == 0:
                    out.append(np.zeros((len(chunk), 0), dtype=chunk.dtype))
                else:
                    out.append(scaler.inverse_transform(pca.inverse_transform(chunk)))
            return np.vstack(out)
        final_features = [_inv(p, s, b, bs=100_000 if device.type=="cuda" else 50_000) for (p, s, b) in zip(pcas, scalers, x_blocks)]
        
        # Create balanced dataset
        if is_temporal:
            # For temporal data, need to expand back to sequences (using mean pooling as placeholder)
            seq_len = WINDOW_SIZE
            final_features_expanded = []
            for feat in final_features:
                if feat.shape[1] > 0:
                    # Expand to sequence by repeating the pooled feature
                    expanded = np.tile(feat[:, None, :], (1, seq_len, 1))
                    final_features_expanded.append(expanded)
                else:
                    final_features_expanded.append(np.zeros((len(feat), seq_len, 0), dtype=feat.dtype))
            
            # Create attention masks (all True since we don't have original temporal structure)
            attention_masks = np.ones((len(y_final), seq_len), dtype=bool)
            
            dataset = TensorDataset(*[torch.from_numpy(data).float() for data in final_features_expanded + [y_final]] + [torch.from_numpy(attention_masks).bool()])
        else:
            dataset = TensorDataset(*[torch.from_numpy(data).float() for data in final_features + [y_final]])
        
        print(f"\nBalanced dataset: {len(dataset)} samples")
        anomaly_count = (torch.from_numpy(y_final).sum(dim=1) > 0).sum().item()
        print(f"  normal: {len(dataset) - anomaly_count} ({(len(dataset) - anomaly_count)/len(dataset):.2%})")
        print(f"  anomalies: {anomaly_count} ({anomaly_count/len(dataset):.2%})")

        # Set prefetch_factor only when using multiprocessing
        prefetch_factor = 4 if NUM_WORKERS > 0 else None
        
        return DataLoader(dataset, batch_size=getattr(train_loader, 'batch_size', BATCH_SIZE), shuffle=True, 
                         num_workers=NUM_WORKERS, pin_memory=(device.type=="cuda"), 
                         persistent_workers=PERSISTENT_WORKERS, prefetch_factor=prefetch_factor)

    except Exception as e:
        print(f"Resampling failed: {e}")
        return train_loader
    finally:
        try:
            spinner.stop_and_persist(text="Resampling completed")
        except:
            pass

def compute_severity_weights(top_targets, top_preds_probs):
    """Compute cost-sensitive weights based on severity downgrade penalty"""
    batch_size = top_targets.shape[0]
    weights = torch.ones_like(top_targets)
    
    for i in range(batch_size):
        # Get true and predicted severity levels
        true_classes = torch.nonzero(top_targets[i] > 0.5, as_tuple=True)[0]
        pred_probs = top_preds_probs[i]
        
        if len(true_classes) > 0:
            # Get maximum true severity
            true_severities = [SEVERITY_MAP.get(TOP_LEVEL_CLASSES[j], 0) for j in true_classes]
            max_true_severity = max(true_severities)
            
            # Estimate predicted severity (weighted by probabilities)
            pred_severity = sum(SEVERITY_MAP.get(TOP_LEVEL_CLASSES[j], 0) * pred_probs[j].item() 
                              for j in range(len(TOP_LEVEL_CLASSES)))
            
            # Compute downgrade cost
            downgrade = max(0, max_true_severity - pred_severity)
            
            # Apply higher weights to severe false negatives
            for j in true_classes:
                class_name = TOP_LEVEL_CLASSES[j]
                severity = SEVERITY_MAP.get(class_name, 0)
                if severity >= 2:  # High-severity classes
                    weights[i, j] = 1.0 + LAMBDA_DOWN * downgrade
                else:
                    weights[i, j] = 1.0 + 0.5 * downgrade
    
    return weights

class CostSensitiveFocalLoss(nn.Module):
    """Focal loss with cost-sensitive weighting"""
    def __init__(self, alpha=1.0, gamma=2.0, pos_weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
        
    def forward(self, input, target, sample_weights=None):
        # Standard BCE with logits
        if self.pos_weight is not None:
            bce_loss = F.binary_cross_entropy_with_logits(
                input, target, pos_weight=self.pos_weight, reduction='none'
            )
        else:
            bce_loss = F.binary_cross_entropy_with_logits(input, target, reduction='none')
        
        # Focal loss modification
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        # Apply cost-sensitive weights
        if sample_weights is not None:
            focal_loss = focal_loss * sample_weights
            
        return focal_loss.mean()

def train_model(model, train_loader, val_loader, epochs=None, lambda_recon=ALPHA_RECON, lambda_hier=LAMBDA_HIER, resample=True):
    if epochs is None:
        epochs = EPOCHS
        
    # Disable SMOTE for MPS to save memory
    if device.type == "mps":
        resample = False
        
    train = smote_data(train_loader, val_loader) if resample else train_loader
    if len(train) == 0:
        train = train_loader
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4, fused=(device.type=="cuda"))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-4, epochs=epochs, steps_per_epoch=len(train))
    mse_loss = nn.MSELoss()
    use_amp = (device.type == "cuda")  # Only use GradScaler on CUDA
    scaler = torch.amp.GradScaler(enabled=use_amp)

    spinner = halo.Halo(text="Training hierarchical model", spinner="dots")
    spinner.start()

    # Detect temporal vs non-temporal data
    sample_batch = next(iter(train_loader))
    is_temporal = len(sample_batch) == 6  # Has attention mask
    
    # Get label names and compute class weights from original training data
    with torch.no_grad():
        if is_temporal:
            c0, m0, x0, a0, _, mask0 = sample_batch
            c0, m0, x0, a0, mask0 = c0.to(device), m0.to(device), x0.to(device), a0.to(device), mask0.to(device)
            _, _, top_out0, sub_out0, _ = model(c0, m0, x0, a0, mask0)
        else:
            c0, m0, x0, a0, _ = sample_batch
            c0, m0, x0, a0 = c0.to(device), m0.to(device), x0.to(device), a0.to(device)
            _, _, top_out0, sub_out0, _ = model(c0, m0, x0, a0)
        
        # Collect all targets for weight computation
        all_targets = []
        for batch in train:
            targets = batch[4] if is_temporal else batch[4]  # Labels are always at index 4
            all_targets.append(targets.cpu())
        Y = torch.cat(all_targets)

        # Compute class weights for hierarchy heads
        sub_label_names = list(sub_out0.keys()) if isinstance(sub_out0, dict) else [f"sub_label_{i}" for i in range(len(sub_out0))]
        sub_pos_weights = []
        for j in range(len(sub_label_names)):
            if j < Y.shape[1]:
                cnt = int(Y[:, j].sum().item())
                if cnt > 0:
                    weight = (Y.shape[0] - cnt) / max(cnt, 1)
                    weight = min(max(weight, 5.0), 50.0)  # Higher floor for rare labels
                else:
                    weight = 1.0
                sub_pos_weights.append(weight)
        
        # Top-level class weights (for severity-based loss)
        top_pos_weights = [1.0] * len(TOP_LEVEL_CLASSES)  # Will be adjusted by severity weights

        # Initialize loss functions
        top_level_loss = CostSensitiveFocalLoss(alpha=1.0, gamma=2.0, pos_weight=torch.tensor(top_pos_weights, device=device))
        sub_level_loss = CostSensitiveFocalLoss(alpha=1.0, gamma=1.5, pos_weight=torch.tensor(sub_pos_weights, device=device))

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
        
        for batch in train:
            if is_temporal:
                cls_tokens, mean_pooling, max_pooling, attn, targets, attention_mask = batch
                attention_mask = attention_mask.to(device, non_blocking=True)
            else:
                cls_tokens, mean_pooling, max_pooling, attn, targets = batch
                attention_mask = None
                
            cls_tokens = cls_tokens.to(device, non_blocking=True)
            mean_pooling = mean_pooling.to(device, non_blocking=True)
            max_pooling = max_pooling.to(device, non_blocking=True)
            attn = attn.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # Device-specific autocast for optimal performance
            if device.type == "cuda":
                autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            elif device.type == "mps":
                autocast_ctx = torch.autocast(device_type="mps", dtype=torch.float16)
            else:
                autocast_ctx = torch.no_grad()  # No autocast for CPU
                
            with autocast_ctx:
                recon, z, top_logits, sub_outputs, recon_target = model(
                    cls_tokens, mean_pooling, max_pooling, attn, attention_mask
                )

                # Reconstruction loss
                recon_loss = mse_loss(recon, recon_target)
                recon_loss = torch.clamp(recon_loss, 0, 100.0)

                # Create top-level targets from propagated hierarchy targets
                batch_size = targets.shape[0]
                node_names, name_to_idx, *_ = build_maps(hierarchy)
                top_targets = torch.zeros(batch_size, len(TOP_LEVEL_CLASSES), device=device)

                # normal = no positives anywhere
                top_targets[:, TOP_LEVEL_CLASSES.index("normal")] = (targets.sum(dim=1) == 0).float()

                # all other top-level classes read their (now propagated) column directly
                for k, cls_name in enumerate(TOP_LEVEL_CLASSES):
                    if cls_name == "normal": 
                        continue
                    if cls_name in name_to_idx and name_to_idx[cls_name] < targets.shape[1]:
                        top_targets[:, k] = targets[:, name_to_idx[cls_name]]

                # Compute severity-based weights
                top_probs = torch.sigmoid(top_logits)
                severity_weights = compute_severity_weights(top_targets, top_probs)
                
                # Top-level loss with cost-sensitive weighting
                top_loss = top_level_loss(top_logits, top_targets, severity_weights)
                top_loss = torch.clamp(top_loss, 0, 100.0)

                # Sub-level loss (masked by parent activity)
                sub_loss = torch.tensor(0.0, device=device)
                if sub_outputs:
                    # Create masks for sub-classes based on parent activity
                    for parent, children in model.parent_child_map.items():
                        if parent in TOP_LEVEL_CLASSES:
                            parent_idx = TOP_LEVEL_CLASSES.index(parent)
                            parent_active = top_targets[:, parent_idx] > 0.5
                            
                            for child in children:
                                if child in sub_outputs and child in node_names:
                                    child_idx = node_names.index(child)
                                    if child_idx < targets.shape[1]:
                                        child_targets = targets[:, child_idx]
                                        child_logits = sub_outputs[child]
                                        
                                        # Only compute loss for samples where parent is active
                                        if parent_active.any():
                                            masked_logits = child_logits[parent_active]
                                            masked_targets = child_targets[parent_active]
                                            if len(masked_targets) > 0:
                                                # Use simple BCE for individual child classes
                                                child_loss = F.binary_cross_entropy_with_logits(
                                                    masked_logits.squeeze(-1), 
                                                    masked_targets, 
                                                    reduction='mean'
                                                )
                                                sub_loss += LAMBDA_INTRA * child_loss

                # Hierarchy consistency loss
                hier_loss = model.hierarchy_consistency_loss(sub_outputs)
                hier_loss = torch.clamp(hier_loss, 0, 10.0)

                # Total loss with new coefficients
                total_loss = lambda_recon * recon_loss + top_loss + sub_loss + lambda_hier * hier_loss

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
            for batch in val_loader:
                if is_temporal:
                    cls_tokens, mean_pooling, max_pooling, attn, targets, attention_mask = batch
                    attention_mask = attention_mask.to(device, non_blocking=True)
                else:
                    cls_tokens, mean_pooling, max_pooling, attn, targets = batch
                    attention_mask = None
                    
                cls_tokens = cls_tokens.to(device, non_blocking=True)
                mean_pooling = mean_pooling.to(device, non_blocking=True)
                max_pooling = max_pooling.to(device, non_blocking=True)
                attn = attn.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                
                # Device-specific autocast for validation
                if device.type == "cuda":
                    val_autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                elif device.type == "mps":
                    val_autocast_ctx = torch.autocast(device_type="mps", dtype=torch.float16)
                else:
                    val_autocast_ctx = torch.no_grad()
                    
                with val_autocast_ctx:
                    recon, z, top_logits, sub_outputs, recon_target = model(
                        cls_tokens, mean_pooling, max_pooling, attn, attention_mask
                    )
                    
                    # Reconstruction loss
                    recon_loss = mse_loss(recon, recon_target)
                    recon_loss = torch.clamp(recon_loss, 0, 100.0)

                    # Create top-level targets from propagated hierarchy targets (same as training)
                    batch_size = targets.shape[0]
                    node_names, name_to_idx, *_ = build_maps(hierarchy)
                    top_targets = torch.zeros(batch_size, len(TOP_LEVEL_CLASSES), device=device)

                    # normal = no positives anywhere
                    top_targets[:, TOP_LEVEL_CLASSES.index("normal")] = (targets.sum(dim=1) == 0).float()

                    # all other top-level classes read their (now propagated) column directly
                    for k, cls_name in enumerate(TOP_LEVEL_CLASSES):
                        if cls_name == "normal": 
                            continue
                        if cls_name in name_to_idx and name_to_idx[cls_name] < targets.shape[1]:
                            top_targets[:, k] = targets[:, name_to_idx[cls_name]]

                    # Compute losses (without gradients)
                    top_probs = torch.sigmoid(top_logits)
                    severity_weights = compute_severity_weights(top_targets, top_probs)
                    top_loss = top_level_loss(top_logits, top_targets, severity_weights)
                    top_loss = torch.clamp(top_loss, 0, 100.0)

                    # Sub-level loss
                    sub_loss = torch.tensor(0.0, device=device)
                    if sub_outputs:
                        for parent, children in model.parent_child_map.items():
                            if parent in TOP_LEVEL_CLASSES:
                                parent_idx = TOP_LEVEL_CLASSES.index(parent)
                                parent_active = top_targets[:, parent_idx] > 0.5
                                
                                for child in children:
                                    if child in sub_outputs and child in node_names:
                                        child_idx = node_names.index(child)
                                        if child_idx < targets.shape[1]:
                                            child_targets = targets[:, child_idx]
                                            child_logits = sub_outputs[child]
                                            
                                            if parent_active.any():
                                                masked_logits = child_logits[parent_active]
                                                masked_targets = child_targets[parent_active]
                                                if len(masked_targets) > 0:
                                                    # Use simple BCE for individual child classes
                                                    child_loss = F.binary_cross_entropy_with_logits(
                                                        masked_logits.squeeze(-1), 
                                                        masked_targets, 
                                                        reduction='mean'
                                                    )
                                                    sub_loss += LAMBDA_INTRA * child_loss

                    hier_loss = model.hierarchy_consistency_loss(sub_outputs)
                    hier_loss = torch.clamp(hier_loss, 0, 10.0)

                    total_loss = lambda_recon * recon_loss + top_loss + sub_loss + lambda_hier * hier_loss
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

def evaluate_model(model, test_loader, log_type, embedding_type="", window_to_event_map=None):
    model.eval()
    all_window_preds, all_window_targets = [], []
    all_top_preds, all_top_targets = [], []
    
    # Detect temporal vs non-temporal
    sample_batch = next(iter(test_loader))
    is_temporal = len(sample_batch) == 6
    
    with torch.no_grad():
        for batch in test_loader:
            if is_temporal:
                cls_tokens, mean_pooling, max_pooling, attn, targets, attention_mask = batch
                attention_mask = attention_mask.to(device)
            else:
                cls_tokens, mean_pooling, max_pooling, attn, targets = batch
                attention_mask = None
                
            cls_tokens = cls_tokens.to(device)
            mean_pooling = mean_pooling.to(device)
            max_pooling = max_pooling.to(device)
            attn = attn.to(device)
            
            _, _, top_logits, sub_outputs, _ = model(cls_tokens, mean_pooling, max_pooling, attn, attention_mask)
            
            # Hierarchical decoding
            hierarchical_preds = model.decode_hierarchical(top_logits, sub_outputs)
            
            # Convert to arrays for window-level evaluation
            node_names = list(flatten_hierarchy(hierarchy))
            batch_window_preds = np.zeros((cls_tokens.shape[0], len(node_names)))
            
            for name, preds in hierarchical_preds.items():
                if name in node_names:
                    idx = node_names.index(name)
                    # Ensure proper shape by flattening and taking first element if needed
                    pred_array = preds.cpu().numpy()
                    if pred_array.ndim > 1:
                        pred_array = pred_array.flatten()
                    if len(pred_array) > batch_size:
                        pred_array = pred_array[:batch_size]  # Take only what we need
                    batch_window_preds[:, idx] = pred_array
            
            # Top-level predictions
            top_probs = torch.sigmoid(top_logits)
            batch_top_preds = (top_probs > 0.5).cpu().numpy()
            
            # Create top-level targets from propagated hierarchy targets
            batch_size = targets.shape[0]
            node_names, name_to_idx, *_ = build_maps(hierarchy)
            batch_top_targets = np.zeros((batch_size, len(TOP_LEVEL_CLASSES)))
            
            # normal = no positives anywhere
            batch_top_targets[:, TOP_LEVEL_CLASSES.index("normal")] = (targets.sum(dim=1) == 0).cpu().numpy()
            
            # all other top-level classes read their (now propagated) column directly
            for k, cls_name in enumerate(TOP_LEVEL_CLASSES):
                if cls_name == "normal": 
                    continue
                if cls_name in name_to_idx and name_to_idx[cls_name] < targets.shape[1]:
                    batch_top_targets[:, k] = targets[:, name_to_idx[cls_name]].cpu().numpy()
            
            all_window_preds.append(batch_window_preds)
            all_window_targets.append(targets.cpu().numpy())
            all_top_preds.append(batch_top_preds)
            all_top_targets.append(batch_top_targets)
    
    all_window_preds = np.vstack(all_window_preds)
    all_window_targets = np.vstack(all_window_targets)
    all_top_preds = np.vstack(all_top_preds)
    all_top_targets = np.vstack(all_top_targets)
    
    # Event-level evaluation for temporal data
    if is_temporal and window_to_event_map:
        print("\n=== Event-Level Evaluation (Optimistic Mapping) ===")
        # Map window predictions back to events
        event_preds_dict = {}
        for name in node_names:
            if name in node_names:
                idx = node_names.index(name)
                window_probs = all_window_preds[:, idx]
                event_pred = np.zeros(max(max(events) for events in window_to_event_map.values()) + 1)
                
                for window_idx, event_indices in window_to_event_map.items():
                    if window_idx < len(window_probs):
                        window_prob = window_probs[window_idx]
                        for event_idx in event_indices:
                            event_pred[event_idx] = max(event_pred[event_idx], window_prob)
                
                event_preds_dict[name] = event_pred
        
        # For event-level evaluation, we'd need original event-level targets
        # This would require modification of the dataset loading to preserve event mappings
        print("Event-level mapping completed (detailed metrics require original event targets)")
    
    # Top-level severity-weighted metrics
    print("\n=== Top-Level Classification Metrics ===")
    print(f"{'Class':<15} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Severity':>10}")
    print("-" * 65)
    
    top_class_metrics = []
    severity_weighted_f1_sum = 0.0
    severity_weight_sum = 0.0
    
    for i, class_name in enumerate(TOP_LEVEL_CLASSES):
        if i < all_top_targets.shape[1]:
            y_true = all_top_targets[:, i]
            y_pred = all_top_preds[:, i]
            severity = SEVERITY_MAP.get(class_name, 0)

            if y_true.sum() > 0 or y_pred.sum() > 0:
                prec, rec, f1, _ = precision_recall_fscore_support(
                    y_true, y_pred, average='binary', zero_division=0
                )
                top_class_metrics.append([prec, rec, f1, severity])
                print(f"{class_name:<15} {prec:>10.3f} {rec:>10.3f} {f1:>10.3f} {severity:>10}")
                
                # Severity-weighted F1
                severity_weighted_f1_sum += f1 * (severity + 1)  # +1 to avoid zero weight for normal
                severity_weight_sum += (severity + 1)
    
    # Window-level hierarchy metrics
    print("\n=== Hierarchy Node Metrics (Window-Level) ===")
    print(f"{'Class':<25} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 55)

    node_names = list(flatten_hierarchy(hierarchy))
    class_metrics = []
    for i, name in enumerate(node_names):
        if i < all_window_targets.shape[1]:
            y_true = all_window_targets[:, i]
            y_pred = all_window_preds[:, i]

            if y_true.sum() > 0:
                prec, rec, f1, _ = precision_recall_fscore_support(
                    y_true, y_pred, average='binary', zero_division=0
                )
                class_metrics.append([prec, rec, f1])
                print(f"{name:<25} {prec:>10.3f} {rec:>10.3f} {f1:>10.3f}")
    
    print("\n=== Overall Metrics ===")

    # Window-level micro/macro metrics
    micro_prec, micro_rec, micro_f1, _ = precision_recall_fscore_support(
        all_window_targets.ravel(), all_window_preds.ravel(), average='micro', zero_division=0
    )
    print(f"Window Micro-avg: Precision={micro_prec:.3f}, Recall={micro_rec:.3f}, F1={micro_f1:.3f}")

    if class_metrics:
        macro_metrics = np.mean(class_metrics, axis=0)
        print(f"Window Macro-avg: Precision={macro_metrics[0]:.3f}, Recall={macro_metrics[1]:.3f}, F1={macro_metrics[2]:.3f}")

    # Top-level micro/macro metrics
    top_micro_prec, top_micro_rec, top_micro_f1, _ = precision_recall_fscore_support(
        all_top_targets.ravel(), all_top_preds.ravel(), average='micro', zero_division=0
    )
    print(f"Top-level Micro-avg: Precision={top_micro_prec:.3f}, Recall={top_micro_rec:.3f}, F1={top_micro_f1:.3f}")

    if top_class_metrics:
        top_macro_metrics = np.mean(top_class_metrics, axis=0)
        print(f"Top-level Macro-avg: Precision={top_macro_metrics[0]:.3f}, Recall={top_macro_metrics[1]:.3f}, F1={top_macro_metrics[2]:.3f}")
        
        # Severity-weighted F1
        if severity_weight_sum > 0:
            severity_weighted_f1 = severity_weighted_f1_sum / severity_weight_sum
            print(f"Severity-weighted F1: {severity_weighted_f1:.3f}")

    # Jaccard and anomaly detection
    jaccard = jaccard_score(all_window_targets, all_window_preds, average='samples', zero_division=0)
    print(f"Jaccard Score (samples): {jaccard:.3f}")

    any_attack_true = (all_window_targets.sum(axis=1) > 0).astype(int)
    any_attack_pred = (all_window_preds.sum(axis=1) > 0).astype(int)
    
    if len(np.unique(any_attack_true)) > 1:
        anomaly_prec, anomaly_rec, anomaly_f1, _ = precision_recall_fscore_support(
            any_attack_true, any_attack_pred, average='binary', zero_division=0
        )
        print(f"Anomaly Detection: Precision={anomaly_prec:.3f}, Recall={anomaly_rec:.3f}, F1={anomaly_f1:.3f}")
    
    # Save detailed report
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    embedding_suffix = f"_{embedding_type}" if embedding_type else ""
    temporal_suffix = "_temporal" if is_temporal else ""
    report_path = results_dir / f"hierarchical_{log_type}{embedding_suffix}{temporal_suffix}_evaluation_{timestamp}.txt"
    
    with open(report_path, 'w') as f:
        f.write(f"Enhanced Hierarchical Transformer Evaluation Report\n")
        f.write(f"Log Type: {log_type}\n")
        if embedding_type:
            f.write(f"Embedding Type: {embedding_type}\n")
        f.write(f"Temporal Processing: {'Yes' if is_temporal else 'No'}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Dataset: {len(all_window_targets)} {'windows' if is_temporal else 'samples'}\n")
        f.write("="*70 + "\n\n")
        
        # Top-level metrics
        f.write("Top-Level Classification Metrics:\n")
        f.write(f"{'Class':<15} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Severity':>10}\n")
        f.write("-" * 65 + "\n")
        
        for i, class_name in enumerate(TOP_LEVEL_CLASSES):
            if i < all_top_targets.shape[1]:
                y_true = all_top_targets[:, i]
                y_pred = all_top_preds[:, i]
                severity = SEVERITY_MAP.get(class_name, 0)
                if y_true.sum() > 0 or y_pred.sum() > 0:
                    prec, rec, f1, _ = precision_recall_fscore_support(
                        y_true, y_pred, average='binary', zero_division=0
                    )
                    f.write(f"{class_name:<15} {prec:>10.3f} {rec:>10.3f} {f1:>10.3f} {severity:>10}\n")
        
        # Hierarchy metrics
        f.write(f"\nHierarchy Node Metrics ({'Window-Level' if is_temporal else 'Sample-Level'}):\n")
        f.write(f"{'Class':<25} {'Precision':>10} {'Recall':>10} {'F1':>10}\n")
        f.write("-" * 55 + "\n")
        
        for i, name in enumerate(node_names):
            if i < all_window_targets.shape[1]:
                y_true = all_window_targets[:, i]
                y_pred = all_window_preds[:, i]
                if y_true.sum() > 0:
                    prec, rec, f1, _ = precision_recall_fscore_support(
                        y_true, y_pred, average='binary', zero_division=0
                    )
                    f.write(f"{name:<25} {prec:>10.3f} {rec:>10.3f} {f1:>10.3f}\n")
        
        # Overall metrics
        f.write(f"\nOverall Metrics:\n")
        f.write(f"Window Micro-avg: Precision={micro_prec:.3f}, Recall={micro_rec:.3f}, F1={micro_f1:.3f}\n")
        if class_metrics:
            f.write(f"Window Macro-avg: Precision={macro_metrics[0]:.3f}, Recall={macro_metrics[1]:.3f}, F1={macro_metrics[2]:.3f}\n")
        f.write(f"Top-level Micro-avg: Precision={top_micro_prec:.3f}, Recall={top_micro_rec:.3f}, F1={top_micro_f1:.3f}\n")
        if top_class_metrics:
            f.write(f"Top-level Macro-avg: Precision={top_macro_metrics[0]:.3f}, Recall={top_macro_metrics[1]:.3f}, F1={top_macro_metrics[2]:.3f}\n")
            if severity_weight_sum > 0:
                f.write(f"Severity-weighted F1: {severity_weighted_f1:.3f}\n")
        f.write(f"Jaccard Score: {jaccard:.3f}\n")
        if len(np.unique(any_attack_true)) > 1:
            f.write(f"Anomaly Detection: Precision={anomaly_prec:.3f}, Recall={anomaly_rec:.3f}, F1={anomaly_f1:.3f}\n")
        
        if is_temporal and window_to_event_map:
            f.write(f"\nTemporal Processing Details:\n")
            f.write(f"Window Size: {WINDOW_SIZE}\n")
            f.write(f"Stride: {STRIDE}\n")
            f.write(f"Total Windows: {len(all_window_targets)}\n")
            f.write(f"Event-level mapping: Available\n")
    
    print(f"\nEvaluation report saved: {report_path}")

def main():
    
    parser = argparse.ArgumentParser(description='Train hierarchical transformer on log embeddings')
    parser.add_argument('--embedding_type', type=str, default='all', 
                       choices=['all', 'logbert', 'fasttext', 'word2vec'],
                       help='Type of embeddings to use (default: all)')
    parser.add_argument('--log_type', type=str, default=None,
                       help='Specific log type to process (processes all if not specified)')
    parser.add_argument('--sample_size', type=int, default=None,
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
            # Use device-specific batch size
            batch_size = BATCH_SIZE if data.get('is_temporal', False) else BATCH_SIZE * 2
            
            # Set prefetch_factor only when using multiprocessing
            prefetch_factor = 4 if NUM_WORKERS > 0 else None
            
            # Use weighted sampler for non-temporal datasets to avoid all-normal batches
            if not data.get('is_temporal', False):
                sampler = make_weighted_sampler(train_set)
                train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler,
                                        num_workers=NUM_WORKERS, pin_memory=(device.type=="cuda"), 
                                        persistent_workers=PERSISTENT_WORKERS, prefetch_factor=prefetch_factor)
            else:
                train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  
                                        num_workers=NUM_WORKERS, pin_memory=(device.type=="cuda"), 
                                        persistent_workers=PERSISTENT_WORKERS, prefetch_factor=prefetch_factor)
                                        
            val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, 
                                    num_workers=NUM_WORKERS, pin_memory=(device.type=="cuda"), 
                                    persistent_workers=PERSISTENT_WORKERS, prefetch_factor=prefetch_factor)
            test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, 
                                    num_workers=NUM_WORKERS, pin_memory=(device.type=="cuda"), 
                                    persistent_workers=PERSISTENT_WORKERS, prefetch_factor=prefetch_factor)
            
            print(f"Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")

            in_dims = data.get('in_dims', (768, 768, 768, 10))
            model = HierarchicalTransformer(hierarchy, cls_in=in_dims[0], mean_in=in_dims[1], max_in=in_dims[2], attn_in=in_dims[3]).to(device)
            
            # Only use torch.compile on CUDA for optimal performance
            if device.type == "cuda":
                try:
                    import triton  # type: ignore  # noqa: F401
                    model = torch.compile(model, mode="max-autotune")
                    print("✓ Model compiled with torch.compile for CUDA")
                except Exception:
                    print("Triton not available or compile failed; using eager mode.")
            else:
                print(f"Using eager mode on {device.type} (torch.compile disabled)")
                
            # Use device-specific resampling
            use_resample = (embedding_type == 'logbert' and device.type != "mps")
            model = train_model(model, train_loader, val_loader, resample=use_resample)

            # Get window_to_event_map for evaluation
            window_to_event_map = data.get('window_to_event_map', None)
            evaluate_model(model, test_loader, log_type, embedding_type, window_to_event_map)

            Path("models").mkdir(exist_ok=True)
            model_path = f"models/hierarchical_{log_type}_{embedding_type}.pth"
            torch.save(model.state_dict(), model_path)
            print(f"\nModel saved to {model_path}")

if __name__ == "__main__":
    main()