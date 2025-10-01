import argparse
import pickle
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import math
import warnings

import halo
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support, jaccard_score
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset, Sampler, Subset

warnings.filterwarnings(
    "ignore",
    message="enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True",
    category=UserWarning,
)

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

# Training configuration
TRAIN_SPLIT = 0.8           # Train/test split ratio
MAX_CLASS_TRAIN = 0.8       # Max proportion of any class in training

# Balanced sampling configuration  
LEAF_QUOTA = 20000          # Base samples per leaf class per epoch
RARE_THRESHOLD = 100        # Classes with fewer samples considered rare
RARE_BOOST = 10.0           # Quota multiplier for rare classes
HARD_NEG_RATIO = 0.3        # Hard negatives from siblings for rare classes


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
        self.register_buffer(
            "decision_thresholds",
            torch.full((len(self.node_to_index),), 0.5, dtype=torch.float32),
            persistent=False,
        )

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
    
    def contrastive_sibling_loss(self, z: torch.Tensor, targets: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
        """Contrastive loss to separate sibling classes in latent space (unsupervised)."""
        
        if targets.size(0) < 2:
            return torch.tensor(0.0, device=self.device)
        
        # Build sibling pairs
        sibling_pairs = []
        for parent_name, children in self.children_map.items():
            if len(children) > 1:
                child_indices = [self.node_to_index[c] for c in children if c in self.node_to_index]
                sibling_pairs.extend([(i, j) for i in child_indices for j in child_indices if i < j])
        
        if not sibling_pairs:
            return torch.tensor(0.0, device=self.device)
        
        # Normalize embeddings for contrastive learning
        z_norm = F.normalize(z, p=2, dim=1)
        
        losses = []
        for idx1, idx2 in sibling_pairs:
            if idx1 >= targets.size(1) or idx2 >= targets.size(1):
                continue
            
            # Find samples with class1 but not class2 and vice versa
            mask1 = (targets[:, idx1] == 1) & (targets[:, idx2] == 0)
            mask2 = (targets[:, idx2] == 1) & (targets[:, idx1] == 0)
            
            if mask1.sum() > 0 and mask2.sum() > 0:
                emb1 = z_norm[mask1]
                emb2 = z_norm[mask2]
                
                # Compute similarity matrix
                sim_matrix = torch.matmul(emb1, emb2.t()) / temperature
                
                # We want low similarity between siblings (push apart)
                # Use negative samples from sibling class as contrastive target
                contrastive_loss = torch.logsumexp(sim_matrix, dim=1).mean()
                losses.append(contrastive_loss)
        
        if losses:
            return torch.stack(losses).mean()
        return torch.tensor(0.0, device=self.device)

    def propagate_labels(self, outputs: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Returns hierarchy-consistent probabilities for inference."""

        _, aggregated = self._dp_probabilities(outputs)
        return aggregated


def build_parent_lookup(tree: Mapping[str, Mapping[str, Sequence[str]]]) -> Dict[str, Optional[str]]:
    """Creates a lookup from each node to its parent in the hierarchy."""

    parents: Dict[str, Optional[str]] = {}

    def recurse(current: Mapping[str, Mapping[str, Sequence[str]]], parent: Optional[str]) -> None:
        for node, children in current.items():
            parents.setdefault(node, parent)
            if isinstance(children, Mapping):
                recurse(children, node)
                for intermediate, leaves in children.items():
                    parents.setdefault(intermediate, node)
                    for leaf in leaves:
                        parents.setdefault(str(leaf), intermediate)

    recurse(tree, None)
    return parents


def _resolve_embedding_roots(embedding_type: str) -> Tuple[List[Path], List[Path]]:
    """Returns candidate root directories that may contain embeddings."""

    mapping = {
        "fasttext": ["fasttext", "fasttext_embeddings", ""],
        "word2vec": ["word2vec", "word2vec_embeddings"],
        "logbert": ["logbert", "logbert_embeddings"],
    }

    subdirs = mapping.get(embedding_type, [""])
    search_order: List[Path] = []
    seen: set[str] = set()

    for subdir in subdirs:
        path = Path("embeddings") / subdir if subdir else Path("embeddings")
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        search_order.append(path)
        seen.add(key)

    existing = [path for path in search_order if path.exists()]
    return (existing if existing else search_order, search_order)


def _iter_embedding_dirs(candidate_roots: Iterable[Path]) -> Iterable[Path]:
    """Yields child directories under candidate roots that may store embeddings."""

    for root in candidate_roots:
        if not root.exists() or not root.is_dir():
            continue
        yield from sorted([child for child in root.iterdir() if child.is_dir()])


def _load_embeddings_from_roots(
    embedding_type: str,
    candidate_roots: Iterable[Path],
    target_log_type: Optional[str],
) -> Tuple[Dict[str, np.ndarray], Dict[str, dict], List[str]]:
    embeddings: Dict[str, np.ndarray] = {}
    labels: Dict[str, dict] = {}
    skipped: List[str] = []

    def load_single(embed_dir: Path, logical_name: str) -> None:
        if logical_name in embeddings:
            return
        log_pkl = embed_dir / f"log_{logical_name}.pkl"
        label_pkl = embed_dir / f"label_{logical_name}.pkl"
        if not (log_pkl.exists() and label_pkl.exists()):
            return
        print(f"Loading {logical_name} ({embedding_type})...", end=" ")
        log_data = safe_load(log_pkl)
        label_data = safe_load(label_pkl)
        if log_data is not None and label_data is not None:
            embeddings[logical_name] = log_data
            labels[logical_name] = label_data
            print("✓")
        else:
            print("✗")
            skipped.append(logical_name)

    if target_log_type:
        slug = target_log_type
        for root in candidate_roots:
            embed_dir = root / slug
            if embed_dir.exists():
                load_single(embed_dir, slug)
        return embeddings, labels, skipped

    for embed_dir in _iter_embedding_dirs(candidate_roots):
        logical_name = embed_dir.name
        load_single(embed_dir, logical_name)

    return embeddings, labels, skipped


def _load_embedding_dispatch(embedding_type: str, target_log_type: Optional[str]):
    roots, search_order = _resolve_embedding_roots(embedding_type)
    embeddings, labels, skipped = _load_embeddings_from_roots(embedding_type, roots, target_log_type)

    if target_log_type and target_log_type not in embeddings:
        searched = [str((root / target_log_type).resolve()) for root in search_order]
        raise FileNotFoundError(
            f"Embeddings for '{target_log_type}' not found for {embedding_type}. Checked: {', '.join(searched)}"
        )

    if not embeddings:
        searched = ", ".join(str(path.resolve()) if path.exists() else str(path) for path in search_order)
        generate_hint = {
            "fasttext": "python src/fasttext_embedding.py --output-subdir fasttext",
            "word2vec": "python src/word2vec_embedding_thesis.py --output-subdir word2vec",
            "logbert": "python src/logbert_embeddings.py --output-subdir logbert",
        }.get(embedding_type, "<embedding script>")
        raise FileNotFoundError(
            f"No valid {embedding_type} embeddings located. Checked directories: {searched}.\n"
            f"Generate embeddings via:\n  {generate_hint}"
        )

    print(
        f"Loaded {len(embeddings)} log types for {embedding_type}" +
        (f" (skipped {len(skipped)})" if skipped else "")
    )
    return embeddings, labels


def load_logbert_embeddings(target_log_type: str = None):
    return _load_embedding_dispatch("logbert", target_log_type)


def load_fasttext_embeddings(target_log_type: str = None):
    return _load_embedding_dispatch("fasttext", target_log_type)


def load_word2vec_embeddings(target_log_type: str = None):
    return _load_embedding_dispatch("word2vec", target_log_type)

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

def _slice_and_pad(
    array: np.ndarray,
    start: int,
    width: int,
) -> np.ndarray:
    """Returns a slice of given width, padding or truncating to match exactly."""

    if width <= 0:
        return np.zeros((array.shape[0], 0), dtype=array.dtype)

    end = start + width
    if start >= array.shape[1]:
        return np.zeros((array.shape[0], width), dtype=array.dtype)

    chunk = array[:, start:min(end, array.shape[1])]
    cur_width = chunk.shape[1]
    if cur_width == width:
        return chunk
    if cur_width > width:
        return chunk[:, :width]

    pad_width = width - cur_width
    padding = np.zeros((array.shape[0], pad_width), dtype=array.dtype)
    return np.hstack([chunk, padding])


def load_datasets(embeddings, labels, batch_size=128):
    datasets = {}
    base_config = TransformerConfig()
    cls_dim = 768
    mean_dim = 768
    max_dim = 768
    attn_dim = base_config.attn_input_dim

    for log_type, log_vectors in embeddings.items():
        cls_tokens = _slice_and_pad(log_vectors, 0, cls_dim)
        mean_pooling = _slice_and_pad(log_vectors, cls_dim, mean_dim)
        max_pooling = _slice_and_pad(log_vectors, cls_dim + mean_dim, max_dim)
        attn = _slice_and_pad(log_vectors, cls_dim + mean_dim + max_dim, attn_dim)

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

def smote_data(train_loader, val_loader=None):
    """Generates a balanced training loader with per-class quotas."""

    spinner = halo.Halo(text="Applying balanced resampling", spinner="dots")
    spinner.start()

    class BalancedBatchSampler(Sampler[List[int]]):
        """Sampler that interleaves class indices to satisfy per-class quotas without materializing data."""

        def __init__(
            self,
            class_indices: Mapping[str, np.ndarray],
            class_targets: Mapping[str, int],
            batch_size: int,
            seed: int = 42,
        ) -> None:
            self.class_indices: Dict[str, np.ndarray] = {
                key: np.asarray(indices, dtype=np.int64)
                for key, indices in class_indices.items()
                if indices.size > 0
            }
            self.class_targets: Dict[str, int] = {
                key: int(class_targets[key])
                for key in self.class_indices.keys()
                if class_targets.get(key, 0) > 0
            }
            self.batch_size = max(1, int(batch_size))
            self.seed = int(seed)
            self.epoch = 0
            self.total = sum(self.class_targets.values())
            if self.total == 0:
                raise ValueError("BalancedBatchSampler requires at least one positive target")

        def __len__(self) -> int:
            return max(1, math.ceil(self.total / self.batch_size))

        def __iter__(self):
            rng = np.random.default_rng(self.seed + self.epoch)
            self.epoch += 1
            remaining = self.class_targets.copy()
            active = [key for key, count in remaining.items() if count > 0]
            iterators: Dict[str, Iterable[int]] = {
                key: self._cycle(self.class_indices[key], rng)
                for key in active
            }

            batch: List[int] = []
            idx = 0
            while active:
                key = active[idx % len(active)]
                idx += 1
                if remaining[key] <= 0:
                    active = [item for item in active if remaining[item] > 0]
                    continue
                batch.append(next(iterators[key]))
                remaining[key] -= 1
                if remaining[key] == 0:
                    active = [item for item in active if remaining[item] > 0]
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []

            if batch:
                yield batch

        @staticmethod
        def _cycle(indices: np.ndarray, rng: np.random.Generator) -> Iterable[int]:
            if indices.size == 0:
                raise ValueError("Cannot cycle over an empty index pool")
            if indices.size == 1:
                single = int(indices[0])
                while True:
                    yield single
            while True:
                perm = rng.permutation(len(indices))
                for idx in perm:
                    yield int(indices[idx])

    try:
        dataset = getattr(train_loader, "dataset", None)
        subset_indices: Optional[np.ndarray] = None
        base_dataset = dataset
        if isinstance(dataset, Subset):
            base_dataset = dataset.dataset
            subset_indices = np.asarray(dataset.indices, dtype=np.int64)

        if not isinstance(base_dataset, TensorDataset) or len(base_dataset.tensors) < 5:
            spinner.stop_and_persist(text="Streaming resampler requires TensorDataset with targets; using original loader")
            return train_loader

        label_tensor = base_dataset.tensors[4]
        if not isinstance(label_tensor, torch.Tensor):
            label_tensor = torch.as_tensor(label_tensor)
        label_tensor = label_tensor.detach().cpu()
        if subset_indices is not None:
            label_tensor = label_tensor[subset_indices]

        y = label_tensor.numpy().astype(np.float32)
        if int(y.sum()) == 0:
            spinner.stop_and_persist(text="Skipping resampling (only normal samples detected)")
            return train_loader

        node_names = list(flatten_hierarchy(hierarchy))
        if len(node_names) < y.shape[1]:
            node_names.extend([f"label_{i}" for i in range(len(node_names), y.shape[1])])
        node_to_index = {name: idx for idx, name in enumerate(node_names)}
        parent_lookup = build_parent_lookup(hierarchy)
        parent_nodes = {name for name in parent_lookup.values() if name is not None}
        leaf_indices = [
            node_to_index[name]
            for name in node_names
            if name in node_to_index and name not in parent_nodes and node_to_index[name] < y.shape[1]
        ]
        if not leaf_indices:
            leaf_indices = list(range(y.shape[1]))

        idx_to_name = {
            node_to_index[name]: name
            for name in node_names
            if name in node_to_index and node_to_index[name] < y.shape[1]
        }

        index_children: Dict[int, List[int]] = {}
        for name, parent_name in parent_lookup.items():
            if parent_name is None:
                continue
            child_idx = node_to_index.get(name)
            parent_idx = node_to_index.get(parent_name)
            if child_idx is None or parent_idx is None:
                continue
            if child_idx >= y.shape[1] or parent_idx >= y.shape[1]:
                continue
            index_children.setdefault(parent_idx, []).append(child_idx)

        def class_counts(labels: np.ndarray) -> np.ndarray:
            return labels.sum(axis=0)

        original_normal = int((y.sum(axis=1) == 0).sum())
        original_anomaly = int(len(y) - original_normal)
        original_class_counts = class_counts(y)

        normal_mask = y.sum(axis=1) == 0
        normal_indices = np.where(normal_mask)[0]

        class_indices: Dict[str, np.ndarray] = {}
        class_targets: Dict[str, int] = {}

        # Normal samples matched to leaf quota
        if normal_indices.size > 0:
            class_indices["normal"] = normal_indices
            class_targets["normal"] = LEAF_QUOTA

        active_leaf_indices = [
            idx for idx in leaf_indices if idx < y.shape[1] and np.any(y[:, idx] == 1)
        ]
        leaf_positive_map: Dict[int, np.ndarray] = {}
        for leaf_idx in active_leaf_indices:
            positives = np.where(y[:, leaf_idx] == 1)[0]
            if positives.size:
                leaf_positive_map[leaf_idx] = positives

        parent_candidate_map: Dict[int, np.ndarray] = {}
        for parent_idx, children in index_children.items():
            if parent_idx >= y.shape[1]:
                continue
            parent_mask = y[:, parent_idx] == 1
            child_candidates = [child for child in children if child < y.shape[1]]
            if child_candidates:
                parent_mask = np.logical_and(parent_mask, (y[:, child_candidates].sum(axis=1) == 0))
            candidates = np.where(parent_mask)[0]
            if candidates.size:
                parent_candidate_map[parent_idx] = candidates
            else:
                fallback_parent = np.where(y[:, parent_idx] == 1)[0]
                if fallback_parent.size:
                    parent_candidate_map[parent_idx] = fallback_parent

        # Build sibling map for hard negative mining
        sibling_map: Dict[int, List[int]] = {}
        for parent_idx, children in index_children.items():
            if len(children) > 1:
                for child in children:
                    siblings = [c for c in children if c != child]
                    if siblings:
                        sibling_map[child] = siblings

        # Process leaf classes with simplified quota logic
        leaf_target_map: Dict[int, int] = {}
        rare_classes: List[int] = []
        
        for leaf_idx, positives in leaf_positive_map.items():
            if positives is None or positives.size == 0:
                continue
            
            original_count = int(original_class_counts[leaf_idx]) if leaf_idx < len(original_class_counts) else 0
            is_rare = original_count < RARE_THRESHOLD
            
            # Simple quota: base quota with boost for rare classes
            quota = LEAF_QUOTA
            if is_rare:
                rare_classes.append(leaf_idx)
                quota = int(min(quota * RARE_BOOST, quota * 2))
            
            # Add hard negatives from siblings for rare classes
            indices = positives
            if is_rare and leaf_idx in sibling_map:
                combined = list(positives)
                for sibling_idx in sibling_map[leaf_idx]:
                    if sibling_idx in leaf_positive_map:
                        n_hard = int(len(positives) * HARD_NEG_RATIO)
                        if n_hard > 0:
                            sibling_samples = leaf_positive_map[sibling_idx]
                            if sibling_samples.size > 0:
                                sampled = np.random.choice(
                                    sibling_samples, 
                                    size=min(n_hard, sibling_samples.size),
                                    replace=False
                                )
                                combined.extend(sampled)
                indices = np.array(combined, dtype=np.int64)
            
            class_indices[f"leaf:{leaf_idx}"] = indices
            class_targets[f"leaf:{leaf_idx}"] = quota
            leaf_target_map[leaf_idx] = quota

        # Process parent-only samples (parent label but no children)
        parent_target_map: Dict[int, int] = {}
        for parent_idx, candidates in parent_candidate_map.items():
            if candidates is None or candidates.size == 0:
                continue
            
            # Parent quota: same as leaf quota (don't sum children)
            quota = LEAF_QUOTA
            has_rare_child = any(child in rare_classes for child in index_children.get(parent_idx, []))
            if has_rare_child:
                quota = int(quota * 1.5)
            
            class_indices[f"parent:{parent_idx}"] = candidates
            class_targets[f"parent:{parent_idx}"] = quota
            parent_target_map[parent_idx] = quota

        # Fallback if no specific classes found
        if not leaf_target_map and not parent_target_map:
            anomaly_mask = ~normal_mask
            attack_indices = np.where(anomaly_mask)[0]
            if attack_indices.size > 0:
                class_indices["anomaly"] = attack_indices
                class_targets["anomaly"] = LEAF_QUOTA
            else:
                spinner.stop_and_persist(text="Resampling skipped (no attack samples)")
                return train_loader

        total_targets = sum(class_targets.values())
        if total_targets == 0:
            spinner.stop_and_persist(text="Resampling skipped (no class targets computed)")
            return train_loader

        batch_size = getattr(train_loader, "batch_size", 128)
        sampler = BalancedBatchSampler(class_indices, class_targets, batch_size=batch_size, seed=42)

        print(f"\nBalanced sampling: {total_targets} samples/epoch")
        print(f"  Base quota: {LEAF_QUOTA} | Rare boost: {RARE_BOOST}x")
        print(f"  Classes: {len(leaf_target_map)} leaves, {len(parent_target_map)} parents")
        if rare_classes:
            rare_names = [idx_to_name.get(idx, str(idx)) for idx in rare_classes]
            print(f"  Rare (boosted): {', '.join(rare_names)}")

        print("\nPer-class targets vs originals:")
        adjustments = []
        normal_target_display = class_targets.get("normal", 0)
        adjustments.append(("normal", original_normal, normal_target_display, normal_target_display - original_normal))
        for leaf_idx, target in leaf_target_map.items():
            name = idx_to_name.get(leaf_idx, str(leaf_idx))
            before = int(original_class_counts[leaf_idx])
            adjustments.append((name, before, target, target - before))
        for parent_idx, target in parent_target_map.items():
            name = idx_to_name.get(parent_idx, str(parent_idx))
            before = int(original_class_counts[parent_idx])
            adjustments.append((name, before, target, target - before))
        if "anomaly" in class_targets:
            before = int(anomaly_mask.sum())
            adjustments.append(("anomaly", before, class_targets["anomaly"], class_targets["anomaly"] - before))

        adjustments.sort(key=lambda item: abs(item[3]), reverse=True)
        for name, before, after, delta in adjustments:
            trend = "++" if delta > 0 else "--" if delta < 0 else "=="
            print(f"  {name:<25} {after:>8} ({trend} {delta:+d})")

        spin_msg = "Streaming resampler ready"
        spinner.stop_and_persist(text=spin_msg)

        loader_kwargs = {
            "batch_sampler": sampler,
            "num_workers": getattr(train_loader, "num_workers", 0),
            "pin_memory": getattr(train_loader, "pin_memory", device.type == "cuda"),
        }
        if loader_kwargs["num_workers"] > 0:
            loader_kwargs["persistent_workers"] = getattr(train_loader, "persistent_workers", True)
            loader_kwargs["prefetch_factor"] = getattr(train_loader, "prefetch_factor", 2)
        collate_fn = getattr(train_loader, "collate_fn", None)
        if collate_fn is not None:
            loader_kwargs["collate_fn"] = collate_fn

        return DataLoader(dataset, **loader_kwargs)

    except Exception as exc:  # pylint: disable=broad-except
        print(f"Resampling failed: {exc}")
        spinner.stop_and_persist(text="Resampling failed (fallback to original loader)")
        return train_loader


def calibrate_thresholds(model: HierarchicalTransformer, loader: DataLoader) -> Optional[np.ndarray]:
    """Calibrates per-class decision thresholds using validation predictions."""

    node_names = list(model.node_to_index.keys())
    if not node_names:
        return None

    model.eval()
    all_probs: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []

    with torch.no_grad():
        for cls_tokens, mean_pooling, max_pooling, attn, targets in loader:
            cls_tokens = cls_tokens.to(device, non_blocking=True)
            mean_pooling = mean_pooling.to(device, non_blocking=True)
            max_pooling = max_pooling.to(device, non_blocking=True)
            attn = attn.to(device, non_blocking=True)

            _, _, outputs = model(cls_tokens, mean_pooling, max_pooling, attn)
            aggregated = model.propagate_labels(outputs)
            batch_probs = torch.zeros((cls_tokens.size(0), len(node_names)), device=device)
            for name, prob in aggregated.items():
                idx = model.node_to_index[name]
                batch_probs[:, idx] = prob.view(-1)
            all_probs.append(batch_probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    if not all_probs:
        return None

    probs = np.vstack(all_probs)
    targets = np.vstack(all_targets)
    thresholds = np.full(probs.shape[1], 0.5, dtype=np.float32)
    
    # More granular threshold search for rare classes
    standard_thresholds = np.linspace(0.1, 0.95, 18)
    fine_thresholds = np.linspace(0.05, 0.95, 50)  # Finer granularity for rare classes

    for idx in range(probs.shape[1]):
        y_true = targets[:, idx]
        pos_count = y_true.sum()
        if pos_count == 0:
            thresholds[idx] = 0.99
            continue

        # Use finer threshold search for rare classes
        is_rare = pos_count < RARE_THRESHOLD
        candidate_thresholds = fine_thresholds if is_rare else standard_thresholds
        
        best_f1 = -1.0
        best_threshold = 0.5
        p = probs[:, idx]
        
        for thr in candidate_thresholds:
            y_pred = (p >= thr).astype(np.int32)
            tp = np.logical_and(y_pred == 1, y_true == 1).sum()
            fp = np.logical_and(y_pred == 1, y_true == 0).sum()
            fn = np.logical_and(y_pred == 0, y_true == 1).sum()
            
            # For rare classes, prioritize recall while maintaining reasonable precision
            if is_rare and tp > 0:
                precision = tp / max(tp + fp, 1)
                recall = tp / max(tp + fn, 1)
                # Weight recall more heavily for rare classes (F2 score)
                beta = 2.0 if is_rare else 1.0
                f_score = (1 + beta**2) * precision * recall / max(beta**2 * precision + recall, 1e-10)
            else:
                precision = tp / max(tp + fp, 1)
                recall = tp / max(tp + fn, 1)
                f_score = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
            
            if f_score > best_f1:
                best_f1 = f_score
                best_threshold = thr
            elif f_score == best_f1 and is_rare and recall > 0:
                # For rare classes with equal F-score, prefer lower threshold (higher recall)
                best_threshold = min(best_threshold, thr)
        
        thresholds[idx] = best_threshold
        
        # Print threshold info for rare classes
        if is_rare:
            name = node_names[idx] if idx < len(node_names) else f"class_{idx}"
            print(f"  Rare class '{name}': threshold={best_threshold:.3f}, samples={int(pos_count)}")

    return thresholds


def train_model(model, train_loader, val_loader=None, epochs=10, lambda_recon=0.05, lambda_hier=0.02, lambda_contrastive=0.01):
    """Trains the transformer with reconstruction and hierarchy regularizers."""
    train = smote_data(train_loader, val_loader)
    if len(train) == 0:
        train = train_loader
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4, fused=(device.type=="cuda"))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-4, epochs=epochs, steps_per_epoch=len(train))
    mse_loss = nn.MSELoss()
    use_amp = device.type in ["cuda", "mps"]
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

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
                bce_loss = super().forward(input, target) # TODO: change to class entropy
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

            autocast_ctx = (
                torch.amp.autocast(device_type="cuda") if device.type == "cuda" else nullcontext()
            )
            with autocast_ctx:
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
                
                # Add contrastive loss to separate sibling classes
                contrastive_loss = model.contrastive_sibling_loss(z, targets)
                contrastive_loss = torch.clamp(contrastive_loss, 0, 10.0)

                total_loss = lambda_recon * recon_loss + ml_loss + lambda_hier * hier_loss + lambda_contrastive * contrastive_loss

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
        avg_train_loss = float(np.mean(train_losses)) if train_losses else 0.0

        if val_loader is not None and len(val_loader) > 0:
            val_losses = []
            with torch.no_grad():
                for cls_tokens, mean_pooling, max_pooling, attn, targets in val_loader:
                    cls_tokens = cls_tokens.to(device, non_blocking=True)
                    mean_pooling = mean_pooling.to(device, non_blocking=True)
                    max_pooling = max_pooling.to(device, non_blocking=True)
                    attn = attn.to(device, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)
                    use_cuda_amp = (device.type=="cuda")
                    ctx = (
                        torch.amp.autocast(device_type="cuda") if use_cuda_amp else nullcontext()
                    )
                    with ctx:
                        recon, z, outputs = model(cls_tokens, mean_pooling, max_pooling, attn)
                        feats = torch.cat([cls_tokens, mean_pooling, max_pooling, attn], dim=1)
                        recon_loss = mse_loss(recon, feats)
                        recon_loss = torch.clamp(recon_loss, 0, 100.0)

                        if isinstance(outputs, dict):
                            if outputs:
                                logits = torch.cat([v.view(v.size(0), -1) for v in outputs.values()], dim=1)
                            else:
                                logits = torch.zeros(targets.size(0), len(label_names), device=device)
                        else:
                            logits = outputs

                        logits = torch.clamp(logits, -10.0, 10.0)

                        ml_loss = bce_loss(logits, targets)
                        ml_loss = torch.clamp(ml_loss, 0, 100.0)

                        hier_loss = model.hierarchy_consistency_loss(outputs)
                        hier_loss = torch.clamp(hier_loss, 0, 10.0)
                        
                        contrastive_loss = model.contrastive_sibling_loss(z, targets)
                        contrastive_loss = torch.clamp(contrastive_loss, 0, 10.0)

                        total_loss = lambda_recon * recon_loss + ml_loss + lambda_hier * hier_loss + lambda_contrastive * contrastive_loss

                        val_losses.append(total_loss.item())

            avg_val_loss = float(np.mean(val_losses)) if val_losses else avg_train_loss
        else:
            avg_val_loss = avg_train_loss

        if (epoch + 1) % max(1, epochs // 20) == 0:
            if val_loader is not None and len(val_loader) > 0:
                spinner.text = f"Epoch {epoch+1}/{epochs} | Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f}"
            else:
                spinner.text = f"Epoch {epoch+1}/{epochs} | Train: {avg_train_loss:.6f}"

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

    calibration_loader = val_loader if val_loader is not None and len(val_loader) > 0 else train
    thresholds = calibrate_thresholds(model, calibration_loader)
    if thresholds is not None:
        threshold_tensor = torch.from_numpy(thresholds).to(device)
        if hasattr(model, "decision_thresholds") and model.decision_thresholds.shape[0] == threshold_tensor.shape[0]:
            model.decision_thresholds.data.copy_(threshold_tensor)
        else:
            model.register_buffer("decision_thresholds", threshold_tensor, persistent=False)

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
            thresholds = getattr(model, "decision_thresholds", None)
            batch_preds = np.zeros((cls_tokens.shape[0], len(node_names)))
            for name, probs in propagated.items():
                if name in model.node_to_index:
                    idx = model.node_to_index[name]
                    threshold = 0.5
                    if thresholds is not None and idx < thresholds.shape[0]:
                        threshold = float(thresholds[idx].item())
                    batch_preds[:, idx] = (probs >= threshold).cpu().numpy().astype(int).squeeze()
            
            all_preds.append(batch_preds)
            all_targets.append(targets.cpu().numpy())
    
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    
    print("\n=== Per-Class Metrics ===")
    print(f"{'Class':<25} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 67)

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
                support = int(y_true.sum())
                print(f"{name:<25} {prec:>10.3f} {rec:>10.3f} {f1:>10.3f} {support:>10d}")
    
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
        f.write(f"{'Class':<25} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}\n")
        f.write("-" * 67 + "\n")
        
        for i, name in enumerate(node_names):
            if i < all_targets.shape[1]:
                y_true = all_targets[:, i]
                y_pred = all_preds[:, i]
                if y_true.sum() > 0:
                    prec, rec, f1, _ = precision_recall_fscore_support(
                        y_true, y_pred, average='binary', zero_division=0
                    )
                    support = int(y_true.sum())
                    f.write(f"{name:<25} {prec:>10.3f} {rec:>10.3f} {f1:>10.3f} {support:>10d}\n")
        
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
    print("Hierarchical Transformer for Log Analysis")
    print("=" * 50)
    print("Note: This script requires pre-generated embeddings.")
    print("If you see 'Embeddings not found' errors, please generate embeddings first:")
    print("  - FastText: python src/fasttext_embedding.py --output-subdir fasttext")
    print("  - Word2Vec: python src/word2vec_embedding_thesis.py --output-subdir word2vec") 
    print("  - LogBERT: python src/logbert_embeddings.py --output-subdir logbert")
    print("=" * 50)
    
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

            def stratified_split(indices: torch.Tensor, train_ratio: float = TRAIN_SPLIT):
                n = len(indices)
                if n == 0:
                    empty = torch.tensor([], dtype=torch.long)
                    return empty, empty

                n_train = int(round(train_ratio * n))
                if n > 1:
                    n_train = max(1, min(n_train, n - 1))
                else:
                    n_train = n

                perm = torch.randperm(n)
                train_split = indices[perm[:n_train]]
                test_split = indices[perm[n_train:]] if n_train < n else torch.tensor([], dtype=torch.long)
                return train_split, test_split

            def concat_indices(chunks: List[torch.Tensor]) -> torch.Tensor:
                filtered = [chunk for chunk in chunks if len(chunk) > 0]
                if not filtered:
                    return torch.tensor([], dtype=torch.long)
                return torch.cat(filtered)

            normal_train, normal_test = stratified_split(normal_idx)

            if len(normal_train) == 0:
                print("Insufficient normal samples for training; defaulting to all normal logs.")
                normal_train = normal_idx

            if len(anomaly_idx) > 0:
                anomaly_total = len(anomaly_idx)
                anomaly_train_target = int(round(TRAIN_SPLIT * anomaly_total))
                if anomaly_total > 1:
                    anomaly_train_target = max(1, min(anomaly_train_target, anomaly_total - 1))
                else:
                    anomaly_train_target = anomaly_total

                anomaly_idx_np = anomaly_idx.cpu().numpy()
                coverage_rng = np.random.default_rng(42)
                coverage_set: set[int] = set()

                if all_targets.numel() > 0:
                    label_tensor = all_targets.to(dtype=torch.bool, device="cpu")
                    for class_idx in range(label_tensor.shape[1]):
                        class_indices = torch.nonzero(label_tensor[:, class_idx], as_tuple=True)[0].cpu().numpy()
                        if class_indices.size == 0:
                            continue
                        class_anomalies = np.intersect1d(class_indices, anomaly_idx_np, assume_unique=False)
                        total_class = class_anomalies.size
                        if total_class == 0:
                            continue
                        max_fractional = int(total_class * MAX_CLASS_TRAIN)
                        quota_target = int(round(TRAIN_SPLIT * total_class))
                        train_quota = min(total_class, max(1, min(quota_target, max_fractional)))
                        train_quota = min(train_quota, total_class)
                        sampled = coverage_rng.choice(class_anomalies, size=train_quota, replace=False)
                        coverage_set.update(int(x) for x in sampled)

                coverage_indices = np.array(sorted(coverage_set), dtype=np.int64)
                coverage_tensor = torch.from_numpy(coverage_indices).to(dtype=torch.long, device=anomaly_idx.device)
                if coverage_tensor.numel() > anomaly_train_target:
                    perm = coverage_tensor[torch.randperm(coverage_tensor.numel())]
                    anomaly_train = perm[:anomaly_train_target]
                    overflow = perm[anomaly_train_target:]
                    remaining_base = torch.from_numpy(
                        np.setdiff1d(anomaly_idx_np, anomaly_train.cpu().numpy(), assume_unique=False)
                    ).to(dtype=torch.long, device=anomaly_idx.device)
                    remaining_anomalies = torch.cat([overflow, remaining_base])
                else:
                    remaining_pool_np = np.setdiff1d(anomaly_idx_np, coverage_indices, assume_unique=False)
                    remaining_pool = torch.from_numpy(remaining_pool_np).to(dtype=torch.long, device=anomaly_idx.device)
                    if remaining_pool.numel() > 0:
                        remaining_pool = remaining_pool[torch.randperm(remaining_pool.numel())]
                    additional_needed = max(0, anomaly_train_target - coverage_tensor.numel())
                    additional = remaining_pool[:additional_needed]
                    anomaly_train = concat_indices([coverage_tensor, additional]) if additional.numel() else coverage_tensor
                    remaining_anomalies = remaining_pool[additional_needed:]

                anomaly_test = remaining_anomalies
            else:
                anomaly_train = torch.tensor([], dtype=torch.long)
                anomaly_test = torch.tensor([], dtype=torch.long)

            train_indices = concat_indices([normal_train, anomaly_train])
            test_indices = concat_indices([normal_test, anomaly_test])

            if len(train_indices) == 0:
                raise ValueError("Dataset does not contain normal samples for unsupervised training.")

            train_set = torch.utils.data.Subset(dataset, train_indices)
            test_set = torch.utils.data.Subset(dataset, test_indices if len(test_indices) > 0 else train_indices)
            train_loader = DataLoader(train_set, batch_size=128, shuffle=True,  num_workers=8, pin_memory=(device.type=="cuda"), persistent_workers=True, prefetch_factor=4)
            test_loader  = DataLoader(test_set,  batch_size=128, shuffle=False, num_workers=8, pin_memory=(device.type=="cuda"), persistent_workers=True, prefetch_factor=4)

            print(f"Train: {len(train_set)}, Test: {len(test_set)}")

            model = HierarchicalTransformer(hierarchy).to(device)
            if device.type == "cuda":
                try:
                    import triton  # type: ignore  # noqa: F401
                    model = torch.compile(model, mode="max-autotune")
                except Exception:
                    print("Triton not available or compile failed; using eager mode.")
            model = train_model(model, train_loader, val_loader=None, epochs=10)

            evaluate_model(model, test_loader, log_type, embedding_type)

            Path("models").mkdir(exist_ok=True)
            model_path = f"models/hierarchical_{log_type}_{embedding_type}.pth"
            torch.save(model.state_dict(), model_path)
            print(f"\nModel saved to {model_path}")

if __name__ == "__main__":
    main()
