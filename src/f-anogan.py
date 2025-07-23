#!/usr/bin/env python3
"""
f-AnoGAN Anomaly Detection Evaluation Pipeline
==============================================

Implements f-AnoGAN (Schlegl et al., 2019) for anomaly detection on log embeddings.

Features:
- Wasserstein GAN + Gradient Penalty (WGAN-GP) for stable training
- Encoder (E) trained via feature matching to learn x→z mapping after G,D converged  
- Optional bidirectional update (BiGAN) to co-train E alongside G,D
- Returns anomaly score per sample = λ·‖x - G(E(x))‖₂ + (1-λ)·‖φ_D(x) - φ_D(G(E(x)))‖₂

Usage:
    python src/f-anogan.py --log-type wp-error
    python src/f-anogan.py --log-type wp-access --latent-dim 256 --epochs 300
"""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Import system configuration from transformer module
import sys
sys.path.append('.')
from src.transformer import detect_system_resources, SystemConfig

# ---------------------------------------------------------------------------
#  NETWORK BUILDING BLOCKS
# ---------------------------------------------------------------------------

class GBlock(nn.Module):
    """Single hidden block for the Generator"""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )
    def forward(self, z):
        return self.block(z)

class DBlock(nn.Module):
    """Single hidden block for the Discriminator / Critic"""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )
    def forward(self, x):
        return self.block(x)

# ---------------------------------------------------------------------------
#  f‑AnoGAN components
# ---------------------------------------------------------------------------

class Generator(nn.Module):
    def __init__(self, latent_dim: int, embed_dim: int, hidden: int = 512, n_layers: int = 3):
        super().__init__()
        layers: List[nn.Module] = []
        dims = [latent_dim] + [hidden] * n_layers + [embed_dim]
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            layers.append(GBlock(d_in, d_out))
        layers[-1] = nn.Linear(dims[-2], dims[-1])  # last layer w/o activation
        self.net = nn.Sequential(*layers)
    def forward(self, z):
        return self.net(z)

class Discriminator(nn.Module):
    def __init__(self, embed_dim: int, hidden: int = 512, n_layers: int = 3):
        super().__init__()
        layers: List[nn.Module] = []
        dims = [embed_dim] + [hidden] * n_layers
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            layers.append(DBlock(d_in, d_out))
        self.feature_extractor = nn.Sequential(*layers)
        self.output = nn.Linear(dims[-1], 1)
    def forward(self, x):
        feat = self.feature_extractor(x)
        return self.output(feat), feat  # score, deep feature

class Encoder(nn.Module):
    """Maps embedding → latent vector"""
    def __init__(self, embed_dim: int, latent_dim: int, hidden: int = 512, n_layers: int = 3):
        super().__init__()
        layers: List[nn.Module] = []
        dims = [embed_dim] + [hidden] * n_layers + [latent_dim]
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            layers.append(DBlock(d_in, d_out))
        layers[-1] = nn.Linear(dims[-2], dims[-1])
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

# ---------------------------------------------------------------------------
#  Training utilities
# ---------------------------------------------------------------------------

def gradient_penalty(D: Discriminator, real: torch.Tensor, fake: torch.Tensor, device: torch.device):
    batch = real.size(0)
    eps = torch.rand(batch, 1, device=device)
    eps = eps.expand_as(real)
    inter = eps * real + (1 - eps) * fake
    inter.requires_grad_(True)
    d_inter, _ = D(inter)
    grad = torch.autograd.grad(outputs=d_inter, inputs=inter,
                               grad_outputs=torch.ones_like(d_inter),
                               create_graph=True, retain_graph=True, only_inputs=True)[0]
    gp = ((grad.norm(2, dim=1) - 1) ** 2).mean()
    return gp

@dataclass
class FANOGANConfig:
    embed_dim: int
    latent_dim: int = 128
    batch_size: int = 256
    lr: float = 1e-4
    n_critic: int = 5
    gp_lambda: float = 10.0
    n_epochs: int = 200
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    joint_training: bool = False  # if True train E together with G,D (BiGAN‑style)
    lambda_recon: float = 0.9  # λ in anomaly score

class FANOGAN:
    """High‑level wrapper for f‑AnoGAN training & inference"""
    def __init__(self, embed_dim: int, latent_dim: int = 128, **kwargs):
        self.cfg = FANOGANConfig(embed_dim=embed_dim, latent_dim=latent_dim, **kwargs)
        self.G = Generator(self.cfg.latent_dim, self.cfg.embed_dim).to(self.cfg.device)
        self.D = Discriminator(self.cfg.embed_dim).to(self.cfg.device)
        self.E = Encoder(self.cfg.embed_dim, self.cfg.latent_dim).to(self.cfg.device)

    # -------------------------- TRAIN ------------------------------------
    def fit(self, embeddings: np.ndarray):
        x = torch.from_numpy(embeddings).float()
        
        # Adjust batch size if dataset is too small
        effective_batch_size = min(self.cfg.batch_size, len(embeddings))
        if len(embeddings) < self.cfg.batch_size:
            print(f"⚠️  Dataset too small ({len(embeddings)} samples), adjusting batch size to {effective_batch_size}")
        
        ds = DataLoader(TensorDataset(x), batch_size=effective_batch_size, shuffle=True, drop_last=False)
        opt_G = optim.Adam(self.G.parameters(), lr=self.cfg.lr, betas=(0.5, 0.9))
        opt_D = optim.Adam(self.D.parameters(), lr=self.cfg.lr, betas=(0.5, 0.9))
        opt_E = optim.Adam(self.E.parameters(), lr=self.cfg.lr, betas=(0.5, 0.9))
        
        # Initialize loss variables
        loss_D = loss_G = loss_E = torch.tensor(0.0, device=self.cfg.device)
        
        start = time.time()
        for epoch in range(self.cfg.n_epochs):
            epoch_has_batches = False
            for i, (x_real,) in enumerate(ds):
                epoch_has_batches = True
                x_real = x_real.to(self.cfg.device)
                
                # Skip if batch is too small for BatchNorm
                if x_real.size(0) < 2:
                    print(f"⚠️  Skipping batch with size {x_real.size(0)} (too small for BatchNorm)")
                    continue
                
                # ----------------- Train D ---------------------
                for _ in range(self.cfg.n_critic):
                    z = torch.randn(x_real.size(0), self.cfg.latent_dim, device=self.cfg.device)
                    with torch.no_grad():
                        x_fake = self.G(z)
                    d_real, _ = self.D(x_real)
                    d_fake, _ = self.D(x_fake)
                    gp = gradient_penalty(self.D, x_real, x_fake, self.cfg.device)
                    loss_D = d_fake.mean() - d_real.mean() + self.cfg.gp_lambda * gp
                    opt_D.zero_grad(); loss_D.backward(); opt_D.step()
                # ----------------- Train G ---------------------
                z = torch.randn(x_real.size(0), self.cfg.latent_dim, device=self.cfg.device)
                x_fake = self.G(z)
                d_fake, _ = self.D(x_fake)
                loss_G = -d_fake.mean()
                opt_G.zero_grad(); loss_G.backward(); opt_G.step()
                # ----------------- Train E (feature‑matching) --
                z_enc = self.E(x_real)
                x_rec = self.G(z_enc)
                _, feat_real = self.D(x_real.detach())
                _, feat_rec  = self.D(x_rec)
                loss_rec = F.mse_loss(x_rec, x_real)
                loss_fm  = F.mse_loss(feat_rec, feat_real)
                loss_E = loss_rec + 0.1 * loss_fm
                opt_E.zero_grad(); loss_E.backward(); opt_E.step()
                # Optional joint training of E with G,D
                if self.cfg.joint_training:
                    # back‑propagate into G as well
                    opt_G.step()
            
            if not epoch_has_batches:
                print(f"⚠️  No valid batches in epoch {epoch+1}, dataset too small for training")
                break
                
            if (epoch + 1) % 20 == 0:
                print(f"[f‑AnoGAN] epoch {epoch+1}/{self.cfg.n_epochs} | D {loss_D.item():.3f} | G {loss_G.item():.3f} | E {loss_E.item():.3f}")
        print(f"Training finished in {(time.time()-start)/60:.1f} min.")
        return self

    # -------------------------- INFERENCE -------------------------------
    @torch.no_grad()
    def score(self, embeddings: np.ndarray) -> np.ndarray:
        """Return anomaly score per embedding."""
        self.G.eval(); self.D.eval(); self.E.eval()
        x = torch.from_numpy(embeddings).float().to(self.cfg.device)
        z = self.E(x)
        x_rec = self.G(z)
        rec_err = F.mse_loss(x_rec, x, reduction='none').mean(dim=1)
        _, feat_real = self.D(x)
        _, feat_rec  = self.D(x_rec)
        fm_err = F.mse_loss(feat_rec, feat_real, reduction='none').mean(dim=1)
        scores = self.cfg.lambda_recon * rec_err + (1 - self.cfg.lambda_recon) * fm_err
        return scores.cpu().numpy()

    # -------------------------- SAVE / LOAD -----------------------------
    def save(self, path: str | Path):
        torch.save({
            'cfg': self.cfg,
            'G': self.G.state_dict(),
            'D': self.D.state_dict(),
            'E': self.E.state_dict()
        }, Path(path))
    @classmethod
    def load(cls, path: str | Path) -> 'FANOGAN':
        chk = torch.load(Path(path), map_location='cpu')
        obj = cls(chk['cfg'].embed_dim, chk['cfg'].latent_dim)
        obj.cfg = chk['cfg']
        obj.G.load_state_dict(chk['G']); obj.D.load_state_dict(chk['D']); obj.E.load_state_dict(chk['E'])
        obj.G.to(obj.cfg.device); obj.D.to(obj.cfg.device); obj.E.to(obj.cfg.device)
        obj.G.eval(); obj.D.eval(); obj.E.eval()
        return obj

# ---------------------------------------------------------------------------
#  ANOMALY DETECTION EVALUATOR
# ---------------------------------------------------------------------------

class FANOGANEvaluator:
    """f-AnoGAN anomaly detection evaluator following project evaluation standards"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.device = torch.device(config.device)
    
    def load_embeddings_and_labels(self, log_type: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Load LogBERT embeddings and true labels for anomaly detection"""
        
        embeddings_dir = Path("embeddings") / log_type
        
        # Load embeddings
        log_file = embeddings_dir / f"log_{log_type}.pkl"
        if not log_file.exists():
            raise FileNotFoundError(f"Embeddings not found: {log_file}")
        
        print(f"📂 Loading embeddings from {log_file}")
        with open(log_file, 'rb') as f:
            X = pickle.load(f)
        
        # Load labels
        label_file = embeddings_dir / f"label_{log_type}.pkl"
        if not label_file.exists():
            raise FileNotFoundError(f"Labels not found: {label_file}")
        
        print(f"📂 Loading labels from {label_file}")
        with open(label_file, 'rb') as f:
            label_data = pickle.load(f)
        
        y_true = label_data["vectors"]
        classes = label_data["classes"]
        
        # For anomaly detection, convert multi-label to anomaly binary (any attack = anomaly)
        # Assuming "normal" is the first class or no labels means normal
        if len(classes) > 1:
            # Any positive label indicates anomalous behavior
            anomaly_labels = (y_true.sum(axis=1) > 0).astype(int)
            print(f"✅ Converted multi-label to binary anomaly detection")
            print(f"   Normal samples: {(anomaly_labels == 0).sum():,}")
            print(f"   Anomaly samples: {(anomaly_labels == 1).sum():,}")
        else:
            anomaly_labels = y_true.flatten()
        
        print(f"✅ Loaded dataset: {len(X):,} samples with {X.shape[1]}D embeddings")
        print(f"📊 Anomaly rate: {anomaly_labels.mean():.3f}")
        
        # Check for extreme imbalance
        normal_count = (anomaly_labels == 0).sum()
        total_count = len(anomaly_labels)
        
        # Warning for extreme imbalance
        if normal_count < 50:
            print(f"⚠️  EXTREME IMBALANCE DETECTED!")
            print(f"   Only {normal_count} normal samples out of {total_count} total")
            print(f"   This dataset may not be suitable for unsupervised anomaly detection")
            print(f"   Consider using supervised classification instead")
        
        return X, anomaly_labels, classes
    
    def train_fanogan(self, X: np.ndarray, y_labels: np.ndarray, 
                     latent_dim: int = 128, n_epochs: int = 200, 
                     batch_size: int = 256) -> FANOGAN:
        """Train f-AnoGAN model on normal samples only"""
        
        # Filter to normal samples only (label = 0) for unsupervised training
        normal_mask = y_labels == 0
        X_normal = X[normal_mask]
        
        print(f"🤖 Training f-AnoGAN on {len(X_normal):,} normal samples")
        print(f"   Embedding dimension: {X.shape[1]}")
        print(f"   Latent dimension: {latent_dim}")
        print(f"   Epochs: {n_epochs}")
        print(f"   Batch size: {batch_size}")
        
        # Check if we have enough normal samples
        if len(X_normal) < 2:
            raise ValueError(f"Insufficient normal samples for training: {len(X_normal)} < 2. "
                           "f-AnoGAN requires at least 2 normal samples for BatchNorm layers.")
        
        if len(X_normal) < 10:
            print(f"⚠️  Very few normal samples ({len(X_normal)}). Consider:")
            print(f"   - Using a different anomaly detection method")
            print(f"   - Rebalancing your dataset") 
            print(f"   - Using data augmentation")
            print(f"   Proceeding with training but results may be unreliable...")
        
        # Initialize and train f-AnoGAN
        fanogan = FANOGAN(
            embed_dim=X.shape[1],
            latent_dim=latent_dim,
            n_epochs=n_epochs,
            batch_size=batch_size,
            device=self.config.device
        )
        
        fanogan.fit(X_normal)
        return fanogan
    
    def evaluate_anomaly_detection(self, fanogan: FANOGAN, X: np.ndarray, 
                                  y_true: np.ndarray) -> Dict[str, Any]:
        """Evaluate anomaly detection performance"""
        from sklearn.metrics import (
            roc_auc_score, average_precision_score, precision_recall_curve,
            roc_curve, accuracy_score, precision_score, recall_score, f1_score
        )
        
        print(f"🔍 Computing anomaly scores...")
        anomaly_scores = fanogan.score(X)
        
        # ROC curve and AUC
        fpr, tpr, roc_thresholds = roc_curve(y_true, anomaly_scores)
        roc_auc = roc_auc_score(y_true, anomaly_scores)
        
        # Precision-Recall curve and AP
        precision, recall, pr_thresholds = precision_recall_curve(y_true, anomaly_scores)
        avg_precision = average_precision_score(y_true, anomaly_scores)
        
        # Find optimal threshold using F1 score
        f1_scores = []
        thresholds_f1 = np.linspace(anomaly_scores.min(), anomaly_scores.max(), 100)
        
        for threshold in thresholds_f1:
            y_pred = (anomaly_scores >= threshold).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            f1_scores.append(f1)
        
        best_f1_idx = np.argmax(f1_scores)
        best_threshold = thresholds_f1[best_f1_idx]
        best_f1 = f1_scores[best_f1_idx]
        
        # Predictions with optimal threshold
        y_pred_optimal = (anomaly_scores >= best_threshold).astype(int)
        
        # Calculate metrics with optimal threshold
        metrics = {
            'roc_auc': float(roc_auc),
            'average_precision': float(avg_precision),
            'best_threshold': float(best_threshold),
            'best_f1': float(best_f1),
            'accuracy': float(accuracy_score(y_true, y_pred_optimal)),
            'precision': float(precision_score(y_true, y_pred_optimal, zero_division=0)),
            'recall': float(recall_score(y_true, y_pred_optimal, zero_division=0)),
            'f1_score': float(f1_score(y_true, y_pred_optimal, zero_division=0)),
            'anomaly_rate_true': float(y_true.mean()),
            'anomaly_rate_pred': float(y_pred_optimal.mean()),
            'n_samples': len(y_true),
            'score_mean': float(anomaly_scores.mean()),
            'score_std': float(anomaly_scores.std()),
            'score_min': float(anomaly_scores.min()),
            'score_max': float(anomaly_scores.max()),
        }
        
        # Additional threshold analysis
        percentiles = [90, 95, 99]
        for p in percentiles:
            threshold_p = np.percentile(anomaly_scores, p)
            y_pred_p = (anomaly_scores >= threshold_p).astype(int)
            metrics[f'threshold_p{p}'] = float(threshold_p)
            metrics[f'precision_p{p}'] = float(precision_score(y_true, y_pred_p, zero_division=0))
            metrics[f'recall_p{p}'] = float(recall_score(y_true, y_pred_p, zero_division=0))
            metrics[f'f1_p{p}'] = float(f1_score(y_true, y_pred_p, zero_division=0))
        
        return metrics, anomaly_scores, y_pred_optimal
    
    def print_results(self, metrics: Dict[str, Any]):
        """Print comprehensive anomaly detection results"""
        
        print(f"\n📊 f-AnoGAN ANOMALY DETECTION RESULTS")
        print("=" * 60)
        print(f"Test samples:     {metrics['n_samples']:,}")
        print(f"Anomaly rate:     {metrics['anomaly_rate_true']:.4f}")
        print("")
        
        print("MAIN METRICS:")
        print("-" * 40)
        print(f"ROC AUC:          {metrics['roc_auc']:.4f}")
        print(f"Average Precision: {metrics['average_precision']:.4f}")
        print(f"Best F1 Score:    {metrics['best_f1']:.4f}")
        print(f"Best Threshold:   {metrics['best_threshold']:.4f}")
        print("")
        
        print("OPTIMAL THRESHOLD PERFORMANCE:")
        print("-" * 40)
        print(f"Accuracy:         {metrics['accuracy']:.4f}")
        print(f"Precision:        {metrics['precision']:.4f}")
        print(f"Recall:           {metrics['recall']:.4f}")
        print(f"F1 Score:         {metrics['f1_score']:.4f}")
        print(f"Pred Anomaly Rate: {metrics['anomaly_rate_pred']:.4f}")
        print("")
        
        print("SCORE DISTRIBUTION:")
        print("-" * 40)
        print(f"Mean:             {metrics['score_mean']:.4f}")
        print(f"Std:              {metrics['score_std']:.4f}")
        print(f"Min:              {metrics['score_min']:.4f}")
        print(f"Max:              {metrics['score_max']:.4f}")
        print("")
        
        print("PERCENTILE THRESHOLDS:")
        print("-" * 50)
        print(f"{'Percentile':<10} {'Threshold':<12} {'Precision':<10} {'Recall':<8} {'F1':<8}")
        print("-" * 50)
        for p in [90, 95, 99]:
            thresh = metrics[f'threshold_p{p}']
            prec = metrics[f'precision_p{p}']
            rec = metrics[f'recall_p{p}']
            f1 = metrics[f'f1_p{p}']
            print(f"{p}%{'':<7} {thresh:<12.4f} {prec:<10.3f} {rec:<8.3f} {f1:<8.3f}")
    
    def save_results(self, metrics: Dict[str, Any], anomaly_scores: np.ndarray,
                    y_pred: np.ndarray, y_true: np.ndarray, log_type: str,
                    fanogan_model: FANOGAN):
        """Save evaluation results and trained model"""
        
        output_dir = Path("results") / log_type
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save evaluation results
        results = {
            'metrics': metrics,
            'anomaly_scores': anomaly_scores.astype(np.float32),
            'predictions': y_pred.astype(np.int8),
            'true_labels': y_true.astype(np.int8),
            'evaluation_type': 'fanogan_anomaly_detection',
            'config': self.config.__dict__,
            'model_config': fanogan_model.cfg.__dict__,
            'timestamp': time.time()
        }
        
        results_file = output_dir / f"fanogan_evaluation_{log_type}_{self.config.node_name}_{self.config.job_id}.pkl"
        with open(results_file, 'wb') as f:
            pickle.dump(results, f)
        
        # Save trained model
        model_file = output_dir / f"fanogan_model_{log_type}_{self.config.node_name}_{self.config.job_id}.pth"
        fanogan_model.save(model_file)
        
        print(f"💾 Results saved to: {results_file}")
        print(f"💾 Model saved to: {model_file}")
        
        return results_file, model_file


# ---------------------------------------------------------------------------
#  MAIN EVALUATION PIPELINE
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="f-AnoGAN anomaly detection evaluation")
    parser.add_argument("--log-type", type=str, required=True,
                       help="Log type to evaluate (e.g., wp-access, wp-error)")
    parser.add_argument("--latent-dim", type=int, default=128,
                       help="Latent dimension for f-AnoGAN")
    parser.add_argument("--epochs", type=int, default=200,
                       help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=256,
                       help="Batch size for training")
    parser.add_argument("--lambda-recon", type=float, default=0.9,
                       help="Reconstruction loss weight in anomaly score")
    
    args = parser.parse_args()
    
    # Detect system configuration
    config = detect_system_resources()
    
    print("🚀 f-AnoGAN Anomaly Detection Evaluation")
    print("=" * 60)
    print(f"Log type: {args.log_type}")
    print(f"Device: {config.device}")
    print(f"Node: {config.node_name} | Job: {config.job_id}")
    print(f"Latent dim: {args.latent_dim}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Lambda recon: {args.lambda_recon}")
    print("")
    
    try:
        # Initialize evaluator
        evaluator = FANOGANEvaluator(config)
        
        # 1. Load embeddings and labels
        X, y_true, classes = evaluator.load_embeddings_and_labels(args.log_type)
        
        # 2. Train f-AnoGAN model
        fanogan_model = evaluator.train_fanogan(
            X, y_true, 
            latent_dim=args.latent_dim,
            n_epochs=args.epochs,
            batch_size=args.batch_size
        )
        
        # Update lambda_recon if specified
        fanogan_model.cfg.lambda_recon = args.lambda_recon
        
        # 3. Evaluate anomaly detection
        metrics, anomaly_scores, y_pred = evaluator.evaluate_anomaly_detection(
            fanogan_model, X, y_true
        )
        
        # 4. Print results
        evaluator.print_results(metrics)
        
        # 5. Save results and model
        results_file, model_file = evaluator.save_results(
            metrics, anomaly_scores, y_pred, y_true, args.log_type, fanogan_model
        )
        
        print(f"\n✅ Evaluation completed for {args.log_type}")
        print(f"📁 Results: {results_file}")
        print(f"📁 Model: {model_file}")
        
        # Summary
        print(f"\n🎯 SUMMARY")
        print(f"   ROC AUC:  {metrics['roc_auc']:.4f}")
        print(f"   Avg Precision: {metrics['average_precision']:.4f}")
        print(f"   Best F1:  {metrics['best_f1']:.4f}")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
