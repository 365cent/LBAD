"""
 f_AnoGAN downstream module for Transformer‑based log‑embedding pipelines
 -----------------------------------------------------------------------
 •  Works on *fixed‑length embedding vectors* (e.g. 2314‑D Enhanced LogBERT, 768‑D BERT CLS, 300‑D FastText).
 •  Implements **f‑AnoGAN** [Schlegl et al., 2019] with:
     – Wasserstein GAN + Gradient Penalty (WGAN‑GP) for stable training
     – Encoder (E) trained via feature matching to learn x→z mapping after G,D converged
     – Optional **bidirectional** update (BiGAN) to co‑train E alongside G,D (set `joint_training=True`)
 •  Returns *anomaly score* per sample = λ⋅‖x − G(E(x))‖₂ + (1−λ)⋅‖φ_D(x) − φ_D(G(E(x)))‖₂
     – φ_D(·) is penultimate layer feature representation of discriminator
 •  Ready to plug into existing Transformer project:
     >>> from fanogan_downstream import FANOGAN
     >>> gan = FANOGAN(embed_dim=2314, latent_dim=128).fit(train_embeddings)
     >>> scores = gan.score(test_embeddings)
     >>> anomalies = scores > np.percentile(scores, 95)  # flag top‑5 % as anomalous
 """

 from __future__ import annotations

 import time
 from pathlib import Path
 from dataclasses import dataclass
 from typing import Tuple, Optional, List

 import numpy as np
 import torch
 import torch.nn as nn
 import torch.optim as optim
 import torch.nn.functional as F
 from torch.utils.data import DataLoader, TensorDataset

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
         ds = DataLoader(TensorDataset(x), batch_size=self.cfg.batch_size, shuffle=True, drop_last=True)
         opt_G = optim.Adam(self.G.parameters(), lr=self.cfg.lr, betas=(0.5, 0.9))
         opt_D = optim.Adam(self.D.parameters(), lr=self.cfg.lr, betas=(0.5, 0.9))
         opt_E = optim.Adam(self.E.parameters(), lr=self.cfg.lr, betas=(0.5, 0.9))
         one = torch.tensor(1.0, device=self.cfg.device)
         mone = -one
         start = time.time()
         for epoch in range(self.cfg.n_epochs):
             for i, (x_real,) in enumerate(ds):
                 x_real = x_real.to(self.cfg.device)
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
             if (epoch + 1) % 20 == 0:
                 print(f"[f‑AnoGAN] epoch {epoch+1}/{self.cfg.n_epochs} | D {loss_D.item():.3f} | G {loss_G.item():.3f} | E {loss_E.item():.3f}")
         print(f"Training finished in {(time.time()-start)/60:.1f} min.")
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
 #  Simple CLI for standalone usage
 # ---------------------------------------------------------------------------
 if __name__ == '__main__':
     import argparse, pickle
     parser = argparse.ArgumentParser("f‑AnoGAN downstream anomaly detector")
     parser.add_argument('--embeddings', required=True, help='Path to *.pkl containing numpy embeddings')
     parser.add_argument('--save-model', default=None, help='Path to save trained FANOGAN model')
     parser.add_argument('--latent-dim', type=int, default=128)
     args = parser.parse_args()
     with open(args.embeddings, 'rb') as f:
         emb = pickle.load(f)
     gan = FANOGAN(embed_dim=emb.shape[1], latent_dim=args.latent_dim).fit(emb)
     scores = gan.score(emb)
     print('Score summary: μ %.4f | σ %.4f' % (scores.mean(), scores.std()))
     if args.save_model:
         gan.save(args.save_model)
         print('Model saved to', args.save_model)
