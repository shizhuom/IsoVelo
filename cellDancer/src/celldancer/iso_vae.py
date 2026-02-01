"""Isoform-level VAE for IsoVelo.

Input dataframe columns: [gene_name, isoform_name, unspice, splice, cellID, clusters, embedding1, embedding2]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


@dataclass
class IsoVeloMatrices:
    U: np.ndarray  # (N_cells, G_genes)
    S: np.ndarray  # (N_cells, sum K_g)
    gene_names: List[str]
    isoform_names: List[str]
    cell_ids: List[str]
    isoform_gene_index: np.ndarray  # (sum K_g, ) map isoform -> gene index
    gene_to_isoform_indices: List[np.ndarray]


def _pivot_matrices(df: pd.DataFrame) -> IsoVeloMatrices:
    required = {
        "gene_name",
        "isoform_name",
        "unsplice",
        "splice",
        "cellID",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Gene-level unspliced matrix U (sum isoforms within gene)
    gene_pivot = (
        df.pivot_table(
            index="cellID",
            columns="gene_name",
            values="unsplice",
            aggfunc="sum",
            fill_value=0.0,
        )
        .sort_index(axis=0)
        .sort_index(axis=1)
    )

    # Isoform-level spliced matrix S (each isoform as its own column)
    iso_pivot = (
        df.pivot_table(
            index="cellID",
            columns="isoform_name",
            values="splice",
            aggfunc="sum",
            fill_value=0.0,
        )
        .sort_index(axis=0)
        .sort_index(axis=1)
    )

    # Align cell order across U and S
    common_cells = gene_pivot.index.intersection(iso_pivot.index)
    gene_pivot = gene_pivot.loc[common_cells]
    iso_pivot = iso_pivot.loc[common_cells]

    gene_names = list(gene_pivot.columns)
    isoform_names = list(iso_pivot.columns)
    cell_ids = list(common_cells)

    # Map isoforms to genes
    isoform_gene = (
        df[["isoform_name", "gene_name"]]
        .drop_duplicates(subset=["isoform_name", "gene_name"])
        .drop_duplicates(subset=["isoform_name"], keep="first")
        .set_index("isoform_name")
        .reindex(isoform_names)
    )
    isoform_gene_index = np.array(
        [gene_names.index(g) for g in isoform_gene["gene_name"].values],
        dtype=np.int64,
    )

    gene_to_isoform_indices: List[np.ndarray] = []
    for g_idx, g_name in enumerate(gene_names):
        idxs = np.where(isoform_gene_index == g_idx)[0]
        gene_to_isoform_indices.append(idxs)

    return IsoVeloMatrices(
        U=gene_pivot.to_numpy(dtype=np.float32),
        S=iso_pivot.to_numpy(dtype=np.float32),
        gene_names=gene_names,
        isoform_names=isoform_names,
        cell_ids=cell_ids,
        isoform_gene_index=isoform_gene_index,
        gene_to_isoform_indices=gene_to_isoform_indices,
    )


class IsoVeloDataset(Dataset):
    """Dataset that pivots isoform dataframe into U and S matrices."""

    def __init__(self, df: pd.DataFrame):
        self.mats = _pivot_matrices(df)
        self.U = torch.from_numpy(self.mats.U)
        self.S = torch.from_numpy(self.mats.S)

    def __len__(self) -> int:
        return self.U.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.U[idx], self.S[idx]


class MLPEncoder(nn.Module):
    def __init__(self, in_dim: int, latent_dim: int, hidden_dims: Iterable[int]):
        super().__init__()
        layers: List[nn.Module] = []
        last_dim = in_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(last_dim, h), nn.BatchNorm1d(h), nn.ReLU()])
            last_dim = h
        self.net = nn.Sequential(*layers)
        self.mu = nn.Linear(last_dim, latent_dim)
        self.logvar = nn.Linear(last_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        return self.mu(h), self.logvar(h)


class MLPDecoder(nn.Module):
    def __init__(self, latent_dim: int, out_dim: int, hidden_dims: Iterable[int]):
        super().__init__()
        layers: List[nn.Module] = []
        last_dim = latent_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(last_dim, h), nn.BatchNorm1d(h), nn.ReLU()])
            last_dim = h
        layers.append(nn.Linear(last_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class IsoVeloVAE(nn.Module):
    """VAE for isoform RNA velocity.

    Returns:
        alpha: (N, G)
        beta: (N, sum K_g)
        gamma: (N, sum K_g)
        tau: (N, 1)
        u_hat: (N, G)
        s_hat: (N, sum K_g)
    """

    def __init__(
        self,
        n_genes: int,
        n_isoforms: int,
        isoform_gene_index: np.ndarray,
        gene_to_isoform_indices: List[np.ndarray],
        latent_dim: int = 50,
        encoder_hidden: Iterable[int] = (256, 128),
        decoder_hidden: Iterable[int] = (128, 256),
    ) -> None:
        super().__init__()
        self.n_genes = n_genes
        self.n_isoforms = n_isoforms
        self.latent_dim = latent_dim

        self.register_buffer(
            "isoform_gene_index",
            torch.from_numpy(isoform_gene_index.astype(np.int64)),
        )
        self.gene_to_isoform_indices = [
            torch.from_numpy(idxs.astype(np.int64)) for idxs in gene_to_isoform_indices
        ]

        in_dim = n_genes + n_isoforms
        self.encoder = MLPEncoder(in_dim, latent_dim, encoder_hidden)
        self.alpha_head = MLPDecoder(latent_dim, n_genes, decoder_hidden)
        self.beta_head = MLPDecoder(latent_dim, n_isoforms, decoder_hidden)
        self.gamma_head = MLPDecoder(latent_dim, n_isoforms, decoder_hidden)
        self.tau_head = MLPDecoder(latent_dim, 1, decoder_hidden)
        self.u0_head = MLPDecoder(latent_dim, n_genes, decoder_hidden)
        self.s0_head = MLPDecoder(latent_dim, n_isoforms, decoder_hidden)
        self.dispersion_head = MLPDecoder(latent_dim, n_genes + n_isoforms, decoder_hidden)
        self.zi_logits_head = MLPDecoder(latent_dim, n_genes + n_isoforms, decoder_hidden)

    @staticmethod
    def _reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self, U: torch.Tensor, S: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.cat([U, S], dim=1)
        mu, logvar = self.encoder(x)
        z = self._reparameterize(mu, logvar)

        alpha = F.softplus(self.alpha_head(z))
        beta = F.softplus(self.beta_head(z))
        gamma = F.softplus(self.gamma_head(z))
        tau = F.softplus(self.tau_head(z))
        u0 = F.softplus(self.u0_head(z))
        s0 = F.softplus(self.s0_head(z))
        dispersion = F.softplus(self.dispersion_head(z)) + 1e-6
        zi_logits = self.zi_logits_head(z)

        u_hat, s_hat = self._ode_solutions(alpha, beta, gamma, tau, u0, s0)
        return (
            alpha,
            beta,
            gamma,
            tau,
            u0,
            s0,
            dispersion,
            zi_logits,
            u_hat,
            s_hat,
            mu,
            logvar,
        )

    def _ode_solutions(
        self,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        gamma: torch.Tensor,
        tau: torch.Tensor,
        u0: torch.Tensor,
        s0: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Closed-form ODE solutions for u(t) and s_k(t)."""
        eps = 1e-6

        # beta_total per gene
        beta_total = torch.zeros(alpha.shape[0], self.n_genes, device=alpha.device)
        for g_idx, idxs in enumerate(self.gene_to_isoform_indices):
            if idxs.numel() == 0:
                continue
            beta_total[:, g_idx] = beta[:, idxs].sum(dim=1)

        beta_total = beta_total + eps
        tau_g = tau  # (N,1)

        u_hat = u0 * torch.exp(-beta_total * tau_g) + (alpha / beta_total) * (
            1.0 - torch.exp(-beta_total * tau_g)
        )

        # Map gene-level alpha, beta_total to isoforms
        alpha_i = alpha[:, self.isoform_gene_index]
        beta_total_i = beta_total[:, self.isoform_gene_index]
        u0_i = u0[:, self.isoform_gene_index]

        g = gamma
        b = beta
        t = tau_g

        exp_gt = torch.exp(-g * t)
        exp_bt = torch.exp(-beta_total_i * t)

        denom_g = g + eps
        denom_gmb = (g - beta_total_i)
        denom_gmb_safe = torch.where(denom_gmb.abs() < eps, denom_gmb + eps, denom_gmb)

        term1 = (1.0 - exp_gt) / denom_g
        term2 = (exp_bt - exp_gt) / denom_gmb_safe

        s_hat = (
            s0 * exp_gt
            + (alpha_i * b / beta_total_i) * term1
            + (b * u0_i - (alpha_i * b / beta_total_i)) * term2
        )
        return u_hat, s_hat


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()


def _physics_loss(
    alpha: torch.Tensor,
    beta: torch.Tensor,
    gamma: torch.Tensor,
    tau: torch.Tensor,
    u_hat: torch.Tensor,
    s_hat: torch.Tensor,
    isoform_gene_index: torch.Tensor,
) -> torch.Tensor:
    # ODE residuals at tau
    beta_total = torch.zeros(alpha.shape[0], alpha.shape[1], device=alpha.device)
    for g_idx in range(alpha.shape[1]):
        idxs = (isoform_gene_index == g_idx).nonzero(as_tuple=True)[0]
        if idxs.numel() == 0:
            continue
        beta_total[:, g_idx] = beta[:, idxs].sum(dim=1)

    u_res = alpha - beta_total * u_hat

    alpha_i = alpha[:, isoform_gene_index]
    beta_i = beta
    gamma_i = gamma
    u_i = u_hat[:, isoform_gene_index]

    s_res = beta_i * u_i - gamma_i * s_hat

    return F.mse_loss(u_res, torch.zeros_like(u_res)) + F.mse_loss(
        s_res, torch.zeros_like(s_res)
    )



def _nb_nll(x: torch.Tensor, mu: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Negative binomial negative log-likelihood.

    x, mu, theta are broadcastable; theta is inverse dispersion.
    """
    eps = 1e-8
    mu = mu + eps
    theta = theta + eps
    log_prob = (
        torch.lgamma(theta + x)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1.0)
        + theta * (torch.log(theta) - torch.log(theta + mu))
        + x * (torch.log(mu) - torch.log(theta + mu))
    )
    return -log_prob.mean()


def _zinb_nll(
    x: torch.Tensor,
    mu: torch.Tensor,
    theta: torch.Tensor,
    zi_logits: torch.Tensor,
) -> torch.Tensor:
    """Zero-inflated negative binomial negative log-likelihood."""
    eps = 1e-8
    mu = mu + eps
    theta = theta + eps
    log_nb = (
        torch.lgamma(theta + x)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1.0)
        + theta * (torch.log(theta) - torch.log(theta + mu))
        + x * (torch.log(mu) - torch.log(theta + mu))
    )

    logit_pi = zi_logits
    log_pi = -F.softplus(-logit_pi)
    log_1m_pi = -F.softplus(logit_pi)

    log_zero = torch.logsumexp(
        torch.stack([log_pi, log_1m_pi + log_nb], dim=0), dim=0
    )
    log_nonzero = log_1m_pi + log_nb
    log_prob = torch.where(x < 1e-8, log_zero, log_nonzero)
    return -log_prob.mean()

def train_isovelo_vae(
    df: pd.DataFrame,
    batch_size: int = 128,
    epochs: int = 50,
    lr: float = 1e-3,
    device: Optional[str] = None,
    kl_weight: float = 1.0,
    physics_weight: float = 1.0,
    use_zinb: bool = False,
) -> Tuple[IsoVeloVAE, Dict[str, List[float]], IsoVeloDataset]:
    """Train IsoVeloVAE on a dataframe.

    Returns:
        model, history, dataset
    """
    dataset = IsoVeloDataset(df)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = IsoVeloVAE(
        n_genes=dataset.mats.U.shape[1],
        n_isoforms=dataset.mats.S.shape[1],
        isoform_gene_index=dataset.mats.isoform_gene_index,
        gene_to_isoform_indices=dataset.mats.gene_to_isoform_indices,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history: Dict[str, List[float]] = {"loss": [], "recon": [], "kl": [], "physics": []}

    for _ in range(epochs):
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        total_phy = 0.0
        n_batches = 0

        for U, S in loader:
            U = U.to(device)
            S = S.to(device)

            (
                alpha,
                beta,
                gamma,
                tau,
                u0,
                s0,
                dispersion,
                zi_logits,
                u_hat,
                s_hat,
                mu,
                logvar,
            ) = model(U, S)

            theta_u = dispersion[:, : model.n_genes]
            theta_s = dispersion[:, model.n_genes :]

            if use_zinb:
                zi_u = zi_logits[:, : model.n_genes]
                zi_s = zi_logits[:, model.n_genes :]
                recon_loss = _zinb_nll(U, u_hat, theta_u, zi_u) + _zinb_nll(
                    S, s_hat, theta_s, zi_s
                )
            else:
                recon_loss = _nb_nll(U, u_hat, theta_u) + _nb_nll(S, s_hat, theta_s)
            kl = kl_divergence(mu, logvar)
            phy = _physics_loss(
                alpha,
                beta,
                gamma,
                tau,
                u_hat,
                s_hat,
                model.isoform_gene_index,
            )

            loss = recon_loss + kl_weight * kl + physics_weight * phy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl.item()
            total_phy += phy.item()
            n_batches += 1

        history["loss"].append(total_loss / max(1, n_batches))
        history["recon"].append(total_recon / max(1, n_batches))
        history["kl"].append(total_kl / max(1, n_batches))
        history["physics"].append(total_phy / max(1, n_batches))

    return model, history, dataset
