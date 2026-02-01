"""Neighborhood-based kinetic refinement for IsoVelo.

Inputs:
- latent_z: (N, 50)
- initial_params: dict with alpha, beta, gamma (arrays)
- adata: object or dict that provides U, S (arrays)
- tau: (N,) latent time for directional neighbor selection
- isoform_gene_index: (sum K_g,) mapping isoform -> gene index
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def _get_matrices(adata) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(adata, dict):
        U = adata.get("U")
        S = adata.get("S")
    else:
        U = getattr(adata, "U", None)
        S = getattr(adata, "S", None)
    if U is None or S is None:
        raise ValueError("adata must provide U and S matrices.")
    return np.asarray(U, dtype=np.float32), np.asarray(S, dtype=np.float32)


def _knn(latent_z: np.ndarray, k: int) -> np.ndarray:
    try:
        from scipy.spatial import KDTree

        tree = KDTree(latent_z)
        _, idx = tree.query(latent_z, k=k + 1)
        return idx[:, 1:]
    except Exception:
        try:
            import pynndescent

            index = pynndescent.NNDescent(latent_z, n_neighbors=k + 1, random_state=0)
            idx, _ = index.neighbor_graph
            return idx[:, 1:]
        except Exception as exc:
            raise ImportError(
                "Neighbor search requires scipy or pynndescent."
            ) from exc


def _cosine_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    num = np.sum(a * b, axis=1)
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + eps
    return num / denom


def refine_step(
    adata,
    latent_z: np.ndarray,
    initial_params: Dict[str, np.ndarray],
    tau: np.ndarray,
    isoform_gene_index: np.ndarray,
    k: int = 30,
    smooth_sigma: float = 1.0,
    gd_steps: int = 50,
    lr: float = 1e-2,
    alignment_weight: float = 1.0,
    anchor_weight: float = 0.1,
    batch_size: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Refine kinetics by aligning velocities with future-neighbor displacement.

    Returns refined alpha, beta, gamma with same shapes as initial_params.
    """
    U, S = _get_matrices(adata)
    alpha = np.asarray(initial_params["alpha"], dtype=np.float32)
    beta = np.asarray(initial_params["beta"], dtype=np.float32)
    gamma = np.asarray(initial_params["gamma"], dtype=np.float32)
    tau = np.asarray(tau, dtype=np.float32).reshape(-1)
    isoform_gene_index = np.asarray(isoform_gene_index, dtype=np.int64)

    if latent_z.shape[0] != U.shape[0]:
        raise ValueError("latent_z and U must have same number of cells.")

    neighbors = _knn(latent_z, k=k)

    # Average displacement in isoform space among future neighbors
    disp = np.zeros_like(S, dtype=np.float32)
    mask = np.zeros((S.shape[0],), dtype=np.float32)
    for i in range(S.shape[0]):
        nb = neighbors[i]
        future_mask = tau[nb] > tau[i]
        nb = nb[future_mask]
        if nb.size == 0:
            mask[i] = 0.0
            continue
        d = S[nb] - S[i]
        if smooth_sigma > 0:
            w = np.exp(
                -np.linalg.norm(latent_z[nb] - latent_z[i], axis=1)
                / max(smooth_sigma, 1e-6)
            )
            d = (w[:, None] * d).sum(axis=0) / (w.sum() + 1e-8)
        else:
            d = d.mean(axis=0)
        disp[i] = d.astype(np.float32)
        mask[i] = 1.0

    # Vectorized optimization (optionally mini-batched)
    U_t = torch.from_numpy(U)
    S_t = torch.from_numpy(S)
    disp_t = torch.from_numpy(disp)
    mask_t = torch.from_numpy(mask)
    alpha0_t = torch.from_numpy(alpha)
    beta0_t = torch.from_numpy(beta)
    gamma0_t = torch.from_numpy(gamma)
    iso_idx_t = torch.from_numpy(isoform_gene_index)

    log_alpha_t = torch.nn.Parameter(torch.log(alpha0_t + 1e-6))
    log_beta_t = torch.nn.Parameter(torch.log(beta0_t + 1e-6))
    log_gamma_t = torch.nn.Parameter(torch.log(gamma0_t + 1e-6))

    optimizer = torch.optim.Adam([log_alpha_t, log_beta_t, log_gamma_t], lr=lr)

    n_cells = U_t.shape[0]
    if batch_size is None or batch_size <= 0:
        batch_size = n_cells

    for _ in range(gd_steps):
        perm = torch.randperm(n_cells)
        for start in range(0, n_cells, batch_size):
            idx = perm[start : start + batch_size]
            u_iso = U_t[idx][:, iso_idx_t]
            s_batch = S_t[idx]
            disp_batch = disp_t[idx]
            mask_batch = mask_t[idx]

            a = torch.exp(log_alpha_t[idx])
            b = torch.exp(log_beta_t[idx])
            g = torch.exp(log_gamma_t[idx])

            v = b * u_iso - g * s_batch
            cos = F.cosine_similarity(v, disp_batch, dim=1, eps=1e-8)
            align_loss = -(cos * mask_batch).sum() / (mask_batch.sum() + 1e-8)

            anchor = (
                F.mse_loss(log_alpha_t[idx], torch.log(alpha0_t[idx] + 1e-6))
                + F.mse_loss(log_beta_t[idx], torch.log(beta0_t[idx] + 1e-6))
                + F.mse_loss(log_gamma_t[idx], torch.log(gamma0_t[idx] + 1e-6))
            )

            loss = alignment_weight * align_loss + anchor_weight * anchor
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    alpha_ref = torch.exp(log_alpha_t).detach().cpu().numpy()
    beta_ref = torch.exp(log_beta_t).detach().cpu().numpy()
    gamma_ref = torch.exp(log_gamma_t).detach().cpu().numpy()

    return {"alpha": alpha_ref, "beta": beta_ref, "gamma": gamma_ref}
