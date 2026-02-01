"""Isoform-level velocity and Isoform Switching Score (ISS)."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


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


def calculate_isoform_velocity(
    U: np.ndarray,
    S: np.ndarray,
    beta: np.ndarray,
    gamma: np.ndarray,
    isoform_gene_index: np.ndarray,
) -> np.ndarray:
    """Compute isoform velocity v_k = beta_k * u_g - gamma_k * s_k."""
    U = np.asarray(U, dtype=np.float32)
    S = np.asarray(S, dtype=np.float32)
    beta = np.asarray(beta, dtype=np.float32)
    gamma = np.asarray(gamma, dtype=np.float32)
    isoform_gene_index = np.asarray(isoform_gene_index, dtype=np.int64)

    if U.ndim != 2 or S.ndim != 2:
        raise ValueError("U and S must be 2D arrays.")
    if beta.shape != S.shape or gamma.shape != S.shape:
        raise ValueError("beta and gamma must match S shape.")

    u_iso = U[:, isoform_gene_index]
    v_iso = beta * u_iso - gamma * S
    return v_iso


def calculate_integrated_velocity(
    v_iso: np.ndarray, isoform_gene_index: np.ndarray, n_genes: int
) -> np.ndarray:
    """Sum isoform velocities to gene level."""
    v_iso = np.asarray(v_iso, dtype=np.float32)
    isoform_gene_index = np.asarray(isoform_gene_index, dtype=np.int64)

    v_gene = np.zeros((v_iso.shape[0], n_genes), dtype=np.float32)
    for g in range(n_genes):
        idx = np.where(isoform_gene_index == g)[0]
        if idx.size == 0:
            continue
        v_gene[:, g] = v_iso[:, idx].sum(axis=1)
    return v_gene


def isoform_switching_score(
    S: np.ndarray,
    v_iso: np.ndarray,
    v_gene: np.ndarray,
    isoform_gene_index: np.ndarray,
    n_genes: int,
    eps: float = 1e-8,
) -> np.ndarray:
    """Compute ISS per gene per cell.

    ISS = ||v_switch||^2 / ||v_iso||^2, with v_proj = p * V_gene.
    """
    S = np.asarray(S, dtype=np.float32)
    v_iso = np.asarray(v_iso, dtype=np.float32)
    v_gene = np.asarray(v_gene, dtype=np.float32)
    isoform_gene_index = np.asarray(isoform_gene_index, dtype=np.int64)

    iss = np.zeros((S.shape[0], n_genes), dtype=np.float32)
    for g in range(n_genes):
        idx = np.where(isoform_gene_index == g)[0]
        if idx.size == 0:
            continue
        s_g = S[:, idx]
        v_g = v_iso[:, idx]
        denom = s_g.sum(axis=1, keepdims=True)
        p = s_g / (denom + eps)
        zero_mask = denom.squeeze(1) < eps
        if np.any(zero_mask):
            p[zero_mask] = 1.0 / max(1, idx.size)
        v_proj = p * v_gene[:, g][:, None]
        v_switch = v_g - v_proj
        num = (v_switch**2).sum(axis=1)
        den = (v_g**2).sum(axis=1) + eps
        iss[:, g] = num / den
    return iss


def add_velocity_to_adata(
    adata,
    v_iso: np.ndarray,
    v_gene: np.ndarray,
    iss: np.ndarray,
    layer_prefix: str = "isoVelo",
) -> None:
    """Attach velocity outputs to AnnData-like or dict object."""
    if isinstance(adata, dict):
        adata[f"{layer_prefix}_v_iso"] = v_iso
        adata[f"{layer_prefix}_v_gene"] = v_gene
        adata[f"{layer_prefix}_iss"] = iss
        return

    layers = getattr(adata, "layers", None)
    if layers is None:
        raise ValueError("adata must have .layers or be a dict")
    layers[f"{layer_prefix}_v_iso"] = v_iso
    layers[f"{layer_prefix}_v_gene"] = v_gene
    layers[f"{layer_prefix}_iss"] = iss


def compute_isoform_velocity_and_iss(
    adata,
    refined_params: Dict[str, np.ndarray],
    isoform_gene_index: np.ndarray,
    n_genes: int,
    layer_prefix: str = "isoVelo",
) -> Dict[str, np.ndarray]:
    """Compute and store v_iso, v_gene, ISS.

    Returns a dict with v_iso, v_gene, iss.
    """
    U, S = _get_matrices(adata)
    v_iso = calculate_isoform_velocity(
        U,
        S,
        refined_params["beta"],
        refined_params["gamma"],
        isoform_gene_index,
    )
    v_gene = calculate_integrated_velocity(v_iso, isoform_gene_index, n_genes)
    iss = isoform_switching_score(
        S, v_iso, v_gene, isoform_gene_index, n_genes
    )
    add_velocity_to_adata(adata, v_iso, v_gene, iss, layer_prefix=layer_prefix)
    return {"v_iso": v_iso, "v_gene": v_gene, "iss": iss}
