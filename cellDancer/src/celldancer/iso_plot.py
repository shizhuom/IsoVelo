"""Visualization utilities for isoform-aware velocity."""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def _get_from_adata(adata, key: str):
    if isinstance(adata, dict):
        return adata.get(key)
    return getattr(adata, key, None)


def plot_global_velocity(
    adata,
    latent_z: np.ndarray,
    v_gene: np.ndarray,
    basis: str = "X_isoVAE",
    density: float = 1.0,
    title: Optional[str] = None,
):
    """Project gene-level velocities onto the VAE latent embedding.

    If scVelo is available and adata is AnnData, uses velocity_embedding_stream.
    Otherwise falls back to a quiver plot.
    """
    try:
        import scvelo as scv  # type: ignore

        if not isinstance(adata, dict):
            if basis not in adata.obsm:
                adata.obsm[basis] = np.asarray(latent_z)
            adata.obsm[f"{basis}_velocity"] = np.asarray(v_gene)
            scv.pl.velocity_embedding_stream(
                adata, basis=basis, vkey=f"{basis}_velocity", density=density
            )
            return
    except Exception:
        pass

    z = np.asarray(latent_z)
    v = np.asarray(v_gene)
    if z.shape[1] < 2:
        raise ValueError("latent_z must be at least 2D for plotting.")

    plt.figure(figsize=(6, 5))
    plt.scatter(z[:, 0], z[:, 1], s=8, c="lightgray", alpha=0.6)
    plt.quiver(z[:, 0], z[:, 1], v[:, 0], v[:, 1], angles="xy", scale_units="xy", scale=1)
    if title:
        plt.title(title)
    plt.xlabel("z1")
    plt.ylabel("z2")
    plt.tight_layout()


def plot_isoform_portrait(
    df: pd.DataFrame,
    gene_name: str,
    isoform_name: str,
    alpha_beta_gamma: Optional[Dict[str, np.ndarray]] = None,
):
    """Phase portrait: gene-level unspliced vs isoform spliced.

    Uses clusters column for color. Draws steady-state line with slope gamma/beta.
    """
    required = {"gene_name", "isoform_name", "unsplice", "splice", "clusters"}
    if not required.issubset(df.columns):
        missing = required.difference(df.columns)
        raise ValueError(f"Missing columns in df: {sorted(missing)}")

    sub = df[(df["gene_name"] == gene_name) & (df["isoform_name"] == isoform_name)]
    if sub.empty:
        raise ValueError("No matching gene/isoform rows found.")

    plt.figure(figsize=(6, 5))
    sns.scatterplot(
        data=sub,
        x="splice",
        y="unsplice",
        hue="clusters",
        palette="tab10",
        s=20,
        alpha=0.8,
    )

    if alpha_beta_gamma is not None:
        beta = alpha_beta_gamma.get("beta")
        gamma = alpha_beta_gamma.get("gamma")
        if beta is not None and gamma is not None:
            b = np.asarray(beta).mean()
            g = np.asarray(gamma).mean()
            slope = g / (b + 1e-8)
            xlim = plt.gca().get_xlim()
            xs = np.linspace(xlim[0], xlim[1], 100)
            ys = slope * xs
            plt.plot(xs, ys, color="black", linestyle="--", label="steady-state")
            plt.legend(loc="best")

    plt.title(f"{gene_name} | {isoform_name}")
    plt.tight_layout()


def plot_switching_hotspots(
    latent_z: np.ndarray,
    iss: np.ndarray,
    gene_index: int,
    title: Optional[str] = None,
):
    """UMAP colored by ISS for a specific gene index."""
    z = np.asarray(latent_z)
    if z.shape[1] < 2:
        raise ValueError("latent_z must be at least 2D for plotting.")
    values = np.asarray(iss)[:, gene_index]

    plt.figure(figsize=(6, 5))
    plt.scatter(z[:, 0], z[:, 1], c=values, cmap="viridis", s=12)
    plt.colorbar(label="ISS")
    if title:
        plt.title(title)
    plt.xlabel("z1")
    plt.ylabel("z2")
    plt.tight_layout()


def plot_isoform_proportion_stream(
    S: np.ndarray,
    tau: np.ndarray,
    isoform_indices: Sequence[int],
    isoform_names: Optional[Sequence[str]] = None,
):
    """Stacked area chart of isoform proportions over latent time."""
    S = np.asarray(S, dtype=np.float32)
    tau = np.asarray(tau, dtype=np.float32)
    idx = np.asarray(isoform_indices, dtype=np.int64)

    order = np.argsort(tau)
    S_sel = S[order][:, idx]
    tau_sorted = tau[order]

    denom = S_sel.sum(axis=1, keepdims=True)
    p = S_sel / (denom + 1e-8)

    if isoform_names is None:
        isoform_names = [f"isoform_{i}" for i in idx]

    plt.figure(figsize=(7, 4))
    plt.stackplot(tau_sorted, p.T, labels=isoform_names, alpha=0.8)
    plt.xlabel("latent time (tau)")
    plt.ylabel("isoform proportion")
    plt.legend(loc="upper right", fontsize=8, ncol=2)
    plt.tight_layout()
