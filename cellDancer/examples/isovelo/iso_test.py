"""End-to-end test for iso_vae, iso_refine, iso_velocity, iso_plot."""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import torch

from src.celldancer.iso_vae import IsoVeloDataset, train_isovelo_vae
from src.celldancer.iso_refine import refine_step
from src.celldancer.iso_velocity import compute_isoform_velocity_and_iss
from src.celldancer.iso_plot import (
    plot_global_velocity,
    plot_isoform_portrait,
    plot_switching_hotspots,
    plot_isoform_proportion_stream,
)


def _load_data() -> pd.DataFrame:
    df = pd.read_parquet(
        "./examples/isovelo/data/all_samples_IsoVelo_u_s_intersection.parquet"
    )
    return df


def main() -> None:
    df = _load_data()

    # Optional: focus on DTU genes for a faster demo
    genes_path = "./examples/isovelo/data/isoform_switch_genes.parquet"
    if os.path.exists(genes_path):
        genes_df = pd.read_parquet(genes_path)
        gene_list = (
            genes_df[(genes_df["p_gene_adj"] > 0.99) & (genes_df["p_DTU_gene_adj"] < 0.05)][
                "genes"
            ]
            .unique()
            .tolist()
        )
        if gene_list:
            df = df[df["gene_name"].isin(gene_list)].copy()
    else:
        gene_list = df["gene_name"].unique().tolist()

    # Train VAE
    model, history, dataset = train_isovelo_vae(
        df,
        batch_size=256,
        epochs=5,
        lr=1e-3,
        use_zinb=True,
        kl_weight=1.0,
        physics_weight=1.0,
    )

    # Inference to get latent z and tau
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        U = dataset.U.to(device)
        S = dataset.S.to(device)
        (
            alpha,
            beta,
            gamma,
            tau,
            _u0,
            _s0,
            _disp,
            _zi,
            _u_hat,
            _s_hat,
            mu,
            _logvar,
        ) = model(U, S)

    latent_z = mu.detach().cpu().numpy()
    tau = tau.detach().cpu().numpy().reshape(-1)

    initial_params = {
        "alpha": alpha.detach().cpu().numpy(),
        "beta": beta.detach().cpu().numpy(),
        "gamma": gamma.detach().cpu().numpy(),
    }

    # Refinement
    refined = refine_step(
        adata={"U": dataset.mats.U, "S": dataset.mats.S},
        latent_z=latent_z,
        initial_params=initial_params,
        tau=tau,
        isoform_gene_index=dataset.mats.isoform_gene_index,
        k=30,
        gd_steps=20,
        lr=1e-2,
        alignment_weight=1.0,
        anchor_weight=0.1,
        batch_size=512,
    )

    # Velocity and ISS
    results = compute_isoform_velocity_and_iss(
        adata={"U": dataset.mats.U, "S": dataset.mats.S},
        refined_params=refined,
        isoform_gene_index=dataset.mats.isoform_gene_index,
        n_genes=len(dataset.mats.gene_names),
        layer_prefix="isoVelo",
    )

    # Plot examples
    os.makedirs("./examples/isovelo/figures", exist_ok=True)

    plot_global_velocity(
        adata={"layers": {}},
        latent_z=latent_z,
        v_gene=results["v_gene"],
        basis="X_isoVAE",
        density=1.0,
        title="Global IsoVelo (gene-level)",
    )

    gene_to_plot = gene_list[0] if gene_list else dataset.mats.gene_names[0]
    gene_index = dataset.mats.gene_names.index(gene_to_plot)

    plot_switching_hotspots(
        latent_z=latent_z,
        iss=results["iss"],
        gene_index=gene_index,
        title=f"ISS Hotspots: {gene_to_plot}",
    )

    isoform_indices = dataset.mats.gene_to_isoform_indices[gene_index]
    isoform_names = [dataset.mats.isoform_names[i] for i in isoform_indices]
    plot_isoform_proportion_stream(
        S=dataset.mats.S,
        tau=tau,
        isoform_indices=isoform_indices,
        isoform_names=isoform_names,
    )

    # Phase portrait for first isoform of the gene
    if isoform_names:
        plot_isoform_portrait(
            df=df,
            gene_name=gene_to_plot,
            isoform_name=isoform_names[0],
            alpha_beta_gamma=refined,
        )


if __name__ == "__main__":
    main()
