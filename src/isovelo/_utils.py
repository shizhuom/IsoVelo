import scanpy as sc
import scvelo as scv
import numpy as np
import pandas as pd
import scipy.sparse as sp
from tqdm import tqdm

def preprocess_and_initialize_scvelo(
    adata, 
    isoform_key="isoform_counts", 
    proportion_key = "proportion",
    min_isoform_counts=10, 
    min_cell_counts = 10,
    min_isoform_prop=0.05, 
    n_top_genes=800,
    n_top_splicing = 500,
    min_cells_spanning = 5,
    isoform_delimiter="_",
    normalized = False
):
    """
    Prefilter Isoforms and Genes.
    1. Filter cells, remove low total counts cells.
    2. Filter Isoforms: Remove low expression and low global proportion isoforms.
    3. Filter Genes, keep highly variable genes and isoform proportion variable genes.
    4. Run scVelo dynamical mode.
    5. Return initialization parameters for VAE.
    
    Parameters:
    adata: including layers['unspliced'] and obsm[isoform_key]
    isoform_key: key of isoform count in adata.obsm
    """

    if adata.X is None:
        adata.X = adata.layers['spliced'] + adata.layers['unspliced']

    # 1. Filter cells, remove low total counts cells.
    initial_cell_count = adata.n_obs
    sc.pp.filter_cells(adata, min_counts=min_cell_counts)
    print(f"Filtered cells from {initial_cell_count} to {adata.n_obs} (min_counts={min_cell_counts})")

    # 2. Filter Isoforms: Remove low expression and low global proportion isoforms.
    iso_df = adata.obsm[isoform_key]
    iso_df = iso_df.loc[adata.obs_names]
    isoform_names = iso_df.columns
    try:
        gene_map = pd.Series([x.rsplit(isoform_delimiter, 1)[0] for x in isoform_names], index=isoform_names)
    except Exception as e:
        raise ValueError(f"Error parsing isoform with names '{isoform_delimiter}'. Error: {e}")

    iso_sum = iso_df.sum(axis=0)
    keep_mask_count = iso_sum >= min_isoform_counts

    iso_prop = adata.obsm[proportion_key]
    high_prop_mask = iso_prop >= min_isoform_prop
    cells_passing_count = high_prop_mask.sum(axis=0)
    keep_mask_prop = cells_passing_count >= min_cells_spanning
    keep_isoforms = keep_mask_count & keep_mask_prop

    filtered_iso_df = iso_df.loc[:, keep_isoforms]
    adata.obsm[isoform_key] = filtered_iso_df
    filtered_prop_df = iso_prop.loc[:, keep_isoforms]
    adata.obsm[proportion_key] = filtered_prop_df

    print(f"Filtered isoforms from {iso_df.shape[1]} to {filtered_iso_df.shape[1]} based on isoform expression and global proportion.")

    remaining_isoforms = gene_map[keep_isoforms.values]
    new_counts = pd.Series(remaining_isoforms).value_counts(sort=False)
    adata.var['filtered_n_isoforms'] = 0
    genes_to_update = new_counts.index.intersection(adata.var_names)
    adata.var.loc[genes_to_update, 'filtered_n_isoforms'] = new_counts[genes_to_update]

    # Recalculate proportion dataframe based on remaining isoforms
    current_iso_cols = filtered_iso_df.columns
    current_gene_map = pd.Series([x.rsplit(isoform_delimiter, 1)[0] for x in current_iso_cols], index=current_iso_cols)
    new_gene_counts_df = filtered_iso_df.groupby(current_gene_map.values, axis=1).sum()
    full_spliced_df = pd.DataFrame(
        adata.layers['spliced'], 
        index=adata.obs_names, 
        columns=adata.var_names
    )
    full_spliced_df.update(new_gene_counts_df)
    adata.layers['spliced'] = full_spliced_df.values
    if not normalized:
        adata.X = adata.layers['spliced'] + adata.layers['unspliced']

    gene_ids_per_col = current_gene_map[filtered_iso_df.columns]
    gene_counts_expanded = new_gene_counts_df.loc[:, gene_ids_per_col]
    gene_counts_expanded.columns = filtered_iso_df.columns 
 
    new_props = filtered_iso_df / (gene_counts_expanded)
    adata.obsm[proportion_key] = new_props
      
    # 3. Filter Genes, keep highly variable genes and isoform proportion variable genes.
    adata_hvg = adata.copy()
    if not normalized:
        sc.pp.normalize_total(adata_hvg)
        sc.pp.log1p(adata_hvg)
        sc.pp.highly_variable_genes(adata_hvg, n_top_genes=n_top_genes, flavor='seurat')
        hvg_genes = set(adata_hvg.var_names[adata_hvg.var['highly_variable']])
    else:
        sc.pp.highly_variable_genes(adata_hvg, n_top_genes=n_top_genes, flavor='seurat')
        hvg_genes = set(adata_hvg.var_names[adata_hvg.var['highly_variable']])
    
    print(f"Found {len(hvg_genes)} expression HVGs.")

    iso_variances = new_props.var(axis=0)
    gene_splicing_scores = iso_variances.groupby(current_gene_map.values, sort=False).mean()
    multi_iso_genes = adata.var_names[adata.var['filtered_n_isoforms'] > 1]
    valid_genes = gene_splicing_scores.index.intersection(multi_iso_genes)
    final_scores = gene_splicing_scores.loc[valid_genes]

    if not final_scores.empty:
        top_splicing_genes = final_scores.sort_values(ascending=False).head(n_top_splicing).index.tolist()
        high_splice_genes = set(top_splicing_genes)
    else:
        high_splice_genes = set()
        
    print(f"Identified {len(high_splice_genes)} genes with high splicing variance.")
    
    final_genes_set = (hvg_genes | high_splice_genes)
    final_genes = [gene for gene in adata.var_names if gene in final_genes_set]
    adata = adata[:, final_genes].copy()

    is_isoform_kept = current_gene_map.isin(final_genes_set)
    final_iso_counts = filtered_iso_df.loc[:, is_isoform_kept]
    final_iso_props  = new_props.loc[:, is_isoform_kept]
    adata.obsm[isoform_key] = final_iso_counts
    adata.obsm[proportion_key] = final_iso_props
    
    # 4. Run scVelo dynamical mode.
    print("Preparing scVelo results:")
    if not normalized:
        scv.pp.filter_and_normalize(adata, min_counts=0, min_cells=0, log=True)
        scv.pp.moments(adata, n_pcs=30, n_neighbors=30)
    else:
        scv.pp.moments(adata, n_pcs=30, n_neighbors=30)
    
    scv.tl.recover_dynamics(adata, var_names='all', n_jobs=-1)

    n_total = adata.n_vars
    n_fitted = adata.var['fit_alpha'].notnull().sum()
    print(f"Dynamics recovered for {n_fitted}/{n_total} genes.")

    return adata