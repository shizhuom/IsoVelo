"""
Compute cell velocity from isoform-level velocity estimation.

This module projects isoform-level RNA velocity onto the embedding space.
"""

import os
import sys
import random
import warnings
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

if __name__ == "__main__":
    sys.path.append('.')
    from sampling import downsampling_embedding
else:
    try:
        from .sampling import downsampling_embedding
    except ImportError:
        from sampling import downsampling_embedding


def compute_isovelo_velocity(
    isovelo_df,
    gene_list=None,
    speed_up=(60, 60),
    expression_scale=None,
    projection_neighbor_size=200,
    projection_neighbor_choice='embedding',
    aggregate_method='sum'
):
    """
    Project isoform-level RNA velocity onto the embedding space.
    
    For isoform-level velocity, we aggregate velocities across isoforms before
    computing cell-level velocity in embedding space.
    
    Arguments
    ---------
    isovelo_df: `pandas.DataFrame`
        Dataframe of isoform-level velocity estimation results.
        Columns=['cellIndex', 'gene_name', 'isoform_name', 'unsplice', 'splice',
                 'unsplice_predict', 'splice_predict', 'alpha', 'beta', 'gamma',
                 'loss', 'cellID', 'clusters', 'embedding1', 'embedding2']
                 
    gene_list: optional, `list` (default: None)
        Genes selected to calculate cell velocity. `None` for all genes.
        
    speed_up: optional, `tuple` (default: (60, 60))
        Sampling grid for downsampling cells.
        
    expression_scale: optional, `str` (default: None)
        Expression scaling method. `'power10'` for 10th power scaling.
        
    projection_neighbor_size: optional, `int` (default: 200)
        Number of neighbors for transition probability matrix.
        
    projection_neighbor_choice: optional, `str` (default: 'embedding')
        Method to obtain neighbors: 'embedding' or 'gene'.
        
    aggregate_method: optional, `str` (default: 'sum')
        How to aggregate isoform velocities: 'sum' or 'mean'.
        
    Returns
    -------
    isovelo_df: `pandas.DataFrame`
        Updated dataframe with additional columns ['velocity1', 'velocity2'].
    """
    
    def velocity_correlation(cell_matrix, velocity_matrix):
        """Calculate correlation between predicted velocity and cell displacement.
        
        Arguments
        ---------
        cell_matrix: np.ndarray (ngenes, ncells)
            Gene expression matrix
        velocity_matrix: np.ndarray (ngenes, ncells)
            Velocity matrix
            
        Returns
        -------
        c_matrix: np.ndarray (ncells, ncells)
        """
        c_matrix = np.zeros((cell_matrix.shape[1], velocity_matrix.shape[1]))
        for i in range(cell_matrix.shape[1]):
            c_matrix[i, :] = corr_coeff(cell_matrix, velocity_matrix, i)[0, :]
        np.fill_diagonal(c_matrix, 0)
        return c_matrix

    def velocity_projection(cell_matrix, velocity_matrix, embedding, knn_embedding):
        """Project velocity to embedding space.
        
        Arguments
        ---------
        cell_matrix: np.ndarray (ngenes, ncells)
        velocity_matrix: np.ndarray (ngenes, ncells)
        embedding: np.ndarray (ncells, 2)
        knn_embedding: sparse matrix
        """
        sigma_corr = 0.05
        cell_matrix[np.isnan(cell_matrix)] = 0
        velocity_matrix[np.isnan(velocity_matrix)] = 0
        corrcoef = velocity_correlation(cell_matrix, velocity_matrix)
        knn_mask = knn_embedding.A.astype(bool)
        scaled = np.where(knn_mask, corrcoef / sigma_corr, -np.inf)
        row_max = np.max(scaled, axis=1, keepdims=True)
        row_max[~np.isfinite(row_max)] = 0
        probability_matrix = np.exp(scaled - row_max)
        probability_matrix = np.where(np.isfinite(probability_matrix), probability_matrix, 0)
        row_sum = probability_matrix.sum(1, keepdims=True)
        probability_matrix = np.divide(
            probability_matrix, row_sum, out=np.zeros_like(probability_matrix), where=row_sum != 0
        )
        unitary_vectors = embedding.T[:, None, :] - embedding.T[:, :, None]
        with np.errstate(divide='ignore', invalid='ignore'):
            unitary_vectors /= np.linalg.norm(unitary_vectors, ord=2, axis=0)
            np.fill_diagonal(unitary_vectors[0, ...], 0)
            np.fill_diagonal(unitary_vectors[1, ...], 0)
        velocity_embedding = (probability_matrix * unitary_vectors).sum(2)
        velocity_embedding -= (knn_embedding.A * unitary_vectors).sum(2) / \
            knn_embedding.sum(1).A.T
        velocity_embedding = velocity_embedding.T
        return velocity_embedding

    # Remove invalid predictions
    is_NaN = isovelo_df[['alpha', 'beta']].isnull()
    row_has_NaN = is_NaN.any(axis=1)
    isovelo_df = isovelo_df[~row_has_NaN].reset_index(drop=True)
    
    if 'velocity1' in isovelo_df.columns:
        del isovelo_df['velocity1']
    if 'velocity2' in isovelo_df.columns:
        del isovelo_df['velocity2']
    
    if gene_list is None:
        gene_list = list(isovelo_df.gene_name.drop_duplicates())
    
    # Filter to selected genes
    isovelo_df_input = isovelo_df[isovelo_df.gene_name.isin(gene_list)].reset_index(drop=True)
    
    # Aggregate isoform data to gene level for velocity projection
    np_splice_all, np_dMatrix_all = data_reshape_isovelo(
        isovelo_df_input, aggregate_method=aggregate_method
    )
    
    n_genes, n_cells = np_splice_all.shape
    
    # Get one isoform per gene for embedding/downsampling
    first_gene = gene_list[0]
    first_isoform = isovelo_df_input[isovelo_df_input.gene_name == first_gene].isoform_name.iloc[0]
    
    # Create temporary data for downsampling (one row per cell)
    temp_df = isovelo_df_input[
        (isovelo_df_input.gene_name == first_gene) & 
        (isovelo_df_input.isoform_name == first_isoform)
    ].copy()
    temp_df['gene_name'] = first_gene  # Ensure consistent gene name
    
    # Prepare data for downsampling
    data_df = temp_df[['gene_name', 'unsplice', 'splice', 'cellID', 'embedding1', 'embedding2']].copy()
    
    embedding_downsampling, sampling_ixs, knn_embedding = downsampling_embedding(
        data_df,
        para='neighbors',
        target_amount=0,
        step=speed_up,
        n_neighbors=projection_neighbor_size,
        projection_neighbor_choice=projection_neighbor_choice,
        expression_scale=expression_scale,
        pca_n_components=None,
        umap_n=None,
        umap_n_components=None
    )
    
    # Get embedding coordinates
    embedding = temp_df[['embedding1', 'embedding2']].to_numpy()
    
    velocity_embedding = velocity_projection(
        np_splice_all[:, sampling_ixs],
        np_dMatrix_all[:, sampling_ixs],
        embedding[sampling_ixs, :],
        knn_embedding
    )
    
    if set(['velocity1', 'velocity2']).issubset(isovelo_df.columns):
        print("Caution! Overwriting the 'velocity' columns.")
        isovelo_df.drop(['velocity1', 'velocity2'], axis=1, inplace=True)
    
    # Map velocity back to all isoforms
    use_cell_id = 'cellID' in isovelo_df_input.columns
    if use_cell_id:
        sampled_cell_ids = temp_df.iloc[sampling_ixs]['cellID'].tolist()
        velocity_dict = {
            cell_id: (velocity_embedding[idx, 0], velocity_embedding[idx, 1])
            for idx, cell_id in enumerate(sampled_cell_ids)
        }
    else:
        velocity_dict = {
            cell_idx: (velocity_embedding[idx, 0], velocity_embedding[idx, 1])
            for idx, cell_idx in enumerate(sampling_ixs)
        }
    
    # Apply velocity to all rows in isovelo_df_input
    isovelo_df_input['velocity1'] = np.nan
    isovelo_df_input['velocity2'] = np.nan
    
    for cell_key, (v1, v2) in velocity_dict.items():
        if use_cell_id:
            mask = isovelo_df_input.cellID == cell_key
        else:
            mask = isovelo_df_input.cellIndex == cell_key
        isovelo_df_input.loc[mask, 'velocity1'] = v1
        isovelo_df_input.loc[mask, 'velocity2'] = v2
    
    return isovelo_df_input


def corr_coeff(ematrix, vmatrix, i):
    """Calculate correlation between predicted velocity and displacement.
    
    Arguments
    ---------
    ematrix: expression matrix
    vmatrix: velocity matrix
    i: cell index
    """
    ematrix = ematrix.T
    vmatrix = vmatrix.T
    ematrix = ematrix - ematrix[i, :]
    vmatrix = vmatrix[i, :][None, :]
    ematrix_m = ematrix - ematrix.mean(1)[:, None]
    vmatrix_m = vmatrix - vmatrix.mean(1)[:, None]
    
    # Sum of squares across rows
    ematrix_ss = (ematrix_m**2).sum(1)
    vmatrix_ss = (vmatrix_m**2).sum(1)
    cor = np.dot(ematrix_m, vmatrix_m.T)
    N = np.sqrt(np.dot(ematrix_ss[:, None], vmatrix_ss[None]))
    cor = np.divide(cor, N, out=np.zeros_like(cor), where=N != 0)
    return cor.T


def data_reshape_isovelo(isovelo_df, aggregate_method='sum'):
    """
    Reshape isoform-level data for velocity projection.
    
    Aggregates splice velocities across isoforms for each gene.
    
    Arguments
    ---------
    isovelo_df: DataFrame with isoform-level predictions
    aggregate_method: 'sum' or 'mean' for aggregating isoform velocities
    
    Returns
    -------
    np_splice_reshape: np.ndarray (ngenes, ncells)
    np_dMatrix: np.ndarray (ngenes, ncells)
    """
    psc = 1
    
    gene_names = isovelo_df['gene_name'].drop_duplicates().tolist()
    
    # Get number of cells from first gene-isoform
    first_gene = gene_names[0]
    first_gene_data = isovelo_df[isovelo_df.gene_name == first_gene]
    first_isoform = first_gene_data.isoform_name.iloc[0]
    n_cells = first_gene_data[first_gene_data.isoform_name == first_isoform].shape[0]
    
    n_genes = len(gene_names)
    
    # Initialize arrays
    splice_matrix = np.zeros((n_genes, n_cells))
    dmatrix = np.zeros((n_genes, n_cells))
    
    for g_idx, gene in enumerate(gene_names):
        gene_data = isovelo_df[isovelo_df.gene_name == gene]
        isoforms = gene_data.isoform_name.drop_duplicates().tolist()
        
        # Aggregate splice across isoforms
        splice_sum = np.zeros(n_cells)
        dsplice_sum = np.zeros(n_cells)
        
        for isoform in isoforms:
            iso_data = gene_data[gene_data.isoform_name == isoform].sort_values('cellIndex')
            splice_sum += iso_data.splice.values
            dsplice_sum += (iso_data.splice_predict.values - iso_data.splice.values)
        
        if aggregate_method == 'mean':
            splice_sum /= len(isoforms)
            dsplice_sum /= len(isoforms)
        
        splice_matrix[g_idx, :] = splice_sum
        dmatrix[g_idx, :] = dsplice_sum
    
    # Apply transformation for velocity projection
    np_dMatrix2 = np.sqrt(np.abs(dmatrix) + psc) * np.sign(dmatrix)
    
    return splice_matrix, np_dMatrix2


def compute_isovelo_velocity_per_isoform(
    isovelo_df,
    gene=None,
    isoform=None,
    speed_up=(60, 60),
    projection_neighbor_size=200,
    smooth_k=30,
    smooth_sigma=None
):
    """
    Compute velocity projection for a specific gene-isoform pair.
    
    This is useful for visualizing velocity at the isoform level.
    When `gene` is None and `isoform` is None, compute isoform velocities
    for all genes and isoforms in `isovelo_df`.
    When `gene` is a list and `isoform` is None, compute isoform velocities
    for all isoforms in the specified genes.
    
    Arguments
    ---------
    isovelo_df: DataFrame with isoform-level predictions
    gene: Gene name
    isoform: Isoform name
    speed_up: Downsampling grid
    projection_neighbor_size: Number of neighbors
    smooth_k: Number of neighbors for smoothing in embedding space (0/None to disable)
    smooth_sigma: Optional Gaussian kernel sigma for weighted smoothing
    
    Returns
    -------
    Updated DataFrame with velocity columns for the selected isoforms
    """

    def _is_list_like(value):
        return isinstance(value, (list, tuple, set, np.ndarray, pd.Series))

    def _compute_one(gene_name, isoform_name):
        # Filter to specific gene-isoform
        iso_df = isovelo_df[
            (isovelo_df.gene_name == gene_name) &
            (isovelo_df.isoform_name == isoform_name)
        ].copy()

        if len(iso_df) == 0:
            raise ValueError(f"No data found for gene={gene_name}, isoform={isoform_name}")

        # Create expression and velocity matrices (1 x n_cells)
        splice = iso_df.splice.values.reshape(1, -1)
        dsplice = (iso_df.splice_predict.values - iso_df.splice.values).reshape(1, -1)

        # Prepare for downsampling
        data_df = iso_df[['gene_name', 'unsplice', 'splice', 'cellID', 'embedding1', 'embedding2']].copy()

        embedding_downsampling, sampling_ixs, knn_embedding = downsampling_embedding(
            data_df,
            para='neighbors',
            target_amount=0,
            step=speed_up,
            n_neighbors=projection_neighbor_size,
            projection_neighbor_choice='embedding'
        )

        embedding = iso_df[['embedding1', 'embedding2']].to_numpy()

        # Simple velocity projection for single gene/isoform
        psc = 1
        np_dMatrix = np.sqrt(np.abs(dsplice) + psc) * np.sign(dsplice)

        # Velocity correlation and projection
        def velocity_projection_simple(cell_matrix, velocity_matrix, embedding, knn_embedding):
            sigma_corr = 0.05
            cell_matrix[np.isnan(cell_matrix)] = 0
            velocity_matrix[np.isnan(velocity_matrix)] = 0

            n_cells = cell_matrix.shape[1]
            c_matrix = np.zeros((n_cells, n_cells))

            for i in range(n_cells):
                c_matrix[i, :] = corr_coeff(cell_matrix, velocity_matrix, i)[0, :]
            np.fill_diagonal(c_matrix, 0)

            knn_mask = knn_embedding.A.astype(bool)
            scaled = np.where(knn_mask, c_matrix / sigma_corr, -np.inf)
            row_max = np.max(scaled, axis=1, keepdims=True)
            row_max[~np.isfinite(row_max)] = 0
            probability_matrix = np.exp(scaled - row_max)
            probability_matrix = np.where(np.isfinite(probability_matrix), probability_matrix, 0)
            row_sum = probability_matrix.sum(1, keepdims=True)
            probability_matrix = np.divide(
                probability_matrix, row_sum, out=np.zeros_like(probability_matrix), where=row_sum != 0
            )

            unitary_vectors = embedding.T[:, None, :] - embedding.T[:, :, None]
            with np.errstate(divide='ignore', invalid='ignore'):
                unitary_vectors /= np.linalg.norm(unitary_vectors, ord=2, axis=0)
                np.fill_diagonal(unitary_vectors[0, ...], 0)
                np.fill_diagonal(unitary_vectors[1, ...], 0)

            velocity_embedding = (probability_matrix * unitary_vectors).sum(2)
            velocity_embedding -= (knn_embedding.A * unitary_vectors).sum(2) / \
                knn_embedding.sum(1).A.T
            velocity_embedding = velocity_embedding.T
            return velocity_embedding

        velocity_embedding = velocity_projection_simple(
            splice[:, sampling_ixs],
            np_dMatrix[:, sampling_ixs],
            embedding[sampling_ixs, :],
            knn_embedding
        )

        # Add velocity to dataframe
        iso_df['velocity1'] = np.nan
        iso_df['velocity2'] = np.nan

        use_cell_id = 'cellID' in iso_df.columns
        if use_cell_id:
            sampled_cell_ids = iso_df.iloc[sampling_ixs]['cellID'].tolist()
            for idx, cell_id in enumerate(sampled_cell_ids):
                iso_df.loc[iso_df.cellID == cell_id, 'velocity1'] = velocity_embedding[idx, 0]
                iso_df.loc[iso_df.cellID == cell_id, 'velocity2'] = velocity_embedding[idx, 1]
        else:
            for idx, cell_idx in enumerate(sampling_ixs):
                iso_df.loc[iso_df.cellIndex == cell_idx, 'velocity1'] = velocity_embedding[idx, 0]
                iso_df.loc[iso_df.cellIndex == cell_idx, 'velocity2'] = velocity_embedding[idx, 1]

        # Optional smoothing in embedding space
        if smooth_k is not None and smooth_k > 0:
            emb = iso_df[['embedding1', 'embedding2']].to_numpy()
            v = iso_df[['velocity1', 'velocity2']].to_numpy()
            finite = np.isfinite(v).all(axis=1)
            if finite.any() and len(emb) > 1:
                k = min(int(smooth_k), len(emb) - 1)
                if k >= 1:
                    nn = NearestNeighbors(n_neighbors=k + 1)
                    nn.fit(emb)
                    dists, idxs = nn.kneighbors(emb)
                    v_smoothed = np.full_like(v, np.nan)
                    for i in range(len(emb)):
                        nbrs = idxs[i, 1:]  # exclude self
                        vals = v[nbrs]
                        mask = np.isfinite(vals).all(axis=1)
                        if not mask.any():
                            continue
                        if smooth_sigma is None:
                            v_smoothed[i] = vals[mask].mean(axis=0)
                        else:
                            w = np.exp(-(dists[i, 1:] ** 2) / (2 * smooth_sigma ** 2))
                            w = w[mask]
                            v_smoothed[i] = (vals[mask] * w[:, None]).sum(axis=0) / w.sum()
                    use_smoothed = np.isfinite(v_smoothed).all(axis=1)
                    if use_smoothed.any():
                        v_out = v.copy()
                        v_out[use_smoothed] = v_smoothed[use_smoothed]
                        iso_df['velocity1'] = v_out[:, 0]
                        iso_df['velocity2'] = v_out[:, 1]

        return iso_df

    # Resolve target gene/isoform pairs
    if gene is None and isoform is None:
        gene_list = isovelo_df.gene_name.drop_duplicates().tolist()
        pairs = []
        for gene_name in gene_list:
            isoforms = isovelo_df[isovelo_df.gene_name == gene_name].isoform_name.drop_duplicates().tolist()
            if not isoforms:
                raise ValueError(f"No isoforms found for gene={gene_name}")
            for isoform_name in isoforms:
                pairs.append((gene_name, isoform_name))
    else:
        if gene is None:
            raise ValueError("When isoform is specified, gene must also be specified")

        if _is_list_like(gene) and not isinstance(gene, str):
            gene_list = list(gene)
        else:
            gene_list = [gene]

        if isoform is None:
            pairs = []
            missing_genes = []
            for gene_name in gene_list:
                if gene_name not in isovelo_df.gene_name.values:
                    missing_genes.append(gene_name)
                    continue
                isoforms = isovelo_df[isovelo_df.gene_name == gene_name].isoform_name.drop_duplicates().tolist()
                if not isoforms:
                    raise ValueError(f"No isoforms found for gene={gene_name}")
                for isoform_name in isoforms:
                    pairs.append((gene_name, isoform_name))
            if missing_genes:
                warnings.warn(
                    "Genes not found in isovelo_df and skipped: "
                    + ", ".join(map(str, missing_genes)),
                    UserWarning
                )
        else:
            if _is_list_like(isoform) and not isinstance(isoform, str):
                isoform_list = list(isoform)
            else:
                isoform_list = [isoform]
            pairs = []
            missing_genes = []
            for gene_name in gene_list:
                if gene_name not in isovelo_df.gene_name.values:
                    missing_genes.append(gene_name)
                    continue
                gene_isoforms = isovelo_df[isovelo_df.gene_name == gene_name].isoform_name.drop_duplicates().tolist()
                selected_isoforms = [i for i in isoform_list if i in gene_isoforms]
                if not selected_isoforms:
                    raise ValueError(f"No matching isoforms found for gene={gene_name}")
                for isoform_name in selected_isoforms:
                    pairs.append((gene_name, isoform_name))
            if missing_genes:
                warnings.warn(
                    "Genes not found in isovelo_df and skipped: "
                    + ", ".join(map(str, missing_genes)),
                    UserWarning
                )

    results = [_compute_one(gene_name, isoform_name) for gene_name, isoform_name in pairs]
    return pd.concat(results, ignore_index=True)
