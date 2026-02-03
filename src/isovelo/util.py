import pandas as pd
import numpy as np
from scipy.io import mmread
import pickle

def get_splicing_count(file_path):
   with open(file_path, "rb") as f:
       obj = pickle.load(f)

   rows = list(obj["obs"])
   cols = list(obj["var"])

   mtx_path = Path(file_path).with_suffix(".mtx")
   X = mmread(mtx_path).tocsr()

   return pd.DataFrame.sparse.from_spmatrix(X, index=rows, columns=cols)

def integrate_rare_isoform(x: pd.DataFrame, 
                           threshold: float = 0.1, 
                           sep: str = "_",
                           filter_gene_count: int = 10,
                           return_summary: bool = True) -> pd.DataFrame:
    """
    Integrate rare isoforms into one category.

    Parameters:
    x (pd.DataFrame): DataFrame with isoform proportions.
    threshold (float): Proportion threshold below which isoforms are considered rare.
    sep (str): Separator used in the column names to identify isoforms.
    filter_gene_count (int): Minimum number of counts a gene must have to be considered for filtering.
    return_summary (bool): Whether to return a summary of the isoform proportions.
    Returns:
    pd.DataFrame: DataFrame with rare isoforms integrated.
    """
    cols = pd.Index(x.columns)
    genes = cols.to_series().astype(str).str.split(sep).str[0]
    iso_sum = x.sum(axis=0)
    gene_sum = iso_sum.groupby(genes).sum()

    gene_total_for_iso = genes.map(gene_sum)

    with np.errstate(divide='ignore', invalid='ignore'):
        iso_proportion = iso_sum / gene_total_for_iso

    to_merge = iso_proportion < threshold

    low_cols = cols[to_merge.values]

    if len(low_cols) == 0:
        summary = None
        if return_summary:
            summary = pd.DataFrame({
                "isoform": cols,
                "genes": genes.values,
                "pseudobulk": iso_sum.values,
                "gene_total": gene_total_for_iso.values,
                "prop_within_gene": iso_proportion.values,
                "merged_into_other": to_merge.values,
            }).set_index("isoform")
        return x.copy(), summary
    
    low_df = x.loc[:, low_cols]

    low_genes = genes.loc[low_cols]
    grouped_T = low_df.T.groupby(low_genes.values).sum()
    other_counts = grouped_T.T
    other_cols = [f"{g}{sep}other" for g in other_counts.columns]
    other_counts.columns = other_cols

    res = x.drop(columns=low_cols).join(other_counts)

    summary = None
    if return_summary:
        summary = pd.DataFrame({
            "isoform": cols,
            "genes": genes.values,
            "pseudobulk": iso_sum.values,
            "gene_total": gene_total_for_iso.values,
            "prop_within_gene": iso_proportion.values,
            "merged_into_other": to_merge.values,
        }).set_index("isoform").sort_values(by=["genes", "prop_within_gene"])

    return res, summary




    
