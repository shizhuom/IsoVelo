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
                           sep: str = "_") -> pd.DataFrame:
    """
    Integrate rare isoforms into one category.

    Parameters:
    x (pd.DataFrame): DataFrame with isoform proportions.
    threshold (float): Proportion threshold below which isoforms are considered rare.
    sep (str): Separator used in the column names to identify isoforms.

    Returns:
    pd.DataFrame: DataFrame with rare isoforms integrated.
    """
    cols = pd.Index(x.columns)
    genes = cols.to_series().astype(str).str.split(sep).str[0]
    iso_sum = x.sum(axis=0)
    gene_sum = iso_sum.groupby(genes).sum()
