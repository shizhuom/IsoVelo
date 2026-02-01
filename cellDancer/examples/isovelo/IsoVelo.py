import os
import sys
import glob
import pandas as pd
import math
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
from scipy.io import mmread
import celldancer as cd

sys.path.append('./src/celldancer/')

import sys
from src.celldancer.isovelo_estimation import isovelo_velocity
from src.celldancer.compute_isovelo_velocity import compute_isovelo_velocity

def get_splicing_count(file_path):
   with open(file_path, "rb") as f:
       obj = pickle.load(f)

   rows = list(obj["obs"])
   cols = list(obj["var"])

   mtx_path = Path(file_path).with_suffix(".mtx")
   X = mmread(mtx_path).tocsr()

   return pd.DataFrame.sparse.from_spmatrix(X, index=rows, columns=cols)

data_dir = "./examples/isovelo/data"
cell_type_u_s_path = os.path.join(data_dir, "brain_10x_pac_pac_scotch.csv")
meta_raw = pd.read_csv(cell_type_u_s_path)
meta_raw = meta_raw.rename(columns={
    'Unnamed: 0': 'cellID',
    'singleR.labels': 'clusters',
    'umap_1': 'embedding1',
    'umap_2': 'embedding2'
})

def _select_meta_for_sample(meta_df, sample_cellids):
    """Select metadata rows that best match a sample, based on suffix overlap."""
    meta_df = meta_df.copy()
    has_suffix = meta_df['cellID'].str.contains(r'_[0-9]+$').any()
    if not has_suffix:
        return meta_df.drop_duplicates(subset='cellID', keep='first'), None

    meta_df['suffix'] = meta_df['cellID'].str.extract(r'_([0-9]+)$')[0]
    suffixes = meta_df['suffix'].dropna().unique().tolist()
    best_suffix = None
    best_overlap = -1
    sample_set = set(sample_cellids)
    for sfx in suffixes:
        sub = meta_df[meta_df['suffix'] == sfx]
        ids = sub['cellID'].str.replace(r'_[0-9]+$', '', regex=True)
        overlap = len(sample_set & set(ids))
        if overlap > best_overlap:
            best_overlap = overlap
            best_suffix = sfx

    if best_suffix is None or best_overlap == 0:
        # Fallback: use all meta rows with suffix stripped
        meta_df['cellID'] = meta_df['cellID'].str.replace(r'_[0-9]+$', '', regex=True)
        return meta_df.drop_duplicates(subset='cellID', keep='first'), None

    meta_df = meta_df[meta_df['suffix'] == best_suffix].copy()
    meta_df['cellID'] = meta_df['cellID'].str.replace(r'_[0-9]+$', '', regex=True)
    meta_df = meta_df.drop_duplicates(subset='cellID', keep='first')
    return meta_df, best_suffix

def _load_sample(sample_id, s_pickle, u_pickle=None):
    s_df = get_splicing_count(s_pickle)
    if u_pickle is not None and os.path.exists(u_pickle):
        u_df = get_splicing_count(u_pickle)
    else:
        # Fallback: derive gene-level unsplice from spliced isoforms (proxy)
        split_cols = s_df.columns.to_series().str.split("_", n=1, expand=True)
        gene_names = split_cols[0].values
        u_df = s_df.groupby(gene_names, axis=1).sum()

    common_names = set(s_df.index) & set(u_df.index)
    common_names = sorted(list(common_names))
    if len(common_names) == 0:
        return None

    meta_df, suffix = _select_meta_for_sample(meta_raw, common_names)
    common_names = sorted(list(set(common_names) & set(meta_df['cellID'])))
    if len(common_names) == 0:
        return None

    meta_df = meta_df[meta_df['cellID'].isin(common_names)]
    s_df = s_df.loc[common_names].copy()
    u_df = u_df.loc[common_names].copy()

    s_long = s_df.reset_index().rename(columns={'index': 'cellID'})
    s_long = s_long.melt(
        id_vars='cellID',
        var_name='unique_id',
        value_name='splice'
    )

    split_cols = s_long['unique_id'].str.split('_', n=1, expand=True)
    s_long['gene_name'] = split_cols[0]
    s_long['isoform_name'] = split_cols[1]
    s_long.drop(columns=['unique_id'], inplace=True)

    u_long = u_df.reset_index().rename(columns={'index': 'cellID'})
    u_long = u_long.melt(
        id_vars='cellID',
        var_name='gene_name',
        value_name='unsplice'
    )

    common_genes = set(s_long['gene_name']) & set(u_long['gene_name'])
    s_long = s_long.loc[s_long['gene_name'].isin(common_genes)].copy()
    u_long = u_long.loc[u_long['gene_name'].isin(common_genes)].copy()

    merged_counts = pd.merge(s_long, u_long, on=['cellID', 'gene_name'], how='left')
    iso_df = pd.merge(
        merged_counts,
        meta_df[['cellID', 'clusters', 'embedding1', 'embedding2']],
        on='cellID',
        how='left'
    )

    # Make cellIDs unique across samples
    iso_df['cellID'] = sample_id + "_" + iso_df['cellID'].astype(str)
    iso_df['sample_id'] = sample_id

    iso_df = iso_df[[
        "gene_name", "isoform_name", "unsplice", "splice",
        "cellID", "clusters", "embedding1", "embedding2", "sample_id"
    ]]
    return iso_df

# Discover samples based on *_s.pickle / *_u.pickle pairs
s_pickles = sorted(glob.glob(os.path.join(data_dir, "*_s.pickle")))
sample_pairs = []
for s_path in s_pickles:
    base = os.path.basename(s_path).replace("_s.pickle", "")
    u_path = os.path.join(data_dir, base + "_u.pickle")
    if os.path.exists(u_path):
        sample_pairs.append((base, s_path, u_path))
    else:
        sample_pairs.append((base, s_path, None))

sample_dfs = {}
for sample_id, s_path, u_path in sample_pairs:
    df = _load_sample(sample_id, s_path, u_path)
    if df is not None:
        sample_dfs[sample_id] = df

if len(sample_dfs) == 0:
    raise RuntimeError("No valid sample pairs found in ./examples/isovelo/data")

# If only one sample exists, write it directly as the intersection file
if len(sample_dfs) == 1:
    only_id, only_df = next(iter(sample_dfs.items()))
    IsoVelo_u_s_intersection = only_df.copy()
    IsoVelo_u_s_intersection.to_parquet(
        os.path.join(data_dir, "all_samples_IsoVelo_u_s_intersection.parquet"),
        engine="pyarrow",
        index=False,
    )
else:
    # Build gene/isoform sets per sample
    combo_sets = {
        sid: set(zip(df["gene_name"], df["isoform_name"]))
        for sid, df in sample_dfs.items()
    }
    all_combos = set().union(*combo_sets.values())
    common_combos = set.intersection(*combo_sets.values())

    def _fill_missing_combos(df, missing_combos):
        if not missing_combos:
            return df
        cells = df[
            ["cellID", "clusters", "embedding1", "embedding2", "sample_id"]
        ].drop_duplicates()
        missing = pd.DataFrame(list(missing_combos), columns=["gene_name", "isoform_name"])
        cells["key"] = 1
        missing["key"] = 1
        expanded = cells.merge(missing, on="key", how="inner").drop(columns=["key"])
        expanded["unsplice"] = 0
        expanded["splice"] = 0
        expanded = expanded[
            [
                "gene_name",
                "isoform_name",
                "unsplice",
                "splice",
                "cellID",
                "clusters",
                "embedding1",
                "embedding2",
                "sample_id",
            ]
        ]
        return pd.concat([df, expanded], ignore_index=True)

    # Strategy 1: union of combos, fill missing with zeros
    filled_dfs = []
    for sid, df in sample_dfs.items():
        missing = all_combos - combo_sets[sid]
        filled_dfs.append(_fill_missing_combos(df, missing))
    IsoVelo_u_s_union = pd.concat(filled_dfs, ignore_index=True)

    IsoVelo_u_s_union.to_parquet(
        os.path.join(data_dir, "all_samples_IsoVelo_u_s_union_zeros.parquet"),
        engine="pyarrow",
        index=False,
    )

    # Strategy 2: intersection of combos
    filtered_dfs = []
    for sid, df in sample_dfs.items():
        mask = list(zip(df["gene_name"], df["isoform_name"]))
        keep = [c in common_combos for c in mask]
        filtered_dfs.append(df.loc[keep].copy())
    IsoVelo_u_s_intersection = pd.concat(filtered_dfs, ignore_index=True)

    IsoVelo_u_s_intersection.to_parquet(
        os.path.join(data_dir, "all_samples_IsoVelo_u_s_intersection.parquet"),
        engine="pyarrow",
        index=False,
    )

# Prioritize genes (optional)
file_path = "./examples/isovelo/data/Neurons_Progenitors_scotch_pacbio_novel_sep.csv"
if os.path.exists(file_path):
    data = pd.read_csv(file_path, sep=",", header=0)

    # Convert columns to numeric (non-numeric values become NaN, similar to as.numeric() behavior)
    data["p_gene_adj"] = pd.to_numeric(data["p_gene_adj"], errors="coerce")
    data["p_DTU_gene_adj"] = pd.to_numeric(data["p_DTU_gene_adj"], errors="coerce")

    data = data.dropna()

    data = data.sort_values(by=["p_gene_adj", "p_DTU_gene_adj"], ascending=[False, True])
    data.to_parquet('./examples/isovelo/data/isoform_switch_genes.parquet', engine='pyarrow')
