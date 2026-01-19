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

cell_type_u_s_path='./examples/isovelo/data/brain_10x_pac_pac_scotch.csv'
IsoVelo_data=pd.read_csv(cell_type_u_s_path)
IsoVelo_data.head()

s137d_r2_s = get_splicing_count('./examples/isovelo/data/137d_run2_s.pickle')
s137d_r2_s.head()

s137d_r2_u = get_splicing_count('./examples/isovelo/data/137d_run2_u.pickle')
s137d_r2_u.head()

common_names = set(s137d_r2_s.index) & set(s137d_r2_u.index)

meta_df = IsoVelo_data.rename(columns={
    'Unnamed: 0': 'cellID',
    'singleR.labels': 'clusters',
    'umap_1': 'embedding1',
    'umap_2': 'embedding2'
})

meta_df = meta_df[meta_df['cellID'].str.contains(r'_4$')]
meta_df['cellID'] = meta_df['cellID'].str.replace(r'_[0-9]+$', '', regex=True)
meta_df = meta_df.drop_duplicates(subset='cellID', keep='first')

common_names = common_names & set(meta_df['cellID'])
common_names = sorted(list(common_names))

meta_df = meta_df[meta_df['cellID'].isin(common_names)]
s137d_r2_s = s137d_r2_s.loc[common_names].copy()
s137d_r2_u = s137d_r2_u.loc[common_names].copy()

s_long = s137d_r2_s.reset_index().rename(columns={'index': 'cellID'})
s_long = s_long.melt(
    id_vars='cellID', 
    var_name='unique_id', 
    value_name='splice'
)

split_cols = s_long['unique_id'].str.split('_', n=1, expand=True)
s_long['gene_name'] = split_cols[0]
s_long['isoform_name'] = split_cols[1]
s_long.drop(columns=['unique_id'], inplace=True)

u_long = s137d_r2_u.reset_index().rename(columns={'index': 'cellID'})
u_long = u_long.melt(
    id_vars='cellID', 
    var_name='gene_name', 
    value_name='unsplice'
)

common_genes = set(s_long['gene_name']) & set(u_long['gene_name'])
s_long = s_long.loc[s_long['gene_name'].isin(common_genes)].copy()
u_long = u_long.loc[u_long['gene_name'].isin(common_genes)].copy()

merged_counts = pd.merge(s_long, u_long, on=['cellID', 'gene_name'], how='left')
merged_counts["unsplice"].isna().sum()

IsoVelo_u_s = pd.merge(merged_counts, meta_df[['cellID', 'clusters', 'embedding1', 'embedding2']], on='cellID', how='left')

IsoVelo_u_s = IsoVelo_u_s[[
    "gene_name", "isoform_name", "unsplice", "splice", 
    "cellID", "clusters", "embedding1", "embedding2"
]]

IsoVelo_u_s.to_parquet('./examples/isovelo/data/137d_run2_IsoVelo_u_s.parquet', engine='pyarrow')
## IsoVelo_u_s = pd.read_parquet('./examples/isovelo/data/137d_run2_IsoVelo_u_s.parquet')

gene_counts = IsoVelo_u_s['gene_name'].value_counts()
top_5_genes = gene_counts.head(5).index.tolist()

isoform_counts = IsoVelo_u_s.groupby('gene_name')['isoform_name'].nunique()
isoform_counts = isoform_counts.sort_values(ascending=False)

isoform_counts

IsoVelo_u_s = IsoVelo_u_s[IsoVelo_u_s['gene_name'].isin(top_5_genes)]

loss_df, isovelo_df = isovelo_velocity(
    isovelo_u_s=IsoVelo_u_s,
    gene_list=None,  # All genes
    max_epoches=200,
    n_neighbors=30,
    loss_func='cosine'
)

isovelo_df = compute_isovelo_velocity(
    isovelo_df,
    speed_up=(60, 60),
    projection_neighbor_size=200
)