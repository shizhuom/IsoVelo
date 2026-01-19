import sys
sys.path.insert(0, '.')
from src.celldancer.isovelo_estimation import isovelo_velocity
import pandas as pd
from src.celldancer.compute_isovelo_velocity import compute_isovelo_velocity

IsoVelo_u_s = pd.read_parquet('./examples/isovelo/data/137d_run2_IsoVelo_u_s.parquet')
gene_counts = IsoVelo_u_s['gene_name'].value_counts()
top_5_genes = gene_counts.head(5).index.tolist()
IsoVelo_u_s = IsoVelo_u_s[IsoVelo_u_s['gene_name'].isin(top_5_genes)]

loss_df, isovelo_df = isovelo_velocity(
    isovelo_u_s=IsoVelo_u_s,
    gene_list=None,
    max_epoches=10,
    n_neighbors=30,
    loss_func='cosine'
)

print(isovelo_df['alpha'].notna().sum(), "non-NaN alpha values")

with pd.option_context('display.max_columns', None):
    print(isovelo_df.head())

isovelo_df = compute_isovelo_velocity(
    isovelo_df,
    speed_up=(60, 60),
    projection_neighbor_size=200
)