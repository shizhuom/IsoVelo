import sys
sys.path.insert(0, '.')
from src.celldancer.isovelo_estimation import isovelo_velocity
import pandas as pd
from src.celldancer.compute_isovelo_velocity import compute_isovelo_velocity
import numpy as np
import matplotlib.pyplot as plt

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

with pd.option_context('display.max_columns', None):
    print(isovelo_df.head())

if 'cellID' in isovelo_df.columns and 'cellIndex' in isovelo_df.columns:
    isovelo_df = isovelo_df.drop(columns=['cellIndex'])

gene_to_plot = top_5_genes[0]
plot_df = isovelo_df[isovelo_df.gene_name == gene_to_plot].copy()

clusters = pd.Categorical(plot_df['clusters'])
colors = plt.cm.tab20(np.linspace(0, 1, max(len(clusters.categories), 1)))
color_map = dict(zip(clusters.categories, colors))
plot_df['cluster_color'] = plot_df['clusters'].map(color_map)

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(
    plot_df['embedding1'],
    plot_df['embedding2'],
    c=plot_df['cluster_color'],
    s=6,
    alpha=0.6,
    linewidths=0
)

vel_mask = plot_df['velocity1'].notna() & plot_df['velocity2'].notna()
vx = plot_df.loc[vel_mask, 'velocity1'].to_numpy()
vy = plot_df.loc[vel_mask, 'velocity2'].to_numpy()
xs = plot_df.loc[vel_mask, 'embedding1'].to_numpy()
ys = plot_df.loc[vel_mask, 'embedding2'].to_numpy()
vel_mag = np.sqrt(vx**2 + vy**2)
max_mag = np.nanmax(vel_mag) if vel_mag.size else 0.0
if max_mag > 0:
    axis_span = max(plot_df['embedding1'].max() - plot_df['embedding1'].min(),
                    plot_df['embedding2'].max() - plot_df['embedding2'].min())
    scale = 0.05 * axis_span / max_mag
    ax.quiver(
        xs, ys, vx * scale, vy * scale,
        angles='xy', scale_units='xy', scale=1,
        width=0.002, color='black', alpha=0.6
    )

for cluster, color in color_map.items():
    ax.scatter([], [], c=[color], label=str(cluster), s=20)

ax.set_title(f"{gene_to_plot} (all isoforms)")
ax.set_xlabel("embedding1")
ax.set_ylabel("embedding2")
ax.legend(loc='best', fontsize=8, frameon=False)
plt.tight_layout()
plt.show()
