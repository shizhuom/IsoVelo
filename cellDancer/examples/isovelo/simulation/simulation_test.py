import sys
sys.path.insert(0, '.')
from src.celldancer.isovelo_estimation import isovelo_velocity
import pandas as pd
from src.celldancer.compute_isovelo_velocity import compute_isovelo_velocity, compute_isovelo_velocity_per_isoform
import numpy as np
import matplotlib.pyplot as plt

IsoVelo_u_s = pd.read_parquet('./examples/isovelo/simulation/sim_counts.parquet')

loss_df, isovelo_df = isovelo_velocity(
    isovelo_u_s=IsoVelo_u_s,
    speed_up=False,
    permutation_ratio=1,
    n_neighbors=30,
    max_epoches=20000,
    loss_func='cosine'
)

# IsoVelo_u_s[IsoVelo_u_s["gene_name"].isin(genes[0:2])].to_csv('./examples/isovelo/data/137d_run2_IsoVelo_u_s_processed.csv')

print(isovelo_df['alpha'].notna().sum(), "non-NaN alpha values")

with pd.option_context('display.max_columns', None):
    print(isovelo_df.head())

isovelo_df = compute_isovelo_velocity_per_isoform (
    isovelo_df,
    isoform=None,
    speed_up=None,
    projection_neighbor_size=30,
    smooth_k=0
)

with pd.option_context('display.max_columns', None):
    print(isovelo_df.head())

if 'cellID' in isovelo_df.columns and 'cellIndex' in isovelo_df.columns:
    isovelo_df = isovelo_df.drop(columns=['cellIndex'])

gene_to_plot = 'gene1'
plot_df = isovelo_df[isovelo_df.gene_name == gene_to_plot].copy()

clusters = pd.Categorical(plot_df['clusters'])
colors = plt.cm.tab20(np.linspace(0, 1, max(len(clusters.categories), 1)))
color_map = dict(zip(clusters.categories, colors))
plot_df['cluster_color'] = plot_df['clusters'].map(color_map)

isoforms = pd.Categorical(plot_df['isoform_name'])
isoform_colors = plt.cm.Set2(np.linspace(0, 1, max(len(isoforms.categories), 1)))
isoform_color_map = dict(zip(isoforms.categories, isoform_colors))

vel_mask = plot_df['velocity1'].notna() & plot_df['velocity2'].notna()
vx = plot_df.loc[vel_mask, 'velocity1'].to_numpy()
vy = plot_df.loc[vel_mask, 'velocity2'].to_numpy()
vel_mag = np.sqrt(vx**2 + vy**2)
max_mag = np.nanmax(vel_mag) if vel_mag.size else 0.0

n_isoforms = max(len(isoforms.categories), 1)
ncols = min(4, n_isoforms)
nrows = int(np.ceil(n_isoforms / ncols))
fig, axes = plt.subplots(
    nrows=nrows, ncols=ncols, figsize=(3.5 * ncols, 3.2 * nrows),
    sharex=True, sharey=True
)
axes = np.atleast_1d(axes).ravel()

if max_mag > 0:
    min_mag = np.nanpercentile(vel_mag, 5)  # drop only the tiniest velocities
    axis_span = max(plot_df['embedding1'].max() - plot_df['embedding1'].min(),
                    plot_df['embedding2'].max() - plot_df['embedding2'].min())
    scale = 0.05 * axis_span / max_mag

for ax, isoform in zip(axes, isoforms.categories):
    ax.scatter(
        plot_df['embedding1'],
        plot_df['embedding2'],
        c=plot_df['cluster_color'],
        s=4,
        alpha=0.35,
        linewidths=0
    )
    iso_mask = vel_mask & (plot_df['isoform_name'] == isoform)
    if iso_mask.any() and max_mag > 0:
        iso_xs = plot_df.loc[iso_mask, 'embedding1'].to_numpy()
        iso_ys = plot_df.loc[iso_mask, 'embedding2'].to_numpy()
        iso_vx = plot_df.loc[iso_mask, 'velocity1'].to_numpy()
        iso_vy = plot_df.loc[iso_mask, 'velocity2'].to_numpy()
        iso_mag = np.sqrt(iso_vx**2 + iso_vy**2)
        keep = iso_mag >= min_mag
        if keep.any():
            ax.quiver(
                iso_xs[keep], iso_ys[keep], iso_vx[keep] * scale, iso_vy[keep] * scale,
                angles='xy', scale_units='xy', scale=1,
                width=0.002, color=isoform_color_map[isoform], alpha=0.85
            )
    ax.set_title(str(isoform), fontsize=8)

for ax in axes[len(isoforms.categories):]:
    ax.axis('off')

for cluster, color in color_map.items():
    axes[0].scatter([], [], c=[color], label=str(cluster), s=20)

fig.suptitle(f"{gene_to_plot} (all isoforms)", y=1.02)
fig.legend(loc='upper right', fontsize=8, frameon=False)
for ax in axes:
    ax.set_xlabel("embedding1")
    ax.set_ylabel("embedding2")

plt.tight_layout()
fig.savefig(f"./examples/isovelo/simulation/simulation_result_velocity_plot.png", dpi=300, bbox_inches='tight')
plt.show()

dt = 0.5  # use the same dt you pass to isovelo_velocity
isovelo_df["ds_dt_pred"] = (isovelo_df["splice_predict"] - isovelo_df["splice"]) / dt