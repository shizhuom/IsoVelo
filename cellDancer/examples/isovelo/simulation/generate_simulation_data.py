import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)

n_per_cluster = 500
n_clusters = 2
n_cells = n_per_cluster * n_clusters
isoforms = np.array(["s1", "s2", "s3"])

# Cell IDs and clusters
cell_ids = np.array([f"cell{i}" for i in range(1, n_cells + 1)])
clusters = np.array([1]*n_per_cluster + [2]*n_per_cluster)

# Sample time t ~ Unif[0, 40] for each cell
t = rng.uniform(0.0, 40.0, size=n_cells)

# Sample parameters per cell
a = rng.uniform(3.0, 5.0, size=n_cells)

b1 = np.empty(n_cells); b2 = np.empty(n_cells); b3 = np.empty(n_cells)
c1 = np.empty(n_cells); c2 = np.empty(n_cells); c3 = np.empty(n_cells)

# Cluster 1 ranges
idx1 = clusters == 1
b1[idx1] = rng.uniform(0.4, 0.6, size=idx1.sum())
b2[idx1] = rng.uniform(1.0, 3.0, size=idx1.sum())
b3[idx1] = rng.uniform(3.0, 5.0, size=idx1.sum())
c1[idx1] = rng.uniform(0.4, 0.6, size=idx1.sum())
c2[idx1] = rng.uniform(0.4, 0.6, size=idx1.sum())
c3[idx1] = rng.uniform(0.4, 0.6, size=idx1.sum())

# Cluster 2 ranges
idx2 = clusters == 2
b1[idx2] = rng.uniform(4.0, 6.0, size=idx2.sum())
b2[idx2] = rng.uniform(1.0, 3.0, size=idx2.sum())
b3[idx2] = rng.uniform(0.3, 0.5, size=idx2.sum())
c1[idx2] = rng.uniform(0.8, 1.0, size=idx2.sum())
c2[idx2] = rng.uniform(0.4, 0.6, size=idx2.sum())
c3[idx2] = rng.uniform(0.2, 0.3, size=idx2.sum())

# Embeddings per cell (time-correlated trajectories)
t_min = t.min()
t_range = t.max() - t.min()
t_norm = (t - t_min) / (t_range if t_range > 0 else 1.0)

emb1 = np.empty(n_cells); emb2 = np.empty(n_cells)
# Cluster 1: forward trajectory
emb1[idx1] = 1.5 + 4.0 * t_norm[idx1] + rng.normal(0.0, 0.4, size=idx1.sum())
emb2[idx1] = -2.0 + 2.5 * t_norm[idx1] + rng.normal(0.0, 0.4, size=idx1.sum())
# Cluster 2: different direction/offset
emb1[idx2] = -4.0 + 3.0 * t_norm[idx2] + rng.normal(0.0, 0.5, size=idx2.sum())
emb2[idx2] = 3.5 - 2.0 * t_norm[idx2] + rng.normal(0.0, 0.5, size=idx2.sum())

# Closed-form solutions with u0=0, s0=0
B = b1 + b2 + b3
# u(t) = a/B * (1 - exp(-B t))
u = (a / B) * (1.0 - np.exp(-B * t))

def s_of_t(bk, ck, a, B, t):
    """
    s(t) with u0=0, s0=0 for:
        du/dt = a - B u
        ds/dt = bk u - ck s
    Handles ck == B via limit.
    """
    # general terms
    term1 = (bk * a) / (B * ck) * (1.0 - np.exp(-ck * t))  # from steady forcing a/B
    # resonance-safe computation for the transient coupling
    eps = 1e-10
    same = np.abs(ck - B) < eps
    out = np.empty_like(t)

    # ck != B
    not_same = ~same
    out[not_same] = term1[not_same] - (bk[not_same] * a[not_same] / B[not_same]) * (
        (np.exp(-B[not_same] * t[not_same]) - np.exp(-ck[not_same] * t[not_same])) / (ck[not_same] - B[not_same])
    )

    # ck == B: s = bk*a/(B*ck)*(1-e^{-ck t}) + bk*(u0-a/B)*t*e^{-ck t}, with u0=0 => -bk*a/B * t e^{-ck t}
    out[same] = term1[same] - (bk[same] * a[same] / B[same]) * (t[same] * np.exp(-ck[same] * t[same]))
    return out

s1 = s_of_t(b1, c1, a, B, t)
s2 = s_of_t(b2, c2, a, B, t)
s3 = s_of_t(b3, c3, a, B, t)

# Build the sampling results dataframe (long format: 3 isoforms per cell)
gene_name = "gene1"

cell_rep = np.repeat(cell_ids, len(isoforms))
cluster_rep = np.repeat(clusters, len(isoforms))
emb1_rep = np.repeat(emb1, len(isoforms))
emb2_rep = np.repeat(emb2, len(isoforms))
iso_rep = np.tile(isoforms, n_cells)

u_rep = np.repeat(u, len(isoforms))
splice_rep = np.concatenate([s1[:, None], s2[:, None], s3[:, None]], axis=1).reshape(-1)

df_counts = pd.DataFrame({
    "gene_name": gene_name,
    "isoform_name": iso_rep,
    "unsplice": u_rep,
    "splice": splice_rep,
    "cellID": cell_rep,
    "clusters": cluster_rep,
    "embedding1": emb1_rep,
    "embedding2": emb2_rep
})

# Ground-truth dataframe (per isoform row, with alpha=a, beta=bk, gamma=ck, time=t)
alpha_rep = np.repeat(a, len(isoforms))
beta_rep = np.concatenate([b1[:, None], b2[:, None], b3[:, None]], axis=1).reshape(-1)
gamma_rep = np.concatenate([c1[:, None], c2[:, None], c3[:, None]], axis=1).reshape(-1)
time_rep = np.repeat(t, len(isoforms))

df_truth = pd.DataFrame({
    "gene_name": gene_name,
    "isoform_name": iso_rep,
    "cellID": cell_rep,
    "clusters": cluster_rep,
    "embedding1": emb1_rep,
    "embedding2": emb2_rep,
    "alpha": alpha_rep,
    "beta": beta_rep,
    "gamma": gamma_rep,
    "time": time_rep
})

# Save to CSVs for convenience
store_path = "./examples/isovelo/simulation/"
df_counts.to_parquet(
    os.path.join(store_path, "sim_counts.parquet"),
    engine="pyarrow",
    index=False
)
df_truth.to_parquet(
    os.path.join(store_path, "sim_ground_truth.parquet"),
    engine="pyarrow",
    index=False
)

# Plot velocity
u_cell = (
    df_counts.groupby("cellID", as_index=False)
    .agg(unsplice=("unsplice", "first"))
)

cell_params = (
    df_truth.groupby(["cellID", "clusters", "embedding1", "embedding2"], as_index=False)
    .agg(alpha=("alpha", "first"),
         time=("time", "first"))
)

B_cell = (
    df_truth.groupby("cellID", as_index=False)
    .agg(B=("beta", "sum"))
)

cell_level = cell_params.merge(u_cell, on="cellID").merge(B_cell, on="cellID")
cell_level["du_dt"] = cell_level["alpha"] - cell_level["B"] * cell_level["unsplice"]  # du/dt = a - B*u

df = df_counts.merge(cell_level[["cellID", "du_dt"]], on="cellID", how="left")

df = df.merge(
    df_truth[["cellID", "isoform_name", "beta", "gamma"]],
    on=["cellID", "isoform_name"],
    how="left"
)

df["ds_dt"] = df["beta"] * df["unsplice"] - df["gamma"] * df["splice"]

isoforms = ["s1", "s2", "s3"]
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharex=True, sharey=True)
fig.suptitle("gene1 (all isoforms)", fontsize=16, y=1.03)

handles, labels = [], []
speed = np.sqrt(df["du_dt"].to_numpy()**2 + df["ds_dt"].to_numpy()**2)
p95 = np.percentile(speed, 95) if np.any(speed > 0) else 1.0
target_len = 0.35  # in embedding units
factor = target_len / (p95 if p95 > 0 else 1.0)

for ax, iso in zip(axes, isoforms):
    sub = df[df["isoform_name"] == iso]

    # scatter by cluster
    for cl in sorted(sub["clusters"].unique()):
        m = sub["clusters"] == cl
        sc = ax.scatter(sub.loc[m, "embedding1"], sub.loc[m, "embedding2"], s=12, alpha=0.65)
        if iso == isoforms[0]:
            handles.append(sc); labels.append(str(cl))

    # quiver using ALL points (no downsampling)
    ax.quiver(
        sub["embedding1"].to_numpy(),
        sub["embedding2"].to_numpy(),
        sub["du_dt"].to_numpy() * factor,
        sub["ds_dt"].to_numpy() * factor,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.0025,
        alpha=0.65,
    )

    ax.set_title(iso, fontsize=13)
    ax.set_xlabel("embedding1")
    ax.set_ylabel("embedding2")

fig.legend(handles, labels, loc="upper right", frameon=False)
plt.tight_layout()
fig.savefig(f"./examples/isovelo/simulation/simulation_groundtruth_velocity_plot.png", dpi=300, bbox_inches='tight')
plt.show()
