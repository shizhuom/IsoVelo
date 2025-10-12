import numpy as np
import pandas as pd
# from scipy.stats import dirichlet
import anndata
import scanpy as sc
import scvelo as scv
from velovi import preprocess_data, VELOVI
import matplotlib.pyplot as plt

def sigmoid(x, x0, k):
    return 1 / (1 + np.exp(-k * (x - x0)))

def generate_time_pseudo(N_GENES, N_PROGENITOR, N_FATE_A, N_FATE_B, PROGENITOR_START, PROGENITOR_END, FATE_START, FATE_END):
    progenitor_time = np.random.uniform(PROGENITOR_START, PROGENITOR_END, N_PROGENITOR)
    progenitor_time = np.tile(progenitor_time,(N_GENES,1))
    fateA_time = np.random.uniform(FATE_START, FATE_END, N_FATE_A)
    fateA_time = np.tile(fateA_time,(N_GENES,1))
    fateB_time = np.random.uniform(FATE_START, FATE_END, N_FATE_B)
    fateB_time = np.tile(fateB_time,(N_GENES,1))
    pseudo_time = np.concatenate((progenitor_time, fateA_time, fateB_time), axis=1)
    print(pseudo_time.shape)

    progenitor_s = np.ones((N_GENES, N_PROGENITOR))
    # mid = (FATE_START + FATE_END)/2
    # mid += (FATE_END - mid)/2
    # fateA_s = sigmoid(x=fateA_time, x0=mid, k=3.5)
    # fateB_s = sigmoid(x=fateB_time, x0=mid, k=3.5)
    fateA_s = sigmoid(x=fateA_time, x0=5, k=4)
    fateB_s = sigmoid(x=fateB_time, x0=5, k=4)

    s = np.concatenate((progenitor_s, fateA_s, fateB_s), axis=1)
    return pseudo_time, s

def generate_alpha(s, N_CELLS, N_PROGENITOR, N_FATE_A, N_FATE_B, N_HOUSEKEEPING, N_DE, N_IS, hk_low, hk_high, de_low, de_high):
    hk_alpha = np.tile(np.random.uniform(hk_low, hk_high, size=(N_HOUSEKEEPING, 1)),(1,N_CELLS))
    de_alpha_fatea = (1-s[0,N_PROGENITOR:N_PROGENITOR+N_FATE_A])*np.tile(np.random.uniform(hk_low, hk_high, size=(N_DE,1)),(1,N_FATE_A))+(s[0,N_PROGENITOR:N_PROGENITOR+N_FATE_A])*np.tile(np.random.uniform(de_low, de_high, size=(N_DE, 1)),(1,N_FATE_A))
    tmp = np.random.uniform(hk_low, hk_high, size=(N_DE, 1))
    de_alpha = np.concatenate((np.tile(tmp,(1,N_PROGENITOR)), de_alpha_fatea, np.tile(tmp, (1,N_FATE_B))), axis= 1)
    is_alpha = np.tile(np.random.uniform(hk_low, hk_high, size=(N_IS, 1)),(1,N_CELLS))
    alpha = np.concatenate((hk_alpha, de_alpha, is_alpha), axis= 0)
    return alpha

def generate_beta(s, N_CELLS, N_PROGENITOR, N_FATE_A, N_FATE_B, N_HOUSEKEEPING, N_DE, N_IS, hk_low, hk_high, de_low, de_high):
    hk_beta = np.tile(np.random.uniform(hk_low, hk_high, size=(N_HOUSEKEEPING, 1)),(1,N_CELLS))
    de_beta_fatea = (1-s[0,N_PROGENITOR:N_PROGENITOR+N_FATE_A])*np.tile(np.random.uniform(hk_low, hk_high, size=(N_DE, 1)),(1,N_FATE_A))+(s[0,N_PROGENITOR:N_PROGENITOR+N_FATE_A])*np.tile(np.random.uniform(de_low, de_high, size=(N_DE, 1)),(1,N_FATE_A))
    tmp = np.random.uniform(hk_low, hk_high, size=(N_DE, 1))
    de_beta = np.concatenate((np.tile(tmp,(1,N_PROGENITOR)), de_beta_fatea, np.tile(tmp, (1, N_FATE_B))), axis= 1)
    is_beta = np.tile(np.random.uniform(hk_low, hk_high, size=(N_IS, 1)),(1,N_CELLS))
    beta = np.concatenate((hk_beta, de_beta, is_beta), axis= 0)
    return beta

def generate_gamma(s, N_CELLS, N_PROGENITOR, N_FATE_A, N_FATE_B, N_HOUSEKEEPING, N_DE, N_IS, hk_low, hk_high, de_low, de_high):
    hk_gamma = np.tile(np.random.uniform(hk_low, hk_high, size=(N_HOUSEKEEPING, 1)),(1,N_CELLS))
    de_gamma_fatea = (1-s[0,N_PROGENITOR:N_PROGENITOR+N_FATE_A])*np.tile(np.random.uniform(hk_low, hk_high, size=(N_DE, 1)),(1,N_FATE_A))+(s[0,N_PROGENITOR:N_PROGENITOR+N_FATE_A])*np.tile(np.random.uniform(de_low, de_high, size=(N_DE, 1)),(1,N_FATE_A))
    tmp = np.random.uniform(hk_low, hk_high, size=(N_DE, 1))
    de_gamma = np.concatenate((np.tile(tmp,(1,N_PROGENITOR)), de_gamma_fatea, np.tile(tmp, (1,N_FATE_B))), axis= 1)
    is_gamma = np.tile(np.random.uniform(hk_low, hk_high, size=(N_IS, 1)),(1,N_CELLS))
    gamma = np.concatenate((hk_gamma, de_gamma, is_gamma), axis= 0)
    return gamma

def get_expected_count(alpha, beta, gamma, t, t_0):
    if (np.any(alpha <= 0) and np.any(beta <= 0) and np.any(gamma <= 0)):
        raise ValueError("Velocity parameters alpha, beta, and gamma must be positive.")
    
    delta_t = np.maximum(0, t-t_0)
    u = (alpha / beta) * (1 - np.exp(-beta * delta_t))

    exp_beta_t = np.exp(-beta * delta_t)
    exp_gamma_t = np.exp(-gamma * delta_t)
    s = (alpha / gamma) * (1 - exp_gamma_t) - \
        (alpha / (gamma - beta)) * (exp_beta_t - exp_gamma_t)

    return u, s

def get_observed_count(u,s):
    u[u < 0] = 0
    s[s < 0] = 0
    u = np.random.poisson(u)
    s = np.random.poisson(s)
    return u,s
    

def generate_isoform_proportion(s, transition, iso_low, iso_high, seed):
    N_GENES, N_CELLS = s.shape
    rng = np.random.default_rng(seed=seed)
    iso_count = rng.integers(low=iso_low, high=iso_high, size=N_GENES)
    major_iso = rng.integers(iso_count)
    tmp = rng.integers(iso_count - 1)
    alt_major_iso = tmp + (tmp >= major_iso)

    isoform_by_cell = []
    alt_isoform_by_cell = []
    groud_truth_pi = []
    alt_groud_truth_pi = []

    for gene_idx in range(N_GENES):
        #Alpha for dirichlet
        alpha = np.ones(iso_count[gene_idx])
        alpha[major_iso[gene_idx]] = 20
        alt_alpha = np.ones(iso_count[gene_idx])
        alt_alpha[alt_major_iso[gene_idx]] = 20

        gene_matrix = np.zeros((iso_count[gene_idx], N_CELLS), dtype=int)
        alt_gene_matrix = np.zeros((iso_count[gene_idx], N_CELLS), dtype=int)

        ground_truth_pi_matrix = np.zeros((iso_count[gene_idx], N_CELLS), dtype=float)
        alt_ground_truth_pi_matrix = np.zeros((iso_count[gene_idx], N_CELLS), dtype=float)
        for cell_idx in range(N_CELLS):
            pi = rng.dirichlet(alpha)
            alt_pi = rng.dirichlet(alt_alpha)
            trans_pi = (1-transition[gene_idx, cell_idx])*pi + transition[gene_idx, cell_idx]*alt_pi
            gene_matrix[:, cell_idx] = np.array(rng.multinomial(s[gene_idx, cell_idx], pi))
            alt_gene_matrix[:, cell_idx] = np.array(rng.multinomial(s[gene_idx, cell_idx], trans_pi))
            ground_truth_pi_matrix[:, cell_idx] = np.array(pi)
            alt_ground_truth_pi_matrix[:, cell_idx] = np.array(trans_pi)
        
        isoform_by_cell.append(gene_matrix)
        alt_isoform_by_cell.append(alt_gene_matrix)
        groud_truth_pi.append(ground_truth_pi_matrix)
        alt_groud_truth_pi.append(alt_ground_truth_pi_matrix)

    isoform_by_cell = np.vstack(isoform_by_cell)
    alt_isoform_by_cell = np.vstack(alt_isoform_by_cell)
    groud_truth_pi = np.vstack(groud_truth_pi)
    alt_groud_truth_pi = np.vstack(alt_groud_truth_pi)

    gene = np.repeat(s, iso_count, axis = 0)
    observed_proportion = np.divide(isoform_by_cell, gene, out=np.zeros_like(isoform_by_cell, dtype=float), where=gene!=0)
    alt_observed_proportion = np.divide(alt_isoform_by_cell, gene, out=np.zeros_like(alt_isoform_by_cell, dtype=float), where=gene!=0)

    return iso_count, isoform_by_cell, alt_isoform_by_cell, observed_proportion, alt_observed_proportion, groud_truth_pi, alt_groud_truth_pi

def generate_isoform_switching(isoform_by_cell, alt_isoform_by_cell, observed_proportion, alt_observed_proportion, groud_truth_pi, alt_groud_truth_pi, N_PROGENITOR, N_FATE_A, N_HOUSEKEEPING, N_DE, overlap_is):
    n_replace = int(N_FATE_A * overlap_is)
    cols_to_replace = np.random.choice(np.arange(N_PROGENITOR, N_PROGENITOR+N_FATE_A), size=n_replace, replace=False)
    isoform = isoform_by_cell
    isoform[N_HOUSEKEEPING+N_DE:, N_PROGENITOR+N_FATE_A:] = alt_isoform_by_cell[N_HOUSEKEEPING+N_DE:, N_PROGENITOR+N_FATE_A:]
    isoform[N_HOUSEKEEPING:N_HOUSEKEEPING+N_DE, cols_to_replace] = alt_isoform_by_cell[N_HOUSEKEEPING:N_HOUSEKEEPING+N_DE, cols_to_replace]
    observed = observed_proportion
    observed[N_HOUSEKEEPING+N_DE:, N_PROGENITOR+N_FATE_A:] = alt_observed_proportion[N_HOUSEKEEPING+N_DE:, N_PROGENITOR+N_FATE_A:]
    observed[N_HOUSEKEEPING:N_HOUSEKEEPING+N_DE, cols_to_replace] = alt_observed_proportion[N_HOUSEKEEPING:N_HOUSEKEEPING+N_DE, cols_to_replace]
    ground_truth = groud_truth_pi
    ground_truth[N_HOUSEKEEPING+N_DE:, N_PROGENITOR+N_FATE_A:] = alt_groud_truth_pi[N_HOUSEKEEPING+N_DE:, N_PROGENITOR+N_FATE_A:]
    ground_truth[N_HOUSEKEEPING:N_HOUSEKEEPING+N_DE, cols_to_replace] = alt_groud_truth_pi[N_HOUSEKEEPING:N_HOUSEKEEPING+N_DE, cols_to_replace]
    return isoform, observed, ground_truth

def create_anndata(alpha, beta, gamma, u, s, time, iso_count, isoform, observed, ground_truth, N_PROGENITOR, N_FATE_A, N_FATE_B, N_HOUSEKEEPING, N_DE, N_IS, overlap_is):
    fates = ['progenitor'] * N_PROGENITOR + ['fate_A'] * N_FATE_A + ['fate_B'] * N_FATE_B
    pseudotime = time[0,:]
    obs_df = pd.DataFrame({
        'cell_state': fates,
        'pseudotime': pseudotime
    })
    var_df = pd.DataFrame({'gene_category': (['Housekeeping'] * N_HOUSEKEEPING +
                                             ['DE'] * N_DE +
                                             ['IS'] * N_IS)})
    var_df.index = var_df.index.astype(str)
    var_df['n_isoforms'] = iso_count
    adata = anndata.AnnData(
        X=(u + s).T,
        obs=obs_df,
        var=var_df
    )
    adata.layers['unspliced'] = u.T
    adata.layers['spliced'] = s.T
    adata.layers['alpha'] = alpha.T
    adata.layers['beta'] = beta.T
    adata.layers['gamma'] = gamma.T

    split_indices = np.cumsum(iso_count)[:-1]
    spliced_by_gene = np.split(isoform, split_indices, axis=0)
    prop_gt_by_gene = np.split(ground_truth, split_indices, axis=0)
    prop_obs_by_gene = np.split(observed, split_indices, axis=0)
    
    adata.uns['spliced_isoform_counts'] = {str(idx): arr for idx, arr in enumerate(spliced_by_gene)}
    adata.uns['proportions_ground_truth'] = {str(idx): arr for idx, arr in enumerate(prop_gt_by_gene)}
    adata.uns['proportions_observed'] = {str(idx): arr for idx, arr in enumerate(prop_obs_by_gene)}

    return adata

def simulate(N_PROGENITOR, N_FATE_A, N_FATE_B, N_HOUSEKEEPING, N_DE, N_IS, hk_low_alpha, hk_high_alpha, de_low_alpha, de_high_alpha, hk_low_beta, hk_high_beta, de_low_beta, de_high_beta, hk_low_gamma, hk_high_gamma, de_low_gamma, de_high_gamma,PROGENITOR_START, PROGENITOR_END, FATE_START, FATE_END, iso_low, iso_high, overlap_is, seed):
    N_CELLS = N_PROGENITOR + N_FATE_A + N_FATE_B
    N_GENES = N_HOUSEKEEPING + N_DE + N_IS
    time, transition = generate_time_pseudo(N_GENES, N_PROGENITOR, N_FATE_A, N_FATE_B, PROGENITOR_START, PROGENITOR_END, FATE_START, FATE_END)
    alpha = generate_alpha(transition, N_CELLS, N_PROGENITOR, N_FATE_A, N_FATE_B, N_HOUSEKEEPING, N_DE, N_IS, hk_low_alpha, hk_high_alpha, de_low_alpha, de_high_alpha)
    beta = generate_beta(transition, N_CELLS, N_PROGENITOR, N_FATE_A, N_FATE_B, N_HOUSEKEEPING, N_DE, N_IS, hk_low_beta, hk_high_beta, de_low_beta, de_high_beta)
    gamma = generate_gamma(transition, N_CELLS, N_PROGENITOR, N_FATE_A, N_FATE_B, N_HOUSEKEEPING, N_DE, N_IS, hk_low_gamma, hk_high_gamma, de_low_gamma, de_high_gamma)
    gene_u, gene_s = get_expected_count(alpha, beta, gamma, time, 0)
    gene_u, gene_s = get_observed_count(gene_u, gene_s)
    iso_count, isoform_by_cell, alt_isoform_by_cell, observed_proportion, alt_observed_proportion, groud_truth_pi, alt_groud_truth_pi = generate_isoform_proportion(gene_s, transition, iso_low, iso_high, seed)
    isoform, observed, ground_truth =generate_isoform_switching(isoform_by_cell, alt_isoform_by_cell, observed_proportion, alt_observed_proportion, groud_truth_pi, alt_groud_truth_pi, N_PROGENITOR, N_FATE_A, N_HOUSEKEEPING, N_DE, overlap_is)
    adata = create_anndata(alpha, beta, gamma, gene_u, gene_s, time, iso_count, isoform, observed, ground_truth, N_PROGENITOR, N_FATE_A, N_FATE_B, N_HOUSEKEEPING, N_DE, N_IS, overlap_is)
    return adata

def adata_preprocess(adata, seed, plot = False):
    plt.rcParams['savefig.transparent'] = True
    adata.X = adata.layers['spliced'].copy()
    scv.pp.filter_and_normalize(adata, min_shared_counts=20)  
    scv.pp.moments(adata, n_pcs=30, n_neighbors=30)

    sc.tl.pca(adata, svd_solver='arpack', random_state=seed)
    sc.pp.neighbors(adata, n_pcs=30, random_state=seed)
    sc.tl.umap(adata, random_state=seed)
    if plot:
        sc.pl.umap(adata, color='cell_state',
                title='UMAP Based on Gene Counts', show = False)
        plt.savefig('adata_gene_umap.svg')
        plt.close
    
    isoforms = list(adata.uns['spliced_isoform_counts'].values())
    isoforms = np.concatenate(isoforms, axis=0).T
    adata_isoform = anndata.AnnData(isoforms)
    print(adata_isoform)
    adata_isoform.obs['cell_state'] = adata.obs['cell_state'].values
    sc.pp.normalize_total(adata_isoform, target_sum=1e4)
    sc.pp.log1p(adata_isoform)
    sc.tl.pca(adata_isoform, n_comps=50, random_state=seed)
    sc.pp.neighbors(adata_isoform, n_pcs=50, random_state=seed)
    sc.tl.umap(adata_isoform, random_state=seed)
    if plot:
        sc.pl.umap(
            adata_isoform,
            color='cell_state', 
            title='UMAP Based on Isoform Counts',
            frameon=False
        )
        plt.savefig('adata_isoform_umap.svg')
        plt.close
    adata.obsm['X_umap_isoform'] = adata_isoform.obsm['X_umap']

def check_gene_scvelo(adata, latent_time = False):
    plt.rcParams['savefig.transparent'] = True
    scv.tl.velocity(adata)
    scv.tl.velocity_graph(adata)
    scv.pl.velocity_embedding_stream(adata, basis='umap', color='cell_state',
                                     title='Gene-Level ScVelo Velocity on Gene-Based UMAP',
                                     show= False)
    plt.savefig('adata_gene_velocity_scvelo.svg')
    plt.close

    if latent_time:
        scv.tl.recover_dynamics(adata)
        scv.tl.latent_time(adata)
        scv.pl.umap(adata, color='latent_time',
                    title = 'Gene-based UMAP Colored by Latent Time (ScVelo)',
                    color_map = 'viridis', show = False)
        plt.savefig('adata_gene_latent_time_scvelo.svg')
        plt.close

def check_isoform_scvelo(adata, latent_time = False):
    plt.rcParams['savefig.transparent'] = True
    scv.tl.velocity(adata)
    scv.tl.velocity_graph(adata)
    scv.pl.velocity_embedding_stream(
        adata,
        basis='umap_isoform',  # Tell scvelo to use these specific coordinates
        color='cell_state',    # Color by your ground truth cell states
        title='Gene-Level ScVelo Velocity on Isoform-Based UMAP',
        show=False)
    plt.savefig('adata_isoform_velocity_scvelo.svg')
    plt.close

    if latent_time:
        scv.pl.scatter(adata, basis='umap_isoform', color='latent_time',
                    title = 'Isoform-based UMAP Colored by Latent Time (ScVelo)',
                    color_map = 'viridis', show = False)
        plt.savefig('adata_isoform_latent_time_scvelo.svg')
        plt.close

def check_gene_velovi(adata, latent_time = False):
    plt.rcParams['savefig.transparent'] = True
    VELOVI.setup_anndata(adata, spliced_layer="Ms", unspliced_layer="Mu")
    vae = VELOVI(adata)
    vae.train()
    scv.tl.velocity_graph(adata)
    scv.pl.velocity_embedding_stream(adata, basis='umap', color='cell_state',
                                     title='Gene-Level VeloVI Velocity on Gene-Based UMAP',
                                     show = False)
    plt.savefig('adata_gene_velocity_velovi.svg')
    plt.close

    if latent_time:
        latent_time_matrix = vae.get_latent_time()
        print("latent time matrix shape:", latent_time_matrix.shape)
        print(latent_time_matrix.head())
        aggregated_latent_time = latent_time_matrix.mean(axis = 1)
        aggregated_latent_time = np.max(aggregated_latent_time) - aggregated_latent_time
        adata.obs['velovi_latent_time_mean'] = aggregated_latent_time
        scv.pl.umap(adata,
                    color='velovi_latent_time_mean',
                    title = "Gene_based UMAP Colored by Latent Time (Velovi)",
                    color_map='viridis',
                    show = False)
        plt.savefig('adata_gene_vlatent_time_velovi.svg')
        plt.close
    

def check_isoform_velovi(adata, latent_time = False):
    plt.rcParams['savefig.transparent'] = True
    # scv.pp.filter_and_normalize(adata, min_shared_counts=20)
    # scv.pp.moments(adata, n_pcs=30, n_neighbors=30)
        
    VELOVI.setup_anndata(adata, spliced_layer="Ms", unspliced_layer="Mu")
    vae = VELOVI(adata)
    vae.train()
    scv.tl.velocity_graph(adata)
    scv.pl.velocity_embedding_stream(
        adata,
        basis='umap_isoform',  # Use the isoform-based UMAP coordinates for plotting
        color='cell_state',
        title='Isoform-Level VeloVI Velocity on Isoform-Based UMAP'
    )
    if latent_time:
        latent_time_matrix = vae.get_latent_time()
        print("latent time matrix shape:", latent_time_matrix.shape)
        print(latent_time_matrix.head())
        aggregated_latent_time = latent_time_matrix.mean(axis = 1)
        aggregated_latent_time = np.max(aggregated_latent_time) - aggregated_latent_time
        adata.obs['velovi_latent_time_mean'] = aggregated_latent_time
        sc.pl.scatter(adata,
                      basis='umap_isoform',
                      color='velovi_latent_time_mean',
                      title = "Isoform_based UMAP Colored by Latent Time (Velovi)",
                      color_map='viridis',
                      show = False)
        plt.savefig('adata_isoform_vlatent_time_velovi.svg')
        plt.close

N_PROGENITOR = 1000
N_FATE_A = 500
N_FATE_B = 500
N_CELLS = N_PROGENITOR + N_FATE_A + N_FATE_B

N_HOUSEKEEPING = 500
N_DE = 250
N_IS = 250
N_GENES = N_HOUSEKEEPING + N_DE + N_IS

hk_low_alpha = 4
hk_high_alpha = 6
de_low_alpha = 12
de_high_alpha = 18

hk_low_beta = 1.8
hk_high_beta = 2.2
de_low_beta = 3.6
de_high_beta = 4.4

hk_low_gamma = 0.9
hk_high_gamma = 1.1
de_low_gamma = 0.9
de_high_gamma = 1.1

PROGENITOR_START = 2 
PROGENITOR_END = 4
FATE_START = 4
FATE_END = 6

iso_low = 2
iso_high = 6

# Set a random seed for reproducibility
seed = 33
np.random.seed(seed)

overlap_is = 0
adata = simulate(N_PROGENITOR, N_FATE_A, N_FATE_B, N_HOUSEKEEPING, N_DE, N_IS, hk_low_alpha, hk_high_alpha, de_low_alpha, de_high_alpha, hk_low_beta, hk_high_beta, de_low_beta, de_high_beta, hk_low_gamma, hk_high_gamma, de_low_gamma, de_high_gamma,PROGENITOR_START, PROGENITOR_END, FATE_START, FATE_END, iso_low, iso_high, overlap_is, seed)
filename = "simulated_data_continuous_" + str(overlap_is) + ".h5ad"
print(filename)
adata.write_h5ad(filename)
adata = anndata.read_h5ad(filename)
adata_preprocess(adata, seed)
check_gene_scvelo(adata, latent_time = True)
check_isoform_scvelo(adata, latent_time= True)
adata = anndata.read_h5ad(filename)
adata_preprocess(adata, seed, plot=False)
check_gene_velovi(adata, latent_time=True)
check_isoform_velovi(adata, latent_time=True)

print("Object saved successfully.")
