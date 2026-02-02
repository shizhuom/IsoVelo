import numpy as np
import torch
import math
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import anndata
import scanpy as sc
import scvelo as scv
#import wandb

EPS = 1e-6
filename = "test_simulated_data_continuous_downregulated0.h5ad"
adata = anndata.read_h5ad(filename)
print(adata)

np.random.seed(42)

G = adata.n_vars
G_prime = sum(adata.var['n_isoforms'])
isoform_counts = adata.var['n_isoforms'].to_dict()
gene_list = list(isoform_counts.keys())
S_gene = adata.layers['spliced']
U_gene = adata.layers['unspliced']
proportions = list(adata.uns['proportions_observed'].values())
P_proportions = np.concatenate(proportions, axis=0).T
combined_input = np.concatenate([U_gene, S_gene, P_proportions], axis=1) #Before this step, the U, S, P must be aligned using cell rows
combined_input_tensor = torch.tensor(combined_input, dtype=torch.float32)

isoforms = list(adata.uns['spliced_isoform_counts'].values())
isoforms = np.concatenate(isoforms, axis=0).T

class MyDataset(Dataset):
    def __init__(self, adata):
        # 确保转换为 tensor
        U_gene = adata.layers['unspliced']
        U_size = U_gene.sum(axis=1, keepdims= True)
        U_norm = (U_gene/U_size) * 1e3
        
        S_gene = adata.layers['spliced']
        S_size = S_gene.sum(axis=1, keepdims= True)
        S_norm = (S_gene/S_size) * 1e3

        tmp = adata.uns['proportions_observed']
        gene_order = list(adata.var_names)
        arrays = [tmp[g] for g in gene_order]

        P_proportions = np.concatenate(arrays, axis=0).T
        
        combined_input = np.concatenate([U_norm, S_norm, P_proportions], axis=1) #Before this step, the U, S, P must be aligned using cell rows
        self.X = torch.tensor(combined_input, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]
    
np.random.seed(42)
dataset = MyDataset(adata)
dataloader = DataLoader(dataset=dataset, batch_size=200, shuffle=True)
# for idx, img in enumerate(dataloader):
#     print(img.shape)
    
    
from scvi.nn import Encoder, FCLayers
input_dim = combined_input.shape[1]  # Number of features in the input
latent_dim = 600
# Define the input dimensions
n_layers = 3
dropout_rate = 0.1
n_hidden = input_dim * 2
latent_distribution = 'ln'
use_batch_norm_encoder = 'encoder'
use_layer_norm_encoder = 'encoder'
var_activation = torch.nn.Softplus()
encoder = Encoder(
            n_input=input_dim,
            n_output=latent_dim,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            distribution=latent_distribution,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
            var_activation=var_activation,
            activation_fn=torch.nn.ReLU,
        )


class FullDecoder(nn.Module):
    """
    Decodes latent z into:
      - isoform proportions P_mean
      - kinetics-based U/S mean & std
    with four kinetic regimes defined by t_cell vs. t_ss and t_switch:
      1) induction (t < min(t_ss, t_switch))
      2) induction steady-state (t_ss <= t <= t_switch)
      3) repression A (t_switch < t_ss  and t >= t_switch)
      4) repression B (t_ss < t_switch and t >= t_switch)

    Switch times t_switch are sampled per-gene from Beta(a1, b1).
    """
    def __init__(self, latent_dim, isoform_counts, gene_list, t_max = 20, num_repeats=10, eps=1e-3):
        super().__init__()
        self.gene_list = gene_list
        self.t_max = t_max
        self.num_repeats = num_repeats
        self.eps = eps
        # isoform proportion nets per gene FCLayers(n_in=latent_dim, n_out=isoform_counts[gene])
        self.prop_nets = nn.ModuleDict({
            gene: FCLayers(n_in=latent_dim, n_out=isoform_counts[gene])
            for gene in gene_list
        }) # isoform_counts[gene]   
        G = len(gene_list)
        # Beta params for gene switch times
        self.raw_a1 = nn.Parameter(torch.randn(G))
        self.raw_b1 = nn.Parameter(torch.randn(G)) # a1 
        # Beta param nets for cell times per gene
        self.f_a2 = nn.ModuleDict({
            gene: FCLayers(n_in=latent_dim, n_out=1, activation_fn=nn.Softplus)
            for gene in gene_list
        })
        self.f_b2 = nn.ModuleDict({
            gene: FCLayers(n_in=latent_dim, n_out=1, activation_fn=nn.Softplus)
            for gene in gene_list
        })
        # kinetic rates per gene
        self.raw_alpha = nn.Parameter(torch.randn(G))
        self.raw_beta  = nn.Parameter(torch.randn(G))
        self.raw_gamma = nn.Parameter(torch.randn(G))

    def forward(self, z):
        N, G = z.size(0), len(self.gene_list)
        # positive enforcement
        alpha = F.softplus(self.raw_alpha)
        beta  = F.softplus(self.raw_beta)
        gamma = F.softplus(self.raw_gamma)
        a1 = F.softplus(self.raw_a1)
        b1 = F.softplus(self.raw_b1) # t^s

        p_stack, u_stack, s_stack = [], [], []
        for _ in range(self.num_repeats):
            # proportions
            p_list = [F.softmax(self.prop_nets[g](z), dim=1) for g in self.gene_list] # U=1000 S=1000 P=3341 
            p_stack.append(torch.cat(p_list, dim=1))
            # sample times
            a2 = torch.cat([self.f_a2[g](z)+EPS for g in self.gene_list], dim=1)  # (N,G)
            b2 = torch.cat([self.f_b2[g](z)+EPS for g in self.gene_list], dim=1)
            # print(f"a2: {a2}, b2: {b2}")
            t_cell   = torch.distributions.Beta(a2, b2).rsample()
            t_cell = t_cell * self.t_max
            t_switch = torch.distributions.Beta(a1, b1).rsample()            # (G,)
            t_switch = t_switch * self.t_max
            t_switch_mat = t_switch.unsqueeze(0).expand(N,G)
            # steady-state induction time
            t1 = (1.0/beta) * torch.log((alpha/beta)/self.eps)
            A  = alpha/gamma + alpha/(gamma-beta)
            B  = alpha/(gamma-beta)
            t2 = (1.0/torch.min(beta, gamma)) * torch.log((A.abs()+B.abs())/self.eps)
            t_ss = torch.max(t1, t2)  # (N,G)
            # regime masks
            t_min     = torch.min(t_ss, t_switch_mat)
            mask1     = t_cell < t_min
            mask2     = (t_cell >= t_ss) & (t_cell <= t_switch_mat)
            mask3     = (t_switch_mat < t_ss) & (t_cell > t_switch_mat)
            mask4     = (t_ss <= t_switch_mat) & (t_cell > t_switch_mat)
            # regime 1: induction
            u1 = (alpha/beta) * (1 - torch.exp(-beta*t_cell))
            s1 = alpha/gamma - (alpha/(gamma-beta)) * torch.exp(-beta*t_cell) \
               + (-alpha/gamma + alpha/(gamma-beta)) * torch.exp(-gamma*t_cell)
            # regime 2: induction steady-state
            u2 = alpha/beta
            s2 = alpha/gamma
            # regime 3: repression A
            u3 = (alpha/beta)*(1-torch.exp(-beta*t_switch_mat)) * torch.exp(-beta*t_cell)
            s3 = alpha/gamma * torch.exp(-beta*t_cell) \
               - alpha/(gamma-beta) * torch.exp(-beta*(t_cell+t_switch_mat)) \
               + (alpha/gamma - alpha/(gamma-beta))*torch.exp(-gamma*t_cell) \
               + (alpha/(gamma-beta) - alpha/gamma)*torch.exp(-gamma*(t_cell+t_switch_mat))
            # regime 4: repression B
            u4 = - (alpha/beta) * torch.exp(-beta*t_cell)
            s4 = alpha/gamma * torch.exp(-beta*t_cell) \
               + (alpha/gamma - alpha/(gamma-beta)) * torch.exp(-gamma*t_cell)
            # combine
            u_val = torch.where(mask1, u1,
                     torch.where(mask2, u2,
                     torch.where(mask3, u3, u4)))
            s_val = torch.where(mask1, s1,
                     torch.where(mask2, s2,
                     torch.where(mask3, s3, s4)))
            u_stack.append(u_val)
            s_stack.append(s_val)
        # empirical stats
        p_mean = torch.stack(p_stack,0).mean(0)
        u_arr  = torch.stack(u_stack,0)
        s_arr  = torch.stack(s_stack,0)
        u_mean = u_arr.mean(0); u_std = u_arr.std(0) + 1e-6
        s_mean = s_arr.mean(0); s_std = s_arr.std(0) + 1e-6
        return {'p_mean': p_mean,
                'u_mean': u_mean, 'u_std': u_std,
                's_mean': s_mean, 's_std': s_std}
        
        



def vae_loss(mu, logvar, dec_out, U_obs, S_obs, P_obs, lam_p=1.0, c_k=1.0):
    """
    VAE loss = 
      – [ log p(u|z) + log p(s|z) ]
      + λ_p · ||clr(P_obs) – clr(P_pred)||²
      + KL( q(z|x) || p(z) )
    """
    device = mu.device

    # helper to convert numpy→tensor
    def to_tensor(x):
        return torch.tensor(x, dtype=torch.float32, device=device) \
               if isinstance(x, np.ndarray) else x

    U_obs = to_tensor(U_obs)
    S_obs = to_tensor(S_obs)
    P_obs = to_tensor(P_obs)

    # --- U reconstruction log-likelihood ---
    u_mean, u_std = dec_out['u_mean'], dec_out['u_std']
    ru = -0.5 * (
        torch.log(2 * math.pi * (c_k*u_std)**2) +
        ((U_obs - u_mean)**2) / ((c_k*u_std)**2)
    )
    recon_u = F.mse_loss(u_mean, U_obs)
    # recon_u = ru.sum()
    # recon_u = ru.mean()

    # --- S reconstruction log-likelihood ---
    s_mean, s_std = dec_out['s_mean'], dec_out['s_std']
    # rs = -0.5 * (
    #     torch.log(2 * math.pi * (c_k*s_std)**2) +
    #     ((S_obs - s_mean)**2) / ((c_k*s_std)**2)
    # )
    # recon_s = rs.sum()
    recon_s = F.mse_loss(s_mean, S_obs)
    #recon_s = rs.mean()

    # --- Aitchison distance on proportions ---
    p_pred = dec_out['p_mean']
    eps_clr = 1e-9
    log_p_obs  = torch.log(P_obs  + eps_clr)
    log_p_pred = torch.log(p_pred + eps_clr)
    # centered log-ratio
    clr_obs  = log_p_obs  - log_p_obs.mean(dim=1, keepdim=True)
    clr_pred = log_p_pred - log_p_pred.mean(dim=1, keepdim=True)
    dist     = ((clr_obs - clr_pred)**2).sum()
    #dist     = ((clr_obs - clr_pred)**2).mean()
    # recon_p  = lam_p * dist
    recon_p = F.mse_loss(p_pred, P_obs)
    # --- KL divergence q(z|x) || N(0,I) ---
    var = torch.exp(logvar)
    # kl  = 0.5 * torch.sum(mu**2 + var - logvar - 1)
    kl = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0)
    #kl  = 0.5 * torch.mean(mu**2 + var - logvar - 1)

    # --- total loss ---
    # we *maximize* recon_u + recon_s, so we minimize -(recon_u+recon_s)
    loss = (recon_u + recon_s) + recon_p + 5 * kl
    print(f"recon_u: {recon_u.item()}, recon_s: {recon_s.item()}, recon_p: {recon_p.item()}, kl: {kl.item()}")

    return loss, {
        'recon_u': recon_u,
        'recon_s': recon_s,
        'recon_p': recon_p,
        'kl'     : kl
    }
    
    

from torch import optim
best_loss = 1e7
best_epoch = 0
loss_history = []

valid_losses = []
train_losses = []

epochs = 15

# encoder = VAE_Encoder(5541, 128)
# encoder = Encoder(
#             n_input=input_dim,
#             n_output=128,
#             n_layers=n_layers,
#             n_hidden=n_hidden,
#             dropout_rate=dropout_rate,
#             distribution=latent_distribution,
#             use_batch_norm=use_batch_norm_encoder,
#             use_layer_norm=use_layer_norm_encoder,
#             var_activation=var_activation,
#             activation_fn=torch.nn.ReLU,
#         )
G = adata.n_vars
G_prime = sum(adata.var['n_isoforms'])
isoform_counts = adata.var['n_isoforms'].to_dict()
gene_list = list(isoform_counts.keys())
gene_list = list(isoform_counts.keys())

isoforms = list(adata.uns['spliced_isoform_counts'].values())
isoforms = np.concatenate(isoforms, axis=0).T

model = FullDecoder(
    latent_dim=latent_dim,
    isoform_counts=isoform_counts,
    gene_list=gene_list,
    num_repeats=10,
    eps=1e-3
)
device = "cuda"
encoder = encoder.to(device)
model = model.to(device)
params_to_optimize = list(encoder.parameters()) + list(model.parameters())
optimizer = optim.Adam(params_to_optimize, lr=3e-2)
# import debugpy
# debugpy.listen(("localhost", 4566))
# print("Waiting for debugger to attach...")
# debugpy.wait_for_client()
for epoch in range(epochs):
    print(f"Epoch {epoch}")

    model.train()
    train_loss = 0.

    for idx, x in enumerate(dataloader):
        batch = x.shape[0]
        x = x.to(device)
        # qz_m, qz_v, z = self.z_encoder(encoder_input)
        mu, logvar, z = encoder(x)
        # logvar = torch.log(var + 1e-6)  # logvar = log(var + eps)
        # print(z)
        dec_out = model(z)
        loss, comps = vae_loss(mu, logvar, dec_out, x[:,0:G], x[:,G:2*G], x[:,2*G:])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_val = loss.item()
        loss_history.append(loss_val)
        
        print(f"Epoch {epoch}: loss={loss_val:.4f}")


# Set the model to evaluation mode
encoder.eval()
model.eval()

with torch.no_grad():
    alpha = F.softplus(model.raw_alpha).cpu().numpy()
    beta = F.softplus(model.raw_beta).cpu().numpy()
    gamma = F.softplus(model.raw_gamma).cpu().numpy()

adata.var['fit_alpha'] = alpha
adata.var['fit_beta'] = beta
adata.var['fit_gamma'] = gamma

print(f"Extracted beta shape: {alpha.shape}")
print(f"Extracted beta shape: {beta.shape}")
print(f"Extracted gamma shape: {gamma.shape}")


full_dataset = MyDataset(adata)
all_data_tensor = full_dataset.X.to(device)

# 将所有数据传入encoder，得到z的均值
with torch.no_grad():
    mu, logvar, z = encoder(all_data_tensor)

# 接下来，传入decoder计算时间分布的参数a2和b2
# z 是我们需要的输入
with torch.no_grad():
    # 这部分代码从你的FullDecoder的forward函数中提取
    a2 = torch.cat([model.f_a2[g](z) + EPS for g in model.gene_list], dim=1)  # (N, G)
    b2 = torch.cat([model.f_b2[g](z) + EPS for g in model.gene_list], dim=1)  # (N, G)

    # Beta分布的均值是 a / (a + b)
    # 我们用均值作为latent time的估计
    t_cell_gene_specific = model.t_max * (a2 / (a2 + b2)) # (N, G)
    
    # 为了得到每个细胞的一个时间值，我们可以取所有基因时间的平均值或中位数
    # 这里我们用平均值
    latent_time = t_cell_gene_specific.mean(dim=1).cpu().numpy()
    latent_time = (latent_time -np.min(latent_time)) / (np.max(latent_time) - np.min(latent_time))


# 将latent time存储到anndata对象中
adata.obs['latent_time'] = latent_time


U = adata.layers['unspliced']
S = adata.layers['spliced']

# beta 和 gamma 是基因维度的 (G,)
# U 和 S 是细胞*基因维度的 (N, G)
# NumPy的广播机制会自动处理这里的维度匹配
velocity = beta * U - gamma * S

# 将velocity矩阵存储在adata的一个新layer中
adata.layers['velocity'] = velocity


import scanpy as sc
import scvelo as scv

# 步骤 4.1: 计算UMAP坐标
# 如果你的adata对象还没有UMAP坐标，需要先计算它们。
# 通常基于高变基因的PCA结果来计算。
print("\n计算PCA和UMAP...")
seed = 33
adata.X = adata.layers['spliced'].copy()
scv.pp.filter_and_normalize(adata, min_shared_counts=20)  
scv.pp.moments(adata, n_pcs=30, n_neighbors=30)

sc.tl.pca(adata, svd_solver='arpack', random_state=seed)
sc.pp.neighbors(adata, n_pcs=30, random_state=seed)
sc.tl.umap(adata, random_state=seed)

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
sc.pl.umap(
    adata_isoform,
    color='cell_state', 
    title='UMAP Based on Isoform Counts',
    frameon=False
)
adata.obsm['X_umap_isoform'] = adata_isoform.obsm['X_umap']
plt.rcParams['savefig.transparent'] = True
# 步骤 4.2: 可视化Latent Time
# 使用scanpy的绘图功能，将UMAP上的点按latent_time着色
print("正在生成Latent Time的UMAP图...")
sc.pl.umap(
    adata,
    color='latent_time',
    color_map='viridis', # 使用viridis色谱，通常适合表示连续的时间值
    title='Gene_based UMAP Colored by Latent Time (IsoVelo)'
)

sc.pl.scatter(
    adata,
    basis='umap_isoform',
    color='latent_time',
    color_map='viridis', # 使用viridis色谱，通常适合表示连续的时间值
    title='Isoform-based UMAP Colored by Latent Time (IsoVelo)',
    show=False
)
plt.savefig('adata_isoform_latent_time_isovelo.svg')
plt.close

scv.pp.moments(adata, n_pcs=30, n_neighbors=30) # scVelo需要计算一阶和二阶矩
scv.tl.velocity_graph(adata)
scv.pl.velocity_embedding_stream(
    adata,
    basis='umap',
    color='cell_state', # 也可以按latent_time着色
    title='Isoform-level IsoVelo Velocity on Gene-Based UMAP'
)

scv.pl.velocity_embedding_stream(
    adata,
    basis='umap_isoform',
    color='cell_state', # 也可以按latent_time着色
    title='Isoform-level IsoVelo Velocity on Isoform-Based UMAP'
)

adata.write_h5ad("downregulated_handeled.h5ad")

