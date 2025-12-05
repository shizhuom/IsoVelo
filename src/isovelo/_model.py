import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scvelo as scv
from scvi.nn import Encoder, FCLayers
from torch.distributions import Normal
from scvi.distributions import NegativeBinomial
import anndata

# Encoder: use scvi.nn.Encoder, it is specifically designed for single-cell data
class IsoveloEncoder(Encoder):
    """
    Encodes U (unspliced) and Isoform counts into a latent cell embedding.
    Inherits from scvi.nn.Encoder to leverage its VAE structure.
    """
    def __init__(self, 
                 input_dim:int, 
                 hidden_dim=32, 
                 output_dim = 128, 
                 n_layers=2, 
                 dropout_rate=0.1, 
                 distribution="normal", 
                 use_batch_norm=True, 
                 use_layer_norm=False,
                 var_activation=F.Softplus(),
                 activation_fn=F.ReLU,
                 **kwargs):
        super().__init__(n_input=input_dim, 
                         n_output=output_dim,
                         n_layers=n_layers,
                         n_hidden=hidden_dim,
                         dropout_rate=dropout_rate,
                         distribution=distribution,
                         use_batch_norm=use_batch_norm,
                         use_layer_norm=use_layer_norm,
                         var_activation=var_activation,
                         activation_fn=activation_fn,
                         **kwargs
                         )

    def forward(self, x: torch.Tensor, *cat_list: torch.Tensor):
        """
        Forward pass.
        :param x: Concatenated tensor of [U, Isoforms]
        :param library_size: Tensor of cell library sizes
        :return: A dictionary with 'qz_m', 'qz_v', 'ql_m', 'ql_v', 'z', and 'library_size_z'
        """
        # Encode x to get latent parameters
        qz_m, qz_v, z = super().forward(x, *cat_list)
        
        return {"qz_m": qz_m, "qz_v": qz_v, "z": z}


class IsoveloDecoder(nn.Module):
    """
    Decodes latent cell embedding 'z' into kinetic parameters (alpha, beta, gamma, t_c)
    and computes the expected U and Isoform counts using the numerical solution to the RNA velocity ODEs.
    """

    def __init__(
        self,
        n_input_z: int,
        n_genes: int,
        n_isoforms: int,
        gene_to_isoform_map: list[list[int]],
        n_hidden: int = 128,
        n_layers: int = 2
    ):
        super().__init__()
        """
        :param n_input_z: Dimensionality of the latent space (from Encoder)
        :param n_genes: Number of genes (for U counts and alpha/t0 parameters)
        :param n_isoforms: Total number of isoforms (for S_iso counts and beta/gamma parameters)
        :param gene_to_isoform_map: A list of lists, where gene_to_isoform_map[g_idx] is a list of isoform indices [i_idx_1, i_idx_2, ...] belonging to gene g_idx.
        :param n_hidden: Number of hidden units in the NN
        :param n_layers: Number of layers in the NN
        """

        self.n_genes = n_genes
        self.n_isoforms = n_isoforms
        self.gene_to_isoform_map = gene_to_isoform_map

        # Common NN body to process 'z'
        self.param_decoder = FCLayers(
            n_in=n_input_z,
            n_out=n_hidden,
            n_layers=n_layers,
            n_hidden=n_hidden,
            use_batch_norm=True
        )

        # alpha: Transcription rate (one per gene)
        self.alpha_head = nn.Linear(n_hidden, n_genes)
        nn.init.normal_(self.alpha_head.bias, mean=-5, std=0.1)

        # beta: Splicing rate (one per isoform)
        self.beta_head = nn.Linear(n_hidden, n_isoforms)
        nn.init.normal_(self.beta_head.bias, mean=-5, std=0.1)

        # gamma: Degradation rate (one per isoform)
        self.gamma_head = nn.Linear(n_hidden, n_isoforms)
        nn.init.normal_(self.gamma_head.bias, mean=-3, std=0.1)

        # t_c: Cell-specific transcription time constant (one per cell)
        self.t_c_head = nn.Linear(n_hidden, n_cells)
        nn.init.normal_(self.t_c_head.bias, mean=0, std=0.1)











