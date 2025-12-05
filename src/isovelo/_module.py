from typing import Callable, Iterable, Literal, Optional
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn as nn
from scvi.nn import Encoder, FCLayers
from scvi.module.base import BaseModuleClass, LossOutput, auto_move_data

from ._constants import REGISTRY_KEYS

class DecoderIsoVelo(nn.Module):
    def __init__(self,
                 n_genes: int,
                 n_isoforms: int,
                 n_input: int,
                 n_hidden: int,
                 n_layers: int,
                 dropout_rate: float = 0.1,
                 s **kwargs):
        super().__init__()

        self.alpha_g = 
        self.beta_g = 
        self.gamma_g = 

        










    pass

class IsoVAE(BaseModuleClass):
    def __init__(self, 
                 n_genes: int,
                 n_isoforms: int,
                 n_input: int,
                 n_hidden: int = 128,
                 n_latent: int = 20,
                 n_layers: int = 1,
                 dropout_rate: float = 0.1,
                 latent_distribution: str = "normal",
                 use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "both",
                 use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "both",
                 var_activation: Optional[Callable] = torch.nn.Softplus(),
                 **kwargs):
        super().__init__()
        self.n_latent = n_latent
        self.log_variational = log_variational
        self.latent_distribution = latent_distribution
        self.use_observed_lib_size = use_observed_lib_size
        
        if n_input is not None:
            self.n_input = n_input
        else:
            self.n_input = 2 * n_genes + n_isoforms

        use_batch_norm_encoder = use_batch_norm == "encoder" or use_batch_norm == "both"
        use_batch_norm_decoder = use_batch_norm == "decoder" or use_batch_norm == "both"
        use_layer_norm_encoder = use_layer_norm == "encoder" or use_layer_norm == "both"
        use_layer_norm_decoder = use_layer_norm == "decoder" or use_layer_norm == "both"
        self.use_batch_norm_decoder = use_batch_norm_decoder

        n_input_encoder = 2 * n_genes + n_isoforms
        n_input_decoder = n_latent
        


        self.encoder_z = Encoder(
            n_input=n_input_encoder,
            n_output=n_latent,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            distribution=latent_distribution,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
            var_activation=var_activation,
            activation_fn=torch.nn.ReLU,
        )

        self.decoder = DecoderIsoVelo(
            #TODO
        )

    def loss(self, *args, **kwargs):
        return super().loss(*args, **kwargs)


    pass




