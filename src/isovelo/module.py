import torch.nn as nn
import torch
import torch.nn.functional as F
from scvi.nn import Encoder, FCLayers

class IsoveloEncoder(nn.Module):
    def __init__(self,
                 n_input: int,
                 n_hidden: int,
                 n_latent: int,
                 n_layers: int,
                 dropout_rate: float = 0.1):
        super().__init__()
        self.encoder = FCLayers(
            n_in=n_input,
            n_out=n_latent,
            n_hidden=n_hidden,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            use_batch_norm=True,
            use_layer_norm=False,
        )

    def forward(self, x):
        z = self.encoder(x)
        return z
    
class IsoveloDecoder(nn.Module):
    def __init__(self,
                 n_latent: int,
                 n_hidden: int,
                 n_output: int,
                 n_layers: int,
                 dropout_rate: float = 0.1):
        super().__init__()
        self.decoder = FCLayers(
            n_in=n_latent,
            n_out=n_output,
            n_hidden=n_hidden,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            use_batch_norm=True,
            use_layer_norm=False,
        )

    def forward(self, z):
        x_reconstructed = self.decoder(z)
        return x_reconstructed
    

def reparameterize(mu: torch.Tensor, var: torch.Tensor, distribution: str) -> torch.Tensor:
    if distribution == "normal":
        std = torch.sqrt(var)
        eps = torch.randn_like(std)
        return mu + eps * std
    elif distribution == "lognormal":
        std = torch.sqrt(var)
        eps = torch.randn_like(std)
        return torch.exp(mu + eps * std)
    else:
        raise ValueError(f"Unsupported distribution: {distribution}")




