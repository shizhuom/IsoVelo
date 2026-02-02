import torch.nn as nn
import torch
import torch.nn.functional as F
from module import reparameterize, IsoveloDecoder, IsoveloEncoder

class IsoveloVAE(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 32,
                 latent_dim: int = 128,
                 n_layers: int = 2,
                 dropout_rate: float = 0.1,
                 distribution: str = "normal"):
        super().__init__()
        self.encoder = IsoveloEncoder(
            n_input=input_dim,
            n_hidden=hidden_dim,
            n_latent=latent_dim,
            n_layers=n_layers,
            dropout_rate=dropout_rate
        )
        self.decoder = IsoveloDecoder(
            n_latent=latent_dim,
            n_hidden=hidden_dim,
            n_output=input_dim,
            n_layers=n_layers,
            dropout_rate=dropout_rate
        )
        self.distribution = distribution

    def forward(self, x):
        # Encode
        qz_m, qz_v, z = self.encoder(x)
        
        # Reparameterize
        z_sampled = reparameterize(qz_m, qz_v, self.distribution)
        
        # Decode
        x_reconstructed = self.decoder(z_sampled)
        
        return {
            "qz_m": qz_m,
            "qz_v": qz_v,
            "z": z_sampled,
            "x_reconstructed": x_reconstructed
        }
    
    @staticmethod
    def loss_function(x, x_reconstructed, qz_m, qz_v, distribution="normal"):
        # Reconstruction loss
        recon_loss = F.mse_loss(x_reconstructed, x, reduction='sum')
        
        # KL divergence
        if distribution == "normal":
            kl_div = -0.5 * torch.sum(1 + torch.log(qz_v) - qz_m.pow(2) - qz_v)
        elif distribution == "lognormal":
            kl_div = torch.sum(qz_m + 0.5 * qz_v - 1)
        else:
            raise ValueError(f"Unsupported distribution: {distribution}")
        
        return recon_loss + kl_div