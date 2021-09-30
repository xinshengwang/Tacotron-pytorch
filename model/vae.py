import torch
from torch import nn
from utils.config import cfg
from model.layers import ReferenceEncoder

class VAE(nn.Module):
    """
    vae for reference embedding

    VAE for speech synthesis is described in:
        Y. Zhang, S. Pan, L. He, Z, Ling,
        "Learning latent representations for style control and transfer in end-to-end speech synthesis,"
        in IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP).
        https://arxiv.org/abs/1812.04342

    """

    def __init__(self, mel_dim, gru_units=cfg.vae_emb_dim,
                 conv_channels=[32, 32, 64, 64, 128, 128], kernel_size=3, stride=2, padding=1):
        super().__init__()

        self.ref_encoder = ReferenceEncoder(mel_dim, gru_units, conv_channels, kernel_size, stride, padding)   
        self.mean_linear = nn.Linear(cfg.vae_emb_dim, cfg.vae_latent_dim)
        self.logvar_linear = nn.Linear(cfg.vae_emb_dim,cfg.vae_latent_dim)
    
    def encoder(self, inputs):
        hidden = self.relu(self.FC_h(inputs))
        return self.FC1(hidden), self.FC2(hidden)

    def forward(self, inputs, input_lengths=None):
        """
        input:
            inputs --- [B, T, mel_dim]
            input_lengths --- [B]
        output:
            latent_embed --- [B, 1, vae_latent_dim]
        """
        ref_emb = self.ref_encoder(inputs, input_lengths=input_lengths)  # [B, gru_units]
        mean = self.mean_linear(ref_emb)
        logvar = self.logvar_linear(ref_emb)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        # z = eps.mul(std).add_(latent_mean)
        z = eps * std + mean
        z = z.unsqueeze(1)
        return (z, (mean, logvar))
