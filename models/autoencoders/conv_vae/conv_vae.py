import torch
import torch.nn as nn
from .encoder import Encoder
from .decoder import Decoder


class ConvVAE(nn.Module):
    __in_channels = 2
    __z_channels = 4
    __latent_dim = 10

    def __init__(self, in_channels: int = None, z_channels: int = None, latent_dim: int = None):
        super().__init__()
        # Defining channels:
        self.in_channels = in_channels if in_channels is not None else self.__in_channels
        self.z_channels = z_channels if z_channels is not None else self.__z_channels
        self.latent_dim = latent_dim if latent_dim is not None else self.__latent_dim

        # Defining encoder:
        self.encoder = Encoder(in_channels=self.in_channels, z_channels=self.z_channels)
        # Latent mean and (log) variance layers:
        self.layer_mean = nn.Linear(in_features=z_channels, out_features=self.latent_dim)
        self.layer_logvar = nn.Linear(in_features=z_channels, out_features=self.latent_dim)
        # Defining decoder:
        self.decoder = Decoder(z_channels=self.z_channels, out_channels=self.in_channels)

    @staticmethod
    def reparameterize(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # Compute stddev and reparametrize accordingly:
        #   var = stddev^2 ==> log(var) = log(stddev^2) = 2*log(stddev)
        #                  ==> stddev = exp(0.5*log(var))
        stddev = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(stddev)
        sample = mean + epsilon * stddev
        return sample

    def forward(self, x: torch.Tensor) -> tuple:
        # Forward pass through encoder into latent space "z":
        x = self.encoder(x)
        # Getting mean and logvar by passing through corresponding dense layers:
        mean, logvar = self.layer_mean(x), self.layer_logvar(x)
        # Reparametrization (getting latent representation):
        z = self.reparameterize(mean, logvar)
        # Passing through decoder back into output space:
        reconstructed_x = self.decoder(z)
        return reconstructed_x, mean, logvar
