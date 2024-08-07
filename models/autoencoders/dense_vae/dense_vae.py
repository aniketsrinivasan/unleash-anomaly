import torch
import torch.nn as nn
from .encoder import Encoder
from .decoder import Decoder


class DenseVAE(nn.Module):
    __in_features = 400
    __z_features = 50
    __latent_dim = 10
    __dropout = float(0)

    def __init__(self, in_features: int = None, z_features: int = None, latent_dim: int = None,
                 dropout: float = None):
        super().__init__()
        # Defining channels:
        self.in_features = in_features if in_features is not None else self.__in_channels
        self.z_features = z_features if z_features is not None else self.__z_features
        self.latent_dim = latent_dim if latent_dim is not None else self.__latent_dim
        self.dropout = dropout if dropout is not None else self.__dropout

        # Defining encoder:
        self.encoder = Encoder(in_features=self.in_features, z_features=self.z_features)
        # Latent mean and (log) variance layers:
        self.layer_mean = nn.Linear(in_features=self.z_features,
                                    out_features=self.latent_dim)
        self.layer_logvar = nn.Linear(in_features=self.z_features,
                                      out_features=self.latent_dim)
        self.layer_reconstruct = nn.Linear(in_features=self.latent_dim,
                                           out_features=self.z_features)
        # Defining decoder:
        self.decoder = Decoder(z_features=self.z_features, out_features=self.in_features)

    @staticmethod
    def reparameterize(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # Compute stddev and reparameterize accordingly:
        #   var = stddev^2 ==> log(var) = log(stddev^2) = 2*log(stddev)
        #                  ==> stddev = exp(0.5*log(var))
        stddev = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(stddev)
        sample = mean + epsilon * stddev
        return sample

    def forward(self, x: torch.Tensor) -> tuple:
        # Forward pass through encoder into latent space "z":
        #   (batch_size, 1, in_features) => (batch_size, 1, z_features)
        x = self.encoder(x)

        # Getting mean and logvar by passing through corresponding dense layers:
        #   mean, logvar have the following shape:
        #       (batch_size, 1, latent_dim)
        mean, logvar = self.layer_mean(x), self.layer_logvar(x)
        logvar = torch.clamp(logvar, min=-30, max=20)   # clamped for stability

        # Reparameterization (getting latent representation):
        #   z:  (batch_size, 4, latent_dim)
        z = self.reparameterize(mean, logvar)

        # Passing through decoder back into output space:
        #   (batch_size, 1, latent_dim) => (batch_size, 1, z_features)
        z = self.layer_reconstruct(z)
        #   (batch_size, 1, z_features) => (batch_size, 1, in_features)
        reconstructed_x = self.decoder(z)

        return reconstructed_x, mean, logvar
