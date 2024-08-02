import torch
import torch.nn as nn
from .encoder import Encoder
from .decoder import Decoder


class ConvVAE(nn.Module):
    __in_channels = 2
    __z_channels = 4
    __latent_dim = 10
    __image_shape = (20, 20)

    def __init__(self, in_channels: int = None, z_channels: int = None, latent_dim: int = None,
                 image_shape: tuple = None):
        super().__init__()
        # Defining channels:
        self.in_channels = in_channels if in_channels is not None else self.__in_channels
        self.z_channels = z_channels if z_channels is not None else self.__z_channels
        self.latent_dim = latent_dim if latent_dim is not None else self.__latent_dim
        self.image_shape = image_shape if image_shape is not None else self.__image_shape
        self.encoded_image_dim = (self.image_shape[0] // 4) * (self.image_shape[1] // 4)

        # Defining encoder:
        self.encoder = Encoder(in_channels=self.in_channels, z_channels=self.z_channels)
        # Latent mean and (log) variance layers:
        self.layer_mean = nn.Linear(in_features=self.encoded_image_dim,
                                    out_features=self.latent_dim)
        self.layer_logvar = nn.Linear(in_features=self.encoded_image_dim,
                                      out_features=self.latent_dim)
        self.layer_reconstruct = nn.Linear(in_features=self.latent_dim,
                                           out_features=self.encoded_image_dim)
        # Defining decoder:
        self.decoder = Decoder(z_channels=self.z_channels, out_channels=self.in_channels)

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
        #   (batch_size, in_channels, height, width) => (batch_size, 4, height//4, width//4)
        x = self.encoder(x)

        # Reshaping to pass through dense layers:
        batch_size, z_channels, height, width = x.shape
        #   (batch_size, 4, height//4, width//4) => (batch_size, 4, height//4 * width//4)
        x = x.reshape((batch_size, z_channels, height*width))

        # Getting mean and logvar by passing through corresponding dense layers:
        #   mean, logvar have the following shape:
        #       (batch_size, 4, latent_dim)
        mean, logvar = self.layer_mean(x), self.layer_logvar(x)
        logvar = torch.clamp(logvar, min=-30, max=20)   # clamped for stability

        # Reparameterization (getting latent representation):
        #   z:  (batch_size, 4, latent_dim)
        z = self.reparameterize(mean, logvar)

        # Passing through decoder back into output space:
        #   (batch_size, 4, latent_dim) => (batch_size, 4, height//4 * width//4)
        z = self.layer_reconstruct(z)
        #   (batch_size, 4, height//4 * width//4) => (batch_size, 4, height//4, width//4)
        z = z.reshape((batch_size, z_channels, height, width))
        #   (batch_size, 4, height//4, width//4) => (batch_size, out_channels, height, width)
        reconstructed_x = self.decoder(z)

        return reconstructed_x, mean, logvar
