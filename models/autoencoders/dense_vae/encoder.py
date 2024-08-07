import torch
import torch.nn as nn
from ..blocks import AttentionBlock, ResidualBlock


class Encoder(nn.Sequential):
    __in_features = 400
    __z_features = 50

    def __init__(self, in_features: int = None, z_features: int = None, dropout: float = 0):
        """
        Encoder for the DenseVAE.

        :param in_features:     number of input features.
        :param z_features:      number of hidden features.
        """
        # Defining features:
        self.in_features = in_features if (in_features is not None) else self.__in_features
        self.z_features = z_features if (z_features is not None) else self.__z_features
        self.dropout = dropout
        # Creating Sequential Dense layers:
        #   shape of encoded (output) Tensor: (batch_size, 1, z_features)
        super().__init__(
            nn.Dropout(dropout),
            #   (batch_size, 1, in_features) => (batch_size, 1, 256)
            nn.Linear(in_features=self.in_features, out_features=256),
            #   (batch_size, 1, 256) => (batch_size, 1, 256)
            nn.LeakyReLU(negative_slope=0.1),

            nn.Dropout(dropout),
            #   (batch_size, 1, 256) => (batch_size, 1, 256)
            nn.Linear(in_features=256, out_features=256),
            #   (batch_size, 1, 256) => (batch_size, 1, 256)
            nn.LeakyReLU(negative_slope=0.1),

            nn.Dropout(dropout),
            #   (batch_size, 1, 256) => (batch_size, 1, 256)
            nn.Linear(in_features=256, out_features=256),
            #   (batch_size, 1, 256) => (batch_size, 1, 256)
            nn.LeakyReLU(negative_slope=0.1),

            nn.Dropout(dropout),
            #   (batch_size, 1, 256) => (batch_size, 1, z_features)
            nn.Linear(in_features=256, out_features=self.z_features),
            #   (batch_size, 1, z_features) => (batch_size, 1, z_features)
            nn.LeakyReLU(negative_slope=0.1)
        )

    # Forward method:
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x:    (batch_size, 1, in_features)
        x = x.to(dtype=torch.float32)
        # Pass through all layers in Sequential:
        for module in self:
            x = module(x)
        # x:    (batch_size, 1, z_features)
        return x
