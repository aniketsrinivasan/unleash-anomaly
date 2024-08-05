import torch
import torch.nn as nn


class Encoder(nn.Sequential):
    __in_features = 400
    __z_features = 50

    def __init__(self, in_features: int = None, z_features: int = None):
        # Defining features:
        self.in_features = in_features if (in_features is not None) else self.__in_features
        self.z_features = z_features if (z_features is not None) else self.__z_features
        # Creating Sequential Dense layers:
        #   shape of encoded (output) Tensor: (batch_size, z_features)
        super().__init__(
            #   (batch_size, in_features) => (batch_size, 200)
            nn.Linear(in_features=self.in_features, out_features=200),
            #   (batch_size, 200) => (batch_size, 200)
            nn.LeakyReLU(negative_slope=0.1),
            #   (batch_size, 200) => (batch_size, z_features)
            nn.Linear(in_features=200, out_features=self.z_features),
            #   (batch_size, z_features) => (batch_size, z_features)
            nn.LeakyReLU(negative_slope=0.1)
        )

    # Forward method:
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x:    (batch_size, in_features)
        # Pass through all layers in Sequential:
        for module in self:
            x = module(x)
        # x:    (batch_size, z_features)
        return x
