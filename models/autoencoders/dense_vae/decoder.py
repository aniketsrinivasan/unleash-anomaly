import torch
import torch.nn as nn


class Decoder(nn.Sequential):
    __z_features = 50
    __out_features = 400

    def __init__(self, z_features: int, out_features: int):
        # Defining features:
        self.out_features = out_features if (out_features is not None) else self.__out_features
        self.z_features = z_features if (z_features is not None) else self.__z_features
        super().__init__(
            #   (batch_size, z_features) => (batch_size, 200)
            nn.Linear(in_features=self.z_features, out_features=200),
            #   (batch_size, 200) => (batch_size, 200)
            nn.LeakyReLU(negative_slope=0.1),
            #   (batch_size, 200) => (batch_size, out_features)
            nn.Linear(in_features=200, out_features=self.out_features),
            #   (batch_size, out_features) => (batch_size, out_features)
            nn.LeakyReLU(negative_slope=0.1)
        )

    # Forward method:
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x:    (batch_size, z_features)
        for module in self:
            x = module(x)
        # x:    (batch_size, out_features)
        return x