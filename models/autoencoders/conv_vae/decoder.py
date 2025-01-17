import torch
from torch import nn
from ..blocks import ResidualBlock, AttentionBlock
from utils import log_info


class Decoder(nn.Sequential):
    __z_channels = 4
    __out_channels = 2

    def __init__(self, z_channels: int = None, out_channels: int = None):
        # Defining channels:
        self.z_channels = z_channels if (z_channels is not None) else self.__z_channels
        self.out_channels = out_channels if (out_channels is not None) else self.__out_channels
        # Creating parent Sequential model to reverse Encoder process:
        #   shape of encoded (input) image: (batch_size, 4, height//4, width//4)
        super().__init__(
            #   (batch_size, 4, height//4, width//4) => (batch_size, 4, height//4, width//4)
            nn.Conv2d(in_channels=self.z_channels, out_channels=self.z_channels, kernel_size=1, padding=0),

            #   (batch_size, 4, height//4, width//4) => (batch_size, 16, height//4, width//4)
            nn.Conv2d(in_channels=self.z_channels, out_channels=16, kernel_size=3, padding=1),

            #   (batch_size, 16, height//4, width//4) => (batch_size, 16, height//4, width//4)
            ResidualBlock(in_channels=16, out_channels=16),

            #   (batch_size, 16, height//4, width//4) => (batch_size, 16, height//4, width//4)
            AttentionBlock(channels=16),

            #   (batch_size, 16, height//4, width//4) => (batch_size, 16, height//4, width//4)
            ResidualBlock(in_channels=16, out_channels=16),
            ResidualBlock(in_channels=16, out_channels=16),

            # Upsampling:
            #   (batch_size, 16, height//4, width//4) => (batch_size, 16, height//2, width//2)
            nn.Upsample(scale_factor=2),

            #   (batch_size, 16, height//2, width//2) => (batch_size, 16, height//2, width//2)
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),

            #   (batch_size, 16, height//2, width//2) => (batch_size, 8, height//2, width//2)
            ResidualBlock(in_channels=16, out_channels=8),
            ResidualBlock(in_channels=8, out_channels=8),

            # Upsampling:
            #   (batch_size, 8, height//2, width//2) => (batch_size, 8, height, width)
            nn.Upsample(scale_factor=2),

            #   (batch_size, 8, height, width) => (batch_size, 8, height, width)
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding=1),

            # Group normalization:
            #   (batch_size, 8, height, width) => (batch_size, 8, height, width)
            nn.GroupNorm(num_groups=8, num_channels=8),

            # SiLU activation function:
            nn.SiLU(),

            # Final convolution, changing to out_channels:
            nn.Conv2d(in_channels=8, out_channels=self.out_channels, kernel_size=3, padding=1)
        )

    # Forward method:
    @log_info(display_args=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x:    (batch_size, 4, height//4, width//4)
        # We want to undo everything done in the encoder, and pass through Sequential layers.
        x = x.to(torch.float32)

        # Passing through decoder:
        for module in self:
            x = module(x)

        # x:    (batch_size, out_channels, height, width)
        return x
