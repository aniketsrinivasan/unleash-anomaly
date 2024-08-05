import torch
from torch import nn
from torch.nn import functional as F
from ..blocks import AttentionBlock, ResidualBlock
from utils import log_info


class Encoder(nn.Sequential):
    __in_channels = 2
    __z_channels = 4

    def __init__(self, in_channels: int = None, z_channels: int = None):
        # Defining channels:
        self.in_channels = in_channels if (in_channels is not None) else self.__in_channels
        self.z_channels = z_channels if (z_channels is not None) else self.__z_channels
        # Creating parent Sequential model to reverse Encoder process:
        #   shape of encoded (output) image: (batch_size, 4, height//4, width//4)
        super().__init__(
            # Convolutional layer:
            #   (batch_size, in_channels, height, width) => (batch_size, 8, height, width)
            nn.Conv2d(in_channels=self.in_channels, out_channels=8, kernel_size=3, padding=1),

            # Residual block:
            #   a combination of convolutions and normalization
            #   (batch_size, 8, height, width) => (batch_size, 8, height, width)
            ResidualBlock(in_channels=8, out_channels=8),

            # Convolutional layer:
            #   (batch_size, 8, height, width) => (batch_size, 16, height//2, width//2)
            #   here, we want (ideally) that our image has odd height and width (to avoid border ignorance)
            #   however, we fix this issue in the self.forward() method using manual padding otherwise
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=2, padding=0),

            # Residual block:
            #   (batch_size, 8, height//2, width//2) => (batch_size, 16, height//2, width//2)
            ResidualBlock(in_channels=8, out_channels=16),

            # Residual block:
            #   (batch_size, 16, height//2, width//2) => (batch_size, 16, height//2, width//2)
            ResidualBlock(in_channels=16, out_channels=16),

            # Convolutional layer:
            #   (batch_size, 16, height//2, width//2) => (batch_size, 16, height//4, width//4)
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=2, padding=0),

            # Residual block(s) without changing shape:
            ResidualBlock(in_channels=16, out_channels=16),
            ResidualBlock(in_channels=16, out_channels=16),

            # Attention block:
            #   runs self-attention over each pixel.
            #   attention relates tokens to each other in a sentence, so we can think of (here) as a
            #       way to relate pixels to one another.
            #   convolutions relate local neighbourhoods of pixels, but attention can propagate
            #       throughout the image to relate far away pixels.
            #   (batch_size, 16, height//4, width//4) => (batch_size, 16, height//4, width//4)
            AttentionBlock(channels=16),

            # Group normalizations (doesn't change size):
            nn.GroupNorm(num_groups=8, num_channels=16),

            # Activation function SiLU:
            nn.SiLU(),

            # Convolution layer:
            #   this is the "bottleneck" of the encoder
            #   (batch_size, 16, height//4, width//4) => (batch_size, 4, height//4, width//4)
            nn.Conv2d(in_channels=16, out_channels=self.z_channels, kernel_size=3, padding=1),

            # Convolution layer:
            #   (batch_size, 4, height//4, width//4) => (batch_size, 4, height//4, width//4)
            nn.Conv2d(in_channels=self.z_channels, out_channels=self.z_channels,
                      kernel_size=1, padding=0)
        )

    # Forward method:
    @log_info(display_args=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward-pass through the Variational Autoencoder. Being a Variational Autoencoder, it learns how to
        represent images in a latent space, which is modeled by a multi-variate Gaussian distribution (this
        is the functional form learnt by the neural network).

        Shape of output: (batch_size, 4, height//4, width//4)

        :param x:       input image (as a torch.Tensor).
        :param noise:   noise (added to the output, so shape must match output).
        :return:        encoded image (as a torch.Tensor).
        """
        # x:        (batch_size, in_channels, height, width)
        # Conversion to float32 data type:
        x = x.to(torch.float32)

        # Forward pass through Sequential layers:
        for module in self:
            # If the stride attribute is (2, 2) and padding isn't applied, we will manually apply a
            #   symmetrical padding:
            if getattr(module, "stride", None) == (2, 2):
                x = F.pad(x, (0, 1, 0, 1))  # add padding to bottom and right (avoids border ignorance)
            # Passing through Sequential:
            x = module(x)

        return x
