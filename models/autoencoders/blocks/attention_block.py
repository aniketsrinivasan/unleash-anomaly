import torch
from torch import nn
from .attention import SelfAttention


class AttentionBlock(nn.Module):
    """
    The Attention Block utilized in the Autoencoder. Used to create relationships between
    pixels of an image that are arbitrarily far using self-attention.
    """
    __num_groups = 8

    def __init__(self, channels: int, num_groups: int = None):
        """
        Performs self-Attention on an image (Tensor) of shape (batch_size, channels, height, width)
        to create relationships between the pixels of an image that are arbitrarily far.

        :param channels:    number of channels in the image.
        :param num_groups:  number of groups in the group normalization (must divide channels).
        """
        super().__init__()
        # Ensuring that the distribution stays somewhat the same (for stability):
        #   group normalization occurs in groups rather than across the entire layer, so each
        #   group has its own mean and variance.
        #   the idea is that we want to normalize, but maintain neighbourhood information.
        self.num_groups = num_groups if (num_groups is not None) else self.__num_groups
        self.groupnorm = nn.GroupNorm(num_groups=self.num_groups, num_channels=channels)
        self.attention = SelfAttention(n_heads=1, d_embed=channels)

    # Forward method:
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs self-Attention.

        :param x:   image (torch.Tensor) of shape (batch_size, channels, height, width).
        :return:    torch.Tensor of same shape as input.
        """
        # x:    (batch_size, channels, height, width) where (height, width) are not fixed.
        # Residue for residual connection:
        residue = x

        is_4d = False
        batch_size, channels, height, width = None, None, None, None
        if len(x.shape) == 4:
            is_4d = True
            # Extracting shape of the input:
            batch_size, channels, height, width = x.shape
            # We perform Self-Attention between all the pixels of the image:
            #   "flattening" the dimensions of the image
            #   (batch_size, channels, height, width) ==> (batch_size, channels, height*width)
            x = x.view(batch_size, channels, height * width)

        #   (batch_size, channels, height*width) ==> (batch_size, height*width, channels)
        #       each pixel is in dim=1 (height*width) and has an 'embedding' given by dim=2 (channels).
        #       this is the same idea as used in the Transformer model.
        x = x.transpose(dim0=2, dim1=1)

        # Self-Attention:
        x = self.attention(x)  # does not change shape of Tensor

        # Transposing back:
        #   (batch_size, height*width, channels) ==> (batch_size, channels, height*width)
        x = x.transpose(dim0=2, dim1=1)

        if is_4d:
            # Reforming the Tensor into original shape:
            #   (batch_size, channels, height*width) ==> (batch_size, channels, height, width)
            x = x.view((batch_size, channels, height, width))

        # Residual connection:
        x += residue

        return x
