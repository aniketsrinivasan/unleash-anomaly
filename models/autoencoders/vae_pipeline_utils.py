import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable
from conv_vae import ConvVAE
from utils import DatasetTensor, log_info


def __check_vae_model_validity(model):
    supported_types = [ConvVAE, ]
    if type(model) not in supported_types:
        raise ValueError(f"Model is of unsupported type. Currently supported VAEs are listed below. \n"
                         f"  {supported_types}")


def __KL_loss_binary(reconstructed_x: torch.Tensor, x: torch.Tensor,
                     mean: torch.Tensor, logvar: torch.Tensor):
    reconstruction_loss = F.binary_cross_entropy(input=reconstructed_x, target=x, reduction="sum")
    KL_divergence = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    return reconstruction_loss + KL_divergence


def __KL_loss_mse(reconstructed_x: torch.Tensor, x: torch.Tensor,
                  mean: torch.Tensor, logvar: torch.Tensor):
    reconstruction_loss = F.mse_loss(input=reconstructed_x, target=x, reduction="sum")
    KL_divergence = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    return reconstruction_loss + KL_divergence


@log_info(log_path="logs/log_model_training", log_enabled=True)
def vae_train(model, data_loader: torch.DataLoader, optimizer=None, loss_function=None,
              epochs=50, device="cpu", verbose=True):
    if verbose:
        print(f"Training model {type(model)} on provided dataset.")
    # Check validity of the VAE provided:
    __check_vae_model_validity(model)

    # Set default values if not provided:
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    if loss_function is None:
        loss_function = __KL_loss_mse

    # Run training:
    model.to(device)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for data in data_loader:
            data = data["data"].to(device)
            optimizer.zero_grad()
            reconstruction, mean, logvar = model(data)
            this_loss = loss_function(reconstructed_x=reconstruction, x=data,
                                      mean=mean, logvar=logvar)
            this_loss.backward()
            optimizer.step()
            total_loss += this_loss.item()
        if verbose:
            print(f"    Epoch {epoch} with loss: {total_loss / len(data_loader.dataset)}.")
