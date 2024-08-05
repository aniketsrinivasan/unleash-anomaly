import torch
from torch.optim import Adam
import matplotlib.pyplot as plt
from utils import *
from torch.utils.data import DataLoader
from torch.nn import functional as F
from models import ConvVAE
from models.autoencoders.vae_pipeline_utils import *


def __KL_loss_mse(reconstructed_x: torch.Tensor, x: torch.Tensor,
                  mean: torch.Tensor, logvar: torch.Tensor):
    reconstruction_loss = F.mse_loss(input=reconstructed_x, target=x, reduction="sum")
    KL_divergence = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    return reconstruction_loss, KL_divergence


# data_tensor = DataTensor(sqlite_path="0.topi",
#                          skeleton_dict_path="skeleton_dict_1",
#                          table_name="TOPS_80",
#                          key_column="Key",
#                          value_column="StatVal",
#                          verbose=True)
# tensor = data_tensor.get_tensor(this_timestep=1720894800,
#                                 interval_size=300,
#                                 num_intervals=3,
#                                 use_other="Other",
#                                 ignore_entries=("SYS:GROUP_TOTALS",),
#                                 image_shape=(20, 20))
def torch_transform(tensor):
    tensor = tensor.clamp(min=1)
    tensor = torch.log(tensor)
    return tensor



kwargs_get_tensor = dict(
    use_other="Other",
    ignore_entries=("SYS:GROUP_TOTALS", ),
    image_shape=(20, 20),
    transform=torch_transform
)
dataset = DatasetTensor(sqlite_path="0.topi",
                        skeleton_dict_path="skeleton_dict_1",
                        table_name="TOPS_80",
                        key_column="Key",
                        value_column="StatVal",
                        start_timestamp=1720809300,
                        end_timestamp=1720894800,
                        interval_size=300,
                        num_intervals=2,
                        kwargs_get_tensor=kwargs_get_tensor)

dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

VAE = ConvVAE(in_channels=2,
              z_channels=16,
              latent_dim=8,
              image_shape=(20, 20))

# optimizer = Adam(VAE.parameters(), lr=0.0005)
# vae_train(model=VAE, data_loader=dataloader, optimizer=optimizer,
#           loss_function=None, epochs=200, device="cpu", verbose=True,
#           save_path="stubs/sample_conv_vae.pt")

VAE.load_state_dict(torch.load("stubs/sample_conv_vae.pt"))

mean_real_loss = 0
mean_anom_loss = 0
anoms = []
for i in range(0, len(dataset)):
    this_sample = dataset[i]["data"]
    this_sample = torch.unsqueeze(this_sample, 0)
    out, mean, logvar = VAE(this_sample)
    loss = __KL_loss_mse(reconstructed_x=out, x=this_sample, mean=mean, logvar=logvar)
    print(i, loss)
    mean_real_loss += loss[0]
    if loss[0] >= 900:
        anoms.append((i, loss))

    # this_sample[0][0][19][19] += 9.2
    # this_sample[0][0][12][17] += 3.3
    # this_sample[0][0][11][13] += 4.7
    # this_sample[0][0][9][6] += 2.1
    # this_sample[0][0][4][19] += 4.3
    # this_sample[0][0][2][12] += 1.9
    # this_sample[0][0][16][15] += 3.0
    # out, mean, logvar = VAE(this_sample)
    # loss = __KL_loss_mse(reconstructed_x=out, x=this_sample, mean=mean, logvar=logvar)
    # print(loss)
    # mean_anom_loss += loss[0]

    # plt.imshow(this_sample[0][0], vmax=20, vmin=0)
    # plt.show()
    #
    # out = out.detach().numpy()
    # plt.imshow(out[0][0], vmax=20, vmin=0)
    # plt.show()

print(mean_real_loss)
print(mean_anom_loss)
for anom in anoms:
    print(anom)

# for i in range(146, 154):
#     this_sample = dataset[i]["data"]
#     this_sample = torch.unsqueeze(this_sample, 0)
#     out, mean, logvar = VAE(this_sample)
#     loss = __KL_loss_mse(reconstructed_x=out, x=this_sample, mean=mean, logvar=logvar)
#     print(i, loss)
#
#     out = out.detach().numpy()
#     fig, axs = plt.subplots(2, 2)
#     axs[0, 0].imshow(this_sample[0][0], vmax=20, vmin=0)
#     axs[0, 1].imshow(this_sample[0][1], vmax=20, vmin=0)
#     axs[1, 0].imshow(out[0][0], vmax=20, vmin=0)
#     axs[1, 1].imshow(out[0][1], vmax=20, vmin=0)
#     plt.show()

# 214 is an interesting timestep
for i in range(129, 133):
    this_sample = dataset[i]["data"]
    this_sample = torch.unsqueeze(this_sample, 0)
    out, mean, logvar = VAE(this_sample)
    loss = __KL_loss_mse(reconstructed_x=out, x=this_sample, mean=mean, logvar=logvar)
    print(i, loss)

    out = out.detach().numpy()
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].imshow(this_sample[0][0], vmax=20, vmin=0)
    axs[0, 1].imshow(this_sample[0][1], vmax=20, vmin=0)
    axs[1, 0].imshow(out[0][0], vmax=20, vmin=0)
    axs[1, 1].imshow(out[0][1], vmax=20, vmin=0)
    plt.show()
