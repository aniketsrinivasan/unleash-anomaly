from utils import *
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.stats


# Idea:
#   take top 499 and create a group for "other"
#   then run this through various algorithms. possibly autoencoder, multivar gaussian.
#   can create various "groups" e.g. top 19, top 99, top 499 etc. and see performance
#   this top 499 should be taken from at least 2 weeks of past data.
#   an algorithm to update the list of toppers and re-train or re-compute may be necessary if this
#       top-k changes frequently.

# for i in range(50, 51):
#     img_struct = ImageStruct(sqlite_path="0.topi", skeleton_dict_path="skeleton_dict_1",
#                              verbose=True)
#     img_struct.get_data_from_sqlite(table_name="TOPS_80", key_column_name="Key", value_column_name="StatVal",
#                                     timestamp=None)
#     img_struct.get_np_array(use_other="Other", ignore_entries=("SYS:GROUP_TOTALS",), update_self=True)
#     img_struct.get_image_from_array(image_size=(20, 20), array=None, transform=np.log,
#                                     vmax_vmin=(None, 20))


def get_stacked_matrix(image_struct: ImageStruct, table_name: str,
                       key_column_name: str, value_column_name: str,
                       timestep_range: tuple, step: int,
                       use_other=None, ignore_entries=None):
    arrays = list()
    for timestep in range(timestep_range[0], timestep_range[1], step):
        image_struct.get_data_from_sqlite(table_name=table_name,
                                          key_column_name=key_column_name,
                                          value_column_name=value_column_name,
                                          timestamp=timestep)
        this_array = image_struct.get_np_array(use_other=use_other, ignore_entries=ignore_entries,
                                               update_self=False)
        arrays.append(this_array)
    stacked_matrix = np.stack(arrays)
    return stacked_matrix


img_struct = ImageStruct(sqlite_path="0.topi", skeleton_dict_path="skeleton_dict_1",
                         verbose=False)
stacked_matrix = get_stacked_matrix(img_struct, table_name="TOPS_80", key_column_name="Key",
                                    value_column_name="StatVal", timestep_range=(1720809000, 1720894800),
                                    step=300, use_other="Other", ignore_entries=("SYS:GROUP_TOTALS",))
stacked_matrix += -1
print(stacked_matrix.shape, end="\n\n")
covariance = np.cov(stacked_matrix)
print(covariance, end="\n\n")
mean = np.mean(stacked_matrix, axis=0)
print(mean, end="\n\n")
std_dev = np.sqrt(np.var(stacked_matrix, axis=0))
print(std_dev, end="\n\n")

rand_matrix = np.random.randn(400)
print(rand_matrix, end="\n\n")

sample = mean + std_dev * rand_matrix
print(sample, end="\n\n")

img_struct.get_image_from_array(image_size=(20, 20), array=sample, transform=None,
                                vmax_vmin=(None, None), verbose=True)

prob = scipy.stats.norm(loc=mean, scale=std_dev).cdf(sample)
print(prob)
prob = np.reshape(prob, (20, 20))
plt.imshow(prob)
plt.show()

img_struct = ImageStruct(sqlite_path="0.topi", skeleton_dict_path="skeleton_dict_1",
                         verbose=False)
img_struct.get_data_from_sqlite(table_name="TOPS_80", key_column_name="Key",
                                value_column_name="StatVal", timestamp=1720895100-300*19)
this_sample = img_struct.get_np_array(use_other="Other", ignore_entries=("SYS:GROUP_TOTALS",)) - 1
img_struct.get_image_from_array(image_size=(20, 20), array=this_sample, transform=np.log, verbose=True)

prob = scipy.stats.norm(loc=mean, scale=std_dev).cdf(this_sample)
print(prob)
prob = np.reshape(prob, (20, 20))
view_arr = abs(prob - 0.5)
print(view_arr)

plt.imshow(view_arr)
plt.show()
img_struct.get_image_from_array(image_size=(20, 20), array=this_sample, verbose=True)

# 1720827300
# timestep_dict = {}
#
# conn = get_sqlite_connection("0.topi")
# df1 = sqlite_to_pandas(conn, "TOPS_80", timestamp=1720809000+300*61)
# print(df1)
#
# # print(df2)
# # keys = df2["Key"].unique()
# # keys = np.delete(keys, np.where(keys == "SYS:GROUP_TOTALS"))
# # keys = keys[:399]
# # print(keys)
# # print(len(keys))
#
# df_kv = df1[["Key", "StatVal"]]
#
# # skeleton_dict = create_skeleton_dict(keys_list=keys, from_csv=False, init_value=1,
# #                                      save_path="/Users/aniket/PycharmProjects/unleashAnomalies/skeleton_dict_1")
# skeleton_dict = load_skeleton_dict("/Users/aniket/PycharmProjects/unleashAnomalies/skeleton_dict_1")
# populated_array = populate_skeleton_dict(skeleton_dict=skeleton_dict,
#                                          populate_data=df_kv, return_type=np.ndarray, use_other="Other",
#                                          ignore_entries=("SYS:GROUP_TOTALS",))
# populated_dict = populate_skeleton_dict(skeleton_dict=skeleton_dict,
#                                         populate_data=df_kv, return_type=dict, use_other="Other",
#                                         ignore_entries=("SYS:GROUP_TOTALS",))
# print(populated_array)
# view_array = np.log2(populated_array)
# view_array = np.reshape(view_array, (20, 20))
# populated_array = np.reshape(populated_array, (20, 20))
# print(populated_dict)
# plt.imshow(view_array)
# plt.show()
#
# # df_kv = df1[["Key", "StatVal"]]
# #
# # skeleton_dict = load_skeleton_dict("/Users/aniket/PycharmProjects/unleashAnomalies/skeleton_dict")
# # # print(df_kv)
# # populated_array = populate_skeleton_dict(skeleton_dict, df_kv, np.ndarray)
# # print(populated_array.shape)
# # print(populated_array)
