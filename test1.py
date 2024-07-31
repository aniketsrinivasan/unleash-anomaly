from utils import *


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
kwargs_get_tensor = dict(
    use_other="Other",
    ignore_entries=("SYS:GROUP_TOTALS", ),
    image_shape=(20, 20)
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
print(len(dataset))
for i in range(10):
    print(dataset[i]["data"].shape)

print(dataset[10])
