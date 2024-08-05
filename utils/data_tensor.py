import numpy as np
import torch
from torch.utils.data import Dataset
from .core_utils import (get_sqlite_connection, sqlite_to_pandas, SkeletonDict,
                         log_info)


def get_skeleton_dict(skeleton_dict: SkeletonDict = None, skeleton_dict_path: str = None,
                      verbose=True) -> tuple:
    """
    Consumes a SkeletonDict, a dictionary path, or both, and returns a tuple of the loaded
    SkeletonDict and its corresponding filepath.

    If both are provided, the path is ignored and the SkeletonDict provided is used (along with
    its corresponding filepath).

    :param skeleton_dict:       SkeletonDict to retrieve (optional, None).
    :param skeleton_dict_path:  filepath to load a SkeletonDict from (optional, None).
    :param verbose:             prints debugging information.
    :return:                    tuple[SkeletonDict, skeleton_dict_path].
    """
    # If neither a SkeletonDict nor a path is provided:
    if (skeleton_dict is None) and (skeleton_dict_path is None):
        raise ValueError("Either skeleton_dict or skeleton_dict_path must be provided when "
                         "initializing an ImageStruct object.")
    # Otherwise, if a SkeletonDict is provided (regardless of whether a path is provided):
    elif skeleton_dict is not None:
        if skeleton_dict_path is not None:
            print(f"Providing a SkeletonDict and skeleton_dict_path as arguments to ImageStruct. "
                  f"The SkeletonDict object overrides the usage of the path.")
        this_skeleton_dict = skeleton_dict
    # Else, only a path is provided, so load a new SkeletonDict:
    else:
        this_skeleton_dict = SkeletonDict(load_path=skeleton_dict_path, verbose=verbose)
    # Extract the path from the SkeletonDict:
    this_skeleton_dict_path = this_skeleton_dict.load_path
    return this_skeleton_dict, this_skeleton_dict_path


class DataTensor:
    @log_info()
    def __init__(self, sqlite_path: str,
                 skeleton_dict: SkeletonDict = None, skeleton_dict_path: str = None,
                 table_name: str = None, key_column: str = None, value_column: str = None,
                 verbose: bool = True):
        """
        Class that acts as a base for converting SQLite3 database information into torch.Tensors for
        usage in deep learning tasks with PyTorch.

        :param sqlite_path:         path to SQLite3 database to extract information from.
        :param skeleton_dict:       SkeletonDict object that defines data format and conversion.
                                        (Optional, either pass skeleton_dict or skeleton_dict_path).
        :param skeleton_dict_path:  path to skeleton dictionary to generate new SkeletonDict object.
                                        (Optional, either pass skeleton_dict or skeleton_dict_path).
        :param table_name:          name of table in SQLite3 database to read from.
                                        (Optional, but available for ease of use).
        :param key_column:          name of column in SQLite3 database to read keys (IDs).
                                        (Optional, but available for ease of use).
        :param value_column:        name of column in SQLite3 database to read values (data points).
                                        (Optional, but available for ease of use).
        :param verbose:             prints debugging information.
        """
        # Setting all information:
        self.verbose = verbose
        self.table_name = table_name
        self.key_column = key_column
        self.value_column = value_column

        # Get the SkeletonDict and corresponding path information:
        self.skeleton_dict, self.skeleton_dict_path \
            = get_skeleton_dict(skeleton_dict=skeleton_dict,
                                skeleton_dict_path=skeleton_dict_path,
                                verbose=verbose)
        # Defining the connection to SQLite3 database:
        self.connection = get_sqlite_connection(sqlite_path, verbose=self.verbose)
        # Defining the DataFrame consisting of information for this ImageStruct:
        self.dataframe = None
        # Defining torch.Tensor that will store the information of this DataTensor:
        self.tensor = None

    @log_info()
    def retrieve_dataframe(self, table_name: str = None, key_column: str = None, value_column: str = None,
                           update_self=True, timestamp=None, verbose=None):
        """
        Converts data from an SQLite3 database into a pd.DataFrame. If update_self, then this is stored
        within the DataTensor.dataframe attribute.
        """
        # Setting key and value column names:
        table_name = table_name if table_name is not None else self.table_name
        key_column = key_column if key_column is not None else self.key_column
        value_column = value_column if value_column is not None else self.value_column

        # Setting verbose information:
        verbose = self.verbose if (verbose is None) else verbose
        if verbose:
            print(f"Getting DataFrame from SQLite3 connection...")

        # Getting DataFrame from the SQLite3 connection:
        dataframe = sqlite_to_pandas(connection=self.connection, table_name=table_name,
                                     timestamp=timestamp, verbose=verbose)
        # Extracting only key and value column (where keys are IDs and values are the ID-wise data):
        dataframe = dataframe[[key_column, value_column]]

        # Updating instance if necessary:
        if update_self:
            self.dataframe = dataframe
        if verbose:
            print(f"DataFrame extracted from SQLite3 connection.")
        return dataframe

    @log_info()
    def get_array(self, use_other: str = None, ignore_entries: str = None, verbose=None):
        """
        Produces an np.ndarray from the DataFrame in DataTensor.dataframe.

        Usage for use_other is as follows:
            If you want to use an "other" class (where all keys in the DataFrame not found in the
            skeleton dict are stored), set use_other to the desired class name. Else, set to None.

        Usage for ignore_entries is as follows:
            If you want to ignore certain keys from the DataFrame, set ignore_entries to a set containing
            all the key names to be ignored (e.g. ignore_entries=("SYS:GROUP_TOTALS",). Else, set to None.

        :param use_other:       use an "other" class when importing data using the skeleton_dict.
        :param ignore_entries:  ignore a set of keys when reading data from the DataFrame.
        :param verbose:         prints debugging information.
        :return:                np.ndarray containing data.
        """
        verbose = verbose if (verbose is not None) else self.verbose
        if verbose:
            print(f"Setting np.ndarray from DataFrame... \n"
                  f"    use_other:      {use_other if (use_other is not None) else 'N/A'} \n"
                  f"    ignore_entries: {ignore_entries if (ignore_entries is not None) else 'N/A'}")
        if self.dataframe is None:
            raise ValueError("DataFrame for this DataTensor not found in DataTensor.dataframe. \n"
                             "Please run DataTensor.retrieve_dataframe() before calling this method.")
        # Resetting existing array and getting populated array using skeleton_dict:
        self.skeleton_dict.reset()  # if not performed, data gets accumulated (unwanted)
        populated_array = self.skeleton_dict.populate(populate_data=self.dataframe,
                                                      return_type=np.ndarray,
                                                      use_other=use_other,
                                                      ignore_entries=ignore_entries,
                                                      verbose=verbose)
        return populated_array

    @log_info()
    def get_tensor(self, this_timestep: int, interval_size: int, num_intervals: int,
                   table_name: str = None, key_column: str = None, value_column: str = None,
                   use_other: str = None, ignore_entries = None, image_shape: tuple = None,
                   transform = None, verbose: bool = None):
        """
        Retrieve a torch.Tensor from this DataTensor object. The tensor is of the following shape:
            (num_intervals, entries)
        where num_intervals is provided by the user (e.g. 2 entries considers this timestep and
        the previous timestep) and "entries" is defined in the data itself (top-K data would
        give K entries).

        Uses DataTensor.retrieve_dataframe() to retrieve the data.
        Uses DataTensor.get_array() to convert data to np.ndarrays, before torch.Tensor conversion.

        Transformation is applied at the end, on the torch.Tensor of shape
            (num_intervals, image_shape[0], image_shape[1]).
        and should always return a torch.Tensor of the same (unchanged) shape.

        :param this_timestep:   the (latest) timestep to retrieve data for.
        :param interval_size:   the number of seconds between consecutive timesteps considered.
        :param num_intervals:   the number of timesteps to consider, stepped by interval_size, inclusive
                                    of this_timestep (e.g. 2 entries considers this timestep and the
                                    previous timestep)
        :param table_name:      name of the table to extract data from (optional, None).
        :param key_column:      name of the key column to extract data from (optional, None).
        :param value_column:    name of the value column to extract data from (optional, None).
        :param use_other:       whether to use an "other" entry for keys not found in the SkeletonDict
                                    (optional, None). enter name of this "other" entry.
        :param ignore_entries:  whether to ignore any keys present in input data (optional, None).
                                    enter name(s) of the keys to ignore as a set.
        :param image_shape:     shape of the image to return in the tensor (optional, None).
        :param transform:       transformation function to apply to the torch.Tensor before output.
        :param verbose:         prints debugging information.
        :return:                torch.Tensor containing data extracted.
        """
        # Setting table_name, key_column and value_column if provided, else default:
        table_name = table_name if (table_name is not None) else self.table_name
        key_column = key_column if (key_column is not None) else self.key_column
        value_column = value_column if (value_column is not None) else self.value_column

        # Setting verbose information:
        verbose = verbose if verbose is not None else self.verbose
        if verbose:
            print(f"Extracting tensor from DataTensor with the following information:\n "
                  f"    table_name:     {table_name}\n "
                  f"    key_column:     {key_column}\n "
                  f"    value_column:   {value_column}\n "
                  f"    this_timestep:  {this_timestep}\n "
                  f"    num_intervals:  {num_intervals}\n "
                  f"    interval_size:  {interval_size}\n "
                  f"    use_other:      {use_other}\n "
                  f"    ignore_entries: {ignore_entries} ")

        arrays = list()
        # Defining the starting timestep (if num_intervals==1, this should equal this_timestep):
        start_timestep = this_timestep - interval_size * (num_intervals - 1)
        # Iterating over timesteps:
        for timestep in range(start_timestep, this_timestep + interval_size, interval_size):
            if verbose:
                print(f"    Getting information for timestep {timestep}.")
            # Retrieving information for this DataFrame:
            self.retrieve_dataframe(table_name=table_name,
                                    key_column=key_column,
                                    value_column=value_column,
                                    update_self=True,
                                    timestamp=timestep,
                                    verbose=False)
            # Getting np.ndarray from DataFrame:
            this_array = self.get_array(use_other=use_other, ignore_entries=ignore_entries,
                                        verbose=False)
            # Appending to collection:
            arrays.append(this_array)

        # Stacking arrays together:
        stacked_array = np.stack(arrays)
        # Conversion to torch.Tensor:
        data_tensor = torch.from_numpy(stacked_array)
        if image_shape is not None:
            try:
                data_tensor = data_tensor.reshape((len(arrays), *image_shape))
            except Exception as e:
                print(f"SoftWarn: Unable to reshape the tensor to shape {image_shape}. "
                      f"Exception encountered as {e}. \n"
                      f"Continuing with torch.Tensor of shape {data_tensor.shape}.")
                pass
        # Apply transformation if provided:
        if transform is not None:
            data_tensor = transform(data_tensor)
        # Store tensor:
        self.tensor = data_tensor

        if verbose:
            print(f"Tensor created with shape {data_tensor.shape}.", end="\n\n")

        return data_tensor


class DatasetTensor(Dataset):
    @log_info(log_path="logs/log_datasets", log_enabled=True, print_enabled=True)
    def __init__(self, sqlite_path: str,
                 skeleton_dict: SkeletonDict = None, skeleton_dict_path: str = None,
                 table_name: str = None, key_column: str = None, value_column: str = None,
                 start_timestamp: int = None, end_timestamp: int = None,
                 interval_size: int = None, num_intervals: int = None,
                 kwargs_get_tensor: dict = None):
        """
        Class to use torch.utils.data.Dataset and torch.utils.data.DataLoader (inherits from the former).

        :param sqlite_path:         path to SQLite3 database to extract information.
        :param skeleton_dict:       SkeletonDict object defining how to arrange data.
                                        (Optional, provide either skeleton_dict or skeleton_dict_path).
        :param skeleton_dict_path:  path to a skeleton dictionary to create a SkeletonDict object.
                                        (Optional, provide either skeleton_dict or skeleton_dict_path).
        :param table_name:          name of the table in the SQLite3 database to get data.
        :param key_column:          name of the column containing keys (IDs) in the table.
        :param value_column:        name of the column containing values (data) in the table.
        :param start_timestamp:     the earliest timestamp from which a DataTensor is to be constructed.
                                        note that this may NOT be the first available data timestamp,
                                        depending on the value of num_intervals.
        :param end_timestamp:       the latest (most recent) timestamp from which a DataTensor
                                        is to be constructed.
        :param interval_size:       the gap between consecutive timestamps to extract (in seconds).
        :param num_intervals:       the number of consecutive timestamps to stack into one torch.Tensor.
        :param kwargs_get_tensor:   additional kwargs for the DataTensor.get_tensor() method, as follows:
                                        use_other: str,
                                        ignore_entries: tuple|set,
                                        image_shape: tuple,
                                        transform: function (torch.Tensor -> torch.Tensor),
                                        verbose: bool
        """
        # Defining this DataTensor (this will be how we get our torch.Tensors):
        self.data_tensor = DataTensor(sqlite_path=sqlite_path,
                                      skeleton_dict=skeleton_dict,
                                      skeleton_dict_path=skeleton_dict_path,
                                      table_name=table_name,
                                      key_column=key_column,
                                      value_column=value_column,
                                      verbose=False)
        # Defining timestamp information and kwargs_get_tensor (defining how to extract data from
        #   the data source provided):
        self.start_timestamp = start_timestamp      # earliest timestamp to extract from
        self.end_timestamp = end_timestamp          # latest timestamp to extract until
        self.interval_size = interval_size          # gap between consecutive timestamps
        self.num_intervals = num_intervals          # number of consecutive intervals per data tensor
        self.kwargs_get_tensor = kwargs_get_tensor  # kwargs for transforming data into tensor

    def __len__(self):
        # The number of entries we have is the number of entries between start and end timestamp.
        #   (subtract num_intervals-1 to account for overcounting;
        #    e.g. if 3 timestamps are present and num_intervals=2, then we have data length of 2)
        length = (((self.end_timestamp - self.start_timestamp) // self.interval_size)
                  - (self.num_intervals - 1))
        return length

    def __getitem__(self, idx):
        this_timestamp = self.start_timestamp + idx * self.interval_size
        this_tensor = self.data_tensor.get_tensor(this_timestep=this_timestamp,
                                                  interval_size=self.interval_size,
                                                  num_intervals=self.num_intervals,
                                                  **self.kwargs_get_tensor)
        this_data = {"timestamp": this_timestamp, "data": this_tensor}
        return this_data
