import numpy as np
import matplotlib.pyplot as plt
from .core_utils import (get_sqlite_connection, sqlite_to_pandas, SkeletonDict)


class ImageStruct:
    def __init__(self, sqlite_path: str, skeleton_dict_path: str=None,
                 skeleton_dict: SkeletonDict=None, verbose=True):
        """
        Creating an ImageStruct object from a SQLite3 path and skeleton dictionary (path).

        :param sqlite_path:         path to the SQLite3 data to connect to.
        :param skeleton_dict_path:  path to initialize a SkeletonDict object.
        :param skeleton_dict:       skeleton dictionary as a SkeletonDict object.
        :param verbose:             prints information.
        """
        if verbose:
            print(f"Initializing ImageStruct object...")
        self.sqlite_path = sqlite_path
        if (skeleton_dict is None) and (skeleton_dict_path is None):
            raise ValueError("Either skeleton_dict or skeleton_dict_path must be provided when "
                             "initializing an ImageStruct object.")
        elif (skeleton_dict is not None) and (skeleton_dict_path is not None):
            print(f"Providing a SkeletonDict and skeleton_dict_path as arguments to ImageStruct. "
                  f"The SkeletonDict object overrides the usage of the path.")
            self.skeleton_dict = skeleton_dict
        elif skeleton_dict is not None:
            self.skeleton_dict = skeleton_dict
        else:
            self.skeleton_dict = SkeletonDict(skeleton_dict_path)

        # Saving the file path for the skeleton dict loaded:
        self.skeleton_dict_path = self.skeleton_dict.load_path
        # Storing verbose information:
        self.verbose = verbose

        # Defining the connection to SQLite3 database:
        self.connection = get_sqlite_connection(sqlite_path, verbose=self.verbose)
        # Defining the DataFrame consisting of information for this ImageStruct:
        self.dataframe = None
        # Defining np.ndarray containing data from self.dataframe:
        #   this is defined based on the structure of skeleton_dict
        self.np_array = None

        if verbose:
            print(f"ImageStruct initialized.")

    def get_data_from_sqlite(self, table_name, key_column_name, value_column_name,
                             update_self=True, timestamp=None, verbose=None):
        """
        Reads data from the SQLite3 connection formed, and returns a pd.DataFrame consisting of
        keys and values (as Series).

        If provided, the data read will be from a given timestamp. Else, all the data from the SQLite3
        connection is used.

        If update_self is True, the DataFrame will be saved in the instance under ImageStruct.dataframe.

        :param table_name:          name of the table to extract from SQLite3 database.
        :param key_column_name:     name of the "key" column in the SQLite3 table.
        :param value_column_name:   name of the "value" column in the SQLite3 table.
        :param update_self:         whether to update the instance with the DataFrame generated.
        :param timestamp:           extracts information from a particular timestamp only.
        :param verbose:             prints debugging information.
        :return:                    pd.DataFrame containing key and value columns (of data).
        """
        # Setting verbose information:
        verbose = self.verbose if (verbose is None) else verbose
        if verbose:
            print(f"Getting DataFrame from SQLite3 connection...")

        # Getting DataFrame from the SQLite3 connection:
        dataframe = sqlite_to_pandas(connection=self.connection, table_name=table_name,
                                     timestamp=timestamp, verbose=verbose)
        # Extracting only key and value column (where keys are IDs and values are the ID-wise data):
        dataframe = dataframe[[key_column_name, value_column_name]]
        # Updating instance if necessary:
        if update_self:
            self.dataframe = dataframe

        if verbose:
            print(f"DataFrame extracted from SQLite3 connection.")
        return dataframe

    def get_np_array(self, use_other=None, ignore_entries=None, update_self=True, verbose=None):
        """
        Produces an np.ndarray from the DataFrame in ImageStruct.dataframe.

        Usage for use_other is as follows:
            If you want to use an "other" class (where all keys in the DataFrame not found in the
            skeleton dict are stored), set use_other to the desired class name. Else, set to None.

        Usage for ignore_entries is as follows:
            If you want to ignore certain keys from the DataFrame, set ignore_entries to a set containing
            all the key names to be ignored (e.g. ignore_entries=("SYS:GROUP_TOTALS",). Else, set to None.

        :param use_other:       use an "other" class when importing data using the skeleton_dict.
        :param ignore_entries:  ignore a set of keys when reading data from the DataFrame.
        :param update_self:     whether to update the instance with the np.ndarray generated.
        :param verbose:         prints debugging information.
        :return:                np.ndarray containing data.
        """
        verbose = self.verbose if (verbose is None) else verbose
        if verbose:
            print(f"Setting np.ndarray from DataFrame... \n"
                  f"    use_other:      {use_other if (use_other is not None) else 'N/A'} \n"
                  f"    ignore_entries: {ignore_entries if (ignore_entries is not None) else 'N/A'} \n"
                  f"    update_self:    {update_self}")
        if self.dataframe is None:
            raise ValueError("DataFrame for this ImageStruct not found in ImageStruct.dataframe. \n"
                             "Please run ImageStruct.get_data_from_sqlite() before calling this method.")

        # Getting populated array using skeleton_dict:
        populated_array = self.skeleton_dict.populate(populate_data=self.dataframe,
                                                      return_type=np.ndarray,
                                                      use_other=use_other,
                                                      ignore_entries=ignore_entries,
                                                      verbose=verbose)
        # Updating instance if necessary:
        if update_self:
            self.np_array = populated_array
        return populated_array

    def get_image_from_array(self, image_size, array=None, transform=None,
                             vmax_vmin=None, verbose=None):
        """
        Creates an image (as an np.ndarray) from a provided image_size. Runs a transformation on the
        data array if a transform is provided (must function on an np.ndarray).

        :param image_size:  size of the output image (as a tuple, e.g. (20, 20)).
        :param array:       (optional) using a custom array. Defaults to using ImageStruct.np_array.
        :param transform:   running a transformation on the array data (must function on a np.ndarray).
        :param vmax_vmin:   parameters for setting vmax and vmin when plotting (optional).
        :param verbose:     prints debugging information.
        :return:            np.ndarray of image (of shape image_size).
        """
        verbose = self.verbose if (verbose is None) else verbose
        if verbose:
            print(f"Getting image of size {image_size} from np.ndarray in ImageStruct.np_array...")
        if (self.np_array is None) and (array is None):
            raise ValueError("Array data in ImageStruct.np_array not yet set. \n "
                             "Custom array was not provided (array=None). \n"
                             "Please run ImageStruct.get_np_array() before calling this method, "
                             "or provide a compatible custom array.")
        # Setting np_array to array if provided, else self.np_array:
        elif array is not None:
            np_array = array
        else:
            np_array = self.np_array

        # Running the provided image transformation if given:
        if transform is not None:
            np_array = transform(np_array)
        # Reshaping the array to the desired size:
        np_array = np.reshape(np_array, image_size)

        if vmax_vmin is not None:
            vmax = vmax_vmin[0]
            vmin = vmax_vmin[1]
        else:
            vmax = None
            vmin = None
        if verbose:
            print(f"Image generated and plotted.")
            plt.imshow(np_array, vmin=vmax, vmax=vmin)
            plt.show()
        return np_array
