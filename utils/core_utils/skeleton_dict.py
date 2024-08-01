import numpy as np
import torch
import pandas as pd
import pickle
import os
from .decorators import log_info


def _create_skeleton_dict(keys_list=None,
                          from_csv=False,
                          keys_csv_path=None, key_column_name=None,
                          init_value=0,
                          save_path: str = None, verbose=True) -> dict:
    """
    Creates a "skeleton dictionary" provided a .csv file with keys (and values). The dictionary is
    returned in the form {key1: init_value, key2: init_value, ...} where keys appear in the same order
    as in the .csv file.

    :param keys_list:       list of keys.
    :param from_csv:        whether to read from a .csv file. default is False.
    :param keys_csv_path:   path to the .csv file containing keys and values.
    :param key_column_name: name of the keys column in the .csv file.
    :param init_value:      default initialization value for all keys in the skeleton dictionary.
    :param save_path:       path to save the skeleton dictionary (optional).
    :param verbose:         prints debugging information.
    :return:                dictionary with keys initialized to init_value.
    """
    if verbose:
        print(f"Creating skeleton dictionary: \n"
              f"    reading type:       {'csv' if from_csv else 'list'}\n"
              f"    keys_list:          {len(keys_list) if keys_list is not None else 0}\n"
              f"    keys_csv_path:      {keys_csv_path}\n"
              f"    key_column_name:    {key_column_name}")
    # If keys_list is not provided, but the function is asked to read from keys_list:
    if (keys_list is None) and (from_csv is False):
        raise ValueError(f"You must provide either a list of keys, or set from_csv=True and provide a "
                         f".csv file path.")

    if from_csv:
        # Validate that a .csv path and column name are provided:
        if (keys_csv_path is None) or (key_column_name is None):
            raise ValueError(f"Trying to read from a .csv file. You must provide both key_csv_path and"
                             f"key_column_name.")
        # Check that the .csv filepath exists:
        if not os.path.exists(keys_csv_path):
            raise FileNotFoundError(f"File at {keys_csv_path} does not exist.")
        # If keys_list is provided despite reading from a .csv file, SoftWarn:
        if keys_list is not None:
            print(f"SoftWarn: keys_list is provided, but from_csv is set to True. Ignoring keys_list.")
        # Loading the .csv as a DataFrame:
        dataframe = pd.read_csv(keys_csv_path)
        # Extracting keys:
        keys = dataframe[key_column_name]
    # Otherwise, we have from_csv set to False, so we read from keys_list:
    else:
        if type(keys_list) not in (list, np.ndarray):
            raise TypeError(f"Trying to read from keys_list, which is of invalid type {type(keys_list)}. \n"
                            f"Provide keys_list as either a list or np.ndarray.")
        keys = keys_list

    # Initializing a skeleton dictionary using keys and the provided initialization value:
    skeleton_dict = {key: init_value for key in keys}

    # Saving file to provided path:
    if save_path is not None:
        with open(save_path, 'wb') as f:
            pickle.dump(skeleton_dict, f)
        if verbose:
            print(f"Skeleton dictionary saved to {save_path}.", end="\n\n")
    else:
        if verbose:
            print(f"Skeleton dictionary created and returned.", end="\n\n")
    return skeleton_dict


def _load_skeleton_dict(load_path: str, verbose=True) -> dict:
    """
    Loads a skeleton dictionary from a pickle file.

    :param load_path:   path to the skeleton dictionary file.
    :param verbose:     prints debugging information.
    :return:            dictionary in load_path.
    """
    if verbose:
        print(f"Loading skeleton dictionary from {load_path}...")
    with open(load_path, 'rb') as f:
        skeleton_dict = pickle.load(f)
    if verbose:
        print(f"Skeleton dictionary loaded.", end="\n\n")
    return skeleton_dict


def _populate_skeleton_dict(skeleton_dict: dict, populate_data, return_type: type,
                            use_other: str = None, ignore_entries=None, verbose=True):
    """
    Populates a given skeleton dictionary with the information in populate_data.
    If populate_data is a pd.DataFrame, it must ONLY contain the Key and Value columns (in that order).
    If populate_data is a dictionary, this is used to populate the skeleton instead.

    The return_type for this function can be one of the following:
        np.ndarray, pd.DataFrame, dict

    :param skeleton_dict:   skeleton dictionary to populate.
    :param populate_data:   data to populate the skeleton dictionary (see above for types).
    :param return_type:     return type for this function.
    :param use_other:       a string for the entry name "Other" to add any extra values not present
                                in the skeleton dictionary. defaults to None (no use_other).
    :param ignore_entries:  to ignore any entries from keys in populate_data (provided as a set).
    :param verbose:         prints debugging information.
    :return:                populated data.
    """
    # If we have a pd.DataFrame, we create a populate_dict from it first:
    if type(populate_data) is pd.DataFrame:
        # Extracting information from the
        populate_dict = {x[0]: x[1] for x in populate_data.to_dict("split")["data"]}
    # Otherwise if a dict is provided as data, we simply copy it into populate_dict:
    elif type(populate_data) is dict:
        populate_dict = populate_data
    else:
        raise TypeError(f"Parameter populate_data is of unsupported type {type(populate_data)}.")
    if ignore_entries is None:
        ignore_entries = set()

    # Populating the skeleton_dict using populate_dict:
    if use_other is None:
        # Updating values for every key in populate_dict:
        for key, value in populate_dict.items():
            if key in ignore_entries:
                continue
            elif key in skeleton_dict:
                skeleton_dict[key] += value
            else:
                skeleton_dict[key] = value
    # Otherwise use_other must be set and used:
    else:
        skeleton_dict[use_other] = 0
        for key, value in populate_dict.items():
            if key in ignore_entries:
                continue
            elif key in skeleton_dict:
                skeleton_dict[key] += value
            else:
                skeleton_dict[use_other] += value

    # We return a DataFrame:
    if return_type is pd.DataFrame:
        raise NotImplementedError
    # We return a numpy array:
    elif return_type is np.ndarray:
        array = np.fromiter(skeleton_dict.values(), dtype=float)
        return array
    # We return a dictionary:
    elif return_type is dict:
        return skeleton_dict
    # Otherwise unimplemented:
    else:
        raise NotImplementedError


class SkeletonDict:
    @log_info()
    def __init__(self, load_path=None, verbose=True):
        """
        Wrapper class for maintaining skeleton dictionaries. Used to load, create, populate and reset
        skeleton dictionaries based on provided data.

        :param load_path:   path to load existing skeleton dictionary (optional, None).
        :param verbose:     prints debugging information.
        """
        self.load_path = load_path
        self.verbose = verbose

        # Loading the skeleton dictionary if a path is provided:
        if self.load_path is None:
            print(f"Skeleton dictionary does not have a defined load_path. "
                  f"Use the SkeletonDict._load_skeleton_dict() method to load the dictionary.")
            self.raw_skeleton_dict = None
        else:
            self.raw_skeleton_dict = _load_skeleton_dict(self.load_path, verbose=verbose)

        # Initializing populated skeleton dictionary to None:
        self.populated_skeleton_dict = None

    @log_info()
    def load(self, load_path, override=False, verbose=None):
        """
        Loads skeleton dictionary (as a dict) into SkeletonDict.raw_skeleton_dict.
        
        :param load_path:   path to load skeleton dictionary from.
        :param override:    whether to override existing dictionary in SkeletonDict.raw_skeleton_dict.
        :param verbose:     prints debugging information.
        :return:            dictionary (skeleton dictionary).
        """
        if verbose is None:
            verbose = self.verbose
        if (self.raw_skeleton_dict is not None) and (override is False):
            print(f"SoftWarn: Loading skeleton dict from {load_path} but SkeletonDict.raw_skeleton_dict "
                  f"already exists. Overriding existing dictionary.")
        self.raw_skeleton_dict = _load_skeleton_dict(load_path, verbose=verbose)
        return self.raw_skeleton_dict

    @log_info(log_path="logs/log_skeleton_dicts", log_enabled=True, print_enabled=True)
    def create(self, keys_list=None, from_csv=False, keys_csv_path=None, key_column_name=None,
               init_value=0, save_path: str=None, verbose=False) -> dict:
        """
        Defines a new skeleton dictionary (saved in save_path) and stored in self.raw_skeleton_dict.

        :param keys_list:       list defining all the keys present in this dictionary (in desired order).
                                    (Optional, provide either keys_list or keys_csv_path).
        :param from_csv:        whether keys are provided as a .csv file instead of as a list.
        :param keys_csv_path:   .csv path with a column containing all the keys in this dictionary
                                    (in desired order).
                                    (Optional, provide either keys_list or keys_csv_path).
                                    Set from_csv to True if using keys_csv_path, and provide key_column_name.
        :param key_column_name: name of column in .csv file to read keys from.
                                    (Optional, must be provided if keys_csv_path is used).
        :param init_value:      initialization value for all keys in this skeleton dictionary (default 0).
        :param save_path:       path to save the created skeleton dictionary.
        :param verbose:         prints debugging information.
        :return:                dictionary (skeleton dictionary).
        """
        # Setting verbose value:
        if verbose is None:
            verbose = self.verbose
        # Loading the skeleton dictionary:
        self.raw_skeleton_dict = _create_skeleton_dict(keys_list=keys_list,
                                                       from_csv=from_csv,
                                                       keys_csv_path=keys_csv_path,
                                                       key_column_name=key_column_name,
                                                       init_value=init_value,
                                                       save_path=save_path,
                                                       verbose=verbose)
        return self.raw_skeleton_dict

    def populate(self, populate_data, return_type: type, use_other: str = None, ignore_entries=None,
                 verbose=None) -> dict:
        """
        Consumes data used to populate the skeleton dictionary found in SkeletonDict.raw_skeleton_dict.
        Requires that a dictionary is loaded into SkeletonDict.raw_skeleton_dict before usage (use
        SkeletonDict.load() or SkeletonDict.create() if raw_skeleton_dict is not found).

        :param populate_data:   data to populate the skeleton dictionary with
                                    (either as a pd.DataFrame or a dict).
        :param return_type:     return type for the returned data (either dict, pd.DataFrame or np.ndarray).
        :param use_other:       name of "Other" entry to use (all keys not in the skeleton dictionary
                                    go in this entry).
        :param ignore_entries:  keys to ignore when populating the skeleton dictionary, as a tuple or list.
        :param verbose:         prints debugging information.
        :return:                dictionary (populated skeleton dictionary)
        """
        verbose = verbose if (verbose is not None) else self.verbose
        if self.populated_skeleton_dict is not None:
            if verbose:
                print(f"SoftWarn: Populated skeleton dictionary in SkeletonDict.populated_skeleton_dict is "
                      f"not none. Overriding the existing dictionary.")
        # Loading data into skeleton dict and storing information:
        self.populated_skeleton_dict = _populate_skeleton_dict(skeleton_dict=self.raw_skeleton_dict,
                                                               populate_data=populate_data,
                                                               return_type=return_type,
                                                               use_other=use_other,
                                                               ignore_entries=ignore_entries,
                                                               verbose=verbose)
        return self.populated_skeleton_dict

    def reset(self):
        """
        Reloads the skeleton dictionary in self.raw_skeleton_dict from the stored path in
        SkeletonDict.load_path. Overrides any existing skeleton dictionary automatically.

        :return:    None.
        """
        self.load(load_path=self.load_path, override=True)
        return



def timestamp_to_embedding(timestamp: int,
                           weekday_onehot=True, weekend_onehot=True,
                           holiday_onehot=True, holiday_region="IND",
                           hour_of_day=True, cosine_hour_of_day=True,
                           minute_of_hour=True, minute_of_day=True,
                           verbose=True
                           ) -> torch.Tensor:
    # Embedding:
    #   (weekday_onehot,
    #    weekend_onehot,
    #    holiday_onehot,
    #    hour_of_day,
    #    cosine_hour_of_day,
    #    minute_of_hour,
    #    minute_of_day,
    #    0, ..., 0)
    pass


def get_timestamp_dict(skeleton_path: str, timestamp: int, dataframe: pd.DataFrame, verbose=True) -> dict:
    pass
