import ast
import h5py
import pandas as pd
import numpy as np


def recursive_hdf5_save(group, d):
    for k, v in d.items():
        if v is None:
            continue
        elif isinstance(v, dict):
            next_group = group.create_group(k)
            recursive_hdf5_save(next_group, v)
        elif isinstance(v, np.ndarray):
            group.create_dataset(k, data=v)
        elif isinstance(v, pd.DataFrame):
            group.create_dataset(k, data=v.to_records(index=False))
        elif isinstance(v, (int, float, str, list)):
            group.create_dataset(k, data=v)
        else:
            raise TypeError(f"Cannot save datatype {type(v)} as hdf5 dataset.")


def recursive_hdf5_load(group):
    d = {}
    for k, v in group.items():
        if isinstance(v, h5py.Group):
            d[k] = recursive_hdf5_load(v)
        else:
            d[k] = v[...]
            # If the array has column names, load it as a pandas DataFrame
            if d[k].dtype.names is not None:
                d[k] = pd.DataFrame(d[k])
            # Convert arrays of size 1 to scalars
            elif d[k].size == 1:
                d[k] = d[k].item()
                if isinstance(d[k], bytes):
                    # Assume this is a string.
                    d[k] = d[k].decode()
            # If an array is 1D and of type object, assume it originated as a list
            # of strings.
            elif d[k].ndim == 1 and d[k].dtype == "O":
                d[k] = [x.decode() for x in d[k]]
    return d

class SimulatedData(object):

    def __init__(self, file_name=None, dictionary=None):

        self.data_source = None
        self.data_target = None
        self.settings = None

        if file_name is not None:
            self.from_file(file_name)
        elif dictionary is not None:
            self.from_dictionary(dictionary)

    def to_file(self, file_name, mode="w"):
        print("Saving dataset to " + str(file_name))
        save_dict = {
            k: v
            for k, v in vars(self).items()
        }
        with h5py.File(file_name, mode) as f:
            recursive_hdf5_save(f, save_dict)
            if self.settings:
                f.attrs["settings"] = str(self.settings)

    def from_file(self, file_name):
        print("Loading dataset from " + str(file_name) + ".")
        with h5py.File(file_name, "r") as f:
            # Load only the keys that the class expects
            loaded_dict = recursive_hdf5_load(f)
            for k, v in loaded_dict.items():
                vars(self)[k] = v
            self.settings = ast.literal_eval(f.attrs["settings"])

    def to_dictionary(self):
        dictionary = {
            k: v
            for k, v in vars(self).items()
            if v is not None
        }
        return dictionary

    def from_dictionary(self, dictionary: dict):
        for k, v in dictionary.items():
            vars(self)[k] = v

