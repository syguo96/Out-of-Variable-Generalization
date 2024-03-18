import ast
import h5py
import numpy as np
from pathlib import Path
import pandas as pd
from typing import cast, Dict, Any
import logging


def _recursive_hdf5_save(group, d):
    for k, v in d.items():
        if v is None:
            continue
        elif isinstance(v, dict):
            next_group = group.create_group(k)
            _recursive_hdf5_save(next_group, v)
        elif isinstance(v, np.ndarray):
            group.create_dataset(k, data=v)
        elif isinstance(v, pd.DataFrame):
            group.create_dataset(k, data=v.to_records(index=False))
        elif isinstance(v, (int, float, str, list)):
            group.create_dataset(k, data=v)
        else:
            raise TypeError(f"Cannot save datatype {type(v)} as hdf5 dataset.")


def _recursive_hdf5_load(group):
    d = {}
    for k, v in group.items():
        if isinstance(v, h5py.Group):
            d[k] = _recursive_hdf5_load(v)
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

    __slots__ = ("data_source", "data_target", "settings")

    def __init__(self) -> None:
        self.data_source = None
        self.data_target = None
        self.settings = None

    @classmethod
    def from_file(cls, file_path: Path) -> "SimulatedData":
        logger = logging.getLogger("simulated data")
        logger.info(f"loading dataset {file_path}")
        instance = cls()
        with h5py.File(str(file_path), "r") as f:
            # Load only the keys that the class expects
            loaded_dict = _recursive_hdf5_load(f)
            for k, v in loaded_dict.items():
                setattr(instance, k, v)
            instance.settings = ast.literal_eval(f.attrs["settings"])  # type: ignore
        return instance

    def to_file(self, file_path: Path, mode="w") -> None:
        logger = logging.getLogger("simulated data")
        logger.info(f"saving dataset to {file_path}")
        save_dict = {attr: getattr(self, attr) for attr in self.__slots__}
        with h5py.File(str(file_path), mode) as f:
            _recursive_hdf5_save(f, save_dict)
            if self.settings:  # type: ignore
                f.attrs["settings"] = str(self.settings)  # type: ignore

    def to_dictionary(self) -> Dict[str, Any]:
        return {attr: getattr(self, attr) for attr in self.__slots__}

    @classmethod
    def from_dictionary(cls, dictionary: dict) -> "SimulatedData":
        instance = cast("SimulatedData", cls())
        for k, v in dictionary.items():
            setattr(instance, k, v)
        return instance
