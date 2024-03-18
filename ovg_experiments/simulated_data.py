import ast
import logging
from enum import Enum
from pathlib import Path
from typing import Any, Dict, cast

import h5py
import numpy as np
import pandas as pd
from scipy.stats import skew


class ExperimentType(Enum):
    polynomial = 0
    nonlinear = 1
    trigonometric = 2


def scale_max_min(x):
    return 5 * (x - np.min(x)) / (np.max(x) - np.min(x))


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


class DataGenSettings:

    __slots__ = (
        "num_samples",
        "split_fraction",
        "noise_var",
        "noise_skew",
        "noise_mean",
    )

    def __init__(self, num_samples, split_fraction, noise_var, noise_skew, noise_mean):
        self.num_samples = num_samples
        self.split_fraction = split_fraction
        self.noise_var = noise_var
        self.noise_skew = noise_skew
        self.noise_mean = noise_mean

    def to_dict(self) -> Dict[str, Any]:
        return {attr: getattr(self, attr) for attr in self.__slots__}

    @classmethod
    def get_default(cls):
        num_samples = 100000
        split_fraction = 0.9
        noise_var = 0.1
        noise_skew = 0
        noise_mean = 0
        return cls(num_samples, split_fraction, noise_var, noise_skew, noise_mean)


class SimulatedData(object):

    __slots__ = ("data_source", "data_target", "settings")

    def __init__(
        self,
        data_source: pd.DataFrame,
        data_target: pd.DataFrame,
        settings: Dict[str, Any],
    ) -> None:
        self.data_source = data_source
        self.data_target = data_target
        self.settings = settings

    @classmethod
    def from_file(cls, file_path: Path) -> "SimulatedData":
        logger = logging.getLogger("simulated data")
        logger.info(f"loading dataset {file_path}")
        with h5py.File(str(file_path), "r") as f:
            # Load only the keys that the class expects
            loaded_dict = _recursive_hdf5_load(f)
            instance = cls(
                loaded_dict["data_source"],
                loaded_dict["data_target"],
                ast.literal_eval(f.attrs["settings"]),  # type: ignore
            )
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
    def from_dictionary(cls, d: dict) -> "SimulatedData":
        instance = cast(
            "SimulatedData", cls(d["data_source"], d["data_target"], d["settings"])
        )
        return instance


def _y_polynomial(x, coeff):
    x_1, x_2, x_3 = x[:, 0], x[:, 1], x[:, 2]
    polynomial_features = np.array(
        [x_1, x_2, x_3, x_1 * x_2, x_1 * x_3, x_2 * x_3, x_1 * x_2 * x_3]
    )
    return np.dot(coeff, polynomial_features)


def _y_nonlinear(x, coeff):
    x_1, x_2, x_3 = x[:, 0], x[:, 1], x[:, 2]
    y = np.sqrt((coeff[2] * x_3) ** 2 + (coeff[0] * x_1) ** 2 + (coeff[1] * x_2) ** 2)
    return y


def _y_trigonometric(x, coeff):
    x_1, x_2, x_3 = x[:, 0], x[:, 1], x[:, 2]
    y = np.cos(coeff[0] * x_1 + coeff[1] * x_2 + coeff[2] * x_3)
    return y


def generate_simulated_data(
    experiment_type: ExperimentType, datagen_settings: DataGenSettings
) -> SimulatedData:

    num_samples = datagen_settings.num_samples
    split_fraction = datagen_settings.split_fraction
    noise_mean = datagen_settings.noise_mean
    noise_var = datagen_settings.noise_var

    # sample random variables from a scaled Gamma distribution
    shape, scale = 1, 1
    x = scale_max_min(np.random.gamma(shape, scale, (num_samples, 3)))

    resampling = True
    while resampling:
        coeff = np.random.normal(0, 1, (7,))
        if np.abs(coeff[2]) / (np.abs(coeff[1]) + np.abs(coeff[0])) > 2:
            resampling = False
    noise = np.random.normal(noise_mean, noise_var, (num_samples,))

    if experiment_type == ExperimentType.polynomial:
        y = _y_polynomial(x, coeff)
    elif experiment_type == ExperimentType.nonlinear:
        coeff = coeff[:3]
        y = _y_nonlinear(x, coeff)
    elif experiment_type == ExperimentType.trigonometric:
        coeff = coeff[:3]
        y = _y_trigonometric(x, coeff)

    y = y + noise
    num_samples_a = int(num_samples * split_fraction)
    data_dict_source = {
        "Y": y[:num_samples_a],
        "X_0": x[:num_samples_a, 0],
        "X_1": x[:num_samples_a, 1],
        "X_2": x[:num_samples_a, 2],
    }
    data_dict_target = {
        "Y": y[num_samples_a:],
        "X_0": x[num_samples_a:, 0],
        "X_1": x[num_samples_a:, 1],
        "X_2": x[num_samples_a:, 2],
    }

    data_source = pd.DataFrame(data_dict_source)
    data_target = pd.DataFrame(data_dict_target)

    settings = datagen_settings.to_dict()
    settings["coefficients"] = list(coeff)
    settings["noise_stats"] = {
        "mean": np.mean(noise),
        "var": np.var(noise),
        "skew": skew(noise),
    }
    data = {
        "data_source": data_source,
        "data_target": data_target,
        "settings": settings,
    }

    dataset = SimulatedData.from_dictionary(data)

    return dataset
