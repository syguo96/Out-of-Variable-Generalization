from enum import Enum
from typing import cast
from typing import Dict
from typing import Any
import logging
from pathlib import Path
import ast

import h5py
import numpy as np
import pandas as pd

from .datagen_settings import DataGenSettings


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


class SimulatedData(object):

    __slots__ = ("data_source", "data_target", "settings")

    def __init__(self) -> None:
        for attr in self.__slots__:
            setattr(self, attr, None)

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
        logger.info("saving dataset to {file_path}")
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


class Mode(Enum):
    general = 0
    linear = 1


class Level(Enum):
    level0 = 0
    level1 = 1
    level2 = 2

    def __lt__(self, other):
        return self.value < other.value

    def __le__(self, other):
        return self.value <= other.value

    def __gt__(self, other):
        return self.value > other.value

    def __ge__(self, other):
        return self.value >= other.value


def _gaussian_kernel(xdata, l1, sigma_f, sigma_noise=2e-2):
    num_total_points = xdata.shape[0]
    xdata1 = np.expand_dims(xdata, axis=0)
    xdata2 = np.expand_dims(xdata, axis=1)
    diff = xdata1 - xdata2

    norm = np.square(diff[:, :, :] / l1[np.newaxis, np.newaxis, :])
    norm = np.sum(norm, axis=3)

    kernel = np.square(sigma_f)[:, np.newaxis, np.newaxis] * np.exp(-0.5 * norm)
    kernel += (sigma_noise) ** 2 * np.eye(num_total_points)

    return kernel


def _y_gp(x, l1_scale=2, sigmaf_scale=1):
    num_samples = x.shape[0]
    l1 = np.ones((1, 2)) * l1_scale
    sigma_f = np.ones(1) * sigmaf_scale
    kernel = _gaussian_kernel(x, l1, sigma_f)
    cholesky = np.linalg.cholesky(kernel).reshape(num_samples, num_samples)
    noise = np.random.normal(size=(num_samples, 1))
    y = np.matmul(cholesky, noise).reshape(-1)
    return y


def ablation_generation(
    datagen_settings: DataGenSettings,
    level: Level,
    mode: Mode,
    noise=0.1,
    is_heavy_tailed=False,
) -> SimulatedData:
    logger = logging.getLogger("ablation-data-generation")

    num_samples = datagen_settings.num_samples
    split_fraction = datagen_settings.split_fraction

    # sample random variables from a scaled Gamma distribution
    shape, scale = 1, 1
    x = scale_max_min(np.random.gamma(shape, scale, (num_samples, 3)))

    x_source = x[:, :2]
    if mode == Mode.general:
        y_base = _y_gp(x_source)
        y_interaction = _y_gp(x_source)
        y_sqinteraction = _y_gp(x_source)
    if mode == Mode.linear:
        base = np.random.normal(0, 1, (2,))
        y_base = np.dot(x_source, base)
        interact = np.random.normal(0, 1, (3,))
        features = np.array([x[:, 0], x[:, 1], x[:, 0] * x[:, 1]]).T
        y_interaction = np.dot(features, interact)

    if level >= Level.level0:
        alphas = np.random.normal(0, 1)
        y = y_base + alphas * x[:, 2]

    # Ablation 1: with interaction
    if level >= Level.level1:
        y += y_interaction * x[:, 2]

    if level >= Level.level2:
        if mode == Mode.linear:
            gamma = np.random.normal(0, 1, (3,))
            y += np.dot(x, gamma)
        else:
            y += y_sqinteraction * (x[:, 2] ** 2)

    logger.info(f"std of y without noise {np.std(y)}")
    logger.info(f"current noise level {noise}")

    if is_heavy_tailed:
        noise = np.random.lognormal(0, 0.5, (num_samples,))
    else:
        noise = np.random.normal(0, noise, (num_samples,))
    logger.info(f"std of noise {np.std(noise)}")
    y += noise

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

    data = {
        "data_source": data_source,
        "data_target": data_target,
        "settings": datagen_settings.to_dict(),
    }
    dataset = SimulatedData.from_dictionary(data)

    return dataset
