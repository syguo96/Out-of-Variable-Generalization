import numpy as np
import pandas as pd
from scipy.stats import skew

from ovg_experiments.ablation_common import scale_max_min, SimulatedData
from ovg_experiments.datagen_settings import DataGenSettings

from enum import Enum


class ExperimentType(Enum):
    polynomial = 0
    nonlinear = 1
    trigonometric = 2


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
        y = y_polymomial(x, coeff)
    elif experiment_type == ExperimentType.nonlinear:
        coeff = coeff[:3]
        y = y_nonlinear(x, coeff)
    elif experiment_type == ExperimentType.trigonometric:
        coeff = coeff[:3]
        y = y_trigonometric(x, coeff)

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


def y_polymomial(x, coeff):
    x_1, x_2, x_3 = x[:, 0], x[:, 1], x[:, 2]
    polynomial_features = np.array(
        [x_1, x_2, x_3, x_1 * x_2, x_1 * x_3, x_2 * x_3, x_1 * x_2 * x_3]
    )
    return np.dot(coeff, polynomial_features)


def y_nonlinear(x, coeff):
    x_1, x_2, x_3 = x[:, 0], x[:, 1], x[:, 2]
    y = np.sqrt((coeff[2] * x_3) ** 2 + (coeff[0] * x_1) ** 2 + (coeff[1] * x_2) ** 2)
    return y


def y_trigonometric(x, coeff):
    x_1, x_2, x_3 = x[:, 0], x[:, 1], x[:, 2]
    y = np.cos(coeff[0] * x_1 + coeff[1] * x_2 + coeff[2] * x_3)
    return y
