import os
from os.path import join
import numpy as np
import pandas as pd
from data.dataset import SimulatedData
from scipy.stats import skew

def scale_max_min(x):
    return 5 * (x - np.min(x)) / (np.max(x) - np.min(x))


def ablation_generation(base_path, data_generation_settings, level, mode, noise=0.1, is_heavy_tailed=False):
    data_path = join(base_path, 'ablation')
    os.makedirs(data_path, exist_ok=True)
    num_samples = data_generation_settings["num_samples"]
    split_fraction = data_generation_settings["split_fraction"]

    # sample random variables from a scaled Gamma distribution
    shape, scale = 1, 1
    x = scale_max_min(np.random.gamma(shape, scale, (num_samples, 3)))

    x_source = x[:, :2]
    if mode == 'general':
        y_base = y_gp(x_source)
        y_interaction = y_gp(x_source)
        y_sqinteraction = y_gp(x_source)
    if mode == 'linear':
        base = np.random.normal(0, 1, (2,))
        y_base = np.dot(x_source, base)
        interact = np.random.normal(0, 1, (3, ))
        features = np.array([x[:, 0 ], x[:, 1], x[:, 0]*x[:, 1]]).T
        y_interaction = np.dot(features, interact)

    if level >= 0:
        alphas = np.random.normal(0, 1)
        y = y_base + alphas * x[:, 2]

    # Ablation 1: with interaction
    if level >= 1:
        y += y_interaction * x[:, 2]

    if level >= 2:
        if mode == 'linear':
            gamma = np.random.normal(0, 1, (3, ))
            y += np.dot(x, gamma)
        else:
            y += y_sqinteraction * (x[:, 2]**2)

    print('std of y without noise', np.std(y))
    print('current noise level', noise)
    if is_heavy_tailed:
        noise = np.random.lognormal(0, 0.5, (num_samples, ))
    else:
        noise = np.random.normal(0, noise, (num_samples, ))
    print(is_heavy_tailed)
    print('std of noise', np.std(noise))
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

    settings = data_generation_settings
    data = {"data_source": data_source, "data_target": data_target, "settings": settings}
    dataset = SimulatedData(dictionary=data)

    return dataset


def generate_data(base_path, experiment, data_generation_settings, mode = 'mean'):
    data_path = join(base_path, experiment)
    os.makedirs(data_path, exist_ok=True)
    num_samples = data_generation_settings["num_samples"]
    split_fraction = data_generation_settings["split_fraction"]
    noise_mean = data_generation_settings['noise_mean']
    noise_var = data_generation_settings['noise_var']
    noise_skew = data_generation_settings['noise_skew']

    # sample random variables from a scaled Gamma distribution
    shape, scale = 1, 1
    x = scale_max_min(np.random.gamma(shape, scale, (num_samples, 3)))

    resampling = True
    while resampling:
        coeff = np.random.normal(0, 1, (7,))
        if np.abs(coeff[2])/(np.abs(coeff[1]) + np.abs(coeff[0])) > 2:
            resampling = False
    noise = np.random.normal(noise_mean, noise_var, (num_samples,))

    if experiment.lower() == "polynomial":
        y = y_polymomial(x, coeff)
    elif experiment.lower() == "nonlinear":
        coeff = coeff[:3]
        y = y_nonlinear(x, coeff)
    elif experiment.lower() == "trigonometric":
        coeff = coeff[:3]
        y = y_trigonometric(x, coeff)
    else:
        raise NotImplementedError("Please pass a valid dataset name")

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

    settings = data_generation_settings
    settings["coefficients"] = list(coeff)
    settings['noise_stats'] = {'mean': np.mean(noise), 'var': np.var(noise), 'skew': skew(noise)}
    data = {"data_source": data_source, "data_target": data_target, "settings": settings}
    dataset = SimulatedData(dictionary=data)

    return dataset

def gaussian_kernel(xdata, l1, sigma_f, sigma_noise=2e-2):
    num_total_points = xdata.shape[0]
    xdata1 = np.expand_dims(xdata, axis = 0)
    xdata2 = np.expand_dims(xdata, axis = 1)
    diff = xdata1 - xdata2

    norm = np.square(diff[:, :, :] / l1[np.newaxis, np.newaxis, :])
    norm = np.sum(norm, axis = 3)

    kernel = np.square(sigma_f)[:, np.newaxis, np.newaxis] * np.exp(-0.5 * norm)
    kernel += (sigma_noise)**2 * np.eye(num_total_points)

    return kernel

def y_gp(x, l1_scale = 2, sigmaf_scale = 1):
    num_samples = x.shape[0]
    l1 = np.ones((1, 2))* l1_scale
    sigma_f = np.ones(1) * sigmaf_scale
    kernel = gaussian_kernel(x, l1, sigma_f)
    cholesky = np.linalg.cholesky(kernel).reshape(num_samples, num_samples)
    noise = np.random.normal(size=(num_samples, 1))
    y = np.matmul(cholesky, noise).reshape(-1)
    return y

def y_polymomial(x, coeff):
    x_1, x_2, x_3 = x[:, 0], x[:, 1], x[:, 2]
    polynomial_features = np.array(
        [x_1, x_2, x_3, x_1 * x_2, x_1 * x_3, x_2 * x_3, x_1 * x_2 * x_3]
    )
    return np.dot(coeff, polynomial_features)


def y_nonlinear(x, coeff):
    x_1, x_2, x_3 = x[:, 0], x[:, 1], x[:, 2]
    y = (
        np.sqrt((coeff[2] * x_3) ** 2 + (coeff[0] * x_1) ** 2 + (coeff[1] * x_2) ** 2)
    )
    return y


def y_trigonometric(x, coeff):
    x_1, x_2, x_3 = x[:, 0], x[:, 1], x[:, 2]
    y = (
        np.cos(coeff[0] * x_1 + coeff[1] * x_2 + coeff[2] * x_3)
    )
    return y
