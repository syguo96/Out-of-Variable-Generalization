from tabulate import tabulate
from typing import List
from typing import Iterable
import logging
import os
import random
from enum import Enum
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from ovg.evaluation import compute_zero_shot_loss
from ovg.predictors import (
    ImputedPredictor,
    MarginalPredictor,
    OptimalPredictor,
    ProposedPredictor,
)

from .datagen_settings import DataGenSettings
from .simulated_data import SimulatedData


class ExperimentType(Enum):
    polynomial = 0
    nonlinear = 1
    trigonometric = 2


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


class PredictorType(Enum):
    proposed = 0
    oracle = 1
    marginal = 2
    imputation = 3


class AblationStudyConfig:
    def __init__(self, num_samples, lrs, hidden_sizes, epochs, num_runs, noises, modes):
        self.num_samples = num_samples
        self.lrs = lrs
        self.hidden_sizes = hidden_sizes
        self.epochs = epochs
        self.num_runs = num_runs
        self.noises = noises
        self.modes = modes

    @classmethod
    def get_default(cls):
        num_samples = 10000
        lrs = (0.01,)
        hidden_sizes = (64,)
        epochs = (50,)
        num_runs = 5
        noises = (0.01, 0.2, 0.4, 0.6, 0.8, 1)
        modes = (Mode.linear,)
        return cls(num_samples, lrs, hidden_sizes, epochs, num_runs, noises, modes)

    @classmethod
    def get_testing(cls):
        num_samples = 25
        lrs = (0.01,)
        hidden_sizes = (8,)
        epochs = (10,)
        num_runs = 2
        noises = (
            0.01,
            0.2,
        )
        modes = (Mode.linear,)
        return cls(num_samples, lrs, hidden_sizes, epochs, num_runs, noises, modes)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def scale_max_min(x):
    return 5 * (x - np.min(x)) / (np.max(x) - np.min(x))


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


def ablation_simulated_data_generation(
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


def _run_ablation_experiment(
    dataset, lr=0.01, hidden_size=64, num_epochs=50
) -> Dict[PredictorType, float]:

    reference_predictor = OptimalPredictor()
    proposed_predictor = ProposedPredictor(
        lr=lr, hidden_size=hidden_size, num_epochs=num_epochs
    )
    marginal_predictor = MarginalPredictor()
    imputation_predictor = ImputedPredictor()

    for pred in [
        reference_predictor,
        proposed_predictor,
        marginal_predictor,
        imputation_predictor,
    ]:
        pred.fit(dataset.data_source, dataset.data_target)

    predictors_dict = {
        PredictorType.proposed: proposed_predictor,
        PredictorType.oracle: reference_predictor,
        PredictorType.marginal: marginal_predictor,
        PredictorType.imputation: imputation_predictor,
    }

    loss = compute_zero_shot_loss(
        reference_predictor,
        predictors_dict,
        dataset.data_target,
        num_samples=1000,
        systematic=True,
    )

    return loss


class AblationStudyResults:

    def __init__(self):
        self._results = {
            level: {pred_type: [] for pred_type in PredictorType} for level in Level
        }

    def raw(self):
        return self._results

    def add_loss(
        self, level: Level, predictor_type: PredictorType, loss: float
    ) -> None:
        self._results[level][predictor_type].append(loss)

    def get_losses(self, level: Level, predictor_type: PredictorType) -> None:
        return self._results[level][predictor_type]

    def get_mean(self, level: Level, predictor_type: PredictorType):
        return np.mean(self.get_losses(level, predictor_type))

    def get_std(self, level: Level, predictor_type: PredictorType):
        return np.std(self.get_losses(level, predictor_type))

    def summary_dict(self, with_perc: bool = False):
        r = {
            level: {
                pred_type: {
                    "mean": self.get_mean(level, pred_type),
                    "std": self.get_std(level, pred_type),
                }
                for pred_type in PredictorType
            }
            for level in Level
        }
        if not with_perc:
            return r
        for level in Level:
            oracle_mean = r[level][PredictorType.oracle]["mean"]
            for pred_type in PredictorType:
                mean = r[level][pred_type]["mean"]
                increase = (mean - oracle_mean) / oracle_mean
                r[level][pred_type]["perc"] = increase
        return

    def save(self, target_dir: Path) -> None:
        target_dir.mkdir(exist_ok=True, parents=True)
        np.save(target_dir / "results.npy", self._results)
        np.save(target_dir / "summary.npy", self.summary_dict())


def _get_means(
    results: Iterable[AblationStudyResults], level: Level, predictor_type: PredictorType
) -> List[float]:
    return [r.get_mean(level, predictor_type) for r in results]


def _get_stds(
    results: Iterable[AblationStudyResults], level: Level, predictor_type: PredictorType
) -> List[float]:
    return [r.get_std(level, predictor_type) for r in results]


def get_summary(results: Iterable[AblationStudyResults]):
    return {
        level: {
            pred_type: {
                "mean": _get_means(results, level, pred_type),
                "std": _get_stds(results, level, pred_type),
            }
            for pred_type in PredictorType
        }
        for level in Level
    }


def format_summary(summary):
    table_data = []
    headers = ["Level", "Method", "Mean", "Std"]
    for level, methods in summary.items():
        for method, stats in methods.items():
            row = [level, method] + stats["mean"] + stats["std"]
            table_data.append(row)
    return tabulate(table_data, headers=headers)


def ablation_studies(
    data_generation_settings: DataGenSettings,
    mode: Mode,
    num_runs: int,
    lr,
    hidden_size,
    epoch,
    noise=0.1,
) -> AblationStudyResults:
    results = AblationStudyResults()
    for run in range(num_runs):
        for level in (Level.level0, Level.level1, Level.level2):
            dataset = ablation_simulated_data_generation(
                data_generation_settings, level, mode, noise=noise
            )
            loss = _run_ablation_experiment(
                dataset, lr=lr, hidden_size=hidden_size, num_epochs=epoch
            )
            for predictor_type, loss_value in loss.items():
                results.add_loss(level, predictor_type, loss_value)

    return results