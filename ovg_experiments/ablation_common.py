import logging
import os
import random
from enum import Enum
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
import torch
from ovg.evaluation import compute_zero_shot_loss
from ovg.predictors import (
    ImputedPredictor,
    MarginalPredictor,
    OptimalPredictor,
    Predictor,
    PredictorType,
    ProposedPredictor,
)
from tabulate import tabulate

from .simulated_data import DataGenSettings, SimulatedData, scale_max_min


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


class AblationStudyConfig:
    def __init__(
        self,
        num_samples: int = 1000,
        lrs: Iterable[float] = (0.01,),
        hidden_sizes: Iterable[int] = (64,),
        epochs: Iterable[int] = (50,),
        num_runs: int = 5,
        noises: Iterable[float] = (0.01, 0.2, 0.4, 0.6, 0.8, 1),
        modes: Iterable[Mode] = (Mode.linear,),
    ):
        self.num_samples = num_samples
        self.lrs = lrs
        self.hidden_sizes = hidden_sizes
        self.epochs = epochs
        self.num_runs = num_runs
        self.noises = noises
        self.modes = modes

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AblationStudyConfig":
        instance = cls()
        for k, v in d.items():
            if not hasattr(instance, k):
                raise ValueError(
                    f"AblationStudyConfig: cannot set a value for {k} "
                    "(no such attribute)"
                )
            setattr(instance, k, v)
        return instance


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


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
    with_heavy_tailed=False,
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

    if with_heavy_tailed:
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

    dataset = SimulatedData(data_source, data_target, datagen_settings.to_dict())

    return dataset


class AblationStudyResults:
    def __init__(self):
        self._results = {
            level: {pred_type: [] for pred_type in PredictorType} for level in Level
        }

    def raw(self) -> Dict[Level, Dict[PredictorType, List[float]]]:
        return self._results

    def add_loss(
        self, level: Level, predictor_type: PredictorType, loss: float
    ) -> None:
        self._results[level][predictor_type].append(loss)

    def get_losses(self, level: Level, predictor_type: PredictorType) -> List[float]:
        return self._results[level][predictor_type]

    def get_mean(self, level: Level, predictor_type: PredictorType):
        return np.mean(self.get_losses(level, predictor_type))

    def get_std(self, level: Level, predictor_type: PredictorType):
        return np.std(self.get_losses(level, predictor_type))

    def summary_dict(
        self, with_perc: bool = False
    ) -> Dict[Level, Dict[PredictorType, Dict[str, float]]]:
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
        return r

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


def get_summary(
    results: Iterable[AblationStudyResults],
) -> Dict[Level, Dict[PredictorType, Dict[str, List[float]]]]:
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


def format_summary(
    summary: Dict[Level, Dict[PredictorType, Dict[str, List[float]]]]
) -> str:
    table_data = []
    headers = ["Level", "Method", "Mean", "Std"]
    for level, methods in summary.items():
        for method, stats in methods.items():
            row = [level, method, stats["mean"], stats["std"]]
            table_data.append(row)
    return tabulate(table_data, headers=headers)


def ablation_studies(
    data_generation_settings: DataGenSettings,
    mode: Mode,
    num_runs: int,
    lr: float,
    hidden_size: int,
    epoch: int,
    noise: float = 0.1,
    with_heavy_tailed: bool = False,
    loss_num_samples: int = 1000,
    loss_systematic: bool = True,
) -> AblationStudyResults:
    results = AblationStudyResults()
    for run in range(num_runs):
        for level in (Level.level0, Level.level1, Level.level2):
            dataset: SimulatedData = ablation_simulated_data_generation(
                data_generation_settings,
                level,
                mode,
                noise=noise,
                with_heavy_tailed=with_heavy_tailed,
            )
            predictors_dict = train_predictors(
                dataset.data_source,
                dataset.data_target,
                lr=lr,
                hidden_size=hidden_size,
                num_epochs=epoch,
            )
            loss = compute_zero_shot_loss(
                predictors_dict[PredictorType.oracle],
                predictors_dict,
                dataset.data_target,
                num_samples=loss_num_samples,
                systematic=loss_systematic,
            )
            for predictor_type, loss_value in loss.items():
                results.add_loss(level, predictor_type, loss_value)

    return results


def ablation_experiment(
    result_dir: Path,
    data_generation_settings: DataGenSettings,
    num_runs: int,
    lrs: Iterable[float],
    hidden_sizes: Iterable[int],
    epochs: Iterable[int],
    with_heavy_tailed: bool,
) -> None:
    logger = logging.getLogger("ablation-studies")
    result_dir.mkdir(parents=True)

    mode_results: Dict[Mode, List[AblationStudyResults]] = {mode: [] for mode in Mode}

    for mode in (Mode.linear, Mode.general):
        for lr, hidden_size, epoch in product(lrs, hidden_sizes, epochs):
            r: AblationStudyResults = ablation_studies(
                data_generation_settings,
                mode,
                num_runs,
                lr,
                hidden_size,
                epoch,
                with_heavy_tailed=with_heavy_tailed,
            )
            mode_results[mode].append(r)

    for mode, results in mode_results.items():
        summary = get_summary(results)
        np.save(result_dir / f"mean_{mode.name}.npy", summary)
        logger.info(f"\n-- results for {mode.name} --\n" + format_summary(summary))

    logger.info(f"results saved in {result_dir}")
