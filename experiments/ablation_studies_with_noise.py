from ovg_experiments.datagen_settings import DataGenSettings
from pathlib import Path
from datetime import datetime
import sys
import logging
from ovg_experiments.ablation_common import ExperimentConfig
import random
from itertools import product

import numpy as np
import torch
import yaml
from ovg.evaluation import compute_zero_shot_loss
from ovg.predictors import (
    ImputedPredictor,
    MarginalPredictor,
    OptimalPredictor,
    ProposedPredictor,
)

from ovg_experiments.ablation_common import ablation_generation, Level, Mode
import os

predictors = ("Proposed", "Oracle", "Marginal", "Imputation")


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def _run_experiment(dataset, lr=0.01, hidden_size=64, num_epochs=50):

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
        "Proposed": proposed_predictor,
        "Oracle": reference_predictor,
        "Marginal": marginal_predictor,
        "Imputation": imputation_predictor,
    }

    loss = compute_zero_shot_loss(
        reference_predictor,
        predictors_dict,
        dataset.data_target,
        num_samples=1000,
        systematic=True,
    )

    return loss


def ablation_studies(
    datagen_settings: DataGenSettings,
    noise,
    num_runs=5,
    mode=Mode.linear,
    lr=0.01,
    hidden_size=64,
    epoch=50,
):
    results = {level: {k: [] for k in predictors} for level in Level}

    for run in range(num_runs):
        for level in Level:
            dataset = ablation_generation(datagen_settings, level, mode, noise=noise)
            loss = _run_experiment(
                dataset, lr=lr, hidden_size=hidden_size, num_epochs=epoch
            )
            for predictor in predictors:
                results[level][predictor].append(loss[predictor])
    return results


def _calculate_mean_summary_table(summary_noises, modes, noises, predictors, levels):
    mean_table = {
        str(n): {
            mode: {
                level: {predictor: {"mean": [], "std": []} for predictor in predictors}
                for level in levels
            }
            for mode in modes
        }
        for n in noises
    }
    perc_increase_table = {
        str(n): {
            mode: {
                level: {predictor: {"perc": []} for predictor in predictors}
                for level in levels
            }
            for mode in modes
        }
        for n in noises
    }
    for n in noises:
        for mode in modes:
            for key in summary_noises[str(n)][mode].keys():
                for predictor in predictors:
                    mean = np.mean(summary_noises[str(n)][mode][key][predictor])
                    std = np.std(summary_noises[str(n)][mode][key][predictor])
                    mean_table[str(n)][mode][key][predictor]["mean"] = mean
                    mean_table[str(n)][mode][key][predictor]["std"] = std
                for predictor in predictors:
                    oracle_mean = mean_table[str(n)][mode][key]["oracle"]["mean"]
                    cur_mean = mean_table[str(n)][mode][key][predictor]["mean"]
                    perc_increase_table[str(n)][mode][key][predictor]["perc"] = (
                        cur_mean - oracle_mean
                    ) / oracle_mean
    print("mean_table", mean_table)

    return mean_table, perc_increase_table


class _Config:
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


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logger = logging.getLogger("ablation-studies-with-noise")

    _set_seed(42)

    results_dir = (
        Path.cwd()
        / f'results_ablation_with_noise_{datetime.now().strftime("%y_%m_%d_%H_%M_%S")}'
    )
    config = _Config.get_default()

    datagen_settings = DataGenSettings.get_default()

    summary_noises = {
        str(n): {mode: None} for mode in config.modes for n in config.noises
    }
    for noise in config.noises:
        for lr, hidden_size, epoch in product(
            config.lrs, config.hidden_sizes, config.epochs
        ):
            for mode in config.modes:
                summary_noises[str(noise)][mode] = ablation_studies(
                    datagen_settings,
                    noise,
                    config.num_runs,
                    mode,
                    lr,
                    hidden_size,
                    epoch,
                )
                logger.info(f"learning rate:\t{lr}")
                logger.info(f"hidden size:\t{hidden_size}")
                logger.info(f"epochs:\t{epoch}")
                logger.info(f"noise:\t{noise}")
                logger.info(f"mode:\t{mode.name}")
                logger.info(f"\tresult:\t{summary_noises[str(noise)][mode]:.3f}")

    np.save(results_dir / "summary_noises.npy", summary_noises)

    mean_table = _calculate_mean_summary_table(
        summary_noises, config.modes, config.noises, predictors, config.levels
    )
    np.save(results_dir / "mean_table_noises.npy", mean_table)
