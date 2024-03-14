import sys
from datetime import datetime
from pathlib import Path
import logging
import random
from itertools import product

import numpy as np
import torch
from ovg.evaluation import compute_zero_shot_loss
from ovg.predictors import (
    ImputedPredictor,
    MarginalPredictor,
    OptimalPredictor,
    ProposedPredictor,
)

from ovg_experiments.datagen_settings import DataGenSettings
from ovg_experiments.ablation_common import (
    SimulatedData,
    ablation_generation,
    Level,
    Mode,
)

import os


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger = logging.getLogger("ablation-studies")
    logger.info(f"Random seed set as {seed}")


def _empty_summary_dict(mean_std: bool = True, list_values: bool = True):
    def _mean_std_dict_list():
        return {"mean": [], "std": []}

    def _mean_std_dict():
        return {"mean": None, "std": None}

    def _empty_array():
        return []

    if mean_std:
        if list_values:
            value_fn = _mean_std_dict_list
        else:
            value_fn = _mean_std_dict
    else:
        value_fn = _empty_array
    return {
        level: {
            key: value_fn() for key in ("Proposed", "Marginal", "Oracle", "Imputation")
        }
        for level in (Level.level0, Level.level1, Level.level2)
    }


def _run_experiment(dataset: SimulatedData, lr=0.01, hidden_size=64, num_epochs=50):

    logger = logging.getLogger("ablation-studies")

    reference_predictor = OptimalPredictor()
    proposed_predictor = ProposedPredictor(
        lr=lr, hidden_size=hidden_size, num_epochs=num_epochs
    )
    marginal_predictor = MarginalPredictor()
    imputation_predictor = ImputedPredictor()

    for label, pred in zip(
        ("reference", "proposed", "marginal", "imputation"),
        (
            reference_predictor,
            proposed_predictor,
            marginal_predictor,
            imputation_predictor,
        ),
    ):
        logger.info(f"fitting {label}")
        pred.fit(dataset.data_source, dataset.data_target)

    predictors_dict = {"Proposed": proposed_predictor}
    predictors_dict["Oracle"] = reference_predictor
    predictors_dict["Marginal"] = marginal_predictor
    predictors_dict["Imputation"] = imputation_predictor

    loss = compute_zero_shot_loss(
        reference_predictor,
        predictors_dict,
        dataset.data_target,
        num_samples=1000,
        systematic=True,
    )

    return loss


def _ablation_studies(
    data_generation_settings: DataGenSettings,
    num_runs=5,
    mode=Mode.linear,
    lr=0.01,
    hidden_size=64,
    epoch=50,
):

    results = _empty_summary_dict(mean_std=False, list_values=True)
    summary = _empty_summary_dict(mean_std=True, list_values=False)
    keys = ("Proposed", "Marginal", "Oracle", "Imputation")

    for run in range(num_runs):
        for level in (Level.level0, Level.level1, Level.level2):
            dataset = ablation_generation(data_generation_settings, level, mode)
            loss = _run_experiment(
                dataset, lr=lr, hidden_size=hidden_size, num_epochs=epoch
            )
            for attr in keys:
                results[level][attr].append(loss[attr])

    for key in results.keys():
        for predictor in keys:
            mean = np.mean(results[key][predictor])
            std = np.std(results[key][predictor])
            summary[key][predictor]["mean"] = mean
            summary[key][predictor]["std"] = std

    return results, summary


def _save_results(result_dir: Path, mode: Mode, results, summary):
    target_dir = result_dir / str(mode.name)
    if not target_dir.is_dir():
        target_dir.mkdir(parents=True)
    np.save(target_dir / "results.npy", results)
    np.save(target_dir / "summary.npy", summary)


def _main(
    result_dir: Path,
    data_generation_settings: DataGenSettings,
    lrs,
    hidden_sizes,
    epochs,
):
    logger = logging.getLogger("ablation-studies")
    summary_linear = _empty_summary_dict()
    summary_general = _empty_summary_dict()

    keys = ("Proposed", "Marginal", "Oracle", "Imputation")

    for lr, hidden_size, epoch in product(lrs, hidden_sizes, epochs):
        logger.info(
            f"learning rate {lr}, hidden_size {hidden_size}, num of epochs {epoch}"
        )
        for summary, mode in zip(
            (summary_linear, summary_general), (Mode.linear, Mode.general)
        ):
            results, summary_ = _ablation_studies(
                data_generation_settings,
                mode=mode,
                lr=lr,
                hidden_size=hidden_size,
                epoch=epoch,
            )
            for level in (Level.level0, Level.level1, Level.level2):
                for predictor in keys:
                    for key in ("mean", "std"):
                        summary[level][predictor][key].append(
                            summary_[level][predictor][key]
                        )
            _save_results(result_dir, mode, results, summary)
        logger.info(
            f"learning rate {lr}, hidden_size {hidden_size}, num of epochs {epoch}"
        )
        logger.info(f"linear result: {summary_linear}")
        logger.info(f"general result: {summary_general}")

    mean_linear = _empty_summary_dict()
    mean_general = _empty_summary_dict()

    for label, mean, summary in zip(
        ("linear", "general"),
        (mean_linear, mean_general),
        (summary_linear, summary_general),
    ):
        logger.info(f"--- results for mean {label} ---")
        for key in summary.keys():
            logger.info(f"\t{key.name}")
            for predictor in keys:
                logger.info(f"\t\t{predictor}")
                mean_ = np.mean(summary_linear[key][predictor]["mean"])
                mean_std_ = np.mean(summary_linear[key][predictor]["std"])
                logger.info(f"\t\t\tmean: {mean_:.3f}")
                logger.info(f"\t\t\tstd: {mean_std_:.3f}")
                mean[key][predictor]["mean"].append(mean_)
                mean[key][predictor]["std"].append(mean_std_)

    np.save(result_dir / "mean_linear.npy", mean_linear)
    np.save(result_dir / "mean_general.npy", mean_general)

    logger.info(f"results saved in {result_dir}")


class _Config:
    def __init__(self, num_samples, lrs, hidden_sizes, epochs):
        self.num_samples = num_samples
        self.lrs = lrs
        self.hidden_sizes = hidden_sizes
        self.epochs = epochs

    @classmethod
    def get_default(cls):
        num_samples = 10000
        lrs = (0.01, 0.001)
        hidden_sizes = (64, 32)
        epochs = (30, 50)
        return cls(num_samples, lrs, hidden_sizes, epochs)

    @classmethod
    def get_testing(cls):
        num_samples = 25
        lrs = (0.01, 0.001)
        hidden_sizes = (8, 4)
        epochs = (8, 12)
        return cls(num_samples, lrs, hidden_sizes, epochs)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logger = logging.getLogger("ablation-studies")

    datagen_settings = DataGenSettings.get_default()

    config = _Config.get_testing()
    # config = _config.get_default()

    datagen_settings.num_samples = config.num_samples

    seed = 42
    results_dir = (
        Path.cwd() / f'results_ablation_{datetime.now().strftime("%y_%m_%d_%H_%M_%S")}'
    )

    for k, v in datagen_settings.to_dict().items():
        logger.info(f"{k}:\t{v}")
    logger.info(f"seed:\t{seed}")
    logger.info(f"lrs:\t{config.lrs}")
    logger.info(f"hidden sizes\t{config.hidden_sizes}")
    logger.info(f"epochs:\t{config.epochs}")
    logger.info(f"result directory:\t{results_dir}")

    _set_seed(seed)
    _main(results_dir, datagen_settings, config.lrs, config.hidden_sizes, config.epochs)
