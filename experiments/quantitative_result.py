import logging
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from numpy.typing import ArrayLike
from ovg.predictors import Predictor
from ovg_experiments.ablation_common import (PredictorType, SimulatedData,
                                             train_predictors)
from ovg_experiments.simulated_data import (DataGenSettings, ExperimentType,
                                            generate_simulated_data)


class ExperimentsResult:
    # one instance of ExperimentsResult per num samples

    def __init__(self) -> None:
        self._avg_errors: Dict[PredictorType, List[float]] = {
            predictor_type: [] for predictor_type in PredictorType
        }  # one value per seed

    def add(self, predictor_type: PredictorType, avg_error: float) -> None:
        self._avg_errors[predictor_type].append(avg_error)

    def get(self, predictor_type: PredictorType) -> List[float]:
        return self._avg_errors[predictor_type]


def _average_over_experiments(
    num_samples_validation: Iterable[int],
    all_results: List[Dict[int, ExperimentsResult]],
) -> Dict[int, ExperimentsResult]:
    r: Dict[int, ExperimentsResult] = {}
    for num_samples in num_samples_validation:
        experiments_results: List[ExperimentsResult] = [
            er[num_samples] for er in all_results
        ]
        r_ = ExperimentsResult()
        for predictor_type in PredictorType:
            all_values = [er._avg_errors[predictor_type] for er in experiments_results]
            avg_values = [sum(x) / len(x) for x in zip(*all_values)]
            r_._avg_errors[predictor_type] = avg_values
        r[num_samples] = r_
    return r


def plot_losses(
    experiments_results: Dict[int, ExperimentsResult], target_file: Path, show: bool
) -> None:

    # one line plot per predictory type
    # x axis: num of samples
    # y axis: average errors

    for predictor_type in PredictorType:
        # key: num samples, values: list of average errors
        results: Dict[int, List[float]] = {
            num_samples: experiment_results.get(predictor_type)
            for num_samples, experiment_results in experiments_results.items()
        }
        df = pd.DataFrame(results)
        df = df.melt(var_name="Sample Size")
        sns.lineplot(
            data=df,
            x="Sample Size",
            y="value",
            marker="o",
            label=predictor_type.name,
            linewidth=3,
            markersize=8,
        )
    plt.axhline(y=0, color="gray", linestyle="--")
    plt.xscale("log")
    plt.xlabel("Number of samples")
    plt.ylabel("Relative Loss")
    plt.legend()
    plt.savefig(target_file)
    if show:
        plt.show()


def _compute_ground_truth_loss(predictor: Predictor, data_target: pd.DataFrame):
    y_target = predictor(data_target)
    return (np.square(y_target - data_target["Y"])).mean()


def _set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def _data_split(
    dataset: SimulatedData, train_test_split: float
) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
    # returns data_source, data_target_train, data_target_test
    data_target_size = dataset.data_target.shape[0]
    data_target_train, data_target_test = (
        dataset.data_target.iloc[: int(data_target_size * train_test_split), :],
        dataset.data_target.iloc[int(data_target_size * train_test_split) :, :],
    )
    return dataset.data_source, data_target_train, data_target_test


def _run_seed_experiment(
    seed: int,
    experiment_type: ExperimentType,
    datagen_settings: DataGenSettings,
    experiments_results: Dict[int, ExperimentsResult],
    train_test_split: float,
    num_samples_train: int,
    num_samples_validation: Iterable[int],
) -> SimulatedData:

    _set_all_seeds(seed)

    dataset: SimulatedData = generate_simulated_data(experiment_type, datagen_settings)

    data_source, data_target_train, data_target_test = _data_split(
        dataset, train_test_split
    )
    predictors: Dict[PredictorType, Predictor] = train_predictors(
        data_source, data_target_train.iloc[:num_samples_train, :]
    )
    predictors_loss: Dict[PredictorType, float] = {
        predictor_type: _compute_ground_truth_loss(predictor, data_target_test)
        for predictor_type, predictor in predictors.items()
    }

    reference_predictor: Predictor = predictors[PredictorType.oracle]
    for num_samples in num_samples_validation:
        reference_predictor.fit(data_source, data_target_train.iloc[:num_samples, :])
        loss = _compute_ground_truth_loss(reference_predictor, data_target_test)
        for predictor_type, predictor_loss in predictors_loss.items():
            experiments_results[num_samples].add(
                predictor_type, np.log(predictor_loss / loss)
            )

    return dataset


def _run_experiment_type(
    experiment_type: ExperimentType,
    datagen_settings: DataGenSettings,
    num_samples_train: int,
    num_seeds: int,
    num_samples_validation: Iterable[int],
    dataset_save: Optional[Path] = None,
    train_test_split: float = 0.5,
) -> Dict[int, ExperimentsResult]:

    logger = logging.getLogger("quant-studies")

    # key: number of samples for validation
    # value: instance of ExperimentsResult, which will save
    #   on result (average error) per predictory type and per seed
    experiments_results: Dict[int, ExperimentsResult] = {
        num_samples_valid: ExperimentsResult()
        for num_samples_valid in num_samples_validation
    }
    for seed in range(num_seeds):
        logger.info(
            f"running for experiment type {experiment_type.name} and seed {seed}"
        )
        # dataset returned only for the sake of saving it
        dataset = _run_seed_experiment(
            seed,
            experiment_type,
            datagen_settings,
            experiments_results,
            train_test_split,
            num_samples_train,
            num_samples_validation,
        )
        if dataset_save:
            dataset.to_file(dataset_save / f"data_{seed}.hdf5")
    return experiments_results


def _run_experiments(
    results_dir: Path,
    experiment_types: Iterable[ExperimentType],
    datagen_settings: DataGenSettings,
    num_samples_train: int,
    num_seeds: int,
    num_samples_validation: Iterable[int],
    train_test_split: float = 0.5,
    show: bool = True,
):
    # for each experiment type (e.g. 'non linear', 'polynomial')
    # running len(num_samples_validation) * num_seeds experiments.

    all_results: List[  # list: one item per experiment type
        # dict:
        #  - key: num samples
        #  - values: list of average errors (one per seed) for each predictor
        #      (as instances of ExperimentsResult)
        Dict[int, ExperimentsResult]
    ] = []

    for experiment_type in experiment_types:
        experiments_results: Dict[int, ExperimentsResult] = _run_experiment_type(
            experiment_type,
            datagen_settings,
            num_samples_train,
            num_seeds,
            num_samples_validation,
            results_dir,
            train_test_split,
        )
        target_file = results_dir / experiment_type.name / "quantitative_comparison.pdf"
        target_file.parent.mkdir()
        plot_losses(experiments_results, target_file, show)
        all_results.append(experiments_results)

    # averaging over all experiment types
    average_results: Dict[int, ExperimentsResult] = _average_over_experiments(
        num_samples_validation, all_results
    )
    target_file = results_dir / "quantitative_comparison.pdf"
    plot_losses(average_results, target_file, show)


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger("quant-studies")

    datagen_settings = DataGenSettings(
        num_samples=10000,
        split_fraction=0.9,
        noise_var=0.1,
        noise_skew=0.0,
        noise_mean=0.0,
    )
    num_samples_validation: Tuple[int, ...] = (10, 100, 200, 500, 1000, 2000, 5000)
    num_seeds = 5
    num_samples_train = 50
    experiment_types = (ExperimentType.nonlinear, ExperimentType.polynomial)
    train_test_split = 0.5
    show_plots = True

    results_dir = (
        Path.cwd()
        / f'results_quantitative_results_{datetime.now().strftime("%y_%m_%d_%H_%M_%S")}'
    )
    results_dir.mkdir(parents=True)

    _run_experiments(
        results_dir,
        experiment_types,
        datagen_settings,
        num_samples_train,
        num_seeds,
        num_samples_validation,
        train_test_split,
        show_plots,
    )

    logger.info(f"saved results in {results_dir}")
