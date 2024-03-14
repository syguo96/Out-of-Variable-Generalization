from datetime import datetime
from ovg.predictors import Predictor
from typing import Optional
from pathlib import Path
from typing import Dict
import random
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import yaml
from ovg.predictors import (
    ImputedPredictor,
    MarginalPredictor,
    OptimalPredictor,
    ProposedPredictor,
)

from ovg_experiments.generate_data import generate_simulated_data, ExperimentType
from ovg_experiments.datagen_settings import DataGenSettings
from ovg_experiments.ablation_common import SimulatedData


def _compute_ground_truth_loss(predictor, data_target):
    y_target = predictor(data_target)
    return (np.square(y_target - data_target["Y"])).mean()


def _set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def _experiment(
    dataset: SimulatedData, ref_predictor, predictor_dict, num_samples_train
) -> Dict:
    loss_dict = {}
    data_target_size = dataset.data_target.shape[0]
    data_target_train, data_target_test = (
        dataset.data_target.iloc[: int(data_target_size * 0.5), :],
        dataset.data_target.iloc[int(data_target_size * 0.5) :, :],
    )
    print(data_target_train.shape, data_target_test.shape)

    for name, predictor in predictor_dict.items():
        predictor.fit(
            dataset.data_source, data_target_train.iloc[:num_samples_train, :]
        )
        loss_dict[name] = _compute_ground_truth_loss(predictor, data_target_test)

    reference_loss_dict = {}
    for num_samples in NUM_SAMPLES_VALIDATION:
        ref_predictor.fit(dataset.data_source, data_target_train.iloc[:num_samples, :])
        reference_loss_dict[num_samples] = _compute_ground_truth_loss(
            ref_predictor, data_target_test
        )
    loss_dict["reference"] = reference_loss_dict

    return loss_dict


def _run_experiment(
    experiment: ExperimentType,
    datagen_settings: DataGenSettings,
    ref_predictor: Predictor,
    predictor_dict: Dict[str, Predictor],
    num_samples_train: int,
    num_seeds: int,
    dataset_save: Optional[Path] = None,
) -> Dict:

    loss_dict = {seed: {} for seed in range(num_seeds)}
    for seed in range(num_seeds):
        _set_all_seeds(seed)
        dataset: SimulatedData = generate_simulated_data(experiment, datagen_settings)

        if dataset_save:
            dataset.to_file(dataset_save / f"data_{seed}.hdf5")

        loss_dict[seed] = _experiment(
            datagen_settings, ref_predictor, predictor_dict, num_samples_train
        )

    return loss_dict


def plot_losses(plot_dict, target_dir: Path):
    if not target_dir.is_dir():
        target_dir.mkdir(parents=True)
    for predictor in plot_dict.keys():
        df = pd.DataFrame(plot_dict[predictor])
        df = df.melt(var_name="Sample Size")
        sns.lineplot(
            data=df,
            x="Sample Size",
            y="value",
            marker="o",
            label=predictor,
            linewidth=3,
            markersize=8,
        )

    plt.axhline(y=0, color="gray", linestyle="--")
    plt.xscale("log")
    plt.xlabel("Number of samples")
    plt.ylabel("Relative Loss")
    plt.legend()
    plt.savefig(target_dir / "quant_comparison.pdf")
    plt.show()


if __name__ == "__main__":
    NUM_SAMPLES_VALIDATION = [10, 100, 200, 500, 1000, 2000, 5000]
    EXPERIMENT_BASE_DIR = "experiments/"  # "../OOV/experiments/"
    NUM_SEEDS = 5
    NUM_SAMPLES_TRAIN = 50

    results_dir = (
        Path.cwd()
        / f'results_quantitative_results_{datetime.now().strftime("%y_%m_%d_%H_%M_%S")}'
    )

    reference_predictor = OptimalPredictor()
    predictor_dict = {
        "Proposed": ProposedPredictor(),
        "Marginal": MarginalPredictor(),
        "MeanImputed": ImputedPredictor(),
    }

    datagen_settings = DataGenSettings.get_default()

    plot_dict = {
        pred: {
            num_samples: {seed: [] for seed in range(NUM_SEEDS)}
            for num_samples in NUM_SAMPLES_VALIDATION
        }
        for pred in predictor_dict.keys()
    }
    for experiment_type in (ExperimentType.nonlinear, ExperimentType.polynomial):
        exp_loss_dict = _run_experiment(
            experiment_type,
            datagen_settings,
            reference_predictor,
            predictor_dict,
            NUM_SAMPLES_TRAIN,
            NUM_SEEDS,
            dataset_save=results_dir / f"data_{experiment_type}.hdf5",
        )
        exp_plot_dict = {
            pred: {num_samples: [] for num_samples in NUM_SAMPLES_VALIDATION}
            for pred in predictor_dict.keys()
        }
        for pred in plot_dict.keys():
            for num_samples in NUM_SAMPLES_VALIDATION:
                for seed in range(NUM_SEEDS):
                    avg_error = np.log(
                        exp_loss_dict[seed][pred]
                        / exp_loss_dict[seed]["reference"][num_samples]
                    )
                    plot_dict[pred][num_samples][seed].append(avg_error)
                    exp_plot_dict[pred][num_samples].append(avg_error)

        plot_losses(exp_plot_dict, results_dir / experiment_type.name)

    mean_dict = {
        pred: {num_samples: [] for num_samples in NUM_SAMPLES_VALIDATION}
        for pred in predictor_dict.keys()
    }
    for pred in predictor_dict.keys():
        for num_samples in NUM_SAMPLES_VALIDATION:
            for seed in range(NUM_SEEDS):
                mean_dict[pred][num_samples].append(
                    np.mean(plot_dict[pred][num_samples][seed])
                )
    plot_losses(mean_dict, results_dir)
