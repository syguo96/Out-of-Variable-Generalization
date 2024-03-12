import pandas as pd

from data.generate_data import generate_data
import yaml
from os.path import join
import random
import torch
import numpy as np
import seaborn as sns

from predictors import (
    ProposedPredictor,
    OptimalPredictor,
    MarginalPredictor, ImputedPredictor,
)

import matplotlib.pyplot as plt

EXPERIMENTS = ["Nonlinear", "Polynomial"]


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def run_experiment(base_dir, experiment, ref_predictor, predictor_dict, num_seeds=5, num_samples_train=1000):
    experiment_dir = join(base_dir, experiment)

    with open(join(EXPERIMENT_BASE_DIR, "configs.yml"), "r") as f:
        settings = yaml.safe_load(f)

    loss_dict = {seed: {} for seed in range(num_seeds)}
    for seed in range(num_seeds):
        set_all_seeds(seed)
        dataset = generate_data(EXPERIMENT_BASE_DIR, experiment, settings["dataset"])
        dataset.to_file(join(experiment_dir, f"data_{seed}.hdf5"))

        data_target_size = dataset.data_target.shape[0]
        data_target_train, data_target_test = (
            dataset.data_target.iloc[: int(data_target_size * 0.5), :],
            dataset.data_target.iloc[int(data_target_size * 0.5):, :],
        )
        print(data_target_train.shape, data_target_test.shape)

        for name, predictor in predictor_dict.items():
            predictor.fit(dataset.data_source, data_target_train.iloc[:num_samples_train, :])
            loss_dict[seed][name] = compute_ground_truth_loss(
                predictor, data_target_test
            )

        reference_loss_dict = {}
        for num_samples in NUM_SAMPLES_VALIDATION:
            ref_predictor.fit(
                dataset.data_source, data_target_train.iloc[:num_samples, :]
            )
            reference_loss_dict[num_samples] = compute_ground_truth_loss(
                ref_predictor, data_target_test
            )
        loss_dict[seed]["reference"] = reference_loss_dict

    return loss_dict


def plot_losses(plot_dict, experiment_dir):
    for predictor in plot_dict.keys():
        df = pd.DataFrame(plot_dict[predictor])
        df = df.melt(var_name='Sample Size')
        sns.lineplot(data = df, x='Sample Size', y='value', marker = 'o', label = predictor, linewidth = 3, markersize=8)

    plt.axhline(y=0, color="gray", linestyle="--")
    plt.xscale("log")
    plt.xlabel("Number of samples")
    plt.ylabel("Relative Loss")
    plt.legend()
    plt.savefig(join(experiment_dir, f"quant_comparison.pdf"))
    plt.show()


def compute_ground_truth_loss(predictor, data_target):
    y_target = predictor(data_target)
    return (np.square(y_target - data_target["Y"])).mean()


if __name__ == "__main__":
    NUM_SAMPLES_VALIDATION = [10, 100, 200, 500, 1000, 2000, 5000]
    EXPERIMENT_BASE_DIR = "experiments/"  #"../OOV/experiments/"
    NUM_SEEDS = 5
    NUM_SAMPLES_TRAIN = 50

    reference_predictor = OptimalPredictor()
    predictor_dict = {
        "Proposed": ProposedPredictor(),
        "Marginal": MarginalPredictor(),
        "MeanImputed": ImputedPredictor()
    }

    plot_dict = {
        pred: {num_samples: {seed: [] for seed in range(NUM_SEEDS)} for num_samples in NUM_SAMPLES_VALIDATION}
        for pred in predictor_dict.keys()
    }
    for experiment in EXPERIMENTS:
        exp_loss_dict = run_experiment(
            EXPERIMENT_BASE_DIR, experiment, reference_predictor, predictor_dict, num_samples_train=NUM_SAMPLES_TRAIN, num_seeds=NUM_SEEDS
        )
        exp_plot_dict = {
            pred: {num_samples: [] for num_samples in NUM_SAMPLES_VALIDATION}
            for pred in predictor_dict.keys()
        }
        for pred in plot_dict.keys():
            for num_samples in NUM_SAMPLES_VALIDATION:
                for seed in range(NUM_SEEDS):
                    avg_error = np.log(
                            exp_loss_dict[seed][pred] /exp_loss_dict[seed]["reference"][num_samples]
                    )
                    plot_dict[pred][num_samples][seed].append(avg_error)
                    exp_plot_dict[pred][num_samples].append(avg_error)

        plot_losses(exp_plot_dict, join(EXPERIMENT_BASE_DIR, experiment))

    mean_dict = {
        pred: {num_samples: [] for num_samples in NUM_SAMPLES_VALIDATION}
        for pred in predictor_dict.keys()
    }
    for pred in predictor_dict.keys():
        for num_samples in NUM_SAMPLES_VALIDATION:
            for seed in range(NUM_SEEDS):
                mean_dict[pred][num_samples].append(np.mean(plot_dict[pred][num_samples][seed]))
    plot_losses(mean_dict, EXPERIMENT_BASE_DIR)
