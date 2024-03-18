from os.path import join
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging
from numpy.typing import ArrayLike
from pandas.core.frame import DataFrame
import torch.nn as nn

plt.rcParams.update(
    {
        "text.usetex": False,
        "font.family": "sans-serif",
        "axes.titlesize": 24,
        "axes.labelsize": 20,
        "lines.linewidth": 3,
        "lines.markersize": 10,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
    }
)


def compute_zero_shot_loss(
    ReferencePredictor: nn.Module,
    predictors_dict: Dict[str, nn.Module],
    data_target: DataFrame,
    num_samples: int = 1000,
    systematic: bool = False,
) -> Dict[str, float]:
    logger = logging.getLogger("ovg-zero-shot-loss")
    data_target = data_target.loc[: num_samples - 1, :]
    reference = ReferencePredictor(data_target)

    losses: Dict[str, float] = {}
    for name, predictor in predictors_dict.items():
        y_target = predictor(data_target)
        losses[name] = (np.square(y_target - reference)).mean()
        if systematic:
            y_observed = data_target["Y"][:num_samples]
            losses[name] = (np.square(y_target - y_observed)).mean()

    for k, v in losses.items():
        logger.info(f"loss {k}:\t{v:.3f}")
    return losses


def visualize_zero_shot(
    predictors_dict: Dict[str, nn.Module],
    data: DataFrame,
    save_dir: str,
    num_samples: int = 2500,
) -> None:
    logger = logging.getLogger("ovg-visualize_zero_shot")
    num_samples_per_dim = int(np.sqrt(num_samples))
    num_samples = num_samples_per_dim**2

    x1 = np.linspace(0, 3, num_samples_per_dim)
    x2 = np.linspace(0, 3, num_samples_per_dim)
    X1, X2 = np.meshgrid(x1, x2)

    eval_frame = {
        "X_0": data.loc[: num_samples - 1, "X_0"],
        "X_1": X1.flatten(),
        "X_2": X2.flatten(),
    }
    eval_df = pd.DataFrame(eval_frame)
    for i, (name, predictor) in enumerate(predictors_dict.items()):
        logger.info(f"Visualizing {name} predictor")
        Y_pred = predictor(eval_df).reshape(num_samples_per_dim, num_samples_per_dim)
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
        ax.contour(X1, X2, Y_pred, linewidths=10)
        ax.set_xlabel(r"$X_2$", fontsize=60, weight="bold")
        ax.set_ylabel(r"$X_3$", fontsize=60, weight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout()
        plt.savefig(join(save_dir, "zero_shot_" + str(name) + ".pdf"))
        plt.show()
