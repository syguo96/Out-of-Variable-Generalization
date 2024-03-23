from typing import Tuple
import logging
from os.path import join
from typing import Dict, Type

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.nn as nn
from pandas.core.frame import DataFrame

from .predictors import Predictor, PredictorType

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
    reference_predictor: Predictor,
    predictors_dict: Dict[PredictorType, Predictor],
    data_target: DataFrame,
    num_samples: int = 1000,
    systematic: bool = False,
) -> Dict[PredictorType, float]:
    logger = logging.getLogger("ovg-zero-shot-loss")
    data_target = data_target.loc[: num_samples - 1, :]
    reference = reference_predictor(data_target)

    losses: Dict[PredictorType, float] = {}
    for predictor_type, predictor in predictors_dict.items():
        y_target = predictor(data_target)
        losses[predictor_type] = (np.square(y_target - reference)).mean()
        if systematic:
            y_observed = data_target["Y"][:num_samples]
            losses[predictor_type] = (np.square(y_target - y_observed)).mean()

    for k, v in losses.items():
        logger.info(f"loss {k}:\t{v:.3f}")
    return losses


def visualize_zero_shot(
    predictors_dict: Dict[PredictorType, Predictor],
    data: DataFrame,
    save_dir: str,
    num_samples: int = 2500,
    display: bool = True,
    figsize: Tuple[int, int] = (12, 6),
    dpi: int = 100,
    fontsize: int = 20,
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
    fig, axs = plt.subplots(
        nrows=len(predictors_dict), ncols=1, figsize=figsize, dpi=dpi
    )
    for i, (predictor_type, predictor) in enumerate(predictors_dict.items()):
        logger.info(f"Visualizing {predictor_type.name} predictor")
        Y_pred = predictor(eval_df).reshape(num_samples_per_dim, num_samples_per_dim)
        axs[i].contour(X1, X2, Y_pred, linewidths=10)
        axs[i].set_xlabel(r"$X_2$", fontsize=fontsize, weight="bold")
        axs[i].set_ylabel(r"$X_3$", fontsize=fontsize, weight="bold")
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        axs[i].set_title(predictor_type.name)
    plt.tight_layout()
    plt.savefig(join(save_dir, "zero_shot.pdf"))
    if display:
        plt.show()
