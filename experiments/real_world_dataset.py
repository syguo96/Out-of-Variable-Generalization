import logging
import random
import sys
from typing import Dict, List

import numpy as np
import pandas as pd
import statsmodels.api as sm
import torch
from ovg.evaluation import compute_zero_shot_loss
from ovg.predictors import (ImputedPredictor, MarginalPredictor,
                            OptimalPredictor, PredictorType, ProposedPredictor)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def _set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def _run(
    nb_iterations: int, num_samples: int, fixed_seed: bool
) -> Dict[PredictorType, List[float]]:

    def _iterate(num_samples) -> Dict[PredictorType, float]:
        df = sm.datasets.get_rdataset("mtcars", "datasets", cache=True).data
        nsamples = 200
        augmented_df = df.sample(nsamples, replace=True)
        # data augmentation
        df = pd.concat([df, augmented_df], ignore_index=True)

        Y = "mpg"
        [X1, X2, X3] = ["cyl", "qsec", "wt"]
        df = df.rename(columns={Y: "Y", X1: "X_0", X2: "X_1", X3: "X_2"})
        # print(df.shape)
        train, test = train_test_split(df, test_size=0.2)
        data_source = train[["Y", "X_0", "X_1", "X_2"]]
        data_target = test[["Y", "X_0", "X_1", "X_2"]]
        scaler = StandardScaler()
        data_source = pd.DataFrame(
            scaler.fit_transform(data_source), columns=["Y", "X_0", "X_1", "X_2"]
        )
        data_target = pd.DataFrame(
            scaler.fit_transform(data_target), columns=["Y", "X_0", "X_1", "X_2"]
        )

        predictors_dict = {
            PredictorType.oracle: OptimalPredictor(),
            PredictorType.proposed: ProposedPredictor(),
            PredictorType.marginal: MarginalPredictor(),
            PredictorType.imputation: ImputedPredictor(),
        }

        for predictor in predictors_dict.values():
            predictor.fit(data_source, data_target)

        losses = compute_zero_shot_loss(
            predictors_dict[PredictorType.oracle],
            predictors_dict,
            data_target,
            num_samples=num_samples,
        )
        return losses

    predictor_losses: Dict[PredictorType, List[float]] = {
        PredictorType.proposed: [],
        PredictorType.marginal: [],
        PredictorType.imputation: [],
    }

    for iteration in range(nb_iterations):
        if fixed_seed:
            _set_all_seeds(iteration)
        losses = _iterate(num_samples)
        for predictor_type, losses_list in predictor_losses.items():
            losses_list.append(losses[predictor_type])

    return predictor_losses


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    nb_iterations = 10
    num_samples = 1000
    fixed_seed = True

    predictor_losses: Dict[PredictorType, List[float]] = _run(
        nb_iterations, num_samples, fixed_seed
    )

    logger = logging.getLogger("real-world-dataset")
    logger.info("Results summary")
    for predictor_type, losses in predictor_losses.items():
        mean = np.mean(losses)
        variance = np.var(losses)
        logger.info(
            f"{predictor_type.name:<10} mean: {mean:.4f}, variance: {variance:.4f}"
        )
