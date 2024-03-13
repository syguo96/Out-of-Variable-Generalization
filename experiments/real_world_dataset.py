from pprint import pformat
import logging
import random
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
import torch
from ovg.evaluation import compute_zero_shot_loss
from ovg.predictors import (
    ImputedPredictor,
    MarginalPredictor,
    OptimalPredictor,
    ProposedPredictor,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def _set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def _run(nb_iterations=10, fixed_seed=True):

    def _iteration():
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

        reference_predictor = OptimalPredictor()
        proposed_predictor = ProposedPredictor()
        marginal_predictor = MarginalPredictor()
        imputation_predictor = ImputedPredictor()

        reference_predictor.fit(data_source, data_target)
        marginal_predictor.fit(data_source, data_target)
        proposed_predictor.fit(data_source, data_target)
        imputation_predictor.fit(data_source, data_target)

        # compute zero-shot loss
        predictors_dict = {"Proposed": proposed_predictor}
        predictors_dict["Optimal"] = reference_predictor
        predictors_dict["Marginal"] = marginal_predictor
        predictors_dict["MeanImputed"] = imputation_predictor
        losses = compute_zero_shot_loss(
            reference_predictor, predictors_dict, data_target, num_samples=1000
        )
        return losses

    proposed_loss = []
    marginal_loss = []
    imputed_loss = []

    for iteration in range(nb_iterations):
        if fixed_seed:
            _set_all_seeds(iteration)
        losses = _iteration()
        proposed_loss.append(losses["Proposed"])
        marginal_loss.append(losses["Marginal"])
        imputed_loss.append(losses["MeanImputed"])

    return proposed_loss, marginal_loss, imputed_loss


def _log_results(logger, label, data):
    mean = np.mean(data)
    variance = np.var(data)
    logger.info(f"{label:<10} mean: {mean:.4f}, variance: {variance:.4f}")


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    proposed_loss, marginal_loss, imputed_loss = _run(nb_iterations=10, fixed_seed=True)

    logger = logging.getLogger("real-world-dataset")
    logger.info("Results summary")
    _log_results(logger, "Proposed", proposed_loss)
    _log_results(logger, "Marginal", marginal_loss)
    _log_results(logger, "Imputed", imputed_loss)
