from data.generate_data import generate_data
import yaml
from os.path import join

from predictors import MarginalPredictor, ProposedPredictor, OptimalPredictor, ImputedPredictor
from evaluation import compute_zero_shot_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import numpy as np
import pandas as pd
import torch
import random

EXPERIMENT_BASE_DIR = "./experiments"

with open(join(EXPERIMENT_BASE_DIR, "configs.yml"), "r") as f:
    settings = yaml.safe_load(f)


# Fixed seed
def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

proposed_loss = []
marginal_loss = []
imputed_loss = []

for i in range(10):
    # set seed
    set_all_seeds(i)
    df = sm.datasets.get_rdataset("mtcars", "datasets", cache=True).data
    nsamples = 200
    augmented_df = df.sample(nsamples, replace=True)
    # data augmentation
    df = pd.concat([df, augmented_df], ignore_index=True)

    Y = 'mpg'
    [X1, X2, X3] = ['cyl', 'qsec', 'wt']
    df = df.rename(columns={Y: 'Y', X1: 'X_0', X2: 'X_1', X3: 'X_2'})
    # print(df.shape)
    train, test = train_test_split(df, test_size=0.2)
    data_source = train[['Y', 'X_0', 'X_1', 'X_2']]
    data_target = test[['Y', 'X_0', 'X_1', 'X_2']]
    scaler = StandardScaler()
    data_source = pd.DataFrame(scaler.fit_transform(data_source), columns = ['Y', 'X_0', 'X_1', 'X_2'])
    data_target = pd.DataFrame(scaler.fit_transform(data_target), columns=['Y', 'X_0', 'X_1', 'X_2'])

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
    losses = compute_zero_shot_loss(reference_predictor, predictors_dict, data_target, num_samples=1000)
    proposed_loss.append(losses['Proposed'])
    marginal_loss.append(losses['Marginal'])
    imputed_loss.append(losses['MeanImputed'])

print('Results summary')
print('Proposed', '\t mean', np.mean(proposed_loss), '\t var', np.var(proposed_loss))
print('Marginal', '\t mean', np.mean(marginal_loss), '\t var', np.var(marginal_loss))
print('Imputed',  '\t mean', np.mean(imputed_loss), '\t var', np.var(imputed_loss))